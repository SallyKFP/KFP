import asyncio
import html
import hmac
import json as json_lib
import logging
import os
import re
import secrets
from contextlib import asynccontextmanager
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from urllib.parse import parse_qs

import pytz
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from config.loader import load_config
from core.ai import extract_booking_intent, get_ai_response, load_knowledge
from core.handover import compose_handover_reply, handover_reason
from core.dashboard_store import DashboardStore, normalize_phone
from core.confirmation import (
    BOOKING_CONFIRMATION_INSTRUCTION,
    BOOKING_NOT_CONFIRMED,
    REMINDER_NOT_CONFIRMED,
    awaiting_booking_information,
    is_booking_confirmation_prompt,
    is_exact_reply,
    is_short_acknowledgement,
)
from core.history import get_history
from core.patient_dates import validate_patient_dob
from core.pending_booking import (
    PENDING_BOOKINGS,
    existing_proposal,
    extract_general_observations,
    has_booking_cancellation,
    inherit_modification_consultant,
    preserve_observations,
    preserves_pending_booking,
)
from core.transcribe import transcribe_audio
from core.rescheduling import is_within_rescheduling_window
from core.whatsapp import WhatsAppClient, validate_webhook_signature
from modules.booking.calendar import (
    CalendarClient,
    Slot,
    patient_age_group,
    validate_booking_slot,
)
from modules.booking.availability import check_availability_range, check_slot_availability
from modules.booking.consultants import (
    allocate_consultant,
    configured_consultants,
    consultant_slot_is_reserved,
    resolve_consultant,
)
from modules.payments.mercadopago import MPClient, validate_mp_signature
from reminders.scheduler import (
    run_daily_reminders,
    seconds_until_next_reminder as _seconds_until_next_reminder,
    send_reminders,
)

logger = logging.getLogger(__name__)

# Consultant picked during a slot check, keyed by (phone, service, date, time, location),
# so the booking made right after keeps that consultant.
_pending_consultant_assignments: dict[tuple[str, str, str, str, str], str] = {}


def _consultant_assignment_key(phone: str, intent: dict) -> tuple[str, str, str, str, str] | None:
    values = tuple(str(intent.get(k, "")).strip() for k in ("service", "date", "time", "location"))
    return (phone, *values) if all(values) else None

CONFIG = load_config()
KNOWLEDGE = load_knowledge()
HISTORY = get_history()

_dashboard_store: DashboardStore | None = None


WA = WhatsAppClient(
    phone_number_id=os.environ["WHATSAPP_PHONE_NUMBER_ID"],
    access_token=os.environ["WHATSAPP_ACCESS_TOKEN"],
)
APP_SECRET = os.environ["WHATSAPP_APP_SECRET"]
VERIFY_TOKEN = os.environ["WHATSAPP_VERIFY_TOKEN"]
INTERNAL_SECRET = os.environ.get("INTERNAL_SECRET", "")
MP_WEBHOOK_SECRET = os.environ.get("MP_WEBHOOK_SECRET", "")

_calendar_client: CalendarClient | None = None
_calendar_clients: dict[str, CalendarClient] = {}
_mp_client: MPClient | None = None
_pending_redis = None

# In-memory fallback for pending operations (used when Redis is not available)
_pending_modifications: dict[str, dict] = {}
_pending_cancellations: dict[str, list[dict]] = {}
_message_locks: dict[str, float] = {}  # phone -> lock expiry timestamp

# Keywords that indicate the user wants to modify an existing booking
_MODIFICATION_KEYWORDS = {"change", "modify", "move", "reschedule", "switch", "postpone", "cambiar", "modificar", "mover", "reprogramar"}
_SATURDAY_RESCHEDULING_MESSAGE = (
    "Saturday appointments are prepaid at SGD 300 for a single intervention and are "
    "non-refundable. Rescheduling is allowed only when requested more than 36 hours "
    "before the appointment. Within or exactly 36 hours, a valid medical certificate "
    "(MC) is required; without one, the appointment must be cancelled and a new "
    "booking made from scratch, and the 100% cancellation charge applies. Saturday "
    "appointment changes must be handled by the administrative team, so please "
    "contact the clinic directly."
)
_RESCHEDULING_MC_QUESTION = (
    "This appointment is within 36 hours. Rescheduling is only possible if you have "
    "a valid medical certificate (MC). Do you have a valid MC? Please reply YES or NO. "
    "Without one, you will need to cancel and make a new booking from scratch, "
    "providing all booking details again; the 100% cancellation charge applies."
)


def _get_mp_client() -> MPClient | None:
    if not CONFIG.get("modules", {}).get("payments"):
        return None
    global _mp_client
    if _mp_client is None:
        sandbox = CONFIG.get("payments", {}).get("sandbox", True)
        _mp_client = MPClient(sandbox=sandbox)
    return _mp_client


def _get_pending_payment_redis():
    global _pending_redis
    if _pending_redis is None:
        redis_url = os.environ.get("REDIS_URL")
        if redis_url:
            import redis
            _pending_redis = redis.from_url(redis_url, decode_responses=True)
    return _pending_redis


def _acquire_message_lock(phone: str, ttl: int = 15) -> bool:
    """Try to acquire a processing lock for this phone. Returns True if acquired."""
    r = _get_pending_payment_redis()
    if r:
        key = f"msg_lock:{phone}"
        return bool(r.set(key, "1", nx=True, ex=ttl))
    # In-memory fallback
    import time
    now = time.time()
    expiry = _message_locks.get(phone, 0)
    if now < expiry:
        return False  # Lock still held
    _message_locks[phone] = now + ttl
    return True


def _release_message_lock(phone: str):
    r = _get_pending_payment_redis()
    if r:
        r.delete(f"msg_lock:{phone}")
    else:
        _message_locks.pop(phone, None)


def _save_pending_modification(phone: str, event: dict, ttl: int = 600):
    """Store event pending modification (will be deleted when new booking is confirmed)."""
    r = _get_pending_payment_redis()
    if r:
        r.setex(f"pending_modification:{phone}", ttl, json_lib.dumps(event))
    else:
        _pending_modifications[phone] = event


def _get_pending_modification(phone: str) -> dict | None:
    r = _get_pending_payment_redis()
    if r:
        raw = r.get(f"pending_modification:{phone}")
        if raw:
            return json_lib.loads(raw)
        return None
    return _pending_modifications.get(phone)


def _delete_pending_modification(phone: str):
    r = _get_pending_payment_redis()
    if r:
        r.delete(f"pending_modification:{phone}")
    else:
        _pending_modifications.pop(phone, None)


def _save_pending_cancellation(phone: str, events: list[dict], ttl: int = 600):
    """Store events pending cancellation for this phone (10 min TTL)."""
    r = _get_pending_payment_redis()
    if r:
        r.setex(f"pending_cancellation:{phone}", ttl, json_lib.dumps(events))
    else:
        _pending_cancellations[phone] = events


def _get_pending_cancellation(phone: str) -> list[dict] | None:
    """Get events pending cancellation for this phone."""
    r = _get_pending_payment_redis()
    if r:
        raw = r.get(f"pending_cancellation:{phone}")
        if raw:
            return json_lib.loads(raw)
        return None
    return _pending_cancellations.get(phone)


def _delete_pending_cancellation(phone: str):
    r = _get_pending_payment_redis()
    if r:
        r.delete(f"pending_cancellation:{phone}")
    else:
        _pending_cancellations.pop(phone, None)


def _save_pending_payment(phone: str, data: dict, ttl: int = 1800):
    """Store pending payment keyed by phone (used as external_reference in MP)."""
    r = _get_pending_payment_redis()
    if r:
        r.setex(f"pending_payment:{phone}", ttl, json_lib.dumps(data))


def _get_and_delete_pending_payment(payment: dict) -> dict | None:
    """Look up pending payment using external_reference from the MP payment object."""
    r = _get_pending_payment_redis()
    if not r:
        return None
    phone = payment.get("external_reference", "")
    if not phone:
        return None
    key = f"pending_payment:{phone}"
    raw = r.get(key)
    if raw:
        r.delete(key)
        return json_lib.loads(raw)
    return None


def _configured_calendar_consultants() -> list[dict]:
    return configured_consultants(CONFIG.get("booking", {}))


def _get_calendar_client(
    consultant_name: str | None = None,
    *,
    calendar_id: str | None = None,
) -> CalendarClient | None:
    if not CONFIG.get("modules", {}).get("booking"):
        return None
    global _calendar_client
    booking_cfg = CONFIG["booking"]
    consultants = _configured_calendar_consultants()
    consultant = None
    if calendar_id:
        consultant = next(
            (item for item in consultants if item["calendar_id"] == calendar_id),
            None,
        )
        if consultant is None and booking_cfg.get("calendar_id") != calendar_id:
            raise ValueError("The calendar for this appointment is not configured.")
    elif consultants:
        consultant = resolve_consultant(booking_cfg, consultant_name)

    target_calendar_id = (
        calendar_id
        or (consultant["calendar_id"] if consultant else booking_cfg["calendar_id"])
    )
    existing = _calendar_clients.get(target_calendar_id)
    if existing is not None:
        return existing
    if target_calendar_id == booking_cfg.get("calendar_id") and _calendar_client is not None:
        return _calendar_client

    client = CalendarClient(
            calendar_id=target_calendar_id,
            calendar_owner_email=(
                consultant.get("calendar_owner_email", booking_cfg.get("calendar_owner_email", ""))
                if consultant
                else booking_cfg["calendar_owner_email"]
            ),
            business_hours=booking_cfg["business_hours"],
            timezone=CONFIG["client"].get("timezone", "America/Argentina/Buenos_Aires"),
        )
    _calendar_clients[target_calendar_id] = client
    if target_calendar_id == booking_cfg.get("calendar_id"):
        _calendar_client = client
    return client


def _get_calendar_clients() -> list[CalendarClient]:
    consultants = _configured_calendar_consultants()
    if consultants:
        return [
            _get_calendar_client(consultant["name"])
            for consultant in consultants
        ]
    calendar = _get_calendar_client()
    return [calendar] if calendar else []


def _calendar_client_for_event(event: dict) -> CalendarClient | None:
    calendar_id = event.get("calendar_id")
    if calendar_id:
        return _get_calendar_client(calendar_id=calendar_id)
    return _get_calendar_client()


def _find_upcoming_events_by_phone(phone: str) -> list[dict]:
    events = []
    consultants = _configured_calendar_consultants()
    if consultants:
        calendars = [
            (consultant["name"], _get_calendar_client(consultant["name"]))
            for consultant in consultants
        ]
    else:
        calendars = [("", _get_calendar_client())]
    for consultant_name, calendar in calendars:
        if calendar is None:
            continue
        matching = calendar.find_upcoming_events_by_phone(phone)
        if consultant_name:
            for event in matching:
                event["consultant"] = consultant_name
        events.extend(matching)
    return sorted(events, key=lambda event: event.get("start_datetime", ""))


def _patient_has_upcoming_appointment(patient_name: str) -> bool:
    return any(
        calendar.has_upcoming_appointment_for_patient(patient_name)
        for calendar in _get_calendar_clients()
    )


def _find_service(service_name: str) -> dict | None:
    services = CONFIG.get("booking", {}).get("services", [])
    name = service_name.lower()
    # Exact match first
    for s in services:
        if s["name"].lower() == name:
            return s
    # Substring match: prioritize longest match to avoid partial matches (e.g. "Dental Cleaning" matching before "Cleaning + Checkup (bundle)")
    matches = []
    for s in services:
        s_lower = s["name"].lower()
        if name in s_lower or s_lower in name:
            matches.append(s)
    if matches:
        # Return the service with the longest name (most specific match)
        return max(matches, key=lambda s: len(s["name"]))
    logger.warning("Service not found: '%s'. Available: %s", service_name,
                   [s["name"] for s in services])
    return None


def _find_location(location_name: str) -> dict | None:
    locations = CONFIG.get("booking", {}).get("locations", [])
    name = location_name.lower()
    for loc in locations:
        if loc["name"].lower() == name:
            return loc
    # Substring match
    for loc in locations:
        loc_lower = loc["name"].lower()
        if name in loc_lower or loc_lower in name:
            return loc
    logger.warning("Location not found: '%s'. Available: %s", location_name,
                   [loc["name"] for loc in locations])
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    reminder_task = None
    global _dashboard_store
    _dashboard_store = None
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        _dashboard_store = DashboardStore(database_url)
        await asyncio.to_thread(_dashboard_store.initialize)
    if CONFIG.get("modules", {}).get("reminders"):
        reminder_task = asyncio.create_task(_run_daily_reminders())
        timezone = CONFIG.get("client", {}).get("timezone", "Asia/Singapore")
        logger.info("WhatsApp reminders scheduled for 09:00 %s", timezone)
    try:
        yield
    finally:
        if reminder_task:
            reminder_task.cancel()
            try:
                await reminder_task
            except asyncio.CancelledError:
                pass
        _dashboard_store = None


async def _sally_should_reply(phone: str, contact_label: str | None = None) -> bool:
    """Register the contact and report whether the dashboard switches allow Sally to reply."""
    if _dashboard_store is None:
        return True
    try:
        return await asyncio.to_thread(
            _dashboard_store.register_contact, "whatsapp", phone, contact_label
        )
    except Exception:
        logger.exception("Could not read dashboard switches; Sally will reply")
        return True


def _booking_intent_fallback(history: list[dict]) -> str:
    last_assistant_message = next(
        (
            str(message.get("content", ""))
            for message in reversed(history)
            if message.get("role") == "assistant"
        ),
        "",
    )
    if re.search(r"\bappointment confirmed\b", last_assistant_message, re.IGNORECASE):
        return "You're welcome! The appointment was confirmed in my previous message."
    return (
        "I can't complete a booking without first showing you the details and getting "
        "your confirmation. Please ask me to book if you'd like to make an appointment."
    )


async def _reminder_allowed(channel: str, contact_id: str) -> bool:
    """Respect the dashboard switches for reminders; if they can't be read, do not send."""
    if _dashboard_store is None:
        return True
    try:
        return await asyncio.to_thread(_dashboard_store.is_contact_enabled, channel, contact_id)
    except Exception:
        logger.exception("Could not read dashboard switches; reminder not sent")
        return False


async def _pause_for_handover(channel: str, contact_id: str, reason: str) -> None:
    if _dashboard_store is None:
        logger.error("Cannot hand over chat to staff: dashboard database is unavailable")
        return
    try:
        await asyncio.to_thread(_dashboard_store.pause_for_handover, channel, contact_id, reason)
    except Exception:
        logger.exception("Could not pause chat for staff handover")


async def _send_scheduled_reminders() -> dict:
    if not CONFIG.get("modules", {}).get("reminders"):
        logger.info("Scheduled reminders skipped because the reminders module is disabled")
        return {"sent": 0}

    calendars = _get_calendar_clients()
    if not calendars:
        logger.warning("Scheduled reminders skipped because no booking calendars are configured")
        return {"sent": 0}
    calendar_service = calendars[0]._service

    sent = await send_reminders(CONFIG, calendar_service, WA, should_send=_reminder_allowed)
    result = {"sent": sent}
    logger.info("WhatsApp reminders complete: %s", result)
    return result


async def _run_daily_reminders() -> None:
    timezone = CONFIG.get("client", {}).get("timezone", "Asia/Singapore")
    await run_daily_reminders(timezone, "WhatsApp", _send_scheduled_reminders)


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("DASHBOARD_SESSION_SECRET") or secrets.token_urlsafe(32),
    session_cookie="sally_dashboard_session",
    max_age=8 * 60 * 60,
    same_site="lax",
    https_only=os.environ.get("DASHBOARD_COOKIE_SECURE", "true").casefold() != "false",
)


def _dashboard_accounts() -> dict[str, str] | None:
    raw = os.environ.get("DASHBOARD_ACCOUNTS")
    if raw is None:
        username = os.environ.get("DASHBOARD_USERNAME", "")
        password = os.environ.get("DASHBOARD_PASSWORD", "")
        accounts = {username: password}
    else:
        def unique_accounts(pairs: list[tuple[str, str]]) -> dict[str, str]:
            result = {}
            for username, password in pairs:
                if username in result:
                    raise ValueError("Duplicate dashboard username")
                result[username] = password
            return result

        try:
            accounts = json_lib.loads(raw, object_pairs_hook=unique_accounts)
        except (ValueError, TypeError):
            return None

    if not isinstance(accounts, dict) or not accounts:
        return None
    if any(
        not isinstance(username, str)
        or not username.strip()
        or username != username.strip()
        or not isinstance(password, str)
        or len(password) < 12
        for username, password in accounts.items()
    ):
        return None
    return accounts


def _dashboard_auth_configured() -> bool:
    return len(os.environ.get("DASHBOARD_SESSION_SECRET", "").strip()) >= 32 and (
        _dashboard_accounts() is not None
    )


def _dashboard_session_tag(username: str, password: str) -> str:
    payload = json_lib.dumps([username, password], ensure_ascii=True).encode("utf-8")
    return hmac.digest(
        os.environ["DASHBOARD_SESSION_SECRET"].encode("utf-8"), payload, "sha256"
    ).hex()


def _dashboard_session_username(request: Request) -> str | None:
    if not _dashboard_auth_configured():
        return None
    username = request.session.get("dashboard_username")
    tag = request.session.get("dashboard_session_tag")
    accounts = _dashboard_accounts()
    if not isinstance(username, str) or not isinstance(tag, str) or accounts is None:
        return None
    password = accounts.get(username)
    if password is None or not secrets.compare_digest(
        _dashboard_session_tag(username, password), tag
    ):
        return None
    return username


def _dashboard_html_response(content: str, status_code: int = 200) -> HTMLResponse:
    return HTMLResponse(
        content,
        status_code=status_code,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": (
                "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                "form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
            ),
        },
    )


def _csrf_token(request: Request) -> str:
    token = request.session.get("csrf_token")
    if not isinstance(token, str):
        token = secrets.token_urlsafe(32)
        request.session["csrf_token"] = token
    return token


def _csrf_matches(request: Request, submitted: str) -> bool:
    expected = request.session.get("csrf_token")
    return isinstance(expected, str) and secrets.compare_digest(expected, submitted)


def _render_dashboard_login(request: Request, error: str = "", status_code: int = 200) -> HTMLResponse:
    page = (
        Path(__file__).resolve().parent.parent / "public" / "dashboard-login.html"
    ).read_text(encoding="utf-8")
    page = page.replace("{{CSRF_TOKEN}}", _csrf_token(request))
    page = page.replace("{{ERROR}}", html.escape(error))
    page = page.replace("{{ERROR_HIDDEN}}", "" if error else "hidden")
    return _dashboard_html_response(page, status_code)


async def require_dashboard_session(request: Request) -> str:
    username = _dashboard_session_username(request)
    if username is None:
        raise HTTPException(status_code=401, detail="Dashboard login required.")
    return username


@app.get("/")
async def health():
    return {"status": "ok"}


@app.get("/dashboard", include_in_schema=False)
async def dashboard_preview(request: Request):
    if not _dashboard_auth_configured():
        return _render_dashboard_login(
            request,
            "Dashboard login isn't configured yet. Ask the administrator to set "
            "DASHBOARD_ACCOUNTS with passwords of at least 12 characters, and "
            "a 32-character-or-longer DASHBOARD_SESSION_SECRET.",
            status_code=503,
        )
    if not _dashboard_session_username(request):
        return RedirectResponse("/dashboard/login", status_code=303)

    page = (Path(__file__).resolve().parent.parent / "public" / "dashboard.html").read_text(
        encoding="utf-8"
    )
    logout_form = (
        '<form method="post" action="/dashboard/logout">'
        f'<input type="hidden" name="csrf_token" value="{_csrf_token(request)}">'
        '<button type="submit">Sign out</button></form>'
    )
    csrf = _csrf_token(request)
    global_enabled = True
    contacts: list[dict] = []
    if _dashboard_store is not None:
        try:
            global_enabled = await asyncio.to_thread(_dashboard_store.get_global_enabled)
            contacts = await asyncio.to_thread(_dashboard_store.list_contacts)
        except Exception:
            logger.exception("Could not load dashboard switches")
            return _dashboard_html_response("Could not load the switches. Please retry.", 503)
        global_html = _switch_row_html("All chats", "", csrf, global_enabled, "global", "", "")
        new_contacts_html = _switch_row_html(
            "New chats",
            "Whether Sally replies to numbers she hasn't seen before",
            csrf,
            await asyncio.to_thread(_dashboard_store.get_new_contacts_enabled),
            "new_contacts",
            "",
            "",
        )
        import_html = _import_form_html(csrf, request.session.pop("import_result", ""))
        contacts_html = "".join(
            _switch_row_html(
                str(c["contact_label"] or c["contact_id"]),
                f"{'WhatsApp' if c['channel'] == 'whatsapp' else 'Telegram'} · {c['contact_id']}"
                + (
                    f" · Needs staff ({html.escape(c['handover_reason'])} question)"
                    if c.get("handover_reason") and not c["bot_enabled"]
                    else ""
                ),
                csrf,
                bool(c["bot_enabled"]),
                "contact",
                c["channel"],
                c["contact_id"],
            )
            for c in contacts
        ) or '<p class="muted">No chats yet.</p>'
    else:
        global_html = contacts_html = new_contacts_html = import_html = (
            '<p class="muted">Switches are unavailable: DATABASE_URL is not configured.</p>'
        )
    page = (
        page.replace("<!-- DASHBOARD_AUTH -->", logout_form)
        .replace("<!-- DASHBOARD_GLOBAL -->", global_html)
        .replace("<!-- DASHBOARD_NEW_CONTACTS -->", new_contacts_html)
        .replace("<!-- DASHBOARD_IMPORT -->", import_html)
        .replace("<!-- DASHBOARD_CONTACTS -->", contacts_html)
    )
    return _dashboard_html_response(page)


def _import_form_html(csrf: str, result: str) -> str:
    esc = html.escape
    return (
        (f'<p class="state on">{esc(result)}</p>' if result else "")
        + '<form method="post" action="/dashboard/import">'
        f'<input type="hidden" name="csrf_token" value="{esc(csrf)}">'
        '<textarea name="numbers" rows="6" required placeholder="+65 9123 4567, Jamie Tan&#10;'
        '91234567&#10;0812-3456-789, Budi"></textarea>'
        '<label><input type="checkbox" name="enabled" value="true"> '
        "Switch Sally ON for these chats (leave unticked to import them as OFF)</label>"
        '<button type="submit">Import chats</button></form>'
    )


def parse_contact_lines(text: str) -> tuple[list[tuple[str, str | None]], list[str]]:
    """Parse pasted CSV lines into unique (number, label) pairs plus unusable entries."""
    contacts: dict[str, str | None] = {}
    skipped: list[str] = []
    for line in text.splitlines():
        cells = [cell.strip().strip('"') for cell in re.split(r"[,;\t]", line) if cell.strip()]
        raw = next((c for c in cells if sum(ch.isdigit() for ch in c) >= 7), None)
        if raw is None:
            if any(ch.isdigit() for ch in line):
                skipped.append(line.strip()[:40])
            continue
        number = normalize_phone(raw)
        if number is None:
            skipped.append(raw[:40])
            continue
        label = next((c for c in cells if c != raw and not any(ch.isdigit() for ch in c)), None)
        contacts.setdefault(number, label)
    return list(contacts.items()), skipped


@app.post("/dashboard/import", include_in_schema=False)
async def dashboard_import(request: Request):
    if not _dashboard_session_username(request):
        return RedirectResponse("/dashboard/login", status_code=303)
    body = await request.body()
    if len(body) > 262144:
        raise HTTPException(status_code=413, detail="Request too large.")
    fields = parse_qs(body.decode("utf-8", errors="replace"), keep_blank_values=True)
    if not _csrf_matches(request, fields.get("csrf_token", [""])[0]):
        raise HTTPException(status_code=403, detail="Invalid dashboard request token.")
    if _dashboard_store is None:
        raise HTTPException(status_code=503, detail="Dashboard storage is not configured.")
    contacts, skipped = parse_contact_lines(fields.get("numbers", [""])[0])
    enabled = fields.get("enabled", [""])[0] == "true"
    added, existing = await asyncio.to_thread(
        _dashboard_store.import_whatsapp_contacts, contacts, enabled
    )
    result = f"Added {added} chat(s) as {'ON' if enabled else 'OFF'}."
    if existing:
        result += f" {existing} already existed and kept their current switch."
    if skipped:
        result += f" Skipped {len(skipped)} unreadable number(s): {', '.join(skipped[:5])}."
    request.session["import_result"] = result
    return RedirectResponse("/dashboard", status_code=303)


def _switch_row_html(
    title: str, subtitle: str, csrf: str, enabled: bool, scope: str, channel: str, contact_id: str
) -> str:
    state = "ON" if enabled else "OFF"
    esc = html.escape
    return (
        '<form class="row" method="post" action="/dashboard/switch">'
        f'<div><div>{esc(title)}</div>'
        + (f'<div class="muted">{esc(subtitle)}</div>' if subtitle else "")
        + f'</div><div><span class="state {"on" if enabled else "off"}">Sally {state}</span> '
        f'<input type="hidden" name="csrf_token" value="{esc(csrf)}">'
        f'<input type="hidden" name="scope" value="{scope}">'
        f'<input type="hidden" name="channel" value="{esc(channel)}">'
        f'<input type="hidden" name="contact_id" value="{esc(contact_id)}">'
        f'<input type="hidden" name="enabled" value="{"false" if enabled else "true"}">'
        f'<button type="submit">Turn {"off" if enabled else "on"}</button></div></form>'
    )


@app.post("/dashboard/switch", include_in_schema=False)
async def dashboard_switch(request: Request):
    if not _dashboard_session_username(request):
        return RedirectResponse("/dashboard/login", status_code=303)
    body = await request.body()
    if len(body) > 4096:
        raise HTTPException(status_code=413, detail="Request too large.")
    fields = parse_qs(body.decode("utf-8", errors="replace"), keep_blank_values=True)
    if not _csrf_matches(request, fields.get("csrf_token", [""])[0]):
        raise HTTPException(status_code=403, detail="Invalid dashboard request token.")
    if _dashboard_store is None:
        raise HTTPException(status_code=503, detail="Dashboard storage is not configured.")
    enabled = fields.get("enabled", [""])[0] == "true"
    scope = fields.get("scope", [""])[0]
    try:
        if scope == "global":
            await asyncio.to_thread(_dashboard_store.set_global_enabled, enabled)
        elif scope == "new_contacts":
            await asyncio.to_thread(_dashboard_store.set_new_contacts_enabled, enabled)
        elif scope == "contact":
            updated = await asyncio.to_thread(
                _dashboard_store.set_contact_enabled,
                fields.get("channel", [""])[0],
                fields.get("contact_id", [""])[0],
                enabled,
            )
            if not updated:
                raise HTTPException(status_code=404, detail="Unknown chat.")
        else:
            raise HTTPException(status_code=400, detail="Invalid switch.")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid switch.")
    return RedirectResponse("/dashboard", status_code=303)


@app.get("/dashboard/login", include_in_schema=False)
async def dashboard_login_page(request: Request):
    if not _dashboard_auth_configured():
        return _render_dashboard_login(
            request,
            "Dashboard login isn't configured yet. Ask the administrator to set "
            "DASHBOARD_ACCOUNTS with passwords of at least 12 characters, and "
            "a 32-character-or-longer DASHBOARD_SESSION_SECRET.",
            status_code=503,
        )
    if _dashboard_session_username(request):
        return RedirectResponse("/dashboard", status_code=303)
    return _render_dashboard_login(request)


@app.post("/dashboard/login", include_in_schema=False)
async def dashboard_login(request: Request):
    if not _dashboard_auth_configured():
        return _render_dashboard_login(
            request,
            "Dashboard login isn't configured yet. Ask the administrator to complete setup.",
            status_code=503,
        )
    body = await request.body()
    if len(body) > 4096:
        return _render_dashboard_login(request, "Login request is too large.", status_code=413)
    fields = parse_qs(body.decode("utf-8", errors="replace"), keep_blank_values=True)
    submitted_csrf = fields.get("csrf_token", [""])[0]
    if not _csrf_matches(request, submitted_csrf):
        return _render_dashboard_login(
            request, "Your login session expired. Please try again.", status_code=400
        )

    submitted_username = fields.get("username", [""])[0]
    submitted_password = fields.get("password", [""])[0]
    accounts = _dashboard_accounts()
    configured_password = accounts.get(submitted_username) if accounts else None
    if configured_password is None or not secrets.compare_digest(
        configured_password.encode("utf-8"), submitted_password.encode("utf-8")
    ):
        return _render_dashboard_login(
            request, "The username or password is incorrect.", status_code=401
        )

    request.session.clear()
    request.session["dashboard_username"] = submitted_username
    request.session["dashboard_session_tag"] = _dashboard_session_tag(
        submitted_username, configured_password
    )
    return RedirectResponse("/dashboard", status_code=303)


@app.post("/dashboard/logout", include_in_schema=False)
async def dashboard_logout(request: Request):
    if not _dashboard_session_username(request):
        request.session.clear()
        return RedirectResponse("/dashboard/login", status_code=303)
    body = await request.body()
    fields = parse_qs(body.decode("utf-8", errors="replace"), keep_blank_values=True)
    if not _csrf_matches(request, fields.get("csrf_token", [""])[0]):
        raise HTTPException(status_code=403, detail="Invalid dashboard request token.")
    request.session.clear()
    return RedirectResponse("/dashboard/login", status_code=303)


@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(alias="hub.mode"),
    hub_verify_token: str = Query(alias="hub.verify_token"),
    hub_challenge: str = Query(alias="hub.challenge"),
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge)
    raise HTTPException(status_code=403)


@app.post("/webhook")
async def receive_message(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not validate_webhook_signature(body, signature, APP_SECRET):
        raise HTTPException(status_code=403, detail="Invalid signature")

    data = await request.json()

    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            if change.get("field") != "smb_message_echoes":
                continue
            value = change.get("value", {})
            metadata = value.get("metadata", {})
            echoes = value.get("message_echoes")
            if (
                metadata.get("phone_number_id") != os.environ["WHATSAPP_PHONE_NUMBER_ID"]
                or not isinstance(echoes, list)
                or not echoes
            ):
                raise HTTPException(status_code=400, detail="Invalid message echo event.")
            if _dashboard_store is None:
                logger.error("Cannot pause for message echo: dashboard database is unavailable")
                raise HTTPException(status_code=503, detail="Dashboard database is unavailable.")
            for echo in echoes:
                if not isinstance(echo, dict):
                    raise HTTPException(status_code=400, detail="Invalid message echo.")
                phone = echo.get("to")
                message_id = echo.get("id")
                if (
                    not isinstance(phone, str)
                    or not re.fullmatch(r"[1-9][0-9]{9,14}", phone)
                    or not isinstance(message_id, str)
                    or not message_id.strip()
                ):
                    raise HTTPException(status_code=400, detail="Invalid message echo.")
                try:
                    await asyncio.to_thread(_dashboard_store.pause_for_whatsapp_echo, phone, message_id)
                except Exception:
                    logger.exception("Could not save pause for message echo")
                    raise HTTPException(status_code=503, detail="Could not save chat pause.")

    try:
        change = next(
            change["value"]
            for entry in data["entry"]
            for change in entry["changes"]
            if change.get("field") != "smb_message_echoes"
        )

        # Ignore status updates (delivered, read, etc.)
        if "statuses" in change:
            return Response(status_code=200)

        message = change["messages"][0]
        phone = message["from"]
        msg_type = message.get("type")

    except (KeyError, IndexError, StopIteration):
        return Response(status_code=200)

    contact_label = None
    for contact in change.get("contacts", []):
        if contact.get("wa_id") == phone:
            profile = contact.get("profile", {})
            contact_label = str(profile.get("name", "")).strip() or None
            break
    if not await _sally_should_reply(phone, contact_label):
        logger.info("Message from %s left for a consultant (Sally is switched off)", phone)
        return Response(status_code=200)

    try:
        if msg_type == "text":
            user_text = message["text"]["body"]
        elif msg_type == "audio":
            media_id = message["audio"]["id"]
            try:
                audio_bytes, mime_type = await WA.download_media(media_id)
                user_text = transcribe_audio(audio_bytes, mime_type)
                logger.info("Audio transcribed for %s: %s", phone, user_text[:100])
            except Exception as e:
                logger.error("Audio transcription failed: %s", e)
                await WA.send_text(phone, "I couldn't process your audio. Could you send it as text instead?")
                return Response(status_code=200)
        else:
            await WA.send_text(phone, "I can only process text and audio messages for now.")
            return Response(status_code=200)

    except (KeyError, IndexError):
        return Response(status_code=200)

    # Prevent concurrent processing for the same phone number
    if not _acquire_message_lock(phone):
        logger.info("Message from %s skipped (already processing)", phone)
        # Still add to history so the next response has context
        HISTORY.add(phone, "user", user_text)
        return Response(status_code=200)

    try:
        await _process_message(phone, user_text)
    finally:
        _release_message_lock(phone)

    return Response(status_code=200)


async def _process_message(phone: str, user_text: str):
    """Process a single message with lock already held."""
    # Get conversation history
    history = HISTORY.get(phone)
    HISTORY.add(phone, "user", user_text)

    key = f"whatsapp:{phone}"
    pending = PENDING_BOOKINGS.get(key)
    booking_prompt = pending is not None and pending.awaiting_confirmation
    booking_question = awaiting_booking_information(history)
    if is_exact_reply(user_text, "APPROVE NOTE"):
        try:
            response = PENDING_BOOKINGS.approve_observations(key).summary
        except ValueError as exc:
            response = str(exc)
        HISTORY.add(phone, "assistant", response)
        await WA.send_text(phone, response)
        return
    if user_text.lstrip().casefold().startswith("note:") or is_exact_reply(user_text, "REMOVE NOTE"):
        try:
            note = None if is_exact_reply(user_text, "REMOVE NOTE") else user_text.lstrip()[5:]
            pending = PENDING_BOOKINGS.set_observations(key, note)
            response = pending.summary
        except ValueError as exc:
            response = str(exc)
        HISTORY.add(phone, "assistant", response)
        await WA.send_text(phone, response)
        return
    if is_exact_reply(user_text, "CONFIRM"):
        if pending and pending.awaiting_confirmation and pending.proposed_observations is not None:
            response = "Please approve or remove the proposed note before confirming the booking.\n\n" + pending.summary
            HISTORY.add(phone, "assistant", response)
            await WA.send_text(phone, response)
            return
        pending = PENDING_BOOKINGS.take(key)
        if pending is None:
            response = (
                "There is no active booking summary to confirm. "
                "Please ask me to prepare a new booking summary."
            )
            HISTORY.add(phone, "assistant", response)
            await WA.send_text(phone, response)
            return
        await _handle_booking_intent(
            phone, pending.details, "I'm checking your requested appointment.",
            pending_modification=pending.modification, use_saved_modification=True,
        )
        return
    if pending and has_booking_cancellation(user_text):
        PENDING_BOOKINGS.clear(key)
        _delete_pending_modification(phone)
        response = (
            "The pending booking has been cancelled; no appointment was booked. "
            "To cancel an existing appointment, please ask separately."
        )
        HISTORY.add(phone, "assistant", response)
        await WA.send_text(phone, response)
        return
    if booking_prompt and not is_exact_reply(user_text, "CONFIRM"):
        if is_short_acknowledgement(user_text):
            response = (
                "The appointment is not booked yet.\n\n" + pending.summary
                if pending.proposed_observations is not None
                else f"{BOOKING_NOT_CONFIRMED}\n\n{pending.summary}"
            )
            HISTORY.add(phone, "assistant", response)
            await WA.send_text(phone, response)
            return
        if preserves_pending_booking(user_text):
            await WA.send_text(phone, BOOKING_NOT_CONFIRMED)
        else:
            PENDING_BOOKINGS.invalidate(key)
            await WA.send_text(
                phone,
                "Your previous booking summary is no longer awaiting confirmation. "
                "If the details change, I'll show you a new summary to confirm.",
            )
    pending_cancellations = _get_pending_cancellation(phone) or []
    if (
        not (booking_prompt or booking_question)
        and len(pending_cancellations) == 1
        and pending_cancellations[0].get("rebook_from_scratch")
        and is_exact_reply(user_text, "YES")
    ):
        await _handle_cancellation_confirmed(
            phone,
            {"event_index": 1},
            "",
        )
        return
    if (
        not (booking_prompt or booking_question)
        and len(pending_cancellations) == 1
        and pending_cancellations[0].get("rebook_from_scratch")
        and is_exact_reply(user_text, "NO")
    ):
        _delete_pending_cancellation(phone)
        response = (
            "Understood. Your appointment is still booked. Without a valid MC, it "
            "cannot be rescheduled within 36 hours. Please contact the clinic if you "
            "need help."
        )
        HISTORY.add(phone, "assistant", response)
        await WA.send_text(phone, response)
        return

    pending_modification = _get_pending_modification(phone)
    if (
        not (booking_prompt or booking_question)
        and pending_modification
        and pending_modification.get("awaiting_mc_confirmation")
        and (
            is_exact_reply(user_text, "YES")
            or is_exact_reply(user_text, "NO")
        )
    ):
        await _handle_rescheduling_mc_response(
            phone,
            {"has_valid_mc": is_exact_reply(user_text, "YES")},
        )
        return
    if not (booking_prompt or booking_question):
        if await _handle_whatsapp_attendance_confirmation(phone, user_text):
            return

    # Proactively detect modification intent from user text and save pending state
    # This ensures the state survives even if many messages pass before booking_confirmed
    if not booking_prompt and not _get_pending_modification(phone):
        words = set(user_text.lower().split())
        if words & _MODIFICATION_KEYWORDS:
            events = _find_upcoming_events_by_phone(phone)
            if len(events) == 1:
                _save_pending_modification(phone, events[0])
                logger.info("Proactive modification state saved for %s", phone)

    # Get AI response
    ai_response = get_ai_response(user_text, history, CONFIG, KNOWLEDGE)

    # Check for intent
    intent, visible_response = extract_booking_intent(ai_response)

    if intent and intent.get("intent") == "handover":
        reply = compose_handover_reply(visible_response)
        HISTORY.add(phone, "assistant", reply)
        await WA.send_text(phone, reply)
        await _pause_for_handover("whatsapp", phone, handover_reason(intent))
        return

    if intent and intent.get("intent") == "booking_draft":
        details = dict(intent)
        proposal = existing_proposal(details, pending) if pending else extract_general_observations(history, user_text)
        preserve_observations(details, pending)
        modification = pending.modification if pending else _get_pending_modification(phone)
        if modification:
            try:
                needs_mc = (
                    modification.get("awaiting_mc_confirmation")
                    or _modification_requires_mc(phone, modification)
                )
            except (TypeError, ValueError):
                logger.exception("Could not verify rescheduling window for phone %s", phone)
                response = "I couldn't verify the rescheduling window. Please contact the clinic directly."
                HISTORY.add(phone, "assistant", response)
                await WA.send_text(phone, response)
                return
        else:
            needs_mc = False
        if needs_mc:
            modification["awaiting_mc_confirmation"] = True
            _save_pending_modification(phone, modification)
            response = _RESCHEDULING_MC_QUESTION
            HISTORY.add(phone, "assistant", response)
            await WA.send_text(phone, response)
            return
        PENDING_BOOKINGS.clear(key)
        locations = CONFIG.get("booking", {}).get("locations", [])
        if not details.get("location") and len(locations) == 1:
            details["location"] = locations[0]["name"]
        locations = CONFIG.get("booking", {}).get("locations", [])
        if not details.get("location") and len(locations) == 1:
            details["location"] = locations[0]["name"]
        assignment_key = _consultant_assignment_key(phone, details)
        if not modification and not details.get("consultant") and assignment_key is not None:
            details["consultant"] = _pending_consultant_assignments.get(assignment_key)
        try:
            if modification:
                inherit_modification_consultant(details, modification, _configured_calendar_consultants())
            pending = PENDING_BOOKINGS.put(
                key, details, modification, proposed_observations=proposal,
                timezone=CONFIG.get("client", {}).get("timezone", "UTC"),
            )
        except ValueError as exc:
            response = str(exc)
        else:
            response = pending.summary
            if modification:
                _delete_pending_modification(phone)
        HISTORY.add(phone, "assistant", response)
        try:
            await WA.send_text(phone, response)
        except Exception:
            PENDING_BOOKINGS.clear(key)
            raise
        if assignment_key is not None and PENDING_BOOKINGS.get(key) is not None:
            _pending_consultant_assignments.pop(assignment_key, None)
        return

    if intent and intent.get("intent") == "availability_check":
        def remember(checked: dict, consultant_name: str) -> None:
            assignment_key = _consultant_assignment_key(phone, checked)
            if assignment_key is not None:
                _pending_consultant_assignments[assignment_key] = consultant_name

        try:
            response = await asyncio.to_thread(
                check_slot_availability, CONFIG, intent, _get_calendar_client, remember
            )
        except ValueError as exc:
            response = str(exc)
        except Exception:
            logger.exception("WhatsApp calendar availability check failed")
            response = (
                "I couldn't check the calendar just now. The time is not confirmed as available; "
                "please try again shortly."
            )
        HISTORY.add(phone, "assistant", response)
        await WA.send_text(phone, response)
        return

    if intent and intent.get("intent") == "booking_confirmed":
        logger.warning("Ignored AI booking intent without a saved booking confirmation")
        response = _booking_intent_fallback(history)
        HISTORY.add(phone, "assistant", response)
        await WA.send_text(phone, response)
        return

    if intent and intent.get("intent") == "appointment_lookup":
        await _handle_appointment_lookup(phone)
        return

    if intent and intent.get("intent") == "availability_range_check":
        try:
            response = await asyncio.to_thread(
                check_availability_range,
                CONFIG,
                intent,
                _get_calendar_client,
            )
        except ValueError as exc:
            response = str(exc)
        except Exception:
            logger.exception("WhatsApp availability range check failed")
            response = "I couldn't check availability just now. Please try again shortly."
        HISTORY.add(phone, "assistant", response)
        await WA.send_text(phone, response)
        return

    if intent and CONFIG.get("modules", {}).get("booking"):
        intent_type = intent.get("intent", "")
        logger.info("Intent detected: %s", intent)

        if intent_type == "booking_confirmed":
            await _handle_booking_intent(phone, intent, visible_response)
        elif intent_type == "cancellation_request":
            await _handle_cancellation_request(phone, visible_response)
        elif intent_type == "cancellation_confirmed":
            await _handle_cancellation_confirmed(phone, intent, visible_response)
        elif intent_type == "modification_request":
            await _handle_modification_request(
                phone,
                visible_response,
                valid_mc_confirmed=intent.get("valid_mc_confirmed") is True,
            )
        elif intent_type == "modification_confirmed":
            await _handle_modification_confirmed(phone, intent, visible_response)
        elif intent_type == "rescheduling_mc_response":
            await _handle_rescheduling_mc_response(phone, intent)
        elif intent_type == "rescheduling_handoff":
            await _handle_rescheduling_handoff(phone)
        else:
            HISTORY.add(phone, "assistant", visible_response or ai_response)
            await WA.send_text(phone, visible_response or ai_response)
    else:
        response = visible_response if intent else ai_response
        if (
            is_booking_confirmation_prompt(response)
            or re.search(r"\b(?:reply|send|type)\s+[*'\"`]*confirm\b", response, re.IGNORECASE)
        ):
            active = PENDING_BOOKINGS.get(key)
            response = (
                f"{BOOKING_NOT_CONFIRMED}\n\n{active.summary}"
                if active and active.awaiting_confirmation
                else "I couldn't prepare a booking summary yet. Please ask me to review the booking details again."
            )
        HISTORY.add(phone, "assistant", response)
        await WA.send_text(phone, response)


async def _handle_whatsapp_attendance_confirmation(phone: str, user_text: str) -> bool:
    try:
        pending_events = [
            event
            for event in await asyncio.to_thread(_find_upcoming_events_by_phone, phone)
            if event.get("reminder_sent_for")
            and event.get("reminder_sent_for") == event.get("start_datetime", "")[:10]
            and event.get("attendance_confirmed_for") != event.get("reminder_sent_for")
        ]
    except Exception:
        logger.exception("Could not check pending WhatsApp reminders")
        response = (
            "I couldn't check your appointment reminders just now. "
            "Attendance hasn't been confirmed; please try again later."
        )
        await WA.send_text(phone, response)
        return is_exact_reply(user_text, "YES")
    if not is_exact_reply(user_text, "YES"):
        if pending_events:
            HISTORY.add(phone, "assistant", REMINDER_NOT_CONFIRMED)
            await WA.send_text(phone, REMINDER_NOT_CONFIRMED)
            return is_short_acknowledgement(user_text)
        return False
    if not pending_events:
        response = (
            "I couldn't find an appointment reminder awaiting confirmation for this number. "
            "Please reply YES to the reminder message or contact the clinic for help."
        )
    elif len(pending_events) > 1:
        choices = "\n".join(
            f"- {event.get('patient_name') or 'Patient'}: "
            f"{event.get('date')} at {event.get('time')}"
            for event in pending_events
        )
        response = (
            "I found more than one appointment awaiting confirmation, so I haven't marked "
            "any as confirmed. Please contact the clinic and specify which appointment "
            "you are confirming:\n"
            f"{choices}"
        )
    else:
        event = pending_events[0]
        calendar = _calendar_client_for_event(event)
        if calendar is None:
            raise RuntimeError("The calendar for the appointment is unavailable.")
        calendar.mark_event_attendance_confirmed(event["id"])
        response = (
            f"Thank you. Attendance is confirmed for {event.get('patient_name') or 'the patient'} "
            f"on {event.get('date')} at {event.get('time')}."
        )

    HISTORY.add(phone, "assistant", response)
    await WA.send_text(phone, response)
    return True


def _conversation_suggests_modification(history: list[dict]) -> bool:
    """Check recent conversation history for modification-related keywords."""
    # Look at the last 6 messages (3 exchanges) for modification signals
    recent = history[-6:] if len(history) > 6 else history
    for msg in recent:
        if msg["role"] == "user":
            words = set(msg["content"].lower().split())
            if words & _MODIFICATION_KEYWORDS:
                return True
    return False


async def _handle_booking_intent(
    phone: str, intent: dict, visible_response: str, *,
    pending_modification: dict | None = None,
    use_saved_modification: bool = False,
):
    """Process a confirmed booking intent."""
    pending_mod = (
        pending_modification if use_saved_modification else _get_pending_modification(phone)
    )
    if pending_mod:
        try:
            needs_mc = _modification_requires_mc(phone, pending_mod)
        except (TypeError, ValueError):
            logger.exception("Could not verify rescheduling window for phone %s", phone)
            response = "I couldn't verify the rescheduling window. Please contact the clinic directly."
            HISTORY.add(phone, "assistant", response)
            await WA.send_text(phone, response)
            return
        if pending_mod.get("awaiting_mc_confirmation") or needs_mc:
            pending_mod["awaiting_mc_confirmation"] = True
            _save_pending_modification(phone, pending_mod)
            HISTORY.add(phone, "assistant", _RESCHEDULING_MC_QUESTION)
            await WA.send_text(phone, _RESCHEDULING_MC_QUESTION)
            return
    consultant_name = str(intent.get("consultant", "")).strip() or None
    consultants = _configured_calendar_consultants()
    if pending_mod and not consultant_name:
        consultant_name = pending_mod.get("consultant")
        if not consultant_name and consultants:
            consultant_name = next(
                (
                    consultant["name"]
                    for consultant in consultants
                    if consultant["calendar_id"] == pending_mod.get("calendar_id")
                ),
                None,
            )
    selected_consultant = None
    try:
        if consultants and (consultant_name or len(consultants) == 1):
            selected_consultant = resolve_consultant(
                CONFIG["booking"],
                consultant_name,
            )
            if selected_consultant is None:
                raise ValueError("The selected consultant is not configured.")
            consultant_name = selected_consultant["name"]
            cal = _get_calendar_client(consultant_name)
        elif consultants:
            cal = None
        else:
            cal = _get_calendar_client()
    except ValueError as exc:
        logger.warning("Unable to select booking calendar: %s", exc)
        error_msg = "Our administrative team will assist you with this booking."
        HISTORY.add(phone, "assistant", error_msg)
        await WA.send_text(phone, error_msg)
        return
    if cal is None and not (consultants and selected_consultant is None):
        error_msg = "Our administrative team will assist you with this booking."
        HISTORY.add(phone, "assistant", error_msg)
        await WA.send_text(phone, error_msg)
        return

    logger.info("Processing confirmed booking details")
    service = _find_service(intent.get("service", ""))
    location = _find_location(intent.get("location", ""))

    if not service or not location:
        logger.error("Service or location not found. service='%s' location='%s'", intent.get('service'), intent.get('location'))
        error_msg = "There was a problem processing your booking. Please contact us directly."
        HISTORY.add(phone, "assistant", error_msg)
        await WA.send_text(phone, error_msg)
        return

    required_patient_fields = (
        "child_name",
        "parent_name",
        "child_dob",
        "referral_name",
        "patient_resides_in_singapore",
    )
    missing_patient_fields = [
        key for key in required_patient_fields
        if key not in intent or not str(intent[key]).strip()
    ]
    if missing_patient_fields:
        error_msg = (
            "To complete the booking request, please provide the patient's full name "
            "and date of birth, the parent's full name, and the referring person's "
            "name (or say there is no referral), and confirm whether the patient "
            "resides in Singapore."
        )
        HISTORY.add(phone, "assistant", error_msg)
        await WA.send_text(phone, error_msg)
        return
    resides_in_singapore = str(intent["patient_resides_in_singapore"]).strip().casefold()
    if resides_in_singapore not in {"yes", "no", "true", "false"}:
        error_msg = "Please confirm whether the patient resides in Singapore (yes or no)."
        HISTORY.add(phone, "assistant", error_msg)
        await WA.send_text(phone, error_msg)
        return
    if resides_in_singapore in {"no", "false"}:
        logger.warning(
            "Routing WhatsApp booking to admin: booking intent reports patient residency outside Singapore"
        )
        error_msg = "Our administrative team will assist you with this booking."
        HISTORY.add(phone, "assistant", error_msg)
        await WA.send_text(phone, error_msg)
        return
    try:
        child_dob = validate_patient_dob(
            intent["child_dob"], CONFIG.get("client", {}).get("timezone", "UTC"),
        )
    except ValueError as exc:
        HISTORY.add(phone, "assistant", str(exc))
        await WA.send_text(phone, str(exc))
        return

    try:
        booking_date = date.fromisoformat(intent["date"])
        booking_time = dt_time.fromisoformat(intent["time"])
    except (ValueError, KeyError) as e:
        logger.warning("Invalid date/time in intent: %s", e)
        error_msg = "There was a problem with your booking date. Could you try again?"
        HISTORY.add(phone, "assistant", error_msg)
        await WA.send_text(phone, error_msg)
        return

    if booking_date.weekday() == 5:
        error_msg = "Our administrative team will assist you with this booking."
        HISTORY.add(phone, "assistant", error_msg)
        await WA.send_text(phone, error_msg)
        return

    age_group = patient_age_group(child_dob, booking_date)

    try:
        validate_booking_slot(
            booking_date,
            booking_time,
            service["duration_minutes"],
            CONFIG["booking"]["business_hours"],
            location.get("days", []),
            CONFIG["client"].get("timezone", "UTC"),
        )
    except ValueError as e:
        logger.info("Booking slot rejected: %s", e)
        error_msg = "Could you share another preferred date and time?"
        HISTORY.add(phone, "assistant", error_msg)
        await WA.send_text(phone, error_msg)
        return

    slot_available = False
    if consultants and selected_consultant is None:
        available_consultants = []
        for candidate in consultants:
            if consultant_slot_is_reserved(
                candidate["name"],
                booking_date,
                booking_time,
            ):
                continue
            candidate_calendar = _get_calendar_client(candidate["name"])
            if candidate_calendar.is_slot_available(
                booking_date,
                booking_time,
                service["duration_minutes"],
            ):
                available_consultants.append(candidate)

        selected_consultant = allocate_consultant(available_consultants)
        if selected_consultant is None:
            error_msg = (
                f"That time isn't available with either consultant. "
                "Please choose another date and time."
            )
            HISTORY.add(phone, "assistant", error_msg)
            await WA.send_text(phone, error_msg)
            return
        consultant_name = selected_consultant["name"]
        cal = _get_calendar_client(consultant_name)
        slot_available = True

    if consultant_slot_is_reserved(consultant_name, booking_date, booking_time):
        error_msg = (
            f"That time isn't available with {consultant_name}. "
            "Please choose another date and time."
        )
        HISTORY.add(phone, "assistant", error_msg)
        await WA.send_text(phone, error_msg)
        return

    slot = Slot(date=booking_date, start_time=booking_time, location=location["name"])

    if not slot_available and not cal.is_slot_available(
        booking_date,
        booking_time,
        service["duration_minutes"],
    ):
        sorry = (
            f"That time slot ({booking_date.strftime('%m/%d/%Y')} at {booking_time.strftime('%H:%M')}) "
            f"is already taken. Would you like another time that same day or a different day?"
        )
        HISTORY.add(phone, "assistant", sorry)
        await WA.send_text(phone, sorry)
        return

    payments_enabled = CONFIG.get("modules", {}).get("payments", False)

    # Fallback: if Haiku skipped modification_request, detect it from conversation history.
    if not pending_mod and not use_saved_modification:
        history = HISTORY.get(phone)
        if _conversation_suggests_modification(history):
            events = _find_upcoming_events_by_phone(phone)
            if events:
                # Use the first (soonest) event as the one being modified
                pending_mod = events[0]
                if consultants and not intent.get("consultant"):
                    pending_consultant = pending_mod.get("consultant")
                    if pending_consultant:
                        cal = _get_calendar_client(pending_consultant)
                logger.info("Modification detected from conversation context for phone %s", phone)

    if pending_mod and pending_mod.get("date", "").startswith("Saturday"):
        _delete_pending_modification(phone)
        HISTORY.add(phone, "assistant", _SATURDAY_RESCHEDULING_MESSAGE)
        await WA.send_text(phone, _SATURDAY_RESCHEDULING_MESSAGE)
        return

    if not pending_mod and _patient_has_upcoming_appointment(intent["child_name"]):
        logger.warning(
            "Routing WhatsApp booking to admin: matching upcoming patient appointment found"
        )
        error_msg = "Our administrative team will assist you with this booking."
        HISTORY.add(phone, "assistant", error_msg)
        await WA.send_text(phone, error_msg)
        return

    if payments_enabled:
        intent["consultant"] = consultant_name
        await _handle_payment_flow(
            phone,
            intent,
            visible_response,
            service,
            location,
            slot,
            cal,
            age_group,
            pending_mod,
        )
    else:
        cal.create_event(
            service_name=service["name"],
            user_name=intent["parent_name"],
            user_phone=phone,
            slot=slot,
            duration_minutes=service["duration_minutes"],
            price=service["price"],
            cancellation_policy=CONFIG["booking"].get("cancellation_policy", ""),
            child_name=intent["child_name"],
            parent_name=intent["parent_name"],
            referral_name=intent["referral_name"],
            patient_age_group=age_group,
            observations=intent.get("observations", ""),
        )
        modification_note = ""
        if pending_mod:
            try:
                old_calendar = _calendar_client_for_event(pending_mod)
                if old_calendar is None:
                    raise RuntimeError("The previous appointment calendar is unavailable.")
                old_calendar.delete_event(pending_mod["id"])
                logger.info("Old event deleted for modification: %s", pending_mod["id"])
            except Exception:
                logger.exception(
                    "New WhatsApp appointment was created but the old appointment could not be removed"
                )
                modification_note = (
                    " The new appointment is booked, but the previous appointment "
                    "could not be removed. Please contact the clinic to remove the duplicate."
                )
            _delete_pending_modification(phone)
        mod_note = " (previous appointment cancelled)" if pending_mod else ""
        consultant_text = f" with {consultant_name}" if consultant_name else ""
        confirmation = (
            f"{visible_response}\n\n"
            f"Appointment confirmed{mod_note}: {service['name']} on {booking_date.strftime('%m/%d/%Y')} "
            f"at {booking_time.strftime('%H:%M')}{consultant_text} at {location['address']}.\n"
            f"Fee: SGD {service['price']:,}.\n\n"
            f"Cancellation policy: {CONFIG['booking'].get('cancellation_policy', 'Please contact the clinic for the cancellation policy.')}"
            f"{modification_note}"
        )
        HISTORY.add(phone, "assistant", confirmation)
        await WA.send_text(phone, confirmation)


async def _handle_payment_flow(
    phone,
    intent,
    visible_response,
    service,
    location,
    slot,
    cal,
    patient_age_group,
    pending_modification: dict | None = None,
):
    mp = _get_mp_client()
    if mp is None:
        return

    # Lock the slot for 30 minutes
    cal.lock_slot(slot.date, slot.start_time, ttl_seconds=1800)

    try:
        pref = mp.create_preference(
            title=service["name"],
            price=service["price"],
            external_reference=phone,
        )
    except Exception as e:
        logger.error("MP preference creation failed: %s", e)
        cal.release_slot(slot.date, slot.start_time)
        await WA.send_text(phone, "There was an error generating the payment link. Please try again.")
        return

    # Store pending payment keyed by phone (= external_reference used in MP preference)
    _save_pending_payment(phone, {
        "phone": phone,
        "service": service["name"],
        "date": slot.date.isoformat(),
        "time": slot.start_time.strftime("%H:%M"),
        "location": location["name"],
        "user_name": intent["parent_name"],
        "child_name": intent["child_name"],
        "parent_name": intent["parent_name"],
        "referral_name": intent["referral_name"],
        "patient_age_group": patient_age_group,
        "observations": intent.get("observations", ""),
        "price": service["price"],
        "duration_minutes": service["duration_minutes"],
        "location_address": location["address"],
        "consultant": intent.get("consultant", ""),
        "calendar_id": getattr(cal, "_calendar_id", ""),
        "pending_modification": pending_modification,
    })

    consultant_text = (
        f"\nAssigned consultant: {intent['consultant']}."
        if intent.get("consultant")
        else ""
    )
    msg = (
        f"{visible_response}\n\n"
        f"To confirm your appointment, complete the payment here:\n{pref['payment_url']}\n\n"
        f"Your appointment will be reserved once the payment is processed."
        f"{consultant_text}"
    )
    HISTORY.add(phone, "assistant", msg)
    await WA.send_text(phone, msg)


@app.post("/payments/webhook")
async def payment_webhook(request: Request):
    body = await request.body()
    x_signature = request.headers.get("X-Signature", "")

    # In sandbox mode, skip signature validation for easier testing
    sandbox = CONFIG.get("payments", {}).get("sandbox", True)
    if not sandbox and MP_WEBHOOK_SECRET:
        if not validate_mp_signature(body, x_signature, MP_WEBHOOK_SECRET):
            raise HTTPException(status_code=403, detail="Invalid MP signature")

    data = await request.json()

    if data.get("action") != "payment.updated":
        return Response(status_code=200)

    payment_id = str(data.get("data", {}).get("id", ""))
    if not payment_id:
        return Response(status_code=200)

    mp = _get_mp_client()
    if not mp:
        return Response(status_code=200)

    payment = mp.get_payment(payment_id)
    pending = _get_and_delete_pending_payment(payment)

    if not pending:
        return Response(status_code=200)

    phone = pending["phone"]

    if payment.get("status") == "approved":
        modification_note = ""
        cal = (
            _get_calendar_client(calendar_id=pending["calendar_id"])
            if pending.get("calendar_id")
            else _get_calendar_client()
        )
        if cal:
            booking_date = date.fromisoformat(pending["date"])
            booking_time = dt_time.fromisoformat(pending["time"])
            slot = Slot(date=booking_date, start_time=booking_time, location=pending["location"])
            cal.create_event(
                service_name=pending["service"],
                user_name=pending["user_name"],
                user_phone=phone,
                slot=slot,
                duration_minutes=pending["duration_minutes"],
                price=pending["price"],
                cancellation_policy=CONFIG["booking"].get("cancellation_policy", ""),
                child_name=pending["child_name"],
                parent_name=pending["parent_name"],
                referral_name=pending.get("referral_name", "None"),
                patient_age_group=pending.get("patient_age_group", ""),
                observations=pending.get("observations", ""),
            )
            cal.release_slot(booking_date, booking_time)
            pending_modification = pending.get("pending_modification")
            if pending_modification:
                try:
                    old_calendar = _calendar_client_for_event(pending_modification)
                    if old_calendar is None:
                        raise RuntimeError("The previous appointment calendar is unavailable.")
                    old_calendar.delete_event(pending_modification["id"])
                except Exception:
                    logger.exception(
                        "Paid replacement appointment was created but the old appointment could not be removed"
                    )
                    modification_note = (
                        " The new appointment is booked, but the previous appointment "
                        "could not be removed. Please contact the clinic to remove the duplicate."
                    )
                _delete_pending_modification(phone)
        consultant_text = (
            f" with {pending['consultant']}"
            if pending.get("consultant")
            else ""
        )
        confirmation = (
            f"Your payment has been confirmed. Appointment booked: {pending['service']} "
            f"on {pending['date']} at {pending['time']}"
            f"{consultant_text} "
            f"at {pending['location_address']}. "
            f"See you there!{modification_note}"
        )
        await WA.send_text(phone, confirmation)
    else:
        cal = (
            _get_calendar_client(calendar_id=pending["calendar_id"])
            if pending.get("calendar_id")
            else _get_calendar_client()
        )
        if cal:
            cal.release_slot(
                date.fromisoformat(pending["date"]),
                dt_time.fromisoformat(pending["time"])
            )
        await WA.send_text(phone, "The payment was not processed. If you'd like to try again, message us.")

    return Response(status_code=200)


async def _handle_appointment_lookup(phone: str):
    """Tell the patient their upcoming appointments, found by their WhatsApp number."""
    try:
        events = await asyncio.to_thread(_find_upcoming_events_by_phone, phone)
    except Exception:
        logger.exception("WhatsApp appointment lookup failed")
        response = "I couldn't look up your appointment just now. Please try again shortly."
        HISTORY.add(phone, "assistant", response)
        await WA.send_text(phone, response)
        return
    if not events:
        response = compose_handover_reply("I couldn't find an upcoming appointment under this number.")
        HISTORY.add(phone, "assistant", response)
        await WA.send_text(phone, response)
        await _pause_for_handover("whatsapp", phone, "admin")
        return
    lines = []
    for e in events:
        who = f" with {e['consultant']}" if e.get("consultant") else ""
        where = f" at {e['location']}" if e.get("location") else ""
        lines.append(f"{e['summary']}: {e['date']} at {e['time']}{who}{where}")
    if len(lines) == 1:
        response = f"Your next appointment is {lines[0]}."
    else:
        response = "Your upcoming appointments:\n" + "\n".join(f"{i}. {l}" for i, l in enumerate(lines, 1))
    HISTORY.add(phone, "assistant", response)
    await WA.send_text(phone, response)


async def _handle_cancellation_request(
    phone: str,
    visible_response: str,
    *,
    rebook_from_scratch: bool = False,
):
    """User wants to cancel — find their upcoming events."""
    if not _get_calendar_clients():
        await WA.send_text(phone, "The booking module is not available.")
        return

    events = _find_upcoming_events_by_phone(phone)

    if not events:
        msg = "I couldn't find any appointments booked with your number. Is there anything else I can help with?"
        HISTORY.add(phone, "assistant", msg)
        await WA.send_text(phone, msg)
        return

    if rebook_from_scratch:
        events = [{**event, "rebook_from_scratch": True} for event in events]
    _save_pending_cancellation(phone, events)

    if len(events) == 1:
        e = events[0]
        msg = (
            f"{visible_response}\n\n"
            f"I found this appointment:\n"
            f"*{e['summary']}*\n"
            f"Date: {e['date']}\n"
            f"Time: {e['time']}\n"
            f"Location: {e['location']}\n\n"
            f"Do you confirm you want to cancel it?"
        )
    else:
        lines = [f"{visible_response}\n\nI found these appointments:\n"]
        for i, e in enumerate(events, 1):
            lines.append(
                f"{i}. *{e['summary']}* — {e['date']} at {e['time']} at {e['location']}"
            )
        lines.append("\nWhich one would you like to cancel?")
        msg = "\n".join(lines)

    HISTORY.add(phone, "assistant", msg)
    await WA.send_text(phone, msg)


async def _handle_cancellation_confirmed(phone: str, intent: dict, visible_response: str):
    """User confirmed cancellation — delete the event."""
    if not _get_calendar_clients():
        await WA.send_text(phone, "The booking module is not available.")
        return

    events = _get_pending_cancellation(phone)
    if not events:
        msg = "I couldn't find an active cancellation request. Would you like to cancel an appointment? Let me know and I'll look up your bookings."
        HISTORY.add(phone, "assistant", msg)
        await WA.send_text(phone, msg)
        return

    event_index = intent.get("event_index", 1)
    try:
        event_index = int(event_index)
    except (TypeError, ValueError):
        event_index = 1

    if event_index < 1 or event_index > len(events):
        msg = f"Invalid number. Please choose a number from 1 to {len(events)}."
        HISTORY.add(phone, "assistant", msg)
        await WA.send_text(phone, msg)
        return

    event = events[event_index - 1]

    try:
        cal = _calendar_client_for_event(event)
        if cal is None:
            raise RuntimeError("The booking module is not available.")
        cal.delete_event(event["id"])
        _delete_pending_cancellation(phone)
        msg = (
            f"Appointment cancelled: *{event['summary']}* on {event['date']} at {event['time']}.\n\n"
            "To rebook, start a new booking and provide all booking details again; "
            "I will show the complete details for you to confirm."
        )
        logger.info("Event cancelled: %s for phone %s", event["id"], phone)
    except Exception as e:
        logger.error("Failed to cancel event: %s", e)
        msg = "There was a problem cancelling the appointment. Please contact us directly."

    HISTORY.add(phone, "assistant", msg)
    await WA.send_text(phone, msg)


def _modification_requires_mc(phone: str, event: dict) -> bool:
    if event.get("valid_mc_confirmed") is True:
        return False
    start_datetime = event.get("start_datetime")
    if not start_datetime:
        raise ValueError("The appointment start time is missing.")
    return is_within_rescheduling_window(
        start_datetime,
        CONFIG.get("client", {}).get("timezone", "UTC"),
    )


async def _start_modification_for_event(
    phone: str,
    event: dict,
    visible_response: str,
    *,
    valid_mc_confirmed: bool = False,
) -> None:
    if event["date"].startswith("Saturday"):
        msg = f"{visible_response}\n\n{_SATURDAY_RESCHEDULING_MESSAGE}"
        HISTORY.add(phone, "assistant", msg)
        await WA.send_text(phone, msg)
        return

    try:
        needs_mc = not valid_mc_confirmed and _modification_requires_mc(phone, event)
    except (TypeError, ValueError):
        logger.exception("Could not verify rescheduling window for phone %s", phone)
        msg = "I couldn't verify the rescheduling window. Please contact the clinic directly."
        HISTORY.add(phone, "assistant", msg)
        await WA.send_text(phone, msg)
        return

    if needs_mc:
        event["awaiting_mc_confirmation"] = True
        _save_pending_modification(phone, event)
        msg = f"{visible_response}\n\n{_RESCHEDULING_MC_QUESTION}"
    else:
        event.pop("awaiting_mc_confirmation", None)
        if valid_mc_confirmed:
            event["valid_mc_confirmed"] = True
        _save_pending_modification(phone, event)
        msg = (
            f"{visible_response}\n\n"
            f"I found your appointment:\n"
            f"*{event['summary']}*\n"
            f"Date: {event['date']}\n"
            f"Time: {event['time']}\n\n"
            "Rescheduling is allowed only when requested more than 36 hours before "
            "the appointment. Within or exactly 36 hours, a valid medical certificate "
            "(MC) is required.\n\n"
            "What new date and time would you prefer? I will show the updated booking "
            "details for you to confirm."
        )
    HISTORY.add(phone, "assistant", msg)
    await WA.send_text(phone, msg)


async def _handle_modification_request(
    phone: str,
    visible_response: str,
    *,
    valid_mc_confirmed: bool = False,
):
    """User wants to modify — show their events and ask for new date.
    The old event is NOT deleted yet. It gets deleted when the new booking is confirmed."""
    if not _get_calendar_clients():
        await WA.send_text(phone, "The booking module is not available.")
        return

    events = _find_upcoming_events_by_phone(phone)

    if not events:
        msg = "I couldn't find any appointments booked with your number. Would you like to book a new one?"
        HISTORY.add(phone, "assistant", msg)
        await WA.send_text(phone, msg)
        return

    if len(events) == 1:
        await _start_modification_for_event(
            phone,
            events[0],
            visible_response,
            valid_mc_confirmed=valid_mc_confirmed,
        )
        return
    else:
        if valid_mc_confirmed:
            for event in events:
                event["valid_mc_confirmed"] = True
        _save_pending_cancellation(phone, events)
        lines = [f"{visible_response}\n\nI found these appointments:\n"]
        for i, e in enumerate(events, 1):
            lines.append(
                f"{i}. *{e['summary']}* — {e['date']} at {e['time']}"
            )
        lines.append(
            "\nRescheduling is allowed only when requested more than 36 hours before "
            "the appointment. Within or exactly 36 hours, it is allowed only with a "
            "valid medical certificate (MC); without one, cancel and make a new "
            "booking from scratch with all details again, and the 100% cancellation "
            "charge applies. Saturday appointments "
            "are prepaid at SGD 300 for a single intervention and non-refundable. "
            "Saturday changes must be handled by the administrative team. Which "
            "appointment would you like to modify?"
        )
        msg = "\n".join(lines)

    HISTORY.add(phone, "assistant", msg)
    await WA.send_text(phone, msg)


async def _handle_modification_confirmed(phone: str, intent: dict, visible_response: str):
    """User chose which event to modify (multiple events case)."""
    if not _get_calendar_clients():
        await WA.send_text(phone, "The booking module is not available.")
        return

    events = _get_pending_cancellation(phone)
    if not events:
        msg = "I couldn't find an active modification request. Let me know which appointment you'd like to change."
        HISTORY.add(phone, "assistant", msg)
        await WA.send_text(phone, msg)
        return

    event_index = intent.get("event_index", 1)
    try:
        event_index = int(event_index)
    except (TypeError, ValueError):
        event_index = 1

    if event_index < 1 or event_index > len(events):
        msg = f"Invalid number. Please choose a number from 1 to {len(events)}."
        HISTORY.add(phone, "assistant", msg)
        await WA.send_text(phone, msg)
        return

    event = events[event_index - 1]
    _delete_pending_cancellation(phone)
    await _start_modification_for_event(
        phone,
        event,
        visible_response,
        valid_mc_confirmed=intent.get("valid_mc_confirmed") is True,
    )


async def _handle_rescheduling_mc_response(phone: str, intent: dict):
    event = _get_pending_modification(phone)
    if not event or not event.get("awaiting_mc_confirmation"):
        msg = "There is no active rescheduling request waiting for an MC response."
        HISTORY.add(phone, "assistant", msg)
        await WA.send_text(phone, msg)
        return

    has_valid_mc = intent.get("has_valid_mc")
    if not isinstance(has_valid_mc, bool):
        HISTORY.add(phone, "assistant", _RESCHEDULING_MC_QUESTION)
        await WA.send_text(phone, _RESCHEDULING_MC_QUESTION)
        return

    if has_valid_mc:
        event.pop("awaiting_mc_confirmation", None)
        event["valid_mc_confirmed"] = True
        _save_pending_modification(phone, event)
        msg = (
            "Thank you for confirming. What new date and time would you prefer? "
            "I will show the updated booking details for you to confirm."
        )
        HISTORY.add(phone, "assistant", msg)
        await WA.send_text(phone, msg)
        return

    _delete_pending_modification(phone)
    await _handle_cancellation_request(
        phone,
        "Without a valid MC, you need to cancel this appointment and make a new "
        "booking from scratch, providing all booking details again. The 100% "
        "cancellation charge applies. Would you like to proceed with cancellation?",
        rebook_from_scratch=True,
    )


async def _handle_rescheduling_handoff(phone: str):
    """Clear pending reschedule state and direct policy disputes to clinic staff."""
    _delete_pending_modification(phone)
    _delete_pending_cancellation(phone)
    msg = (
        "I understand. A member of our administrative team can assist you with this. "
        "Please contact the clinic directly."
    )
    HISTORY.add(phone, "assistant", msg)
    await WA.send_text(phone, msg)


@app.post("/internal/send-reminders")
async def trigger_reminders(request: Request):
    secret = request.headers.get("X-Internal-Secret", "")
    if secret != INTERNAL_SECRET:
        raise HTTPException(status_code=403)
    return await _send_scheduled_reminders()
