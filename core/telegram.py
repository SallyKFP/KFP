import asyncio
import argparse
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import date, datetime

import anthropic
import httpx
import pytz
from dotenv import load_dotenv

from config.loader import load_config
from core.ai import extract_booking_intent, get_ai_response, load_knowledge
from core.confirmation import (
    BOOKING_CONFIRMATION_INSTRUCTION,
    BOOKING_NOT_CONFIRMED,
    REMINDER_NOT_CONFIRMED,
    awaiting_booking_information,
    is_booking_confirmation_prompt,
    is_exact_reply,
    is_short_acknowledgement,
)
from core.dashboard_store import DashboardStore
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
from core.rescheduling import is_within_rescheduling_window
from modules.booking.calendar import (
    CalendarClient,
    Slot,
    patient_age_group,
    validate_booking_slot,
)
from modules.booking.availability import check_availability_range
from modules.booking.consultants import (
    allocate_consultant,
    configured_consultants,
    consultant_slot_is_reserved,
    resolve_consultant,
)
from reminders.scheduler import run_daily_reminders, send_telegram_reminders

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"
BOOKING_CONFIRMATION_WORDS = re.compile(
    r"\b(?:appointment|booking)\s+(?:is\s+)?confirmed\b"
    r"|\b(?:i(?:'ve| have)?\s+)?booked\s+(?:you|the|your|an|a)\b"
    r"|\byou(?:'re| are)\s+booked\b"
    r"|\bi(?:'ve| have)\s+confirmed\s+your\s+appointment\b",
    re.IGNORECASE,
)
BOOKING_USAGE = (
    "Use: /book Service | YYYY-MM-DD | HH:MM | Patient name | Patient DOB (YYYY-MM-DD) "
    "| Parent name | Referring doctor or None | Patient resides in Singapore? yes/no "
    "| Location (omit if one is configured) | Consultant (optional)."
)
APPOINTMENT_LOOKUP_USAGE = "Use: /next-appointment Patient name | Parent name."
_pending_cancellations: dict[int, list[dict]] = {}
_pending_modifications: dict[int, dict] = {}
_RESCHEDULING_MC_QUESTION = (
    "This appointment is within 36 hours. Rescheduling is only possible if you have "
    "a valid medical certificate (MC). Do you have a valid MC? Please reply YES or NO. "
    "Without one, you will need to cancel and make a new booking from scratch, "
    "providing all booking details again; the 100% cancellation charge applies."
)
_pending_consultant_assignments: dict[tuple[int, str, str, str, str], str] = {}
_dashboard_store: DashboardStore | None = None


class TelegramConflictError(RuntimeError):
    """Raised when Telegram rejects polling because the bot has another update source."""


def _consultant_assignment_key(
    chat_id: int,
    intent: dict,
) -> tuple[int, str, str, str, str] | None:
    values = (
        str(intent.get("service", "")).strip(),
        str(intent.get("date", "")).strip(),
        str(intent.get("time", "")).strip(),
        str(intent.get("location", "")).strip(),
    )
    if not all(values):
        return None
    return (chat_id, *values)


def parse_allowed_user_ids(value: str) -> set[int]:
    try:
        user_ids = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError("TELEGRAM_ALLOWED_USER_IDS must be comma-separated numeric IDs") from exc
    if not user_ids:
        raise ValueError("Set TELEGRAM_ALLOWED_USER_IDS to your numeric Telegram user ID")
    return user_ids


async def _telegram_request(
    client: httpx.AsyncClient,
    token: str,
    method: str,
    *,
    params: dict | None = None,
    payload: dict | None = None,
):
    response = await client.post(
        f"{TELEGRAM_API_BASE}/bot{token}/{method}",
        params=params,
        json=payload,
    )
    if response.status_code == 409:
        data = response.json()
        description = data.get("description", "another update source is active")
        raise TelegramConflictError(f"Telegram {method} conflict: {description}")
    response.raise_for_status()
    data = response.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram {method} failed: {data.get('description', 'unknown error')}")
    return data["result"]


async def _reply_to_message(
    client: httpx.AsyncClient,
    token: str,
    chat_id: int,
    text: str,
) -> None:
    await _telegram_request(
        client,
        token,
        "sendMessage",
        payload={"chat_id": chat_id, "text": text},
    )


async def _sally_should_reply(chat_id: int, contact_label: str | None = None) -> bool:
    """Register the chat and report whether the dashboard switches allow Sally to reply."""
    if _dashboard_store is None:
        return True
    try:
        return await asyncio.to_thread(
            _dashboard_store.register_contact, "telegram", str(chat_id), contact_label
        )
    except Exception:
        logger.exception("Could not read dashboard switches; Sally will reply")
        return True


@asynccontextmanager
async def _telegram_dashboard_storage():
    global _dashboard_store
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        yield
        return

    store = DashboardStore(database_url)
    await asyncio.to_thread(store.initialize)
    _dashboard_store = store
    try:
        yield
    finally:
        _dashboard_store = None

def _configured_consultants(config: dict) -> list[dict]:
    return configured_consultants(config.get("booking", {}))


def _resolve_consultant(config: dict, consultant_name: str | None) -> dict | None:
    return resolve_consultant(config.get("booking", {}), consultant_name)


def _calendar_client(config: dict, consultant_name: str | None = None) -> CalendarClient:
    if not config.get("modules", {}).get("booking"):
        raise ValueError("Calendar booking is disabled. Set modules.booking to true.")
    booking = config.get("booking", {})
    consultants = _configured_consultants(config)
    required = ("business_hours", "services", "locations")
    if not consultants:
        required += ("calendar_id", "calendar_owner_email")
    missing = [key for key in required if not booking.get(key)]
    if missing:
        raise ValueError(
            "Calendar booking is not configured. Add these booking settings: "
            + ", ".join(missing)
        )
    consultant = _resolve_consultant(config, consultant_name)
    return CalendarClient(
        calendar_id=consultant["calendar_id"] if consultant else booking["calendar_id"],
        calendar_owner_email=(
            consultant.get("calendar_owner_email", booking.get("calendar_owner_email", ""))
            if consultant
            else booking["calendar_owner_email"]
        ),
        business_hours=booking["business_hours"],
        timezone=config.get("client", {}).get("timezone", "UTC"),
    )


def _calendar_client_for_event(config: dict, event: dict) -> CalendarClient:
    calendar_id = event.get("calendar_id")
    if calendar_id:
        for consultant in _configured_consultants(config):
            if consultant["calendar_id"] == calendar_id:
                return _calendar_client(config, consultant["name"])
        booking = config.get("booking", {})
        if booking.get("calendar_id") == calendar_id:
            return _calendar_client(config)
        raise ValueError("The calendar for this appointment is not configured.")
    return _calendar_client(config)


def _find_telegram_events(config: dict, chat_id: int) -> list[dict]:
    consultants = _configured_consultants(config)
    if consultants:
        calendars = [
            (consultant["name"], _calendar_client(config, consultant["name"]))
            for consultant in consultants
        ]
    else:
        calendars = [("", _calendar_client(config))]
    events = []
    for consultant_name, calendar in calendars:
        for event in calendar.find_upcoming_events_by_telegram_chat_id(chat_id):
            if consultant_name:
                event["consultant"] = consultant_name
            events.append(event)
    return sorted(events, key=lambda event: event.get("start_datetime", ""))


def _find_next_telegram_appointment(
    config: dict,
    chat_id: int,
    patient_name: str,
    parent_name: str,
) -> dict | None:
    if not config.get("modules", {}).get("booking"):
        raise ValueError("Calendar booking is disabled.")

    consultants = _configured_consultants(config)
    calendars = (
        [
            (consultant["name"], _calendar_client(config, consultant["name"]))
            for consultant in consultants
        ]
        if consultants
        else [("", _calendar_client(config))]
    )
    appointments = []
    for consultant_name, calendar in calendars:
        event = calendar.find_next_event_by_telegram_chat_and_names(
            chat_id,
            patient_name,
            parent_name,
        )
        if event:
            appointments.append({**event, "consultant": consultant_name})
    return min(
        appointments,
        key=lambda appointment: datetime.fromisoformat(appointment["start_datetime"]),
        default=None,
    )


def _pending_telegram_confirmation_events(config: dict, chat_id: int) -> list[dict]:
    return [
        event
        for event in _find_telegram_events(config, chat_id)
        if event.get("reminder_sent_for")
        and event.get("reminder_sent_for") == event.get("start_datetime", "")[:10]
        and event.get("attendance_confirmed_for") != event.get("reminder_sent_for")
    ]


async def _handle_telegram_attendance_confirmation(
    client: httpx.AsyncClient,
    token: str,
    chat_id: int,
    config: dict,
    user_text: str,
) -> str | None:
    if not is_exact_reply(user_text, "YES") and not config.get("modules", {}).get("booking"):
        return None
    try:
        pending_events = await asyncio.to_thread(
            _pending_telegram_confirmation_events,
            config,
            chat_id,
        )
    except Exception:
        logger.exception("Could not check pending Telegram reminders")
        response = (
            "I couldn't check your appointment reminders just now. "
            "Attendance hasn't been confirmed; please try again later."
        )
        await _reply_to_message(client, token, chat_id, response)
        return response if is_exact_reply(user_text, "YES") else None
    if not is_exact_reply(user_text, "YES"):
        if pending_events:
            await _reply_to_message(client, token, chat_id, REMINDER_NOT_CONFIRMED)
            if is_short_acknowledgement(user_text):
                return REMINDER_NOT_CONFIRMED
        return None
    if not pending_events:
        response = (
            "I couldn't find an appointment reminder awaiting confirmation for this chat. "
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
        calendar = await asyncio.to_thread(
            _calendar_client_for_event,
            config,
            event,
        )
        await asyncio.to_thread(
            calendar.mark_event_attendance_confirmed,
            event["id"],
        )
        response = (
            f"Thank you. Attendance is confirmed for {event.get('patient_name') or 'the patient'} "
            f"on {event.get('date')} at {event.get('time')}."
        )
    await _reply_to_message(client, token, chat_id, response)
    return response


def _patient_has_upcoming_appointment(config: dict, patient_name: str) -> bool:
    consultants = _configured_consultants(config)
    if consultants:
        calendars = [
            _calendar_client(config, consultant["name"])
            for consultant in consultants
        ]
    else:
        calendars = [_calendar_client(config)]
    return any(
        calendar.has_upcoming_appointment_for_patient(patient_name)
        for calendar in calendars
    )


def _match_configured_item(items: list[dict], requested_name: str, kind: str) -> dict:
    name = requested_name.strip().casefold()
    exact_matches = [item for item in items if item.get("name", "").casefold() == name]
    matches = exact_matches or [
        item for item in items
        if name in item.get("name", "").casefold() or item.get("name", "").casefold() in name
    ]
    if not matches:
        options = ", ".join(item.get("name", "") for item in items)
        raise ValueError(f"Unknown {kind} '{requested_name}'. Choose from: {options}.")
    if len(matches) > 1:
        raise ValueError(f"More than one {kind} matches '{requested_name}'. Use its exact name.")
    return matches[0]


def _create_telegram_booking(
    config: dict,
    chat_id: int,
    fields: list[str],
    consultant_name: str | None = None,
    pending_modification: dict | None = None,
    observations: str = "",
) -> str:
    booking = config.get("booking", {})
    locations = booking.get("locations", [])
    if len(fields) == 8 and len(locations) == 1:
        (
            service_name,
            requested_date,
            requested_time,
            child_name,
            child_dob,
            parent_name,
            _referral_name,
            patient_resides_in_singapore,
        ) = fields
        location_name = locations[0]["name"]
    elif len(fields) == 9:
        (
            service_name,
            requested_date,
            requested_time,
            child_name,
            child_dob,
            parent_name,
            _referral_name,
            patient_resides_in_singapore,
            location_name,
        ) = fields
    else:
        raise ValueError(BOOKING_USAGE)

    if not child_name.strip() or not parent_name.strip():
        raise ValueError("Please provide both the patient's full name and the parent's full name.")
    if not _referral_name.strip():
        raise ValueError("Provide the referring person's name, or enter None if there is no referral.")
    residency_answer = patient_resides_in_singapore.strip().casefold()
    if residency_answer not in {"yes", "no"}:
        raise ValueError("Confirm whether the patient resides in Singapore by entering yes or no.")
    if residency_answer == "no":
        logger.warning(
            "Routing Telegram booking to admin: booking intent reports patient residency outside Singapore"
        )
        raise ValueError("Our administrative team will assist you with this booking.")
    dob = validate_patient_dob(child_dob.strip(), config.get("client", {}).get("timezone", "UTC"))
    service = _match_configured_item(booking.get("services", []), service_name, "service")
    location = _match_configured_item(locations, location_name, "location")
    try:
        booking_date = date.fromisoformat(requested_date.strip())
        start_time = datetime.strptime(requested_time.strip(), "%H:%M").time()
    except (ValueError, KeyError) as exc:
        raise ValueError("Use a valid YYYY-MM-DD date and HH:MM time.") from exc

    if booking_date.weekday() == 5:
        raise ValueError("Our administrative team will assist you with this booking.")

    age_group = patient_age_group(dob, booking_date)
    timezone = pytz.timezone(config.get("client", {}).get("timezone", "UTC"))
    try:
        validate_booking_slot(
            booking_date,
            start_time,
            service["duration_minutes"],
            booking["business_hours"],
            location.get("days", []),
            str(timezone),
        )
    except ValueError as exc:
        logger.info("Telegram booking slot rejected: %s", exc)
        raise ValueError("Could you share another preferred date and time?") from exc

    consultants = _configured_consultants(config)
    consultant = (
        _resolve_consultant(config, consultant_name)
        if consultant_name or len(consultants) <= 1
        else None
    )
    if consultant and consultant_slot_is_reserved(
        consultant["name"],
        booking_date,
        start_time,
    ):
        raise ValueError(
            f"That time isn't available with {consultant['name']}. "
            "Please choose another date and time."
        )
    slot_available = False
    if consultants and consultant is None:
        available_consultants = []
        for candidate in consultants:
            if consultant_slot_is_reserved(candidate["name"], booking_date, start_time):
                continue
            candidate_calendar = _calendar_client(config, candidate["name"])
            if candidate_calendar.is_slot_available(
                booking_date,
                start_time,
                service["duration_minutes"],
            ):
                available_consultants.append(candidate)
        consultant = allocate_consultant(available_consultants)
        if consultant is None:
            raise ValueError(
                "That time isn't available with either consultant. "
                "Please choose another date and time."
            )
        consultant_name = consultant["name"]
        calendar = _calendar_client(config, consultant_name)
        slot_available = True
    else:
        calendar = (
            _calendar_client(config, consultant_name)
            if consultant_name
            else _calendar_client(config)
        )
    already_booked = _patient_has_upcoming_appointment(config, child_name)
    is_rescheduling_same_patient = bool(
        pending_modification
        and child_name.strip().casefold()
        == pending_modification.get("patient_name", "").strip().casefold()
    )
    if already_booked and not is_rescheduling_same_patient:
        logger.warning(
            "Routing Telegram booking to admin: matching upcoming patient appointment found"
        )
        raise ValueError("Our administrative team will assist you with this booking.")
    slot = Slot(date=booking_date, start_time=start_time, location=location["name"])
    if not slot_available and not calendar.is_slot_available(
        booking_date,
        start_time,
        service["duration_minutes"],
    ):
        raise ValueError("That time is already booked. Choose another date or time.")

    calendar.create_event(
        service_name=service["name"],
        user_name=parent_name.strip(),
        user_phone="",
        slot=slot,
        duration_minutes=service["duration_minutes"],
        price=service.get("price", 0),
        cancellation_policy=booking.get("cancellation_policy", ""),
        telegram_chat_id=chat_id,
        child_name=child_name.strip(),
        parent_name=parent_name.strip(),
        referral_name=_referral_name.strip(),
        patient_age_group=age_group,
        observations=observations,
    )
    modification_note = ""
    if pending_modification:
        if pending_modification.get("awaiting_mc_confirmation"):
            _pending_modifications[chat_id] = pending_modification
            raise ValueError(_RESCHEDULING_MC_QUESTION)
        if (
            pending_modification.get("valid_mc_confirmed") is not True
            and _telegram_modification_requires_mc(config, pending_modification)
        ):
            pending_modification["awaiting_mc_confirmation"] = True
            _pending_modifications[chat_id] = pending_modification
            raise ValueError(_RESCHEDULING_MC_QUESTION)
        try:
            _calendar_client_for_event(config, pending_modification).delete_event(
                pending_modification["id"]
            )
        except Exception:
            logger.exception(
                "New Telegram appointment was created but the old appointment could not be removed"
            )
            modification_note = (
                " The new appointment is booked, but the previous appointment could "
                "not be removed. Please contact the clinic to remove the duplicate."
            )
    consultant_text = f" with {consultant['name']}" if consultant else ""
    return (
        f"Appointment confirmed: {service['name']} for {child_name.strip()} "
        f"(parent: {parent_name.strip()}) on {booking_date.isoformat()} "
        f"at {start_time.strftime('%H:%M')}{consultant_text} at {location['address']}. "
        f"Fee: SGD {service.get('price', 0):,}.{modification_note}"
    )


def _check_telegram_availability(
    config: dict,
    intent: dict,
    chat_id: int | None = None,
) -> str:
    booking = config.get("booking", {})
    locations = booking.get("locations", [])
    service_name = str(intent.get("service", "")).strip()
    location_name = str(intent.get("location", "")).strip()
    if not service_name or not intent.get("date") or not intent.get("time"):
        raise ValueError("Please tell me the service, date, and time you'd like me to check.")
    if not location_name and len(locations) == 1:
        location_name = locations[0]["name"]
    if not location_name:
        raise ValueError("Please tell me which clinic location you'd like me to check.")

    service = _match_configured_item(booking.get("services", []), service_name, "service")
    location = _match_configured_item(locations, location_name, "location")
    try:
        booking_date = date.fromisoformat(str(intent["date"]).strip())
        start_time = datetime.strptime(str(intent["time"]).strip(), "%H:%M").time()
    except (ValueError, KeyError) as exc:
        raise ValueError("I couldn't identify that date and time. Please clarify them.") from exc

    timezone = pytz.timezone(config.get("client", {}).get("timezone", "UTC"))
    try:
        validate_booking_slot(
            booking_date,
            start_time,
            service["duration_minutes"],
            booking["business_hours"],
            location.get("days", []),
            str(timezone),
        )
    except ValueError as exc:
        logger.info("Telegram availability slot rejected: %s", exc)
        raise ValueError("Could you share another preferred date and time?") from exc

    consultant_name = str(intent.get("consultant", "")).strip() or None
    consultants = _configured_consultants(config)
    consultant = (
        _resolve_consultant(config, consultant_name)
        if consultant_name or len(consultants) <= 1
        else None
    )
    if consultant and consultant_slot_is_reserved(
        consultant["name"],
        booking_date,
        start_time,
    ):
        return (
            f"That time isn't available with {consultant['name']}. "
            "What other date and time would work?"
        )
    if consultants and consultant is None:
        available_consultants = []
        for candidate in consultants:
            if consultant_slot_is_reserved(candidate["name"], booking_date, start_time):
                continue
            candidate_calendar = _calendar_client(config, candidate["name"])
            if candidate_calendar.is_slot_available(
                booking_date,
                start_time,
                service["duration_minutes"],
            ):
                available_consultants.append(candidate)
        consultant = allocate_consultant(available_consultants)
        if consultant is None:
            slot_text = (
                f"{service['name']} on {booking_date.strftime('%A, %B')} "
                f"{booking_date.day}, {booking_date.year} "
                f"at {start_time.strftime('%H:%M')}"
            )
            return (
                f"{slot_text} isn't available with either consultant. "
                "What other date and time would work?"
            )
        consultant_name = consultant["name"]
        calendar = _calendar_client(config, consultant_name)
        available = True
        if chat_id is not None:
            assignment_key = _consultant_assignment_key(chat_id, {
                **intent,
                "service": service["name"],
                "date": booking_date.isoformat(),
                "time": start_time.strftime("%H:%M"),
                "location": location["name"],
            })
            if assignment_key is not None:
                _pending_consultant_assignments[assignment_key] = consultant_name
    else:
        calendar = (
            _calendar_client(config, consultant_name)
            if consultant_name
            else _calendar_client(config)
        )
        available = calendar.is_slot_available(
            booking_date,
            start_time,
            service["duration_minutes"],
        )
    consultant_text = f" with {consultant['name']}" if consultant else ""
    slot_text = (
        f"{service['name']} on {booking_date.strftime('%A, %B')} "
        f"{booking_date.day}, {booking_date.year} "
        f"at {start_time.strftime('%H:%M')}"
    )
    if available:
        return (
            f"{slot_text} is available at {location['name']}{consultant_text}. "
            "It isn't reserved yet. Would you like me to book it?"
        )
    return (
        f"{slot_text} is already booked at {location['name']}{consultant_text}. "
        "What other date and time would work? "
        "I'll check that option too."
    )


async def _handle_book_command(
    client: httpx.AsyncClient,
    token: str,
    chat_id: int,
    config: dict,
    fields: list[str],
    consultant_name: str | None = None,
    pending_modification: dict | None = None,
    observations: str = "",
) -> str:
    try:
        if observations:
            confirmation = await asyncio.to_thread(
                _create_telegram_booking, config, chat_id, fields, consultant_name,
                pending_modification, observations,
            )
        elif pending_modification:
            confirmation = await asyncio.to_thread(
                _create_telegram_booking,
                config,
                chat_id,
                fields,
                consultant_name,
                pending_modification,
            )
        elif consultant_name:
            confirmation = await asyncio.to_thread(
                _create_telegram_booking, config, chat_id, fields, consultant_name,
            )
        else:
            confirmation = await asyncio.to_thread(
                _create_telegram_booking, config, chat_id, fields,
            )
    except ValueError as exc:
        response_text = str(exc)
    except Exception:
        logger.exception("Telegram calendar booking failed")
        response_text = "Our administrative team will assist you with this booking."
    else:
        response_text = confirmation
    await _reply_to_message(client, token, chat_id, response_text)
    return response_text


async def _handle_telegram_cancellation_request(
    client: httpx.AsyncClient,
    token: str,
    chat_id: int,
    config: dict,
    visible_response: str,
    *,
    rebook_from_scratch: bool = False,
) -> str:
    events = await asyncio.to_thread(_find_telegram_events, config, chat_id)
    if not events:
        response = (
            "I couldn't find any upcoming appointments linked to this Telegram chat. "
            "Is there anything else I can help with?"
        )
    else:
        if rebook_from_scratch:
            events = [{**event, "rebook_from_scratch": True} for event in events]
        _pending_cancellations[chat_id] = events
        if len(events) == 1:
            event = events[0]
            consultant_text = (
                f"\nConsultant: {event['consultant']}" if event.get("consultant") else ""
            )
            response = (
                f"{visible_response}\n\nI found this appointment:\n"
                f"*{event['summary']}*\nDate: {event['date']}\n"
                f"Time: {event['time']}{consultant_text}\n\n"
                "Do you confirm you want to cancel it?"
            )
        else:
            lines = [f"{visible_response}\n\nI found these appointments:"]
            for index, event in enumerate(events, 1):
                consultant_text = (
                    f" with {event['consultant']}" if event.get("consultant") else ""
                )
                lines.append(
                    f"{index}. *{event['summary']}* â€” {event['date']} at "
                    f"{event['time']}{consultant_text}"
                )
            lines.append("\nWhich one would you like to cancel?")
            response = "\n".join(lines)
    await _reply_to_message(client, token, chat_id, response)
    return response


async def _handle_telegram_cancellation_confirmed(
    client: httpx.AsyncClient,
    token: str,
    chat_id: int,
    config: dict,
    intent: dict,
) -> str:
    events = _pending_cancellations.get(chat_id, [])
    if not events:
        response = (
            "I couldn't find an active cancellation request. Tell me you'd like to "
            "cancel an appointment and I'll look it up."
        )
    else:
        try:
            event_index = int(intent.get("event_index", 1))
        except (TypeError, ValueError):
            event_index = 1
        if event_index < 1 or event_index > len(events):
            response = f"Please choose an appointment number from 1 to {len(events)}."
        else:
            event = events[event_index - 1]
            try:
                calendar = _calendar_client_for_event(config, event)
                await asyncio.to_thread(calendar.delete_event, event["id"])
            except Exception:
                logger.exception("Failed to cancel Telegram appointment %s", event["id"])
                response = (
                    "I couldn't cancel the appointment just now. It has not been "
                    "confirmed as cancelled; please try again or contact the clinic."
                )
            else:
                _pending_cancellations.pop(chat_id, None)
                response = (
                    f"Appointment cancelled: *{event['summary']}* on {event['date']} "
                    f"at {event['time']}.\n\nTo rebook, start a new booking and provide "
                    "all booking details again; I will show the complete details for "
                    "you to confirm."
                )
    await _reply_to_message(client, token, chat_id, response)
    return response


async def _handle_telegram_modification_request(
    client: httpx.AsyncClient,
    token: str,
    chat_id: int,
    config: dict,
    visible_response: str,
    *,
    valid_mc_confirmed: bool = False,
) -> str:
    events = await asyncio.to_thread(_find_telegram_events, config, chat_id)
    if not events:
        response = "I couldn't find any upcoming appointments linked to this Telegram chat."
    elif len(events) == 1:
        event = events[0]
        if event["date"].startswith("Saturday"):
            response = (
                "Saturday appointment changes must be handled by the administrative "
                "team. Please contact the clinic directly."
            )
        else:
            response = await _start_telegram_modification(
                client,
                token,
                chat_id,
                config,
                event,
                visible_response,
                valid_mc_confirmed=valid_mc_confirmed,
            )
            return response
    else:
        if valid_mc_confirmed:
            for event in events:
                event["valid_mc_confirmed"] = True
        _pending_cancellations[chat_id] = events
        lines = [f"{visible_response}\n\nWhich appointment would you like to change?"]
        for index, event in enumerate(events, 1):
            consultant_text = (
                f" with {event['consultant']}" if event.get("consultant") else ""
            )
            lines.append(
                f"{index}. *{event['summary']}* â€” {event['date']} at "
                f"{event['time']}{consultant_text}"
            )
        response = "\n".join(lines)
    await _reply_to_message(client, token, chat_id, response)
    return response


async def _handle_telegram_modification_confirmed(
    client: httpx.AsyncClient,
    token: str,
    chat_id: int,
    config: dict,
    intent: dict,
) -> str:
    events = _pending_cancellations.get(chat_id, [])
    if not events:
        response = "I couldn't find an active appointment-change request. Please ask me to change an appointment first."
    else:
        try:
            event_index = int(intent.get("event_index", 1))
        except (TypeError, ValueError):
            event_index = 1
        if event_index < 1 or event_index > len(events):
            response = f"Please choose an appointment number from 1 to {len(events)}."
        else:
            event = events[event_index - 1]
            if event["date"].startswith("Saturday"):
                response = (
                    "Saturday appointment changes must be handled by the administrative "
                    "team. Please contact the clinic directly."
                )
            else:
                response = await _start_telegram_modification(
                    client,
                    token,
                    chat_id,
                    config,
                    event,
                    "",
                    valid_mc_confirmed=intent.get("valid_mc_confirmed") is True,
                )
                _pending_cancellations.pop(chat_id, None)
                return response
            _pending_cancellations.pop(chat_id, None)
    await _reply_to_message(client, token, chat_id, response)
    return response


def _telegram_modification_requires_mc(config: dict, event: dict) -> bool:
    if event.get("valid_mc_confirmed") is True:
        return False
    start_datetime = event.get("start_datetime")
    if not start_datetime:
        raise ValueError("The appointment start time is missing.")
    return is_within_rescheduling_window(
        start_datetime,
        config.get("client", {}).get("timezone", "UTC"),
    )


async def _start_telegram_modification(
    client: httpx.AsyncClient,
    token: str,
    chat_id: int,
    config: dict,
    event: dict,
    visible_response: str,
    *,
    valid_mc_confirmed: bool = False,
) -> str:
    try:
        needs_mc = (
            not valid_mc_confirmed
            and _telegram_modification_requires_mc(config, event)
        )
    except (TypeError, ValueError):
        logger.exception("Could not verify Telegram rescheduling window for chat %s", chat_id)
        response = "I couldn't verify the rescheduling window. Please contact the clinic directly."
    else:
        if needs_mc:
            event["awaiting_mc_confirmation"] = True
            _pending_modifications[chat_id] = event
            response = f"{visible_response}\n\n{_RESCHEDULING_MC_QUESTION}".strip()
        else:
            event.pop("awaiting_mc_confirmation", None)
            if valid_mc_confirmed:
                event["valid_mc_confirmed"] = True
            _pending_modifications[chat_id] = event
            consultant_text = (
                f" with {event['consultant']}" if event.get("consultant") else ""
            )
            response = (
                f"{visible_response}\n\nI found *{event['summary']}* on "
                f"{event['date']} at {event['time']}{consultant_text}. "
                "Rescheduling is allowed only when requested more than 36 hours "
                "before the appointment; within or exactly 36 hours, a valid MC is "
                "required. "
                "What new date and time would you prefer? I will show the updated "
                "booking details for you to confirm."
            ).strip()
    await _reply_to_message(client, token, chat_id, response)
    return response


async def _handle_telegram_rescheduling_mc_response(
    client: httpx.AsyncClient,
    token: str,
    chat_id: int,
    config: dict,
    intent: dict,
) -> str:
    event = _pending_modifications.get(chat_id)
    if not event or not event.get("awaiting_mc_confirmation"):
        response = "There is no active rescheduling request waiting for an MC response."
    elif not isinstance(intent.get("has_valid_mc"), bool):
        response = _RESCHEDULING_MC_QUESTION
    elif intent["has_valid_mc"]:
        event.pop("awaiting_mc_confirmation", None)
        event["valid_mc_confirmed"] = True
        _pending_modifications[chat_id] = event
        response = (
            "Thank you for confirming. What new date and time would you prefer? "
            "I will show the updated booking details for you to confirm."
        )
    else:
        _pending_modifications.pop(chat_id, None)
        response = await _handle_telegram_cancellation_request(
            client,
            token,
            chat_id,
            config,
            "Without a valid MC, you need to cancel this appointment and make a new "
            "booking from scratch, providing all booking details again. The 100% "
            "cancellation charge applies. Would you like to proceed with cancellation?",
            rebook_from_scratch=True,
        )
        return response
    await _reply_to_message(client, token, chat_id, response)
    return response


async def _handle_update(
    update: dict,
    *,
    client: httpx.AsyncClient,
    token: str,
    allowed_user_ids: set[int],
    history,
    config: dict,
    knowledge: str,
) -> None:
    message = update.get("message", {})
    sender = message.get("from", {})
    user_id = sender.get("id")
    chat = message.get("chat", {})
    chat_id = chat.get("id")

    if chat.get("type") != "private":
        logger.warning("Ignored Telegram message from a non-private chat")
        return
    if user_id not in allowed_user_ids:
        logger.warning(
            "Ignored Telegram message from a user not in the configured allowlist "
            "(%d IDs loaded)",
            len(allowed_user_ids),
        )
        return
    if chat_id is None:
        logger.warning("Ignored Telegram message without a chat ID")
        return

    profile_name = " ".join(
        part.strip()
        for part in (sender.get("first_name", ""), sender.get("last_name", ""))
        if part.strip()
    )
    username = str(sender.get("username", "")).strip() or None
    contact_label = profile_name or (
        f"@{username}" if username else f"Telegram user {chat_id}"
    )
    if username:
        contact_label = f"{contact_label} (@{username})" if profile_name else contact_label
    if not await _sally_should_reply(chat_id, contact_label):
        logger.info("Telegram message left for a consultant (Sally is switched off)")
        return

    user_text = message.get("text", "").strip()
    if not user_text:
        await _reply_to_message(
            client,
            token,
            chat_id,
            "For this development bot, please send a text message.",
        )
        return

    if user_text == "/start":
        history.add(str(user_id), "assistant", "Development bot is ready.")
        await _reply_to_message(
            client,
            token,
            chat_id,
            "Development bot is ready. Send a fictional test message, or use /book to create a test appointment.",
        )
        return

    command, _, arguments = user_text.partition(" ")
    command = command.split("@", 1)[0].casefold()
    key = f"telegram:{chat_id}"
    if command == "/book":
        previous = PENDING_BOOKINGS.get(key)
        modification = previous.modification if previous else _pending_modifications.get(chat_id)
        PENDING_BOOKINGS.clear(key)
        fields = [field.strip() for field in arguments.split("|")]
        if not arguments.strip():
            fields = []
        consultants = config.get("booking", {}).get("consultants", [])
        consultant_names = {
            str(consultant.get("name", "")).strip().casefold()
            for consultant in consultants
        }
        location_names = {
            str(location.get("name", "")).strip().casefold()
            for location in config.get("booking", {}).get("locations", [])
        }
        consultant_name = None
        if len(consultants) > 1 and fields:
            candidate = fields[-1].casefold()
            is_consultant_field = (
                candidate in consultant_names
                or len(fields) >= 10
                or (len(location_names) == 1 and len(fields) == 9 and candidate not in location_names)
            )
            if is_consultant_field:
                consultant_name = fields.pop().strip()
        if len(fields) not in {8, 9} or not all(fields):
            history.add(str(user_id), "assistant", BOOKING_USAGE)
            await _reply_to_message(client, token, chat_id, BOOKING_USAGE)
            return
        names = [
            "service", "date", "time", "child_name", "child_dob",
            "parent_name", "referral_name", "patient_resides_in_singapore", "location",
        ]
        details = dict(zip(names, fields))
        preserve_observations(details, previous)
        if modification:
            try:
                needs_mc = (
                    modification.get("awaiting_mc_confirmation")
                    or _telegram_modification_requires_mc(config, modification)
                )
            except (TypeError, ValueError):
                logger.exception(
                    "Could not verify Telegram rescheduling window for chat %s",
                    chat_id,
                )
                response = "I couldn't verify the rescheduling window. Please contact the clinic directly."
                history.add(str(user_id), "assistant", response)
                await _reply_to_message(client, token, chat_id, response)
                return
            if needs_mc:
                modification["awaiting_mc_confirmation"] = True
                _pending_modifications[chat_id] = modification
                history.add(str(user_id), "assistant", _RESCHEDULING_MC_QUESTION)
                await _reply_to_message(
                    client,
                    token,
                    chat_id,
                    _RESCHEDULING_MC_QUESTION,
                )
                return
        locations = config.get("booking", {}).get("locations", [])
        if "location" not in details and len(locations) == 1:
            details["location"] = locations[0]["name"]
        if consultant_name:
            details["consultant"] = consultant_name
        try:
            if modification:
                inherit_modification_consultant(details, modification, _configured_consultants(config))
            pending = PENDING_BOOKINGS.put(
                key, details, modification, proposed_observations=existing_proposal(details, previous),
                timezone=config.get("client", {}).get("timezone", "UTC"),
            )
        except ValueError as exc:
            response = str(exc)
        else:
            response = pending.summary
            if modification:
                _pending_modifications.pop(chat_id, None)
        history.add(str(user_id), "assistant", response)
        try:
            await _reply_to_message(client, token, chat_id, response)
        except Exception:
            PENDING_BOOKINGS.clear(key)
            raise
        return

    if command == "/next-appointment":
        history.add(str(user_id), "assistant", "Checking the next appointment.")
        fields = [field.strip() for field in arguments.split("|")]
        if len(fields) != 2 or not all(fields):
            await _reply_to_message(
                client,
                token,
                chat_id,
                APPOINTMENT_LOOKUP_USAGE,
            )
            return
        try:
            appointment = await asyncio.to_thread(
                _find_next_telegram_appointment,
                config,
                chat_id,
                fields[0],
                fields[1],
            )
        except Exception:
            logger.exception("Telegram appointment lookup failed")
            response = "I couldn't look up the appointment just now. Please try again later."
        else:
            if appointment is None:
                response = (
                    "I couldn't find an upcoming appointment matching that patient and "
                    "parent for this Telegram chat. Please check the names and try again."
                )
            else:
                consultant = (
                    f" with {appointment['consultant']}"
                    if appointment.get("consultant")
                    else ""
                )
                location = (
                    f" at {appointment['location']}"
                    if appointment.get("location")
                    else ""
                )
                response = (
                    f"The next appointment is {appointment['service']} on "
                    f"{appointment['date']} at {appointment['time']}{consultant}"
                    f"{location}."
                )
        await _reply_to_message(client, token, chat_id, response)
        return

    conversation = history.get(str(user_id))
    history.add(str(user_id), "user", user_text)
    pending = PENDING_BOOKINGS.get(key)
    booking_prompt = pending is not None and pending.awaiting_confirmation
    booking_question = awaiting_booking_information(conversation)
    if is_exact_reply(user_text, "APPROVE NOTE"):
        try:
            response = PENDING_BOOKINGS.approve_observations(key).summary
        except ValueError as exc:
            response = str(exc)
        history.add(str(user_id), "assistant", response)
        await _reply_to_message(client, token, chat_id, response)
        return
    if user_text.lstrip().casefold().startswith("note:") or is_exact_reply(user_text, "REMOVE NOTE"):
        try:
            note = None if is_exact_reply(user_text, "REMOVE NOTE") else user_text.lstrip()[5:]
            pending = PENDING_BOOKINGS.set_observations(key, note)
            response = pending.summary
        except ValueError as exc:
            response = str(exc)
        history.add(str(user_id), "assistant", response)
        await _reply_to_message(client, token, chat_id, response)
        return
    if is_exact_reply(user_text, "CONFIRM"):
        if pending and pending.awaiting_confirmation and pending.proposed_observations is not None:
            response = "Please approve or remove the proposed note before confirming the booking.\n\n" + pending.summary
            history.add(str(user_id), "assistant", response)
            await _reply_to_message(client, token, chat_id, response)
            return
        pending = PENDING_BOOKINGS.take(key)
        if pending is None:
            response = "There is no active booking summary to confirm. Please ask me to prepare a new booking summary."
            history.add(str(user_id), "assistant", response)
            await _reply_to_message(client, token, chat_id, response)
            return
        details = pending.details
        fields = [
            details["service"], details["date"], details["time"], details["child_name"],
            details["child_dob"], details["parent_name"], details["referral_name"],
            details["patient_resides_in_singapore"], details["location"],
        ]
        booking_args = (
            client, token, chat_id, config, fields, details.get("consultant"),
            pending.modification,
        )
        if details.get("observations"):
            response = await _handle_book_command(*booking_args, observations=details["observations"])
        else:
            response = await _handle_book_command(*booking_args)
        if pending.modification and response.startswith("Appointment confirmed:"):
            _pending_modifications.pop(chat_id, None)
        history.add(str(user_id), "assistant", response)
        return
    if pending and has_booking_cancellation(user_text):
        PENDING_BOOKINGS.clear(key)
        _pending_modifications.pop(chat_id, None)
        response = (
            "The pending booking has been cancelled; no appointment was booked. "
            "To cancel an existing appointment, please ask separately."
        )
        history.add(str(user_id), "assistant", response)
        await _reply_to_message(client, token, chat_id, response)
        return
    if booking_prompt and not is_exact_reply(user_text, "CONFIRM"):
        if is_short_acknowledgement(user_text):
            response = (
                "The appointment is not booked yet.\n\n" + pending.summary
                if pending.proposed_observations is not None
                else f"{BOOKING_NOT_CONFIRMED}\n\n{pending.summary}"
            )
            history.add(str(user_id), "assistant", response)
            await _reply_to_message(client, token, chat_id, response)
            return
        if preserves_pending_booking(user_text):
            await _reply_to_message(client, token, chat_id, BOOKING_NOT_CONFIRMED)
        else:
            PENDING_BOOKINGS.invalidate(key)
            await _reply_to_message(
                client, token, chat_id,
                "Your previous booking summary is no longer awaiting confirmation. "
                "If the details change, I'll show you a new summary to confirm.",
            )
    pending_cancellations = _pending_cancellations.get(chat_id, [])
    if (
        not (booking_prompt or booking_question)
        and len(pending_cancellations) == 1
        and pending_cancellations[0].get("rebook_from_scratch")
        and is_exact_reply(user_text, "YES")
    ):
        response = await _handle_telegram_cancellation_confirmed(
            client,
            token,
            chat_id,
            config,
            {"event_index": 1},
        )
        history.add(str(user_id), "assistant", response)
        return
    if (
        not (booking_prompt or booking_question)
        and len(pending_cancellations) == 1
        and pending_cancellations[0].get("rebook_from_scratch")
        and is_exact_reply(user_text, "NO")
    ):
        _pending_cancellations.pop(chat_id, None)
        response = (
            "Understood. Your appointment is still booked. Without a valid MC, it "
            "cannot be rescheduled within 36 hours. Please contact the clinic if you "
            "need help."
        )
        history.add(str(user_id), "assistant", response)
        await _reply_to_message(client, token, chat_id, response)
        return

    pending_modification = _pending_modifications.get(chat_id)
    if (
        not (booking_prompt or booking_question)
        and pending_modification
        and pending_modification.get("awaiting_mc_confirmation")
        and (
            is_exact_reply(user_text, "YES")
            or is_exact_reply(user_text, "NO")
        )
    ):
        response = await _handle_telegram_rescheduling_mc_response(
            client,
            token,
            chat_id,
            config,
            {"has_valid_mc": is_exact_reply(user_text, "YES")},
        )
        history.add(str(user_id), "assistant", response)
        return

    attendance_response = None
    if not (booking_prompt or booking_question):
        attendance_response = await _handle_telegram_attendance_confirmation(
            client,
            token,
            chat_id,
            config,
            user_text,
        )
    if attendance_response is not None:
        history.add(str(user_id), "assistant", attendance_response)
        return
    try:
        raw_answer = await asyncio.to_thread(
            get_ai_response,
            user_text,
            conversation,
            config,
            knowledge,
            telegram_live_availability=True,
        )
    except anthropic.APIError:
        logger.exception("Claude request failed while handling a Telegram message")
        await _reply_to_message(
            client,
            token,
            chat_id,
            "Sorry, I couldn't generate a reply just now. Please try again.",
        )
        return

    intent, answer = extract_booking_intent(raw_answer)
    if intent and intent.get("intent") == "booking_draft":
        details = dict(intent)
        proposal = existing_proposal(details, pending) if pending else extract_general_observations(conversation, user_text)
        preserve_observations(details, pending)
        modification = pending.modification if pending else _pending_modifications.get(chat_id)
        if modification:
            try:
                needs_mc = (
                    modification.get("awaiting_mc_confirmation")
                    or _telegram_modification_requires_mc(config, modification)
                )
            except (TypeError, ValueError):
                logger.exception(
                    "Could not verify Telegram rescheduling window for chat %s",
                    chat_id,
                )
                response = "I couldn't verify the rescheduling window. Please contact the clinic directly."
                history.add(str(user_id), "assistant", response)
                await _reply_to_message(client, token, chat_id, response)
                return
            if needs_mc:
                modification["awaiting_mc_confirmation"] = True
                _pending_modifications[chat_id] = modification
                history.add(str(user_id), "assistant", _RESCHEDULING_MC_QUESTION)
                await _reply_to_message(
                    client,
                    token,
                    chat_id,
                    _RESCHEDULING_MC_QUESTION,
                )
                return
        PENDING_BOOKINGS.clear(key)
        locations = config.get("booking", {}).get("locations", [])
        if not details.get("location") and len(locations) == 1:
            details["location"] = locations[0]["name"]
        assignment_key = _consultant_assignment_key(chat_id, details)
        if not modification and not details.get("consultant") and assignment_key is not None:
            details["consultant"] = _pending_consultant_assignments.get(assignment_key)
        try:
            if modification:
                inherit_modification_consultant(details, modification, _configured_consultants(config))
            pending = PENDING_BOOKINGS.put(
                key, details, modification, proposed_observations=proposal,
                timezone=config.get("client", {}).get("timezone", "UTC"),
            )
        except ValueError as exc:
            response = str(exc)
        else:
            response = pending.summary
            if modification:
                _pending_modifications.pop(chat_id, None)
        history.add(str(user_id), "assistant", response)
        try:
            await _reply_to_message(client, token, chat_id, response)
        except Exception:
            PENDING_BOOKINGS.clear(key)
            raise
        if assignment_key is not None and PENDING_BOOKINGS.get(key) is not None:
            _pending_consultant_assignments.pop(assignment_key, None)
        return
    if intent and intent.get("intent") == "availability_range_check":
        try:
            response = await asyncio.to_thread(
                check_availability_range,
                config,
                intent,
                lambda consultant_name: _calendar_client(config, consultant_name),
            )
        except ValueError as exc:
            response = str(exc)
        except Exception:
            logger.exception("Telegram availability range check failed")
            response = "I couldn't check availability just now. Please try again shortly."
        history.add(str(user_id), "assistant", response)
        await _reply_to_message(client, token, chat_id, response)
        return

    if intent and intent.get("intent") == "appointment_lookup":
        patient_name = str(intent.get("patient_name", "")).strip()
        parent_name = str(intent.get("parent_name", "")).strip()
        if not patient_name or not parent_name:
            response = "Please provide the patient's full name and the parent or guardian's full name."
        else:
            try:
                appointment = await asyncio.to_thread(
                    _find_next_telegram_appointment,
                    config,
                    chat_id,
                    patient_name,
                    parent_name,
                )
            except Exception:
                logger.exception("Telegram appointment lookup failed")
                response = "I couldn't look up the appointment just now. Please try again later."
            else:
                if appointment is None:
                    response = (
                        "I couldn't find an upcoming appointment matching that patient and "
                        "parent for this Telegram chat. Please check the names and try again."
                    )
                else:
                    consultant = (
                        f" with {appointment['consultant']}"
                        if appointment.get("consultant")
                        else ""
                    )
                    location = (
                        f" at {appointment['location']}"
                        if appointment.get("location")
                        else ""
                    )
                    response = (
                        f"The next appointment is {appointment['service']} on "
                        f"{appointment['date']} at {appointment['time']}{consultant}"
                        f"{location}."
                    )
        history.add(str(user_id), "assistant", response)
        await _reply_to_message(client, token, chat_id, response)
        return

    if intent and intent.get("intent") == "availability_check":
        try:
            availability_response = await asyncio.to_thread(
                _check_telegram_availability,
                config,
                intent,
                chat_id,
            )
        except ValueError as exc:
            availability_response = str(exc)
        except Exception:
            logger.exception("Telegram calendar availability check failed")
            availability_response = (
                "I couldn't check the calendar just now. The time is not confirmed as available; "
                "please try again shortly."
            )
        history.add(str(user_id), "assistant", availability_response)
        await _reply_to_message(client, token, chat_id, availability_response)
        return

    if intent and intent.get("intent") == "cancellation_request":
        response = await _handle_telegram_cancellation_request(
            client, token, chat_id, config, answer,
        )
        history.add(str(user_id), "assistant", response)
        return

    if intent and intent.get("intent") == "cancellation_confirmed":
        response = await _handle_telegram_cancellation_confirmed(
            client, token, chat_id, config, intent,
        )
        history.add(str(user_id), "assistant", response)
        return

    if intent and intent.get("intent") == "modification_request":
        response = await _handle_telegram_modification_request(
            client, token, chat_id, config, answer,
            valid_mc_confirmed=intent.get("valid_mc_confirmed") is True,
        )
        history.add(str(user_id), "assistant", response)
        return

    if intent and intent.get("intent") == "modification_confirmed":
        response = await _handle_telegram_modification_confirmed(
            client, token, chat_id, config, intent,
        )
        history.add(str(user_id), "assistant", response)
        return

    if intent and intent.get("intent") == "rescheduling_mc_response":
        response = await _handle_telegram_rescheduling_mc_response(
            client,
            token,
            chat_id,
            config,
            intent,
        )
        history.add(str(user_id), "assistant", response)
        return

    if intent and intent.get("intent") == "booking_confirmed":
        response = "I haven't booked an appointment. Please ask me to prepare a booking summary."
        history.add(str(user_id), "assistant", response)
        await _reply_to_message(client, token, chat_id, response)
        return

    if BOOKING_CONFIRMATION_WORDS.search(answer):
        answer = (
            "I can't confirm this appointment yet. Please share the requested booking "
            "details so I can help with your request."
        )
    if (
        is_booking_confirmation_prompt(answer)
        or re.search(r"\b(?:reply|send|type)\s+[*'\"`]*confirm\b", answer, re.IGNORECASE)
    ):
        active = PENDING_BOOKINGS.get(key)
        answer = (
            f"{BOOKING_NOT_CONFIRMED}\n\n{active.summary}"
            if active and active.awaiting_confirmation
            else "I couldn't prepare a booking summary yet. Please ask me to review the booking details again."
        )
    history.add(str(user_id), "assistant", answer)
    await _reply_to_message(client, token, chat_id, answer)


@asynccontextmanager
async def _telegram_reminder_schedule(token: str, config: dict):
    reminder_task = None
    if config.get("modules", {}).get("reminders"):
        timezone = config.get("client", {}).get("timezone", "Asia/Singapore")

        async def send():
            return await run_telegram_reminders(token, config)

        reminder_task = asyncio.create_task(run_daily_reminders(timezone, "Telegram", send))
        logger.info("Telegram reminders scheduled for 09:00 %s", timezone)
    try:
        yield
    finally:
        if reminder_task:
            reminder_task.cancel()
            try:
                await reminder_task
            except asyncio.CancelledError:
                pass


async def run_bot(token: str, allowed_user_ids: set[int]) -> None:
    config = load_config()
    knowledge = load_knowledge()
    history = get_history()
    offset = None

    async with (
        httpx.AsyncClient(timeout=httpx.Timeout(40.0)) as client,
        _telegram_reminder_schedule(token, config),
        _telegram_dashboard_storage(),
    ):
        bot = await _telegram_request(client, token, "getMe")
        logger.info("Telegram development bot started as @%s", bot.get("username", "unknown"))
        logger.info("Telegram allowlist loaded with %d user IDs", len(allowed_user_ids))

        while True:
            try:
                updates = await _telegram_request(
                    client,
                    token,
                    "getUpdates",
                    params={
                        "timeout": 25,
                        "allowed_updates": json.dumps(["message"]),
                        **({"offset": offset} if offset is not None else {}),
                    },
                )
            except (httpx.HTTPError, RuntimeError) as exc:
                if isinstance(exc, TelegramConflictError):
                    logger.error(
                        "%s Stop any other bot polling processes or webhook, then restart once.",
                        str(exc).replace(token, "[redacted]"),
                    )
                    return
                logger.error(
                    "Telegram polling failed: %s",
                    str(exc).replace(token, "[redacted]"),
                )
                await asyncio.sleep(5)
                continue

            for update in updates:
                offset = update["update_id"] + 1
                try:
                    await _handle_update(
                        update,
                        client=client,
                        token=token,
                        allowed_user_ids=allowed_user_ids,
                        history=history,
                        config=config,
                        knowledge=knowledge,
                    )
                except (httpx.HTTPError, RuntimeError):
                    logger.exception("Failed to handle Telegram update %s", update["update_id"])


async def send_telegram_reminders_with_token(
    token: str,
    config: dict,
    calendar_service,
) -> int:
    if not config.get("modules", {}).get("reminders"):
        raise RuntimeError("Set modules.reminders to true before sending Telegram reminders.")
    async with httpx.AsyncClient(timeout=httpx.Timeout(40.0)) as client:
        async def send_message(chat_id: int, text: str) -> None:
            await _telegram_request(
                client,
                token,
                "sendMessage",
                payload={"chat_id": chat_id, "text": text},
            )

        return await send_telegram_reminders(config, calendar_service, send_message)


async def run_telegram_reminders(token: str, config: dict) -> int:
    if not config.get("modules", {}).get("reminders"):
        raise RuntimeError("Set modules.reminders to true before sending Telegram reminders.")
    consultants = _configured_consultants(config)
    if consultants:
        calendar = await asyncio.to_thread(
            _calendar_client,
            config,
            consultants[0]["name"],
        )
    else:
        calendar = await asyncio.to_thread(_calendar_client, config)
    return await send_telegram_reminders_with_token(token, config, calendar._service)


async def main() -> None:
    load_dotenv()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN in your .env file")
    parser = argparse.ArgumentParser(description="Run the Telegram development bot.")
    parser.add_argument(
        "--send-reminders",
        action="store_true",
        help="Send Telegram appointment reminders for two calendar days ahead once, then exit.",
    )
    args = parser.parse_args()
    if args.send_reminders:
        config = load_config()
        sent = await run_telegram_reminders(token, config)
        logger.info("Sent %d Telegram appointment reminder(s)", sent)
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY in your .env file to enable Claude replies")
    allowed_user_ids = parse_allowed_user_ids(os.environ.get("TELEGRAM_ALLOWED_USER_IDS", ""))
    await run_bot(token, allowed_user_ids)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
