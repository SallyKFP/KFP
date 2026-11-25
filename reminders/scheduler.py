import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import datetime, time, timedelta

import pytz

from core.confirmation import REMINDER_CONFIRMATION_INSTRUCTION
from modules.booking.consultants import configured_consultants

logger = logging.getLogger(__name__)


def seconds_until_next_reminder(now: datetime, timezone: str) -> float:
    tz = pytz.timezone(timezone)
    local_now = now.astimezone(tz)
    next_run = tz.localize(datetime.combine(local_now.date(), time(9)))
    if next_run <= local_now:
        next_run = tz.localize(datetime.combine(local_now.date() + timedelta(days=1), time(9)))
    return (next_run - local_now).total_seconds()


async def run_daily_reminders(
    timezone: str,
    channel: str,
    send: Callable[[], Awaitable[object]],
) -> None:
    while True:
        await asyncio.sleep(seconds_until_next_reminder(datetime.now(pytz.UTC), timezone))
        try:
            result = await send()
            logger.info("%s scheduled reminders complete: %s", channel, result)
        except Exception:
            logger.exception("%s scheduled reminder run failed", channel)


def extract_phone_from_description(description: str) -> str | None:
    match = re.search(r"(?:WhatsApp|Phone):\s*(\+?\d+)", description or "")
    return match.group(1) if match else None


def extract_telegram_chat_id_from_description(description: str) -> int | None:
    match = re.search(r"Telegram Chat ID:\s*(-?\d+)", description or "")
    return int(match.group(1)) if match else None


def extract_patient_name(event: dict) -> str | None:
    description = event.get("description", "")
    match = re.search(r"^Patient:\s*(.+?)\s*$", description, re.MULTILINE)
    if match:
        return match.group(1).strip()

    summary = event.get("summary", "")
    _, separator, patient_details = summary.partition(" - Patient: ")
    if separator:
        patient_name, _, _ = patient_details.partition(" | Parent:")
        return patient_name.strip() or None

    _, separator, legacy_name = summary.partition(" - ")
    return legacy_name.strip() or None if separator else None


def format_reminder_message(
    template: str,
    event: dict,
    consultant_name: str = "",
) -> str:
    start_raw = event["start"].get("dateTime", "")
    try:
        dt = datetime.fromisoformat(start_raw)
        date_str = dt.strftime("%A, %d %B %Y")
        time_str = dt.strftime("%H:%M")
    except ValueError:
        date_str = start_raw
        time_str = start_raw

    summary = event.get("summary", "")
    service_name = summary.split(" - ")[0] if " - " in summary else summary
    location = event.get("location", "")
    patient_name = extract_patient_name(event)
    if not patient_name:
        raise ValueError(f"Unable to determine patient name for event: {summary}")

    message = template.format(
        date=date_str,
        time=time_str,
        location=location,
        service=service_name,
        patient=patient_name,
        consultant=consultant_name,
    )
    if REMINDER_CONFIRMATION_INSTRUCTION not in message:
        message = f"{message.rstrip()}\n\n{REMINDER_CONFIRMATION_INSTRUCTION}"
    return message


def _reminder_window(config: dict) -> tuple[datetime, datetime]:
    tz = pytz.timezone(config["client"].get("timezone", "America/Argentina/Buenos_Aires"))
    now = datetime.now(tz)
    reminder_date = (now + timedelta(days=2)).date()
    start = tz.localize(datetime.combine(reminder_date, datetime.min.time()))
    end = start.replace(hour=23, minute=59, second=59)
    return start, end


def _calendars(config: dict) -> list[tuple[str, str]]:
    booking = config["booking"]
    consultants = configured_consultants(booking)
    if consultants:
        return [
            (consultant["calendar_id"], consultant["name"])
            for consultant in consultants
        ]
    return [(booking["calendar_id"], "")]


def _mark_reminder_sent(
    calendar_service,
    calendar_id: str,
    event: dict,
    channel: str,
    appointment_date: str,
) -> None:
    private_properties = dict(
        event.get("extendedProperties", {}).get("private", {})
    )
    private_properties[f"{channel}_reminder_sent_for"] = appointment_date
    calendar_service.events().patch(
        calendarId=calendar_id,
        eventId=event["id"],
        body={"extendedProperties": {"private": private_properties}},
    ).execute()


async def send_reminders(config: dict, calendar_service, wa_client, should_send=None) -> int:
    """
    Query Calendar for events two local calendar days ahead and send WhatsApp reminders.
    Returns number of reminders sent.
    """
    reminder_start, reminder_end = _reminder_window(config)

    template = config["reminders"]["message_template"]
    sent = 0

    for calendar_id, consultant_name in _calendars(config):
        events_result = calendar_service.events().list(
            calendarId=calendar_id,
            timeMin=reminder_start.isoformat(),
            timeMax=reminder_end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        ).execute()

        for event in events_result.get("items", []):
            description = event.get("description", "")
            phone = extract_phone_from_description(description)
            if not phone:
                logger.warning("No phone in event: %s", event.get("summary"))
                continue
            if not extract_patient_name(event):
                logger.warning("No patient name in event: %s", event.get("summary"))
                continue
            start_raw = event.get("start", {}).get("dateTime", "")
            if not start_raw:
                logger.warning("No start time in event: %s", event.get("summary"))
                continue
            start_date = datetime.fromisoformat(start_raw).astimezone(
                pytz.timezone(config["client"].get("timezone", "UTC"))
            ).date().isoformat()
            private_properties = event.get("extendedProperties", {}).get("private", {})
            if private_properties.get("whatsapp_reminder_sent_for") == start_date:
                continue
            if should_send is not None and not await should_send("whatsapp", phone):
                logger.info("Reminder skipped: Sally is switched off for this contact")
                continue
            msg = format_reminder_message(template, event, consultant_name)
            await wa_client.send_text(phone, msg)
            _mark_reminder_sent(
                calendar_service,
                calendar_id,
                event,
                "whatsapp",
                start_date,
            )
            sent += 1
            logger.info("Reminder sent to %s for event %s", phone, event.get("summary"))

    return sent


async def send_telegram_reminders(config: dict, calendar_service, send_message, should_send=None) -> int:
    """Send reminders for events two local calendar days ahead; mark successful sends."""
    tz = pytz.timezone(config["client"].get("timezone", "America/Argentina/Buenos_Aires"))
    reminder_start, reminder_end = _reminder_window(config)

    template = config["reminders"]["message_template"]
    sent = 0
    for calendar_id, consultant_name in _calendars(config):
        events_result = calendar_service.events().list(
            calendarId=calendar_id,
            timeMin=reminder_start.isoformat(),
            timeMax=reminder_end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        ).execute()

        for event in events_result.get("items", []):
            chat_id = extract_telegram_chat_id_from_description(event.get("description", ""))
            start_raw = event.get("start", {}).get("dateTime", "")
            if chat_id is None or not start_raw:
                continue
            if not extract_patient_name(event):
                logger.warning("No patient name in event: %s", event.get("summary"))
                continue

            start_date = datetime.fromisoformat(start_raw).astimezone(tz).date().isoformat()
            private_properties = event.get("extendedProperties", {}).get("private", {})
            if private_properties.get("telegram_reminder_sent_for") == start_date:
                continue

            if should_send is not None and not await should_send("telegram", str(chat_id)):
                logger.info("Telegram reminder skipped: Sally is switched off for this chat")
                continue
            await send_message(
                chat_id,
                format_reminder_message(template, event, consultant_name),
            )
            _mark_reminder_sent(
                calendar_service,
                calendar_id,
                event,
                "telegram",
                start_date,
            )
            sent += 1
            logger.info("Telegram reminder sent to chat %s for event %s", chat_id, event.get("summary"))

    return sent
