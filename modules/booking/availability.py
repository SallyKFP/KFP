import calendar
from datetime import date, datetime
from typing import Callable

import pytz

from modules.booking.consultants import (
    configured_consultants,
    consultant_slot_is_reserved,
    resolve_consultant,
)
from modules.booking.calendar import CalendarClient

MAX_AVAILABILITY_RANGE_MONTHS = 3
MAX_AVAILABILITY_RESULTS = 10


def _add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _configured_item(items: list[dict], requested_name: str, kind: str) -> dict:
    requested = requested_name.strip().casefold()
    exact_matches = [
        item for item in items
        if str(item.get("name", "")).strip().casefold() == requested
    ]
    matches = exact_matches or [
        item for item in items
        if requested in str(item.get("name", "")).casefold()
        or str(item.get("name", "")).casefold() in requested
    ]
    if not matches:
        options = ", ".join(str(item.get("name", "")) for item in items)
        raise ValueError(f"Unknown {kind} '{requested_name}'. Choose from: {options}.")
    if len(matches) > 1:
        raise ValueError(f"More than one {kind} matches '{requested_name}'. Use its exact name.")
    return matches[0]


def check_availability_range(
    config: dict,
    intent: dict,
    calendar_factory: Callable[[str | None], CalendarClient],
) -> str:
    """Search configured calendars for real appointment openings in a date range."""
    booking = config.get("booking", {})
    if not config.get("modules", {}).get("booking"):
        raise ValueError("Calendar booking is disabled.")

    try:
        start_date = date.fromisoformat(str(intent.get("start_date", "")).strip())
        end_date = date.fromisoformat(str(intent.get("end_date", "")).strip())
    except ValueError as exc:
        raise ValueError("Please tell me the start and end dates you'd like me to check.") from exc
    if end_date < start_date:
        raise ValueError("The end date must be on or after the start date.")
    if end_date > _add_months(start_date, MAX_AVAILABILITY_RANGE_MONTHS):
        raise ValueError(
            f"I can check up to {MAX_AVAILABILITY_RANGE_MONTHS} calendar months at a time. "
            "Please choose a shorter date range."
        )

    service_name = str(intent.get("service", "")).strip()
    if not service_name:
        raise ValueError("Please tell me which service you'd like me to check.")
    service = _configured_item(booking.get("services", []), service_name, "service")

    locations = booking.get("locations", [])
    location_name = str(intent.get("location", "")).strip()
    if not location_name and len(locations) == 1:
        location_name = str(locations[0].get("name", ""))
    if not location_name:
        raise ValueError("Please tell me which clinic location you'd like me to check.")
    location = _configured_item(locations, location_name, "location")

    timezone = pytz.timezone(config.get("client", {}).get("timezone", "UTC"))
    consultants = configured_consultants(booking)
    requested_consultant = str(intent.get("consultant", "")).strip() or None
    if requested_consultant:
        consultant = resolve_consultant(booking, requested_consultant)
        if consultant is None:
            raise ValueError("No consultants are configured for appointment bookings.")
        calendars = [(consultant["name"], calendar_factory(consultant["name"]))]
    elif consultants:
        calendars = [
            (consultant["name"], calendar_factory(consultant["name"]))
            for consultant in consultants
        ]
    else:
        calendars = [("", calendar_factory(None))]

    openings = []
    for consultant_name, calendar in calendars:
        slots = calendar.find_available_slots(
            start_date=start_date,
            end_date=end_date,
            duration_minutes=service["duration_minutes"],
            business_hours=booking["business_hours"],
            booking_days=location.get("days", []),
            location=location.get("name", ""),
        )
        for slot in slots:
            if consultant_name and consultant_slot_is_reserved(
                consultant_name,
                slot.date,
                slot.start_time,
            ):
                continue
            start = timezone.localize(datetime.combine(slot.date, slot.start_time))
            openings.append((start, consultant_name, slot))

    openings.sort(key=lambda result: result[0])
    start_text = start_date.strftime("%A, %d %B %Y")
    end_text = end_date.strftime("%A, %d %B %Y")
    if not openings:
        return (
            f"I couldn't find an available {service['name']} appointment from "
            f"{start_text} to {end_text}. Would another date range work?"
        )

    lines = []
    for start, consultant_name, _slot in openings[:MAX_AVAILABILITY_RESULTS]:
        consultant_text = f" with {consultant_name}" if consultant_name else ""
        lines.append(
            f"- {start.strftime('%A, %d %B')} at {start.strftime('%H:%M')}"
            f"{consultant_text}"
        )
    response = (
        f"These {service['name']} times are currently available from {start_text} "
        f"to {end_text}:\n" + "\n".join(lines)
    )
    if len(openings) > MAX_AVAILABILITY_RESULTS:
        response += "\nThere are more available times in this range if you'd like additional options."
    response += "\nThese times are not booked or reserved yet."
    return response
