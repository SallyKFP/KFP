import os
import random
from datetime import date, time


def configured_consultants(booking: dict) -> list[dict]:
    configured = booking.get("consultants", [])
    if not isinstance(configured, list):
        raise ValueError("booking.consultants must be a list of consultant names and calendar IDs.")

    consultants = []
    for configured_consultant in configured:
        consultant = dict(configured_consultant)
        name = str(consultant.get("name", "")).strip()
        if not name:
            raise ValueError("Each configured consultant needs a name.")

        calendar_id_env = consultant.get("calendar_id_env")
        if calendar_id_env:
            consultant["calendar_id"] = os.environ.get(calendar_id_env, "").strip()
            if not consultant["calendar_id"]:
                raise ValueError(f"Set {calendar_id_env} in your .env file.")
        if not consultant.get("calendar_id"):
            raise ValueError(f"Set a calendar ID for {name} in booking.consultants.")
        consultant["name"] = name
        consultants.append(consultant)

    return consultants


def resolve_consultant(booking: dict, consultant_name: str | None) -> dict | None:
    consultants = configured_consultants(booking)
    if not consultants:
        return None
    if not consultant_name:
        if len(consultants) == 1:
            return consultants[0]
        names = ", ".join(consultant["name"] for consultant in consultants)
        raise ValueError(f"Please tell me which consultant you prefer: {names}.")

    requested = consultant_name.strip().casefold()
    exact_matches = [
        consultant for consultant in consultants
        if consultant["name"].casefold() == requested
    ]
    matches = exact_matches or [
        consultant for consultant in consultants
        if requested in consultant["name"].casefold()
        or consultant["name"].casefold() in requested
    ]
    if not matches:
        names = ", ".join(consultant["name"] for consultant in consultants)
        raise ValueError(f"Unknown consultant '{consultant_name}'. Choose from: {names}.")
    if len(matches) > 1:
        raise ValueError(
            f"More than one consultant matches '{consultant_name}'. Use the exact name."
        )
    return matches[0]


def allocate_consultant(available_consultants: list[dict]) -> dict | None:
    """Choose an available consultant using the configured internal allocation split."""
    if not available_consultants:
        return None
    if len(available_consultants) == 1:
        return available_consultants[0]

    weights = [
        70 if "serena" in consultant["name"].casefold()
        else 30 if "sarah" in consultant["name"].casefold()
        else 1
        for consultant in available_consultants
    ]
    return random.choices(available_consultants, weights=weights, k=1)[0]


def consultant_slot_is_reserved(
    consultant_name: str | None,
    appointment_date: date,
    start_time: time,
) -> bool:
    """Return whether a Singapore-local slot is internally reserved."""
    if not consultant_name:
        return False

    name = consultant_name.casefold()
    if "sarah" in name:
        return (
            start_time in {time(16), time(17)}
            or (
                appointment_date.weekday() == 3
                and start_time <= time(12)
            )
            or appointment_date.weekday() == 4
        )
    if "serena" in name:
        return appointment_date.weekday() == 2
    return False
