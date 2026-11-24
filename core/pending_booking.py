import re
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import date, time as dt_time
from typing import Callable

from core.confirmation import BOOKING_CONFIRMATION_INSTRUCTION, is_short_acknowledgement
from core.patient_dates import validate_patient_dob

BOOKING_FIELDS = (
    ("service", "Service"),
    ("date", "Date"),
    ("time", "Time"),
    ("child_name", "Patient"),
    ("child_dob", "Patient DOB"),
    ("parent_name", "Parent"),
    ("referral_name", "Referral"),
    ("patient_resides_in_singapore", "Patient resides in Singapore"),
    ("location", "Location"),
)

OBSERVATIONS_INSTRUCTION = (
    "Optional: To share general observations or concerns about your child with the "
    "consultant, reply NOTE: followed by your message. Please do not include specific "
    "or sensitive health details. Your note will be included in the consultant's "
    "calendar appointment."
)

_GENERAL_ACTIVITY = re.compile(
    r"\b(?:she|he|they|my child|my daughter|my son)\s+"
    r"(?:likes|loves|enjoys|prefers)\s+"
    r"(?:(?:playing with|playing|reading|drawing|listening to)\s+)?"
    r"(?:blocks|toys|books|music|puzzles|dolls|cars|drawing|painting|singing|dancing)"
    r"\b",
    re.IGNORECASE,
)


def extract_general_observations(history: list[dict], user_text: str) -> str | None:
    """Conservatively extract a non-clinical activity stated by the parent."""
    messages = [*history, {"role": "user", "content": user_text}]
    for message in reversed(messages):
        if message.get("role") == "assistant" and "appointment confirmed" in str(message.get("content", "")).casefold():
            break
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                match = _GENERAL_ACTIVITY.search(content)
                if match:
                    return match.group(0)
    return None


@dataclass(frozen=True)
class PendingBooking:
    details: dict[str, str]
    summary: str
    expires_at: float
    modification: dict | None = None
    awaiting_confirmation: bool = True
    proposed_observations: str | None = None


class PendingBookings:
    def __init__(self, ttl: int = 24 * 60 * 60, clock: Callable[[], float] = time.monotonic):
        self._ttl = ttl
        self._clock = clock
        self._items: dict[str, PendingBooking] = {}
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def _expire(self, key: str, expires_at: float) -> None:
        with self._lock:
            record = self._items.get(key)
            if record and record.expires_at == expires_at:
                self._items.pop(key, None)
                self._timers.pop(key, None)

    def put(
        self, key: str, details: dict, modification: dict | None = None,
        proposed_observations: str | None = None,
        *, timezone: str = "UTC",
    ) -> PendingBooking:
        normalized = validate_details(details, timezone=timezone)
        if modification:
            patient = modification.get("patient_name", "")
            if (
                not modification.get("id")
                or not isinstance(patient, str)
                or not patient.strip()
                or patient.strip().casefold() != normalized["child_name"].casefold()
            ):
                raise ValueError(
                    "I couldn't verify which patient's appointment is being changed. "
                    "Please contact the clinic for assistance."
                )
        summary = format_summary(normalized, proposed_observations)
        if modification:
            summary += (
                "\n\nThis will replace the existing appointment for "
                f"{normalized['child_name']} on {modification.get('date', 'the previously selected date')} "
                f"at {modification.get('time', 'the previously selected time')}."
            )
        record = PendingBooking(
            normalized, summary, self._clock() + self._ttl, deepcopy(modification),
            proposed_observations=proposed_observations,
        )
        with self._lock:
            timer = self._timers.pop(key, None)
            if timer:
                timer.cancel()
            self._items[key] = record
            if self._clock is time.monotonic:
                timer = threading.Timer(self._ttl, self._expire, args=(key, record.expires_at))
                timer.daemon = True
                self._timers[key] = timer
                timer.start()
        return record

    def get(self, key: str) -> PendingBooking | None:
        with self._lock:
            record = self._items.get(key)
            if record and record.expires_at <= self._clock():
                del self._items[key]
                timer = self._timers.pop(key, None)
                if timer:
                    timer.cancel()
                return None
            return record

    def take(self, key: str) -> PendingBooking | None:
        with self._lock:
            record = self._items.get(key)
            if record and (not record.awaiting_confirmation or record.proposed_observations is not None):
                return None
            self._items.pop(key, None)
            timer = self._timers.pop(key, None)
            if timer:
                timer.cancel()
            return record if record and record.expires_at > self._clock() else None

    def invalidate(self, key: str) -> None:
        with self._lock:
            record = self._items.get(key)
            if record:
                self._items[key] = replace(record, awaiting_confirmation=False)

    def set_observations(self, key: str, text: str | None) -> PendingBooking:
        note = " ".join(text.split()) if text is not None else None
        if note is not None and (not note or len(note) > 1500):
            raise ValueError("Please provide a note of 1 to 1500 characters after NOTE:.")
        with self._lock:
            record = self._items.get(key)
            if not record or record.expires_at <= self._clock() or not record.awaiting_confirmation:
                raise ValueError("There is no active booking summary. Please prepare a new booking first.")
            details = dict(record.details)
            details.pop("observations", None)
            summary = format_summary(details, note)
            if record.modification:
                summary += (
                    "\n\nThis will replace the existing appointment for "
                    f"{details['child_name']} on {record.modification.get('date', 'the previously selected date')} "
                    f"at {record.modification.get('time', 'the previously selected time')}."
                )
            updated = replace(record, details=details, summary=summary, proposed_observations=note)
            self._items[key] = updated
            return updated

    def approve_observations(self, key: str) -> PendingBooking:
        with self._lock:
            record = self._items.get(key)
            if (
                not record or record.expires_at <= self._clock()
                or not record.awaiting_confirmation or record.proposed_observations is None
            ):
                raise ValueError("There is no proposed note to approve.")
            details = {**record.details, "observations": record.proposed_observations}
            summary = format_summary(details)
            if record.modification:
                summary += (
                    "\n\nThis will replace the existing appointment for "
                    f"{details['child_name']} on {record.modification.get('date', 'the previously selected date')} "
                    f"at {record.modification.get('time', 'the previously selected time')}."
                )
            updated = replace(record, details=details, summary=summary, proposed_observations=None)
            self._items[key] = updated
            return updated

    def clear(self, key: str) -> None:
        with self._lock:
            self._items.pop(key, None)
            timer = self._timers.pop(key, None)
            if timer:
                timer.cancel()


def validate_details(details: dict, *, timezone: str = "UTC") -> dict[str, str]:
    result = {}
    for key, label in BOOKING_FIELDS:
        value = details.get(key)
        if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
            raise ValueError(f"Please provide a valid {label}.")
        result[key] = value.strip()
    consultant = details.get("consultant")
    if consultant is not None:
        if not isinstance(consultant, str) or "\n" in consultant or "\r" in consultant:
            raise ValueError("Please provide a valid consultant preference.")
        if consultant.strip():
            result["consultant"] = consultant.strip()
    observations = details.get("observations")
    if observations is not None:
        if not isinstance(observations, str) or not observations.strip() or len(observations) > 1500:
            raise ValueError("Please provide a note of 1 to 1500 characters.")
        result["observations"] = " ".join(observations.split())
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", result["date"]) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}", result["child_dob"]
    ) or not re.fullmatch(r"\d{2}:\d{2}", result["time"]):
        raise ValueError("Please provide valid dates (YYYY-MM-DD) and time (HH:MM).")
    try:
        date.fromisoformat(result["date"])
        dt_time.fromisoformat(result["time"])
    except ValueError as exc:
        raise ValueError("Please provide valid dates (YYYY-MM-DD) and time (HH:MM).") from exc
    validate_patient_dob(result["child_dob"], timezone)
    if result["patient_resides_in_singapore"].casefold() not in {"yes", "no"}:
        raise ValueError("Please confirm whether the patient resides in Singapore (yes or no).")
    return result


def format_summary(details: dict[str, str], proposed_observations: str | None = None) -> str:
    lines = ["Please check these booking details:"]
    lines.extend(f"- {label}: {details[key]}" for key, label in BOOKING_FIELDS)
    if details.get("consultant"):
        lines.append(f"- Consultant: {details['consultant']}")
    else:
        lines.append("- Consultant: No preference")
    if proposed_observations is not None:
        lines.append(f"- Proposed note for consultant (not yet approved): {proposed_observations}")
        lines.extend([
            "",
            "May I include this note in the consultant's calendar appointment? "
            "Please check that it contains no specific or sensitive health details. "
            "Reply APPROVE NOTE to share it, or REMOVE NOTE to continue without it. "
            "To change it, reply NOTE: followed by the replacement.",
            "The appointment is not booked yet.",
        ])
        return "\n".join(lines)
    if details.get("observations"):
        lines.append(f"- Observations for consultant: {details['observations']}")
        notes_action = "To remove this note, reply REMOVE NOTE."
    else:
        notes_action = "If you prefer not to share anything, reply CONFIRM to continue."
    lines.extend(["", f"{OBSERVATIONS_INSTRUCTION} {notes_action}", "", BOOKING_CONFIRMATION_INSTRUCTION])
    return "\n".join(lines)


def preserve_observations(details: dict, pending: PendingBooking | None) -> None:
    # Only retain a note explicitly submitted by the parent for the same patient.
    details.pop("observations", None)
    if (
        pending
        and pending.details.get("observations")
        and isinstance(details.get("child_name"), str)
        and details["child_name"].strip().casefold() == pending.details["child_name"].casefold()
    ):
        details["observations"] = pending.details["observations"]


def existing_proposal(details: dict, pending: PendingBooking | None) -> str | None:
    if (
        pending and pending.proposed_observations
        and isinstance(details.get("child_name"), str)
        and details["child_name"].strip().casefold() == pending.details["child_name"].casefold()
    ):
        return pending.proposed_observations
    return None


def inherit_modification_consultant(details: dict, modification: dict | None, consultants: list[dict]) -> None:
    if not modification or details.get("consultant"):
        return
    name = modification.get("consultant")
    if not name and modification.get("calendar_id"):
        name = next(
            (item["name"] for item in consultants if item.get("calendar_id") == modification["calendar_id"]),
            None,
        )
    if name:
        details["consultant"] = name
    elif len(consultants) > 1:
        raise ValueError("I couldn't identify the existing appointment's consultant. Please contact the clinic.")


def preserves_pending_booking(text: str) -> bool:
    if is_short_acknowledgement(text):
        return True
    return bool(re.fullmatch(
        r"\s*(?:where (?:is|are) (?:the |your )?(?:clinic|location|office)|"
        r"what(?:'s| is) (?:the |your )?(?:address|location|fee|price|cost|"
        r"cancellation policy|opening hours)|how much (?:is it|does it cost))\s*\??\s*",
        text, re.IGNORECASE,
    ))


def has_booking_cancellation(text: str) -> bool:
    if "?" in text and not re.match(r"\s*(?:please\s+)?(?:cancel|stop|abandon)\b", text, re.IGNORECASE):
        return False
    return bool(re.search(r"\b(?:cancel|stop|abandon|never mind|nevermind)\b", text, re.IGNORECASE))


PENDING_BOOKINGS = PendingBookings()
