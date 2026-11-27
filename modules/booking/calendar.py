# modules/booking/calendar.py
import base64
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import pytz
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar"]
APPOINTMENT_DURATION_MINUTES = 60


@dataclass
class Slot:
    date: date
    start_time: time
    location: str


def _validate_appointment_shape(start_time: time, duration_minutes: int) -> None:
    if duration_minutes != APPOINTMENT_DURATION_MINUTES:
        raise ValueError("Appointments are 60 minutes long.")
    if start_time.minute != 0 or start_time.second != 0 or start_time.microsecond != 0:
        raise ValueError("Appointments must start on the hour, such as 10:00.")


def validate_booking_slot(
    booking_date: date,
    start_time: time,
    duration_minutes: int,
    business_hours: dict,
    days: list[str],
    timezone: str,
) -> None:
    """Raise ValueError unless a requested hourly, one-hour slot is valid."""
    _validate_appointment_shape(start_time, duration_minutes)

    tz = pytz.timezone(timezone)
    start_dt = tz.localize(datetime.combine(booking_date, start_time))
    if start_dt <= datetime.now(tz):
        raise ValueError("Choose a date and time in the future.")

    try:
        opening_time = time.fromisoformat(business_hours["start"])
        closing_time = time.fromisoformat(business_hours["end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Business hours are not configured correctly.") from exc

    end_dt = start_dt + timedelta(minutes=duration_minutes)
    if start_time < opening_time or end_dt.date() != booking_date or end_dt.time() > closing_time:
        raise ValueError("That appointment time is outside the configured business hours.")

    configured_days = {day.casefold() for day in days}
    if not configured_days:
        raise ValueError("No booking days are configured for this location.")
    if booking_date.strftime("%A").casefold() not in configured_days:
        raise ValueError(f"This location is not open on {booking_date.strftime('%A')}.")


def patient_age_group(date_of_birth: date, appointment_date: date) -> str:
    """Return the patient's age group on the appointment date without retaining their DOB."""
    age = appointment_date.year - date_of_birth.year
    if (appointment_date.month, appointment_date.day) < (date_of_birth.month, date_of_birth.day):
        age -= 1
    if age < 13:
        return "Child (0-12)"
    if age < 18:
        return "Teen (13-17)"
    return "Adult (18+)"


def _get_credentials() -> Credentials:
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    service_account_info = json.loads(base64.b64decode(raw))
    return Credentials.from_service_account_info(service_account_info, scopes=SCOPES)


def _get_redis():
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        return None
    try:
        import redis
        r = redis.from_url(redis_url)
        r.ping()
        return r
    except Exception:
        return None


class CalendarClient:
    def __init__(self, calendar_id: str, calendar_owner_email: str,
                 business_hours: dict, timezone: str = "America/Argentina/Buenos_Aires"):
        self._calendar_id = calendar_id
        self._owner_email = calendar_owner_email
        self._business_hours = business_hours
        self._tz = pytz.timezone(timezone)
        creds = _get_credentials()
        self._service = build("calendar", "v3", credentials=creds)

    def _slot_key(self, d: date, t: time) -> str:
        return f"slot_lock:{self._calendar_id}:{d.isoformat()}:{t.strftime('%H:%M')}"

    def is_slot_available(self, d: date, start_time: time, duration_minutes: int) -> bool:
        """Check Google Calendar + Redis locks."""
        _validate_appointment_shape(start_time, duration_minutes)
        r = _get_redis()
        if r and r.exists(self._slot_key(d, start_time)):
            return False

        start_dt = self._tz.localize(datetime.combine(d, start_time))
        end_dt = start_dt + timedelta(minutes=duration_minutes)

        body = {
            "timeMin": start_dt.isoformat(),
            "timeMax": end_dt.isoformat(),
            "items": [{"id": self._calendar_id}],
        }
        result = self._service.freebusy().query(body=body).execute()
        busy = result["calendars"][self._calendar_id]["busy"]
        return len(busy) == 0

    def find_available_slots(
        self,
        start_date: date,
        end_date: date,
        duration_minutes: int,
        business_hours: dict,
        booking_days: list[str],
        location: str,
    ) -> list[Slot]:
        """Find available on-the-hour slots with one Google Calendar busy query."""
        _validate_appointment_shape(time(0), duration_minutes)
        if end_date < start_date:
            raise ValueError("The end date must be on or after the start date.")
        try:
            opening_time = time.fromisoformat(business_hours["start"])
            closing_time = time.fromisoformat(business_hours["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Business hours are not configured correctly.") from exc

        configured_days = {day.casefold() for day in booking_days}
        range_start = self._tz.localize(datetime.combine(start_date, time.min))
        range_end = self._tz.localize(
            datetime.combine(end_date + timedelta(days=1), time.min)
        )
        result = self._service.freebusy().query(
            body={
                "timeMin": range_start.isoformat(),
                "timeMax": range_end.isoformat(),
                "items": [{"id": self._calendar_id}],
            }
        ).execute()
        busy_intervals = [
            (
                datetime.fromisoformat(interval["start"].replace("Z", "+00:00")),
                datetime.fromisoformat(interval["end"].replace("Z", "+00:00")),
            )
            for interval in result["calendars"][self._calendar_id]["busy"]
        ]

        redis_client = _get_redis()
        now = datetime.now(self._tz)
        slots = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.strftime("%A").casefold() in configured_days:
                start_hour = opening_time.hour
                if opening_time.minute or opening_time.second:
                    start_hour += 1
                end_hour = closing_time.hour
                for hour in range(start_hour, end_hour + 1):
                    start_time = time(hour)
                    if start_time < opening_time:
                        continue
                    slot_start = self._tz.localize(
                        datetime.combine(current_date, start_time)
                    )
                    slot_end = slot_start + timedelta(minutes=duration_minutes)
                    if slot_end.time() > closing_time or slot_end.date() != current_date:
                        continue
                    if slot_start <= now:
                        continue
                    if redis_client and redis_client.exists(
                        self._slot_key(current_date, start_time)
                    ):
                        continue
                    if any(
                        busy_start < slot_end and busy_end > slot_start
                        for busy_start, busy_end in busy_intervals
                    ):
                        continue
                    slots.append(Slot(current_date, start_time, location))
            current_date += timedelta(days=1)
        return slots

    def lock_slot(self, d: date, start_time: time, ttl_seconds: int = 1800) -> None:
        """Soft-lock a slot in Redis for TTL seconds (default 30 min)."""
        r = _get_redis()
        if r:
            r.setex(self._slot_key(d, start_time), ttl_seconds, "1")

    def release_slot(self, d: date, start_time: time) -> None:
        r = _get_redis()
        if r:
            r.delete(self._slot_key(d, start_time))

    def mark_event_attendance_confirmed(self, event_id: str) -> None:
        event = self._service.events().get(
            calendarId=self._calendar_id,
            eventId=event_id,
        ).execute()
        patient_name = self._event_patient_name(event)
        if not patient_name:
            raise ValueError(f"Calendar event {event_id} has no patient name.")

        summary = event.get("summary", "")
        confirmed_name = f"{patient_name} (CONFIRMED)"
        if confirmed_name not in summary:
            if patient_name in summary:
                summary = summary.replace(patient_name, confirmed_name, 1)
            else:
                summary = f"{summary} - Patient: {confirmed_name}"

        private_properties = dict(
            event.get("extendedProperties", {}).get("private", {})
        )
        start_raw = event.get("start", {}).get("dateTime", "")
        if start_raw:
            private_properties["attendance_confirmed_for"] = (
                datetime.fromisoformat(start_raw).astimezone(self._tz).date().isoformat()
            )
        self._service.events().patch(
            calendarId=self._calendar_id,
            eventId=event_id,
            body={
                "summary": summary,
                "extendedProperties": {"private": private_properties},
            },
        ).execute()

    def create_event(
        self, service_name: str, user_name: str, user_phone: str,
        slot: Slot, duration_minutes: int,
        price: int, cancellation_policy: str, telegram_chat_id: int | None = None,
        child_name: str = "", parent_name: str = "",
        referral_name: str = "None", patient_age_group: str = "",
        observations: str = "",
    ) -> str:
        _validate_appointment_shape(slot.start_time, duration_minutes)
        start_dt = self._tz.localize(datetime.combine(slot.date, slot.start_time))
        end_dt = start_dt + timedelta(minutes=duration_minutes)

        description = (
            f"Service: {service_name}\n"
            f"Price: SGD {price:,}"
        )
        if user_phone:
            description += f"\nWhatsApp: {user_phone}"
        if child_name:
            description += f"\nPatient: {child_name}"
        if parent_name:
            description += f"\nParent/guardian: {parent_name}"
        if patient_age_group:
            description += f"\nPatient age group: {patient_age_group}"
        description += f"\nReferring doctor: {referral_name.strip() or 'None'}"
        if telegram_chat_id is not None:
            description += f"\nTelegram Chat ID: {telegram_chat_id}"
        if observations:
            description += f"\nParent's observations for consultant: {' '.join(observations.split())}"

        event = {
            "summary": (
                f"{service_name} - Patient: {child_name} | Parent: {parent_name}"
                if child_name and parent_name
                else f"{service_name} - {child_name or user_name}"
            ),
            "description": description,
            "start": {"dateTime": start_dt.isoformat(), "timeZone": str(self._tz)},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": str(self._tz)},
            "reminders": {"useDefault": False, "overrides": []},
        }

        result = self._service.events().insert(
            calendarId=self._calendar_id,
            body=event,
        ).execute()
        return result["id"]

    def find_upcoming_events_by_phone(self, phone: str, horizon_days: int = 183) -> list[dict]:
        """Find events in the next ~6 months (all of them, not just the next 10) whose description has this phone number."""
        now_dt = datetime.now(self._tz)
        items: list[dict] = []
        page_token = None
        while True:
            result = self._service.events().list(
                calendarId=self._calendar_id,
                timeMin=now_dt.isoformat(),
                timeMax=(now_dt + timedelta(days=horizon_days)).isoformat(),
                maxResults=250,
                singleEvents=True,
                orderBy="startTime",
                pageToken=page_token,
            ).execute()
            items.extend(result.get("items", []))
            page_token = result.get("nextPageToken")
            if not page_token:
                break

        matching = []
        for event in items:
            desc = event.get("description", "")
            if phone in desc:
                start = event["start"].get("dateTime", "")
                # Format date as human-readable English
                date_str = start[:10] if start else ""
                if date_str:
                    from datetime import date as date_cls
                    d = date_cls.fromisoformat(date_str)
                    day_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
                                 4: "Friday", 5: "Saturday", 6: "Sunday"}
                    month_names = {1: "January", 2: "February", 3: "March", 4: "April",
                                   5: "May", 6: "June", 7: "July", 8: "August",
                                   9: "September", 10: "October", 11: "November", 12: "December"}
                    date_str = f"{day_names[d.weekday()]} {month_names[d.month]} {d.day}, {d.year}"
                matching.append({
                    "id": event["id"],
                    "calendar_id": self._calendar_id,
                    "start_datetime": start,
                    "summary": event.get("summary", ""),
                    "patient_name": self._event_patient_name(event),
                    "date": date_str,
                    "time": start[11:16] if start else "",
                    "location": event.get("location", ""),
                    "reminder_sent_for": event.get("extendedProperties", {}).get(
                        "private", {}
                    ).get("whatsapp_reminder_sent_for"),
                    "attendance_confirmed_for": event.get(
                        "extendedProperties", {}
                    ).get("private", {}).get("attendance_confirmed_for"),
                })
        return matching

    def find_upcoming_events_by_telegram_chat_id(self, chat_id: int) -> list[dict]:
        """Find future events associated with a Telegram chat on this calendar."""
        now = datetime.now(self._tz).isoformat()
        result = self._service.events().list(
            calendarId=self._calendar_id,
            timeMin=now,
            maxResults=10,
            singleEvents=True,
            orderBy="startTime",
        ).execute()

        matching = []
        expected = f"Telegram Chat ID: {chat_id}".casefold()
        for event in result.get("items", []):
            description = event.get("description", "")
            if expected not in description.casefold():
                continue
            start = event.get("start", {}).get("dateTime", "")
            date_str = start[:10] if start else ""
            if date_str:
                from datetime import date as date_cls
                event_date = date_cls.fromisoformat(date_str)
                date_str = (
                    f"{event_date.strftime('%A')} {event_date.strftime('%B')} "
                    f"{event_date.day}, {event_date.year}"
                )
            matching.append({
                "id": event["id"],
                "calendar_id": self._calendar_id,
                "start_datetime": start,
                "summary": event.get("summary", ""),
                "patient_name": self._event_patient_name(event),
                "date": date_str,
                "time": start[11:16] if start else "",
                "location": event.get("location", ""),
                "reminder_sent_for": event.get("extendedProperties", {}).get(
                    "private", {}
                ).get("telegram_reminder_sent_for"),
                "attendance_confirmed_for": event.get(
                    "extendedProperties", {}
                ).get("private", {}).get("attendance_confirmed_for"),
            })
        return matching

    def find_next_event_by_telegram_chat_and_names(
        self,
        chat_id: int,
        patient_name: str,
        parent_name: str,
    ) -> dict | None:
        """Find the next future event matching the Telegram chat and both names."""
        now = datetime.now(self._tz).isoformat()
        page_token = None
        expected_chat_id = str(chat_id)
        expected_patient = " ".join(patient_name.casefold().split())
        expected_parent = " ".join(parent_name.casefold().split())

        while True:
            query = {
                "calendarId": self._calendar_id,
                "timeMin": now,
                "maxResults": 2500,
                "singleEvents": True,
                "orderBy": "startTime",
            }
            if page_token:
                query["pageToken"] = page_token
            result = self._service.events().list(**query).execute()

            for event in result.get("items", []):
                if event.get("status") == "cancelled":
                    continue
                start_raw = event.get("start", {}).get("dateTime", "")
                if not start_raw:
                    continue

                fields = {}
                for line in event.get("description", "").splitlines():
                    label, separator, value = line.partition(":")
                    if separator:
                        fields[label.strip().casefold()] = value.strip()
                if fields.get("telegram chat id") != expected_chat_id:
                    continue

                event_patient = " ".join(fields.get("patient", "").casefold().split())
                event_parent = " ".join(
                    fields.get("parent/guardian", "").casefold().split()
                )
                if event_patient != expected_patient or event_parent != expected_parent:
                    continue

                local_start = datetime.fromisoformat(start_raw).astimezone(self._tz)
                summary = event.get("summary", "")
                service_name, _, _ = summary.partition(" - Patient:")
                return {
                    "service": service_name or summary,
                    "date": (
                        f"{local_start.strftime('%A, %B ')}{local_start.day}"
                        f"{local_start.strftime(', %Y')}"
                    ),
                    "time": local_start.strftime("%H:%M"),
                    "location": event.get("location", ""),
                    "start_datetime": local_start.isoformat(),
                }

            page_token = result.get("nextPageToken")
            if not page_token:
                return None

    @staticmethod
    def _event_patient_name(event: dict) -> str:
        for line in event.get("description", "").splitlines():
            label, separator, value = line.partition(":")
            if separator and label.strip().casefold() == "patient":
                return value.strip()
        return ""

    def has_upcoming_appointment_for_patient(self, patient_name: str) -> bool:
        """Check whether the patient already has an upcoming calendar event."""
        target_name = patient_name.strip().casefold()
        if not target_name:
            return False

        page_token = None
        while True:
            result = self._service.events().list(
                calendarId=self._calendar_id,
                timeMin=datetime.now(self._tz).isoformat(),
                maxResults=2500,
                singleEvents=True,
                orderBy="startTime",
                pageToken=page_token,
            ).execute()

            for event in result.get("items", []):
                if event.get("status") == "cancelled":
                    continue
                if not event.get("start", {}).get("dateTime"):
                    continue

                description = event.get("description", "")
                for line in description.splitlines():
                    label, separator, value = line.partition(":")
                    if (
                        separator
                        and label.strip().casefold() == "patient"
                        and value.strip().casefold() == target_name
                    ):
                        logger.info(
                            "Found an upcoming patient appointment in calendar %s (event %s)",
                            self._calendar_id,
                            event.get("id", "unknown"),
                        )
                        return True

                summary = event.get("summary", "")
                _, separator, parent = summary.rpartition(" - Patient: ")
                patient_name_in_summary, parent_separator, _ = parent.partition(" | Parent:")
                if (
                    separator
                    and parent_separator
                    and patient_name_in_summary.strip().casefold() == target_name
                ):
                    logger.info(
                        "Found an upcoming patient appointment in calendar %s (event %s)",
                        self._calendar_id,
                        event.get("id", "unknown"),
                    )
                    return True

            page_token = result.get("nextPageToken")
            if not page_token:
                return False

    def delete_event(self, event_id: str) -> None:
        """Delete an event from the calendar."""
        self._service.events().delete(
            calendarId=self._calendar_id,
            eventId=event_id,
        ).execute()
