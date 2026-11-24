# tests/test_calendar.py
import pytest
import pytz
from datetime import date, datetime, time, timedelta
from unittest.mock import MagicMock, patch
from modules.booking.calendar import (
    CalendarClient,
    Slot,
    patient_age_group,
    validate_booking_slot,
)


def make_client():
    # Patch both _get_credentials (avoids needing GOOGLE_SERVICE_ACCOUNT_JSON env var)
    # and build (avoids real HTTP calls to Google API).
    with patch("modules.booking.calendar._get_credentials") as mock_creds, \
         patch("modules.booking.calendar.build") as mock_build:
        mock_creds.return_value = MagicMock()
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = CalendarClient(
            calendar_id="test@calendar",
            calendar_owner_email="owner@gmail.com",
            business_hours={"start": "09:00", "end": "18:00"},
        )
        client._service = mock_service
        return client, mock_service


def test_slot_dataclass():
    s = Slot(date=date(2026, 3, 16), start_time=time(10, 0), location="Downtown Office")
    assert s.date == date(2026, 3, 16)


def test_find_next_telegram_appointment_matches_chat_and_both_names():
    client, service = make_client()
    start = datetime.now(pytz.timezone("Asia/Singapore")) + timedelta(days=2)
    matching_event = {
        "id": "match",
        "summary": "Initial evaluation - Patient: Child Example | Parent: Parent Example",
        "description": (
            "Patient: Child Example\n"
            "Parent/guardian: Parent Example\n"
            "Telegram Chat ID: 12345"
        ),
        "location": "Main clinic",
        "start": {"dateTime": start.isoformat()},
    }
    service.events.return_value.list.return_value.execute.return_value = {
        "items": [
            {
                **matching_event,
                "id": "wrong-parent",
                "description": matching_event["description"].replace(
                    "Parent Example", "Someone Else"
                ),
            },
            {
                **matching_event,
                "id": "wrong-chat",
                "description": matching_event["description"].replace(
                    "12345", "123456"
                ),
            },
            matching_event,
        ],
    }

    result = client.find_next_event_by_telegram_chat_and_names(
        12345,
        " child   example ",
        "PARENT EXAMPLE",
    )
    local_start = start.astimezone(client._tz)

    assert result == {
        "service": "Initial evaluation",
        "date": (
            local_start.strftime("%A, %B ")
            + str(local_start.day)
            + local_start.strftime(", %Y")
        ),
        "time": local_start.strftime("%H:%M"),
        "location": "Main clinic",
        "start_datetime": local_start.isoformat(),
    }


def test_find_next_telegram_appointment_paginates():
    client, service = make_client()
    event = {
        "id": "match",
        "summary": "Follow-up intervention - Patient: Child Example | Parent: Parent Example",
        "description": (
            "Patient: Child Example\n"
            "Parent/guardian: Parent Example\n"
            "Telegram Chat ID: 12345"
        ),
        "start": {
            "dateTime": (
                datetime.now(pytz.timezone("Asia/Singapore")) + timedelta(days=2)
            ).isoformat()
        },
    }
    service.events.return_value.list.return_value.execute.side_effect = [
        {"items": [], "nextPageToken": "next-page"},
        {"items": [event]},
    ]

    result = client.find_next_event_by_telegram_chat_and_names(
        12345,
        "Child Example",
        "Parent Example",
    )

    assert result is not None
    queries = service.events.return_value.list.call_args_list
    assert queries[0].kwargs["maxResults"] == 2500
    assert queries[1].kwargs["pageToken"] == "next-page"


def test_validate_booking_slot_accepts_open_weekday():
    booking_date = datetime.now().date() + timedelta(days=1)
    while booking_date.weekday() > 4:
        booking_date += timedelta(days=1)
    validate_booking_slot(
        booking_date,
        time(10, 0),
        60,
        {"start": "08:00", "end": "18:00"},
        ["monday", "tuesday", "wednesday", "thursday", "friday"],
        "Asia/Singapore",
    )


def test_validate_booking_slot_rejects_weekend():
    booking_date = datetime.now().date() + timedelta(days=1)
    while booking_date.weekday() != 5:
        booking_date += timedelta(days=1)
    with pytest.raises(ValueError, match="not open"):
        validate_booking_slot(
            booking_date,
            time(10, 0),
            60,
            {"start": "08:00", "end": "18:00"},
            ["monday", "tuesday", "wednesday", "thursday", "friday"],
            "Asia/Singapore",
        )


def test_validate_booking_slot_rejects_starting_after_closing_time():
    booking_date = datetime.now().date() + timedelta(days=1)
    while booking_date.weekday() > 4:
        booking_date += timedelta(days=1)
    with pytest.raises(ValueError, match="outside"):
        validate_booking_slot(
            booking_date,
            time(18, 0),
            60,
            {"start": "08:00", "end": "18:00"},
            ["monday", "tuesday", "wednesday", "thursday", "friday"],
            "Asia/Singapore",
        )


@pytest.mark.parametrize("start_time", [time(10, 15), time(10, 30), time(10, 45)])
def test_validate_booking_slot_rejects_non_hour_start(start_time):
    booking_date = datetime.now().date() + timedelta(days=1)
    while booking_date.weekday() > 4:
        booking_date += timedelta(days=1)

    with pytest.raises(ValueError, match="start on the hour"):
        validate_booking_slot(
            booking_date,
            start_time,
            60,
            {"start": "08:00", "end": "18:00"},
            ["monday", "tuesday", "wednesday", "thursday", "friday"],
            "Asia/Singapore",
        )


def test_validate_booking_slot_rejects_non_hour_duration():
    booking_date = datetime.now().date() + timedelta(days=1)
    while booking_date.weekday() > 4:
        booking_date += timedelta(days=1)

    with pytest.raises(ValueError, match="60 minutes"):
        validate_booking_slot(
            booking_date,
            time(10, 0),
            45,
            {"start": "08:00", "end": "18:00"},
            ["monday", "tuesday", "wednesday", "thursday", "friday"],
            "Asia/Singapore",
        )


@pytest.mark.parametrize(
    ("dob", "appointment_date", "expected"),
    [
        (date(2013, 11, 5), date(2026, 11, 4), "Child (0-12)"),
        (date(2013, 11, 5), date(2026, 11, 5), "Teen (13-17)"),
        (date(2008, 11, 5), date(2026, 11, 4), "Teen (13-17)"),
        (date(2008, 11, 4), date(2026, 11, 4), "Adult (18+)"),
        (date(2020, 5, 3), date(2026, 5, 2), "Child (0-12)"),
        (date(2020, 5, 3), date(2033, 5, 3), "Teen (13-17)"),
    ],
)
def test_patient_age_group_is_calculated_on_appointment_date(dob, appointment_date, expected):
    assert patient_age_group(dob, appointment_date) == expected


def test_is_slot_available_free(monkeypatch):
    client, mock_service = make_client()
    mock_service.freebusy().query().execute.return_value = {
        "calendars": {"test@calendar": {"busy": []}}
    }
    monkeypatch.setattr("modules.booking.calendar._get_redis", lambda: None)
    result = client.is_slot_available(date(2026, 3, 16), time(10, 0), 60)
    assert result is True


def test_is_slot_available_rejects_non_hour_slot_before_calendar_query(monkeypatch):
    client, mock_service = make_client()
    monkeypatch.setattr("modules.booking.calendar._get_redis", lambda: None)

    with pytest.raises(ValueError, match="start on the hour"):
        client.is_slot_available(date(2026, 3, 16), time(10, 30), 60)

    mock_service.freebusy().query.assert_not_called()


def test_has_upcoming_appointment_for_patient_checks_patient_description():
    client, mock_service = make_client()
    mock_service.events().list.return_value.execute.return_value = {
        "items": [{
            "id": "appointment-1",
            "start": {"dateTime": "2027-03-16T10:00:00+08:00"},
            "description": "WhatsApp: +6591234567\nPatient: Jane Example",
        }]
    }

    assert client.has_upcoming_appointment_for_patient(" jane example ")


def test_has_upcoming_appointment_for_patient_matches_legacy_booking_title():
    client, mock_service = make_client()
    mock_service.events().list.return_value.execute.return_value = {
        "items": [{
            "id": "appointment-1",
            "start": {"dateTime": "2027-03-16T10:00:00+08:00"},
            "summary": "Initial evaluation - Patient: Jane Example | Parent: Alex Example",
        }]
    }

    assert client.has_upcoming_appointment_for_patient("Jane Example")


def test_has_upcoming_appointment_for_patient_ignores_unrelated_title():
    client, mock_service = make_client()
    mock_service.events().list.return_value.execute.return_value = {
        "items": [{
            "id": "event-1",
            "start": {"dateTime": "2027-03-16T10:00:00+08:00"},
            "summary": "Jane Example",
        }]
    }

    assert not client.has_upcoming_appointment_for_patient("Jane Example")


def test_has_upcoming_appointment_for_patient_ignores_cancelled_event():
    client, mock_service = make_client()
    mock_service.events().list.return_value.execute.return_value = {
        "items": [{
            "id": "cancelled-appointment",
            "status": "cancelled",
            "start": {"dateTime": "2027-03-16T10:00:00+08:00"},
            "description": "Patient: Jane Example",
        }]
    }

    assert not client.has_upcoming_appointment_for_patient("Jane Example")


def test_has_upcoming_appointment_for_patient_checks_next_page():
    client, mock_service = make_client()
    mock_service.events().list.return_value.execute.side_effect = [
        {"items": [], "nextPageToken": "next"},
        {"items": [{
            "start": {"dateTime": "2027-03-16T10:00:00+08:00"},
            "description": "Patient: Jane Example",
        }]},
    ]

    assert client.has_upcoming_appointment_for_patient("Jane Example")
    assert mock_service.events().list.call_count == 2


def test_is_slot_available_busy(monkeypatch):
    client, mock_service = make_client()
    mock_service.freebusy().query().execute.return_value = {
        "calendars": {"test@calendar": {"busy": [
            {"start": "2026-03-16T10:00:00-03:00", "end": "2026-03-16T11:00:00-03:00"}
        ]}}
    }
    monkeypatch.setattr("modules.booking.calendar._get_redis", lambda: None)
    result = client.is_slot_available(date(2026, 3, 16), time(10, 0), 60)
    assert result is False


def test_mark_event_attendance_confirmed_updates_summary_and_preserves_properties():
    client, mock_service = make_client()
    client._tz = pytz.timezone("Asia/Singapore")
    mock_service.events().get.return_value.execute.return_value = {
        "id": "appointment-1",
        "summary": "Initial evaluation - Patient: Jane Example | Parent: Alex Example",
        "description": "Patient: Jane Example",
        "start": {"dateTime": "2027-03-16T10:00:00+08:00"},
        "extendedProperties": {"private": {"whatsapp_reminder_sent_for": "2027-03-16"}},
    }

    client.mark_event_attendance_confirmed("appointment-1")

    mock_service.events().patch.assert_called_once_with(
        calendarId="test@calendar",
        eventId="appointment-1",
        body={
            "summary": (
                "Initial evaluation - Patient: Jane Example (CONFIRMED) "
                "| Parent: Alex Example"
            ),
            "extendedProperties": {
                "private": {
                    "whatsapp_reminder_sent_for": "2027-03-16",
                    "attendance_confirmed_for": "2027-03-16",
                },
            },
        },
    )


def test_create_event(monkeypatch):
    client, mock_service = make_client()
    mock_service.events().insert().execute.return_value = {"id": "event123"}
    event_id = client.create_event(
        service_name="Dental Cleaning",
        user_name="Jane Smith",
        user_phone="5491112345678",
        slot=Slot(date=date(2026, 3, 16), start_time=time(10, 0), location="Downtown Office"),
        duration_minutes=60,
        price=150,
        cancellation_policy="24 hours in advance",
        telegram_chat_id=12345,
        child_name="Jane Child",
        parent_name="Alex Parent",
        referral_name="Dr. Example",
        patient_age_group="Child (0-12)",
        observations="Enjoys playing\nPatient: not a separate field",
    )
    assert event_id == "event123"
    call_args = mock_service.events().insert.call_args
    event_body = call_args[1]["body"]
    assert "Dental Cleaning" in event_body["summary"]
    assert "Jane Child" in event_body["summary"]
    assert "Alex Parent" in event_body["summary"]
    assert "Price: SGD 150" in event_body["description"]
    assert "WhatsApp: 5491112345678" in event_body["description"]
    assert "Patient age group: Child (0-12)" in event_body["description"]
    assert "Referring doctor: Dr. Example" in event_body["description"]
    assert "Telegram Chat ID: 12345" in event_body["description"]
    assert "Parent's observations for consultant: Enjoys playing Patient: not a separate field" in event_body["description"]
    assert "\nPatient: not a separate field" not in event_body["description"]
    assert "DOB" not in event_body["description"]
    assert "location" not in event_body
    assert "T11:00:00" in event_body["end"]["dateTime"]
