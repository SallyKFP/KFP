import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
import pytz
from config.loader import load_config
from reminders.scheduler import (
    extract_patient_name,
    extract_telegram_chat_id_from_description,
    extract_phone_from_description,
    format_reminder_message,
    send_reminders,
    send_telegram_reminders,
)


def make_mock_event(summary, description, start_datetime, location="456 Oak Avenue, Springfield"):
    return {
        "id": "event-1",
        "summary": summary,
        "description": description,
        "location": location,
        "start": {"dateTime": start_datetime},
    }


def test_configured_reminder_includes_rescheduling_policy():
    template = load_config()["reminders"]["message_template"]
    assert "more than 36 hours before the appointment" in template
    assert "valid medical certificate (MC)" in template
    assert "cancel and make a new booking from scratch" in template
    assert "all booking details again" in template


@pytest.mark.asyncio
async def test_failed_whatsapp_send_does_not_mark_reminder_sent():
    config = {
        "client": {"timezone": "Asia/Singapore"},
        "booking": {"calendar_id": "test@calendar"},
        "reminders": {"message_template": "Reminder for {patient}"},
    }
    service = MagicMock()
    service.events().list().execute.return_value = {"items": [
        make_mock_event("Consultation - Jane", "Phone: 6591234567", "2026-11-11T10:00:00+08:00")
    ]}
    wa = MagicMock()
    wa.send_text = AsyncMock(side_effect=RuntimeError("send failed"))
    with pytest.raises(RuntimeError, match="send failed"):
        await send_reminders(config, service, wa)
    service.events().patch.assert_not_called()


@pytest.mark.asyncio
async def test_daily_scheduler_logs_failure_and_runs_again(monkeypatch, caplog):
    import asyncio
    import reminders.scheduler as scheduler

    sleep = AsyncMock(side_effect=[None, None, asyncio.CancelledError()])
    monkeypatch.setattr(scheduler.asyncio, "sleep", sleep)
    send = AsyncMock(side_effect=[RuntimeError("test failure"), 1])
    with pytest.raises(asyncio.CancelledError):
        await scheduler.run_daily_reminders("Asia/Singapore", "Telegram", send)
    assert send.await_count == 2
    assert "Telegram scheduled reminder run failed" in caplog.text


def test_extract_phone_from_description():
    desc = "Service: Dental Cleaning\nPrice: $150\nPhone: 5491112345678\nPolicy: 24 hours"
    assert extract_phone_from_description(desc) == "5491112345678"
    assert extract_phone_from_description("WhatsApp: +6591234567") == "+6591234567"


def test_extract_phone_missing():
    assert extract_phone_from_description("No phone here") is None


def test_extract_telegram_chat_id_from_description():
    assert extract_telegram_chat_id_from_description(
        "Service: Consultation\nTelegram Chat ID: 12345"
    ) == 12345
    assert extract_telegram_chat_id_from_description("No Telegram chat") is None


def test_extract_patient_name_from_calendar_description():
    event = make_mock_event(
        "Initial evaluation - Patient: Yew Mei Green | Parent: Vanda Tan",
        "Service: Initial evaluation\nPatient: Yew Mei Green\nParent/guardian: Vanda Tan",
        "2026-11-10T09:00:00+08:00",
    )

    assert extract_patient_name(event) == "Yew Mei Green"


def test_format_reminder_message():
    template = (
        "Good day,\n\n{patient} has an appointment with {consultant} on:\n\n"
        "{date}, {time}\n\nPlease reply YES."
    )
    event = make_mock_event(
        "Initial evaluation - Patient: Yew Mei Green | Parent: Vanda Tan",
        "Phone: 5491112345678\nPatient: Yew Mei Green",
        "2026-11-10T09:00:00+08:00",
    )
    msg = format_reminder_message(template, event, "Consultant Sarah")
    assert msg == (
        "Good day,\n\nYew Mei Green has an appointment with Consultant Sarah on:\n\n"
        "Tuesday, 10 November 2026, 09:00\n\nPlease reply YES.\n\n"
        "To confirm attendance, reply YES only, without punctuation or other words. "
        "Other replies will not confirm attendance."
    )


@pytest.mark.asyncio
async def test_send_reminders_sends_message_for_each_event():
    config = {
        "client": {"timezone": "America/Argentina/Buenos_Aires"},
        "booking": {"calendar_id": "test@calendar"},
        "reminders": {"message_template": "Reminder for {patient} with {consultant} on {date} at {time}."},
    }

    mock_service = MagicMock()
    mock_service.events().list().execute.return_value = {
        "items": [
            make_mock_event(
                "Dental Cleaning - Jane",
                "Phone: 5491112345678\nPatient: Jane",
                "2026-03-19T10:00:00-03:00",
            )
        ]
    }

    mock_wa = MagicMock()
    mock_wa.send_text = AsyncMock()

    count = await send_reminders(config, mock_service, mock_wa)
    assert count == 1
    query = mock_service.events().list.call_args.kwargs
    assert datetime.fromisoformat(query["timeMin"]).date() == (
        datetime.now(pytz.timezone("America/Argentina/Buenos_Aires")).date() + timedelta(days=2)
    )
    mock_wa.send_text.assert_called_once()
    call_phone = mock_wa.send_text.call_args[0][0]
    assert call_phone == "5491112345678"
    assert "Jane" in mock_wa.send_text.call_args.args[1]
    mock_service.events().patch.assert_called_once()


@pytest.mark.asyncio
async def test_send_reminders_skips_event_without_phone():
    config = {
        "client": {"timezone": "America/Argentina/Buenos_Aires"},
        "booking": {"calendar_id": "test@calendar"},
        "reminders": {"message_template": "Reminder for {patient} with {consultant} at {time}."},
    }

    mock_service = MagicMock()
    mock_service.events().list().execute.return_value = {
        "items": [
            make_mock_event("Dental Cleaning - Jane", "No phone here", "2026-03-19T10:00:00-03:00")
        ]
    }

    mock_wa = MagicMock()
    mock_wa.send_text = AsyncMock()

    count = await send_reminders(config, mock_service, mock_wa)
    assert count == 0
    mock_wa.send_text.assert_not_called()


@pytest.mark.asyncio
async def test_send_reminders_skips_event_already_reminded_for_appointment_date():
    config = {
        "client": {"timezone": "America/Argentina/Buenos_Aires"},
        "booking": {"calendar_id": "test@calendar"},
        "reminders": {"message_template": "Reminder for {patient} with {consultant}."},
    }
    event = make_mock_event(
        "Initial evaluation - Patient: Jane",
        "Phone: 5491112345678\nPatient: Jane",
        "2026-03-19T10:00:00-03:00",
    )
    event["extendedProperties"] = {
        "private": {"whatsapp_reminder_sent_for": "2026-03-19"}
    }
    mock_service = MagicMock()
    mock_service.events().list().execute.return_value = {"items": [event]}
    mock_wa = MagicMock()
    mock_wa.send_text = AsyncMock()

    count = await send_reminders(config, mock_service, mock_wa)

    assert count == 0
    mock_wa.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_telegram_reminders_sends_and_marks_event():
    config = {
        "client": {"timezone": "Asia/Singapore"},
        "booking": {"calendar_id": "test@calendar"},
        "reminders": {"message_template": "Reminder for {patient} with {consultant} at {time}."},
    }
    event = make_mock_event(
        "Consultation - Jane",
        "Telegram Chat ID: 12345",
        (
            datetime.now(pytz.timezone("Asia/Singapore")) + timedelta(days=2)
        ).isoformat(),
    )
    event["id"] = "event-1"
    mock_service = MagicMock()
    mock_service.events().list().execute.return_value = {"items": [event]}
    send_message = AsyncMock()

    count = await send_telegram_reminders(config, mock_service, send_message)

    assert count == 1
    query = mock_service.events().list.call_args.kwargs
    assert datetime.fromisoformat(query["timeMin"]).date() == (
        datetime.now(pytz.timezone("Asia/Singapore")).date() + timedelta(days=2)
    )
    send_message.assert_awaited_once()
    assert send_message.call_args.args[0] == 12345
    mock_service.events().patch.assert_called_once()


@pytest.mark.asyncio
async def test_failed_telegram_send_does_not_mark_reminder_sent():
    config = {
        "client": {"timezone": "Asia/Singapore"},
        "booking": {"calendar_id": "test@calendar"},
        "reminders": {"message_template": "Reminder for {patient}"},
    }
    service = MagicMock()
    service.events().list().execute.return_value = {"items": [
        make_mock_event("Consultation - Jane", "Telegram Chat ID: 12345", "2026-11-11T10:00:00+08:00")
    ]}
    send = AsyncMock(side_effect=RuntimeError("telegram failed"))
    with pytest.raises(RuntimeError, match="telegram failed"):
        await send_telegram_reminders(config, service, send)
    service.events().patch.assert_not_called()


@pytest.mark.asyncio
async def test_channel_markers_are_independent():
    config = {
        "client": {"timezone": "Asia/Singapore"},
        "booking": {"calendar_id": "test@calendar"},
        "reminders": {"message_template": "Reminder for {patient}"},
    }
    event = make_mock_event(
        "Consultation - Jane", "Telegram Chat ID: 12345\nPhone: 6591234567",
        "2026-11-11T10:00:00+08:00",
    )
    event["extendedProperties"] = {"private": {"whatsapp_reminder_sent_for": "2026-11-11"}}
    service = MagicMock()
    service.events().list().execute.return_value = {"items": [event]}
    send = AsyncMock()
    assert await send_telegram_reminders(config, service, send) == 1
    patch = service.events().patch.call_args.kwargs["body"]["extendedProperties"]["private"]
    assert patch["whatsapp_reminder_sent_for"] == "2026-11-11"
    assert patch["telegram_reminder_sent_for"] == "2026-11-11"


@pytest.mark.asyncio
async def test_send_telegram_reminders_skips_already_reminded_event():
    config = {
        "client": {"timezone": "Asia/Singapore"},
        "booking": {"calendar_id": "test@calendar"},
        "reminders": {"message_template": "Reminder for {patient} with {consultant} at {time}."},
    }
    reminder_day = datetime.now(pytz.timezone("Asia/Singapore")) + timedelta(days=2)
    event = make_mock_event(
        "Consultation - Jane",
        "Telegram Chat ID: 12345",
        reminder_day.isoformat(),
    )
    event["id"] = "event-1"
    event["extendedProperties"] = {
        "private": {"telegram_reminder_sent_for": reminder_day.date().isoformat()}
    }
    mock_service = MagicMock()
    mock_service.events().list().execute.return_value = {"items": [event]}
    send_message = AsyncMock()

    count = await send_telegram_reminders(config, mock_service, send_message)

    assert count == 0
    send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_telegram_reminders_searches_each_consultant_calendar():
    config = {
        "client": {"timezone": "Asia/Singapore"},
        "booking": {
            "consultants": [
                {"name": "Consultant One", "calendar_id": "one@calendar"},
                {"name": "Consultant Two", "calendar_id": "two@calendar"},
            ],
        },
        "reminders": {"message_template": "Reminder for {patient} with {consultant} at {time}."},
    }
    event = make_mock_event(
        "Consultation - Jane",
        "Telegram Chat ID: 12345",
        (datetime.now(pytz.timezone("Asia/Singapore")) + timedelta(days=2)).isoformat(),
    )
    event["id"] = "event-1"
    mock_service = MagicMock()
    list_operation = mock_service.events.return_value.list
    list_operation.return_value.execute.side_effect = [
        {"items": [event]},
        {"items": []},
    ]
    send_message = AsyncMock()

    count = await send_telegram_reminders(config, mock_service, send_message)

    assert count == 1
    queries = list_operation.call_args_list
    assert [query.kwargs["calendarId"] for query in queries] == [
        "one@calendar",
        "two@calendar",
    ]
    assert mock_service.events().patch.call_args.kwargs["calendarId"] == "one@calendar"
    send_message.assert_awaited_once()
    assert "Jane" in send_message.call_args.args[1]
    assert "Consultant One" in send_message.call_args.args[1]


def _switch_off_setup():
    config = {
        "client": {"timezone": "America/Argentina/Buenos_Aires"},
        "booking": {"calendar_id": "test@calendar"},
        "reminders": {"message_template": "Reminder for {patient} with {consultant} on {date} at {time}."},
    }
    service = MagicMock()
    service.events().list().execute.return_value = {
        "items": [
            make_mock_event(
                "Dental Cleaning - Jane",
                "Phone: 5491112345678\nPatient: Jane",
                "2026-03-19T10:00:00-03:00",
            )
        ]
    }
    wa = MagicMock()
    wa.send_text = AsyncMock()
    return config, service, wa


@pytest.mark.asyncio
async def test_send_reminders_skips_contact_switched_off():
    config, service, wa = _switch_off_setup()
    should_send = AsyncMock(return_value=False)

    count = await send_reminders(config, service, wa, should_send=should_send)

    assert count == 0
    should_send.assert_awaited_once_with("whatsapp", "5491112345678")
    wa.send_text.assert_not_called()
    service.events().patch.assert_not_called()


@pytest.mark.asyncio
async def test_send_reminders_sends_when_switch_on():
    config, service, wa = _switch_off_setup()

    count = await send_reminders(config, service, wa, should_send=AsyncMock(return_value=True))

    assert count == 1
    wa.send_text.assert_called_once()
