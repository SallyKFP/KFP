import asyncio
import logging
from datetime import date, datetime, time, timedelta

import pytest
import pytz
from unittest.mock import AsyncMock, MagicMock

import core.telegram as telegram
from core.telegram import (
    TelegramConflictError,
    _check_telegram_availability,
    _configured_consultants,
    _create_telegram_booking,
    _handle_telegram_attendance_confirmation,
    _handle_update,
    _telegram_request,
    parse_allowed_user_ids,
)
from modules.booking.consultants import allocate_consultant, consultant_slot_is_reserved


@pytest.fixture(autouse=True)
def isolate_calendar_search(monkeypatch):
    monkeypatch.setattr("core.telegram._find_telegram_events", lambda _config, _chat_id: [])


@pytest.fixture(autouse=True)
def isolate_dashboard_storage(monkeypatch):
    monkeypatch.setattr("core.telegram._dashboard_store", None)


@pytest.mark.asyncio
async def test_telegram_registers_contact_without_storing_message(mocker, monkeypatch):
    store = mocker.MagicMock()
    store.register_contact.return_value = True
    monkeypatch.setattr(telegram, "_dashboard_store", store)
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)
    from core.history import InMemoryHistory

    await _handle_update(
        {
            "message": {
                "message_id": 67,
                "from": {"id": 12345, "first_name": "Jamie", "last_name": "Tan", "username": "jamietest"},
                "chat": {"id": 12345, "type": "private"},
                "text": "/start",
            }
        },
        client=None,
        token="",
        allowed_user_ids={12345},
        history=InMemoryHistory(),
        config={},
        knowledge="",
    )

    store.register_contact.assert_called_once_with("telegram", "12345", "Jamie Tan (@jamietest)")
    assert not store.add_message.called
    reply.assert_awaited_once()


@pytest.mark.asyncio
async def test_telegram_stays_silent_when_switched_off(mocker, monkeypatch):
    store = mocker.MagicMock()
    store.register_contact.return_value = False
    monkeypatch.setattr(telegram, "_dashboard_store", store)
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)
    answer = MagicMock()
    monkeypatch.setattr("core.telegram.get_ai_response", answer)
    from core.history import InMemoryHistory

    await _handle_update(
        {
            "message": {
                "from": {"id": 12345},
                "chat": {"id": 12345, "type": "private"},
                "text": "Hello",
            }
        },
        client=None,
        token="",
        allowed_user_ids={12345},
        history=InMemoryHistory(),
        config={},
        knowledge="",
    )

    reply.assert_not_awaited()
    answer.assert_not_called()


@pytest.mark.asyncio
async def test_telegram_replies_when_switch_lookup_fails(mocker, monkeypatch):
    store = mocker.MagicMock()
    store.register_contact.side_effect = RuntimeError("db down")
    monkeypatch.setattr(telegram, "_dashboard_store", store)

    assert await telegram._sally_should_reply(12345) is True


@pytest.mark.asyncio
async def test_telegram_dashboard_storage_initializes_from_database_url(mocker, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    store = mocker.MagicMock()
    monkeypatch.setattr("core.telegram.DashboardStore", mocker.Mock(return_value=store))

    async with telegram._telegram_dashboard_storage():
        assert telegram._dashboard_store is store

    store.initialize.assert_called_once_with()
    assert telegram._dashboard_store is None


@pytest.mark.asyncio
async def test_telegram_scheduler_runs_without_whatsapp_credentials(monkeypatch):
    import asyncio

    for name in (
        "WHATSAPP_PHONE_NUMBER_ID", "WHATSAPP_ACCESS_TOKEN",
        "WHATSAPP_APP_SECRET", "WHATSAPP_VERIFY_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    config = {"modules": {"reminders": True}, "client": {"timezone": "Asia/Singapore"}}
    sent = AsyncMock(return_value=1)
    monkeypatch.setattr("core.telegram.run_telegram_reminders", sent)
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def daily(timezone, channel, send):
        assert timezone == "Asia/Singapore"
        assert channel == "Telegram"
        await send()
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr("core.telegram.run_daily_reminders", daily)
    async with telegram._telegram_reminder_schedule("test-token", config):
        await asyncio.wait_for(started.wait(), timeout=1)
        sent.assert_awaited_once_with("test-token", config)
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_telegram_scheduler_not_started_when_reminders_disabled(monkeypatch):
    daily = MagicMock()
    monkeypatch.setattr("core.telegram.run_daily_reminders", daily)
    async with telegram._telegram_reminder_schedule("test-token", {"modules": {"reminders": False}}):
        pass
    daily.assert_not_called()


@pytest.mark.asyncio
async def test_running_telegram_bot_owns_scheduler_and_cleans_up_on_polling_exit(monkeypatch):
    import asyncio

    config = {"modules": {"reminders": True}, "client": {"timezone": "Asia/Singapore"}}
    monkeypatch.setattr("core.telegram.load_config", lambda: config)
    monkeypatch.setattr("core.telegram.load_knowledge", lambda: "")
    monkeypatch.setattr("core.telegram.get_history", MagicMock())
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def daily(_timezone, _channel, _send):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def request(_client, _token, method, **_kwargs):
        if method == "getMe":
            await asyncio.wait_for(started.wait(), timeout=1)
            return {"username": "test"}
        raise TelegramConflictError("polling conflict")

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr("core.telegram.run_daily_reminders", daily)
    monkeypatch.setattr("core.telegram._telegram_request", request)
    monkeypatch.setattr("core.telegram.asyncio.sleep", no_sleep)
    with pytest.raises(RuntimeError, match="did not clear"):
        await telegram.run_bot("test-token", {123})
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_run_bot_retries_after_polling_conflict_then_polls(monkeypatch):
    calls = {"getUpdates": 0}

    async def daily(_timezone, _channel, _send):
        await asyncio.Event().wait()

    async def request(_client, _token, method, **_kwargs):
        if method == "getMe":
            return {"username": "test"}
        calls["getUpdates"] += 1
        if calls["getUpdates"] == 1:
            raise TelegramConflictError("polling conflict")
        raise asyncio.CancelledError

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr("core.telegram.run_daily_reminders", daily)
    monkeypatch.setattr("core.telegram._telegram_request", request)
    monkeypatch.setattr("core.telegram.asyncio.sleep", no_sleep)
    with pytest.raises(asyncio.CancelledError):
        await telegram.run_bot("test-token", {123})
    assert calls["getUpdates"] == 2


@pytest.fixture(autouse=True)
def isolated_pending_bookings(monkeypatch):
    from core.pending_booking import PendingBookings
    store = PendingBookings(clock=lambda: 0.0)
    monkeypatch.setattr("core.telegram.PENDING_BOOKINGS", store)
    return store


def booking_details(**overrides):
    return {
        "service": "Initial evaluation", "date": "2026-11-23", "time": "10:00",
        "child_name": "Child Example", "child_dob": "2020-05-03",
        "parent_name": "Parent Example", "referral_name": "None",
        "patient_resides_in_singapore": "yes", "location": "Main Office", **overrides,
    }


@pytest.mark.parametrize(
    ("consultant", "appointment_date", "start_time", "reserved"),
    [
        ("Consultant Sarah", date(2026, 11, 9), time(16), True),
        ("Sarah", date(2026, 11, 15), time(17), True),
        ("Consultant Sarah", date(2026, 11, 12), time(12), True),
        ("Consultant Sarah", date(2026, 11, 13), time(10), True),
        ("Consultant Serena", date(2026, 11, 11), time(8), True),
        ("Consultant Sarah", date(2026, 11, 9), time(15), False),
        ("Consultant Sarah", date(2026, 11, 12), time(13), False),
        ("Consultant Serena", date(2026, 11, 10), time(16), False),
        ("Consultant One", date(2026, 11, 11), time(10), False),
        (None, date(2026, 11, 11), time(10), False),
    ],
)
def test_consultant_slot_reservations(
    consultant,
    appointment_date,
    start_time,
    reserved,
):
    assert consultant_slot_is_reserved(
        consultant,
        appointment_date,
        start_time,
    ) is reserved


def test_configured_consultants_resolve_calendar_ids_from_environment(monkeypatch):
    monkeypatch.setenv("CONSULTANT_1_CALENDAR_ID", "consultant-one@calendar")

    consultants = _configured_consultants({
        "booking": {
            "consultants": [{
                "name": "Consultant One",
                "calendar_id_env": "CONSULTANT_1_CALENDAR_ID",
            }],
        },
    })

    assert consultants[0]["calendar_id"] == "consultant-one@calendar"


@pytest.mark.asyncio
async def test_telegram_cancellation_deletes_event_from_its_calendar(monkeypatch):
    chat_id = 12345
    event = {
        "id": "event-2",
        "calendar_id": "two@calendar",
        "summary": "Initial evaluation - Patient: Child",
        "date": "Monday November 9, 2026",
        "time": "10:00",
    }
    telegram._pending_cancellations[chat_id] = [event]
    calendar = MagicMock()
    monkeypatch.setattr("core.telegram._calendar_client_for_event", lambda _config, _event: calendar)
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)

    response = await telegram._handle_telegram_cancellation_confirmed(
        None,
        "",
        chat_id,
        {"booking": {}},
        {"event_index": 1},
    )

    calendar.delete_event.assert_called_once_with("event-2")
    assert chat_id not in telegram._pending_cancellations
    assert "Appointment cancelled" in response
    reply.assert_awaited_once()


@pytest.mark.asyncio
async def test_telegram_modification_preserves_consultant_assignment(monkeypatch):
    chat_id = 12345
    event = {
        "id": "event-2",
        "calendar_id": "two@calendar",
        "consultant": "Consultant Two",
        "summary": "Initial evaluation - Patient: Child",
        "date": "Monday November 9, 2026",
        "time": "10:00",
        "start_datetime": (
            datetime.now(pytz.timezone("Asia/Singapore")) + timedelta(days=3)
        ).isoformat(),
    }
    monkeypatch.setattr("core.telegram._find_telegram_events", lambda _config, _chat_id: [event])
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)

    await telegram._handle_telegram_modification_request(
        None,
        "",
        chat_id,
        {"client": {"timezone": "Asia/Singapore"}, "booking": {}},
        "I can help with that.",
    )

    assert telegram._pending_modifications[chat_id] is event
    assert "with Consultant Two" in reply.await_args.args[3]
    telegram._pending_modifications.pop(chat_id, None)


@pytest.mark.asyncio
async def test_telegram_modification_within_36_hours_requires_mc(monkeypatch):
    chat_id = 12345
    event = {
        "id": "event-3",
        "summary": "Initial evaluation - Patient: Child",
        "date": "Wednesday November 25, 2026",
        "time": "10:00",
        "start_datetime": (
            datetime.now(pytz.timezone("Asia/Singapore")) + timedelta(hours=12)
        ).isoformat(),
    }
    monkeypatch.setattr("core.telegram._find_telegram_events", lambda _config, _chat_id: [event])
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)

    response = await telegram._handle_telegram_modification_request(
        None,
        "",
        chat_id,
        {"client": {"timezone": "Asia/Singapore"}, "booking": {}},
        "I can help with that.",
    )

    assert telegram._pending_modifications[chat_id]["awaiting_mc_confirmation"] is True
    assert "valid medical certificate (MC)" in response
    assert "reply YES or NO" in response
    reply.assert_awaited_once()
    telegram._pending_modifications.pop(chat_id, None)


@pytest.mark.asyncio
async def test_telegram_no_mc_starts_cancel_and_rebook_flow(monkeypatch):
    chat_id = 12345
    telegram._pending_modifications[chat_id] = {
        "id": "event-3",
        "awaiting_mc_confirmation": True,
    }
    cancellation = AsyncMock(return_value="Do you confirm you want to cancel it?")
    monkeypatch.setattr(
        "core.telegram._handle_telegram_cancellation_request",
        cancellation,
    )
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)

    await telegram._handle_telegram_rescheduling_mc_response(
        None,
        "",
        chat_id,
        {"booking": {}},
        {"has_valid_mc": False},
    )

    assert chat_id not in telegram._pending_modifications
    assert "new booking from scratch" in cancellation.await_args.args[4]


def test_parse_allowed_user_ids():
    assert parse_allowed_user_ids("123, 456") == {123, 456}


def test_parse_allowed_user_ids_requires_numeric_id():
    with pytest.raises(ValueError, match="numeric IDs"):
        parse_allowed_user_ids("not-an-id")


def test_parse_allowed_user_ids_requires_at_least_one_id():
    with pytest.raises(ValueError, match="Set TELEGRAM_ALLOWED_USER_IDS"):
        parse_allowed_user_ids("")


@pytest.mark.asyncio
async def test_telegram_yes_confirms_appointment_from_recent_reminder(monkeypatch):
    event = {
        "id": "appointment-1",
        "calendar_id": "serena@calendar",
        "patient_name": "Jane Example",
        "date": "Tuesday November 10, 2026",
        "time": "09:00",
        "start_datetime": "2026-11-10T09:00:00+08:00",
        "reminder_sent_for": "2026-11-10",
        "attendance_confirmed_for": None,
    }
    monkeypatch.setattr(
        "core.telegram._pending_telegram_confirmation_events",
        lambda _config, _chat_id: [event],
    )
    calendar = MagicMock()
    monkeypatch.setattr("core.telegram._calendar_client_for_event", lambda *_args: calendar)
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)

    response = await _handle_telegram_attendance_confirmation(
        None,
        "token",
        123,
        {},
        "YES",
    )

    assert response and "Attendance is confirmed" in response
    calendar.mark_event_attendance_confirmed.assert_called_once_with("appointment-1")
    reply.assert_awaited_once()


@pytest.mark.asyncio
async def test_telegram_request_reports_polling_conflict_without_retrying_response():
    response = MagicMock()
    response.status_code = 409
    response.json.return_value = {
        "ok": False,
        "description": "Conflict: terminated by other getUpdates request",
    }
    client = MagicMock()
    client.post = AsyncMock(return_value=response)

    with pytest.raises(
        TelegramConflictError,
        match="terminated by other getUpdates request",
    ):
        await _telegram_request(client, "test-token", "getUpdates")

    response.raise_for_status.assert_not_called()


@pytest.mark.asyncio
async def test_private_message_from_unapproved_user_logs_allowlist_reason(caplog):
    update = {
        "message": {
            "from": {"id": 789},
            "chat": {"id": 789, "type": "private"},
            "text": "Hello",
        }
    }

    with caplog.at_level(logging.WARNING):
        await _handle_update(
            update,
            client=None,
            token="",
            allowed_user_ids={123, 456},
            history=None,
            config={},
            knowledge="",
        )

    assert "not in the configured allowlist (2 IDs loaded)" in caplog.text


@pytest.mark.asyncio
async def test_message_from_non_private_chat_logs_chat_reason(caplog):
    update = {
        "message": {
            "from": {"id": 123},
            "chat": {"id": -123, "type": "group"},
            "text": "Hello",
        }
    }

    with caplog.at_level(logging.WARNING):
        await _handle_update(
            update,
            client=None,
            token="",
            allowed_user_ids={123},
            history=None,
            config={},
            knowledge="",
        )

    assert "non-private chat" in caplog.text


@pytest.mark.asyncio
async def test_book_command_requires_confirm_before_creating_booking(monkeypatch):
    create_booking = MagicMock(return_value="Appointment confirmed.")
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._create_telegram_booking", create_booking)
    monkeypatch.setattr("core.telegram._reply_to_message", reply)
    update = {
        "message": {
            "from": {"id": 123},
            "chat": {"id": 123, "type": "private"},
        "text": (
            "/book Consultation | 2026-11-06 | 10:00 | Child Example | "
            "2020-05-03 | Parent Example | None | yes"
        ),
        }
    }

    from core.history import InMemoryHistory
    history = InMemoryHistory()
    config = {"booking": {"locations": [{"name": "Main Office"}]}}
    await _handle_update(
        update,
        client=None,
        token="",
        allowed_user_ids={123},
        history=history,
        config=config,
        knowledge="",
    )

    create_booking.assert_not_called()
    assert "reply CONFIRM only" in reply.await_args.args[3]
    update["message"]["text"] = "CONFIRM"
    await _handle_update(
        update,
        client=None,
        token="",
        allowed_user_ids={123},
        history=history,
        config=config,
        knowledge="",
    )
    create_booking.assert_called_once_with(
        config,
        123,
        [
            "Consultation", "2026-11-06", "10:00", "Child Example",
            "2020-05-03", "Parent Example", "None", "yes", "Main Office",
        ],
    )
    assert reply.await_count == 2


@pytest.mark.asyncio
async def test_other_command_preserves_staged_booking(monkeypatch):
    from core.history import InMemoryHistory

    history = InMemoryHistory()
    create_booking = MagicMock()
    monkeypatch.setattr("core.telegram._create_telegram_booking", create_booking)
    monkeypatch.setattr("core.telegram._reply_to_message", AsyncMock())
    monkeypatch.setattr(
        "core.telegram.get_ai_response",
        MagicMock(return_value='{"intent": "booking_confirmed"}'),
    )
    update = {"message": {
        "from": {"id": 123}, "chat": {"id": 123, "type": "private"},
    }}
    async def send(text):
        update["message"]["text"] = text
        await _handle_update(
            update, client=None, token="", allowed_user_ids={123}, history=history,
            config={"booking": {"locations": [{"name": "Main Office"}]}}, knowledge="",
        )

    await send(
        "/book Consultation | 2026-11-06 | 10:00 | Child Example | "
        "2020-05-03 | Parent Example | None | yes"
    )
    await send("/next-appointment")
    await send("CONFIRM")
    create_booking.assert_called_once()


@pytest.mark.asyncio
async def test_next_appointment_command_uses_chat_id_and_both_names(monkeypatch):
    lookup = MagicMock(return_value={
        "service": "Initial evaluation",
        "date": "Monday, November 9, 2026",
        "time": "10:00",
        "consultant": "Consultant One",
        "location": "Main clinic",
    })
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._find_next_telegram_appointment", lookup)
    monkeypatch.setattr("core.telegram._reply_to_message", reply)
    update = {
        "message": {
            "from": {"id": 123},
            "chat": {"id": 123, "type": "private"},
            "text": "/next-appointment Child Example | Parent Example",
        }
    }

    await _handle_update(
        update,
        client=None,
        token="",
        allowed_user_ids={123},
        history=MagicMock(),
        config={"modules": {"booking": True}},
        knowledge="",
    )

    lookup.assert_called_once_with(
        {"modules": {"booking": True}},
        123,
        "Child Example",
        "Parent Example",
    )
    reply.assert_awaited_once_with(
        None,
        "",
        123,
        "The next appointment is Initial evaluation on Monday, November 9, 2026 "
        "at 10:00 with Consultant One at Main clinic.",
    )


@pytest.mark.asyncio
async def test_next_appointment_command_requires_both_names(monkeypatch):
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)
    update = {
        "message": {
            "from": {"id": 123},
            "chat": {"id": 123, "type": "private"},
            "text": "/next-appointment Child Example",
        }
    }

    await _handle_update(
        update,
        client=None,
        token="",
        allowed_user_ids={123},
        history=MagicMock(),
        config={},
        knowledge="",
    )

    reply.assert_awaited_once()
    assert "Patient name | Parent name" in reply.call_args.args[3]


@pytest.mark.asyncio
async def test_natural_language_appointment_lookup_uses_intent_and_chat_id(monkeypatch):
    history = MagicMock()
    history.get.return_value = []
    lookup = MagicMock(return_value={
        "service": "Initial evaluation",
        "date": "Monday, November 9, 2026",
        "time": "10:00",
        "consultant": "Consultant One",
        "location": "Main clinic",
        "start_datetime": "2026-11-09T10:00:00+08:00",
    })
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram.get_ai_response", lambda *_args, **_kwargs: (
        'I am checking. {"intent": "appointment_lookup", '
        '"patient_name": "Child Example", "parent_name": "Parent Example"}'
    ))
    monkeypatch.setattr("core.telegram._find_next_telegram_appointment", lookup)
    monkeypatch.setattr("core.telegram._reply_to_message", reply)
    update = {
        "message": {
            "from": {"id": 123},
            "chat": {"id": 123, "type": "private"},
            "text": "When is my child's next appointment? Child Example, Parent Example",
        }
    }

    await _handle_update(
        update,
        client=None,
        token="",
        allowed_user_ids={123},
        history=history,
        config={"modules": {"booking": True}},
        knowledge="",
    )

    lookup.assert_called_once_with(
        {"modules": {"booking": True}},
        123,
        "Child Example",
        "Parent Example",
    )
    reply.assert_awaited_once_with(
        None,
        "",
        123,
        "The next appointment is Initial evaluation on Monday, November 9, 2026 "
        "at 10:00 with Consultant One at Main clinic.",
    )


@pytest.mark.asyncio
async def test_telegram_availability_range_intent_returns_calendar_results(monkeypatch):
    history = MagicMock()
    history.get.return_value = []
    intent = {
        "intent": "availability_range_check",
        "service": "Initial evaluation",
        "start_date": "2026-11-23",
        "end_date": "2026-11-29",
    }
    monkeypatch.setattr(
        "core.telegram.get_ai_response",
        lambda *_args, **_kwargs: (
            'I am checking. {"intent": "availability_range_check", '
            '"service": "Initial evaluation", "start_date": "2026-11-23", '
            '"end_date": "2026-11-29"}'
        ),
    )
    range_check = MagicMock(return_value="These times are currently available.")
    monkeypatch.setattr("core.telegram.check_availability_range", range_check)
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)
    update = {
        "message": {
            "from": {"id": 123},
            "chat": {"id": 123, "type": "private"},
            "text": "Any availability the week of November 23?",
        }
    }

    await _handle_update(
        update,
        client=None,
        token="",
        allowed_user_ids={123},
        history=history,
        config={"modules": {"booking": True}},
        knowledge="",
    )

    range_check.assert_called_once()
    assert range_check.call_args.args[:2] == (
        {"modules": {"booking": True}},
        intent,
    )
    reply.assert_awaited_once_with(
        None,
        "",
        123,
        "These times are currently available.",
    )


def test_next_telegram_appointment_selects_earliest_across_consultant_calendars(
    monkeypatch,
):
    class Calendar:
        def __init__(self, appointment):
            self.appointment = appointment

        def find_next_event_by_telegram_chat_and_names(
            self,
            chat_id,
            patient_name,
            parent_name,
        ):
            assert (chat_id, patient_name, parent_name) == (
                123,
                "Child Example",
                "Parent Example",
            )
            return self.appointment

    appointments = {
        "one@calendar": {
            "service": "Initial evaluation",
            "start_datetime": "2026-11-12T10:00:00+08:00",
            "consultant": "Consultant One",
        },
        "two@calendar": {
            "service": "Follow-up intervention",
            "start_datetime": "2026-11-10T10:00:00+08:00",
            "consultant": "Consultant Two",
        },
    }
    monkeypatch.setattr(
        "core.telegram._calendar_client",
        lambda _config, name: Calendar(
            appointments[
                "one@calendar" if name == "Consultant One" else "two@calendar"
            ]
        ),
    )
    config = {
        "modules": {"booking": True},
        "booking": {
            "consultants": [
                {"name": "Consultant One", "calendar_id": "one@calendar"},
                {"name": "Consultant Two", "calendar_id": "two@calendar"},
            ],
        },
    }

    result = telegram._find_next_telegram_appointment(
        config,
        123,
        "Child Example",
        "Parent Example",
    )

    assert result["consultant"] == "Consultant Two"
    assert result["start_datetime"] == "2026-11-10T10:00:00+08:00"


@pytest.mark.asyncio
async def test_natural_language_booking_intent_creates_calendar_event(monkeypatch, isolated_pending_bookings):
    timezone = pytz.timezone("Asia/Singapore")
    booking_date = (datetime.now(timezone) + timedelta(days=1)).date()
    while booking_date.weekday() >= 5:
        booking_date += timedelta(days=1)
    calendar = MagicMock()
    calendar.is_slot_available.return_value = True
    calendar.has_upcoming_appointment_for_patient.return_value = False
    monkeypatch.setattr("core.telegram._calendar_client", lambda _config: calendar)
    monkeypatch.setattr(
        "core.telegram.get_ai_response",
        lambda *_args, **_kwargs: (
            "Checking that time now.\n"
            '{"intent": "booking_confirmed", "service": "Initial evaluation", '
            f'"date": "{booking_date.isoformat()}", "time": "10:00", '
            '"location": "Main Office", "child_name": "Child Example", '
            '"child_dob": "2020-05-03", "parent_name": "Parent Example", '
            '"referral_name": "None", "patient_resides_in_singapore": "yes"}'
        ),
    )
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)
    config = {
        "client": {"timezone": "Asia/Singapore"},
        "modules": {"booking": True},
        "booking": {
            "business_hours": {"start": "08:00", "end": "18:00"},
            "services": [{"name": "Initial evaluation", "duration_minutes": 60, "price": 300}],
            "locations": [{
                "name": "Main Office",
                "address": "123 Main Street",
                "days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
            }],
        },
    }

    history = MagicMock()
    isolated_pending_bookings.put("telegram:123", booking_details(date=booking_date.isoformat()))
    isolated_pending_bookings.set_observations("telegram:123", "Enjoys playing")
    isolated_pending_bookings.approve_observations("telegram:123")
    history.get.return_value = [{
        "role": "assistant",
        "content": (
            "Please check these booking details and reply CONFIRM:\n"
            "- Service: Initial evaluation\n- Date: 2026-11-23\n- Time: 10:00\n"
            "- Patient: Child Example\n- Patient DOB: 2020-05-03\n- Parent: Parent Example\n"
            "- Referral: None\n- Patient resides in Singapore: Yes"
        ),
    }]
    await _handle_update(
        {
            "message": {
                "from": {"id": 123},
                "chat": {"id": 123, "type": "private"},
                "text": "CONFIRM",
            }
        },
        client=None,
        token="",
        allowed_user_ids={123},
        history=history,
        config=config,
        knowledge="",
    )

    calendar.create_event.assert_called_once()
    assert calendar.create_event.call_args.kwargs["observations"] == "Enjoys playing"
    assert "Appointment confirmed" in reply.await_args.args[3]


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_text", ["YES", "yes please", "CONFIRM!", "correct"])
async def test_booking_intent_cannot_create_event_without_exact_confirm(
    monkeypatch, reply_text, isolated_pending_bookings,
):
    from core.history import InMemoryHistory

    history = InMemoryHistory()
    isolated_pending_bookings.put("telegram:123", booking_details())
    history.add(
        "123",
        "assistant",
        "Please check these booking details and reply CONFIRM:\n"
        "- Service: Initial evaluation\n- Date: 2026-11-23\n- Time: 10:00\n"
        "- Patient: Child Example\n- Patient DOB: 2020-05-03\n- Parent: Parent Example\n"
        "- Referral: None\n- Patient resides in Singapore: Yes",
    )
    attendance = AsyncMock(return_value=None)
    monkeypatch.setattr("core.telegram._handle_telegram_attendance_confirmation", attendance)
    monkeypatch.setattr(
        "core.telegram.get_ai_response",
        lambda *_args, **_kwargs: '{"intent": "booking_confirmed"}',
    )
    booking = AsyncMock()
    monkeypatch.setattr("core.telegram._handle_book_command", booking)
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)
    await _handle_update(
        {"message": {
            "from": {"id": 123},
            "chat": {"id": 123, "type": "private"},
            "text": reply_text,
        }},
        client=None,
        token="",
        allowed_user_ids={123},
        history=history,
        config={"modules": {"booking": True}},
        knowledge="",
    )

    booking.assert_not_awaited()
    if reply_text == "YES":
        attendance.assert_not_awaited()
        assert "reply CONFIRM only" in reply.await_args.args[3]


@pytest.mark.asyncio
async def test_booking_summary_yes_then_confirm_routes_to_booking_not_reminder(monkeypatch):
    from core.history import InMemoryHistory

    history = InMemoryHistory()
    answer = MagicMock(side_effect=[
        '{"intent":"booking_draft","service":"Initial evaluation",'
        '"date":"2026-11-23","time":"10:00","child_name":"Child Example",'
        '"child_dob":"2020-05-03","parent_name":"Parent Example",'
        '"referral_name":"None","patient_resides_in_singapore":"yes","location":"Main Office"}',
    ])
    monkeypatch.setattr("core.telegram.get_ai_response", answer)
    reminder = AsyncMock(return_value=None)
    monkeypatch.setattr("core.telegram._handle_telegram_attendance_confirmation", reminder)
    booking = AsyncMock(return_value="Appointment confirmed.")
    monkeypatch.setattr("core.telegram._handle_book_command", booking)
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)
    update = {
        "message": {
            "from": {"id": 123},
            "chat": {"id": 123, "type": "private"},
            "text": "My booking details",
        }
    }

    async def send(text):
        update["message"]["text"] = text
        await _handle_update(
            update,
            client=None,
            token="",
            allowed_user_ids={123},
            history=history,
            config={"modules": {"booking": True}},
            knowledge="",
        )

    await send("My booking details")
    assert "Reply YES to confirm" not in reply.await_args.args[3]
    assert "reply CONFIRM only" in reply.await_args.args[3]
    reminder.reset_mock()
    await send("YES")
    reminder.assert_not_awaited()
    booking.assert_not_awaited()
    await send("CONFIRM")
    booking.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(("dob", "accepted"), [("2026-10-10", True), ("2026-11-25", False)])
async def test_telegram_booking_draft_validates_dob_against_clinic_date(
    monkeypatch, isolated_pending_bookings, dob, accepted,
):
    import json
    from core.history import InMemoryHistory

    def clinic_today(timezone):
        assert timezone == "Asia/Singapore"
        return date(2026, 11, 24)

    monkeypatch.setattr("core.patient_dates.clinic_today", clinic_today)
    monkeypatch.setattr("core.telegram.get_ai_response", MagicMock(return_value=json.dumps({
        "intent": "booking_draft", **booking_details(child_dob=dob),
    })))
    monkeypatch.setattr("core.telegram._handle_telegram_attendance_confirmation", AsyncMock(return_value=None))
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)
    await _handle_update(
        {"message": {"from": {"id": 123}, "chat": {"id": 123, "type": "private"}, "text": "Here are my booking details"}},
        client=None, token="", allowed_user_ids={123}, history=InMemoryHistory(),
        config={"client": {"timezone": "Asia/Singapore"}, "modules": {"booking": True}, "booking": {}},
        knowledge="",
    )
    pending = isolated_pending_bookings.get("telegram:123")
    assert (pending is not None) == accepted
    if accepted:
        assert f"Patient DOB: {dob}" in reply.await_args.args[3]
    else:
        assert "date of birth is in the future" in reply.await_args.args[3]


@pytest.mark.asyncio
async def test_telegram_proposes_initial_general_observation(monkeypatch, isolated_pending_bookings):
    import json
    from core.history import InMemoryHistory

    history = InMemoryHistory()
    history.add("123", "user", "My son likes books and is not yet crawling.")
    monkeypatch.setattr("core.telegram.get_ai_response", MagicMock(return_value=json.dumps({
        "intent": "booking_draft", **booking_details(),
    })))
    monkeypatch.setattr("core.telegram._handle_telegram_attendance_confirmation", AsyncMock(return_value=None))
    booking = AsyncMock(return_value="Appointment confirmed.")
    monkeypatch.setattr("core.telegram._handle_book_command", booking)
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)

    async def send(text):
        await _handle_update(
            {"message": {"from": {"id": 123}, "chat": {"id": 123, "type": "private"}, "text": text}},
            client=None, token="", allowed_user_ids={123}, history=history,
            config={"modules": {"booking": True}, "booking": {}}, knowledge="",
        )

    await send("Here are the remaining booking details")
    summary = reply.await_args.args[3]
    assert "Proposed note for consultant (not yet approved): My son likes books" in summary
    assert "not yet crawling" not in summary
    await send("CONFIRM")
    booking.assert_not_awaited()
    await send("APPROVE NOTE")
    await send("CONFIRM")
    assert booking.await_args.kwargs["observations"] == "My son likes books"


@pytest.mark.asyncio
async def test_telegram_note_is_reviewed_before_confirmation(monkeypatch, isolated_pending_bookings):
    from core.history import InMemoryHistory

    isolated_pending_bookings.put("telegram:123", booking_details())
    ai = MagicMock()
    monkeypatch.setattr("core.telegram.get_ai_response", ai)
    booking = AsyncMock(return_value="Appointment confirmed.")
    monkeypatch.setattr("core.telegram._handle_book_command", booking)
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)
    history = InMemoryHistory()

    async def send(text):
        await _handle_update(
            {"message": {"from": {"id": 123}, "chat": {"id": 123, "type": "private"}, "text": text}},
            client=None, token="", allowed_user_ids={123}, history=history,
            config={"modules": {"booking": True}}, knowledge="",
        )

    await send("NOTE: Enjoys playing with blocks")
    assert "- Proposed note for consultant (not yet approved): Enjoys playing with blocks" in reply.await_args.args[3]
    booking.assert_not_awaited()
    ai.assert_not_called()
    await send("REMOVE NOTE")
    assert "- Observations for consultant:" not in reply.await_args.args[3]
    await send("NOTE: Enjoys playing with blocks")
    await send("CONFIRM")
    booking.assert_not_awaited()
    await send("APPROVE NOTE")
    booking.assert_not_awaited()
    await send("CONFIRM")
    assert booking.await_args.kwargs["observations"] == "Enjoys playing with blocks"


@pytest.mark.asyncio
async def test_telegram_question_does_not_replace_saved_booking(monkeypatch, isolated_pending_bookings):
    from core.history import InMemoryHistory

    history = InMemoryHistory()
    isolated_pending_bookings.put("telegram:123", booking_details(time="10:00"))
    ai = MagicMock(return_value="The clinic is on Main Street.")
    monkeypatch.setattr("core.telegram.get_ai_response", ai)
    booking = AsyncMock(return_value="Appointment confirmed.")
    monkeypatch.setattr("core.telegram._handle_book_command", booking)
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)
    update = {"message": {"from": {"id": 123}, "chat": {"id": 123, "type": "private"}}}

    async def send(text):
        update["message"]["text"] = text
        await _handle_update(
            update, client=None, token="", allowed_user_ids={123}, history=history,
            config={"modules": {"booking": True}}, knowledge="",
        )

    await send("Where is the clinic?")
    assert "Main Street" in reply.await_args.args[3]
    await send("CONFIRM")
    assert booking.await_args.args[4][2] == "10:00"
    ai.assert_called_once()
    await send("CONFIRM")
    booking.assert_awaited_once()
    assert "no active booking summary" in reply.await_args.args[3]


@pytest.mark.asyncio
async def test_telegram_change_requires_new_summary_and_confirmation(monkeypatch, isolated_pending_bookings):
    import json
    from core.history import InMemoryHistory

    history = InMemoryHistory()
    isolated_pending_bookings.put("telegram:123", booking_details())
    monkeypatch.setattr("core.telegram.get_ai_response", MagicMock(return_value=json.dumps({
        "intent": "booking_draft", **booking_details(time="12:00"),
    })))
    booking = AsyncMock(return_value="Appointment confirmed.")
    monkeypatch.setattr("core.telegram._handle_book_command", booking)
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)
    update = {"message": {"from": {"id": 123}, "chat": {"id": 123, "type": "private"}}}

    async def send(text):
        update["message"]["text"] = text
        await _handle_update(
            update, client=None, token="", allowed_user_ids={123}, history=history,
            config={"modules": {"booking": True}}, knowledge="",
        )

    await send("Could we change the time to 12:00?")
    assert "- Time: 12:00" in reply.await_args.args[3]
    booking.assert_not_awaited()
    await send("CONFIRM")
    assert booking.await_args.args[4][2] == "12:00"


@pytest.mark.asyncio
async def test_telegram_summary_includes_previously_assigned_consultant(monkeypatch, isolated_pending_bookings):
    import json
    from core.history import InMemoryHistory

    details = booking_details()
    assignment_key = telegram._consultant_assignment_key(123, details)
    telegram._pending_consultant_assignments[assignment_key] = "Consultant One"
    monkeypatch.setattr("core.telegram.get_ai_response", MagicMock(return_value=json.dumps({
        "intent": "booking_draft", **details,
    })))
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)
    await _handle_update(
        {"message": {"from": {"id": 123}, "chat": {"id": 123, "type": "private"},
                     "text": "Here are my details"}},
        client=None, token="", allowed_user_ids={123}, history=InMemoryHistory(),
        config={"modules": {"booking": True}}, knowledge="",
    )
    assert "- Consultant: Consultant One" in reply.await_args.args[3]
    assert isolated_pending_bookings.get("telegram:123").details["consultant"] == "Consultant One"
    assert assignment_key not in telegram._pending_consultant_assignments


@pytest.mark.asyncio
async def test_telegram_cancel_pending_does_not_call_ai_or_booking(monkeypatch, isolated_pending_bookings):
    from core.history import InMemoryHistory
    isolated_pending_bookings.put("telegram:123", booking_details())
    ai = MagicMock()
    monkeypatch.setattr("core.telegram.get_ai_response", ai)
    booking = AsyncMock()
    monkeypatch.setattr("core.telegram._handle_book_command", booking)
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)
    await _handle_update(
        {"message": {"from": {"id": 123}, "chat": {"id": 123, "type": "private"},
                     "text": "Cancel this booking"}},
        client=None, token="", allowed_user_ids={123}, history=InMemoryHistory(),
        config={"modules": {"booking": True}}, knowledge="",
    )
    assert isolated_pending_bookings.get("telegram:123") is None
    assert "no appointment was booked" in reply.await_args.args[3]
    booking.assert_not_awaited()
    ai.assert_not_called()


@pytest.mark.asyncio
async def test_yes_to_booking_information_question_does_not_confirm_reminder(monkeypatch):
    from core.history import InMemoryHistory

    history = InMemoryHistory()
    history.add("123", "assistant", "Does the patient reside in Singapore?")
    attendance = AsyncMock(return_value=None)
    monkeypatch.setattr("core.telegram._handle_telegram_attendance_confirmation", attendance)
    ai = MagicMock(return_value="Thank you.")
    monkeypatch.setattr("core.telegram.get_ai_response", ai)
    monkeypatch.setattr("core.telegram._reply_to_message", AsyncMock())
    await _handle_update(
        {"message": {"from": {"id": 123}, "chat": {"id": 123, "type": "private"}, "text": "YES"}},
        client=None,
        token="",
        allowed_user_ids={123},
        history=history,
        config={"modules": {"booking": True}},
        knowledge="",
    )
    attendance.assert_not_awaited()
    ai.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_text", ["YES!", "yes please", "CONFIRM"])
async def test_telegram_reminder_requires_exact_yes(monkeypatch, reply_text):
    pending = MagicMock(return_value=[{"id": "appointment-1"}])
    monkeypatch.setattr("core.telegram._pending_telegram_confirmation_events", pending)
    calendar = MagicMock()
    monkeypatch.setattr("core.telegram._calendar_client_for_event", calendar)
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)

    result = await _handle_telegram_attendance_confirmation(
        None, "token", 123, {"modules": {"booking": True}}, reply_text,
    )

    assert "reply YES only" in result
    assert "reply YES only" in reply.await_args.args[3]
    pending.assert_called_once()
    calendar.assert_not_called()


@pytest.mark.asyncio
async def test_telegram_reminder_question_gets_keyword_guidance_and_normal_answer(monkeypatch):
    from core.history import InMemoryHistory

    pending = MagicMock(return_value=[{"id": "appointment-1"}])
    monkeypatch.setattr("core.telegram._pending_telegram_confirmation_events", pending)
    monkeypatch.setattr("core.telegram.get_ai_response", MagicMock(return_value="The clinic is on Main Street."))
    calendar = MagicMock()
    monkeypatch.setattr("core.telegram._calendar_client_for_event", calendar)
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)
    await _handle_update(
        {"message": {"from": {"id": 123}, "chat": {"id": 123, "type": "private"},
                     "text": "Where is the clinic?"}},
        client=None, token="", allowed_user_ids={123}, history=InMemoryHistory(),
        config={"modules": {"booking": True}}, knowledge="",
    )
    assert reply.await_count == 2
    assert "reply YES only" in reply.await_args_list[0].args[3]
    assert "Main Street" in reply.await_args_list[1].args[3]
    calendar.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_text", ["okay", "CONFIRM!", "yes please"])
async def test_telegram_book_command_rejects_non_exact_then_accepts_confirm(monkeypatch, reply_text):
    from core.history import InMemoryHistory

    history = InMemoryHistory()
    create_booking = MagicMock(return_value="Appointment confirmed.")
    monkeypatch.setattr("core.telegram._create_telegram_booking", create_booking)
    attendance = AsyncMock()
    monkeypatch.setattr("core.telegram._handle_telegram_attendance_confirmation", attendance)
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)
    update = {"message": {"from": {"id": 123}, "chat": {"id": 123, "type": "private"}}}

    async def send(text):
        update["message"]["text"] = text
        await _handle_update(
            update, client=None, token="", allowed_user_ids={123}, history=history,
            config={"booking": {"locations": [{"name": "Main Office"}]}}, knowledge="",
        )

    await send(
        "/book Consultation | 2026-11-06 | 10:00 | Child Example | "
        "2020-05-03 | Parent Example | None | yes"
    )
    await send(reply_text)
    assert "reply CONFIRM only" in reply.await_args.args[3]
    create_booking.assert_not_called()
    attendance.assert_not_awaited()
    await send("CONFIRM")
    create_booking.assert_called_once()


@pytest.mark.asyncio
async def test_telegram_reminder_lookup_failure_is_reported(monkeypatch):
    monkeypatch.setattr(
        "core.telegram._pending_telegram_confirmation_events",
        MagicMock(side_effect=RuntimeError("unavailable")),
    )
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)
    result = await _handle_telegram_attendance_confirmation(None, "", 123, {}, "YES")
    assert "couldn't check" in result
    assert "hasn't been confirmed" in reply.await_args.args[3]


def test_telegram_availability_check_queries_calendar_and_reports_result(monkeypatch):
    timezone = pytz.timezone("Asia/Singapore")
    booking_date = (datetime.now(timezone) + timedelta(days=1)).date()
    while booking_date.weekday() >= 5:
        booking_date += timedelta(days=1)
    config = {
        "client": {"timezone": "Asia/Singapore"},
        "modules": {"booking": True},
        "booking": {
            "calendar_id": "test@calendar",
            "calendar_owner_email": "owner@example.com",
            "business_hours": {"start": "08:00", "end": "18:00"},
            "services": [{"name": "Initial evaluation", "duration_minutes": 60, "price": 300}],
            "locations": [{
                "name": "Main Office",
                "address": "123 Main Street",
                "days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
            }],
        },
    }
    calendar = MagicMock()
    calendar.is_slot_available.return_value = True
    monkeypatch.setattr("core.telegram._calendar_client", lambda _config: calendar)

    answer = _check_telegram_availability(
        config,
        {
            "service": "Initial evaluation",
            "date": booking_date.isoformat(),
            "time": "10:00",
            "location": "Main Office",
        },
    )

    assert "Google Calendar" not in answer
    assert "is available" in answer
    assert "isn't reserved yet" in answer
    calendar.is_slot_available.assert_called_once()
    calendar.create_event.assert_not_called()


def test_telegram_availability_allocates_available_consultant_without_preference(monkeypatch):
    timezone = pytz.timezone("Asia/Singapore")
    booking_date = (datetime.now(timezone) + timedelta(days=1)).date()
    while booking_date.weekday() >= 5:
        booking_date += timedelta(days=1)
    config = {
        "client": {"timezone": "Asia/Singapore"},
        "modules": {"booking": True},
        "booking": {
            "consultants": [
                {"name": "Consultant One", "calendar_id": "one@calendar"},
                {"name": "Consultant Two", "calendar_id": "two@calendar"},
            ],
            "business_hours": {"start": "08:00", "end": "18:00"},
            "services": [{"name": "Initial evaluation", "duration_minutes": 60, "price": 300}],
            "locations": [{
                "name": "Main Office",
                "address": "123 Main Street",
                "days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
            }],
        },
    }
    calendar = MagicMock()
    calendar.is_slot_available.return_value = True
    calendar_client = MagicMock(return_value=calendar)
    monkeypatch.setattr("core.telegram.CalendarClient", calendar_client)
    monkeypatch.setattr(
        "core.telegram.allocate_consultant",
        lambda available: available[1],
    )
    intent = {
        "service": "Initial evaluation",
        "date": booking_date.isoformat(),
        "time": "10:00",
        "location": "Main Office",
    }

    response = _check_telegram_availability(config, intent, chat_id=123)

    assert calendar_client.call_count == 3
    assert calendar_client.call_args.kwargs["calendar_id"] == "two@calendar"
    assert "with Consultant Two" in response
    assignment_key = (
        123,
        "Initial evaluation",
        booking_date.isoformat(),
        "10:00",
        "Main Office",
    )
    assert telegram._pending_consultant_assignments[assignment_key] == "Consultant Two"
    telegram._pending_consultant_assignments.pop(assignment_key)


def test_allocate_consultant_uses_serena_70_sarah_30_when_both_are_available(monkeypatch):
    available = [
        {"name": "Consultant Sarah", "calendar_id": "sarah@calendar"},
        {"name": "Consultant Serena", "calendar_id": "serena@calendar"},
    ]
    choices = MagicMock(return_value=[available[1]])
    monkeypatch.setattr("modules.booking.consultants.random.choices", choices)

    assigned = allocate_consultant(available)

    assert assigned is available[1]
    choices.assert_called_once_with(available, weights=[30, 70], k=1)


def test_telegram_availability_rejects_reserved_consultant_slot_before_calendar_check(
    monkeypatch,
):
    timezone = pytz.timezone("Asia/Singapore")
    today = datetime.now(timezone).date()
    days_until_thursday = (3 - today.weekday()) % 7 or 7
    booking_date = today + timedelta(days=days_until_thursday)
    calendar = MagicMock()
    monkeypatch.setattr("core.telegram._calendar_client", lambda *_args: calendar)
    config = {
        "client": {"timezone": "Asia/Singapore"},
        "modules": {"booking": True},
        "booking": {
            "consultants": [
                {"name": "Consultant Sarah", "calendar_id": "sarah@calendar"},
                {"name": "Consultant Serena", "calendar_id": "serena@calendar"},
            ],
            "business_hours": {"start": "08:00", "end": "18:00"},
            "services": [{"name": "Initial evaluation", "duration_minutes": 60, "price": 300}],
            "locations": [{
                "name": "Main Office",
                "address": "123 Main Street",
                "days": ["thursday"],
            }],
        },
    }

    answer = _check_telegram_availability(
        config,
        {
            "service": "Initial evaluation",
            "date": booking_date.isoformat(),
            "time": "12:00",
            "location": "Main Office",
            "consultant": "Sarah",
        },
    )

    assert answer == (
        "That time isn't available with Consultant Sarah. "
        "What other date and time would work?"
    )
    calendar.is_slot_available.assert_not_called()


def test_telegram_booking_rejects_reserved_consultant_slot_before_calendar_check(
    monkeypatch,
):
    timezone = pytz.timezone("Asia/Singapore")
    today = datetime.now(timezone).date()
    days_until_friday = (4 - today.weekday()) % 7 or 7
    booking_date = today + timedelta(days=days_until_friday)
    calendar = MagicMock()
    calendar_client = MagicMock(return_value=calendar)
    monkeypatch.setattr("core.telegram._calendar_client", calendar_client)
    config = {
        "client": {"timezone": "Asia/Singapore"},
        "modules": {"booking": True},
        "booking": {
            "consultants": [
                {"name": "Consultant Sarah", "calendar_id": "sarah@calendar"},
                {"name": "Consultant Serena", "calendar_id": "serena@calendar"},
            ],
            "business_hours": {"start": "08:00", "end": "18:00"},
            "services": [{"name": "Initial evaluation", "duration_minutes": 60, "price": 300}],
            "locations": [{
                "name": "Main Office",
                "address": "123 Main Street",
                "days": ["friday"],
            }],
        },
    }

    with pytest.raises(ValueError, match="isn't available with Consultant Sarah"):
        _create_telegram_booking(
            config,
            12345,
            [
                "Initial evaluation", booking_date.isoformat(), "10:00",
                "Child Example", "2020-05-03", "Parent Example", "None", "yes",
            ],
            "Sarah",
        )

    calendar_client.assert_not_called()
    calendar.is_slot_available.assert_not_called()
    calendar.create_event.assert_not_called()


@pytest.mark.asyncio
async def test_telegram_availability_intent_checks_calendar_before_reply(monkeypatch):
    timezone = pytz.timezone("Asia/Singapore")
    booking_date = (datetime.now(timezone) + timedelta(days=1)).date()
    while booking_date.weekday() >= 5:
        booking_date += timedelta(days=1)
    calendar = MagicMock()
    calendar.is_slot_available.return_value = False
    monkeypatch.setattr("core.telegram._calendar_client", lambda _config: calendar)
    monkeypatch.setattr(
        "core.telegram.get_ai_response",
        lambda *_args, **_kwargs: (
            "I'll check that time for you.\n"
            '{"intent": "availability_check", "service": "Initial evaluation", '
            f'"date": "{booking_date.isoformat()}", "time": "10:00", '
            '"location": "Main Office"}'
        ),
    )
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)
    history = MagicMock()
    config = {
        "client": {"timezone": "Asia/Singapore"},
        "modules": {"booking": True},
        "booking": {
            "calendar_id": "test@calendar",
            "calendar_owner_email": "owner@example.com",
            "business_hours": {"start": "08:00", "end": "18:00"},
            "services": [{"name": "Initial evaluation", "duration_minutes": 60, "price": 300}],
            "locations": [{
                "name": "Main Office",
                "address": "123 Main Street",
                "days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
            }],
        },
    }

    await _handle_update(
        {
            "message": {
                "from": {"id": 123},
                "chat": {"id": 123, "type": "private"},
                "text": "Would Tuesday at 10 work?",
            }
        },
        client=None,
        token="",
        allowed_user_ids={123},
        history=history,
        config=config,
        knowledge="",
    )

    response = reply.await_args.args[3]
    assert "already booked" in response
    calendar.is_slot_available.assert_called_once()
    calendar.create_event.assert_not_called()
    history.add.assert_any_call("123", "assistant", response)


@pytest.mark.asyncio
async def test_natural_language_confirmation_is_rejected_without_calendar_intent(monkeypatch):
    monkeypatch.setattr(
        "core.telegram.get_ai_response",
        lambda *_args, **_kwargs: "Your appointment is confirmed for Monday.",
    )
    reply = AsyncMock()
    monkeypatch.setattr("core.telegram._reply_to_message", reply)
    history = MagicMock()
    history.get.return_value = []

    await _handle_update(
        {
            "message": {
                "from": {"id": 123},
                "chat": {"id": 123, "type": "private"},
                "text": "Can I book a session?",
            }
        },
        client=None,
        token="",
        allowed_user_ids={123},
        history=history,
        config={},
        knowledge="",
    )

    response = reply.await_args.args[3]
    assert "I can't confirm this appointment yet" in response
    assert "Google Calendar" not in response
    assert "confirmed for Monday" not in response


def test_create_telegram_booking_checks_availability_and_creates_event(monkeypatch):
    timezone = pytz.timezone("Asia/Singapore")
    booking_date = (datetime.now(timezone) + timedelta(days=2)).date()
    while booking_date.weekday() >= 5:
        booking_date += timedelta(days=1)
    config = {
        "client": {"timezone": "Asia/Singapore"},
        "modules": {"booking": True},
        "booking": {
            "calendar_id": "test@calendar",
            "calendar_owner_email": "owner@example.com",
            "business_hours": {"start": "09:00", "end": "18:00"},
            "services": [{"name": "Consultation", "duration_minutes": 60, "price": 100}],
            "locations": [{
                "name": "Main Office",
                "address": "123 Main Street",
                "days": [booking_date.strftime("%A").lower()],
            }],
        },
    }
    calendar = MagicMock()
    calendar.is_slot_available.return_value = True
    calendar.has_upcoming_appointment_for_patient.return_value = False
    monkeypatch.setattr("core.telegram._calendar_client", lambda _config: calendar)

    result = _create_telegram_booking(
        config,
        12345,
        [
            "Consultation", booking_date.isoformat(), "10:00", "Child Example",
            "2020-05-03", "Parent Example", "None", "yes",
        ],
    )

    assert "Appointment confirmed" in result
    assert "SGD 100" in result
    calendar.create_event.assert_called_once()
    assert calendar.create_event.call_args.kwargs["telegram_chat_id"] == 12345
    assert calendar.create_event.call_args.kwargs["child_name"] == "Child Example"
    assert calendar.create_event.call_args.kwargs["parent_name"] == "Parent Example"
    assert calendar.create_event.call_args.kwargs["referral_name"] == "None"
    assert calendar.create_event.call_args.kwargs["patient_age_group"] == "Child (0-12)"
    assert calendar.create_event.call_args.kwargs["duration_minutes"] == 60
    assert "location_address" not in calendar.create_event.call_args.kwargs


def test_telegram_booking_creates_event_in_selected_consultant_calendar(monkeypatch):
    timezone = pytz.timezone("Asia/Singapore")
    booking_date = (datetime.now(timezone) + timedelta(days=2)).date()
    while booking_date.weekday() >= 5:
        booking_date += timedelta(days=1)
    config = {
        "client": {"timezone": "Asia/Singapore"},
        "modules": {"booking": True},
        "booking": {
            "consultants": [
                {"name": "Consultant One", "calendar_id": "one@calendar"},
                {"name": "Consultant Two", "calendar_id": "two@calendar"},
            ],
            "business_hours": {"start": "09:00", "end": "18:00"},
            "services": [{"name": "Consultation", "duration_minutes": 60, "price": 100}],
            "locations": [{
                "name": "Main Office",
                "address": "123 Main Street",
                "days": [booking_date.strftime("%A").lower()],
            }],
        },
    }
    calendar = MagicMock()
    calendar.is_slot_available.return_value = True
    calendar.has_upcoming_appointment_for_patient.return_value = False
    calendar_client = MagicMock(return_value=calendar)
    monkeypatch.setattr("core.telegram.CalendarClient", calendar_client)

    result = _create_telegram_booking(
        config,
        12345,
        [
            "Consultation", booking_date.isoformat(), "10:00", "Child Example",
            "2020-05-03", "Parent Example", "None", "yes",
        ],
        "Consultant Two",
    )

    assert calendar_client.call_args.kwargs["calendar_id"] == "two@calendar"
    calendar.create_event.assert_called_once()
    assert "with Consultant Two" in result


def test_create_telegram_booking_routes_nonresident_to_admin(monkeypatch):
    from core.telegram import _create_telegram_booking

    config = {"booking": {"locations": [{"name": "Main Office"}]}}
    calendar = MagicMock()
    monkeypatch.setattr("core.telegram._calendar_client", lambda _config: calendar)

    with pytest.raises(ValueError, match="Our administrative team will assist"):
        _create_telegram_booking(
            config,
            12345,
            [
                "Consultation", "2026-11-06", "10:00", "Child Example",
                "2020-05-03", "Parent Example", "None", "no",
            ],
        )

    calendar.create_event.assert_not_called()


def test_create_telegram_booking_rejects_unavailable_slot(monkeypatch):
    timezone = pytz.timezone("Asia/Singapore")
    booking_date = (datetime.now(timezone) + timedelta(days=2)).date()
    while booking_date.weekday() >= 5:
        booking_date += timedelta(days=1)
    config = {
        "client": {"timezone": "Asia/Singapore"},
        "modules": {"booking": True},
        "booking": {
            "calendar_id": "test@calendar",
            "calendar_owner_email": "owner@example.com",
            "business_hours": {"start": "09:00", "end": "18:00"},
            "services": [{"name": "Consultation", "duration_minutes": 60, "price": 100}],
            "locations": [{
                "name": "Main Office",
                "address": "123 Main Street",
                "days": [booking_date.strftime("%A").lower()],
            }],
        },
    }
    calendar = MagicMock()
    calendar.is_slot_available.return_value = False
    calendar.has_upcoming_appointment_for_patient.return_value = False
    monkeypatch.setattr("core.telegram._calendar_client", lambda _config: calendar)

    with pytest.raises(ValueError, match="already booked"):
        _create_telegram_booking(
            config,
            12345,
            [
                "Consultation", booking_date.isoformat(), "10:00", "Child Example",
                    "2020-05-03", "Parent Example", "None", "yes",
            ],
        )
    calendar.create_event.assert_not_called()
