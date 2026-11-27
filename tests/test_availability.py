from datetime import date, datetime, time, timedelta
from unittest.mock import MagicMock

import pytest
import pytz

from modules.booking.availability import check_availability_range
from modules.booking.calendar import CalendarClient, Slot


def _next_weekday(day: int = 0) -> date:
    today = datetime.now(pytz.timezone("Asia/Singapore")).date()
    days_ahead = (day - today.weekday()) % 7 or 7
    return today + timedelta(days=days_ahead)


def _config(start_date: date, end_date: date) -> tuple[dict, dict]:
    config = {
        "client": {"timezone": "Asia/Singapore"},
        "modules": {"booking": True},
        "booking": {
            "business_hours": {"start": "08:00", "end": "18:00"},
            "services": [
                {"name": "Initial evaluation", "duration_minutes": 60},
            ],
            "locations": [{
                "name": "Main clinic",
                "days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
            }],
        },
    }
    intent = {
        "service": "Initial evaluation",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "location": "Main clinic",
    }
    return config, intent


def test_find_available_slots_queries_calendar_once_and_excludes_busy_times(monkeypatch):
    service = MagicMock()
    start_date = _next_weekday()
    calendar_id = "test@calendar"
    timezone = pytz.timezone("Asia/Singapore")
    service.freebusy.return_value.query.return_value.execute.return_value = {
        "calendars": {
            calendar_id: {
                "busy": [{
                    "start": f"{start_date.isoformat()}T09:00:00+08:00",
                    "end": f"{start_date.isoformat()}T10:00:00+08:00",
                }],
            }
        }
    }
    monkeypatch.setattr("modules.booking.calendar._get_redis", lambda: None)
    client = CalendarClient.__new__(CalendarClient)
    client._calendar_id = calendar_id
    client._tz = timezone
    client._service = service

    slots = client.find_available_slots(
        start_date,
        start_date,
        60,
        {"start": "08:00", "end": "18:00"},
        ["monday", "tuesday", "wednesday", "thursday", "friday"],
        "Main clinic",
    )

    assert slots[0] == Slot(start_date, time(8), "Main clinic")
    assert Slot(start_date, time(9), "Main clinic") not in slots
    assert Slot(start_date, time(10), "Main clinic") in slots
    service.freebusy.return_value.query.assert_called_once()


def test_check_availability_range_checks_all_consultants_and_returns_real_options():
    start_date = _next_weekday()
    end_date = start_date + timedelta(days=6)
    config, intent = _config(start_date, end_date)
    config["booking"]["consultants"] = [
        {"name": "Consultant Sarah", "calendar_id": "sarah@calendar"},
        {"name": "Consultant Serena", "calendar_id": "serena@calendar"},
    ]
    calendars = {
        "Consultant Sarah": MagicMock(),
        "Consultant Serena": MagicMock(),
    }
    calendars["Consultant Sarah"].find_available_slots.return_value = [
        Slot(start_date, time(14), "Main clinic"),
    ]
    available_date = start_date + timedelta(days=1)
    calendars["Consultant Serena"].find_available_slots.return_value = [
        Slot(available_date, time(10), "Main clinic"),
    ]

    response = check_availability_range(
        config,
        intent,
        lambda name: calendars[name],
    )

    assert "available" in response
    assert "14:00 with Consultant Sarah" in response
    assert "10:00 with Consultant Serena" in response
    for calendar in calendars.values():
        calendar.find_available_slots.assert_called_once()


def test_check_availability_range_accepts_three_calendar_months():
    start_date = date(2027, 1, 31)
    config, intent = _config(start_date, date(2027, 4, 30))
    calendar = MagicMock()
    calendar.find_available_slots.return_value = []

    check_availability_range(config, intent, lambda _name: calendar)

    calendar.find_available_slots.assert_called_once()


def test_check_availability_range_rejects_more_than_three_calendar_months():
    start_date = date(2027, 1, 31)
    config, intent = _config(start_date, date(2027, 5, 1))
    calendar_factory = MagicMock()

    with pytest.raises(ValueError, match="up to 3 calendar months"):
        check_availability_range(config, intent, calendar_factory)

    calendar_factory.assert_not_called()


def test_check_availability_range_reports_no_open_slots():
    start_date = _next_weekday()
    config, intent = _config(start_date, start_date)
    calendar = MagicMock()
    calendar.find_available_slots.return_value = []

    response = check_availability_range(config, intent, lambda _name: calendar)

    assert "couldn't find an available Initial evaluation appointment" in response


def test_check_availability_range_omits_consultant_reserved_times():
    start_date = _next_weekday()
    thursday = start_date + timedelta(days=3)
    config, intent = _config(start_date, start_date + timedelta(days=6))
    config["booking"]["consultants"] = [{
        "name": "Consultant Sarah",
        "calendar_id": "sarah@calendar",
    }]
    calendar = MagicMock()
    calendar.find_available_slots.return_value = [
        Slot(thursday, time(12), "Main clinic"),
        Slot(thursday, time(13), "Main clinic"),
    ]

    response = check_availability_range(config, intent, lambda _name: calendar)

    assert "Thursday, " in response
    assert "13:00 with Consultant Sarah" in response
    assert "12:00" not in response


def test_check_availability_range_limits_reply_to_ten_options():
    start_date = _next_weekday()
    config, intent = _config(start_date, start_date + timedelta(days=30))
    calendar = MagicMock()
    slots = [
        Slot(start_date + timedelta(days=day), time(10), "Main clinic")
        for day in range(20)
    ]
    calendar.find_available_slots.return_value = slots

    response = check_availability_range(config, intent, lambda _name: calendar)

    assert response.count("\n- ") == 10
    assert "more available times" in response


@pytest.fixture(autouse=False)
def _env(monkeypatch):
    monkeypatch.setenv("A", "a@cal")
    monkeypatch.setenv("B", "b@cal")


def _slot_config():
    return {
        "client": {"timezone": "Asia/Singapore"},
        "booking": {
            "consultants": [
                {"name": "Consultant Sarah", "calendar_id_env": "A"},
                {"name": "Consultant Serena", "calendar_id_env": "B"},
            ],
            "business_hours": {"start": "08:00", "end": "18:00"},
            "services": [{"name": "Initial evaluation", "duration_minutes": 60, "price": 300}],
            "locations": [{"name": "Main Office", "days": ["monday"]}],
        },
    }


def _future_monday():
    from datetime import date, timedelta
    d = date.today() + timedelta(days=14)
    return d + timedelta(days=(7 - d.weekday()) % 7)


class _Cal:
    def __init__(self, free):
        self.free = free

    def is_slot_available(self, *_):
        return self.free


def test_slot_check_assigns_a_free_consultant_and_remembers_it(_env):
    from modules.booking.availability import check_slot_availability
    remembered = []
    intent = {"intent": "availability_check", "service": "Initial evaluation",
              "date": _future_monday().isoformat(), "time": "10:00"}
    reply = check_slot_availability(
        _slot_config(), intent,
        lambda name: _Cal(name == "Consultant Serena"),
        lambda checked, name: remembered.append((checked["location"], name)),
    )
    assert "is available at Main Office with Consultant Serena" in reply
    assert "isn't reserved yet" in reply
    assert remembered == [("Main Office", "Consultant Serena")]


def test_slot_check_reports_when_both_consultants_are_busy(_env):
    from modules.booking.availability import check_slot_availability
    intent = {"service": "Initial evaluation", "date": _future_monday().isoformat(), "time": "10:00"}
    reply = check_slot_availability(_slot_config(), intent, lambda name: _Cal(False))
    assert "isn't available with either consultant" in reply


def test_whatsapp_prompt_has_availability_check_and_lookup():
    from core.ai import build_system_prompt
    prompt = build_system_prompt(
        {"client": {"name": "T"}, "modules": {"booking": True},
         "booking": {"services": [{"name": "A", "price": 1, "duration_minutes": 60}],
                     "locations": [{"name": "M", "address": "x", "days": ["monday"]}],
                     "business_hours": {"start": "08:00", "end": "18:00"}}},
        "knowledge",
    )
    assert '"intent": "availability_check"' in prompt
    assert "WHATSAPP APPOINTMENT LOOKUPS" in prompt
