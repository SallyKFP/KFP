from concurrent.futures import ThreadPoolExecutor

import pytest

from core.pending_booking import (
    PendingBookings, extract_general_observations, has_booking_cancellation,
    preserve_observations, preserves_pending_booking,
)


def details(**changes):
    return {
        "service": "Initial evaluation", "date": "2026-11-23", "time": "10:00",
        "child_name": "Jane", "child_dob": "2020-05-03", "parent_name": "Alex",
        "referral_name": "None", "patient_resides_in_singapore": "yes",
        "location": "Main clinic", **changes,
    }


def test_expiry_and_conversation_isolation():
    now = [0.0]
    store = PendingBookings(clock=lambda: now[0])
    store.put("whatsapp:one", details())
    store.put("telegram:one", details(child_name="Other"))
    now[0] = 86399
    assert store.get("whatsapp:one").details["child_name"] == "Jane"
    assert store.get("telegram:one").details["child_name"] == "Other"
    now[0] = 86400
    assert store.take("whatsapp:one") is None
    assert store.get("telegram:one") is None


def test_only_one_concurrent_confirm_can_take_booking():
    store = PendingBookings(clock=lambda: 0.0)
    store.put("whatsapp:one", details())
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: store.take("whatsapp:one"), range(20)))
    assert sum(result is not None for result in results) == 1


def test_snapshot_is_independent_of_original_details():
    store = PendingBookings(clock=lambda: 0.0)
    source = details()
    record = store.put("whatsapp:one", source)
    source["time"] = "12:00"
    assert record.details["time"] == "10:00"
    assert "- Time: 10:00" in record.summary


def test_future_birth_date_is_not_staged(monkeypatch):
    from datetime import date
    monkeypatch.setattr("core.patient_dates.clinic_today", lambda timezone: date(2026, 11, 24))
    store = PendingBookings(clock=lambda: 0.0)
    record = store.put("whatsapp:one", details(child_dob="2026-10-10"), timezone="Asia/Singapore")
    assert record.details["child_dob"] == "2026-10-10"
    assert store.put("whatsapp:two", details(child_dob="2026-11-24")).details["child_dob"] == "2026-11-24"
    with pytest.raises(ValueError, match="future"):
        store.put("whatsapp:three", details(child_dob="2026-11-25"), timezone="Asia/Singapore")
    assert store.get("whatsapp:three") is None


def test_optional_observations_are_reviewed_replaced_and_removed():
    store = PendingBookings(clock=lambda: 0.0)
    record = store.put("whatsapp:one", details())
    assert "sensitive health details" in record.summary
    assert "calendar appointment" in record.summary
    assert "observations" not in record.details
    record = store.set_observations("whatsapp:one", " Enjoys playing\nwith blocks ")
    assert "observations" not in record.details
    assert "- Proposed note for consultant (not yet approved): Enjoys playing with blocks" in record.summary
    assert store.take("whatsapp:one") is None
    record = store.set_observations("whatsapp:one", "Prefers quiet activities")
    assert "Enjoys playing" not in record.summary
    record = store.approve_observations("whatsapp:one")
    assert record.details["observations"] == "Prefers quiet activities"
    assert store.take("whatsapp:one").details["observations"] == "Prefers quiet activities"
    with pytest.raises(ValueError, match="no active booking"):
        store.set_observations("whatsapp:one", "Another note")
    store.put("whatsapp:one", details())
    store.set_observations("whatsapp:one", "A note")
    record = store.set_observations("whatsapp:one", None)
    assert "observations" not in record.details
    assert "- Observations for consultant:" not in record.summary


def test_observations_reject_invalid_or_inactive_booking_and_preserve_same_patient():
    now = [0.0]
    store = PendingBookings(ttl=1, clock=lambda: now[0])
    record = store.put("whatsapp:one", details())
    for note in ("", " " * 3, "x" * 1501):
        with pytest.raises(ValueError):
            store.set_observations("whatsapp:one", note)
    record = store.set_observations("whatsapp:one", "Prefers quiet activities")
    record = store.approve_observations("whatsapp:one")
    revised = details(observations="invented")
    preserve_observations(revised, record)
    assert revised["observations"] == "Prefers quiet activities"
    revised = details(child_name="Other", observations="invented")
    preserve_observations(revised, record)
    assert "observations" not in revised
    store.invalidate("whatsapp:one")
    with pytest.raises(ValueError, match="no active booking"):
        store.set_observations("whatsapp:one", "note")
    store.put("whatsapp:one", details())
    now[0] = 2
    with pytest.raises(ValueError, match="no active booking"):
        store.set_observations("whatsapp:one", "note")


def test_extract_general_observations_excludes_developmental_and_medical_information():
    history = [
        {"role": "user", "content": "My daughter is 10.5 months old and not yet crawling."},
        {"role": "assistant", "content": "A consultant can review that."},
    ]
    assert extract_general_observations(history, "I'd like an appointment") is None
    history[0]["content"] += " She enjoys playing with blocks, and has difficulty crawling."
    assert extract_general_observations(history, "I'd like an appointment") == "She enjoys playing with blocks"
    history.append({"role": "assistant", "content": "Appointment confirmed."})
    assert extract_general_observations(history, "I'd like another appointment") is None


@pytest.mark.parametrize("changes", [
    {"date": "tomorrow"}, {"time": "10:00:00"}, {"child_dob": "2020-02-31"},
    {"patient_resides_in_singapore": "perhaps"}, {"location": ""},
])
def test_invalid_data_is_not_staged(changes):
    store = PendingBookings(clock=lambda: 0.0)
    with pytest.raises(ValueError):
        store.put("whatsapp:one", details(**changes))
    assert store.get("whatsapp:one") is None


def test_rescheduling_requires_matching_patient():
    store = PendingBookings(clock=lambda: 0.0)
    with pytest.raises(ValueError, match="verify"):
        store.put("whatsapp:one", details(), {"id": "old", "patient_name": "Other"})
    assert store.get("whatsapp:one") is None
    record = store.put("whatsapp:one", details(), {
        "id": "old", "patient_name": "Jane", "date": "2026-11-20", "time": "11:00",
    })
    assert "replace the existing appointment for Jane on 2026-11-20 at 11:00" in record.summary
    store.invalidate("whatsapp:one")
    assert store.take("whatsapp:one") is None
    assert store.get("whatsapp:one").modification["id"] == "old"


@pytest.mark.parametrize(("text", "preserved"), [
    ("I want to confirm it", True),
    ("Where is the clinic?", True),
    ("What is the cancellation policy?", True),
    ("Can we change the time?", False),
    ("What about next Tuesday?", False),
    ("Can it be in her name?", False),
    ("12:00 instead", False),
])
def test_safety_of_follow_up(text, preserved):
    assert preserves_pending_booking(text) is preserved


def test_question_about_cancellation_is_not_a_cancellation_request():
    assert not has_booking_cancellation("What happens if I cancel?")
    assert has_booking_cancellation("Please cancel this booking.")
