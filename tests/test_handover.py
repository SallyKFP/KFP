from core.ai import extract_booking_intent
from core.handover import HANDOVER_PLACEHOLDER, compose_handover_reply, handover_reason


def test_reply_keeps_answerable_part_then_placeholder():
    reply = compose_handover_reply("We are open Monday to Friday, 8 am to 6 pm.")
    assert reply.startswith("We are open Monday to Friday")
    assert reply.endswith(HANDOVER_PLACEHOLDER)


def test_reply_is_only_placeholder_when_nothing_answerable():
    assert compose_handover_reply("  ") == HANDOVER_PLACEHOLDER


def test_handover_intent_is_split_from_visible_text():
    raw = 'We open at 8 am.\n{"intent": "handover", "reason": "clinical"}'
    intent, visible = extract_booking_intent(raw)
    assert intent == {"intent": "handover", "reason": "clinical"}
    assert visible == "We open at 8 am."
    assert handover_reason(intent) == "clinical"


def test_unknown_reason_falls_back_to_admin():
    assert handover_reason({"intent": "handover", "reason": "other"}) == "admin"
    assert handover_reason({"intent": "handover"}) == "admin"


import pytest


@pytest.mark.asyncio
async def test_lookup_lists_appointments_found_by_number(mocker):
    from core.main import _handle_appointment_lookup
    mocker.patch("core.main.HISTORY", mocker.MagicMock())
    mocker.patch("core.main._find_upcoming_events_by_phone", return_value=[
        {"summary": "Physio", "date": "Monday March 1, 2027", "time": "10:00",
         "location": "Clinic", "consultant": "Sarah"}])
    send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    await _handle_appointment_lookup("6591234567")
    assert send.call_args.args[1] == (
        "Your next appointment is Physio: Monday March 1, 2027 at 10:00 with Sarah at Clinic.")


@pytest.mark.asyncio
async def test_lookup_with_no_match_hands_over_to_staff(mocker):
    from core.main import _handle_appointment_lookup
    mocker.patch("core.main.HISTORY", mocker.MagicMock())
    mocker.patch("core.main._find_upcoming_events_by_phone", return_value=[])
    pause = mocker.patch("core.main._pause_for_handover", new_callable=mocker.AsyncMock)
    send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    await _handle_appointment_lookup("6591234567")
    assert HANDOVER_PLACEHOLDER in send.call_args.args[1]
    pause.assert_awaited_once_with("whatsapp", "6591234567", "admin")
