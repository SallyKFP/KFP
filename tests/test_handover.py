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
