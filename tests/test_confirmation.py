import pytest

from core.confirmation import (
    awaiting_booking_confirmation,
    awaiting_booking_information,
    is_exact_reply,
    prepare_booking_confirmation_prompt,
)


@pytest.mark.parametrize(
    ("text", "word", "expected"),
    [
        ("CONFIRM", "CONFIRM", True),
        (" confirm ", "CONFIRM", True),
        ("CONFIRM!", "CONFIRM", False),
        ("yes confirm", "CONFIRM", False),
        ("YES", "YES", True),
        (" yes ", "YES", True),
        ("YES!", "YES", False),
        ("yes please", "YES", False),
    ],
)
def test_exact_replies(text, word, expected):
    assert is_exact_reply(text, word) is expected


def test_booking_prompt_replaces_yes_instruction_and_retains_details():
    text = (
        "Please check these details:\n"
        "- Service: Initial evaluation\n"
        "- Date: 2026-11-23\n"
        "- Time: 10:00\n"
        "- Patient: Example\n"
        "- Patient DOB: 2020-05-03\n"
        "- Parent: Parent Example\n"
        "- Referral: None\n"
        "- Patient resides in Singapore: Yes\n"
        "Reply YES to confirm and I'll book the appointment."
    )

    response = prepare_booking_confirmation_prompt(text)

    assert "Reply YES to confirm" not in response
    assert "Reply CONFIRM to confirm" in response
    assert "Other replies will not book it" in response
    assert awaiting_booking_confirmation([{"role": "assistant", "content": response}])


@pytest.mark.parametrize("text", [
    "Please confirm the booking details when asked so I can check and move the appointment.",
    "Please confirm these patient details before I book.",
    "Please confirm your booking:\n- Service: Initial evaluation\n- Patient: Example",
])
def test_incomplete_or_generic_prompt_does_not_authorize_booking(text):
    assert not awaiting_booking_confirmation([{"role": "assistant", "content": text}])


def test_yes_to_booking_information_question_is_not_a_reminder_reply():
    history = [{"role": "assistant", "content": "Does the patient reside in Singapore?"}]
    assert awaiting_booking_information(history)
