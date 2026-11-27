import re

BOOKING_CONFIRMATION_INSTRUCTION = (
    "To book this appointment, reply CONFIRM only, without punctuation or other words. "
    "Other replies will not book it. "
    "For an appointment reminder, reply YES only to confirm attendance."
)
REMINDER_CONFIRMATION_INSTRUCTION = (
    "To confirm attendance, reply YES only, without punctuation or other words. "
    "Other replies will not confirm attendance."
)

BOOKING_NOT_CONFIRMED = (
    "I haven't booked the appointment from that reply. "
    + BOOKING_CONFIRMATION_INSTRUCTION
)
REMINDER_NOT_CONFIRMED = (
    "Your attendance has not been confirmed by that reply. "
    + REMINDER_CONFIRMATION_INSTRUCTION
)


def is_exact_reply(text: str, word: str) -> bool:
    return text.strip().casefold() == word.casefold()


def is_short_acknowledgement(text: str) -> bool:
    return bool(re.fullmatch(
        r"\s*(?:yes|yeah|yep|yup|ok|okay|sure|correct|confirm|confirmed|"
        r"agree|alright|right|fine|sounds good|i want to confirm it)(?:\s+please)?[.!?]*\s*",
        text,
        re.IGNORECASE,
    ))


def is_booking_confirmation_prompt(text: str) -> bool:
    if not (
        re.search(r"\bconfirm\b", text, re.IGNORECASE)
        and re.search(r"\b(?:book|booking|appointment)\b", text, re.IGNORECASE)
        and not re.search(r"\bappointment (?:is )?confirmed\b", text, re.IGNORECASE)
    ):
        return False
    labels = (
        r"service",
        r"date",
        r"time",
        r"patient(?: name)?",
        r"patient (?:dob|date of birth)",
        r"parent(?: name)?",
        r"referral",
        r"patient resides in singapore",
    )
    return all(
        re.search(
            rf"^\s*[-*]?\s*\*{{0,2}}{label}\s*:\*{{0,2}}\s*\S",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        for label in labels
    )


def awaiting_booking_confirmation(history: list[dict]) -> bool:
    for message in reversed(history):
        if message.get("role") == "assistant":
            return is_booking_confirmation_prompt(str(message.get("content", "")))
    return False


def awaiting_booking_information(history: list[dict]) -> bool:
    for message in reversed(history):
        if message.get("role") == "assistant":
            text = str(message.get("content", ""))
            return bool(
                "?" in text
                and re.search(
                    r"\b(?:book(?:ing)?|appointment|patient|child|parent|"
                    r"referral|reside|resides|singapore)\b",
                    text,
                    re.IGNORECASE,
                )
            )
    return False


def prepare_booking_confirmation_prompt(text: str) -> str:
    if not is_booking_confirmation_prompt(text):
        return text
    text = re.sub(
        r"(?i)\b(reply|type|send)\s*['\"`*]*yes['\"`*]*",
        r"\1 CONFIRM",
        text,
    )
    if BOOKING_CONFIRMATION_INSTRUCTION not in text:
        text = f"{text.rstrip()}\n\n{BOOKING_CONFIRMATION_INSTRUCTION}"
    return text
