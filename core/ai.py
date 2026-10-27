import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import anthropic
import pytz

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 1024

_client: anthropic.Anthropic | None = None

_APPOINTMENT_WORDS = re.compile(r"\b(appointment|book|booking|slot)\b", re.IGNORECASE)
_IMMEDIATE_WORDS = re.compile(
    r"\b(right now|now|currently|at the moment)\b",
    re.IGNORECASE,
)
_FEE_QUESTION_WORDS = re.compile(
    r"\b(fee|fees|charge|charges|cost|costs|price|prices)\b",
    re.IGNORECASE,
)
_SLOT_AVAILABILITY_WORDS = re.compile(
    r"\b(availability|available|openings|free slots?)\b",
    re.IGNORECASE,
)
_CLINICAL_CONTEXT_WORDS = re.compile(
    r"\b(child|baby|infant|son|daughter|torticollis|plagiocephaly|"
    r"condition|diagnos\w*|disease|symptom\w*|disorder|delay|"
    r"medical|clinical|referral)\b",
    re.IGNORECASE,
)
_CONSULTANT_REQUEST_WORDS = re.compile(
    r"\b(referral|referred|appointment|consult\w*|see|meet|request)\b",
    re.IGNORECASE,
)
_OCCUPATIONAL_THERAPY_WORDS = re.compile(
    r"\b(occupational therap(?:y|ist)s?|OTs?)\b",
    re.IGNORECASE,
)
_EVALUATION_REPORT_WORDS = re.compile(
    r"\b(?:evaluation|assessment)\s+reports?\b"
    r"|\breports?\b.{0,40}\b(?:fee|fees|cost|price|request|get|receive|obtain)\b"
    r"|\b(?:request|get|receive|obtain)\b.{0,40}\breports?\b",
    re.IGNORECASE,
)
_REPORT_FOLLOW_UP_WORDS = re.compile(
    r"\b(?:fee|fees|charge|charges|cost|costs|price|prices|pay|payment|when|"
    r"how much|how long|turnaround|ready|receive|get it|that|it)\b",
    re.IGNORECASE,
)
_WEEKEND_WORDS = re.compile(r"\b(weekends?|saturdays?|sundays?)\b", re.IGNORECASE)
_WEEKEND_APPOINTMENT_WORDS = re.compile(
    r"\b(appointment|book|booking|schedule|slot|follow[\s-]?up|come|visit|request)\b",
    re.IGNORECASE,
)
_REFERRAL_WORDS = re.compile(
    r"\b(referr?al|referred|referring|sent by|recommended by)\b",
    re.IGNORECASE,
)
_CONSULTANT_APPOINTMENT_REQUEST = re.compile(
    r"\b(?:consultant|physiotherapist|physio)\b.{0,60}"
    r"\b(?:see|meet|appointment|consult|request|book|speak)\b"
    r"|\b(?:see|meet|appointment|consult|request|book|speak)\b.{0,60}"
    r"\b(?:consultant|physiotherapist|physio)\b",
    re.IGNORECASE,
)


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def load_knowledge(path: str = "knowledge/client.txt") -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Knowledge file not found: %s", path)
        return ""


def _get_faq_answer(knowledge: str, faq_id: str) -> str | None:
    faq_match = re.search(
        rf"^FAQ ID: {re.escape(faq_id)}\s*$([\s\S]*?)(?=^FAQ ID:|\Z)",
        knowledge,
        re.MULTILINE,
    )
    if not faq_match:
        return None

    answer_match = re.search(
        r"^ANSWER:\s*(.*?)(?=^HUMAN REVIEW REQUIRED:|^FAQ ID:|\Z)",
        faq_match.group(1),
        re.MULTILINE | re.DOTALL,
    )
    return answer_match.group(1).strip() if answer_match else None


def _fee_question_answer(user_message: str, knowledge: str) -> str | None:
    if not _FEE_QUESTION_WORDS.search(user_message):
        return None
    if re.search(r"\b(cancell?ation|report)\b", user_message, re.IGNORECASE):
        return None
    return _get_faq_answer(knowledge, "appointment_fees")


def _slot_availability_answer(user_message: str, knowledge: str) -> str | None:
    if not (
        _APPOINTMENT_WORDS.search(user_message)
        and _SLOT_AVAILABILITY_WORDS.search(user_message)
    ):
        return None
    if _FEE_QUESTION_WORDS.search(user_message):
        return None
    return _get_faq_answer(knowledge, "appointment_slot_availability")


def _evaluation_report_answer(
    user_message: str,
    history: list[dict],
    knowledge: str,
) -> str | None:
    report_request = bool(_EVALUATION_REPORT_WORDS.search(user_message))
    if not report_request and _REPORT_FOLLOW_UP_WORDS.search(user_message):
        report_request = any(
            message.get("role") in {"user", "assistant"}
            and re.search(r"\bevaluation report\b|\breport fees?\b", message.get("content", ""), re.IGNORECASE)
            for message in history[-6:]
        )
    if not report_request:
        return None
    return _get_faq_answer(knowledge, "evaluation_report")


def _occupational_therapy_answer(user_message: str, knowledge: str) -> str | None:
    if not _OCCUPATIONAL_THERAPY_WORDS.search(user_message):
        return None
    return _get_faq_answer(knowledge, "occupational_therapy")


def _patient_type(user_message: str) -> str | None:
    if re.search(
        r"\b(?:not\s+(?:a\s+)?new|aren't\s+new|are\s+not\s+new)\b"
        r"|\b(?:returning|existing|current|old)\s+patients?\b"
        r"|\b(?:been|visited|seen|attended|came|went|was|were)\b.{0,40}"
        r"\b(?:to\s+)?(?:the\s+)?(?:clinic|us|you|here|there)\b"
        r"(?:.{0,20}\b(?:before|ago|previously|last\s+(?:week|month|year))\b)?"
        r"|\b(?:been|visited|seen|attended|came|went)\b.{0,40}"
        r"\b(?:before|previously|last\s+(?:week|month|year)|"
        r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:days?|weeks?|months?|years?)\s+ago)\b"
        r"|\b(?:returning|existing|old)\b",
        user_message,
        re.IGNORECASE,
    ):
        return "existing"
    if re.search(
        r"\b(?:my|our)\s+(?:child|son|daughter)\s+(?:is|will be)\s+a?\s*new\s+patient\b"
        r"|\bwe(?:'re| are)\s+new\s+patients?\b"
        r"|\bi(?:'m| am)\s+(?:a\s+)?new(?:\s+patient)?\b"
        r"|\bi(?:'m| am)\s+the\s+parent\s+of\s+(?:a\s+)?new\s+(?:patient|child)\b"
        r"|\bwe(?:'re| are)\s+new\b"
        r"|\b(?:first|initial)\s+(?:time|visit|appointment|evaluation)\b"
        r"|\bnever\s+(?:been|visited|seen)\s+(?:at|by|with)\s+(?:the\s+)?(?:clinic|us)\b"
        r"|\A\s*new(?:\s+patient)?\s*\Z",
        user_message,
        re.IGNORECASE,
    ):
        return "new"
    if re.fullmatch(r"\s*(?:existing|returning|current|old)(?:\s+patient)?\s*", user_message, re.IGNORECASE):
        return "existing"
    return None


def _remembered_patient_type(user_message: str, history: list[dict]) -> str | None:
    patient_type = _patient_type(user_message)
    if patient_type:
        return patient_type
    for message in reversed(history):
        if message.get("role") != "user":
            continue
        patient_type = _patient_type(message.get("content", ""))
        if patient_type:
            return patient_type
    return None


def _weekend_appointment_answer(
    user_message: str,
    history: list[dict],
    knowledge: str,
) -> str | None:
    is_weekend_request = bool(
        _WEEKEND_WORDS.search(user_message)
        and _WEEKEND_APPOINTMENT_WORDS.search(user_message)
    )
    last_assistant_message = next(
        (
            message.get("content", "")
            for message in reversed(history)
            if message.get("role") == "assistant"
        ),
        "",
    )
    awaiting_patient_type = (
        "new patient" in last_assistant_message.casefold()
        and "visited" in last_assistant_message.casefold()
        and any(
            message.get("role") == "user"
            and _WEEKEND_WORDS.search(message.get("content", ""))
            and _WEEKEND_APPOINTMENT_WORDS.search(message.get("content", ""))
            for message in history[-6:]
        )
    )
    if not (is_weekend_request or awaiting_patient_type):
        return None

    patient_type = _remembered_patient_type(user_message, history)
    if patient_type is None:
        return "Is your child a new patient, or have they visited Kids Focus Physiotherapy before?"
    faq_id = "new_patient_saturday" if patient_type == "new" else "existing_patient_saturday"
    return _get_faq_answer(knowledge, faq_id)


def _requested_consultant(user_message: str, knowledge: str) -> str | None:
    if not _CONSULTANT_REQUEST_WORDS.search(user_message):
        return None

    consultant_section = re.search(
        r"^CONSULTANT INFORMATION\s*[\s\S]*?(?=^DAYS, LOCATIONS|^COMMON QUESTIONS)",
        knowledge,
        re.MULTILINE,
    )
    if not consultant_section:
        return None

    consultants = re.findall(
        r"^([A-Z][A-Za-z'’-]+ [A-Z][A-Za-z'’-]+),\s+(?:Chief|Consultant)\b",
        consultant_section.group(),
        re.MULTILINE,
    )
    message_lower = user_message.casefold()
    return next(
        (name for name in consultants if name.casefold() in message_lower),
        None,
    )


def _needs_appointment_details(user_message: str) -> bool:
    return bool(_CLINICAL_CONTEXT_WORDS.search(user_message))


def _should_include_appointment_details(user_message: str) -> bool:
    if _REFERRAL_WORDS.search(user_message) or _patient_type(user_message) == "new":
        return True

    has_lengthy_clinical_intro = (
        len(user_message.split()) >= 20
        and _CLINICAL_CONTEXT_WORDS.search(user_message)
    )
    return bool(
        has_lengthy_clinical_intro
        and _CONSULTANT_APPOINTMENT_REQUEST.search(user_message)
    )


def _safe_fallback_answer(
    user_message: str,
    knowledge: str,
    consultant: str | None,
    history: list[dict],
) -> str:
    parts = []
    fee_answer = _fee_question_answer(user_message, knowledge)
    if fee_answer:
        parts.append(fee_answer)

    if (
        (_needs_appointment_details(user_message) or _should_include_appointment_details(user_message))
        and not any("date of birth" in part.casefold() for part in parts)
    ):
        appointment_answer = _get_faq_answer(knowledge, "make_appointment")
        if appointment_answer:
            parts.append(appointment_answer)

    if consultant:
        parts.append(f"Just a moment please, I'll reach out to {consultant}.")

    if parts:
        return _ensure_requested_next_steps(
            "\n\n".join(parts),
            user_message,
            knowledge,
            consultant,
            history,
        )
    if _should_include_appointment_details(user_message):
        appointment_answer = _get_faq_answer(knowledge, "make_appointment")
        if appointment_answer:
            return _ensure_requested_next_steps(
                appointment_answer,
                user_message,
                knowledge,
                consultant,
                history,
            )
    if _remembered_patient_type(user_message, history) is None:
        return "Sorry, I couldn't generate a reply just now. May I check whether your child is a new patient or has visited Kids Focus Physiotherapy before?"
    return "Sorry, I couldn't generate a reply just now. Please try again."


def _ensure_requested_next_steps(
    answer: str,
    user_message: str,
    knowledge: str,
    consultant: str | None,
    history: list[dict] | None = None,
) -> str:
    if _remembered_patient_type(user_message, history or []) is not None:
        answer = _remove_patient_type_questions(answer)
    parts = [answer] if answer else []
    if (
        _should_include_appointment_details(user_message)
        and "date of birth" not in answer.casefold()
        and "to schedule an appointment, please provide:" not in answer.casefold()
    ):
        appointment_answer = _get_faq_answer(knowledge, "make_appointment")
        if appointment_answer:
            parts.append(appointment_answer)

    if consultant and f"reach out to {consultant}".casefold() not in answer.casefold():
        parts.append(f"Just a moment please, I'll reach out to {consultant}.")

    if (
        _remembered_patient_type(user_message, history or []) is None
        and not _already_asks_patient_type(answer)
    ):
        parts.append(
            "May I check whether your child is a new patient or has visited "
            "Kids Focus Physiotherapy before?"
        )

    return "\n\n".join(parts)


def _already_asks_patient_type(answer: str) -> bool:
    questions = re.findall(r"[^?]*\?", answer)
    return any(
        (
            "new" in question.casefold()
            and any(word in question.casefold() for word in ("existing", "returning"))
            and "patient" in question.casefold()
        )
        or (
            any(word in question.casefold() for word in ("visited", "been"))
            and "before" in question.casefold()
        )
        for question in questions
    )


def _remove_patient_type_questions(answer: str) -> str:
    def remove_if_redundant(match: re.Match) -> str:
        question = match.group()
        return "" if _already_asks_patient_type(question) else question

    answer = re.sub(
        r"(?:^|(?<=[.!?\n]))\s*[^.!?\n]*\?",
        remove_if_redundant,
        answer,
    )
    return re.sub(r"[ \t]{2,}", " ", answer).strip()


def _immediate_appointment_answer(
    user_message: str,
    config: dict,
    *,
    now: datetime | None = None,
) -> str | None:
    if not (_APPOINTMENT_WORDS.search(user_message) and _IMMEDIATE_WORDS.search(user_message)):
        return None

    client = config["client"]
    timezone = client.get("timezone", "UTC")
    local_timezone = pytz.timezone(timezone)
    local_now = now.astimezone(local_timezone) if now else datetime.now(local_timezone)
    hours = client.get("opening_hours", {})

    if local_now.weekday() < 5:
        day_hours = hours.get("weekdays")
        day_label = "weekday"
    elif local_now.weekday() == 5:
        day_hours = hours.get("saturday")
        day_label = "Saturday"
    else:
        day_hours = None
        day_label = "Sunday"

    if day_hours is not None:
        opens = datetime.strptime(day_hours["start"], "%H:%M").time()
        closes = datetime.strptime(day_hours["end"], "%H:%M").time()
        hours_text = (
            f"{opens.strftime('%I:%M %p').lstrip('0')} to "
            f"{closes.strftime('%I:%M %p').lstrip('0')}"
        )
        if opens <= local_now.time() < closes:
            status = f"within the clinic's regular {day_label} hours ({hours_text})"
        else:
            status = f"outside the clinic's regular {day_label} hours ({hours_text})"
    elif local_now.weekday() == 6:
        status = "on a day when the clinic is normally closed"
    else:
        status = "when regular hours are not configured"

    current_time = local_now.strftime("%I:%M %p").lstrip("0")
    current_day = local_now.strftime("%A")
    time_label = (
        "Singapore time"
        if timezone == "Asia/Singapore"
        else f"local time ({timezone})"
    )
    return (
        f"It's currently {current_day} at {current_time} {time_label}, {status}. "
        "The clinic is strictly by appointment, so opening hours do not guarantee a "
        "same-day slot. Appointment requests need to be reviewed and confirmed by "
        "the clinic; public-holiday hours and live availability should be confirmed "
        "with the administrative team."
    )


def _is_multi_part_query(user_message: str) -> bool:
    question_count = user_message.count("?")
    interrogative_count = len(
        re.findall(
            r"\b(?:what|which|who|where|when|how)\b",
            user_message,
            re.IGNORECASE,
        )
    )
    return question_count > 1 or interrogative_count > 1


def _multi_part_clinic_answer(
    user_message: str,
    history: list[dict],
    knowledge: str,
) -> str | None:
    parts = []

    weekend_answer = _weekend_appointment_answer(user_message, history, knowledge)
    if weekend_answer:
        parts.append(f"Saturday appointment: {weekend_answer}")

    if re.search(
        r"\b(?:best|most suitable|recommend|which one)\b",
        user_message,
        re.IGNORECASE,
    ) and re.search(r"\b(?:consultants?|physios?|physiotherapists?)\b", user_message, re.IGNORECASE):
        parts.append(
            "Consultant: Our consultants are all suitable to help your child. "
            "All our physiotherapists have similar depth of paediatric physiotherapy "
            "training and are fully competent to care for children with a wide spectrum "
            "of special needs, though their therapy styles differ."
        )

    fee_answer = _fee_question_answer(user_message, knowledge)
    if fee_answer:
        parts.append(f"Fees: {fee_answer}")

    if re.search(
        r"\b(?:payment|pay|CDA|CDAC|Baby Bonus|PayNow|cash)\b",
        user_message,
        re.IGNORECASE,
    ):
        payment_answer = _get_faq_answer(knowledge, "cda_payment")
        if payment_answer:
            parts.append(f"Payment: {payment_answer}")

    if re.search(
        r"\b(?:car|drive|driving|private transport|directions?)\b",
        user_message,
        re.IGNORECASE,
    ):
        directions_match = re.search(
            r"^Getting here by private transport:\s*(.+)$",
            knowledge,
            re.MULTILINE | re.IGNORECASE,
        )
        location_answer = _get_faq_answer(knowledge, "clinic_location")
        if location_answer:
            directions = f"{location_answer} "
            if directions_match:
                directions += directions_match.group(1).strip()
            if directions:
                parts.append(f"Driving directions: {directions}")

    if re.search(
        r"\b(?:arriv\w*|how early|what time.{0,30}(?:there|arrive|clinic|appointment))\b",
        user_message,
        re.IGNORECASE,
    ):
        punctuality_answer = _get_faq_answer(knowledge, "appointment_punctuality")
        if punctuality_answer:
            parts.append(f"Arrival: {punctuality_answer}")

    if len(parts) < 2:
        return None
    return "\n\n".join(parts)


def build_system_prompt(
    config: dict,
    knowledge: str,
    *,
    is_first_response: bool = False,
) -> str:
    client_name = config["client"]["name"]
    modules = config.get("modules", {})
    timezone = config["client"].get("timezone", "UTC")
    local_now = datetime.now(pytz.timezone(timezone))
    local_time_context = (
        f"{local_now.strftime('%A, %Y-%m-%d %I:%M %p')} ({timezone})"
    )

    lines = [
        f"You are Sally, the warm, thoughtful assistant for {client_name}.",
        f"Current local date and time: {local_time_context}.",
        "Keep every reply polite, friendly, and concise.",
        "Speak in the third person about the professional (do not impersonate them).",
        f"If asked who you are, introduce yourself simply as Sally from {client_name}. Do not describe yourself as a virtual receptionist, chatbot, or AI.",
        "Use the knowledge base as your source of truth, but do not recite or dump it.",
        "Do not include clinic phone numbers in replies; the patient is already contacting the clinic on this channel. When contact details are relevant, provide the clinic email address(es) from the knowledge base instead.",
        "Answer the user's specific question first, then include only the context or next step that is useful to them.",
        "For a message with multiple questions or requests, answer every distinct part in the same reply. Use a short numbered list or separate brief paragraphs when that makes the answers easier to follow. Do not stop after answering the first request; use the knowledge base for every factual answer, and clearly say when a requested detail is not available there.",
        "For simple factual questions, keep the answer brief. For policies or multi-part questions, explain the relevant conditions clearly without omitting important details.",
        "When asked generally who the consultants are, give only their names; do not include biographies, specialties, or qualifications unless the user asks for more detail. For a follow-up, share only the requested biographical details.",
        "Sound natural, calm, and empathetic. Acknowledge a concern briefly when appropriate; avoid canned openings, filler, and overly formal language.",
        "Ask at most one focused follow-up question, and only when needed to answer or help with the user's request.",
        "Treat each FAQ's SAMPLE QUESTIONS as examples of requests with the same meaning, even when the user's wording differs. For a matching FAQ, use its ANSWER as the response pattern and source of facts; you may adapt the wording naturally, but preserve its facts, prices, eligibility rules, conditions, and level of detail. Use the conversation history to distinguish cases such as new versus existing patients, and do not apply one case's policy to another.",
        "For a Saturday or weekend appointment request, ask whether the child is a new or returning patient only if the conversation does not establish this. If the patient has already identified themselves as new, existing, returning, or an old patient, use that information without asking again. Remember the patient's answer in the conversation and use it for later follow-ups; then follow the matching new_patient_saturday or existing_patient_saturday FAQ. Initial evaluations are only on weekdays; Saturday follow-ups may be requested only after a child has completed an initial evaluation, and are subject to extremely high demand and availability. State the full upfront SGD 300 payment and applicable Saturday payment/rescheduling policy for an existing patient's Saturday request.",
        "For every patient inquiry, if the conversation does not make clear whether the patient is new or existing, answer the inquiry and also ask whether the child is a new patient or has visited the clinic before. If the patient has already stated they are new, existing, returning, or an old patient in the current message or conversation history, do not ask this question again; acknowledge or use the information as relevant and answer the inquiry. Remember the answer in conversation history. A general question about new patients does not establish the user's own patient type.",
        "When a message indicates the patient is being referred, is a new patient, or gives a detailed account of the child's condition while requesting to see a consultant, answer the other parts of the question and also include the exact appointment-information heading and point-form list from the make_appointment FAQ. Do not repeat that list if it is already present.",
        "If a matching FAQ says HUMAN REVIEW REQUIRED: YES, explain that the administrative team must review and confirm the request. Never imply that the appointment is confirmed before the clinic confirms it.",
        "Whenever asking the patient for information to schedule an appointment, use this exact heading and point-form list, without converting it into a sentence: 'To schedule an appointment, please provide:' followed by '- Your child's full name and date of birth', '- Your full name', '- Your WhatsApp contact number', '- The referring person's name, if any', and '- Your preferred day and time'.",
        "Answer practical, non-clinical logistics questions (including directions, building access, elevators, parking, and what to bring) from verified facts in the knowledge base. For driving directions, use the 'Getting here by private transport' information when present; do not say directions are unavailable if they are documented. These are not clinical advice. Do not infer accessibility features from the building type or address; if the specific fact is not documented, say it is not confirmed and offer to ask the administrative team.",
        "When asked whether the clinic is open or an appointment can happen now, compare the current local date and time above with the opening-hours FAQ. Clearly distinguish regular opening hours from appointment availability: the clinic is strictly by appointment, and never claim that a slot is available or an appointment is confirmed unless the system explicitly confirms it. If public-holiday status or a live appointment slot cannot be checked, say so rather than guessing. Do not answer 'NULL'.",
        "Do not diagnose a child, assess their condition, or prescribe exercises. For requests for diagnosis, interpretation, or treatment advice, explain that child-specific clinical advice requires a physiotherapist's assessment and offer an appropriate next step.",
        "For questions asking whether Kids Focus Physiotherapy is the right facility or whether the consultants are suitable for the child's condition, this instruction takes precedence over the general clinical-advice guidance: reply only 'Our consultants are all suitable to help your child.' Then immediately provide the exact appointment-information heading and point-form list. Do not say you cannot assess the child, cannot recommend a physiotherapist, or that an assessment is needed; do not add any other clinical caveat or recommendation.",
        "If a message combines a factual clinic question (such as fees, hours, or services) with a child's clinical background, answer the factual question from its matching FAQ first. Then briefly explain that assessing the child's needs or recommending treatment requires a physiotherapist's assessment.",
        "When a caregiver asks about fees or charges and also mentions a child's condition or clinical background, follow a matching referral-and-fees FAQ if one exists. Otherwise answer the fee question using the appointment_fees FAQ, then ask for the appointment-request details listed in the make_appointment FAQ. Do not give medical advice.",
        "When asked about available appointment slots or live appointment availability, use the appointment_slot_availability FAQ. Explain that the administrative team will check live slots; do not say the AI is unable to check them.",
        "When the user mentions a referral or requests an appointment/consultation with a specific consultant whose full name appears in the knowledge base, include this exact sentence: 'Just a moment please, I'll reach out to <consultant's full name>.' Replace the placeholder with that consultant's name. Do not claim this for general biography questions or when no consultant is named.",
        "If a caregiver asks for an interpretation of their child's condition, symptoms, or development, or asks what treatment or exercises to use, do not assess or explain the condition. Acknowledge them briefly, then use a matching clinical appointment FAQ's ANSWER verbatim if available; otherwise use the make_appointment FAQ's ANSWER so they can schedule an appointment. The short suitability response above is the exception to the general clinical-advice guidance. Do not suggest that the request or appointment is already confirmed.",
        "If the knowledge base does not contain the answer, say so plainly and offer the relevant contact or next step if one is provided. Do not guess.",
        "Respond in the user's language.",
        "Be concise (usually 1-3 short paragraphs). Do not use emojis.",
        "FORMAT: Use WhatsApp formatting, NOT markdown. Bold with *single asterisks* (not **double**). Italics with _underscores_. Do not use # or other markdown formatting.",
        "",
        "KNOWLEDGE BASE:",
        knowledge,
    ]

    if is_first_response:
        lines.insert(
            1,
            "This is your first reply in this conversation. Begin with exactly: "
            f"'Hi, I'm Sally from {client_name}.' Do not repeat this introduction in later replies.",
        )

    if modules.get("booking"):
        booking = config.get("booking", {})
        services_text = "\n".join(
            f"- {s['name']}: ${s['price']:,} ({s['duration_minutes']} min)"
            for s in booking.get("services", [])
        )
        locations_text = "\n".join(
            f"- {loc['name']}: {loc['address']} ({', '.join(loc['days'])})"
            for loc in booking.get("locations", [])
        )
        hours = booking.get("business_hours", {})
        from datetime import date as date_cls, timedelta
        today = date_cls.today()
        day_names_en = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
                        4: "Friday", 5: "Saturday", 6: "Sunday"}
        today_str = f"{day_names_en[today.weekday()]} {today.isoformat()}"

        # Calculate next available dates based on booking days
        day_name_to_num = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                           "friday": 4, "saturday": 5, "sunday": 6}
        booking_days = set()
        for loc in booking.get("locations", []):
            for d in loc.get("days", []):
                booking_days.add(day_name_to_num.get(d.lower(), -1))
        next_dates = []
        check = today + timedelta(days=1)
        while len(next_dates) < 5:
            if check.weekday() in booking_days:
                next_dates.append(f"{day_names_en[check.weekday()]} {check.day}/{check.month} ({check.isoformat()})")
            check += timedelta(days=1)
        next_dates_str = ", ".join(next_dates)

        lines += [
            "",
            "BOOKINGS:",
            f"Today is {today_str}.",
            f"Next available dates for appointments: {next_dates_str}.",
            "Use ONLY these dates when proposing appointments. Do not make up other dates.",
            "When the user wants to book an appointment, you need:",
            "1. Type of service (from the services list)",
            "2. Date and time",
            "3. Full name",
            "",
            "IMPORTANT about the booking experience:",
            "- 'Tomorrow' means the day after today. 'Day after tomorrow' is two days later. ALWAYS resolve these expressions to the actual date.",
            "- If the user says something vague like 'tomorrow', 'Wednesday' or 'next week', YOU resolve the actual date and propose a time.",
            "  Example: if today is Tuesday 03/31, 'tomorrow' = Wednesday 04/01. Reply: 'Tomorrow Wednesday 04/01 at 10:00. Does that work for you?'",
            "- NEVER ask for multiple pieces of information at once. Ask ONE thing per message. If you already have the date, ask for the service. If you already have the service, ask for the name. Never numbered lists with multiple questions.",
            "- Try to complete the booking in as few messages as possible.",
            f"Available hours: {hours.get('start', '09:00')} to {hours.get('end', '18:00')}",
            f"Available services:\n{services_text}",
            f"Locations:\n{locations_text}",
            f"Cancellation policy: {booking.get('cancellation_policy', '')}",
            "",
            "Once you have all the data confirmed by the user, respond with a confirmation message",
            "AND at the end include this JSON on a single line (NO markdown, NO ```json, NO backticks):",
            '{"intent": "booking_confirmed", "service": "<EXACT service name from the list>", '
            '"date": "<YYYY-MM-DD>", "time": "<HH:MM>", "location": "<location name>", '
            '"user_name": "<full name>"}',
            "IMPORTANT: The 'service' field in the JSON MUST be the exact name from the services list. "
            "For example, if the user asks for the bundle, use 'Cleaning + Checkup (bundle)', NOT 'Dental Cleaning'.",
            "IMPORTANT: Do not mention payment, the system handles it automatically.",
            "IMPORTANT: The JSON goes in plain text at the end of the message, never inside code blocks.",
            "",
            "CANCELLATIONS:",
            "When the user wants to cancel an appointment, respond politely and at the end include:",
            '{"intent": "cancellation_request"}',
            "Do not ask for name or other details. The system searches for appointments automatically by phone number.",
            "If the system shows the appointments and the user confirms which one to cancel, respond with:",
            '{"intent": "cancellation_confirmed", "event_index": <appointment number>}',
            "If the user says 'yes' and there is only one appointment, use event_index: 1.",
            "",
            "MODIFICATIONS:",
            "When the user wants to change, modify, move or reschedule an appointment, ALWAYS respond with:",
            '{"intent": "modification_request"}',
            "IMPORTANT: ALWAYS emit modification_request first, even if the user already provides the new date in the same message. "
            "The system needs to look up the current appointment before proceeding. Do not try to handle the modification conversationally.",
            "Do not ask for name or extra details. The system searches by phone and shows the appointments.",
            "After the system shows the appointment, the user will tell you the new date/time.",
            "IMPORTANT: When the user confirms the new date/time for their modified appointment, emit a booking_confirmed JSON (NOT modification_confirmed). "
            "Use the same service as the original appointment. Example:",
            '{"intent": "booking_confirmed", "service": "Dental Cleaning", "date": "2026-04-03", "time": "15:00", "location": "Downtown Office", "user_name": "John Smith"}',
            "modification_confirmed is ONLY used when there are MULTIPLE appointments and the user chooses which one to modify:",
            '{"intent": "modification_confirmed", "event_index": <appointment number>}',
        ]

    return "\n".join(lines)


def extract_intent(response: str) -> tuple[dict | None, str]:
    """
    Look for any intent JSON block in Claude's response.
    Returns (intent_dict, visible_text) — strips the JSON from user-visible text.
    """
    pattern = re.compile(r'\{[^{}]*"intent"\s*:\s*"[^"]*"[^{}]*\}', re.DOTALL)
    match = pattern.search(response)
    if not match:
        return None, response
    try:
        intent = json.loads(match.group())
    except json.JSONDecodeError:
        return None, response
    visible = response[:match.start()].strip()
    # Remove any trailing markdown code fences that Claude might add
    visible = re.sub(r'```\s*json\s*$', '', visible, flags=re.MULTILINE).strip()
    visible = re.sub(r'```\s*$', '', visible, flags=re.MULTILINE).strip()
    return intent, visible


# Keep backward compatibility
extract_booking_intent = extract_intent


def get_ai_response(
    user_message: str,
    history: list[dict],
    config: dict,
    knowledge: str,
) -> str:
    is_multi_part_query = _is_multi_part_query(user_message)

    if is_multi_part_query:
        multi_part_answer = _multi_part_clinic_answer(
            user_message,
            history,
            knowledge,
        )
        if multi_part_answer:
            return _ensure_requested_next_steps(
                multi_part_answer,
                user_message,
                knowledge,
                None,
                history,
            )

    evaluation_report_answer = _evaluation_report_answer(user_message, history, knowledge)
    if evaluation_report_answer and not is_multi_part_query:
        return _ensure_requested_next_steps(
            evaluation_report_answer,
            user_message,
            knowledge,
            None,
            history,
        )

    occupational_therapy_answer = _occupational_therapy_answer(user_message, knowledge)
    if occupational_therapy_answer and not is_multi_part_query:
        return _ensure_requested_next_steps(
            occupational_therapy_answer,
            user_message,
            knowledge,
            None,
            history,
        )

    weekend_appointment_answer = _weekend_appointment_answer(user_message, history, knowledge)
    if weekend_appointment_answer and not is_multi_part_query:
        return _ensure_requested_next_steps(
            weekend_appointment_answer,
            user_message,
            knowledge,
            None,
            history,
        )

    slot_availability_answer = _slot_availability_answer(user_message, knowledge)
    if slot_availability_answer and not is_multi_part_query:
        return _ensure_requested_next_steps(
            slot_availability_answer,
            user_message,
            knowledge,
            None,
            history,
        )

    requested_consultant = _requested_consultant(user_message, knowledge)
    fee_answer = _fee_question_answer(user_message, knowledge)
    if fee_answer and not is_multi_part_query and not (
        requested_consultant or _needs_appointment_details(user_message)
    ):
        return _ensure_requested_next_steps(
            fee_answer,
            user_message,
            knowledge,
            None,
            history,
        )

    immediate_appointment_answer = _immediate_appointment_answer(user_message, config)
    if immediate_appointment_answer and not is_multi_part_query:
        return _ensure_requested_next_steps(
            immediate_appointment_answer,
            user_message,
            knowledge,
            None,
            history,
        )

    is_first_response = not any(message.get("role") == "assistant" for message in history)
    system_prompt = build_system_prompt(
        config,
        knowledge,
        is_first_response=is_first_response,
    )
    messages = history + [{"role": "user", "content": user_message}]
    resp = get_client().messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=messages,
    )
    answer = next(
        (
            block.text
            for block in resp.content
            if getattr(block, "type", "text") == "text" and block.text.strip()
        ),
        "",
    ).strip()
    if not answer or answer.casefold() in {"null", "none"}:
        logger.warning("Model returned an empty/null response; using a safe fallback")
        return _safe_fallback_answer(user_message, knowledge, requested_consultant, history)
    return _ensure_requested_next_steps(
        answer,
        user_message,
        knowledge,
        requested_consultant,
        history,
    )
