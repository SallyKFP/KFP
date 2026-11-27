import json
from datetime import datetime

import pytest
import pytz
from types import SimpleNamespace

from core.ai import (
    MODEL,
    _fee_question_answer,
    _immediate_appointment_answer,
    _slot_availability_answer,
    extract_booking_intent,
    build_system_prompt,
    get_ai_response,
)


def test_configured_model_is_claude_haiku_5_5():
    assert MODEL == "claude-haiku-5-5"


def test_extract_booking_intent_present():
    response = 'Great, your appointment is confirmed. {"intent": "booking_confirmed", "service": "Dental Cleaning", "date": "2026-03-17", "time": "10:00", "location": "Downtown Office", "user_name": "Jane"}'
    intent, visible = extract_booking_intent(response)
    assert intent is not None
    assert intent["intent"] == "booking_confirmed"
    assert intent["service"] == "Dental Cleaning"
    assert "intent" not in visible
    assert "Great" in visible


def test_extract_booking_intent_absent():
    response = "Hi, how can I help you?"
    intent, visible = extract_booking_intent(response)
    assert intent is None
    assert visible == response


def test_build_system_prompt_core_only():
    config = {
        "client": {"name": "Test Dentist"},
        "modules": {"booking": False, "payments": False},
    }
    knowledge = "Services: Dental Cleaning $150"
    prompt = build_system_prompt(config, knowledge)
    assert "Test Dentist" in prompt
    assert "Services: Dental Cleaning" in prompt
    assert "booking_confirmed" not in prompt  # booking disabled


def test_booking_prompt_asks_consultant_preference_and_allows_internal_allocation():
    prompt = build_system_prompt(
        {
            "client": {"name": "Test Clinic", "timezone": "Asia/Singapore"},
            "modules": {"booking": True},
            "booking": {
                "consultants": [
                    {"name": "Consultant One"},
                    {"name": "Consultant Two"},
                ],
                "services": [],
                "locations": [],
            },
        },
        "",
        telegram_live_availability=True,
    )

    assert "Ask the patient whether they prefer either consultant or have no preference" in prompt
    assert "do not assume that silence means no preference" in prompt
    assert "Do you have a consultant preference, or no preference?" in prompt
    assert "Consultant One, Consultant Two" in prompt
    assert '"consultant": "<EXACT consultant name when explicitly preferred or already assigned>"' in prompt
    assert "A stated 'no preference' allows the system to assign an available consultant internally" in prompt
    assert "without disclosing the internal allocation split" in prompt


def test_build_system_prompt_encourages_natural_grounded_answers():
    prompt = build_system_prompt(
        {"client": {"name": "Kids Focus Physiotherapy"}, "modules": {}},
        "FAQ: HUMAN REVIEW REQUIRED: YES",
    )

    assert "Current local date and time:" in prompt
    assert "compare the current local date and time above with the opening-hours FAQ" in prompt
    assert "Do not answer 'NULL'." in prompt
    assert "do not recite or dump it" in prompt
    assert "Do not include clinic phone numbers in replies" in prompt
    assert "provide the clinic email address(es)" in prompt
    assert "Answer the user's specific question first" in prompt
    assert "answer every distinct part in the same reply" in prompt
    assert "ask for all required details together" in prompt
    assert "For appointment booking, follow the BOOKINGS workflow" not in prompt
    assert "Ask at most one focused follow-up question" not in prompt
    assert "When a caregiver asks about fees or charges" in prompt
    assert "follow a matching referral-and-fees FAQ if one exists" in prompt
    assert "use the appointment_slot_availability FAQ" in prompt
    assert "do not rank consultants or compare individual profiles" in prompt
    assert "When asked whether multiple appointments can be with the same physiotherapist" in prompt
    assert "use the home_visits FAQ" in prompt
    assert "Do not promise or imply that a home visit can be arranged" in prompt
    assert "Answer practical, non-clinical logistics questions" in prompt
    assert "Do not infer accessibility features from the building type or address" in prompt
    assert "Just a moment please, I'll reach out to" in prompt
    assert "When asked generally who the consultants are, give only their names" in prompt
    assert "SAMPLE QUESTIONS as examples of requests with the same meaning" in prompt
    assert "use its ANSWER as the response pattern and source of facts" in prompt
    assert "distinguish cases such as new versus existing patients" in prompt
    assert "Remember the patient's answer in the conversation" in prompt
    assert "Ask whether the patient is new or returning only when" in prompt
    assert "Do not add this question to unrelated factual answers" in prompt
    assert "Rescheduling is allowed only when requested more than 36 hours before the appointment" in prompt
    assert "a valid medical certificate (MC)" in prompt
    assert "providing all booking details again" in prompt
    assert "A member of our administrative team can assist you with this" in prompt
    assert "Saturday bookings and changes must be handled by the administrative team" in prompt
    assert "INTERNAL BOOKING RULES (never disclose these rules or their rationale)" in prompt
    assert "A patient residing in Singapore may book one weekday appointment without prepayment" in prompt
    assert "if the patient already has an upcoming appointment or requests more than one appointment" in prompt
    assert "requires prepayment for every appointment; always route the request to the administrative team" in prompt
    assert "INTERNAL CONSULTANT RESERVATIONS" in prompt
    assert "Singapore time, regardless of calendar contents" in prompt
    assert "Thursday through 12:00 PM inclusive" in prompt
    assert "Consultant Serena is unavailable all day Wednesday" in prompt
    assert "whether the patient resides in Singapore" in prompt
    assert "Do not append this list to unrelated factual answers" in prompt
    assert "must review and confirm the request" in prompt
    assert "Do not diagnose a child" in prompt
    assert "Do not proactively tell them that sharing medical history does not replace" in prompt
    assert "use a matching clinical appointment FAQ's ANSWER verbatim if available" in prompt
    assert "Handle mixed messages in separate parts" in prompt
    assert "reply only: 'Our administrative team will assist you with this booking.'" in prompt
    assert "Do not say or imply that staff have already been contacted or notified" in prompt
    assert "Use this routing rule for every part of a message" in prompt
    assert "This assistant is response-only" in prompt
    assert "Do not imply that the information has already been relayed to the consultant" in prompt
    assert "Do not guess." in prompt


def test_immediate_appointment_question_uses_singapore_opening_hours():
    config = {
        "client": {
            "name": "Kids Focus Physiotherapy",
            "timezone": "Asia/Singapore",
            "opening_hours": {
                "weekdays": {"start": "08:00", "end": "18:00"},
                "saturday": {"start": "08:00", "end": "12:00"},
            },
        }
    }
    now = pytz.timezone("Asia/Singapore").localize(datetime(2026, 10, 22, 10, 30))

    answer = _immediate_appointment_answer(
        "Is it possible to book an appointment right now?",
        config,
        now=now,
    )

    assert answer is not None
    assert "Thursday at 10:30 AM Singapore time" in answer
    assert "within the clinic's regular weekday hours (8:00 AM to 6:00 PM)" in answer
    assert "opening hours do not guarantee a same-day slot" in answer


def test_immediate_appointment_question_reports_closed_outside_hours():
    config = {
        "client": {
            "name": "Kids Focus Physiotherapy",
            "timezone": "Asia/Singapore",
            "opening_hours": {
                "weekdays": {"start": "08:00", "end": "18:00"},
                "saturday": {"start": "08:00", "end": "12:00"},
            },
        }
    }
    now = pytz.timezone("Asia/Singapore").localize(datetime(2026, 10, 22, 22, 30))

    answer = _immediate_appointment_answer(
        "Can I book an appointment right now?",
        config,
        now=now,
    )

    assert answer is not None
    assert "Thursday at 10:30 PM Singapore time" in answer
    assert "outside the clinic's regular weekday hours" in answer


def test_get_ai_response_answers_immediate_appointment_without_calling_model(monkeypatch):
    config = {
        "client": {
            "name": "Kids Focus Physiotherapy",
            "timezone": "Asia/Singapore",
            "opening_hours": {
                "weekdays": {"start": "08:00", "end": "18:00"},
                "saturday": {"start": "08:00", "end": "12:00"},
            },
        }
    }
    monkeypatch.setattr(
        "core.ai.get_client",
        lambda: pytest.fail("Immediate appointment query should not call the LLM"),
    )

    answer = get_ai_response(
        "Is it possible to book an appointment right now?",
        [],
        config,
        "",
    )

    assert answer
    assert "Singapore time" in answer
    assert "same-day slot" in answer


def test_home_visit_request_gets_direct_policy_without_contacting_consultant(monkeypatch):
    knowledge = (
        "CONSULTANT INFORMATION\n"
        "Sarah Wong, Chief Paediatric Physiotherapist\n"
        "FAQ ID: home_visits\n"
        "ANSWER: Thank you for asking. We’re unable to offer home visits, as all appointments "
        "take place at our clinic, "
        "which has purpose-built facilities for paediatric physiotherapy.\n"
        "HUMAN REVIEW REQUIRED: NO\n"
    )
    monkeypatch.setattr(
        "core.ai.get_client",
        lambda: pytest.fail("A home-visit policy answer should not call the LLM"),
    )

    answer = get_ai_response(
        "Can Sarah Wong provide a home visit for my child?",
        [],
        {"client": {"name": "Kids Focus Physiotherapy"}},
        knowledge,
    )

    assert answer == (
        "Thank you for asking. We’re unable to offer home visits, as all appointments "
        "take place at our clinic, "
        "which has purpose-built facilities for paediatric physiotherapy."
    )
    assert "reach out" not in answer
    assert "administrative team" not in answer
    assert "@" not in answer


def test_fee_question_with_referral_uses_llm_then_adds_required_details(monkeypatch):
    config = {"client": {"name": "Kids Focus Physiotherapy", "timezone": "Asia/Singapore"}}
    knowledge = (
        "CONSULTANT INFORMATION\n"
        "Sarah Wong, Chief Paediatric Physiotherapist\n"
        "Serena Kwa, Consultant Physiotherapist\n"
        "Diana Ng, Consultant Physiotherapist\n"
        "DAYS, LOCATIONS AND CONTACT INFORMATION\n"
        "FAQ ID: appointment_fees\n"
        "ANSWER: The initial evaluation is SGD 300. "
        "A follow-up intervention session is SGD 280.\n"
        "HUMAN REVIEW REQUIRED: NO\n"
        "FAQ ID: make_appointment\n"
        "ANSWER: Please provide the patient's full name and date of birth, your full name, "
        "your contact number, and the referring person's name, if any. The team will review "
        "and confirm your request.\n"
    )
    prompts = []

    class FakeMessages:
        def create(self, **kwargs):
            prompts.append(kwargs["system"])
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="The initial evaluation is SGD 300.")])

    monkeypatch.setattr("core.ai.get_client", lambda: SimpleNamespace(messages=FakeMessages()))

    answer = get_ai_response(
        "Hello! I have a referral letter to see Serena Kwa for my one month old baby girl "
        "- torticollis and plagiocephaly intervention. May I enquire on the charges please?",
        [],
        config,
        knowledge,
    )

    assert len(prompts) == 1
    assert "fee question using the appointment_fees FAQ" in prompts[0]
    assert "The initial evaluation is SGD 300." in answer
    assert "patient's full name and date of birth" in answer
    assert "Just a moment please, I'll reach out to Serena Kwa." in answer


def test_null_model_response_uses_fees_details_and_consultant_fallback(monkeypatch):
    config = {"client": {"name": "Kids Focus Physiotherapy", "timezone": "Asia/Singapore"}}
    knowledge = (
        "CONSULTANT INFORMATION\nSerena Kwa, Consultant Physiotherapist\n"
        "DAYS, LOCATIONS AND CONTACT INFORMATION\n"
        "FAQ ID: appointment_fees\nANSWER: Initial evaluation: SGD 300.\n"
        "FAQ ID: make_appointment\n"
        "ANSWER: Please provide the patient's full name and date of birth, your full name, "
        "your contact number, and referring person's name, if any.\n"
    )

    class FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="Null")])

    monkeypatch.setattr("core.ai.get_client", lambda: SimpleNamespace(messages=FakeMessages()))

    answer = get_ai_response(
        "I have a referral to see Serena Kwa. My baby has torticollis; what are the charges?",
        [],
        config,
        knowledge,
    )

    assert "Initial evaluation: SGD 300." in answer
    assert "patient's full name and date of birth" in answer
    assert "Just a moment please, I'll reach out to Serena Kwa." in answer


@pytest.mark.parametrize("model_text", [
    "Could you double-check Lily's date of birth? The date you gave, 10 Oct 2026, is later than today's date.",
    "Lily's birth date is in the future.",
    "Your baby hasn't been born yet.",
    "The DOB is invalid, so please provide it again.",
])
def test_model_must_not_claim_birth_date_is_in_future(monkeypatch, model_text):
    class FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=model_text)])

    monkeypatch.setattr("core.ai.get_client", lambda: SimpleNamespace(messages=FakeMessages()))
    answer = get_ai_response(
        "1. Initial evaluation\n2. Next Thursday morning\n3. Lily Tan; 10 Oct 2026",
        [],
        {"client": {"name": "Clinic", "timezone": "Asia/Singapore"},
         "modules": {"booking": True}, "booking": {}},
        "",
    )
    assert answer != model_text
    assert "remaining booking details" in answer


def test_model_dob_judgement_using_conversation_context_is_suppressed(monkeypatch):
    messages = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(
        content=[SimpleNamespace(type="text", text="The date you gave is later than today.")]
    ))
    monkeypatch.setattr("core.ai.get_client", lambda: SimpleNamespace(messages=messages))
    answer = get_ai_response(
        "10 Oct 2026", [{"role": "assistant", "content": "What is your child's date of birth?"}],
        {"client": {"name": "Clinic"}, "modules": {"booking": True}, "booking": {}}, "",
    )
    assert "later than today" not in answer


def test_fee_question_does_not_use_appointment_fee_for_report_or_cancellation():
    knowledge = (
        "FAQ ID: appointment_fees\n"
        "ANSWER: The initial evaluation is SGD 300.\n"
        "FAQ ID: report_fees\n"
        "ANSWER: Please contact the clinic about physiotherapy report fees.\n"
    )

    assert _fee_question_answer("What are the cancellation charges?", knowledge) is None
    assert _fee_question_answer("What is the report fee?", knowledge) is None


@pytest.mark.parametrize(
    "user_message",
    [
        "My child's doctor referred us. What is the evaluation report fee?",
        "My child is a new patient. What is the evaluation report fee?",
    ],
)
def test_fast_report_answer_includes_appointment_details_for_referral_or_new_patient(user_message):
    from core.ai import load_knowledge

    answer = get_ai_response(
        user_message,
        [],
        {"client": {"name": "Kids Focus Physiotherapy"}},
        load_knowledge(),
    )

    assert "SGD 250 to SGD 800 per report" in answer
    assert "May I check whether your child is a new patient" not in answer
    assert "To schedule an appointment, please provide:" in answer
    assert "- The patient's full name and date of birth" in answer
    assert "- The referring person's name, or confirm that there is no referral" in answer
    assert "- Your preferred date and time" in answer


def test_long_condition_intro_with_consultant_request_includes_appointment_details(monkeypatch):
    from core.ai import load_knowledge

    class FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="The consultant can assess your child.")],
            )

    monkeypatch.setattr(
        "core.ai.get_client",
        lambda: SimpleNamespace(messages=FakeMessages()),
    )

    answer = get_ai_response(
        "My daughter has had difficulty standing and walking since she was very young, "
        "and her doctors have noticed delays with her movement, balance, strength, and "
        "coordination at home and school. We would like to see a consultant.",
        [],
        {"client": {"name": "Kids Focus Physiotherapy"}},
        load_knowledge(),
    )

    assert "The consultant can assess your child." in answer
    assert "To schedule an appointment, please provide:" in answer
    assert answer.count("To schedule an appointment, please provide:") == 1


def test_general_patient_query_does_not_clarify_patient_type(monkeypatch):
    from core.ai import load_knowledge

    class FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="Our clinic is open today.")],
            )

    monkeypatch.setattr(
        "core.ai.get_client",
        lambda: SimpleNamespace(messages=FakeMessages()),
    )
    answer = get_ai_response(
        "What are your opening hours?",
        [],
        {"client": {"name": "Kids Focus Physiotherapy"}},
        load_knowledge(),
    )

    assert "Our clinic is open" in answer
    assert "May I check whether your child is a new patient" not in answer


def test_known_patient_type_in_history_suppresses_clarification(monkeypatch):
    from core.ai import load_knowledge

    class FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="Our clinic is open today.")],
            )

    monkeypatch.setattr(
        "core.ai.get_client",
        lambda: SimpleNamespace(messages=FakeMessages()),
    )

    answer = get_ai_response(
        "What are your opening hours?",
        [
            {"role": "user", "content": "We are existing patients; my child has been seen here before."},
            {"role": "assistant", "content": "How can I help?"},
        ],
        {"client": {"name": "Kids Focus Physiotherapy"}},
        load_knowledge(),
    )

    assert "Our clinic is open" in answer
    assert "May I check whether your child is a new patient" not in answer


@pytest.mark.parametrize(
    ("user_message", "history"),
    [
        ("My child is a new patient. Are you open today?", []),
        ("I am an old patient. Are you open today?", []),
        (
            "Are you open today?",
            [{"role": "user", "content": "We are existing patients."}],
        ),
    ],
)
def test_model_patient_type_question_is_removed_when_status_is_known(
    monkeypatch,
    user_message,
    history,
):
    class FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="text",
                        text=(
                            "Our clinic is open on weekdays. "
                            "May I check whether your child is a new patient or has "
                            "visited Kids Focus Physiotherapy before?"
                        ),
                    )
                ],
            )

    monkeypatch.setattr(
        "core.ai.get_client",
        lambda: SimpleNamespace(messages=FakeMessages()),
    )

    answer = get_ai_response(
        user_message,
        history,
        {"client": {"name": "Kids Focus Physiotherapy"}},
        "No matching FAQ.",
    )

    assert "Our clinic is open on weekdays." in answer
    assert "May I check whether your child is a new patient" not in answer


def test_model_response_does_not_add_patient_type_question_to_unrelated_answer(monkeypatch):
    class FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="We are open on weekdays.")],
            )

    monkeypatch.setattr(
        "core.ai.get_client",
        lambda: SimpleNamespace(messages=FakeMessages()),
    )

    answer = get_ai_response(
        "Are you open today?",
        [],
        {"client": {"name": "Kids Focus Physiotherapy"}},
        "No matching FAQ.",
    )

    assert "We are open on weekdays." in answer
    assert "May I check whether your child is a new patient" not in answer


def test_new_patient_fee_response_includes_appointment_details_once(monkeypatch):
    from core.ai import load_knowledge

    monkeypatch.setattr(
        "core.ai.get_client",
        lambda: pytest.fail("Standalone fee answer should not call the LLM"),
    )

    answer = get_ai_response(
        "We are new patients. What are your physiotherapy fees?",
        [],
        {"client": {"name": "Kids Focus Physiotherapy"}},
        load_knowledge(),
    )

    assert "initial evaluation is SGD 300" in answer
    assert answer.count("To schedule an appointment, please provide:") == 1


def test_evaluation_report_request_uses_report_fees_and_timing_without_model(monkeypatch):
    from core.ai import load_knowledge

    monkeypatch.setattr(
        "core.ai.get_client",
        lambda: pytest.fail("Evaluation report FAQ should not call the LLM"),
    )

    answer = get_ai_response(
        "Could I request an evaluation report for my child?",
        [],
        {"client": {"name": "Kids Focus Physiotherapy"}},
        load_knowledge(),
    )

    assert answer.startswith(
        "Our evaluation report fees ranged from SGD 250 to SGD 800 per report. "
        "I will need to confirm the exact report fees that apply to your child with "
        "our Consultant. After fee payment, you will receive the report within 4 weeks."
    )
    assert "May I check whether your child is a new patient" not in answer


@pytest.mark.parametrize("follow_up", ["How long will it take?", "What are the charges?"])
def test_evaluation_report_follow_up_uses_conversation_context(monkeypatch, follow_up):
    from core.ai import load_knowledge

    monkeypatch.setattr(
        "core.ai.get_client",
        lambda: pytest.fail("Evaluation report follow-up should not call the LLM"),
    )

    answer = get_ai_response(
        follow_up,
        [
            {"role": "user", "content": "Can I request an evaluation report?"},
            {
                "role": "assistant",
                "content": "Our evaluation report fees ranged from SGD 250 to SGD 800 per report.",
            },
        ],
        {"client": {"name": "Kids Focus Physiotherapy"}},
        load_knowledge(),
    )

    assert "After fee payment, you will receive the report within 4 weeks." in answer


def test_occupational_therapy_request_uses_clinic_referral_answer_without_model(monkeypatch):
    from core.ai import load_knowledge

    monkeypatch.setattr(
        "core.ai.get_client",
        lambda: pytest.fail("Occupational therapy FAQ should not call the LLM"),
    )

    answer = get_ai_response(
        "Do you have an OT for my child?",
        [],
        {"client": {"name": "Kids Focus Physiotherapy"}},
        load_knowledge(),
    )

    assert "does not have any OTs" in answer
    assert "All new patients are evaluated comprehensively." in answer
    assert "refer patients to other allied health professionals" in answer


def test_weekend_appointment_asks_patient_type_before_routing(monkeypatch):
    monkeypatch.setattr(
        "core.ai.get_client",
        lambda: pytest.fail("Patient type should be clarified before routing"),
    )

    answer = get_ai_response(
        "Can I book an appointment this weekend?",
        [],
        {"client": {"name": "Kids Focus Physiotherapy"}},
        "FAQ ID: new_patient_saturday\nANSWER: New patient policy.\n"
        "FAQ ID: existing_patient_saturday\nANSWER: Existing patient policy.\n",
    )

    assert answer == (
        "Is your child a new patient, or have they visited Kids Focus Physiotherapy before?"
    )


def test_asking_about_new_patients_does_not_assume_patient_type(monkeypatch):
    monkeypatch.setattr(
        "core.ai.get_client",
        lambda: pytest.fail("Patient status should be clarified before routing"),
    )

    answer = get_ai_response(
        "Can new patients book on Saturday?",
        [],
        {"client": {"name": "Kids Focus Physiotherapy"}},
        "FAQ ID: new_patient_saturday\nANSWER: New patient policy.\n"
        "FAQ ID: existing_patient_saturday\nANSWER: Existing patient policy.\n",
    )

    assert answer == (
        "Is your child a new patient, or have they visited Kids Focus Physiotherapy before?"
    )


@pytest.mark.parametrize(
    ("current_message", "history", "expected"),
    [
        (
            "My child is a new patient. Can we book Saturday?",
            [],
            "New patient policy.",
        ),
        (
            "Hello. I am the parent of a new patient. I would like a saturday appointment with one of your consultants.",
            [],
            "New patient policy.",
        ),
        (
            "We are returning patients.",
            [
                {"role": "user", "content": "Can I book an appointment this weekend?"},
                {
                    "role": "assistant",
                    "content": "Is your child a new patient, or have they visited Kids Focus Physiotherapy before?",
                },
            ],
            "Existing patient policy.",
        ),
        (
            "My child was at your clinic six months ago. Can we book Saturday?",
            [],
            "Existing patient policy.",
        ),
        (
            "We visited six months ago. Can we book Saturday?",
            [],
            "Existing patient policy.",
        ),
        (
            "new",
            [
                {"role": "user", "content": "Can I book an appointment this weekend?"},
                {
                    "role": "assistant",
                    "content": "Is your child a new patient, or have they visited Kids Focus Physiotherapy before?",
                },
            ],
            "New patient policy.",
        ),
    ],
)
def test_weekend_appointment_routes_using_current_or_remembered_patient_type(
    monkeypatch,
    current_message,
    history,
    expected,
):
    monkeypatch.setattr(
        "core.ai.get_client",
        lambda: pytest.fail("Known patient type should route directly to its policy"),
    )

    answer = get_ai_response(
        current_message,
        history,
        {"client": {"name": "Kids Focus Physiotherapy"}},
        "FAQ ID: new_patient_saturday\nANSWER: New patient policy.\n"
        "FAQ ID: existing_patient_saturday\nANSWER: Existing patient policy.\n",
    )

    assert answer == expected


def test_multipart_saturday_request_answers_each_part_from_knowledge(monkeypatch):
    from core.ai import load_knowledge

    monkeypatch.setattr(
        "core.ai.get_client",
        lambda: pytest.fail("The multipart answer should use the verified FAQs directly"),
    )

    answer = get_ai_response(
        "Hello. I am the parent of a new patient. I would like a saturday appointment "
        "with one of your consultants. Which one of your consultants is the best with "
        "children? What are the fees for the initial consultant and each subsequent "
        "evaluation? What are the payment options? Can I pay using my child's CDAC "
        "account? How do I get to the clinic by car? What time do I have to be there?",
        [],
        {"client": {"name": "Kids Focus Physiotherapy"}},
        load_knowledge(),
    )

    assert "Saturday appointments are not available to first-time patients" in answer
    assert "initial evaluation must be booked on a weekday" in answer
    assert "Monday to Friday: 8:00 AM to 6:00 PM" in answer
    assert "Saturday: 8:00 AM to 12:00 noon" in answer
    assert "Our consultants are all suitable to help your child" in answer
    assert "SGD 300" in answer
    assert "CDA account, also known as Baby Bonus" in answer
    assert "cash or PayNow" in answer
    assert "38 Stevens Road" in answer
    assert "Turn left into Orange Grove road" in answer
    assert "arrive at least 10 minutes early" in answer
    assert "To schedule an appointment, please provide:" in answer


def test_multipart_clinical_and_time_request_includes_human_review_notes():
    from core.ai import load_knowledge

    answer = get_ai_response(
        "My child has muscle stiffness and symptoms that concern us. What are the fees? "
        "What payment options do you accept? We are free Tuesday at 3 pm for an appointment.",
        [],
        {"client": {"name": "Kids Focus Physiotherapy"}},
        load_knowledge(),
    )

    assert "SGD 300" in answer
    assert "cash or PayNow" in answer
    assert "Thank you for sharing your child's background." in answer
    assert "A consultant would need to review these details" in answer
    assert "I can't provide medical advice over chat." in answer
    assert "cannot be skipped" not in answer
    assert "To schedule an appointment, please provide:" in answer
    assert "Thank you for sharing your preferred timing." in answer
    assert "Our administrative team can assist you with this booking." in answer
    assert "available options" not in answer
    assert "appointment is confirmed only when" not in answer


def test_assessment_booking_does_not_repeat_intake_or_add_clinical_handoff():
    from core.ai import _ensure_requested_next_steps, load_knowledge

    user_message = (
        "I need an initial evaluation Thursday at 2pm for Khai Wong "
        "(24 November 2002). Sarah Wong. No referral."
    )
    answer = (
        "The initial evaluation is SGD 300. "
        "To book, please share:\n"
        "- Your preferred date and time\n"
        "- The patient's full name and date of birth\n"
        "- Your full name as the parent or guardian\n"
        "- The referring person's name, or confirm there is no referral"
    )

    result = _ensure_requested_next_steps(
        answer,
        user_message,
        load_knowledge(),
        consultant=None,
        history=[],
        booking_enabled=True,
    )

    assert result.count("To book, please share:") == 1
    assert "To schedule an appointment, please provide:" not in result
    assert "A consultant would need to review" not in result
    assert "I can't provide medical advice over chat" not in result


def test_booking_summary_does_not_append_intake_faq():
    from core.ai import _ensure_requested_next_steps, load_knowledge

    answer = (
        "I'm checking the slot for Friday at 10:00 AM with Consultant Sarah now.\n\n"
        "Please review the details below:\n"
        "- Service: Initial evaluation\n"
        "- Date and time: Friday, 6 November 2026, 10:00 AM\n"
        "- Patient: YK Wong, date of birth 24 November 2020\n"
        "- Parent: TY Wong\n"
        "- Residency: Singapore\n"
        "- Referral: None"
    )
    result = _ensure_requested_next_steps(
        answer,
        "Initial evaluation Friday at 10am for YK Wong, 24 November 2020. "
        "Parent TY Wong. Residence in Singapore. No referring doctor.",
        load_knowledge(),
        consultant=None,
        history=[],
        booking_enabled=True,
    )

    assert result == answer
    assert "To schedule an appointment, please provide:" not in result


def test_booking_summary_removes_repeated_checklist_after_consultant_question():
    from core.ai import _ensure_requested_next_steps, load_knowledge

    answer = (
        "Thank you, I have most of the details for your child's initial evaluation:\n\n"
        "- Service: Initial evaluation (SGD 300)\n"
        "- Date and time: Friday 6 November 2026 at 10:00 AM\n"
        "- Patient's full name: YK Wong\n"
        "- Patient's date of birth: 24 November 2020\n"
        "- Parent's full name: TY Wong\n"
        "- Resides in Singapore: Yes\n"
        "- Referral: No referring doctor\n\n"
        "Which consultant would you prefer, Consultant Sarah or Consultant Serena?\n\n"
        "To schedule an appointment, please provide:\n"
        "- The service you would like to book\n"
        "- Your preferred date and time\n"
        "- The patient's full name and date of birth\n"
        "- Your full name as the parent or guardian\n"
        "- Whether the patient resides in Singapore\n"
        "- The referring person's name, or confirm that there is no referral\n\n"
        "Our administrative team can assist you with your appointment request."
    )
    result = _ensure_requested_next_steps(
        answer,
        "Initial evaluation this Friday at 10am, YK Wong, 24 November 2020, "
        "TY Wong, Singapore resident, no referring doctor.",
        load_knowledge(),
        consultant=None,
        history=[],
        booking_enabled=True,
    )

    assert "Which consultant would you prefer" in result
    assert "To schedule an appointment, please provide:" not in result
    assert "Patient's full name: YK Wong" in result


def test_followup_with_booking_details_does_not_repeat_requested_checklist():
    from core.ai import _ensure_requested_next_steps, load_knowledge

    history = [{
        "role": "assistant",
        "content": (
            "To get started, please let me know:\n"
            "1. The service you would like to book\n"
            "2. Your preferred date and time\n"
            "3. The patient's full name and date of birth\n"
            "4. Your full name as the parent or guardian\n"
            "5. Whether the patient resides in Singapore\n"
            "6. The referring doctor's name, or confirm there is no referral"
        ),
    }, {
        "role": "user",
        "content": "Which of the two consultants is more experienced?",
    }, {
        "role": "assistant",
        "content": (
            "Both consultants are experienced in various areas of paediatric "
            "physiotherapy. When ready, please let me know which consultant you "
            "prefer, along with the appointment details listed earlier."
        ),
    }]
    result = _ensure_requested_next_steps(
        "Thank you for sharing those details. Would you like me to proceed?",
        "Initial evaluation, this Friday at 10am. YK Wong, 24 November 2020. "
        "TY Wong. Residence in Singapore. No referring doctor.",
        load_knowledge(),
        consultant=None,
        history=history,
        booking_enabled=True,
    )

    assert result == "Thank you for sharing those details. Would you like me to proceed?"
    assert "To schedule an appointment, please provide:" not in result


def test_multipart_canned_answers_do_not_hide_an_unhandled_question(monkeypatch):
    class FakeMessages:
        def create(self, **kwargs):
            assert "Where is the clinic?" in kwargs["messages"][-1]["content"]
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="text",
                        text=(
                            "The initial evaluation is SGD 300. We accept cash or PayNow. "
                            "The clinic is at 30 Orange Grove Road, Singapore."
                        ),
                    ),
                ],
            )

    monkeypatch.setattr(
        "core.ai.get_client",
        lambda: SimpleNamespace(messages=FakeMessages()),
    )

    answer = get_ai_response(
        "What are the fees? What payment methods do you accept? Where is the clinic?",
        [],
        {"client": {"name": "Kids Focus Physiotherapy"}},
        (
            "FAQ ID: appointment_fees\n"
            "ANSWER: The initial evaluation is SGD 300.\n"
            "FAQ ID: cda_payment\n"
            "ANSWER: We accept cash or PayNow.\n"
            "FAQ ID: clinic_location\n"
            "ANSWER: The clinic is at 30 Orange Grove Road, Singapore.\n"
        ),
    )

    assert "The initial evaluation is SGD 300." in answer
    assert "We accept cash or PayNow." in answer
    assert "The clinic is at 30 Orange Grove Road, Singapore." in answer


def test_multipart_experience_and_same_physiotherapist_queries_are_answered():
    from core.ai import load_knowledge

    answer = get_ai_response(
        "Which physiotherapist has the most relevant experience treating infants with "
        "torticollis and fibromatosis colli? Could all three appointments be with the "
        "same physiotherapist? What are the fees for the initial and follow-up sessions?",
        [],
        {"client": {"name": "Kids Focus Physiotherapy"}},
        load_knowledge(),
    )

    assert (
        "All of our consultants are experienced in various areas of paediatric "
        "physiotherapy and will be able to help your child."
    ) in answer
    assert "Serena Kwa's profile" not in answer
    assert "Diana Ng's profile" not in answer
    assert "fibromatosis colli" not in answer
    assert "You can request the same physiotherapist for all your visits, subject to availability." in answer
    assert "you can choose another physiotherapist or wait for your preferred one" in answer
    assert "SGD 300" in answer
    assert "SGD 280" in answer
    assert "Paediatric experience:" not in answer
    assert "Same physiotherapist:" not in answer
    assert "Fees:" not in answer


def test_initial_evaluation_cannot_be_skipped_and_appointment_details_are_requested(monkeypatch):
    class FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="We can arrange follow-up care.")],
            )

    monkeypatch.setattr(
        "core.ai.get_client",
        lambda: SimpleNamespace(messages=FakeMessages()),
    )

    answer = get_ai_response(
        "We have provided the medical history already. Can we skip the initial evaluation?",
        [],
        {"client": {"name": "Kids Focus Physiotherapy"}},
        (
            "FAQ ID: make_appointment\n"
            "ANSWER: To schedule an appointment, please provide:\n"
            "- The patient's full name and date of birth\n"
            "- Your full name\n"
            "- Your WhatsApp contact number\n"
            "- The referring person's name, if any\n"
            "- Your preferred day and time\n"
        ),
    )

    assert "initial evaluation is required for new patients before follow-up care" in answer
    assert "Sharing medical history or records in advance does not replace the evaluation." not in answer
    assert "We can arrange follow-up care." not in answer
    assert "To schedule an appointment, please provide:" in answer


def test_model_response_adds_handoff_notes_to_mixed_clinical_and_booking_request(monkeypatch):
    class FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="Our clinic is open on weekdays.")],
            )

    monkeypatch.setattr(
        "core.ai.get_client",
        lambda: SimpleNamespace(messages=FakeMessages()),
    )

    answer = get_ai_response(
        "My child has been having unusual stiffness. What are your opening hours? "
        "We would prefer Tuesday at 3 pm for an appointment.",
        [],
        {"client": {"name": "Kids Focus Physiotherapy"}},
        "FAQ ID: clinic_opening_hours\n"
        "ANSWER: Our clinic is open from 8:00 AM to 6:00 PM on weekdays.\n",
    )

    assert "Our clinic is open on weekdays." in answer
    assert "Thank you for sharing your child's background." in answer
    assert "Thank you for sharing your preferred timing." in answer


def test_model_response_answers_availability_alongside_fees_and_clinical_context(monkeypatch):
    from core.ai import load_knowledge

    class FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="text",
                        text=(
                            "The initial evaluation is SGD 300, and a follow-up "
                            "intervention session is SGD 280."
                        ),
                    ),
                ],
            )

    monkeypatch.setattr(
        "core.ai.get_client",
        lambda: SimpleNamespace(messages=FakeMessages()),
    )

    answer = get_ai_response(
        "My child has muscle stiffness and symptoms that concern us. What are the "
        "fees for physiotherapy sessions and assessments, and what appointment "
        "slots are available?",
        [],
        {"client": {"name": "Kids Focus Physiotherapy"}},
        load_knowledge(),
    )

    assert "SGD 300" in answer
    assert "SGD 280" in answer
    assert (
        "Please let me know which service you need and the date or date range"
        in answer
    )
    assert "Thank you for sharing your child's background." in answer
    assert "A consultant would need to review these details" in answer


def test_slot_availability_question_uses_admin_check_faq_without_model(monkeypatch):
    config = {"client": {"name": "Kids Focus Physiotherapy", "timezone": "Asia/Singapore"}}
    knowledge = (
        "FAQ ID: appointment_slot_availability\n"
        "ANSWER: Our administrative team will assist you with this booking.\n"
        "HUMAN REVIEW REQUIRED: YES\n"
    )
    monkeypatch.setattr(
        "core.ai.get_client",
        lambda: pytest.fail("Slot availability FAQ should not call the LLM"),
    )

    answer = get_ai_response(
        "Do you have any appointment slots available this week?",
        [],
        config,
        knowledge,
    )

    assert answer == "Our administrative team will assist you with this booking."


def test_slot_availability_does_not_match_opening_hours_or_fee_questions():
    knowledge = (
        "FAQ ID: appointment_slot_availability\n"
        "ANSWER: The admin team would need to check live appointment slots.\n"
    )

    assert _slot_availability_answer("What are your opening hours?", knowledge) is None
    assert _slot_availability_answer("What are your appointment fees?", knowledge) is None


def test_immediate_appointment_answer_only_handles_time_sensitive_booking_questions():
    config = {"client": {"name": "Kids Focus Physiotherapy", "timezone": "Asia/Singapore"}}

    assert _immediate_appointment_answer("What are your opening hours?", config) is None
    assert _immediate_appointment_answer("Can I book an appointment next week?", config) is None


def test_knowledge_has_concise_consultant_faq():
    from core.ai import load_knowledge

    knowledge = load_knowledge()

    assert "FAQ ID: consultant_names" in knowledge
    assert "FAQ ID: physiotherapist_experience_and_continuity" in knowledge
    assert "ANSWER: Our consultants are Sarah Wong, Diana Ng, and Serena Kwa." in knowledge


def test_knowledge_includes_verified_elevator_access():
    from core.ai import load_knowledge

    knowledge = load_knowledge()

    assert "FAQ ID: clinic_elevator_access" in knowledge
    assert "Building management has confirmed that there is elevator access to the clinic." in knowledge


def test_knowledge_includes_clinical_and_saturday_response_examples():
    from core.ai import load_knowledge

    knowledge = load_knowledge()

    assert "FAQ ID: child_development_appointment" in knowledge
    assert "preferred day and time" in knowledge
    assert "FAQ ID: consultant_referral_and_fees" in knowledge
    assert "There is no discount for a referral letter" in knowledge
    assert "FAQ ID: insurance_and_payment" in knowledge
    assert "We do not bill insurance providers directly." in knowledge
    assert "Credit card payment is available on-site with an additional small service fee." in knowledge
    assert "FAQ ID: phone_and_contact_channels" in knowledge
    assert "FAQ ID: check_next_appointment" in knowledge
    assert "To help the administrative team look up your appointment" in knowledge
    assert "FAQ ID: schedule_multiple_appointments" in knowledge
    assert "FAQ ID: new_patient_saturday" in knowledge
    assert "FAQ ID: existing_patient_saturday" in knowledge
    assert "PayNow to UEN 200802130M" in knowledge
    assert "Saturday bookings and changes are handled by our administrative team" in knowledge
    assert "All our consultants are suitable to help your child." in knowledge
    assert "An initial evaluation must be booked on a weekday" in knowledge
    assert "Saturday appointments are not available to first-time patients" in knowledge
    assert "Saturday appointments are only for follow-up patients who have completed an initial evaluation" in knowledge
    assert "Rescheduling is allowed only when requested more than 36 hours before the appointment." in knowledge
    assert "the 100% cancellation charge applies" in knowledge
    assert "cancellations within 36 hours of the appointment incur a charge" in knowledge
    assert "a valid medical certificate is required" in knowledge
    assert "Saturday appointments are prepaid at SGD 300 for a single intervention" in build_system_prompt(
        {"client": {"name": "Kids Focus Physiotherapy"}, "modules": {}},
        knowledge,
    )
    assert "Treat questions about an insurer being on the clinic's insurance panel as questions about direct billing." in build_system_prompt(
        {"client": {"name": "Kids Focus Physiotherapy"}, "modules": {}},
        knowledge,
    )
    assert "FAQ ID: evaluation_report" in knowledge
    assert "SGD 250 to SGD 800 per report" in knowledge
    assert "FAQ ID: occupational_therapy" in knowledge


def test_appointment_faq_uses_multiline_point_form():
    from core.ai import _get_faq_answer, load_knowledge

    answer = _get_faq_answer(load_knowledge(), "make_appointment")

    assert answer is not None
    assert answer.startswith("To schedule an appointment, please provide:\n")
    assert "\n- The service you would like to book" in answer
    assert "\n- Your preferred date and time" in answer
    assert "\n- The patient's full name and date of birth" in answer
    assert "\n- Your full name as the parent or guardian" in answer
    assert "\n- The referring person's name, or confirm that there is no referral" in answer


def test_build_system_prompt_adds_greeting_only_for_first_response():
    config = {"client": {"name": "Kids Focus Physiotherapy"}, "modules": {}}

    first_prompt = build_system_prompt(config, "", is_first_response=True)
    later_prompt = build_system_prompt(config, "", is_first_response=False)

    assert "Begin with exactly: 'Hi, I'm Sally from Kids Focus Physiotherapy.'" in first_prompt
    assert "Begin with exactly: 'Hi, I'm Sally from Kids Focus Physiotherapy.'" not in later_prompt
    assert "Do not describe yourself as a virtual receptionist" in first_prompt
    assert "For a new appointment request, ask for all required details together using this exact heading and point-form list" in first_prompt
    assert "never repeat the full checklist" in first_prompt
    assert "To schedule an appointment, please provide:" in first_prompt
    assert "For questions asking whether Kids Focus Physiotherapy is the right facility" in first_prompt
    assert "Do not proactively tell them that sharing medical history does not replace" in first_prompt


def test_get_ai_response_uses_history_to_control_greeting(monkeypatch):
    prompts = []

    class FakeMessages:
        def create(self, **kwargs):
            prompts.append(kwargs["system"])
            return SimpleNamespace(content=[SimpleNamespace(text="Reply")])

    monkeypatch.setattr(
        "core.ai.get_client",
        lambda: SimpleNamespace(messages=FakeMessages()),
    )
    config = {"client": {"name": "Kids Focus Physiotherapy"}, "modules": {}}

    get_ai_response("Hello", [], config, "")
    get_ai_response(
        "Another question",
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
        config,
        "",
    )

    assert "Begin with exactly: 'Hi, I'm Sally from Kids Focus Physiotherapy.'" in prompts[0]
    assert "Begin with exactly: 'Hi, I'm Sally from Kids Focus Physiotherapy.'" not in prompts[1]


def test_prompt_routes_child_condition_questions_to_appointment_faq():
    prompt = build_system_prompt(
        {"client": {"name": "Kids Focus Physiotherapy"}, "modules": {}},
        "FAQ ID: make_appointment",
    )

    assert "If a message combines a factual clinic question" in prompt
    assert "answer the factual question from its matching FAQ first" in prompt
    assert "If a caregiver asks for an interpretation of their child's condition" in prompt
    assert "otherwise use the make_appointment FAQ's ANSWER" in prompt
    assert "Do not suggest that the request or appointment is already confirmed." in prompt


def test_build_system_prompt_with_booking():
    config = {
        "client": {"name": "Test"},
        "modules": {"booking": True, "payments": False},
        "booking": {
            "services": [{"name": "Dental Cleaning", "price": 150, "duration_minutes": 60}],
            "locations": [{"name": "Downtown Office", "address": "456 Oak Avenue, Springfield", "days": ["monday"]}],
            "business_hours": {"start": "09:00", "end": "18:00"},
            "cancellation_policy": "24 hours in advance",
        }
    }
    prompt = build_system_prompt(config, "knowledge")
    assert "booking_confirmed" in prompt
    assert "Dental Cleaning" in prompt
    assert "booking system successfully creates it" in prompt
    assert "do not repeat the intake checklist" in prompt
    assert "Treat a request to book an initial evaluation or physiotherapy assessment as a scheduling request" in prompt
    assert "The application handles an exact CONFIRM using its saved pending booking." in prompt
    assert "rescheduling_mc_response" in prompt
    assert "This assistant is response-only" not in prompt


def test_build_system_prompt_adds_telegram_live_availability_workflow():
    prompt = build_system_prompt(
        {
            "client": {"name": "Test"},
            "modules": {"booking": True},
            "booking": {
                "services": [{"name": "Initial evaluation", "price": 300, "duration_minutes": 60}],
                "locations": [{
                    "name": "Main Office",
                    "address": "123 Main Street",
                    "days": ["monday"],
                }],
                "business_hours": {"start": "08:00", "end": "18:00"},
            },
        },
        "knowledge",
        telegram_live_availability=True,
    )

    assert '"intent": "availability_check"' in prompt
    assert '"intent": "appointment_lookup"' in prompt
    assert "do not ask for a Telegram ID" in prompt
    assert "Do not follow any FAQ that asks for a WhatsApp number for this Telegram lookup" in prompt
    assert "does not already contain an availability result for that same service" in prompt
    assert "An available time is not booked yet" in prompt
    assert "If the conversation already says the requested slot is available and names an assigned consultant" in prompt
    assert "do not emit availability_check again" in prompt
    assert "Never mention calendars or event creation to the patient" in prompt


def test_weekend_booking_is_rejected_when_configured_weekdays_only(monkeypatch):
    from core.ai import load_knowledge

    config = {
        "client": {"name": "Kids Focus Physiotherapy", "timezone": "Asia/Singapore"},
        "modules": {"booking": True},
    }
    answer = get_ai_response(
        "My child is an existing patient and has completed an initial evaluation. "
        "Can I book a Saturday follow-up?",
        [],
        config,
        load_knowledge(),
    )

    assert "Saturday appointments are only for follow-up patients" in answer
    assert "administrative team" in answer
    assert "assist you with your request" in answer
    assert "prepaid at SGD 300" not in answer
    assert "configured for weekdays only" not in answer


def test_build_system_prompt_with_booking_and_no_configured_days():
    config = {
        "client": {"name": "Test"},
        "modules": {"booking": True, "payments": False},
    }

    prompt = build_system_prompt(config, "knowledge")

    assert "Configured clinic appointment weekdays: No appointment days are configured." in prompt
    assert "not a date-range limit" in prompt
    assert '"intent": "availability_range_check"' in prompt
    assert "Do not ask for patient or parent names just to check availability." in prompt
    assert "Use ONLY these dates" not in prompt


def test_build_system_prompt_with_booking_and_invalid_configured_days():
    config = {
        "client": {"name": "Test"},
        "modules": {"booking": True, "payments": False},
        "booking": {
            "locations": [{
                "name": "Test location",
                "address": "123 Test Street",
                "days": ["not-a-weekday"],
            }],
        },
    }

    prompt = build_system_prompt(config, "knowledge")

    assert "Configured clinic appointment weekdays: No appointment days are configured." in prompt
