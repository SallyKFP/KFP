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
    assert "When a caregiver asks about fees or charges" in prompt
    assert "follow a matching referral-and-fees FAQ if one exists" in prompt
    assert "use the appointment_slot_availability FAQ" in prompt
    assert "Answer practical, non-clinical logistics questions" in prompt
    assert "Do not infer accessibility features from the building type or address" in prompt
    assert "Just a moment please, I'll reach out to" in prompt
    assert "When asked generally who the consultants are, give only their names" in prompt
    assert "SAMPLE QUESTIONS as examples of requests with the same meaning" in prompt
    assert "use its ANSWER as the response pattern and source of facts" in prompt
    assert "distinguish cases such as new versus existing patients" in prompt
    assert "Remember the patient's answer in the conversation" in prompt
    assert "do not ask this question again" in prompt
    assert "must review and confirm the request" in prompt
    assert "Do not diagnose a child" in prompt
    assert "use a matching clinical appointment FAQ's ANSWER verbatim if available" in prompt
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


def test_fee_question_with_referral_uses_llm_then_adds_required_details(monkeypatch):
    config = {"client": {"name": "Kids Focus Physiotherapy", "timezone": "Asia/Singapore"}}
    knowledge = (
        "CONSULTANT INFORMATION\n"
        "Sarah Wong, Chief Paediatric Physiotherapist\n"
        "Serena Kwa, Consultant Physiotherapist\n"
        "Diana Ng, Consultant Physiotherapist\n"
        "DAYS, LOCATIONS AND CONTACT INFORMATION\n"
        "FAQ ID: appointment_fees\n"
        "ANSWER: The initial evaluation is SGD 300 and takes approximately 45 minutes. "
        "A follow-up intervention session is SGD 280 and also takes approximately 45 minutes.\n"
        "HUMAN REVIEW REQUIRED: NO\n"
        "FAQ ID: make_appointment\n"
        "ANSWER: Please provide your child's full name and date of birth, your full name, "
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
    assert "child's full name and date of birth" in answer
    assert "Just a moment please, I'll reach out to Serena Kwa." in answer


def test_null_model_response_uses_fees_details_and_consultant_fallback(monkeypatch):
    config = {"client": {"name": "Kids Focus Physiotherapy", "timezone": "Asia/Singapore"}}
    knowledge = (
        "CONSULTANT INFORMATION\nSerena Kwa, Consultant Physiotherapist\n"
        "DAYS, LOCATIONS AND CONTACT INFORMATION\n"
        "FAQ ID: appointment_fees\nANSWER: Initial evaluation: SGD 300.\n"
        "FAQ ID: make_appointment\n"
        "ANSWER: Please provide your child's full name and date of birth, your full name, "
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
    assert "child's full name and date of birth" in answer
    assert "Just a moment please, I'll reach out to Serena Kwa." in answer


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
    if "new patient" not in user_message:
        assert "May I check whether your child is a new patient" in answer
    assert "To schedule an appointment, please provide:" in answer
    assert "- Your child's full name and date of birth" in answer
    assert "- Your preferred day and time" in answer


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


def test_general_patient_query_also_clarifies_patient_type(monkeypatch):
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
    assert "May I check whether your child is a new patient or has visited Kids Focus Physiotherapy before?" in answer


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


def test_model_response_clarifies_patient_type_alongside_original_answer(monkeypatch):
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
    assert "May I check whether your child is a new patient" in answer


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
    assert "May I check whether your child is a new patient" in answer


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

    assert "Saturday appointments are not available for new patients" in answer
    assert "Our consultants are all suitable to help your child" in answer
    assert "SGD 300" in answer
    assert "CDA account, also known as Baby Bonus" in answer
    assert "cash or PayNow" in answer
    assert "38 Stevens Road" in answer
    assert "Turn left into Orange Grove road" in answer
    assert "arrive at least 10 minutes early" in answer
    assert "To schedule an appointment, please provide:" in answer


def test_slot_availability_question_uses_admin_check_faq_without_model(monkeypatch):
    config = {"client": {"name": "Kids Focus Physiotherapy", "timezone": "Asia/Singapore"}}
    knowledge = (
        "FAQ ID: appointment_slot_availability\n"
        "ANSWER: Live appointment availability needs to be checked by our administrative team. "
        "Please let us know your preferred days or times, and the team will check the available "
        "slots and get back to you. An appointment is not confirmed until the clinic confirms it.\n"
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

    assert "checked by our administrative team" in answer
    assert "get back to you" in answer


def test_slot_availability_does_not_match_opening_hours_or_fee_questions():
    knowledge = (
        "FAQ ID: appointment_slot_availability\n"
        "ANSWER: The admin team will check live appointment slots.\n"
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
    assert "FAQ ID: new_patient_saturday" in knowledge
    assert "FAQ ID: existing_patient_saturday" in knowledge
    assert "PayNow to UEN 200802130M" in knowledge
    assert "reply AGREE" in knowledge
    assert "replying AGREE does not confirm an appointment" in knowledge
    assert "All our consultants are suitable to help your child." in knowledge
    assert "all initial evaluations are held on weekdays" in knowledge
    assert "Weekend appointment demand is extremely high" in knowledge
    assert "Saturday follow-up appointments are available only to existing patients who have completed an initial evaluation" in knowledge
    assert "must be made upfront by PayNow to UEN 200802130M" in knowledge
    assert "full upfront SGD 300 payment" in build_system_prompt(
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
    assert "\n- Your child's full name and date of birth" in answer
    assert "\n- Your full name" in answer
    assert "\n- Your WhatsApp contact number" in answer
    assert "\n- The referring person's name, if any" in answer
    assert "\n- Your preferred day and time" in answer


def test_build_system_prompt_adds_greeting_only_for_first_response():
    config = {"client": {"name": "Kids Focus Physiotherapy"}, "modules": {}}

    first_prompt = build_system_prompt(config, "", is_first_response=True)
    later_prompt = build_system_prompt(config, "", is_first_response=False)

    assert "Begin with exactly: 'Hi, I'm Sally from Kids Focus Physiotherapy.'" in first_prompt
    assert "Begin with exactly: 'Hi, I'm Sally from Kids Focus Physiotherapy.'" not in later_prompt
    assert "Do not describe yourself as a virtual receptionist" in first_prompt
    assert "use this exact heading and point-form list" in first_prompt
    assert "To schedule an appointment, please provide:" in first_prompt
    assert "For questions asking whether Kids Focus Physiotherapy is the right facility" in first_prompt
    assert "this instruction takes precedence over the general clinical-advice guidance" in first_prompt
    assert "Do not say you cannot assess the child, cannot recommend a physiotherapist" in first_prompt


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
            "services": [{"name": "Dental Cleaning", "price": 150, "duration_minutes": 45}],
            "locations": [{"name": "Downtown Office", "address": "456 Oak Avenue, Springfield", "days": ["monday"]}],
            "business_hours": {"start": "09:00", "end": "18:00"},
            "cancellation_policy": "24 hours in advance",
        }
    }
    prompt = build_system_prompt(config, "knowledge")
    assert "booking_confirmed" in prompt
    assert "Dental Cleaning" in prompt
