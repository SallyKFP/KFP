from datetime import date, datetime

import pytz


def clinic_today(timezone: str) -> date:
    return datetime.now(pytz.timezone(timezone)).date()


def validate_patient_dob(value: str, timezone: str) -> date:
    try:
        dob = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Please provide the patient's date of birth in YYYY-MM-DD format.") from exc
    if dob > clinic_today(timezone):
        raise ValueError("The patient's date of birth is in the future. Please check and send it again.")
    return dob
