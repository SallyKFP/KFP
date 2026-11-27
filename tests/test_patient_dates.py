from datetime import date, datetime, timezone

import pytest

from core.patient_dates import validate_patient_dob


class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        instant = datetime(2026, 11, 23, 16, 30, tzinfo=timezone.utc)
        return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)


def test_dob_compares_to_clinic_date_not_server_date(monkeypatch):
    monkeypatch.setattr("core.patient_dates.datetime", FrozenDatetime)
    assert validate_patient_dob("2026-10-10", "Asia/Singapore") == date(2026, 10, 10)
    assert validate_patient_dob("2026-11-24", "Asia/Singapore") == date(2026, 11, 24)
    with pytest.raises(ValueError, match="future"):
        validate_patient_dob("2026-11-24", "UTC")
    with pytest.raises(ValueError, match="future"):
        validate_patient_dob("2026-11-25", "Asia/Singapore")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        validate_patient_dob("2026-02-30", "Asia/Singapore")
