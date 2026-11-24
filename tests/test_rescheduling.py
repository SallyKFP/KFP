from datetime import datetime, timedelta

import pytz

from core.rescheduling import is_within_rescheduling_window


def test_36_hour_boundary_requires_mc():
    timezone = pytz.timezone("Asia/Singapore")
    now = timezone.localize(datetime(2026, 11, 24, 10, 0))

    assert is_within_rescheduling_window(
        (now + timedelta(hours=36)).isoformat(),
        "Asia/Singapore",
        now=now,
    )
    assert not is_within_rescheduling_window(
        (now + timedelta(hours=36, minutes=1)).isoformat(),
        "Asia/Singapore",
        now=now,
    )
    assert is_within_rescheduling_window(
        (now + timedelta(hours=35, minutes=59)).isoformat(),
        "Asia/Singapore",
        now=now,
    )
