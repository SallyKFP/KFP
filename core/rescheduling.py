from datetime import datetime, timedelta

import pytz


RESCHEDULING_WINDOW = timedelta(hours=36)


def is_within_rescheduling_window(
    start_datetime: str,
    timezone: str,
    *,
    now: datetime | None = None,
) -> bool:
    appointment = datetime.fromisoformat(start_datetime)
    clinic_timezone = pytz.timezone(timezone)
    if appointment.tzinfo is None:
        appointment = clinic_timezone.localize(appointment)
    else:
        appointment = appointment.astimezone(clinic_timezone)
    current_time = now or datetime.now(clinic_timezone)
    if current_time.tzinfo is None:
        current_time = clinic_timezone.localize(current_time)
    else:
        current_time = current_time.astimezone(clinic_timezone)
    return appointment - current_time <= RESCHEDULING_WINDOW
