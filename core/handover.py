"""Hand a chat over to staff when part of a patient's message needs a human."""

# Sent after the answerable parts of the patient's message. Edit this wording to change
# what patients see when their question needs a consultant or the administrative team.
HANDOVER_PLACEHOLDER = (
    "Our consultants are with patients at the moment. "
    "Please leave your message here and our team will get back to you."
)

HANDOVER_REASONS = {"clinical", "admin"}


def handover_reason(intent: dict) -> str:
    reason = str(intent.get("reason", "")).strip().lower()
    return reason if reason in HANDOVER_REASONS else "admin"


def compose_handover_reply(visible_text: str) -> str:
    """Answerable parts first, then the placeholder for the part Sally cannot answer."""
    visible_text = (visible_text or "").strip()
    if not visible_text:
        return HANDOVER_PLACEHOLDER
    return f"{visible_text}\n\n{HANDOVER_PLACEHOLDER}"
