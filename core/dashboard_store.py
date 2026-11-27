"""PostgreSQL storage for the staff dashboard on/off switches.

Switch state and processed human-message identifiers are stored.
No message content is kept.
"""

import re

import psycopg
from psycopg.rows import dict_row

CHANNELS = ("whatsapp", "telegram")
DEFAULT_COUNTRY_CODE = "65"
INDONESIA_COUNTRY_CODE = "62"


def normalize_phone(raw: str, default_country_code: str = DEFAULT_COUNTRY_CODE) -> str | None:
    """Return the WhatsApp-style number (digits, international, no '+') or None if unusable.

    A leading 0 is read as an Indonesian local number; 8-digit numbers are read as
    local numbers of the default country (Singapore).
    """
    text = str(raw).strip()
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    if text.startswith("+"):
        pass
    elif digits.startswith("00"):
        digits = digits[2:]
    elif digits.startswith("0"):
        digits = INDONESIA_COUNTRY_CODE + digits[1:]
    elif len(digits) == 8:
        digits = default_country_code + digits
    return digits if 10 <= len(digits) <= 15 and not digits.startswith("0") else None


def _validate(channel: str, contact_id: str) -> str:
    if channel not in CHANNELS:
        raise ValueError("Unsupported channel.")
    contact_id = str(contact_id).strip()
    if channel == "whatsapp":
        contact_id = re.sub(r"\D", "", contact_id)
    if not contact_id:
        raise ValueError("A contact ID is required.")
    return contact_id


class DashboardStore:
    def __init__(self, database_url: str):
        if not database_url.strip():
            raise ValueError("A PostgreSQL database URL is required.")
        self._database_url = database_url

    def initialize(self) -> None:
        with psycopg.connect(self._database_url) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dashboard_settings (
                    id BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id),
                    bot_enabled BOOLEAN NOT NULL DEFAULT TRUE
                )
                """
            )
            conn.execute(
                "ALTER TABLE dashboard_settings "
                "ADD COLUMN IF NOT EXISTS new_contacts_enabled BOOLEAN NOT NULL DEFAULT TRUE"
            )
            conn.execute(
                "INSERT INTO dashboard_settings (id) VALUES (TRUE) ON CONFLICT DO NOTHING"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dashboard_contacts (
                    channel TEXT NOT NULL CHECK (channel IN ('whatsapp', 'telegram')),
                    contact_id TEXT NOT NULL,
                    contact_label TEXT,
                    bot_enabled BOOLEAN NOT NULL DEFAULT TRUE,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (channel, contact_id)
                )
                """
            )
            conn.execute(
                "ALTER TABLE dashboard_contacts ADD COLUMN IF NOT EXISTS handover_reason TEXT"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dashboard_processed_echoes (
                    message_id TEXT PRIMARY KEY
                )
                """
            )

    def register_contact(
        self, channel: str, contact_id: str, contact_label: str | None = None
    ) -> bool:
        """Record the contact and return True if Sally should reply to it."""
        contact_id = _validate(channel, contact_id)
        label = (contact_label or "").strip() or None
        with psycopg.connect(self._database_url) as conn:
            row = conn.execute(
                """
                WITH upserted AS (
                    INSERT INTO dashboard_contacts (channel, contact_id, contact_label, bot_enabled)
                    VALUES (%s, %s, %s, COALESCE((SELECT new_contacts_enabled FROM dashboard_settings), TRUE))
                    ON CONFLICT (channel, contact_id) DO UPDATE
                    SET contact_label = COALESCE(EXCLUDED.contact_label, dashboard_contacts.contact_label)
                    RETURNING bot_enabled
                )
                SELECT upserted.bot_enabled AND settings.bot_enabled
                FROM upserted, dashboard_settings settings
                """,
                (channel, contact_id, label),
            ).fetchone()
        return bool(row[0]) if row else True

    def is_contact_enabled(self, channel: str, contact_id: str) -> bool:
        """Read-only check: False if the global switch or this contact's switch is off."""
        contact_id = _validate(channel, contact_id)
        with psycopg.connect(self._database_url) as conn:
            global_row = conn.execute("SELECT bot_enabled FROM dashboard_settings").fetchone()
            contact_row = conn.execute(
                "SELECT bot_enabled FROM dashboard_contacts WHERE channel = %s AND contact_id = %s",
                (channel, contact_id),
            ).fetchone()
        if global_row and not global_row[0]:
            return False
        return bool(contact_row[0]) if contact_row else True

    def get_global_enabled(self) -> bool:
        with psycopg.connect(self._database_url) as conn:
            row = conn.execute("SELECT bot_enabled FROM dashboard_settings").fetchone()
        return bool(row[0]) if row else True

    def set_global_enabled(self, enabled: bool) -> None:
        with psycopg.connect(self._database_url) as conn:
            conn.execute(
                "INSERT INTO dashboard_settings (id, bot_enabled) VALUES (TRUE, %s) "
                "ON CONFLICT (id) DO UPDATE SET bot_enabled = EXCLUDED.bot_enabled",
                (bool(enabled),),
            )

    def get_new_contacts_enabled(self) -> bool:
        with psycopg.connect(self._database_url) as conn:
            row = conn.execute("SELECT new_contacts_enabled FROM dashboard_settings").fetchone()
        return bool(row[0]) if row else True

    def set_new_contacts_enabled(self, enabled: bool) -> None:
        with psycopg.connect(self._database_url) as conn:
            conn.execute(
                "INSERT INTO dashboard_settings (id, new_contacts_enabled) VALUES (TRUE, %s) "
                "ON CONFLICT (id) DO UPDATE SET new_contacts_enabled = EXCLUDED.new_contacts_enabled",
                (bool(enabled),),
            )

    def import_whatsapp_contacts(
        self, contacts: list[tuple[str, str | None]], enabled: bool
    ) -> tuple[int, int]:
        """Add WhatsApp numbers with the given switch state; return (added, already_existing).

        Numbers already in the list keep their current switch.
        """
        added = 0
        with psycopg.connect(self._database_url) as conn:
            for number, label in contacts:
                number = _validate("whatsapp", number)
                cursor = conn.execute(
                    "INSERT INTO dashboard_contacts (channel, contact_id, contact_label, bot_enabled) "
                    "VALUES ('whatsapp', %s, %s, %s) ON CONFLICT (channel, contact_id) DO NOTHING",
                    (number, (label or "").strip() or None, bool(enabled)),
                )
                added += cursor.rowcount
        return added, len(contacts) - added

    def set_contact_enabled(self, channel: str, contact_id: str, enabled: bool) -> bool:
        contact_id = _validate(channel, contact_id)
        with psycopg.connect(self._database_url) as conn:
            cursor = conn.execute(
                "UPDATE dashboard_contacts SET bot_enabled = %s, updated_at = CURRENT_TIMESTAMP, "
                "handover_reason = CASE WHEN %s THEN NULL ELSE handover_reason END "
                "WHERE channel = %s AND contact_id = %s",
                (bool(enabled), bool(enabled), channel, contact_id),
            )
            return cursor.rowcount > 0

    def pause_for_handover(self, channel: str, contact_id: str, reason: str) -> None:
        """Switch Sally off for this chat because a human needs to answer part of it."""
        contact_id = _validate(channel, contact_id)
        with psycopg.connect(self._database_url) as conn:
            conn.execute(
                "INSERT INTO dashboard_contacts (channel, contact_id, bot_enabled, handover_reason) "
                "VALUES (%s, %s, FALSE, %s) "
                "ON CONFLICT (channel, contact_id) DO UPDATE SET bot_enabled = FALSE, "
                "handover_reason = EXCLUDED.handover_reason, updated_at = CURRENT_TIMESTAMP",
                (channel, contact_id, reason),
            )

    def pause_for_whatsapp_echo(self, contact_id: str, message_id: str) -> bool:
        """Pause the recipient once per human message, even if its webhook is retried."""
        contact_id = _validate("whatsapp", contact_id)
        if not isinstance(message_id, str) or not message_id.strip():
            raise ValueError("A message ID is required.")
        with psycopg.connect(self._database_url) as conn:
            inserted = conn.execute(
                "INSERT INTO dashboard_processed_echoes (message_id) VALUES (%s) "
                "ON CONFLICT DO NOTHING",
                (message_id,),
            )
            if not inserted.rowcount:
                return False
            conn.execute(
                """
                INSERT INTO dashboard_contacts (channel, contact_id, bot_enabled)
                VALUES ('whatsapp', %s, FALSE)
                ON CONFLICT (channel, contact_id) DO UPDATE
                SET bot_enabled = FALSE, updated_at = CURRENT_TIMESTAMP
                """,
                (contact_id,),
            )
        return True

    def list_contacts(self) -> list[dict]:
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            return conn.execute(
                "SELECT channel, contact_id, contact_label, bot_enabled, handover_reason "
                "FROM dashboard_contacts ORDER BY bot_enabled, updated_at DESC, contact_id"
            ).fetchall()
