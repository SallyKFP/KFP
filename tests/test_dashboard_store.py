import os
import uuid

import psycopg
import pytest

from core.dashboard_store import DashboardStore, normalize_phone


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("+65 9123 4567", "6591234567"),
        ("91234567", "6591234567"),
        ("6591234567", "6591234567"),
        ("0065 9123 4567", "6591234567"),
        ("+62 812-3456-789", "62812345678" + "9"),
        ("0812-3456-789", "628123456789"),
        ("62812345678", "62812345678"),
        ("12345", None),
        ("abc", None),
    ],
)
def test_normalize_phone(raw, expected):
    assert normalize_phone(raw) == expected


def test_whatsapp_contact_ids_are_digits_only():
    store = DashboardStore("postgresql://unused")
    with pytest.raises(ValueError):
        store.register_contact("whatsapp", "+ -")


def test_rejects_empty_database_url():
    with pytest.raises(ValueError, match="database URL"):
        DashboardStore(" ")


@pytest.mark.parametrize(
    "channel,contact_id",
    [("other", "123"), ("telegram", " "), ("whatsapp", "")],
)
def test_rejects_invalid_contacts(channel, contact_id):
    store = DashboardStore("postgresql://unused")
    with pytest.raises(ValueError):
        store.register_contact(channel, contact_id)
    with pytest.raises(ValueError):
        store.set_contact_enabled(channel, contact_id, False)


def test_database_errors_are_not_silently_ignored(mocker):
    mocker.patch("core.dashboard_store.psycopg.connect", side_effect=psycopg.OperationalError("unavailable"))
    with pytest.raises(psycopg.OperationalError, match="unavailable"):
        DashboardStore("postgresql://unused").initialize()


def test_echo_pause_deduplicates_before_upserting_contact(mocker):
    conn = mocker.MagicMock()
    mocker.patch("core.dashboard_store.psycopg.connect").return_value.__enter__.return_value = conn
    conn.execute.return_value.rowcount = 1
    store = DashboardStore("postgresql://unused")

    assert store.pause_for_whatsapp_echo("6591234567", "echo-1") is True
    assert conn.execute.call_count == 2
    assert "dashboard_processed_echoes" in conn.execute.call_args_list[0].args[0]
    assert conn.execute.call_args_list[1].args[1] == ("6591234567",)

    conn.reset_mock()
    conn.execute.return_value.rowcount = 0
    assert store.pause_for_whatsapp_echo("6591234567", "echo-1") is False
    assert conn.execute.call_count == 1


def test_echo_pause_propagates_upsert_error_to_transaction(mocker):
    context = mocker.patch("core.dashboard_store.psycopg.connect").return_value
    conn = context.__enter__.return_value
    conn.execute.side_effect = [mocker.Mock(rowcount=1), psycopg.OperationalError("unavailable")]

    with pytest.raises(psycopg.OperationalError, match="unavailable"):
        DashboardStore("postgresql://unused").pause_for_whatsapp_echo("6591234567", "echo-1")

    assert context.__exit__.call_args.args[0] is psycopg.OperationalError


@pytest.mark.parametrize("contact_id,message_id", [("", "echo-1"), ("6591234567", "")])
def test_echo_pause_rejects_missing_identifiers(contact_id, message_id):
    with pytest.raises(ValueError):
        DashboardStore("postgresql://unused").pause_for_whatsapp_echo(contact_id, message_id)


@pytest.mark.skipif(
    not os.environ.get("DASHBOARD_TEST_DATABASE_URL"),
    reason="Set DASHBOARD_TEST_DATABASE_URL to a dedicated test database.",
)
def test_postgresql_switches_round_trip():
    url = os.environ["DASHBOARD_TEST_DATABASE_URL"]
    store = DashboardStore(url)
    contact_id = "test-" + uuid.uuid4().hex
    store.initialize()
    original_global = store.get_global_enabled()
    try:
        store.set_global_enabled(True)
        assert store.register_contact("telegram", contact_id, "Jamie") is True
        assert store.set_contact_enabled("telegram", contact_id, False) is True
        assert store.register_contact("telegram", contact_id) is False
        assert store.set_contact_enabled("whatsapp", contact_id, False) is False
        store.set_contact_enabled("telegram", contact_id, True)
        store.set_global_enabled(False)
        assert store.register_contact("telegram", contact_id) is False
        store.set_global_enabled(True)
        assert store.register_contact("telegram", contact_id) is True
        row = next(r for r in store.list_contacts() if r["contact_id"] == contact_id)
        assert row["contact_label"] == "Jamie"
        assert set(row) == {"channel", "contact_id", "contact_label", "bot_enabled"}
    finally:
        store.set_global_enabled(original_global)
        with psycopg.connect(url) as conn:
            conn.execute("DELETE FROM dashboard_contacts WHERE contact_id = %s", (contact_id,))


@pytest.mark.skipif(
    not os.environ.get("DASHBOARD_TEST_DATABASE_URL"),
    reason="Set DASHBOARD_TEST_DATABASE_URL to a dedicated test database.",
)
def test_postgresql_echo_pause_and_retry():
    url = os.environ["DASHBOARD_TEST_DATABASE_URL"]
    store = DashboardStore(url)
    phone = "65" + str(uuid.uuid4().int % 10**13).zfill(13)
    first_echo = "test-" + uuid.uuid4().hex
    second_echo = "test-" + uuid.uuid4().hex
    store.initialize()
    try:
        assert store.pause_for_whatsapp_echo(phone, first_echo) is True
        row = next(r for r in store.list_contacts() if r["contact_id"] == phone)
        assert row["bot_enabled"] is False
        assert store.set_contact_enabled("whatsapp", phone, True)
        assert store.pause_for_whatsapp_echo(phone, first_echo) is False
        row = next(r for r in store.list_contacts() if r["contact_id"] == phone)
        assert row["bot_enabled"] is True
        assert store.pause_for_whatsapp_echo(phone, second_echo) is True
        row = next(r for r in store.list_contacts() if r["contact_id"] == phone)
        assert row["bot_enabled"] is False
    finally:
        with psycopg.connect(url) as conn:
            conn.execute("DELETE FROM dashboard_contacts WHERE channel = 'whatsapp' AND contact_id = %s", (phone,))
            conn.execute("DELETE FROM dashboard_processed_echoes WHERE message_id IN (%s, %s)", (first_echo, second_echo))
