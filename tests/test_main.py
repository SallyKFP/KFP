import os
import json
import hashlib
import hmac
import re
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ANTHROPIC_API_KEY", "test")
os.environ.setdefault("WHATSAPP_ACCESS_TOKEN", "test")
os.environ.setdefault("WHATSAPP_PHONE_NUMBER_ID", "123")
os.environ.setdefault("WHATSAPP_VERIFY_TOKEN", "mytoken")
os.environ.setdefault("WHATSAPP_APP_SECRET", "appsecret")
os.environ.setdefault("INTERNAL_SECRET", "internalsecret")
os.environ.setdefault("GOOGLE_CALENDAR_ID", "test@calendar")
os.environ.setdefault("GOOGLE_CALENDAR_OWNER_EMAIL", "test@test.com")

from core.main import app, parse_contact_lines

client = TestClient(app)


def test_parse_contact_lines():
    contacts, skipped = parse_contact_lines(
        'phone,name\n+65 9123 4567,"Jamie Tan"\n91234567\n0812-3456-789;Budi\n12345678901234567\n'
    )
    assert contacts == [("6591234567", "Jamie Tan"), ("628123456789", "Budi")]
    assert skipped == ["12345678901234567"]


@pytest.fixture(autouse=True)
def isolated_pending_bookings(monkeypatch):
    from core.pending_booking import PendingBookings
    store = PendingBookings(clock=lambda: 0.0)
    monkeypatch.setattr("core.main.PENDING_BOOKINGS", store)
    return store


def booking_details(**overrides):
    return {
        "service": "Initial evaluation", "date": "2026-11-23", "time": "10:00",
        "child_name": "Jane", "child_dob": "2020-05-03", "parent_name": "Alex",
        "referral_name": "None", "patient_resides_in_singapore": "yes",
        "location": "Main Office", **overrides,
    }


def test_health_check():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_dashboard_shows_only_on_off_switches(dashboard_client, mocker, monkeypatch):
    store = mocker.MagicMock()
    store.get_global_enabled.return_value = True
    store.list_contacts.return_value = [
        {"channel": "whatsapp", "contact_id": "15551234567", "contact_label": "Alex <b>", "bot_enabled": False},
    ]
    monkeypatch.setattr("core.main._dashboard_store", store)
    test_client = dashboard_client
    response = test_client.get("/dashboard", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/dashboard/login"
    resp = test_client.get(response.headers["location"])
    csrf_token = re.search(r'name="csrf_token" value="([^"]+)"', resp.text).group(1)
    test_client.post(
        "/dashboard/login",
        data={
            "username": "staff",
            "password": "test-dashboard-password",
            "csrf_token": csrf_token,
        },
    )
    resp = test_client.get("/dashboard")
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-store"
    assert "Sign out" in resp.text
    assert "Sally ON" in resp.text
    assert "Sally OFF" in resp.text
    assert "Turn off" in resp.text
    assert "Alex &lt;b&gt;" in resp.text
    assert "Preview only" not in resp.text

    csrf = re.search(r'name="csrf_token" value="([^"]+)"', resp.text).group(1)
    toggled = test_client.post(
        "/dashboard/switch",
        data={"csrf_token": csrf, "scope": "global", "enabled": "false"},
        follow_redirects=False,
    )
    assert toggled.status_code == 303
    store.set_global_enabled.assert_called_once_with(False)
    store.set_contact_enabled.return_value = True
    test_client.post(
        "/dashboard/switch",
        data={
            "csrf_token": csrf, "scope": "contact", "channel": "whatsapp",
            "contact_id": "15551234567", "enabled": "true",
        },
    )
    store.set_contact_enabled.assert_called_once_with("whatsapp", "15551234567", True)
    bad = test_client.post(
        "/dashboard/switch",
        data={"csrf_token": "wrong", "scope": "global", "enabled": "true"},
    )
    assert bad.status_code == 403


def test_dashboard_switch_requires_login(dashboard_client):
    response = dashboard_client.post(
        "/dashboard/switch", data={"scope": "global", "enabled": "false"}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/dashboard/login"


@pytest.fixture
def dashboard_client(monkeypatch):
    monkeypatch.delenv("DASHBOARD_ACCOUNTS", raising=False)
    monkeypatch.setenv("DASHBOARD_USERNAME", "staff")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-dashboard-password")
    monkeypatch.setenv("DASHBOARD_SESSION_SECRET", "test-session-secret-" * 3)
    return TestClient(app, base_url="https://testserver")


def _dashboard_csrf(response):
    return re.search(r'name="csrf_token" value="([^"]+)"', response.text).group(1)


def test_dashboard_requires_login_and_supports_logout(dashboard_client):
    response = dashboard_client.get("/dashboard", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/dashboard/login"

    login_page = dashboard_client.get("/dashboard/login")
    csrf_token = _dashboard_csrf(login_page)
    response = dashboard_client.post(
        "/dashboard/login",
        data={
            "username": "staff",
            "password": "test-dashboard-password",
            "csrf_token": csrf_token,
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    dashboard = dashboard_client.get("/dashboard")
    assert dashboard.status_code == 200
    assert "Sign out" in dashboard.text
    cookie_header = dashboard.headers.get("set-cookie", "").lower()
    assert "secure" in cookie_header
    assert "httponly" in cookie_header
    assert "samesite=lax" in cookie_header
    logout_response = dashboard_client.post(
        "/dashboard/logout",
        data={"csrf_token": _dashboard_csrf(dashboard)},
        follow_redirects=False,
    )
    assert logout_response.status_code == 303
    assert dashboard_client.get(
        "/dashboard", follow_redirects=False
    ).headers["location"] == "/dashboard/login"


def test_dashboard_rejects_bad_credentials(dashboard_client):
    login_page = dashboard_client.get("/dashboard/login")
    response = dashboard_client.post(
        "/dashboard/login",
        data={
            "username": "staff",
            "password": "wrong-password",
            "csrf_token": _dashboard_csrf(login_page),
        },
    )
    assert response.status_code == 401
    assert "username or password is incorrect" in response.text
    assert "wrong-password" not in response.text
    assert dashboard_client.get("/dashboard", follow_redirects=False).status_code == 303


def test_dashboard_rejects_invalid_csrf_token(dashboard_client):
    response = dashboard_client.post(
        "/dashboard/login",
        data={
            "username": "staff",
            "password": "test-dashboard-password",
            "csrf_token": "invalid",
        },
    )
    assert response.status_code == 400
    assert dashboard_client.get("/dashboard", follow_redirects=False).status_code == 303


def test_dashboard_is_unavailable_without_secure_credentials(monkeypatch):
    for name in (
        "DASHBOARD_ACCOUNTS",
        "DASHBOARD_USERNAME",
        "DASHBOARD_PASSWORD",
        "DASHBOARD_SESSION_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    response = TestClient(app).get("/dashboard")
    assert response.status_code == 503
    assert "DASHBOARD_SESSION_SECRET" in response.text


def _dashboard_login_as(test_client, username, password):
    login_page = test_client.get("/dashboard/login")
    return test_client.post(
        "/dashboard/login",
        data={"username": username, "password": password, "csrf_token": _dashboard_csrf(login_page)},
        follow_redirects=False,
    )


def test_dashboard_individual_accounts(dashboard_client, monkeypatch):
    monkeypatch.setenv(
        "DASHBOARD_ACCOUNTS",
        json.dumps({"alice": "alice-password-123", "bob": "bob-password-123"}),
    )
    assert _dashboard_login_as(dashboard_client, "alice", "bob-password-123").status_code == 401
    assert _dashboard_login_as(dashboard_client, "staff", "test-dashboard-password").status_code == 401
    assert _dashboard_login_as(dashboard_client, "alice", "alice-password-123").status_code == 303
    assert dashboard_client.get("/dashboard").status_code == 200
    dashboard = dashboard_client.get("/dashboard")
    dashboard_client.post("/dashboard/logout", data={"csrf_token": _dashboard_csrf(dashboard)})
    assert _dashboard_login_as(dashboard_client, "bob", "bob-password-123").status_code == 303
    assert dashboard_client.get("/dashboard").status_code == 200


def test_removing_another_account_does_not_revoke_session(dashboard_client, monkeypatch):
    monkeypatch.setenv(
        "DASHBOARD_ACCOUNTS",
        json.dumps({"alice": "alice-password-123", "bob": "bob-password-123"}),
    )
    assert _dashboard_login_as(dashboard_client, "bob", "bob-password-123").status_code == 303
    monkeypatch.setenv("DASHBOARD_ACCOUNTS", json.dumps({"bob": "bob-password-123"}))
    assert dashboard_client.get("/dashboard").status_code == 200


@pytest.mark.parametrize(
    "accounts",
    ["", "{", "{}", "[]", '{"alice":"short"}', '{"alice":123}',
     '{"alice":"password-long-one","alice":"password-long-two"}'],
)
def test_invalid_dashboard_accounts_fail_closed(dashboard_client, monkeypatch, accounts):
    monkeypatch.setenv("DASHBOARD_ACCOUNTS", accounts)
    assert dashboard_client.get("/dashboard").status_code == 503
    assert dashboard_client.get("/dashboard/login").status_code == 503


@pytest.mark.parametrize("change", ["removed", "password_changed", "invalid_config"])
def test_dashboard_account_changes_revoke_session(dashboard_client, monkeypatch, change):
    monkeypatch.setenv(
        "DASHBOARD_ACCOUNTS",
        json.dumps({"alice": "alice-password-123", "bob": "bob-password-123"}),
    )
    assert _dashboard_login_as(dashboard_client, "alice", "alice-password-123").status_code == 303
    dashboard = dashboard_client.get("/dashboard")
    csrf = _dashboard_csrf(dashboard)
    if change == "removed":
        new_accounts = {"bob": "bob-password-123"}
    elif change == "password_changed":
        new_accounts = {"alice": "different-password-123", "bob": "bob-password-123"}
    else:
        new_accounts = {}
    monkeypatch.setenv("DASHBOARD_ACCOUNTS", json.dumps(new_accounts))
    for path in ("/dashboard/switch", "/dashboard/import"):
        response = dashboard_client.post(path, data={"csrf_token": csrf}, follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/dashboard/login"
    response = dashboard_client.get("/dashboard", follow_redirects=False)
    assert response.status_code == (503 if change == "invalid_config" else 303)


def test_configured_database_is_initialized_on_app_startup(mocker, monkeypatch):
    from core.main import app

    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    mocker.patch("core.main.CONFIG", {"modules": {"reminders": False}})
    store = mocker.MagicMock()
    mocker.patch("core.main.DashboardStore", return_value=store)

    with TestClient(app) as app_client:
        assert app_client.get("/").status_code == 200

    store.initialize.assert_called_once_with()


def test_daily_reminder_scheduler_targets_nine_am_singapore():
    from datetime import datetime
    import pytz

    from core.main import _seconds_until_next_reminder

    timezone = pytz.timezone("Asia/Singapore")
    before_nine = timezone.localize(datetime(2026, 11, 6, 8, 30))
    after_nine = timezone.localize(datetime(2026, 11, 6, 9, 1))

    assert _seconds_until_next_reminder(before_nine, "Asia/Singapore") == 30 * 60
    assert _seconds_until_next_reminder(after_nine, "Asia/Singapore") == (
        23 * 60 * 60 + 59 * 60
    )


def test_reminder_endpoint_sends_only_whatsapp_reminders(mocker, monkeypatch):
    config = {
        "modules": {"booking": True, "reminders": True},
    }
    mocker.patch("core.main.CONFIG", config)
    mock_calendar = mocker.MagicMock()
    mocker.patch("core.main._get_calendar_client", return_value=mock_calendar)
    whatsapp_reminders = mocker.patch(
        "core.main.send_reminders",
        new_callable=mocker.AsyncMock,
        return_value=2,
    )
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-telegram-token")

    response = client.post(
        "/internal/send-reminders",
        headers={"X-Internal-Secret": "internalsecret"},
    )

    assert response.status_code == 200
    assert response.json() == {"sent": 2}
    whatsapp_reminders.assert_awaited_once_with(
        config,
        mock_calendar._service,
        mocker.ANY,
    )


def test_reminder_endpoint_skips_telegram_without_token(mocker, monkeypatch):
    mocker.patch("core.main.CONFIG", {"modules": {"booking": True, "reminders": True}})
    mock_calendar = mocker.MagicMock()
    mocker.patch("core.main._get_calendar_client", return_value=mock_calendar)
    mocker.patch("core.main.send_reminders", new_callable=mocker.AsyncMock, return_value=2)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    response = client.post(
        "/internal/send-reminders",
        headers={"X-Internal-Secret": "internalsecret"},
    )

    assert response.status_code == 200
    assert response.json() == {"sent": 2}


def test_webhook_verification():
    resp = client.get("/webhook", params={
        "hub.mode": "subscribe",
        "hub.verify_token": "mytoken",
        "hub.challenge": "12345",
    })
    assert resp.status_code == 200
    assert resp.text == "12345"


def test_webhook_verification_wrong_token():
    resp = client.get("/webhook", params={
        "hub.mode": "subscribe",
        "hub.verify_token": "wrong",
        "hub.challenge": "12345",
    })
    assert resp.status_code == 403


def test_webhook_invalid_signature():
    resp = client.post(
        "/webhook",
        content=b'{"test": "data"}',
        headers={"X-Hub-Signature-256": "sha256=invalid", "Content-Type": "application/json"},
    )
    assert resp.status_code == 403


def test_webhook_status_update_ignored():
    body = json.dumps({
        "entry": [{"changes": [{"value": {"statuses": [{"status": "delivered"}]}}]}]
    }).encode()
    sig = "sha256=" + hmac.new(b"appsecret", body, hashlib.sha256).hexdigest()
    resp = client.post(
        "/webhook",
        content=body,
        headers={"X-Hub-Signature-256": sig, "Content-Type": "application/json"},
    )
    assert resp.status_code == 200


def _echo_change(*echoes):
    return {
        "field": "smb_message_echoes",
        "value": {
            "metadata": {"phone_number_id": "123"},
            "message_echoes": list(echoes),
        },
    }


def _post_signed_webhook(payload):
    body = json.dumps(payload).encode()
    signature = "sha256=" + hmac.new(b"appsecret", body, hashlib.sha256).hexdigest()
    return client.post(
        "/webhook", content=body,
        headers={"X-Hub-Signature-256": signature, "Content-Type": "application/json"},
    )


def test_message_echo_pauses_each_recipient_across_entries(mocker, monkeypatch):
    store = mocker.MagicMock()
    monkeypatch.setattr("core.main._dashboard_store", store)
    process = mocker.patch("core.main._process_message", new_callable=mocker.AsyncMock)
    payload = {"entry": [
        {"changes": [
            {"field": "other", "value": {}},
            _echo_change(
                {"id": "echo-1", "from": "6591111111", "to": "6591234567", "type": "text"},
                {"id": "echo-2", "from": "6591111111", "to": "6598765432", "type": "image"},
            ),
        ]},
        {"changes": [_echo_change({"id": "echo-3", "to": "628123456789", "type": "text"})]},
    ]}

    response = _post_signed_webhook(payload)

    assert response.status_code == 200
    assert store.pause_for_whatsapp_echo.call_args_list == [
        mocker.call("6591234567", "echo-1"),
        mocker.call("6598765432", "echo-2"),
        mocker.call("628123456789", "echo-3"),
    ]
    process.assert_not_awaited()


@pytest.mark.parametrize("reason", ["missing_store", "database_error"])
def test_message_echo_returns_error_if_pause_cannot_be_saved(mocker, monkeypatch, reason):
    store = mocker.MagicMock() if reason == "database_error" else None
    if store is not None:
        store.pause_for_whatsapp_echo.side_effect = RuntimeError("database unavailable")
    monkeypatch.setattr("core.main._dashboard_store", store)
    payload = {"entry": [{"changes": [_echo_change({"id": "echo-1", "to": "6591234567"})]}]}

    assert _post_signed_webhook(payload).status_code == 503


@pytest.mark.parametrize("echo", [{}, {"id": "echo-1"}, {"id": "", "to": "6591234567"},
                                   {"id": "echo-1", "to": "not-a-number"}])
def test_message_echo_rejects_malformed_entries(mocker, monkeypatch, echo):
    store = mocker.MagicMock()
    monkeypatch.setattr("core.main._dashboard_store", store)
    response = _post_signed_webhook({"entry": [{"changes": [_echo_change(echo)]}]})
    assert response.status_code == 400
    store.pause_for_whatsapp_echo.assert_not_called()


def test_message_echo_rejects_wrong_number_and_invalid_signature(mocker, monkeypatch):
    store = mocker.MagicMock()
    monkeypatch.setattr("core.main._dashboard_store", store)
    change = _echo_change({"id": "echo-1", "to": "6591234567"})
    change["value"]["metadata"]["phone_number_id"] = "other"
    payload = {"entry": [{"changes": [change]}]}
    assert _post_signed_webhook(payload).status_code == 400
    assert client.post("/webhook", json=payload).status_code == 403
    store.pause_for_whatsapp_echo.assert_not_called()


def test_message_echo_pauses_before_a_batched_incoming_message(mocker, monkeypatch):
    store = mocker.MagicMock()
    store.register_contact.return_value = False
    monkeypatch.setattr("core.main._dashboard_store", store)
    process = mocker.patch("core.main._process_message", new_callable=mocker.AsyncMock)
    incoming = json.loads(_whatsapp_text_body())["entry"][0]["changes"][0]
    payload = {"entry": [{"changes": [
        _echo_change({"id": "echo-1", "to": "15551234567"}), incoming,
    ]}]}

    assert _post_signed_webhook(payload).status_code == 200
    assert store.pause_for_whatsapp_echo.call_count == 1
    store.register_contact.assert_called_once_with("whatsapp", "15551234567", "Alex Parent")
    process.assert_not_awaited()


def test_human_takeover_stays_paused_until_dashboard_resume(dashboard_client, mocker, monkeypatch):
    class FakeStore:
        def __init__(self):
            self.enabled = {}
            self.seen = set()

        def pause_for_whatsapp_echo(self, phone, message_id):
            if message_id in self.seen:
                return False
            self.seen.add(message_id)
            self.enabled[phone] = False
            return True

        def register_contact(self, channel, phone, label=None):
            return self.enabled.setdefault(phone, True)

        def get_global_enabled(self):
            return True

        def get_new_contacts_enabled(self):
            return True

        def list_contacts(self):
            return [
                {"channel": "whatsapp", "contact_id": phone, "contact_label": None, "bot_enabled": enabled}
                for phone, enabled in self.enabled.items()
            ]

        def set_contact_enabled(self, channel, phone, enabled):
            self.enabled[phone] = enabled
            return True

    store = FakeStore()
    monkeypatch.setattr("core.main._dashboard_store", store)
    process = mocker.patch("core.main._process_message", new_callable=mocker.AsyncMock)
    echo = {"entry": [{"changes": [_echo_change({"id": "echo-1", "to": "15551234567"})]}]}
    incoming = json.loads(_whatsapp_text_body())
    other_incoming = json.loads(_whatsapp_text_body())
    other_value = other_incoming["entry"][0]["changes"][0]["value"]
    other_value["messages"][0]["from"] = "15557654321"

    assert _post_signed_webhook(echo).status_code == 200
    assert _post_signed_webhook(incoming).status_code == 200
    assert _post_signed_webhook(other_incoming).status_code == 200
    assert process.await_count == 1
    assert process.await_args.args[0] == "15557654321"

    _dashboard_login_as(dashboard_client, "staff", "test-dashboard-password")
    dashboard = dashboard_client.get("/dashboard")
    response = dashboard_client.post("/dashboard/switch", data={
        "csrf_token": _dashboard_csrf(dashboard), "scope": "contact",
        "channel": "whatsapp", "contact_id": "15551234567", "enabled": "true",
    })
    assert response.status_code == 200
    assert _post_signed_webhook(echo).status_code == 200
    assert _post_signed_webhook(incoming).status_code == 200
    assert process.await_count == 2
    assert process.await_args.args[0] == "15551234567"


def _whatsapp_text_body():
    return json.dumps({
        "entry": [{
            "changes": [{
                "value": {
                    "contacts": [{"wa_id": "15551234567", "profile": {"name": "Alex Parent"}}],
                    "messages": [{
                        "id": "wamid.test-message",
                        "from": "15551234567",
                        "type": "text",
                        "text": {"body": "I need help from a consultant"},
                    }],
                },
            }],
        }],
    })


@pytest.mark.parametrize("should_reply", [True, False])
def test_whatsapp_contact_is_registered_and_switch_controls_reply(mocker, monkeypatch, should_reply):
    store = mocker.MagicMock()
    store.register_contact.return_value = should_reply
    monkeypatch.setattr("core.main._dashboard_store", store)
    monkeypatch.setattr("core.main.validate_webhook_signature", lambda *args: True)
    process = mocker.AsyncMock(return_value=None)
    monkeypatch.setattr("core.main._process_message", process)

    response = client.post("/webhook", content=_whatsapp_text_body())

    assert response.status_code == 200
    store.register_contact.assert_called_once_with("whatsapp", "15551234567", "Alex Parent")
    assert not store.add_message.called
    assert process.await_count == (1 if should_reply else 0)


def test_booking_intent_creates_event_without_payments(mocker, isolated_pending_bookings):
    """When payments are disabled, booking intent creates calendar event immediately."""
    mocker.patch("core.main.CONFIG", {
        "client": {"name": "Test", "timezone": "America/Argentina/Buenos_Aires"},
        "modules": {"booking": True, "payments": False, "reminders": False},
        "booking": {
            "calendar_id": "test",
            "calendar_owner_email": "test@test.com",
            "business_hours": {"start": "09:00", "end": "18:00"},
            "services": [{"name": "Dental Cleaning", "price": 150, "duration_minutes": 60}],
            "locations": [{"name": "Downtown Office", "address": "456 Oak Avenue, Springfield", "days": ["monday"]}],
            "cancellation_policy": "24 hours in advance",
        }
    })
    mock_calendar = mocker.MagicMock()
    mock_calendar.is_slot_available.return_value = True
    mock_calendar.has_upcoming_appointment_for_patient.return_value = False
    mocker.patch("core.main._get_calendar_client", return_value=mock_calendar)
    mock_send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    history = mocker.MagicMock()
    history.get.return_value = [{
        "role": "assistant",
        "content": (
            "Thank you. Before I book, please confirm these details:\n"
            "- Service: Dental Cleaning\n- Date: 2027-03-15\n- Time: 10:00\n"
            "- Patient: Jane Child\n- Patient DOB: 2020-05-03\n- Parent: Alex Parent\n"
            "- Referral: None\n- Patient resides in Singapore: Yes\n"
            "Please reply Confirm if correct."
        ),
    }]
    mocker.patch("core.main.HISTORY", history)
    isolated_pending_bookings.put("whatsapp:5491112345678", booking_details(
        service="Dental Cleaning", date="2027-03-15", child_name="Jane Child",
        parent_name="Alex Parent", location="Downtown Office",
    ))
    isolated_pending_bookings.set_observations("whatsapp:5491112345678", "Enjoys playing")
    isolated_pending_bookings.approve_observations("whatsapp:5491112345678")
    mocker.patch("core.main.get_ai_response", return_value=(
        'Your appointment is confirmed. {"intent": "booking_confirmed", "service": "Dental Cleaning", '
        '"date": "2027-03-15", "time": "10:00", "location": "Downtown Office", '
        '"child_name": "Jane Child", "child_dob": "2020-05-03", '
        '"parent_name": "Alex Parent", "referral_name": "None", '
        '"patient_resides_in_singapore": "yes"}'
    ))

    import json
    import hashlib
    import hmac
    body = json.dumps({"entry": [{"changes": [{"value": {"messages": [
        {"from": "5491112345678", "type": "text", "text": {"body": "Confirm"}}
    ]}}]}]}).encode()
    sig = "sha256=" + hmac.new(b"appsecret", body, hashlib.sha256).hexdigest()

    from fastapi.testclient import TestClient
    from core.main import app
    test_client = TestClient(app)
    resp = test_client.post("/webhook", content=body,
        headers={"X-Hub-Signature-256": sig, "Content-Type": "application/json"})

    assert resp.status_code == 200
    mock_calendar.create_event.assert_called_once()
    assert mock_send.called
    confirmation = mock_send.call_args.args[1]
    assert "Fee: SGD 150." in confirmation
    assert mock_calendar.create_event.call_args.kwargs["child_name"] == "Jane Child"
    assert mock_calendar.create_event.call_args.kwargs["parent_name"] == "Alex Parent"
    event_args = mock_calendar.create_event.call_args.kwargs
    assert "child_dob" not in event_args
    assert event_args["referral_name"] == "None"
    assert event_args["patient_age_group"] == "Child (0-12)"
    assert event_args["observations"] == "Enjoys playing"
    assert event_args["duration_minutes"] == 60
    assert "location_address" not in event_args
    assert "24 hours in advance" in confirmation


@pytest.mark.asyncio
async def test_booking_intent_is_ignored_for_acknowledgement_after_confirmation(mocker):
    from core.main import _process_message

    history = mocker.MagicMock()
    history.get.return_value = [
        {
            "role": "assistant",
            "content": "Appointment confirmed: Initial evaluation for Jane on Thursday.",
        }
    ]
    mocker.patch("core.main.HISTORY", history)
    mocker.patch("core.main._find_upcoming_events_by_phone", return_value=[])
    mocker.patch("core.main.CONFIG", {"modules": {"booking": True}})
    mocker.patch(
        "core.main.get_ai_response",
        return_value=(
            'You are welcome. {"intent": "booking_confirmed", '
            '"service": "Initial evaluation"}'
        ),
    )
    booking_handler = mocker.patch(
        "core.main._handle_booking_intent",
        new_callable=mocker.AsyncMock,
    )
    send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)

    await _process_message("6591234567", "Ok perfect")

    booking_handler.assert_not_awaited()
    send.assert_awaited_once_with(
        "6591234567",
        "You're welcome! The appointment was confirmed in my previous message.",
    )
    assert '{"intent"' not in send.call_args.args[1]


@pytest.mark.asyncio
async def test_booking_intent_requires_confirmation_prompt_and_explicit_reply(mocker):
    from core.main import _process_message

    history = mocker.MagicMock()
    history.get.return_value = [
        {"role": "assistant", "content": "The requested time is available."}
    ]
    mocker.patch("core.main.HISTORY", history)
    mocker.patch("core.main.CONFIG", {"modules": {"booking": True}})
    mocker.patch(
        "core.main._handle_whatsapp_attendance_confirmation",
        new_callable=mocker.AsyncMock,
        return_value=False,
    )
    mocker.patch(
        "core.main.get_ai_response",
        return_value='{"intent": "booking_confirmed", "service": "Initial evaluation"}',
    )
    booking_handler = mocker.patch(
        "core.main._handle_booking_intent",
        new_callable=mocker.AsyncMock,
    )
    send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)

    await _process_message("6591234567", "Yes")

    booking_handler.assert_not_awaited()
    assert "without first showing you the details" in send.call_args.args[1]
    assert '{"intent"' not in send.call_args.args[1]


@pytest.mark.asyncio
async def test_whatsapp_availability_range_intent_returns_calendar_results(mocker):
    from core.main import _process_message

    history = mocker.MagicMock()
    history.get.return_value = []
    mocker.patch("core.main.HISTORY", history)
    mocker.patch("core.main.CONFIG", {"modules": {"booking": True}})
    mocker.patch(
        "core.main._handle_whatsapp_attendance_confirmation",
        new_callable=mocker.AsyncMock,
        return_value=False,
    )
    intent = {
        "intent": "availability_range_check",
        "service": "Initial evaluation",
        "start_date": "2026-11-23",
        "end_date": "2026-11-29",
    }
    mocker.patch(
        "core.main.get_ai_response",
        return_value=(
            'I am checking. {"intent": "availability_range_check", '
            '"service": "Initial evaluation", "start_date": "2026-11-23", '
            '"end_date": "2026-11-29"}'
        ),
    )
    range_check = mocker.patch(
        "core.main.check_availability_range",
        return_value="These times are currently available.",
    )
    send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)

    await _process_message("6591234567", "Any availability the week of November 23?")

    range_check.assert_called_once()
    assert range_check.call_args.args[:2] == ({"modules": {"booking": True}}, intent)
    send.assert_awaited_once_with("6591234567", "These times are currently available.")


@pytest.mark.asyncio
async def test_yes_does_not_confirm_booking_or_reminder_after_booking_summary(mocker, isolated_pending_bookings):
    from core.main import _process_message

    history = mocker.MagicMock()
    history.get.return_value = [{
        "role": "assistant",
        "content": (
            "Please confirm these booking details:\n"
            "- Service: Initial evaluation\n- Date: 2026-11-23\n- Time: 10:00\n"
            "- Patient: Jane\n- Patient DOB: 2020-05-03\n- Parent: Alex\n"
            "- Referral: None\n- Patient resides in Singapore: Yes"
        ),
    }]
    mocker.patch("core.main.HISTORY", history)
    isolated_pending_bookings.put("whatsapp:6591234567", booking_details())
    mocker.patch("core.main.CONFIG", {"modules": {"booking": True}})
    attendance_confirmation = mocker.patch(
        "core.main._handle_whatsapp_attendance_confirmation",
        new_callable=mocker.AsyncMock,
    )
    mocker.patch(
        "core.main.get_ai_response",
        return_value='{"intent": "booking_confirmed", "service": "Initial evaluation"}',
    )
    booking_handler = mocker.patch(
        "core.main._handle_booking_intent",
        new_callable=mocker.AsyncMock,
    )
    send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)

    await _process_message("6591234567", "Yes")

    attendance_confirmation.assert_not_awaited()
    booking_handler.assert_not_awaited()
    assert "reply CONFIRM only" in send.call_args.args[1]


@pytest.mark.asyncio
async def test_yes_to_booking_information_question_does_not_confirm_reminder(mocker):
    from core.main import _process_message

    history = mocker.MagicMock()
    history.get.return_value = [{
        "role": "assistant", "content": "Does the patient reside in Singapore?",
    }]
    mocker.patch("core.main.HISTORY", history)
    mocker.patch("core.main.CONFIG", {"modules": {"booking": True}})
    attendance = mocker.patch(
        "core.main._handle_whatsapp_attendance_confirmation",
        new_callable=mocker.AsyncMock,
    )
    ai = mocker.patch("core.main.get_ai_response", return_value="Thank you.")
    mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    await _process_message("6591234567", "YES")
    attendance.assert_not_awaited()
    ai.assert_called_once()


@pytest.mark.asyncio
async def test_whatsapp_yes_confirms_appointment_from_recent_reminder(mocker):
    from core.main import _handle_whatsapp_attendance_confirmation

    event = {
        "id": "appointment-1",
        "calendar_id": "sarah@calendar",
        "patient_name": "Jane Example",
        "date": "Tuesday November 10, 2026",
        "time": "09:00",
        "start_datetime": "2026-11-10T09:00:00+08:00",
        "reminder_sent_for": "2026-11-10",
        "attendance_confirmed_for": None,
    }
    mocker.patch("core.main._find_upcoming_events_by_phone", return_value=[event])
    calendar = mocker.MagicMock()
    mocker.patch("core.main._calendar_client_for_event", return_value=calendar)
    mocker.patch("core.main.HISTORY.add")
    send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)

    handled = await _handle_whatsapp_attendance_confirmation("6591234567", "YES")

    assert handled
    calendar.mark_event_attendance_confirmed.assert_called_once_with("appointment-1")
    assert "Attendance is confirmed" in send.call_args.args[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_text", ["YES!", "yes please", "CONFIRM"])
async def test_whatsapp_reminder_requires_exact_yes(mocker, reply_text):
    from core.main import _handle_whatsapp_attendance_confirmation

    event = {
        "id": "appointment-1", "start_datetime": "2026-11-10T09:00:00+08:00",
        "reminder_sent_for": "2026-11-10",
    }
    mocker.patch("core.main._find_upcoming_events_by_phone", return_value=[event])
    calendar = mocker.patch("core.main._calendar_client_for_event")
    mocker.patch("core.main.HISTORY.add")
    send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    handled = await _handle_whatsapp_attendance_confirmation("6591234567", reply_text)

    assert handled
    calendar.assert_not_called()
    assert "reply YES only" in send.call_args.args[1]


@pytest.mark.asyncio
async def test_whatsapp_reminder_question_gets_keyword_guidance_and_normal_answer(mocker):
    from core.history import InMemoryHistory
    from core.main import _process_message

    history = InMemoryHistory()
    mocker.patch("core.main.HISTORY", history)
    mocker.patch("core.main.CONFIG", {"modules": {"booking": True}})
    mocker.patch("core.main._find_upcoming_events_by_phone", return_value=[{
        "id": "appointment-1", "start_datetime": "2026-11-10T09:00:00+08:00",
        "reminder_sent_for": "2026-11-10",
    }])
    mocker.patch("core.main.get_ai_response", return_value="The clinic is on Main Street.")
    calendar = mocker.patch("core.main._calendar_client_for_event")
    send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)

    await _process_message("6591234567", "Where is the clinic?")

    assert send.await_count == 2
    assert "reply YES only" in send.await_args_list[0].args[1]
    assert "Main Street" in send.await_args_list[1].args[1]
    calendar.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_text", ["okay", "CONFIRM!", "yes please"])
async def test_whatsapp_booking_rejects_non_exact_then_accepts_confirm(mocker, reply_text, isolated_pending_bookings):
    from core.history import InMemoryHistory
    from core.main import _process_message

    history = InMemoryHistory()
    history.add("6591234567", "assistant", (
        "Please confirm these booking details:\n"
        "- Service: Initial evaluation\n- Date: 2026-11-23\n- Time: 10:00\n"
        "- Patient: Jane\n- Patient DOB: 2020-05-03\n- Parent: Alex\n"
        "- Referral: None\n- Patient resides in Singapore: Yes"
    ))
    mocker.patch("core.main.HISTORY", history)
    isolated_pending_bookings.put("whatsapp:6591234567", booking_details())
    mocker.patch("core.main.CONFIG", {"modules": {"booking": True}})
    mocker.patch("core.main.get_ai_response", return_value='{"intent": "booking_confirmed"}')
    attendance = mocker.patch(
        "core.main._handle_whatsapp_attendance_confirmation", new_callable=mocker.AsyncMock,
    )
    booking = mocker.patch("core.main._handle_booking_intent", new_callable=mocker.AsyncMock)
    send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)

    await _process_message("6591234567", reply_text)
    assert "reply CONFIRM only" in send.await_args.args[1]
    booking.assert_not_awaited()
    attendance.assert_not_awaited()
    await _process_message("6591234567", "CONFIRM")
    booking.assert_awaited_once()


@pytest.mark.asyncio
async def test_whatsapp_booking_correction_gets_guidance_and_normal_answer(mocker, isolated_pending_bookings):
    from core.history import InMemoryHistory
    from core.main import _process_message

    history = InMemoryHistory()
    history.add("6591234567", "assistant", (
        "Please confirm these booking details:\n"
        "- Service: Initial evaluation\n- Date: 2026-11-23\n- Time: 10:00\n"
        "- Patient: Jane\n- Patient DOB: 2020-05-03\n- Parent: Alex\n"
        "- Referral: None\n- Patient resides in Singapore: Yes"
    ))
    mocker.patch("core.main.HISTORY", history)
    isolated_pending_bookings.put("whatsapp:6591234567", booking_details())
    mocker.patch("core.main.CONFIG", {"modules": {"booking": True}})
    mocker.patch("core.main._get_pending_modification", return_value={"id": "existing"})
    ai = mocker.patch("core.main.get_ai_response", return_value="What time would you prefer?")
    send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    booking = mocker.patch("core.main._handle_booking_intent", new_callable=mocker.AsyncMock)
    await _process_message("6591234567", "Could we change the time?")
    assert send.await_count == 2
    assert "no longer awaiting confirmation" in send.await_args_list[0].args[1]
    assert send.await_args_list[1].args[1] == "What time would you prefer?"
    ai.assert_called_once()
    booking.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(("dob", "accepted"), [("2026-10-10", True), ("2026-11-25", False)])
async def test_whatsapp_booking_draft_validates_dob_against_clinic_date(
    mocker, monkeypatch, isolated_pending_bookings, dob, accepted,
):
    from datetime import date
    from core.history import InMemoryHistory
    from core.main import _process_message

    def clinic_today(timezone):
        assert timezone == "Asia/Singapore"
        return date(2026, 11, 24)

    monkeypatch.setattr("core.patient_dates.clinic_today", clinic_today)
    mocker.patch("core.main.HISTORY", InMemoryHistory())
    mocker.patch("core.main.CONFIG", {
        "modules": {"booking": True}, "client": {"timezone": "Asia/Singapore"}, "booking": {},
    })
    mocker.patch("core.main._get_pending_modification", return_value=None)
    mocker.patch("core.main._handle_whatsapp_attendance_confirmation", new_callable=mocker.AsyncMock, return_value=False)
    mocker.patch("core.main.get_ai_response", return_value=json.dumps({
        "intent": "booking_draft", **booking_details(child_dob=dob),
    }))
    send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    await _process_message("6591234567", "Here are my booking details")
    pending = isolated_pending_bookings.get("whatsapp:6591234567")
    assert (pending is not None) == accepted
    if accepted:
        assert f"Patient DOB: {dob}" in send.await_args.args[1]
    else:
        assert "date of birth is in the future" in send.await_args.args[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("approve", [True, False])
async def test_whatsapp_initial_general_observation_requires_separate_approval(
    mocker, isolated_pending_bookings, approve,
):
    from core.history import InMemoryHistory
    from core.main import _process_message

    history = InMemoryHistory()
    history.add("6591234567", "user", (
        "I'd like an appointment for my daughter. She enjoys playing with blocks. "
        "She is not yet crawling."
    ))
    mocker.patch("core.main.HISTORY", history)
    mocker.patch("core.main.CONFIG", {"modules": {"booking": True}, "booking": {}})
    mocker.patch("core.main._get_pending_modification", return_value=None)
    mocker.patch("core.main._handle_whatsapp_attendance_confirmation", new_callable=mocker.AsyncMock, return_value=False)
    mocker.patch("core.main.get_ai_response", return_value=json.dumps({
        "intent": "booking_draft", **booking_details(),
    }))
    send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    booking = mocker.patch("core.main._handle_booking_intent", new_callable=mocker.AsyncMock)
    await _process_message("6591234567", "Here are the rest of the booking details")
    summary = send.await_args.args[1]
    assert "Proposed note for consultant (not yet approved): She enjoys playing with blocks" in summary
    assert "not yet crawling" not in summary
    await _process_message("6591234567", "CONFIRM")
    booking.assert_not_awaited()
    await _process_message("6591234567", "APPROVE NOTE" if approve else "REMOVE NOTE")
    booking.assert_not_awaited()
    await _process_message("6591234567", "CONFIRM")
    if approve:
        assert booking.await_args.args[1]["observations"] == "She enjoys playing with blocks"
    else:
        assert "observations" not in booking.await_args.args[1]


@pytest.mark.asyncio
async def test_whatsapp_note_is_reviewed_before_confirmation(mocker, isolated_pending_bookings):
    from core.history import InMemoryHistory
    from core.main import _process_message

    mocker.patch("core.main.HISTORY", InMemoryHistory())
    mocker.patch("core.main.CONFIG", {"modules": {"booking": True}})
    isolated_pending_bookings.put("whatsapp:6591234567", booking_details())
    ai = mocker.patch("core.main.get_ai_response")
    send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    booking = mocker.patch("core.main._handle_booking_intent", new_callable=mocker.AsyncMock)
    await _process_message("6591234567", "NOTE: She enjoys playing with blocks")
    assert "- Proposed note for consultant (not yet approved): She enjoys playing with blocks" in send.await_args.args[1]
    booking.assert_not_awaited()
    ai.assert_not_called()
    await _process_message("6591234567", "CONFIRM")
    booking.assert_not_awaited()
    await _process_message("6591234567", "APPROVE NOTE")
    booking.assert_not_awaited()
    await _process_message("6591234567", "CONFIRM")
    assert booking.await_args.args[1]["observations"] == "She enjoys playing with blocks"


@pytest.mark.asyncio
async def test_whatsapp_note_survives_updated_draft_for_same_patient(mocker, isolated_pending_bookings):
    from core.history import InMemoryHistory
    from core.main import _process_message

    mocker.patch("core.main.HISTORY", InMemoryHistory())
    mocker.patch("core.main.CONFIG", {"modules": {"booking": True}, "booking": {}})
    mocker.patch("core.main._get_pending_modification", return_value=None)
    isolated_pending_bookings.put("whatsapp:6591234567", booking_details())
    isolated_pending_bookings.set_observations("whatsapp:6591234567", "Enjoys playing")
    isolated_pending_bookings.approve_observations("whatsapp:6591234567")
    mocker.patch("core.main.get_ai_response", return_value=json.dumps({
        "intent": "booking_draft", **booking_details(time="12:00", observations="fabricated"),
    }))
    mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    booking = mocker.patch("core.main._handle_booking_intent", new_callable=mocker.AsyncMock)
    await _process_message("6591234567", "Could we change the time to 12:00?")
    await _process_message("6591234567", "CONFIRM")
    assert booking.await_args.args[1]["observations"] == "Enjoys playing"
    assert booking.await_args.args[1]["time"] == "12:00"


@pytest.mark.asyncio
async def test_payment_flow_preserves_optional_observations(mocker):
    from datetime import date, time
    from core.main import _handle_payment_flow
    from modules.booking.calendar import Slot

    payment = mocker.MagicMock()
    payment.create_preference.return_value = {"payment_url": "https://example.test"}
    mocker.patch("core.main._get_mp_client", return_value=payment)
    save = mocker.patch("core.main._save_pending_payment")
    mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    mocker.patch("core.main.HISTORY")
    calendar = mocker.MagicMock()
    intent = {
        "parent_name": "Alex", "child_name": "Jane", "referral_name": "None",
        "observations": "Enjoys playing",
    }
    await _handle_payment_flow(
        "6591234567", intent, "Checking", {"name": "Evaluation", "price": 100, "duration_minutes": 60},
        {"name": "Main", "address": "Street"},
        Slot(date=date(2027, 3, 15), start_time=time(10), location="Main"),
        calendar, "Child (0-12)",
    )
    assert save.call_args.args[1]["observations"] == "Enjoys playing"


@pytest.mark.asyncio
async def test_whatsapp_saved_summary_survives_question_and_books_saved_values(mocker, isolated_pending_bookings):
    from core.history import InMemoryHistory
    from core.main import _process_message

    history = InMemoryHistory()
    mocker.patch("core.main.HISTORY", history)
    mocker.patch("core.main.CONFIG", {"modules": {"booking": True}})
    mocker.patch("core.main._get_pending_modification", return_value=None)
    mocker.patch("core.main._handle_whatsapp_attendance_confirmation", new_callable=mocker.AsyncMock, return_value=False)
    ai = mocker.patch("core.main.get_ai_response", side_effect=[
        json.dumps({"intent": "booking_draft", **booking_details()}),
        "The clinic is on Main Street.",
    ])
    send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    booking = mocker.patch("core.main._handle_booking_intent", new_callable=mocker.AsyncMock)

    await _process_message("6591234567", "Here are my details")
    assert "- Time: 10:00" in send.await_args.args[1]
    await _process_message("6591234567", "Where is the clinic?")
    assert "Main Street" in send.await_args.args[1]
    await _process_message("6591234567", "CONFIRM")
    assert booking.await_args.args[1]["time"] == "10:00"
    assert ai.call_count == 2
    await _process_message("6591234567", "CONFIRM")
    booking.assert_awaited_once()
    assert "no active booking summary" in send.await_args.args[1]


@pytest.mark.asyncio
async def test_whatsapp_change_invalidates_old_summary_until_new_draft(mocker, isolated_pending_bookings):
    from core.history import InMemoryHistory
    from core.main import _process_message

    mocker.patch("core.main.HISTORY", InMemoryHistory())
    mocker.patch("core.main.CONFIG", {"modules": {"booking": True}})
    mocker.patch("core.main._get_pending_modification", return_value=None)
    isolated_pending_bookings.put("whatsapp:6591234567", booking_details())
    mocker.patch("core.main.get_ai_response", return_value="What time would you prefer?")
    send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    booking = mocker.patch("core.main._handle_booking_intent", new_callable=mocker.AsyncMock)
    await _process_message("6591234567", "Could we change the time?")
    await _process_message("6591234567", "CONFIRM")
    booking.assert_not_awaited()
    assert "no active booking summary" in send.await_args.args[1]


@pytest.mark.asyncio
async def test_whatsapp_new_draft_requires_confirmation_and_preserves_reschedule(mocker, isolated_pending_bookings):
    from core.history import InMemoryHistory
    from core.main import _process_message

    mocker.patch("core.main.HISTORY", InMemoryHistory())
    mocker.patch("core.main.CONFIG", {
        "client": {"timezone": "Asia/Singapore"},
        "modules": {"booking": True},
        "booking": {},
    })
    mocker.patch("core.main._get_pending_modification", return_value=None)
    mocker.patch("core.main._delete_pending_modification")
    from datetime import datetime, timedelta
    import pytz
    modification = {
        "id": "old",
        "patient_name": "Jane",
        "date": "2026-11-20",
        "time": "11:00",
        "start_datetime": (
            datetime.now(pytz.timezone("Asia/Singapore")) + timedelta(days=3)
        ).isoformat(),
    }
    isolated_pending_bookings.put("whatsapp:6591234567", booking_details(), modification)
    mocker.patch("core.main.get_ai_response", return_value=json.dumps({
        "intent": "booking_draft", **booking_details(time="12:00"),
    }))
    mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    booking = mocker.patch("core.main._handle_booking_intent", new_callable=mocker.AsyncMock)
    await _process_message("6591234567", "Could we change the time to 12:00?")
    booking.assert_not_awaited()
    assert "- Time: 12:00" in isolated_pending_bookings.get("whatsapp:6591234567").summary
    await _process_message("6591234567", "CONFIRM")
    assert booking.await_args.args[1]["time"] == "12:00"
    assert booking.await_args.kwargs["pending_modification"] == modification


@pytest.mark.asyncio
async def test_whatsapp_cancel_pending_does_not_cancel_existing_event(mocker, isolated_pending_bookings):
    from core.history import InMemoryHistory
    from core.main import _process_message

    mocker.patch("core.main.HISTORY", InMemoryHistory())
    mocker.patch("core.main._delete_pending_modification")
    isolated_pending_bookings.put("whatsapp:6591234567", booking_details())
    ai = mocker.patch("core.main.get_ai_response")
    booking = mocker.patch("core.main._handle_booking_intent", new_callable=mocker.AsyncMock)
    send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    await _process_message("6591234567", "Cancel this booking")
    assert isolated_pending_bookings.get("whatsapp:6591234567") is None
    assert "no appointment was booked" in send.await_args.args[1]
    booking.assert_not_awaited()
    ai.assert_not_called()


@pytest.mark.asyncio
async def test_whatsapp_reminder_lookup_failure_is_reported(mocker):
    from core.main import _handle_whatsapp_attendance_confirmation
    mocker.patch("core.main._find_upcoming_events_by_phone", side_effect=RuntimeError("unavailable"))
    send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    assert await _handle_whatsapp_attendance_confirmation("6591234567", "YES")
    assert "couldn't check" in send.await_args.args[1]
    assert "hasn't been confirmed" in send.await_args.args[1]


@pytest.mark.asyncio
async def test_whatsapp_yes_does_not_confirm_when_multiple_reminders_are_pending(mocker):
    from core.main import _handle_whatsapp_attendance_confirmation

    events = [
        {
            "id": f"appointment-{index}",
            "start_datetime": f"2026-11-{index + 10}T09:00:00+08:00",
            "date": f"Tuesday November {index + 10}, 2026",
            "time": "09:00",
            "patient_name": "Jane Example",
            "reminder_sent_for": f"2026-11-{index + 10}",
        }
        for index in range(2)
    ]
    mocker.patch("core.main._find_upcoming_events_by_phone", return_value=events)
    calendar = mocker.MagicMock()
    mocker.patch("core.main._calendar_client_for_event", return_value=calendar)
    mocker.patch("core.main.HISTORY.add")
    send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)

    handled = await _handle_whatsapp_attendance_confirmation("6591234567", "yes")

    assert handled
    calendar.mark_event_attendance_confirmed.assert_not_called()
    assert "more than one appointment" in send.call_args.args[1]


@pytest.mark.asyncio
async def test_booking_intent_uses_patient_selected_consultant_calendar(mocker):
    from core.main import _handle_booking_intent

    mocker.patch("core.main.CONFIG", {
        "client": {"timezone": "Asia/Singapore"},
        "modules": {"booking": True, "payments": False},
        "booking": {
            "consultants": [
                {"name": "Consultant One", "calendar_id": "one@calendar"},
                {"name": "Consultant Two", "calendar_id": "two@calendar"},
            ],
            "business_hours": {"start": "08:00", "end": "18:00"},
            "services": [{"name": "Initial evaluation", "duration_minutes": 60, "price": 300}],
            "locations": [{
                "name": "Clinic",
                "address": "Clinic address",
                "days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
            }],
        },
    })
    mocker.patch("core.main.HISTORY.add")
    mocker.patch("core.main._get_pending_modification", return_value=None)
    mock_send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    calendar_one = mocker.MagicMock()
    calendar_one.has_upcoming_appointment_for_patient.return_value = False
    calendar_two = mocker.MagicMock()
    calendar_two.is_slot_available.return_value = True
    calendar_two.has_upcoming_appointment_for_patient.return_value = False
    get_calendar = mocker.patch(
        "core.main._get_calendar_client",
        side_effect=lambda name=None: {
            "Consultant One": calendar_one,
            "Consultant Two": calendar_two,
        }[name],
    )

    await _handle_booking_intent(
        "6591234567",
        {
            "intent": "booking_confirmed",
            "consultant": "Consultant Two",
            "service": "Initial evaluation",
            "location": "Clinic",
            "date": "2027-03-15",
            "time": "10:00",
            "child_name": "Child Example",
            "child_dob": "2020-05-03",
            "parent_name": "Parent Example",
            "referral_name": "None",
            "patient_resides_in_singapore": "yes",
        },
        "Checking the appointment.",
    )

    get_calendar.assert_any_call("Consultant Two")
    calendar_one.create_event.assert_not_called()
    calendar_two.create_event.assert_called_once()
    assert "Appointment confirmed" in mock_send.call_args.args[1]


@pytest.mark.asyncio
async def test_booking_intent_allocates_available_consultant_without_preference(mocker):
    from datetime import datetime, timedelta

    import pytz

    from core.main import _handle_booking_intent

    timezone = pytz.timezone("Asia/Singapore")
    today = datetime.now(timezone).date()
    days_until_monday = (0 - today.weekday()) % 7 or 7
    booking_date = today + timedelta(days=days_until_monday)
    mocker.patch("core.main.CONFIG", {
        "client": {"timezone": "Asia/Singapore"},
        "modules": {"booking": True, "payments": False},
        "booking": {
            "consultants": [
                {"name": "Consultant Sarah", "calendar_id": "sarah@calendar"},
                {"name": "Consultant Serena", "calendar_id": "serena@calendar"},
            ],
            "business_hours": {"start": "08:00", "end": "18:00"},
            "services": [{"name": "Initial evaluation", "duration_minutes": 60, "price": 300}],
            "locations": [{
                "name": "Clinic",
                "address": "Clinic address",
                "days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
            }],
        },
    })
    mocker.patch("core.main.HISTORY.add")
    mocker.patch("core.main.HISTORY.get", return_value=[])
    mocker.patch("core.main._get_pending_modification", return_value=None)
    mocker.patch("core.main._patient_has_upcoming_appointment", return_value=False)
    mock_send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    calendar_sarah = mocker.MagicMock()
    calendar_sarah.is_slot_available.return_value = True
    calendar_serena = mocker.MagicMock()
    calendar_serena.is_slot_available.return_value = True
    get_calendar = mocker.patch(
        "core.main._get_calendar_client",
        side_effect=lambda name=None: {
            "Consultant Sarah": calendar_sarah,
            "Consultant Serena": calendar_serena,
        }[name],
    )
    mocker.patch(
        "core.main.allocate_consultant",
        return_value={"name": "Consultant Serena", "calendar_id": "serena@calendar"},
    )

    await _handle_booking_intent(
        "6591234567",
        {
            "intent": "booking_confirmed",
            "service": "Initial evaluation",
            "location": "Clinic",
            "date": booking_date.isoformat(),
            "time": "10:00",
            "child_name": "Child Example",
            "child_dob": "2020-05-03",
            "parent_name": "Parent Example",
            "referral_name": "None",
            "patient_resides_in_singapore": "yes",
        },
        "Checking the appointment.",
    )

    get_calendar.assert_any_call("Consultant Sarah")
    get_calendar.assert_any_call("Consultant Serena")
    calendar_sarah.create_event.assert_not_called()
    calendar_serena.create_event.assert_called_once()
    assert "Appointment confirmed" in mock_send.call_args.args[1]
    assert "with Consultant Serena" in mock_send.call_args.args[1]


@pytest.mark.asyncio
async def test_booking_intent_rejects_consultant_reserved_time(mocker):
    from datetime import datetime, timedelta

    import pytz

    from core.main import _handle_booking_intent

    timezone = pytz.timezone("Asia/Singapore")
    today = datetime.now(timezone).date()
    days_until_monday = (0 - today.weekday()) % 7 or 7
    booking_date = today + timedelta(days=days_until_monday)
    mocker.patch("core.main.CONFIG", {
        "client": {"timezone": "Asia/Singapore"},
        "modules": {"booking": True, "payments": False},
        "booking": {
            "consultants": [
                {"name": "Consultant Sarah", "calendar_id": "sarah@calendar"},
                {"name": "Consultant Serena", "calendar_id": "serena@calendar"},
            ],
            "business_hours": {"start": "08:00", "end": "18:00"},
            "services": [{"name": "Initial evaluation", "duration_minutes": 60, "price": 300}],
            "locations": [{
                "name": "Clinic",
                "address": "Clinic address",
                "days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
            }],
        },
    })
    mocker.patch("core.main.HISTORY.add")
    mocker.patch("core.main._get_pending_modification", return_value=None)
    mock_send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    mock_calendar = mocker.MagicMock()
    get_calendar = mocker.patch("core.main._get_calendar_client", return_value=mock_calendar)

    await _handle_booking_intent(
        "6591234567",
        {
            "intent": "booking_confirmed",
            "consultant": "Sarah",
            "service": "Initial evaluation",
            "location": "Clinic",
            "date": booking_date.isoformat(),
            "time": "16:00",
            "child_name": "Child Example",
            "child_dob": "2020-05-03",
            "parent_name": "Parent Example",
            "referral_name": "None",
            "patient_resides_in_singapore": "yes",
        },
        "Checking the appointment.",
    )

    get_calendar.assert_called_once_with("Consultant Sarah")
    mock_calendar.is_slot_available.assert_not_called()
    mock_calendar.create_event.assert_not_called()
    assert mock_send.call_args.args[1] == (
        "That time isn't available with Consultant Sarah. "
        "Please choose another date and time."
    )


@pytest.mark.asyncio
async def test_booking_intent_requires_child_parent_dob_and_referral(mocker):
    from core.main import _handle_booking_intent

    mocker.patch("core.main.CONFIG", {
        "client": {"timezone": "Asia/Singapore"},
        "modules": {"booking": True, "payments": False},
        "booking": {
            "services": [{"name": "Initial evaluation", "duration_minutes": 60, "price": 300}],
            "locations": [{"name": "Clinic", "address": "Clinic address", "days": ["monday"]}],
        },
    })
    mocker.patch("core.main.HISTORY.add")
    mock_send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    mock_calendar = mocker.MagicMock()
    mocker.patch("core.main._get_calendar_client", return_value=mock_calendar)

    await _handle_booking_intent(
        "6591234567",
        {
            "intent": "booking_confirmed",
            "service": "Initial evaluation",
            "location": "Clinic",
            "date": "2027-03-15",
            "time": "10:00",
            "child_name": "Child Example",
            "parent_name": "Parent Example",
        },
        "Checking the appointment.",
    )

    mock_calendar.create_event.assert_not_called()
    assert "referring person's name" in mock_send.call_args.args[1]


@pytest.mark.asyncio
async def test_booking_intent_routes_non_singapore_resident_to_admin(mocker):
    from core.main import _handle_booking_intent

    mocker.patch("core.main.CONFIG", {
        "client": {"timezone": "Asia/Singapore"},
        "modules": {"booking": True, "payments": False},
        "booking": {
            "business_hours": {"start": "08:00", "end": "18:00"},
            "services": [{"name": "Initial evaluation", "duration_minutes": 60, "price": 300}],
            "locations": [{"name": "Clinic", "address": "Clinic address", "days": ["monday"]}],
        },
    })
    mocker.patch("core.main.HISTORY.add")
    mock_send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    mock_calendar = mocker.MagicMock()
    mocker.patch("core.main._get_calendar_client", return_value=mock_calendar)

    await _handle_booking_intent(
        "6591234567",
        {
            "intent": "booking_confirmed",
            "service": "Initial evaluation",
            "location": "Clinic",
            "date": "2027-03-15",
            "time": "10:00",
            "child_name": "Child Example",
            "child_dob": "2020-05-03",
            "parent_name": "Parent Example",
            "referral_name": "None",
            "patient_resides_in_singapore": "no",
        },
        "Checking the appointment.",
    )

    mock_calendar.create_event.assert_not_called()
    assert mock_send.call_args.args[1] == (
        "Our administrative team will assist you with this booking."
    )


@pytest.mark.asyncio
async def test_booking_intent_routes_second_appointment_to_admin(mocker):
    from core.main import _handle_booking_intent

    mocker.patch("core.main.CONFIG", {
        "client": {"timezone": "Asia/Singapore"},
        "modules": {"booking": True, "payments": True},
        "booking": {
            "business_hours": {"start": "08:00", "end": "18:00"},
            "services": [{"name": "Initial evaluation", "duration_minutes": 60, "price": 300}],
            "locations": [{"name": "Clinic", "address": "Clinic address", "days": ["monday"]}],
        },
    })
    mocker.patch("core.main.HISTORY.add")
    mock_send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    mock_calendar = mocker.MagicMock()
    mock_calendar.is_slot_available.return_value = True
    mock_calendar.has_upcoming_appointment_for_patient.return_value = True
    mocker.patch("core.main._get_calendar_client", return_value=mock_calendar)

    await _handle_booking_intent(
        "6591234567",
        {
            "intent": "booking_confirmed",
            "service": "Initial evaluation",
            "location": "Clinic",
            "date": "2027-03-15",
            "time": "10:00",
            "child_name": "Child Example",
            "child_dob": "2020-05-03",
            "parent_name": "Parent Example",
            "referral_name": "None",
            "patient_resides_in_singapore": "yes",
        },
        "Checking the appointment.",
    )

    mock_calendar.create_event.assert_not_called()
    assert mock_send.call_args.args[1] == (
        "Our administrative team will assist you with this booking."
    )


@pytest.mark.asyncio
async def test_booking_intent_cannot_bypass_saturday_reschedule_handoff(mocker):
    from core.main import _handle_booking_intent

    mocker.patch("core.main.CONFIG", {
        "client": {"timezone": "Asia/Singapore"},
        "modules": {"booking": True, "payments": False},
        "booking": {
            "business_hours": {"start": "08:00", "end": "18:00"},
            "services": [{"name": "Initial evaluation", "duration_minutes": 60, "price": 300}],
            "locations": [{"name": "Clinic", "address": "Clinic address", "days": ["monday"]}],
        },
    })
    mocker.patch("core.main.HISTORY.add")
    mocker.patch("core.main.HISTORY.get", return_value=[])
    mocker.patch("core.main._get_pending_modification", return_value=None)
    mocker.patch("core.main._conversation_suggests_modification", return_value=True)
    mock_send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)
    mock_calendar = mocker.MagicMock()
    mock_calendar.is_slot_available.return_value = True
    mock_calendar.find_upcoming_events_by_phone.return_value = [{
        "id": "saturday-event",
        "date": "Saturday November 7, 2026",
    }]
    mocker.patch("core.main._get_calendar_client", return_value=mock_calendar)

    await _handle_booking_intent(
        "6591234567",
        {
            "intent": "booking_confirmed",
            "service": "Initial evaluation",
            "location": "Clinic",
            "date": "2027-03-15",
            "time": "10:00",
            "child_name": "Child Example",
            "child_dob": "2020-05-03",
            "parent_name": "Parent Example",
            "referral_name": "None",
            "patient_resides_in_singapore": "yes",
        },
        "Checking the appointment.",
    )

    mock_calendar.delete_event.assert_not_called()
    mock_calendar.create_event.assert_not_called()
    assert "Saturday appointment changes must be handled" in mock_send.call_args.args[1]


@pytest.mark.asyncio
async def test_modification_request_states_weekday_policy(mocker):
    from core.main import _handle_modification_request
    from datetime import datetime, timedelta
    import pytz

    mocker.patch("core.main.CONFIG", {
        "client": {"timezone": "Asia/Singapore"},
        "modules": {"booking": True},
        "booking": {"calendar_id": "test@calendar"},
    })
    mocker.patch("core.main.HISTORY.add")
    mock_calendar = mocker.patch("core.main._get_calendar_client").return_value
    mock_calendar.find_upcoming_events_by_phone.return_value = [{
        "id": "weekday-event",
        "summary": "Initial evaluation",
        "date": "Monday November 9, 2026",
        "time": "10:00",
        "start_datetime": (
            datetime.now(pytz.timezone("Asia/Singapore")) + timedelta(days=3)
        ).isoformat(),
        "location": "",
    }]
    save_pending = mocker.patch("core.main._save_pending_modification")
    mock_send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)

    await _handle_modification_request("6591234567", "I can help with that.")

    assert "more than 36 hours before the appointment" in mock_send.call_args.args[1]
    assert "updated booking details" in mock_send.call_args.args[1]
    save_pending.assert_called_once_with("6591234567", mock_calendar.find_upcoming_events_by_phone.return_value[0])


@pytest.mark.asyncio
async def test_modification_within_36_hours_requires_mc_before_rescheduling(mocker):
    from core.main import _handle_modification_request
    from datetime import datetime, timedelta
    import pytz

    mocker.patch("core.main.CONFIG", {
        "client": {"timezone": "Asia/Singapore"},
        "modules": {"booking": True},
        "booking": {"calendar_id": "test@calendar"},
    })
    mocker.patch("core.main.HISTORY.add")
    mock_calendar = mocker.patch("core.main._get_calendar_client").return_value
    event = {
        "id": "weekday-event",
        "summary": "Initial evaluation",
        "date": "Wednesday November 25, 2026",
        "time": "10:00",
        "start_datetime": (
            datetime.now(pytz.timezone("Asia/Singapore")) + timedelta(hours=24)
        ).isoformat(),
        "location": "",
    }
    mock_calendar.find_upcoming_events_by_phone.return_value = [event]
    save_pending = mocker.patch("core.main._save_pending_modification")
    mock_send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)

    await _handle_modification_request("6591234567", "I can help with that.")

    assert "valid medical certificate (MC)" in mock_send.call_args.args[1]
    assert "reply YES or NO" in mock_send.call_args.args[1]
    save_pending.assert_called_once()
    assert event["awaiting_mc_confirmation"] is True


@pytest.mark.asyncio
async def test_no_mc_starts_cancel_and_rebook_flow(mocker):
    from core.main import _handle_rescheduling_mc_response

    event = {
        "id": "weekday-event",
        "awaiting_mc_confirmation": True,
    }
    mocker.patch("core.main._get_pending_modification", return_value=event)
    delete_pending = mocker.patch("core.main._delete_pending_modification")
    cancellation = mocker.patch(
        "core.main._handle_cancellation_request",
        new_callable=mocker.AsyncMock,
    )

    await _handle_rescheduling_mc_response(
        "6591234567",
        {"has_valid_mc": False},
    )

    delete_pending.assert_called_once_with("6591234567")
    assert "new booking from scratch" in cancellation.await_args.args[1]
    assert "all booking details again" in cancellation.await_args.args[1]


@pytest.mark.asyncio
async def test_modification_request_routes_saturday_to_admin(mocker):
    from core.main import _handle_modification_request

    mocker.patch("core.main.CONFIG", {
        "modules": {"booking": True},
        "booking": {"calendar_id": "test@calendar"},
    })
    mocker.patch("core.main.HISTORY.add")
    mock_calendar = mocker.patch("core.main._get_calendar_client").return_value
    mock_calendar.find_upcoming_events_by_phone.return_value = [{
        "id": "saturday-event",
        "summary": "Follow-up intervention",
        "date": "Saturday November 7, 2026",
        "time": "10:00",
        "location": "",
    }]
    save_pending = mocker.patch("core.main._save_pending_modification")
    mock_send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)

    await _handle_modification_request("6591234567", "I can help with that.")

    response = mock_send.call_args.args[1]
    assert "prepaid at SGD 300" in response
    assert "must be handled by the administrative team" in response
    save_pending.assert_not_called()


@pytest.mark.asyncio
async def test_rescheduling_handoff_clears_pending_state_and_asks_patient_to_contact_clinic(mocker):
    from core.main import _handle_rescheduling_handoff

    mocker.patch("core.main.HISTORY.add")
    delete_modification = mocker.patch("core.main._delete_pending_modification")
    delete_cancellation = mocker.patch("core.main._delete_pending_cancellation")
    mock_send = mocker.patch("core.main.WA.send_text", new_callable=mocker.AsyncMock)

    await _handle_rescheduling_handoff("6591234567")

    delete_modification.assert_called_once_with("6591234567")
    delete_cancellation.assert_called_once_with("6591234567")
    assert "administrative team can assist" in mock_send.call_args.args[1]
    assert "Please contact the clinic directly" in mock_send.call_args.args[1]
