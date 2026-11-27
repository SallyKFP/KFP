import json

import httpx
import pytest
from fastapi.testclient import TestClient

from core.sandbox import app
from core.whatsapp import DialogSandboxClient

RECIPIENT = "6591234567"
@pytest.fixture
def sandbox_client(monkeypatch, mocker):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-ai-key")
    monkeypatch.setenv("D360_SANDBOX_API_KEY", "fake-api-key")
    monkeypatch.setenv("D360_SANDBOX_RECIPIENT", RECIPIENT)
    monkeypatch.delenv("D360_SANDBOX_PLATFORM_SECRET", raising=False)
    config = {"modules": {"booking": True, "payments": True, "reminders": True}}
    mocker.patch("core.sandbox.load_config", return_value=config)
    mocker.patch("core.sandbox.load_knowledge", return_value="fictional knowledge")
    send = mocker.patch.object(DialogSandboxClient, "send_text", new_callable=mocker.AsyncMock)
    ai = mocker.patch("core.sandbox.get_ai_response", return_value="Test reply")
    with TestClient(app) as client:
        yield client, send, ai, config


def _payload(message_id="wamid.one", sender=RECIPIENT):
    return {"entry": [{"changes": [{"field": "messages", "value": {
        "contacts": [{"wa_id": sender, "profile": {"name": "Test"}}],
        "messages": [{"id": message_id, "from": sender, "type": "text",
                      "text": {"body": "Hello"}}],
    }}]}]}


def _post(client, payload):
    return client.post("/webhook", json=payload)


def test_sandbox_receives_unsigned_text_and_replies(sandbox_client):
    client, send, ai, original_config = sandbox_client
    response = _post(client, _payload())
    assert response.status_code == 200
    send.assert_awaited_once_with(RECIPIENT, "Test reply")
    assert ai.call_args.args[2]["modules"] == {
        "booking": False, "payments": False, "reminders": False,
    }
    assert original_config["modules"]["booking"] is True


def test_sandbox_rejects_other_recipients(sandbox_client):
    client, send, ai, _ = sandbox_client
    assert _post(client, _payload(sender="6598765432")).status_code == 403
    send.assert_not_awaited()
    ai.assert_not_called()


def test_sandbox_ignores_duplicate_and_status_updates(sandbox_client):
    client, send, ai, _ = sandbox_client
    payload = _payload()
    assert _post(client, payload).status_code == 200
    assert _post(client, payload).status_code == 200
    status = {"entry": [{"changes": [{"field": "messages", "value": {
        "statuses": [{"id": "test", "status": "delivered"}],
    }}]}]}
    assert _post(client, status).status_code == 200
    assert send.await_count == 1
    assert ai.call_count == 1


def test_sandbox_retries_after_send_failure(sandbox_client):
    client, send, ai, _ = sandbox_client
    send.side_effect = [RuntimeError("delivery error"), None]
    assert _post(client, _payload()).status_code == 503
    assert _post(client, _payload()).status_code == 200
    assert send.await_count == 2
    assert ai.call_args.args[1] == []


def test_sandbox_non_text_does_not_call_ai_or_retrieve_media(sandbox_client):
    client, send, ai, _ = sandbox_client
    payload = _payload()
    message = payload["entry"][0]["changes"][0]["value"]["messages"][0]
    message["type"] = "audio"
    message["audio"] = {"id": "test-media"}
    assert _post(client, payload).status_code == 200
    ai.assert_not_called()
    send.assert_awaited_once_with(
        RECIPIENT, "The sandbox supports text messages only. Please send a text message."
    )


@pytest.mark.parametrize("env_name", [
    "ANTHROPIC_API_KEY", "D360_SANDBOX_API_KEY", "D360_SANDBOX_RECIPIENT",
])
def test_sandbox_refuses_startup_without_required_config(monkeypatch, env_name):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-ai-key")
    monkeypatch.setenv("D360_SANDBOX_API_KEY", "fake-api-key")
    monkeypatch.setenv("D360_SANDBOX_RECIPIENT", RECIPIENT)
    monkeypatch.delenv(env_name)
    with pytest.raises(RuntimeError, match=env_name), TestClient(app):
        pass


def test_sandbox_refuses_invalid_recipient(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-ai-key")
    monkeypatch.setenv("D360_SANDBOX_API_KEY", "fake-api-key")
    monkeypatch.setenv("D360_SANDBOX_RECIPIENT", "+6591234567")
    with pytest.raises(RuntimeError, match="international phone number"), TestClient(app):
        pass


def test_sandbox_text_end_to_end_with_mocked_http(monkeypatch, mocker, httpx_mock):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-ai-key")
    monkeypatch.setenv("D360_SANDBOX_API_KEY", "fake-api-key")
    monkeypatch.setenv("D360_SANDBOX_RECIPIENT", RECIPIENT)
    monkeypatch.delenv("D360_SANDBOX_PLATFORM_SECRET", raising=False)
    mocker.patch("core.sandbox.load_config", return_value={"modules": {}})
    mocker.patch("core.sandbox.load_knowledge", return_value="fictional knowledge")
    mocker.patch("core.sandbox.get_ai_response", return_value="Hello from Sally")
    httpx_mock.add_response(url="https://waba-sandbox.360dialog.io/v1/messages", status_code=201)
    with TestClient(app) as client:
        assert _post(client, _payload()).status_code == 200
    request = httpx_mock.get_request()
    assert json.loads(request.content)["text"]["body"] == "Hello from Sally"
    assert request.headers["D360-API-KEY"] == "fake-api-key"


@pytest.mark.asyncio
async def test_sandbox_client_uses_sandbox_endpoint_and_key(httpx_mock):
    httpx_mock.add_response(url="https://waba-sandbox.360dialog.io/v1/messages", status_code=201)
    client = DialogSandboxClient("fake-api-key", RECIPIENT)
    try:
        await client.send_text(RECIPIENT, "Hello")
        request = httpx_mock.get_request()
        assert request.headers["D360-API-KEY"] == "fake-api-key"
        assert json.loads(request.content) == {
            "messaging_product": "whatsapp", "to": RECIPIENT,
            "type": "text", "text": {"body": "Hello"},
        }
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_sandbox_client_rejects_other_recipient_before_sending(httpx_mock):
    client = DialogSandboxClient("fake-api-key", RECIPIENT)
    try:
        with pytest.raises(ValueError):
            await client.send_text("6598765432", "Hello")
        assert httpx_mock.get_requests() == []
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_sandbox_client_surfaces_api_failure(httpx_mock):
    httpx_mock.add_response(url="https://waba-sandbox.360dialog.io/v1/messages", status_code=429)
    client = DialogSandboxClient("fake-api-key", RECIPIENT)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.send_text(RECIPIENT, "Hello")
    finally:
        await client.aclose()
