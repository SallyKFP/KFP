"""Isolated text-only 360dialog sandbox; never use for patient conversations."""

import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response

from config.loader import load_config
from core.ai import get_ai_response, load_knowledge
from core.history import InMemoryHistory
from core.whatsapp import DialogSandboxClient

logger = logging.getLogger(__name__)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be configured for the 360dialog sandbox.")
    return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    _required_env("ANTHROPIC_API_KEY")
    api_key = _required_env("D360_SANDBOX_API_KEY")
    recipient = _required_env("D360_SANDBOX_RECIPIENT")
    if not re.fullmatch(r"[1-9][0-9]{9,14}", recipient):
        raise RuntimeError("D360_SANDBOX_RECIPIENT must be an international phone number without '+'.")
    config = load_config()
    app.state.config = {
        **config,
        "modules": {**config.get("modules", {}), "booking": False, "payments": False, "reminders": False},
    }
    app.state.knowledge = load_knowledge()
    app.state.history = InMemoryHistory()
    app.state.recipient = recipient
    app.state.processed_ids = set()
    app.state.lock = asyncio.Lock()
    app.state.whatsapp = DialogSandboxClient(api_key, recipient)
    try:
        yield
    finally:
        await app.state.whatsapp.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def health():
    return {"status": "ok", "mode": "360dialog sandbox"}


@app.post("/webhook")
async def webhook(request: Request):
    try:
        data = await request.json()
        changes = [change for entry in data["entry"] for change in entry["changes"]]
        messages = [
            message
            for change in changes if change.get("field") == "messages"
            for message in change["value"].get("messages", [])
        ]
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid webhook payload.")

    for message in messages:
        if not isinstance(message, dict):
            raise HTTPException(status_code=400, detail="Invalid message.")
        sender = message.get("from")
        message_id = message.get("id")
        if sender != request.app.state.recipient:
            raise HTTPException(status_code=403, detail="Unexpected sandbox recipient.")
        if not isinstance(message_id, str) or not message_id:
            raise HTTPException(status_code=400, detail="Missing message ID.")
        async with request.app.state.lock:
            if message_id in request.app.state.processed_ids:
                continue
            if message.get("type") == "text":
                text = message.get("text", {}).get("body")
                if not isinstance(text, str) or not text:
                    raise HTTPException(status_code=400, detail="Invalid text message.")
                history = request.app.state.history.get(sender)
                reply = await asyncio.to_thread(
                    get_ai_response, text, history, request.app.state.config, request.app.state.knowledge
                )
            else:
                text = None
                reply = "The sandbox supports text messages only. Please send a text message."
            try:
                await request.app.state.whatsapp.send_text(sender, reply)
            except Exception:
                logger.exception("Could not send sandbox reply")
                raise HTTPException(status_code=503, detail="Could not send sandbox reply.")
            if text is not None:
                request.app.state.history.add(sender, "user", text)
                request.app.state.history.add(sender, "assistant", reply)
            request.app.state.processed_ids.add(message_id)
    return Response(status_code=200)
