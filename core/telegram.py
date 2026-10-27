import asyncio
import json
import logging
import os

import anthropic
import httpx
from dotenv import load_dotenv

from config.loader import load_config
from core.ai import get_ai_response, load_knowledge
from core.history import get_history

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"


def parse_allowed_user_ids(value: str) -> set[int]:
    try:
        user_ids = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError("TELEGRAM_ALLOWED_USER_IDS must be comma-separated numeric IDs") from exc
    if not user_ids:
        raise ValueError("Set TELEGRAM_ALLOWED_USER_IDS to your numeric Telegram user ID")
    return user_ids


async def _telegram_request(
    client: httpx.AsyncClient,
    token: str,
    method: str,
    *,
    params: dict | None = None,
    payload: dict | None = None,
):
    response = await client.post(
        f"{TELEGRAM_API_BASE}/bot{token}/{method}",
        params=params,
        json=payload,
    )
    response.raise_for_status()
    data = response.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram {method} failed: {data.get('description', 'unknown error')}")
    return data["result"]


async def _reply_to_message(
    client: httpx.AsyncClient,
    token: str,
    chat_id: int,
    text: str,
) -> None:
    await _telegram_request(
        client,
        token,
        "sendMessage",
        payload={"chat_id": chat_id, "text": text},
    )


async def _handle_update(
    update: dict,
    *,
    client: httpx.AsyncClient,
    token: str,
    allowed_user_ids: set[int],
    history,
    config: dict,
    knowledge: str,
) -> None:
    message = update.get("message", {})
    sender = message.get("from", {})
    user_id = sender.get("id")
    chat = message.get("chat", {})
    chat_id = chat.get("id")

    if chat.get("type") != "private" or user_id not in allowed_user_ids or chat_id is None:
        logger.warning("Ignored Telegram message from an unapproved or non-private chat")
        return

    user_text = message.get("text", "").strip()
    if not user_text:
        await _reply_to_message(
            client,
            token,
            chat_id,
            "For this development bot, please send a text message.",
        )
        return

    if user_text == "/start":
        await _reply_to_message(
            client,
            token,
            chat_id,
            "Development bot is ready. Send a fictional test message to try the assistant.",
        )
        return

    conversation = history.get(str(user_id))
    history.add(str(user_id), "user", user_text)
    try:
        answer = await asyncio.to_thread(
            get_ai_response,
            user_text,
            conversation,
            config,
            knowledge,
        )
    except anthropic.APIError:
        logger.exception("Claude request failed while handling a Telegram message")
        await _reply_to_message(
            client,
            token,
            chat_id,
            "Sorry, I couldn't generate a reply just now. Please try again.",
        )
        return

    history.add(str(user_id), "assistant", answer)
    await _reply_to_message(client, token, chat_id, answer)


async def run_bot(token: str, allowed_user_ids: set[int]) -> None:
    config = load_config()
    knowledge = load_knowledge()
    history = get_history()
    offset = None

    async with httpx.AsyncClient(timeout=httpx.Timeout(40.0)) as client:
        bot = await _telegram_request(client, token, "getMe")
        logger.info("Telegram development bot started as @%s", bot.get("username", "unknown"))

        while True:
            try:
                updates = await _telegram_request(
                    client,
                    token,
                    "getUpdates",
                    params={
                        "timeout": 25,
                        "allowed_updates": json.dumps(["message"]),
                        **({"offset": offset} if offset is not None else {}),
                    },
                )
            except (httpx.HTTPError, RuntimeError) as exc:
                logger.error("Telegram polling failed: %s", exc)
                await asyncio.sleep(5)
                continue

            for update in updates:
                offset = update["update_id"] + 1
                try:
                    await _handle_update(
                        update,
                        client=client,
                        token=token,
                        allowed_user_ids=allowed_user_ids,
                        history=history,
                        config=config,
                        knowledge=knowledge,
                    )
                except (httpx.HTTPError, RuntimeError):
                    logger.exception("Failed to handle Telegram update %s", update["update_id"])


async def main() -> None:
    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN in your .env file")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY in your .env file to enable Claude replies")
    allowed_user_ids = parse_allowed_user_ids(os.environ.get("TELEGRAM_ALLOWED_USER_IDS", ""))
    await run_bot(token, allowed_user_ids)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
