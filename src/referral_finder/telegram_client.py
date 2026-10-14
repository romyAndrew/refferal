"""Telethon client construction, event handling, and application lifecycle.

No Telegram/network access happens at import time; a connection is only opened
when :meth:`TelegramRunner.start` / :meth:`TelegramRunner.run_forever` is
explicitly awaited.

Stage 2 does not persist messages. The handler emits a minimal technical log
record (chat id, message id, timestamp, text length) and deliberately keeps the
message text out of the logs. No media is downloaded and no extra profile data
is requested.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from telethon import TelegramClient, events

from .config import Config, ConfigError
from .models import IncomingMessage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MessageLogRecord:
    """Minimal technical log record; intentionally excludes the message text."""

    chat_id: int
    message_id: int
    date_utc: int
    text_length: int


def parse_chat_ids(chats: Iterable[str]) -> Tuple[int, ...]:
    """Convert the configured chat list to integer chat IDs.

    ``config/chats.txt`` stores one numeric ID per line. Non-numeric entries are
    rejected with a clear error instead of being silently ignored.
    """
    ids = []
    for raw in chats:
        try:
            ids.append(int(raw))
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"Invalid chat id in config/chats.txt: {raw!r} (numeric ID expected)"
            ) from exc
    return tuple(ids)


def create_client(config: Config) -> TelegramClient:
    """Create a Telethon client from the configuration.

    This only constructs the client (and its local session file); it does not
    connect to Telegram or authorize the account.
    """
    return TelegramClient(str(config.session_path), config.api_id, config.api_hash)


def new_message_builder(chat_ids: Iterable[int]) -> events.NewMessage:
    """Build the incoming-only ``NewMessage`` filter for the selected chats."""
    ids = list(chat_ids)
    if not ids:
        raise ConfigError(
            "No chats configured. Add numeric chat IDs to config/chats.txt."
        )
    return events.NewMessage(chats=ids, incoming=True)


def extract_message(event) -> IncomingMessage:
    """Map a Telethon ``NewMessage`` event to an :class:`IncomingMessage`.

    Works against any object exposing the same attributes (``chat_id``,
    ``id``, ``date``, ``sender_id``, ``raw_text``), which keeps it testable
    without a real Telegram event.
    """
    text = getattr(event, "raw_text", None)
    sender_id = getattr(event, "sender_id", None)
    date = getattr(event, "date", None)
    date_utc = int(date.timestamp()) if date is not None else 0
    return IncomingMessage(
        chat_id=int(event.chat_id),
        message_id=int(event.id),
        date_utc=date_utc,
        received_at_utc=int(time.time()),
        text=text if text else None,
        sender_id=int(sender_id) if sender_id is not None else None,
    )


def to_log_record(message: IncomingMessage) -> MessageLogRecord:
    """Build the technical log record, keeping only the text length."""
    text_length = len(message.text) if message.text else 0
    return MessageLogRecord(
        chat_id=message.chat_id,
        message_id=message.message_id,
        date_utc=message.date_utc,
        text_length=text_length,
    )


async def handle_new_message(event) -> None:
    """Handle a new incoming message.

    Logs a minimal technical record and never the message text. Extraction
    failures are logged explicitly so they are not silently swallowed.
    """
    try:
        message = extract_message(event)
    except Exception:
        logger.exception("Failed to extract fields from a new message event")
        return

    record = to_log_record(message)
    logger.info(
        "New message: chat_id=%s message_id=%s date_utc=%s text_length=%s",
        record.chat_id,
        record.message_id,
        record.date_utc,
        record.text_length,
    )


class TelegramRunner:
    """Owns a :class:`TelegramClient` and the application lifecycle.

    The client is injectable so lifecycle ordering and shutdown can be tested
    without any network access.
    """

    def __init__(self, config: Config, client: Optional[TelegramClient] = None):
        self._config = config
        self._client = client if client is not None else create_client(config)
        self._registered = False

    @property
    def client(self) -> TelegramClient:
        return self._client

    def register(self, handler=handle_new_message) -> None:
        """Register the inbound-message handler.

        Must happen before :meth:`start` (and therefore before ``catch_up``) so
        that recovered updates are processed by our handler.
        """
        chat_ids = parse_chat_ids(self._config.chats)
        builder = new_message_builder(chat_ids)
        self._client.add_event_handler(handler, builder)
        self._registered = True
        logger.info(
            "Registered NewMessage handler for %d chat(s).", len(chat_ids)
        )

    async def start(self) -> None:
        """Connect, authorize if needed, and catch up on missed updates."""
        if not self._registered:
            self.register()
        await self._client.start()
        # Telethon docs: call catch_up() only after event handlers are
        # registered, so the fetched updates can be processed by our handler.
        await self._client.catch_up()
        logger.info("Telegram client connected and caught up.")

    async def run_forever(self) -> None:
        """Register, connect, recover, then wait until disconnected."""
        await self.start()
        await self._client.run_until_disconnected()

    async def shutdown(self) -> None:
        """Disconnect the client if it is still connected."""
        try:
            if self._client.is_connected():
                await self._client.disconnect()
                logger.info("Telegram client disconnected.")
        except Exception:
            logger.exception("Error while shutting down the Telegram client.")
