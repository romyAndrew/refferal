"""Telethon client construction, event handling, and application lifecycle.

No Telegram/network access happens at import time; a connection is only opened
when :meth:`TelegramRunner.start` / :meth:`TelegramRunner.run_forever` is
explicitly awaited.

The handler emits a minimal technical log record (chat id, message id,
timestamp, text length) and deliberately keeps the message text out of the logs.
When a :class:`~referral_finder.ingest.MessageWriter` is attached, text-bearing
messages are also forwarded to it for persistence; messages without text or a
caption are skipped. No media is downloaded and no extra profile data is
requested.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable, Optional, Tuple

from telethon import TelegramClient, events

from .config import Config, ConfigError
from .ingest import MessageWriter, MessageWriterError
from .models import IncomingMessage
from .storage import MessageStore

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


async def handle_new_message(event, writer: Optional[MessageWriter] = None) -> None:
    """Handle a new incoming message.

    Logs a minimal technical record and never the message text. Messages without
    text or a caption are skipped (per SPEC FR-11). When ``writer`` is provided,
    text-bearing messages are queued for persistence. Extraction failures are
    logged explicitly so they are not silently swallowed.
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

    if message.text is None:
        logger.debug(
            "Skipping message without text chat_id=%s message_id=%s",
            message.chat_id,
            message.message_id,
        )
        return

    if writer is None:
        return

    # Backpressure: awaits if the bounded queue is full instead of dropping.
    try:
        await writer.submit(message)
    except MessageWriterError:
        # The writer has failed or stopped; the message cannot be persisted.
        # Log it explicitly (identifiers only) so the loss is not silent.
        logger.exception(
            "Message writer unavailable; message not persisted "
            "chat_id=%s message_id=%s",
            message.chat_id,
            message.message_id,
        )
    except asyncio.CancelledError:
        # Telethon cancels in-flight handler tasks on disconnect. The message
        # was not queued; log it (identifiers only) and propagate cancellation.
        logger.warning(
            "Message handling cancelled during shutdown; message not persisted "
            "chat_id=%s message_id=%s",
            message.chat_id,
            message.message_id,
        )
        raise


def make_message_handler(
    writer: Optional[MessageWriter],
) -> Callable[[object], Awaitable[None]]:
    """Bind ``writer`` to :func:`handle_new_message` for Telethon registration.

    Telethon calls handlers with a single ``event`` argument, so the writer is
    captured in a closure rather than passed by the framework.
    """

    async def handler(event) -> None:
        await handle_new_message(event, writer=writer)

    return handler


class TelegramRunner:
    """Owns a :class:`TelegramClient`, the message writer, and the lifecycle.

    The client, writer, and store are injectable so lifecycle ordering and
    shutdown can be tested without any network access. When a ``writer`` is
    supplied the inbound handler forwards text messages to it; otherwise the
    handler only logs.
    """

    def __init__(
        self,
        config: Config,
        client: Optional[TelegramClient] = None,
        writer: Optional[MessageWriter] = None,
        store: Optional[MessageStore] = None,
    ):
        self._config = config
        self._client = client if client is not None else create_client(config)
        self._writer = writer
        self._store = store
        self._registered = False

    @property
    def client(self) -> TelegramClient:
        return self._client

    @property
    def writer(self) -> Optional[MessageWriter]:
        return self._writer

    def register(self, handler=None) -> None:
        """Register the inbound-message handler.

        Must happen before :meth:`start` (and therefore before ``catch_up``) so
        that recovered updates are processed by our handler. When no explicit
        ``handler`` is passed and a writer is attached, the handler is bound to
        that writer.
        """
        if handler is None:
            handler = make_message_handler(self._writer)
        chat_ids = parse_chat_ids(self._config.chats)
        builder = new_message_builder(chat_ids)
        self._client.add_event_handler(handler, builder)
        self._registered = True
        logger.info(
            "Registered NewMessage handler for %d chat(s).", len(chat_ids)
        )

    async def start(self) -> None:
        """Start the writer, connect, authorize if needed, and catch up."""
        if self._writer is not None:
            await self._writer.start()
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
        """Stop the pipeline and release resources, in a safe order.

        Order matters, and is driven by how Telethon actually behaves:

        1. **Disconnect the client first.** This stops Telegram update delivery
           (``_update_loop`` runs only while connected) and Telethon cancels and
           awaits any in-flight event-handler tasks. After this returns, no new
           call to :meth:`MessageWriter.submit` can happen.
        2. **Stop the writer.** Everything already accepted into the queue is
           drained and persisted before the writer task ends.
        3. **Close the SQLite store last**, only once the writer has finished.

        Each step is isolated, and a pending cancellation is remembered and
        re-raised at the end, so one failing component does not skip the others.
        """
        cancellation: Optional[BaseException] = None

        try:
            if self._client.is_connected():
                await self._client.disconnect()
                logger.info("Telegram client disconnected.")
        except asyncio.CancelledError as exc:  # remember, keep cleaning up
            cancellation = cancellation or exc
        except Exception:
            logger.exception("Error while disconnecting the Telegram client.")

        if self._writer is not None:
            try:
                await self._writer.stop()
            except asyncio.CancelledError as exc:
                cancellation = cancellation or exc
            except Exception:
                logger.exception("Error while stopping the message writer.")

        if self._store is not None:
            try:
                self._store.close()
            except Exception:
                logger.exception("Error while closing the message store.")

        if cancellation is not None:
            raise cancellation
