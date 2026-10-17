"""Telethon client construction, event handling, and application lifecycle.

No Telegram/network access happens at import time; a connection is only opened
when :meth:`TelegramRunner.start` / :meth:`TelegramRunner.run_forever` is
explicitly awaited.

The handler logs only the timestamp, text length, and topic status.
Chat and message IDs and message text are deliberately excluded from logs.
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

from .config import Config, ConfigError, parse_chat_entries
from .history import sync_history
from .ingest import MessageWriter, MessageWriterError
from .models import IncomingMessage, build_incoming, incoming_from_object
from .storage import MessageStore
from .topics import TopicResolver

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
    return incoming_from_object(event, received_at_utc=int(time.time()))


def to_log_record(message: IncomingMessage) -> MessageLogRecord:
    """Build the technical log record, keeping only the text length."""
    text_length = len(message.text) if message.text else 0
    return MessageLogRecord(
        chat_id=message.chat_id,
        message_id=message.message_id,
        date_utc=message.date_utc,
        text_length=text_length,
    )


async def handle_new_message(
    event,
    writer: Optional[MessageWriter] = None,
    chat_entity=None,
    topic_resolver: Optional[TopicResolver] = None,
) -> None:
    """Handle a new incoming message.

    Logs a minimal technical record and never the message text. Messages without
    text or a caption are skipped (per SPEC FR-11). When ``writer`` is provided,
    text-bearing messages are queued for persistence. Extraction failures are
    logged explicitly so they are not silently swallowed. ``chat_entity`` and
    ``topic_resolver`` are optional and only used to enrich forum metadata.
    """
    try:
        message = await build_incoming(
            event,
            received_at_utc=int(time.time()),
            chat_entity=chat_entity,
            topic_resolver=topic_resolver,
        )
    except Exception:
        logger.exception("Failed to extract fields from a new message event")
        return

    record = to_log_record(message)
    logger.info(
        "New message: date_utc=%s text_length=%s topic_resolved=%s "
        "topic_title_known=%s",
        record.date_utc,
        record.text_length,
        message.topic_id is not None,
        message.topic_title is not None,
    )
    if message.topic_id is not None:
        logger.debug(
            "Topic resolved: topic_title_known=%s",
            message.topic_title is not None,
        )
    elif chat_entity is not None and getattr(chat_entity, "forum", False):
        logger.warning("Forum message without a topic id")

    if message.text is None:
        logger.debug("Skipping message without text")
        return

    if writer is None:
        return

    # Backpressure: awaits if the bounded queue is full instead of dropping.
    try:
        await writer.submit(message)
    except MessageWriterError:
        # The writer has failed or stopped; the message cannot be persisted.
        # Log it explicitly (without Telegram identifiers) so it is not silent.
        logger.exception("Message writer unavailable; message not persisted")
    except asyncio.CancelledError:
        # Telethon cancels in-flight handler tasks on disconnect. The message
        # was not queued; log it (without identifiers) and propagate.
        logger.warning("Message handling cancelled during shutdown; message not persisted")
        raise


def make_message_handler(
    writer: Optional[MessageWriter],
    entities_by_chat_id=None,
    topic_resolver: Optional[TopicResolver] = None,
) -> Callable[[object], Awaitable[None]]:
    """Bind ``writer`` (and optional forum metadata) to the handler.

    Telethon calls handlers with a single ``event`` argument, so the writer and
    the per-chat entity map are captured in a closure. The dict is captured by
    reference: entities resolved after registration are still visible.
    """

    entities = entities_by_chat_id if entities_by_chat_id is not None else {}

    async def handler(event) -> None:
        entity = entities.get(getattr(event, "chat_id", None))
        await handle_new_message(
            event, writer=writer, chat_entity=entity, topic_resolver=topic_resolver
        )

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
        history_sync: bool = False,
    ):
        self._config = config
        self._client = client if client is not None else create_client(config)
        self._writer = writer
        self._store = store
        self._history_sync = history_sync
        self._registered = False
        # Per-chat history-sync outcomes from the most recent start(). Empty
        # when history sync is disabled.
        self._history_results = ()
        # chat_id -> resolved Telegram entity. Captured by reference in the
        # handler, so entities resolved after registration remain visible.
        self._entities_by_chat_id: dict = {}
        self._topic_resolver = TopicResolver(self._client)

    @property
    def client(self) -> TelegramClient:
        return self._client

    @property
    def writer(self) -> Optional[MessageWriter]:
        return self._writer

    def _configured_chat_ids(self):
        entries = parse_chat_entries(self._config.chats)
        ids = list(entries.ids)
        for chat_id in self._entities_by_chat_id:
            if chat_id not in ids:
                ids.append(chat_id)
        return tuple(ids)

    def configured_chat_ids(self):
        """Configured chat ids plus successfully resolved ``title:`` entities.

        This is the authoritative set of chats the application actually watches,
        derived from the configuration and the entities resolved from Telegram
        (never from ``DISTINCT chat_id`` over stored messages). Call it after
        :meth:`start` so title entries have been resolved.
        """
        return self._configured_chat_ids()

    @property
    def history_sync_enabled(self) -> bool:
        """Whether startup history reconciliation is part of this runner."""
        return self._history_sync

    @property
    def history_results(self):
        """Per-chat results of the last startup history sync (may be empty)."""
        return self._history_results

    def history_failed_chat_ids(self):
        """Configured chats whose last startup history sync did not complete.

        A failed/incomplete sync must prevent the bootstrap snapshot for that
        chat: fixing a watermark over an incomplete history could permanently
        exclude old messages from analysis.
        """
        return frozenset(
            int(result.chat_id) for result in self._history_results if not result.ok
        )

    @staticmethod
    def _normalize_title(title: str) -> str:
        return " ".join(str(title).split()).casefold()

    async def _resolve_titles(self, titles) -> dict:
        """Resolve exact ``title:`` entries to forum supergroup entities.

        Uses only dialogs visible to the account. Exact (normalized) match; no
        fuzzy search. Missing, ambiguous, or non-forum matches raise
        :class:`ConfigError`.
        """
        dialogs = await self._client.get_dialogs()
        by_title: dict = {}
        for dialog in dialogs:
            name = getattr(dialog, "name", None)
            entity = getattr(dialog, "entity", None)
            if name is None or entity is None:
                continue
            by_title.setdefault(self._normalize_title(name), []).append(entity)

        resolved: dict = {}
        for title in titles:
            matches = by_title.get(self._normalize_title(title), [])
            if not matches:
                logger.error("Configured chat title not found: title=%r", title)
                raise ConfigError(
                    f"Supergroup with title {title!r} was not found among your "
                    "dialogs. Check the exact title or use a numeric chat id."
                )
            if len(matches) > 1:
                logger.error(
                    "Configured chat title is ambiguous: title=%r matches=%d",
                    title,
                    len(matches),
                )
                raise ConfigError(
                    f"Multiple supergroups match the title {title!r}; "
                    "specify the numeric chat ID instead."
                )
            entity = matches[0]
            is_forum = getattr(entity, "megagroup", False) and getattr(
                entity, "forum", False
            )
            if not is_forum:
                logger.error(
                    "Configured title is not a forum supergroup: title=%r", title
                )
                raise ConfigError(
                    f"Chat {title!r} is not a forum supergroup with topics enabled."
                )
            chat_id = int(entity.id)
            resolved[chat_id] = entity
            logger.info(
                "Resolved configured title %r to a forum supergroup",
                title,
            )
        return resolved

    async def _load_missing_entities(self, chat_ids) -> None:
        """Best-effort entity lookup for numeric chats (for forum metadata)."""
        if not hasattr(self._client, "get_entity"):
            return
        for chat_id in chat_ids:
            chat_id = int(chat_id)
            if chat_id in self._entities_by_chat_id:
                continue
            try:
                entity = await self._client.get_entity(chat_id)
                self._entities_by_chat_id[chat_id] = entity
            except Exception:
                logger.warning(
                    "Could not resolve entity for a configured chat; "
                    "topic metadata may be unavailable."
                )

    def register(self, handler=None, chat_ids=None) -> None:
        """Register the inbound-message handler for ``chat_ids``.

        Must happen before ``catch_up`` so recovered updates are processed.
        ``chat_ids`` may be passed explicitly (after resolving title entries);
        when omitted, numeric ids are parsed from the config (title entries
        cannot be resolved without a connected client).
        """
        if handler is None:
            handler = make_message_handler(
                self._writer, self._entities_by_chat_id, self._topic_resolver
            )
        if chat_ids is None:
            entries = parse_chat_entries(self._config.chats)
            if entries.has_titles:
                raise ConfigError(
                    "Title-based chats require a connected client; use start()."
                )
            chat_ids = entries.ids
        chat_ids = tuple(chat_ids)
        builder = new_message_builder(chat_ids)
        self._client.add_event_handler(handler, builder)
        self._registered = True
        logger.info("Registered NewMessage handler for %d chat(s).", len(chat_ids))

    async def start(self) -> None:
        """Start the writer, connect, resolve config, and catch up.

        Numeric-only configs keep the original order (register before connect) so
        recovered updates are handled. When ``title:`` entries are present, the
        client must connect first to resolve them from dialogs; registration then
        happens after resolution but still before ``catch_up()``.
        """
        if self._writer is not None:
            await self._writer.start()

        entries = parse_chat_entries(self._config.chats)
        if entries.has_titles:
            await self._client.start()
            self._entities_by_chat_id.clear()
            self._entities_by_chat_id.update(
                await self._resolve_titles(entries.titles)
            )
            chat_ids = self._configured_chat_ids()
            if not self._registered:
                self.register(chat_ids=chat_ids)
            await self._load_missing_entities(chat_ids)
        else:
            chat_ids = entries.ids
            if not self._registered:
                self.register(chat_ids=chat_ids)
            await self._client.start()
            await self._load_missing_entities(chat_ids)

        # Telethon docs: call catch_up() only after event handlers are
        # registered, so the fetched updates can be processed by our handler.
        await self._client.catch_up()
        logger.info("Telegram client connected and caught up.")

        # catch_up() is not a delivery guarantee; reconcile recent history for
        # each configured chat to close small gaps left while offline. Keep the
        # per-chat outcomes so the startup wiring can avoid snapshotting a chat
        # whose history sync did not complete.
        if self._history_sync and self._store is not None and self._writer is not None:
            self._history_results = tuple(await self.sync_history())

    async def sync_history(self):
        """Run bounded history reconciliation for the configured chats."""
        chat_ids = self._configured_chat_ids()
        return await sync_history(
            self._client,
            chat_ids,
            self._store,
            self._writer,
            entities_by_chat_id=self._entities_by_chat_id,
            topic_resolver=self._topic_resolver,
        )

    async def wait(self) -> None:
        """Wait until the client disconnects, without starting it again.

        Split out of :meth:`run_forever` so a caller can run its own post-start
        work (for example launching the AI worker) before blocking here.
        """
        await self._client.run_until_disconnected()

    async def run_forever(self) -> None:
        """Register, connect, recover, then wait until disconnected."""
        await self.start()
        await self.wait()

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
