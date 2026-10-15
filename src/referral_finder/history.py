"""Startup history reconciliation (bounded, one-shot).

Telethon's ``catch_up()`` asks the server for the update difference, but it is
not a guarantee that every message missed while the process was offline is
re-delivered: the server only keeps a finite update window, and the local
database may already contain gaps. This module therefore performs an explicit,
bounded history scan of each configured chat at startup and fills those gaps
through the existing single-writer pipeline.

Design constraints:

* Not a periodic poller and never downloads media. The scan runs once per
  process start, per configured chat.
* The first scan of a chat (no checkpoint yet) is *bounded* to the most recent
  :data:`DEFAULT_INITIAL_HISTORY_LIMIT` messages, which is enough to close small
  gaps (e.g. a stored id 28 with 26 and 27 missing) without importing the whole
  archive. This is why ``MAX(message_id)`` from the ``messages`` table is *not*
  used as a checkpoint: it would hide exactly those gaps.
* Later scans read everything after the stored checkpoint (paginated), so a long
  offline period is not silently truncated.
* The per-chat checkpoint lives in ``chat_sync_state`` and is advanced only
  after each page has been fully processed and flushed to SQLite. On any error
  the checkpoint is left at the last fully processed page, so no unprocessed
  message is ever skipped.
* Messages are persisted through the shared :class:`MessageWriter`, so the live
  handler and reconciliation never write concurrently; the existing
  ``(chat_id, message_id)`` deduplication makes overlapping work harmless.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Iterable, List, Optional

from .ingest import MessageWriter
from .models import incoming_from_object
from .storage import MessageStore

logger = logging.getLogger(__name__)

# Bounded size of the very first scan for a chat that has no checkpoint yet.
DEFAULT_INITIAL_HISTORY_LIMIT = 200
# How many messages are examined/stored between checkpoint commits.
DEFAULT_HISTORY_PAGE_SIZE = 100


@dataclass(frozen=True)
class ChatSyncResult:
    """Outcome of reconciling a single chat's history."""

    chat_id: int
    initial: bool
    examined: int
    submitted: int
    stored: int
    skipped: int
    checkpoint_before: Optional[int]
    checkpoint_after: Optional[int]
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


async def sync_history(
    client,
    chat_ids: Iterable[int],
    store: MessageStore,
    writer: MessageWriter,
    *,
    initial_limit: int = DEFAULT_INITIAL_HISTORY_LIMIT,
    page_size: int = DEFAULT_HISTORY_PAGE_SIZE,
    clock: Callable[[], float] = time.time,
) -> List[ChatSyncResult]:
    """Reconcile each configured chat and return per-chat results.

    ``client`` only needs an async ``iter_messages(entity, ...)`` method (the
    real :class:`telethon.TelegramClient` and the test fakes both provide it).
    """
    ids = [int(chat_id) for chat_id in chat_ids]
    logger.info("History sync starting for %d chat(s).", len(ids))
    results: List[ChatSyncResult] = []
    for chat_id in ids:
        results.append(
            await _sync_one_chat(
                client,
                chat_id,
                store,
                writer,
                initial_limit=initial_limit,
                page_size=page_size,
                clock=clock,
            )
        )
    return results


async def _sync_one_chat(
    client,
    chat_id: int,
    store: MessageStore,
    writer: MessageWriter,
    *,
    initial_limit: int,
    page_size: int,
    clock: Callable[[], float],
) -> ChatSyncResult:
    checkpoint_before = store.get_checkpoint(chat_id)
    initial = checkpoint_before is None

    errors_before = writer.errors
    saved_before = writer.saved
    examined = submitted = skipped = 0
    committed = checkpoint_before
    error: Optional[str] = None

    try:
        if initial:
            messages = _iter_initial(client, chat_id, initial_limit)
        else:
            messages = client.iter_messages(chat_id, min_id=checkpoint_before, reverse=True)

        async for page in _chunks(messages, page_size):
            page_max = max(int(message.id) for message in page)

            batch = []
            for message in page:
                examined += 1
                if getattr(message, "out", False):
                    skipped += 1
                    continue
                incoming = incoming_from_object(message, received_at_utc=int(clock()))
                if incoming.text is None:
                    skipped += 1
                    continue
                batch.append(incoming)

            # Goes through the same single-writer pipeline as live events, as a
            # single batched transaction per page.
            if batch:
                await writer.submit_batch(batch)
                submitted += len(batch)

            # Barrier: do not advance the checkpoint until the page has been
            # handed to SQLite. Persistence failures bump writer.errors and
            # abort the sync here.
            await writer.flush()
            if writer.errors > errors_before:
                error = "persistence errors during history sync"
                logger.error(
                    "History sync aborted before checkpoint advance "
                    "chat_id=%s page_max=%s",
                    chat_id,
                    page_max,
                )
                break

            store.set_checkpoint(chat_id, page_max, int(clock()))
            committed = page_max
    except Exception as exc:  # noqa: BLE001 - record, never crash the app
        error = type(exc).__name__
        logger.exception("History sync failed chat_id=%s", chat_id)

    result = ChatSyncResult(
        chat_id=chat_id,
        initial=initial,
        examined=examined,
        submitted=submitted,
        stored=writer.saved - saved_before,
        skipped=skipped,
        checkpoint_before=checkpoint_before,
        checkpoint_after=committed,
        error=error,
    )
    logger.info(
        "History sync chat_id=%s initial=%s examined=%s submitted=%s stored=%s "
        "skipped=%s checkpoint=%s->%s result=%s",
        chat_id,
        initial,
        examined,
        submitted,
        result.stored,
        skipped,
        checkpoint_before,
        committed,
        error or "ok",
    )
    return result


async def _iter_initial(client, chat_id: int, limit: int) -> AsyncIterator:
    """Yield the most recent ``limit`` messages oldest-first (bounded scan)."""
    recent = []
    async for message in client.iter_messages(chat_id, limit=limit):
        recent.append(message)
    for message in reversed(recent):
        yield message


async def _chunks(messages: AsyncIterator, size: int) -> AsyncIterator[List]:
    """Group an async iterator into lists of at most ``size`` items."""
    buffer: List = []
    async for item in messages:
        buffer.append(item)
        if len(buffer) >= size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer
