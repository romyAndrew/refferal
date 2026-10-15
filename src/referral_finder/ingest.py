"""Message ingestion pipeline: bounded queue + single writer (stage 3/4).

The handler converts a Telethon event into an :class:`IncomingMessage` and
hands it to a :class:`MessageWriter`. The writer owns one ``asyncio.Queue`` and
one background task, so all SQLite writes happen from a single coroutine and no
concurrent write transactions are created.

If the queue is full, :meth:`MessageWriter.submit` awaits until there is room
(backpressure) instead of silently dropping the message. On shutdown a sentinel
is placed *after* the already accepted items, so every accepted message is
persisted before the writer task ends. Persisting errors are logged explicitly
and never reported as success.

Failure handling (stage 4): if the background writer task terminates
unexpectedly, the exception is recorded on the writer and surfaced to callers.
:meth:`submit` then raises :class:`MessageWriterError` instead of blocking
forever on a queue nobody drains, and :meth:`stop` raises it after the task has
ended. ``start``/``stop`` are re-entrant, so a writer can be restarted without
leaking tasks.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .models import IncomingMessage
from .storage import MessageStore

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_MAX_SIZE = 1000

# Sentinels are distinct from any real IncomingMessage, so ``None``/objects can
# never be mistaken for a message to persist.
_STOP = object()


class _Batch:
    """Internal queue item carrying several messages for a single transaction."""

    __slots__ = ("messages",)

    def __init__(self, messages):
        self.messages = messages


class MessageWriterError(RuntimeError):
    """Raised when the writer is no longer able to accept messages."""


class MessageWriter:
    """Single background writer draining an :class:`asyncio.Queue` to SQLite."""

    def __init__(
        self,
        store: MessageStore,
        maxsize: int = DEFAULT_QUEUE_MAX_SIZE,
        on_saved=None,
    ):
        self._store = store
        # Optional post-persist callback (e.g. classification). It runs in this
        # single writer task; failures are logged and never break ingestion.
        self._on_saved = on_saved
        self._queue: "asyncio.Queue[object]" = asyncio.Queue(maxsize=maxsize)
        self._task: Optional[asyncio.Task] = None
        self._stopping = False
        self._failure: Optional[BaseException] = None
        # Number of submit() calls that passed the shutdown check but have not
        # finished queueing yet, and an event set when that number is zero.
        # ``stop`` waits on it so the sentinel is never enqueued ahead of a
        # message that was already being accepted.
        self._active_submits = 0
        self._no_active_submits = asyncio.Event()
        self._no_active_submits.set()
        self.saved = 0
        self.duplicates = 0
        self.errors = 0

    @property
    def pending(self) -> int:
        """Number of items currently waiting in the queue."""
        return self._queue.qsize()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def failure(self) -> Optional[BaseException]:
        """The exception that killed the writer task, if any."""
        return self._failure

    async def start(self) -> None:
        """Start the background writer task (re-entrant / idempotent)."""
        if self._task is not None and not self._task.done():
            return
        self._stopping = False
        self._failure = None
        self._task = asyncio.create_task(self._run(), name="message-writer")
        self._task.add_done_callback(self._on_task_done)

    async def submit(self, message: IncomingMessage) -> None:
        """Queue a single ``message`` for persistence (live path)."""
        await self._enqueue(message)

    async def submit_batch(self, messages) -> None:
        """Queue several messages for one persistence transaction (history)."""
        batch = _Batch(list(messages))
        if not batch.messages:
            return
        await self._enqueue(batch)

    async def _enqueue(self, item) -> None:
        """Queue ``item`` (a message or a batch), applying backpressure if full.

        Raises :class:`MessageWriterError` if the writer has failed, has been
        stopped, or is shutting down, so an item is never silently withheld
        when nobody will drain the queue. If the caller (for example a Telethon
        handler task) is cancelled while waiting for queue space, the pending
        ``put`` is cancelled too and ``CancelledError`` is re-raised.
        """
        if self._failure is None and self._task is not None and self._task.done():
            self._record_failure(self._task)
        if self._failure is not None:
            raise MessageWriterError(
                "message writer is not running; item was not queued"
            ) from self._failure
        if self._stopping:
            raise MessageWriterError(
                "message writer is shutting down; item was not queued"
            )

        # No await between the shutdown check above and this registration, so a
        # submit that passed the check is always counted before ``stop`` starts
        # waiting for in-flight submissions.
        self._active_submits += 1
        self._no_active_submits.clear()
        try:
            put_task = asyncio.ensure_future(self._queue.put(item))
            try:
                if self._task is None:
                    # Not started yet: queue now, a later start() will drain it.
                    await put_task
                    return
                # Wait for either queue space or the writer task ending, so a
                # dead writer can never make submit() block forever.
                done, _pending = await asyncio.wait(
                    {put_task, self._task}, return_when=asyncio.FIRST_COMPLETED
                )
            except asyncio.CancelledError:
                # The awaiting handler was cancelled (e.g. Telethon disconnect):
                # drop the half-finished put so no orphan task outlives shutdown.
                put_task.cancel()
                raise

            if put_task in done:
                exc = put_task.exception()
                if exc is not None:
                    raise exc
                if self._failure is not None:
                    raise MessageWriterError(
                        "message writer stopped before the message could be stored"
                    ) from self._failure
                return

            put_task.cancel()
            if self._failure is None and self._task is not None:
                self._record_failure(self._task)
            raise MessageWriterError(
                "message writer stopped before the message was queued"
            ) from self._failure
        finally:
            self._active_submits -= 1
            if self._active_submits == 0:
                self._no_active_submits.set()

    async def flush(self) -> None:
        """Wait until everything queued so far has been processed by the writer.

        Used by history reconciliation as a barrier: after ``flush`` returns, all
        messages submitted before the call have been handed to the store (either
        inserted or rejected as duplicates). Raises :class:`MessageWriterError`
        if the writer task has already failed or stops before draining, so the
        caller never advances a checkpoint over unpersisted messages.

        Per-message persistence errors are counted in :attr:`errors` (and the
        writer keeps running); callers that need a strict barrier compare that
        counter before and after the flush.
        """
        if self._failure is not None:
            raise MessageWriterError(
                "message writer is not running; flush failed"
            ) from self._failure
        if self._task is None:
            # Nothing was ever started; the queue is not being drained.
            if not self._queue.empty():
                raise MessageWriterError(
                    "message writer has not been started; flush failed"
                )
            return

        join_task = asyncio.ensure_future(self._queue.join())
        try:
            done, _pending = await asyncio.wait(
                {join_task, self._task}, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            join_task.cancel()
            raise

        if join_task in done:
            exc = join_task.exception()
            if exc is not None:
                raise exc
            if self._failure is not None:
                raise MessageWriterError(
                    "message writer failed before flush completed"
                ) from self._failure
            return

        join_task.cancel()
        if self._failure is None and self._task is not None:
            self._record_failure(self._task)
        raise MessageWriterError(
            "message writer stopped before flush completed"
        ) from self._failure

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is _STOP:
                    return
                if isinstance(item, _Batch):
                    self._persist_batch(item.messages)
                else:
                    self._persist(item)  # type: ignore[arg-type]
            finally:
                self._queue.task_done()

    def _notify_saved(self, messages) -> None:
        """Run the optional post-persist hook; never let it break ingestion."""
        if self._on_saved is None:
            return
        payload = [message for message in messages if getattr(message, "text", None)]
        if not payload:
            return
        try:
            self._on_saved(payload)
        except Exception:
            # Classification (or any post-persist step) must not stop collection.
            logger.exception("Post-persist hook failed for %d message(s)", len(payload))

    def _persist_batch(self, messages) -> None:
        try:
            inserted, duplicates = self._store.save_many(messages)
        except Exception:
            self.errors += 1
            logger.exception(
                "Failed to store a batch of %d message(s)", len(messages)
            )
            return
        self._notify_saved(messages)
        self.saved += inserted
        self.duplicates += duplicates
        logger.info(
            "Stored message batch size=%d inserted=%d duplicates=%d",
            len(messages),
            inserted,
            duplicates,
        )

    def _persist(self, message: IncomingMessage) -> None:
        try:
            inserted = self._store.save(message)
        except Exception:
            self.errors += 1
            # Log the identifiers only; never the message text.
            logger.exception(
                "Failed to store message chat_id=%s message_id=%s",
                message.chat_id,
                message.message_id,
            )
            return

        self._notify_saved((message,))
        if inserted:
            self.saved += 1
            logger.info(
                "Stored message chat_id=%s message_id=%s",
                message.chat_id,
                message.message_id,
            )
        else:
            self.duplicates += 1
            logger.debug(
                "Duplicate message ignored chat_id=%s message_id=%s",
                message.chat_id,
                message.message_id,
            )

    def _on_task_done(self, task: asyncio.Task) -> None:
        self._record_failure(task)
        if self._failure is not None:
            logger.error("Message writer stopped unexpectedly: %r", self._failure)

    def _record_failure(self, task: asyncio.Task) -> None:
        """Capture why ``task`` ended, unless it ended via a clean stop."""
        if self._failure is not None:
            return
        if task.cancelled():
            self._failure = MessageWriterError("message writer was cancelled")
            return
        exc = task.exception()
        if exc is not None:
            self._failure = exc
        elif not self._stopping:
            self._failure = MessageWriterError(
                "message writer stopped without a shutdown request"
            )

    async def stop(self) -> None:
        """Drain accepted items, then stop the writer task (re-entrant).

        Raises :class:`MessageWriterError` if the writer task ended with an
        unexpected failure (the original exception is chained).
        """
        task = self._task
        if task is None or self._stopping:
            return
        self._stopping = True
        # Wait for submissions that already passed the shutdown check to finish
        # queueing. The writer keeps draining while we wait, so the queue frees
        # up. Only then is the sentinel enqueued, guaranteeing it is the last
        # item the writer sees.
        await self._no_active_submits.wait()
        if not task.done():
            # Place the sentinel after all currently queued items.
            await self._queue.put(_STOP)
        try:
            await task
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - record, then re-wrap below
            if self._failure is None:
                self._failure = exc
        finally:
            self._task = None
        self._record_failure(task)
        if self._failure is not None:
            raise MessageWriterError("message writer stopped unexpectedly") from self._failure
