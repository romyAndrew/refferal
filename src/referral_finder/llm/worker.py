"""Background AI-analysis worker.

Polls the existing ``message_ai_analysis`` state (there is no separate queue
table) and runs the existing :class:`~referral_finder.llm.service.MessageAnalysisService`
for each still-pending message. All pipeline behaviour — prefilter, scrubbing,
provider call, validation, versioning and the bounded retry policy — lives in
that service and in :class:`~referral_finder.storage.MessageStore`; this module
only drives them one message at a time.

Guarantees and boundaries:

* Concurrency is 1 (one provider call at a time); the batch is materialised
  before analysis so no SQLite cursor stays open across network calls.
* A normal :meth:`stop` asks the worker to finish the in-flight analysis but not
  start another one; it does not cancel an active provider request (the provider
  timeout bounds the wait).
* The worker never closes the :class:`MessageStore`; the application lifecycle
  owns it.
* It is not wired into the Telegram handler, the writer, or the message-save
  path: no network call happens inside synchronous ingestion.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, List, Optional, Tuple

from ..storage import DEFAULT_MAX_ATTEMPTS, MessageStore
from ..prefilter import PREFILTER_VERSION
from .base import LLMProvider
from .service import MessageAnalysisService
from .versioning import effective_analysis_version

logger = logging.getLogger(__name__)

# Conservative defaults, mirroring config.DEFAULT_LLM_* values.
DEFAULT_POLL_INTERVAL = 5.0
DEFAULT_BATCH_SIZE = 25

# A row is (chat_id, message_id, text).
PendingRow = Tuple[int, int, Optional[str]]


class AiWorker:
    """Single-task background worker that analyses pending messages."""

    def __init__(
        self,
        store: MessageStore,
        provider: LLMProvider,
        *,
        criteria: str = "",
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        enabled: bool = False,
        wait: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        self._store = store
        self._provider = provider
        self._criteria = criteria
        self._poll_interval = poll_interval
        self._batch_size = batch_size
        self._max_attempts = max_attempts
        self._enabled = enabled
        self._wait = wait or self._default_wait
        self._service = MessageAnalysisService(
            store, provider, enabled=enabled, criteria=criteria
        )
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

        # Diagnostics (no secrets, no message text).
        self.processed = 0
        self.cycles = 0
        self.last_error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def effective_version(self) -> str:
        """Effective analysis version (template prompt version + criteria hash)."""
        return effective_analysis_version(
            self._provider.prompt_version, self._criteria
        )

    async def start(self) -> None:
        """Start the background task (re-entrant, idempotent).

        A disabled worker starts no task at all, so it can never call the
        provider or write results.
        """
        if not self._enabled:
            return
        if self._task is not None and not self._task.done():
            return
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="ai-worker")

    async def stop(self) -> None:
        """Stop the worker after the in-flight analysis; safe to call twice."""
        task = self._task
        if task is None:
            return
        self._stop.set()
        try:
            await task
        except asyncio.CancelledError:
            # External cancellation is propagated, never turned into success.
            raise
        except Exception:
            logger.exception("AI worker stopped with an unexpected error")
        finally:
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                batch = self._next_batch()
            except asyncio.CancelledError:
                raise
            except Exception:
                # A systematic storage/configuration failure must not be hidden
                # or retried in a tight loop: log it and stop the worker.
                self.last_error = "read_failed"
                logger.exception("AI worker could not read pending messages; stopping")
                return

            if not batch:
                await self._wait_once()
                continue

            for chat_id, message_id, text in batch:
                if self._stop.is_set():
                    return
                try:
                    await self._service.analyze(chat_id, message_id, text)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # The service normally records failures and returns; an
                    # unexpected error must not kill the worker. Log (no text,
                    # no identifiers) and continue with the next message.
                    self.last_error = "analyze_failed"
                    logger.exception("AI worker failed to analyze a message")
                self.processed += 1

            await self._wait_once()

    def _next_batch(self) -> List[PendingRow]:
        """Return one bounded, materialised batch of pending messages.

        The storage query already applies ``LIMIT batch_size`` and ``fetchall``,
        so only the first (bounded) chunk is taken and no cursor stays open.
        """
        for chunk in self._store.iter_ai_unprocessed(
            self._provider.provider_name,
            self._provider.model,
            self.effective_version,
            max_attempts=self._max_attempts,
            batch_size=self._batch_size,
            prefilter_version=PREFILTER_VERSION,
        ):
            return list(chunk)
        return []

    async def _wait_once(self) -> None:
        self.cycles += 1
        await self._wait()

    async def _default_wait(self) -> None:
        """Sleep for the poll interval, waking early on stop."""
        if self._stop.is_set():
            return
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=self._poll_interval)
        except asyncio.TimeoutError:
            return
