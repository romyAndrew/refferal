"""Bootstrap and steady AI-analysis worker integrated with the application lifecycle.

This module drives the two analysis phases for the configured/allowed chats:

* **Bootstrap** — the fixed per-chat snapshot (up to ``LLM_BOOTSTRAP_LIMIT``
  most recent text messages) that is captured once from the local SQLite
  database. The snapshot deliberately bypasses the prefilter (it is evaluated in
  full) but the text is still scrubbed before it reaches the provider. The
  snapshot and the per-message progress are persisted, so a restart neither
  re-picks a different snapshot nor re-analyses a stored success.
* **Steady** — only messages strictly newer than the chat's saved
  ``steady_after_message_id``. They pass through the broad prefilter and the
  company catalog, exactly like the existing pipeline, so history beyond the
  bootstrap window can never enter the steady stream.

The worker does not create its own request counter: it shares the provider's
persistent :class:`~referral_finder.llm.budget.ApiRequestBudget`. When the daily
UTC budget is exhausted, nothing is deleted and no analysis attempt is recorded;
the message simply stays queued for the next UTC day.

The lifecycle owner closes the store; the worker is integrated with application
startup through the app entry point.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, List, Optional, Sequence, Tuple

from ..config import (
    DEFAULT_LLM_BATCH_SIZE,
    DEFAULT_LLM_BOOTSTRAP_LIMIT,
    DEFAULT_LLM_POLL_INTERVAL,
)
from ..prefilter import effective_prefilter_version
from ..storage import DEFAULT_MAX_ATTEMPTS, MessageStore
from .base import LLMProvider
from .service import STATUS_BUDGET_EXHAUSTED, MessageAnalysisService
from .versioning import effective_analysis_version

logger = logging.getLogger(__name__)

# A row is (chat_id, message_id, text).
PendingRow = Tuple[int, int, Optional[str]]


class BootstrapSteadyWorker:
    """Single-task worker that runs bootstrap first, then the steady stream."""

    def __init__(
        self,
        store: MessageStore,
        provider: LLMProvider,
        chat_ids: Sequence[int],
        *,
        criteria: str = "",
        bootstrap_limit: int = DEFAULT_LLM_BOOTSTRAP_LIMIT,
        poll_interval: float = DEFAULT_LLM_POLL_INTERVAL,
        batch_size: int = DEFAULT_LLM_BATCH_SIZE,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        enabled: bool = False,
        wait: Optional[Callable[[], Awaitable[None]]] = None,
        catalog=None,
        clock: Optional[Callable[[], int]] = None,
    ):
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if bootstrap_limit < 1:
            raise ValueError("bootstrap_limit must be at least 1")

        self._store = store
        self._provider = provider
        self._chat_ids = [int(chat_id) for chat_id in chat_ids]
        self._criteria = criteria
        self._bootstrap_limit = bootstrap_limit
        self._poll_interval = poll_interval
        self._batch_size = batch_size
        self._max_attempts = max_attempts
        self._enabled = enabled
        self._catalog = catalog
        self._clock = clock
        self._wait = wait or self._default_wait
        self._service = MessageAnalysisService(
            store,
            provider,
            enabled=enabled,
            criteria=criteria,
            catalog=catalog,
            clock=clock,
        )
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._snapshotted = False
        self._bootstrap_done = False

        # Diagnostics (no secrets, no message text).
        self.bootstrap_processed = 0
        self.steady_processed = 0
        self.cycles = 0
        self.last_error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def bootstrap_done(self) -> bool:
        return self._bootstrap_done

    @property
    def phase(self) -> str:
        """Current phase (``bootstrap`` until its fixed queue is exhausted)."""
        return "steady" if self._bootstrap_done else "bootstrap"

    @property
    def effective_version(self) -> str:
        """Effective analysis version (template prompt version + criteria hash)."""
        return effective_analysis_version(self._provider.prompt_version, self._criteria)

    def snapshot(self) -> list:
        """Fix the per-chat bootstrap snapshot (idempotent across restarts).

        Must be called only after the normal history sync, so the high-water
        mark reflects everything already stored. Calling it again reuses the
        existing state instead of re-picking a different snapshot.
        """
        summaries = self._store.snapshot_bootstrap(
            self._chat_ids, limit=self._bootstrap_limit, clock=self._clock
        )
        self._snapshotted = True
        return summaries

    async def start(self) -> None:
        """Snapshot the chats and start the background task (idempotent).

        A disabled worker starts no task and takes no snapshot, so it can never
        write analysis state or call the provider.
        """
        if not self._enabled:
            return
        if self._task is not None and not self._task.done():
            return
        if not self._snapshotted:
            self.snapshot()
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="ai-bootstrap-worker")

    async def stop(self) -> None:
        """Stop after the in-flight analysis; safe to call more than once."""
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
            logger.exception("Bootstrap/steady worker stopped with an unexpected error")
        finally:
            self._task = None

    async def _run(self) -> None:
        if not self._snapshotted:
            self.snapshot()
        while not self._stop.is_set():
            try:
                if not self._bootstrap_done:
                    await self._run_bootstrap_cycle()
                if self._bootstrap_done and not self._stop.is_set():
                    await self._run_steady_cycle()
            except asyncio.CancelledError:
                raise
            except Exception:
                # A systematic storage/configuration failure must not be hidden
                # or retried in a tight loop: log it and stop the worker.
                self.last_error = "cycle_failed"
                logger.exception("Bootstrap/steady worker cycle failed; stopping")
                return
            await self._wait_once()

    # --- bootstrap phase ---------------------------------------------------

    async def _run_bootstrap_cycle(self) -> None:
        """Analyse one bootstrap batch, bypassing the prefilter.

        When the whole fixed bootstrap queue is exhausted (every item is a
        stored success or has reached ``max_attempts``) the phase ends and all
        snapshot chats are marked ``done``.
        """
        batch = self._bootstrap_batch()
        if not batch:
            self._finish_bootstrap()
            return
        for chat_id, message_id, text in batch:
            if self._stop.is_set():
                return
            outcome = await self._service.analyze(
                chat_id, message_id, text, bypass_prefilter=True
            )
            if outcome.status == STATUS_BUDGET_EXHAUSTED:
                # Keep the phase: nothing is deleted and no attempt is recorded,
                # so the item stays queued for the next UTC day.
                return
            self.bootstrap_processed += 1

    def _bootstrap_batch(self) -> List[PendingRow]:
        for chunk in self._store.iter_ai_bootstrap(
            self._provider.provider_name,
            self._provider.model,
            self.effective_version,
            max_attempts=self._max_attempts,
            batch_size=self._batch_size,
            chat_ids=self._chat_ids,
        ):
            return list(chunk)
        return []

    def _finish_bootstrap(self) -> None:
        if self._bootstrap_done:
            return
        # Only this worker's current/configured chats are completed. State rows
        # left over from a previous configuration must stay untouched.
        for chat_id in self._chat_ids:
            try:
                self._store.set_bootstrap_status(chat_id, "done", clock=self._clock)
            except Exception:
                logger.exception("Failed to mark bootstrap status done")
        self._bootstrap_done = True

    # --- steady phase ------------------------------------------------------

    async def _run_steady_cycle(self) -> None:
        """Analyse one steady batch (prefilter + catalog, high-water mark)."""
        batch = self._steady_batch()
        for chat_id, message_id, text in batch:
            if self._stop.is_set():
                return
            outcome = await self._service.analyze(chat_id, message_id, text)
            if outcome.status == STATUS_BUDGET_EXHAUSTED:
                return
            self.steady_processed += 1

    def _steady_batch(self) -> List[PendingRow]:
        for chunk in self._store.iter_ai_unprocessed(
            self._provider.provider_name,
            self._provider.model,
            self.effective_version,
            max_attempts=self._max_attempts,
            batch_size=self._batch_size,
            prefilter_version=effective_prefilter_version(self._catalog),
            steady_only=True,
            chat_ids=self._chat_ids,
        ):
            return list(chunk)
        return []

    # --- waiting -----------------------------------------------------------

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
