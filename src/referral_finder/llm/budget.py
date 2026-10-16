"""Persistent daily budget for outgoing DeepSeek HTTP requests.

The budget is stored in the same SQLite database as the rest of the app (table
``api_request_budget``) so it survives restarts and is shared by bootstrap, the
steady worker, manual backfill and the smoke command. Counting is per calendar
day in UTC, and every actual HTTP attempt (including retries) reserves one unit
*before* the transport is invoked.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Optional

from .base import LLMConfigError


def utc_day(epoch_seconds: float) -> str:
    """Return ``YYYY-MM-DD`` for the given instant in UTC (ignores local TZ)."""
    return datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc).strftime(
        "%Y-%m-%d"
    )


class ApiRequestBudget:
    """Tracks and reserves the per-UTC-day outgoing request budget."""

    def __init__(
        self,
        store,
        limit: int,
        clock: Optional[Callable[[], float]] = None,
    ):
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise LLMConfigError("LLM_MAX_API_REQUESTS_PER_DAY must be a positive integer")
        self._store = store
        self._limit = int(limit)
        self._clock = clock or time.time

    @property
    def limit(self) -> int:
        return self._limit

    def day(self) -> str:
        """Current UTC calendar day (``YYYY-MM-DD``)."""
        return utc_day(self._clock())

    def used(self) -> int:
        """Requests already reserved for the current UTC day."""
        return self._store.api_requests_used(self.day())

    def remaining(self) -> int:
        return max(0, self._limit - self.used())

    def reserve(self) -> bool:
        """Atomically reserve one request; ``False`` when the limit is reached.

        A reservation is never returned, even if the request later times out:
        counting it conservatively is safer than risking an overrun.
        """
        return self._store.reserve_api_request(self.day(), self._limit, int(self._clock()))
