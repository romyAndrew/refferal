"""Offline message-analysis pipeline: prefilter → scrub → provider → store.

One message at a time, and one local record key (``chat_id``, ``message_id``)
which is used only for the local storage round-trip. The key and the raw text
are never placed in :class:`~referral_finder.llm.base.LLMRequest`, the prompt,
the HTTP body, or any log/diagnostic message: the provider sees only the
scrubbed text and the configured search criteria.

The service is safe by default: with ``enabled=False`` it returns immediately
without calling any provider. It is deliberately not wired into the Telegram
collector.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from ..prefilter import prefilter
from ..scrubbing import scrub
from ..storage import MessageStore
from .base import (
    LLMConfigError,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
    LLMRequest,
    LLMResponseError,
    LLMServerError,
    LLMTimeoutError,
)
from .schema import (
    STATUS_ERROR,
    STATUS_INVALID,
    STATUS_OK,
    AnalysisResult,
    ValidationError,
    ai_analysis_row,
    validate_analysis,
)
from .versioning import effective_analysis_version

logger = logging.getLogger(__name__)

# Outcome statuses (kept as plain strings for simple aggregation).
STATUS_DISABLED = "disabled"
STATUS_PREFILTER_SKIPPED = "prefilter_skipped"
STATUS_SAVED = "saved"
STATUS_DUPLICATE = "duplicate"
STATUS_INVALID = "invalid"
STATUS_STORAGE_ERROR = "storage_error"

# Short, non-sensitive error codes (no text, no identifiers).
_ERROR_TIMEOUT = "timeout"
_ERROR_RATE_LIMIT = "rate_limit"
_ERROR_SERVER = "server_error"
_ERROR_CONFIG = "config_error"
_ERROR_PROVIDER = "provider_error"
_ERROR_INVALID_RESPONSE = "invalid_response"
_ERROR_INVALID_RESULT = "invalid_result"
_ERROR_STORAGE = "storage_error"


@dataclass(frozen=True)
class AnalysisOutcome:
    """Result of analysing one message (no message text or PII)."""

    status: str
    matched: bool = False
    prefilter_rules: Tuple[str, ...] = ()
    scrub_rules: Tuple[str, ...] = ()
    stored: int = 0
    error: Optional[str] = None
    analysis: Optional[AnalysisResult] = None


class MessageAnalysisService:
    """Runs the offline analysis pipeline for a single message at a time."""

    def __init__(
        self,
        store: MessageStore,
        provider: LLMProvider,
        *,
        enabled: bool = False,
        criteria: str = "",
        clock: Optional[Callable[[], int]] = None,
    ):
        self._store = store
        self._provider = provider
        self._enabled = enabled
        self._criteria = criteria
        self._clock = clock or (lambda: int(time.time()))

    async def analyze(self, chat_id: int, message_id: int, text: str) -> AnalysisOutcome:
        """Analyse one message and persist the outcome.

        The local key (``chat_id``, ``message_id``) is used only for storage.
        """
        if not self._enabled:
            return AnalysisOutcome(STATUS_DISABLED)

        gate = prefilter(text)
        if not gate.matched:
            return AnalysisOutcome(
                STATUS_PREFILTER_SKIPPED,
                matched=False,
                prefilter_rules=gate.rule_ids,
            )

        scrubbed = scrub(text)
        request = LLMRequest(text=scrubbed.text, criteria=self._criteria)

        try:
            result = await self._provider.analyze(request)
        except LLMTimeoutError:
            return self._store_failure(
                chat_id, message_id, STATUS_ERROR, _ERROR_TIMEOUT, gate.rule_ids, scrubbed.rule_ids
            )
        except LLMRateLimitError:
            return self._store_failure(
                chat_id, message_id, STATUS_ERROR, _ERROR_RATE_LIMIT, gate.rule_ids, scrubbed.rule_ids
            )
        except LLMServerError:
            return self._store_failure(
                chat_id, message_id, STATUS_ERROR, _ERROR_SERVER, gate.rule_ids, scrubbed.rule_ids
            )
        except LLMConfigError:
            return self._store_failure(
                chat_id, message_id, STATUS_ERROR, _ERROR_CONFIG, gate.rule_ids, scrubbed.rule_ids
            )
        except LLMResponseError:
            return self._store_failure(
                chat_id, message_id, STATUS_INVALID, _ERROR_INVALID_RESPONSE,
                gate.rule_ids, scrubbed.rule_ids,
            )
        except LLMError:
            return self._store_failure(
                chat_id, message_id, STATUS_ERROR, _ERROR_PROVIDER, gate.rule_ids, scrubbed.rule_ids
            )
        except Exception:
            # A single provider failure must not abort a batch; record it and
            # let the caller continue with the next message. No text/ids logged.
            logger.exception("AI provider call failed")
            return self._store_failure(
                chat_id, message_id, STATUS_ERROR, _ERROR_PROVIDER, gate.rule_ids, scrubbed.rule_ids
            )

        validated = self._revalidate(result)
        if validated is None:
            return self._store_failure(
                chat_id, message_id, STATUS_INVALID, _ERROR_INVALID_RESULT,
                gate.rule_ids, scrubbed.rule_ids,
            )

        return self._store_ok(
            chat_id, message_id, validated, gate.rule_ids, scrubbed.rule_ids
        )

    @staticmethod
    def _revalidate(result) -> Optional[AnalysisResult]:
        """Re-check a provider result against the contract.

        Providers normally validate already, but the service does not trust that
        blindly: a non-``AnalysisResult`` or an inconsistent one is rejected and
        must never be stored as a successful classification.
        """
        if not isinstance(result, AnalysisResult):
            return None
        try:
            return validate_analysis(
                {
                    "category": result.category,
                    "relevant": result.relevant,
                    "confidence": result.confidence,
                    "explanation": result.explanation,
                }
            )
        except ValidationError:
            return None

    def _provider_meta(self) -> dict:
        return {
            "provider": self._provider.provider_name,
            "model": self._provider.model,
            "prompt_version": self._effective_version(),
            "prompt_hash": self._provider.prompt_hash,
        }

    def _effective_version(self) -> str:
        """Effective analysis version: template prompt version + criteria hash."""
        return effective_analysis_version(self._provider.prompt_version, self._criteria)

    def _store_ok(
        self,
        chat_id: int,
        message_id: int,
        result: AnalysisResult,
        prefilter_rules: Tuple[str, ...],
        scrub_rules: Tuple[str, ...],
    ) -> AnalysisOutcome:
        row = ai_analysis_row(
            chat_id,
            message_id,
            result,
            analyzed_at_utc=self._clock(),
            status=STATUS_OK,
            **self._provider_meta(),
        )
        try:
            inserted = self._store.save_ai_analysis([row])
        except Exception:
            # Never log the text or the local ids here.
            logger.exception("Failed to store AI analysis result")
            return AnalysisOutcome(
                STATUS_STORAGE_ERROR,
                matched=True,
                prefilter_rules=prefilter_rules,
                scrub_rules=scrub_rules,
                error=_ERROR_STORAGE,
            )
        return AnalysisOutcome(
            STATUS_SAVED if inserted else STATUS_DUPLICATE,
            matched=True,
            prefilter_rules=prefilter_rules,
            scrub_rules=scrub_rules,
            stored=inserted,
            analysis=result,
        )

    def _store_failure(
        self,
        chat_id: int,
        message_id: int,
        status: str,
        error_code: str,
        prefilter_rules: Tuple[str, ...],
        scrub_rules: Tuple[str, ...],
    ) -> AnalysisOutcome:
        row = ai_analysis_row(
            chat_id,
            message_id,
            None,
            analyzed_at_utc=self._clock(),
            status=status,
            error=error_code,
            **self._provider_meta(),
        )
        try:
            self._store.save_ai_analysis([row])
        except Exception:
            logger.exception("Failed to store AI analysis failure record")
        return AnalysisOutcome(
            status,
            matched=True,
            prefilter_rules=prefilter_rules,
            scrub_rules=scrub_rules,
            error=error_code,
        )
