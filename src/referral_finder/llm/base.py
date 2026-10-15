"""Provider-agnostic LLM interface and request/error types.

Every provider analyses exactly one already-scrubbed message at a time. The
interface deliberately hides transport details so a real provider can be swapped
for the offline :class:`~referral_finder.llm.fake.FakeProvider` without changing
callers. No network access happens in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .schema import AnalysisResult


@dataclass(frozen=True)
class LLMRequest:
    """One message to analyse.

    ``text`` must already be scrubbed (see :mod:`referral_finder.scrubbing`);
    ``criteria`` is the configured, human-authored search description. No
    Telegram identifiers are carried here by design.
    """

    text: str
    criteria: str = ""


class LLMError(Exception):
    """Base class for all LLM integration errors."""


class LLMConfigError(LLMError):
    """Misconfiguration (for example a missing API key)."""


class LLMTimeoutError(LLMError):
    """The provider did not answer within the configured timeout."""


class LLMRateLimitError(LLMError):
    """The provider reported a rate limit (HTTP 429)."""


class LLMServerError(LLMError):
    """The provider reported a server-side failure (HTTP 5xx)."""


class LLMResponseError(LLMError):
    """The provider answered, but the body/JSON/schema was unusable."""


@runtime_checkable
class LLMProvider(Protocol):
    """Common surface of any analysis provider."""

    provider_name: str
    model: str
    prompt_version: str
    prompt_hash: str

    async def analyze(self, request: LLMRequest) -> AnalysisResult:
        """Analyse one (scrubbed) message; raise :class:`LLMError` on failure."""
        ...
