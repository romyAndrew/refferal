"""Deterministic offline provider for tests.

``FakeProvider`` never touches the network. It records every ``LLMRequest`` it
receives (so tests can assert only scrubbed text was sent) and returns a result
from an injected handler or a static mapping.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

from ..classifier import UNCERTAIN
from .base import LLMRequest
from .prompt import PROMPT_HASH, PROMPT_VERSION
from .schema import AnalysisResult

Outcome = Union[AnalysisResult, BaseException]
Handler = Callable[[LLMRequest], Outcome]


def _default_handler(_request: LLMRequest) -> AnalysisResult:
    return AnalysisResult(UNCERTAIN, None, 0.0, "fake provider: no handler")


class FakeProvider:
    """In-memory provider that returns deterministic results.

    ``handler`` maps a request to either an :class:`AnalysisResult` or an
    exception instance to raise. When omitted, every request yields
    ``uncertain``. ``calls`` keeps the received requests for assertions.
    """

    provider_name = "fake"

    def __init__(
        self,
        handler: Optional[Handler] = None,
        *,
        model: str = "fake-model",
        prompt_version: str = PROMPT_VERSION,
        prompt_hash: str = PROMPT_HASH,
    ):
        self.model = model
        self.prompt_version = prompt_version
        self.prompt_hash = prompt_hash
        self._handler: Handler = handler or _default_handler
        self.calls: List[LLMRequest] = []

    @classmethod
    def from_mapping(
        cls,
        mapping: Dict[str, Outcome],
        *,
        default: Optional[Outcome] = None,
        **kwargs,
    ) -> "FakeProvider":
        """Build a provider keyed by message text (with an optional fallback)."""
        fallback = default if default is not None else (
            AnalysisResult(UNCERTAIN, None, 0.0, "fake provider: unmapped")
        )

        def handler(request: LLMRequest) -> Outcome:
            return mapping.get(request.text, fallback)

        return cls(handler, **kwargs)

    async def analyze(self, request: LLMRequest) -> AnalysisResult:
        self.calls.append(request)
        outcome = self._handler(request)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
