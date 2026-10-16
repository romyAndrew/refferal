"""Structured AI-analysis contract shared by every provider.

The categories are reused from the shared taxonomy in
:mod:`referral_finder.classifier`, so the LLM, the rule baseline and reporting
speak the same vocabulary; their meaning is not changed here.

Relationship between ``relevant`` and ``category`` (enforced by
:func:`validate_analysis`):

========================= ===========
``category``              ``relevant``
========================= ===========
referral_request          True
job_interest              True
referral_offer            True
recruitment_offer         True
company_contact_search    True
irrelevant                False
uncertain                 None (abstain)
========================= ===========

``uncertain`` is therefore kept as a distinct result and is never folded into
``irrelevant``. ``confidence`` is a heuristic self-assessment in ``[0, 1]``; it
must not be presented as the (calibrated) probability of a correct answer.

``company_contact_search`` is a **weak** signal: the author looks for a person,
employee or acquaintance connected to a company/service without asking for a
job or a referral link. It is stored and reported separately and must not be
treated as a confirmed lead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..classifier import (
    CATEGORIES,
    COMPANY_CONTACT_SEARCH,
    IRRELEVANT,
    JOB_INTEREST,
    RECRUITMENT_OFFER,
    REFERRAL_OFFER,
    REFERRAL_REQUEST,
    UNCERTAIN,
)

RELEVANT_CATEGORIES = (
    REFERRAL_REQUEST,
    JOB_INTEREST,
    REFERRAL_OFFER,
    RECRUITMENT_OFFER,
    COMPANY_CONTACT_SEARCH,
)

STATUS_OK = "ok"
STATUS_INVALID = "invalid"
STATUS_ERROR = "error"
STATUS_REFUSED = "refused"
STATUSES = (STATUS_OK, STATUS_INVALID, STATUS_ERROR, STATUS_REFUSED)


class ValidationError(ValueError):
    """Raised when a provider response does not match the analysis contract."""


@dataclass(frozen=True)
class AnalysisResult:
    """Validated result of analysing one message.

    ``relevant`` is ``None`` only for ``uncertain`` (abstain). ``confidence`` is
    a heuristic number in ``[0, 1]`` and is not a calibrated probability.
    """

    category: str
    relevant: Optional[bool]
    confidence: float
    explanation: str


def expected_relevant(category: str) -> Optional[bool]:
    """Return the ``relevant`` value mandated for ``category``."""
    if category == UNCERTAIN:
        return None
    return category in RELEVANT_CATEGORIES


def validate_analysis(payload) -> AnalysisResult:
    """Validate a raw provider payload and return an :class:`AnalysisResult`.

    Raises :class:`ValidationError` on any type, enum or consistency problem.
    """
    if not isinstance(payload, dict):
        raise ValidationError("analysis payload must be an object")

    category = payload.get("category")
    if category not in CATEGORIES:
        raise ValidationError(f"invalid category: {category!r}")

    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValidationError("confidence must be a number")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValidationError("confidence must be within [0, 1]")

    explanation = payload.get("explanation")
    if not isinstance(explanation, str):
        raise ValidationError("explanation must be a string")

    relevant = payload.get("relevant")
    if category == UNCERTAIN:
        if relevant is not None:
            raise ValidationError("uncertain must abstain (relevant=null)")
    else:
        if not isinstance(relevant, bool):
            raise ValidationError("relevant must be a boolean")
        if relevant != expected_relevant(category):
            raise ValidationError(
                f"relevant={relevant} is inconsistent with category={category!r}"
            )

    return AnalysisResult(
        category=category,
        relevant=relevant,
        confidence=round(confidence, 4),
        explanation=explanation,
    )


def ai_analysis_row(
    chat_id: int,
    message_id: int,
    result: Optional[AnalysisResult],
    *,
    provider: str,
    model: str,
    prompt_version: str,
    prompt_hash: str,
    analyzed_at_utc: int,
    status: str = STATUS_OK,
    error: Optional[str] = None,
) -> tuple:
    """Build a storage row for the ``message_ai_analysis`` table.

    Use ``result=None`` together with ``status`` in
    ``{STATUS_INVALID, STATUS_ERROR, STATUS_REFUSED}`` to record a failure
    without any analysis fields.
    """
    if result is None:
        relevant_int = None
        category = None
        confidence = None
        explanation = None
    else:
        relevant_int = None if result.relevant is None else int(result.relevant)
        category = result.category
        confidence = result.confidence
        explanation = result.explanation
    return (
        int(chat_id),
        int(message_id),
        str(provider),
        str(model),
        str(prompt_version),
        str(prompt_hash),
        str(status),
        relevant_int,
        category,
        confidence,
        explanation,
        error,
        int(analyzed_at_utc),
    )
