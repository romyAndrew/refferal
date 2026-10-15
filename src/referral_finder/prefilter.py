"""Independent keyword/template prefilter.

A pure, deterministic gate: no network, no Telegram, no AI, no database. It
decides whether a message text is *potentially* related to the courier-job /
referral domain, so that only a reduced set of messages is later sent for a
semantic (AI) classification.

This module is intentionally independent of
:func:`referral_finder.classifier.classify`: it reuses only the shared
:func:`~referral_finder.classifier.normalize` text normalization and never
consults the classifier's category. The rule-based classifier stays a separate
baseline for comparison.

The gate favours recall (missing a relevant message is worse than a few extra
false positives); every rule has a stable id so False Positive/False Negative
analysis can be done per rule without storing the message text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple

from .classifier import normalize

# Bump whenever the rules change; evaluations and stored selections must be
# attributable to an explicit prefilter version.
PREFILTER_VERSION = "prefilter-2026.10"


@dataclass(frozen=True)
class PrefilterResult:
    """Outcome of the prefilter for one message text.

    ``matched`` is ``True`` when at least one rule fired (the message should be
    considered for the next stage). ``rule_ids`` are the stable ids of the fired
    rules (no message text). ``version`` identifies the rule set used.
    """

    matched: bool
    rule_ids: Tuple[str, ...]
    version: str = PREFILTER_VERSION


@dataclass(frozen=True)
class PrefilterRule:
    id: str
    pattern: "re.Pattern[str]"


# --- recognition fragments -----------------------------------------------
# Distributed domain terms that are distinctive enough to fire on their own.
_REFERRAL = (
    r"(?:реферал\w*|рефк\w*|реф[-\s]?код\w*|пригласительн\w*|инвайт\w*|"
    r"refer\w*|refk\w*|invite\w*)"
)
_COURIER = r"(?:курьер\w*|курер\w*|courier\w*|kuryer\w*|kurier\w*)"
_DELIVERY = r"(?:доставк\w*|развоз\w*|deliver\w*|dostavk\w*)"
_JOB_KIND = r"(?:пеш(?:ий|ем|ком|еход\w*)|вело\w*|самокат\w*|водител\w*)"

# Generic job-search phrasing. Deliberately *not* bare "работ"/"work"/"job":
# those words alone are too broad. A generic term fires only inside a precise
# phrase or collocation (a search/intent verb plus a work noun, or a distinct
# job-search word such as "подработка"/"вакансия").
_GENERIC_JOB = (
    r"(?:подработ\w*|подрабатыва\w*|ваканси\w*|"
    r"(?:ищу|ищем|найти|найду|поиск|в\s+поиске)\s+(?:работ\w*|заработ\w*)|"
    r"предлага\w*\s+работ\w*|"
    r"нужн\w*\s+работ\w*|"
    r"looking\s+for\s+(?:a\s+|an\s+)?(?:job|work)|"
    r"need\s+(?:a\s+)?(?:job|work)|"
    r"part[-\s]?time\s+(?:job|work)|"
    r"hiring|vacanc(?:y|ies)|recruit\w*)"
)


def _rule(rule_id: str, pattern: str) -> PrefilterRule:
    return PrefilterRule(rule_id, re.compile(pattern, re.IGNORECASE))


# Order is stable and defines the order of ``rule_ids`` in a result.
RULES: Tuple[PrefilterRule, ...] = (
    _rule("prefilter.referral", _REFERRAL),
    _rule("prefilter.courier", _COURIER),
    _rule("prefilter.delivery", _DELIVERY),
    _rule("prefilter.job_kind", _JOB_KIND),
    _rule("prefilter.generic_job", _GENERIC_JOB),
)


def prefilter(text: str) -> PrefilterResult:
    """Return whether ``text`` potentially matches the target domain.

    Empty/whitespace-only text (or ``None``) yields ``matched=False`` with no
    rules. The decision is a plain union of independent rule matches and never
    depends on :func:`referral_finder.classifier.classify`.
    """
    normalized = normalize(text)
    if not normalized:
        return PrefilterResult(False, ())
    matched = tuple(rule.id for rule in RULES if rule.pattern.search(normalized))
    return PrefilterResult(bool(matched), matched)


def matches(text: str) -> bool:
    """Convenience boolean wrapper around :func:`prefilter`."""
    return prefilter(text).matched
