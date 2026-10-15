"""Rule-based message classifier.

A pure, deterministic function: no network, no external LLM/API, no randomness.
It classifies the stored message text (or media caption) into one of six
categories. The result carries the matched rule ids so a decision can be
explained without copying the message text.

Language coverage: Russian, common English delivery/job/referral terms, and
common transliteration. ``score`` is a heuristic weight sum, **not** a
calibrated probability, and must not be presented as one.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Tuple

# Active classifier version. Bump this whenever rules change; the store keeps
# one result per (chat_id, message_id, classifier_version).
CLASSIFIER_VERSION = "rules-2026.06"

REFERRAL_REQUEST = "referral_request"
JOB_INTEREST = "job_interest"
REFERRAL_OFFER = "referral_offer"
RECRUITMENT_OFFER = "recruitment_offer"
IRRELEVANT = "irrelevant"
UNCERTAIN = "uncertain"

CATEGORIES: Tuple[str, ...] = (
    REFERRAL_REQUEST,
    JOB_INTEREST,
    REFERRAL_OFFER,
    RECRUITMENT_OFFER,
    IRRELEVANT,
    UNCERTAIN,
)

# Categories that form the candidate queue (see ``MessageStore.iter_candidates``).
CANDIDATE_CATEGORIES: Tuple[str, ...] = (REFERRAL_REQUEST, JOB_INTEREST)

# A category must reach this weight to be chosen; below it the result is
# ``uncertain`` (e.g. a lone weak signal).
MIN_SCORE = 3
# Heuristic normalisation cap used only to map the weight sum into [0, 1].
SCORE_CAP = 9.0


@dataclass(frozen=True)
class Classification:
    """Result of classifying one message.

    ``score`` is a heuristic weight (normalised to [0, 1] via ``SCORE_CAP``),
    not a probability. ``matched_rules`` are stable rule ids (no message text).
    """

    category: str
    score: float
    matched_rules: Tuple[str, ...]
    classifier_version: str = CLASSIFIER_VERSION


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    weight: float
    pattern: "re.Pattern[str]"


# --- normalization -------------------------------------------------------

_CYRILLIC = re.compile(r"[а-я]")
_LATIN = re.compile(r"[a-z]")
# Only used inside a token that mixes Cyrillic and Latin, to fold stray Latin
# lookalikes onto their Cyrillic counterparts (e.g. "рeфералка" -> "рефералка").
_LATIN_TO_CYRILLIC = str.maketrans(
    {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у",
     "k": "к", "m": "м", "t": "т"}
)


def _unify_mixed_script(text: str) -> str:
    parts = []
    for token in text.split(" "):
        if _CYRILLIC.search(token) and _LATIN.search(token):
            token = token.translate(_LATIN_TO_CYRILLIC)
        parts.append(token)
    return " ".join(parts)


def normalize(text: str) -> str:
    """Lowercase, NFC, ``ё``->``е``, collapsed whitespace, mixed-script folding."""
    if not text:
        return ""
    value = unicodedata.normalize("NFC", text).lower().replace("ё", "е")
    value = " ".join(value.split())
    return _unify_mixed_script(value)


# --- recognition fragments -----------------------------------------------

_REFERRAL = (
    r"(?:реферал\w*|рефк\w*|реф[-\s]?код\w*|пригласительн\w*|инвайт\w*|"
    r"refer\w*|refk\w*|\bref\b|invite\w*)"
)
_JOB = (
    r"(?:курьер\w*|курер\w*|доставк\w*|пеш(?:ий|ем|ком|еход\w*)|вело\w*|"
    r"самокат\w*|водител\w*|развоз\w*|courier\w*|kuryer\w*|kurier\w*|"
    r"dostavk\w*|deliver\w*)"
)
_WORK = r"(?:работ\w*|подработ\w*|job|work)"
# Domains that are specific to the target task (used to split irrelevant vs
# uncertain when no rule matched).
_TARGET = r"(?:" + _REFERRAL + r"|" + _JOB + r")"

_REQUEST_CUE = (
    r"(?:дай(?:те)?|дад\w*|скин\w*|кин\w*|поделись|поделит\w*|"
    r"можешь|может\w*|можно|нужен|нужна|ищу|где\s+взять|"
    r"где\s+найти|где\s+получить|как\s+получить|как\s+взять|подскаж\w*|"
    r"кто\s+даст|есть\s+у\s+кого|хочу|need|want|give|send|share|"
    r"looking\s+for|where\s+to\s+get|how\s+to\s+get|\bdai\b|\bskin\b|"
    r"\bpodelis\b|gde\s+vzyat|kak\s+poluchit|podskazh\w*)"
)
_OFFER_CUE = (
    r"(?:вот|держи|моя|мой|дам|делюсь|поделюсь|оформл\w*|подключ\w*|"
    r"регистрируй\w*|присоединяй\w*|приглаша\w*|используй|here|take|use|join|"
    r"\bvot\b|\bderzhi\b|\bdam\b|\bmoya\b|podklyuch\w*|oforml\w*)"
)
_RECRUIT_CUE = (
    r"(?:набира\w*|набор\w*|требуются|требуется|ищем|ищет|ваканси\w*|нужны|"
    r"приглашаем|hiring|vacancy|vacancies|recruiting|nabirayu|nabor|vakansi\w*)"
)
_SEEK_CUE = (
    r"(?:ищу|хочу|нужна|подскажите|рассматриваю|интересует|"
    r"как\s+устроиться|как\s+попасть|как\s+начать|возьмут|берут|"
    r"условия|какие\s+условия|сколько\s+платят|какая\s+оплата|"
    r"сколько\s+зарабатыва\w*|график|"
    r"looking\s+for|want|need|interested|how\s+to\s+apply|"
    r"\bishu\b|\bhochu\b|nuzhna|kak\s+ustroitsya)"
)

_TARGET_RE = re.compile(_TARGET, re.IGNORECASE)
_ANY_CUE_RE = re.compile(
    r"(?:%s|%s|%s|%s)" % (_REQUEST_CUE, _OFFER_CUE, _RECRUIT_CUE, _SEEK_CUE),
    re.IGNORECASE,
)


def _rule(rule_id: str, category: str, weight: float, pattern: str) -> Rule:
    return Rule(rule_id, category, weight, re.compile(pattern, re.IGNORECASE))


# Intent rules for the same category add up; different categories at the same
# weight that both fire produce ``uncertain`` (never guess). Recruitment has a
# slightly higher weight so a recruiter post mentioning terms also present in a
# job-seeker post is not misread as job interest.
RULES: Tuple[Rule, ...] = (
    _rule(
        "req.referral.give",
        REFERRAL_REQUEST,
        4.0,
        rf"(?:{_REQUEST_CUE}.{{0,25}}{_REFERRAL})|(?:{_REFERRAL}.{{0,25}}{_REQUEST_CUE})",
    ),
    _rule(
        "job.interest.seek",
        JOB_INTEREST,
        4.0,
        rf"(?:{_SEEK_CUE}.{{0,30}}{_JOB})|(?:{_JOB}.{{0,30}}{_SEEK_CUE})",
    ),
    _rule(
        "offer.referral.share",
        REFERRAL_OFFER,
        4.0,
        rf"(?:{_OFFER_CUE}.{{0,25}}{_REFERRAL})|(?:{_REFERRAL}.{{0,25}}{_OFFER_CUE})",
    ),
    _rule(
        "offer.recruit.vacancy",
        RECRUITMENT_OFFER,
        5.0,
        rf"(?:{_RECRUIT_CUE}.{{0,30}}{_JOB})|(?:{_JOB}.{{0,30}}{_RECRUIT_CUE})",
    ),
    _rule(
        "offer.recruit.join",
        RECRUITMENT_OFFER,
        5.0,
        rf"(?:подключ\w*|оформл\w*|приглаша\w*|podklyuch\w*|oforml\w*)"
        rf".{{0,25}}{_JOB}",
    ),
    _rule(
        "context.past_experience",
        IRRELEVANT,
        3.0,
        r"(?:уже|раньше|вчера|оформил\w*|оформился|работаю|получил|"
        r"зарегистрировался|был\w*|already|used\s+to)"
        rf".{{0,25}}(?:по\s+(?:ссылке|{_REFERRAL})|{_JOB}|{_WORK})",
    ),
    _rule(
        "irrelevant.negation",
        IRRELEVANT,
        5.0,
        rf"(?:(?:не\s+(?:нужн\w*|надо|ищу|интерес\w*)|неинтересно|не\s+нужен|"
        rf"не\s+нужна|работа\s+не\s+нужна).{{0,25}}(?:{_JOB}|{_WORK}|{_REFERRAL}))|"
        rf"(?:(?:{_JOB}|{_WORK}|{_REFERRAL}).{{0,25}}(?:не\s+(?:нужн\w*|надо|"
        rf"интерес\w*)|неинтересно|\bnot\s+interested\b))",
    ),
)


def _weight_to_score(weight: float) -> float:
    return round(min(1.0, weight / SCORE_CAP), 4)


def classify(text: str) -> Classification:
    """Classify message ``text`` into one of :data:`CATEGORIES`.

    Empty text yields ``uncertain``. When no rule matched: text that carries a
    target domain word or a generic intent cue is ``uncertain``; otherwise it is
    ``irrelevant``. A category is chosen only if it strictly outscores every
    other category and reaches :data:`MIN_SCORE`.
    """
    normalized = normalize(text)
    if not normalized:
        return Classification(UNCERTAIN, 0.0, ())

    matched = tuple(rule for rule in RULES if rule.pattern.search(normalized))
    if not matched:
        if _TARGET_RE.search(normalized) or _ANY_CUE_RE.search(normalized):
            return Classification(UNCERTAIN, 0.0, ())
        return Classification(IRRELEVANT, 0.0, ())

    matched_ids = tuple(rule.id for rule in matched)
    scores: dict[str, float] = {}
    for rule in matched:
        scores[rule.category] = scores.get(rule.category, 0.0) + rule.weight

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top_category, top_score = ranked[0]

    if top_score < MIN_SCORE:
        return Classification(UNCERTAIN, _weight_to_score(top_score), matched_ids)
    if len(ranked) > 1 and ranked[1][1] == top_score:
        return Classification(UNCERTAIN, _weight_to_score(top_score), matched_ids)
    return Classification(top_category, _weight_to_score(top_score), matched_ids)
