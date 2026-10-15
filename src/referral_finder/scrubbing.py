"""Best-effort text scrubbing before sending text to an external AI provider.

Removes or replaces a set of *machine-detectable* sensitive patterns: email
addresses, phone numbers, ``@usernames``, Telegram profile/message links, other
URLs, and a few recognizable technical identifiers (for example ``-100...``
supergroup ids or ``user_id=...`` fragments).

Scope and honest limits:

* This is pattern-based and **does not guarantee anonymization**. Arbitrary
  personal names and many forms of sensitive data cannot be reliably removed
  with regular expressions, and a person can identify themselves directly in
  free text. Callers must not treat the output as fully de-identified.
* Structured Telegram metadata (``sender_id``, ``chat_id``, ``message_id``,
  ``topic_id``, username, source URL, chat title) is *never* part of the text
  passed here; this module only cleans a message body.

The module is pure and offline: no network, no Telegram, no AI, no database.
It returns the cleaned text plus the stable ids of the rules that fired; the
original fragments are never returned or logged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

# Bump whenever the rules change; stored AI results reference this version.
SCRUBBING_VERSION = "scrub-2026.10"

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# Telegram-specific links first, so they are not swallowed by the generic URL
# rule (they carry profile/message references).
_TELEGRAM_LINK = re.compile(
    r"(?:(?:https?://)?(?:t\.me|telegram\.me|telegram\.dog)/[^\s<>\"']*)"
    r"|(?:tg://[^\s<>\"']*)",
    re.IGNORECASE,
)
_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
_USERNAME = re.compile(r"(?<![\w@])@[A-Za-z][A-Za-z0-9_]{2,31}\b")
# Recognizable technical identifiers: supergroup/channel marked ids and labelled
# id fragments. Bare long digit runs are handled by the phone rule below.
_TECH_ID = re.compile(
    r"(?:-100\d{6,})"
    r"|(?:\b(?:id|chat[_\s\-]?id|user[_\s\-]?id|message[_\s\-]?id|msg[_\s\-]?id"
    r"|topic[_\s\-]?id|peer)\s*[:=#]?\s*\d{4,}\b)",
    re.IGNORECASE,
)
_PHONE_CANDIDATE = re.compile(r"(?<!\d)\+?\d[\d\s().\-]{7,}\d(?!\d)")


def _phone_replacement(match: "re.Match[str]") -> str:
    """Replace only digit runs that look like a phone number (10-15 digits)."""
    digits = re.sub(r"\D", "", match.group(0))
    if 10 <= len(digits) <= 15:
        return "[phone]"
    return match.group(0)


# (rule_id, pattern, replacement). Replacement may be a callable.
_RULES: Tuple[Tuple[str, "re.Pattern[str]", object], ...] = (
    ("scrub.telegram_link", _TELEGRAM_LINK, "[link]"),
    ("scrub.email", _EMAIL, "[email]"),
    ("scrub.url", _URL, "[url]"),
    ("scrub.username", _USERNAME, "[username]"),
    ("scrub.tech_id", _TECH_ID, "[id]"),
    ("scrub.phone", _PHONE_CANDIDATE, _phone_replacement),
)


@dataclass(frozen=True)
class ScrubResult:
    """Cleaned text and the ids of the rules that fired (no source fragments)."""

    text: str
    rule_ids: Tuple[str, ...]
    version: str = SCRUBBING_VERSION


def scrub(text: str) -> ScrubResult:
    """Return ``text`` with machine-detectable sensitive patterns replaced.

    Empty/``None`` input yields empty text and no rules. The returned text is
    best-effort de-identification only (see the module docstring); original
    fragments are never returned, stored in diagnostics, or logged.
    """
    if not text:
        return ScrubResult("", ())

    cleaned = text
    fired: List[str] = []
    for rule_id, pattern, replacement in _RULES:
        updated = pattern.sub(replacement, cleaned)  # type: ignore[arg-type]
        if updated != cleaned:
            fired.append(rule_id)
            cleaned = updated
    return ScrubResult(cleaned, tuple(fired))
