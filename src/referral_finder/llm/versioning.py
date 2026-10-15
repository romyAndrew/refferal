"""Effective analysis version: prompt template version + criteria hash.

The stored identity of an AI result is ``(chat_id, message_id, provider, model,
prompt_version)``. To make a change of the configured search *criteria* produce a
distinct identity (and never reuse a result computed under different criteria),
the value stored in the ``prompt_version`` column is the **effective analysis
version**:

    <template prompt version> + "c" + <short hash of normalized criteria>

Contract (deliberate and documented):

* The criteria text is normalized (trimmed, internal whitespace collapsed) and
  hashed with SHA-256; only the short hash is embedded, never the raw criteria.
* The template prompt version is kept as a prefix, so the template revision and
  the criteria are both part of the identity without conflating them.
* Empty criteria are allowed and still produce a stable, deterministic version.
"""

from __future__ import annotations

import hashlib
import re

CRITERIA_HASH_LENGTH = 12

_WHITESPACE = re.compile(r"\s+")


def normalize_criteria(criteria: str) -> str:
    """Trim and collapse internal whitespace (stable across formatting)."""
    return _WHITESPACE.sub(" ", (criteria or "").strip())


def criteria_hash(criteria: str) -> str:
    """Short, stable hash of the normalized criteria."""
    normalized = normalize_criteria(criteria)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:CRITERIA_HASH_LENGTH]


def effective_analysis_version(prompt_version: str, criteria: str) -> str:
    """Combine the prompt template version with the criteria hash."""
    return f"{prompt_version}+c{criteria_hash(criteria)}"
