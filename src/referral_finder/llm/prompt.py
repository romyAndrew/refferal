"""Prompt definition, version and content hash.

The prompt is static (system instructions) and versioned explicitly. The hash
covers the instructions plus the version, but **not** the per-message text, so a
stored analysis can be attributed to an exact prompt revision. Telegram
identifiers, links and other metadata are never placed in the prompt; callers
pass only the scrubbed message text (see :mod:`referral_finder.scrubbing`).
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence

from .base import LLMRequest

PROMPT_VERSION = "prompt-2026.10"

_SYSTEM_INSTRUCTIONS = (
    "You classify a single message for a courier-job / referral lead finder. "
    "Reply with strict JSON only, no prose, with exactly these fields: "
    '"category", "relevant", "confidence", "explanation". '
    '"category" must be one of: referral_request, job_interest, '
    "referral_offer, recruitment_offer, irrelevant, uncertain. "
    '"relevant" must be true for referral_request, job_interest, '
    "referral_offer, recruitment_offer; false for irrelevant; null for "
    "uncertain. "
    '"confidence" is a number in [0, 1] (your own heuristic certainty, not a '
    "calibrated probability). "
    '"explanation" is a short reason. '
    "If evidence is insufficient or conflicting, use category=uncertain and "
    "relevant=null. "
    "Classify only the given message. Do not request or invent metadata. Do not "
    "repeat personal data. Output JSON only."
)

# Stable content hash of the prompt revision (version + instructions).
PROMPT_HASH = hashlib.sha256(
    (PROMPT_VERSION + "\n" + _SYSTEM_INSTRUCTIONS).encode("utf-8")
).hexdigest()


def build_messages(request: LLMRequest) -> List[Dict[str, str]]:
    """Build chat messages for ``request`` (system + user).

    The user message contains only the configured search criteria and the
    scrubbed message text. No Telegram metadata is included.
    """
    criteria = (request.criteria or "").strip()
    parts: List[str] = []
    if criteria:
        parts.append("Search criteria:\n" + criteria)
    parts.append("Message:\n" + (request.text or ""))
    return [
        {"role": "system", "content": _SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def system_instructions() -> str:
    """Expose the static instructions (used by tests/diagnostics)."""
    return _SYSTEM_INSTRUCTIONS


def prompt_material() -> Sequence[str]:
    """The exact pieces covered by :data:`PROMPT_HASH`."""
    return (PROMPT_VERSION, _SYSTEM_INSTRUCTIONS)
