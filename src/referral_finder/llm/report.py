"""Bootstrap-only aggregated, read-only AI-analysis report (no network, no data changes).

The report is computed with derived SQL queries over the existing tables
(``ai_bootstrap_state``, ``ai_bootstrap_item``, ``message_ai_analysis``,
``api_request_budget``), so there is no separate counter that could drift from
the stored data. It is scoped to one explicit analysis version
(``provider``/``model``/effective ``prompt_version``) so results produced under a
different prompt revision or a different criteria set are never presented as the
current version.

Aggregates cover only the fixed bootstrap snapshot/queue (``ai_bootstrap_item``).
Messages processed exclusively in the steady phase are not included in these
counts.

Output is aggregate-only: temporary chat labels and counts, never message text,
usernames,
links, provider responses or API keys. Running the CLI never connects to
Telegram or DeepSeek and never writes analysis data.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ..classifier import (
    COMPANY_CONTACT_SEARCH,
    JOB_INTEREST,
    REFERRAL_REQUEST,
    UNCERTAIN,
)
from ..config import (
    DEFAULT_LLM_MAX_API_REQUESTS_PER_DAY,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
)
from ..storage import (
    DEFAULT_MAX_ATTEMPTS,
    IncompatibleSchemaError,
    MessageStore,
    ReadOnlyMessageStore,
)
from .budget import ApiRequestBudget
from .prompt import PROMPT_VERSION
from .schema import STATUS_INVALID, STATUS_OK
from .versioning import criteria_hash, effective_analysis_version

# Categories that must always be visible in the report (0 when absent). Any other
# observed category is appended dynamically, so a growing taxonomy is handled
# without a code change.
_REPORT_KEY_CATEGORIES = (
    COMPANY_CONTACT_SEARCH,
    REFERRAL_REQUEST,
    JOB_INTEREST,
    UNCERTAIN,
)


def _empty_counts() -> dict:
    return {"success": 0, "invalid": 0, "error": 0, "pending": 0, "categories": {}}


def _ordered_categories(categories: Mapping[str, int]) -> dict:
    ordered = {name: int(categories.get(name, 0)) for name in _REPORT_KEY_CATEGORIES}
    for name in sorted(categories):
        if name not in ordered:
            ordered[name] = int(categories[name])
    return ordered


def _tally(counts: dict, status, category, attempts, max_attempts: int) -> None:
    """Accumulate one bootstrap item into ``counts`` (derived, no text).

    ``pending`` includes both unprocessed items and retryable ``invalid``/
    ``error`` items, so it can overlap with those counters; the three are not
    disjoint categories and must not be summed as if they were.
    """
    if status is None:
        counts["pending"] += 1
        return
    if status == STATUS_OK:
        counts["success"] += 1
        if category:
            counts["categories"][category] = counts["categories"].get(category, 0) + 1
        return
    if status == STATUS_INVALID:
        counts["invalid"] += 1
    else:
        # Unknown/refused statuses are failures, never successes.
        counts["error"] += 1
    attempts_value = 0 if attempts is None else int(attempts)
    if attempts_value < max_attempts:
        counts["pending"] += 1


def build_report(
    store: MessageStore,
    *,
    chat_ids,
    provider: str,
    model: str,
    prompt_version: str = PROMPT_VERSION,
    criteria: str = "",
    budget_limit: int = DEFAULT_LLM_MAX_API_REQUESTS_PER_DAY,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    clock=None,
) -> dict:
    """Build a bootstrap-only aggregate report for the given analysis version.

    Counts are derived from ``ai_bootstrap_item``; messages processed only in
    the steady phase are outside this report. ``prompt_version`` is combined
    with the criteria hash into the effective analysis version, so only results
    of the active prompt+criteria are counted. Chats without any stored state
    contribute zero counts rather than an error.
    """
    effective = effective_analysis_version(prompt_version, criteria)
    ordered_chat_ids = [int(chat_id) for chat_id in chat_ids]

    per_chat = {chat_id: _empty_counts() for chat_id in ordered_chat_ids}
    for chat_id, status, category, attempts in store.bootstrap_analysis_rows(
        provider, model, effective
    ):
        counts = per_chat.setdefault(chat_id, _empty_counts())
        _tally(counts, status, category, attempts, max_attempts)

    budget = ApiRequestBudget(store, budget_limit, clock=clock)

    chats = []
    totals = _empty_counts()
    for index, chat_id in enumerate(ordered_chat_ids, 1):
        state = store.get_bootstrap_state(chat_id) or {}
        counts = per_chat[chat_id]
        chats.append(
            {
                # Per-report temporary label; the real Telegram id is never
                # exposed in text or structured output and no mapping is kept.
                "chat_id": f"chat_{index}",
                "available_count": int(state.get("available_count", 0)),
                "selected_count": int(state.get("selected_count", 0)),
                "success": counts["success"],
                "invalid": counts["invalid"],
                "error": counts["error"],
                "pending": counts["pending"],
                "categories": _ordered_categories(counts["categories"]),
            }
        )
        totals["success"] += counts["success"]
        totals["invalid"] += counts["invalid"]
        totals["error"] += counts["error"]
        totals["pending"] += counts["pending"]
        for name, value in counts["categories"].items():
            totals["categories"][name] = totals["categories"].get(name, 0) + value

    return {
        "provider": str(provider),
        "model": str(model),
        "prompt_version": effective,
        "criteria_hash": criteria_hash(criteria),
        "budget": {
            "used": budget.used(),
            "limit": budget.limit,
            "remaining": budget.remaining(),
        },
        "chats": chats,
        "totals": {
            "available": sum(chat["available_count"] for chat in chats),
            "selected": sum(chat["selected_count"] for chat in chats),
            "success": totals["success"],
            "invalid": totals["invalid"],
            "error": totals["error"],
            "pending": totals["pending"],
            "categories": _ordered_categories(totals["categories"]),
        },
    }


def format_report(report: Mapping) -> str:
    """Render a report as stable, aggregate-only text (no ids/text beyond chat)."""
    budget = report["budget"]
    lines = [
        "report_scope=bootstrap (bootstrap snapshot/queue only; "
        "steady-only messages are excluded)",
        f"provider={report['provider']} model={report['model']} "
        f"prompt_version={report['prompt_version']}",
        f"budget_used={budget['used']} budget_limit={budget['limit']} "
        f"budget_remaining={budget['remaining']}",
        "pending_note=pending includes unprocessed items and retryable "
        "invalid/error items; pending, invalid and error may overlap and are "
        "not disjoint categories",
    ]
    for chat in report["chats"]:
        lines.append(
            f"chat_id={chat['chat_id']} available={chat['available_count']} "
            f"selected={chat['selected_count']} pending={chat['pending']} "
            f"success={chat['success']} invalid={chat['invalid']} "
            f"error={chat['error']}"
        )
        lines.append(
            "  categories: " + " ".join(
                f"{name}={value}" for name, value in chat["categories"].items()
            )
        )
    totals = report["totals"]
    lines.append(
        f"total available={totals['available']} selected={totals['selected']} "
        f"pending={totals['pending']} success={totals['success']} "
        f"invalid={totals['invalid']} error={totals['error']}"
    )
    lines.append(
        "total categories: " + " ".join(
            f"{name}={value}" for name, value in totals["categories"].items()
        )
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m referral_finder.llm.report",
        description=(
            "Print an aggregate, bootstrap-only AI-analysis report for one "
            "explicit analysis version. Counts cover the bootstrap snapshot/"
            "queue; messages processed only in steady are not included "
            "(read-only, no network, no data changes)."
        ),
    )
    parser.add_argument("--db", required=True, help="path to the SQLite database")
    parser.add_argument(
        "--provider",
        default=DEFAULT_LLM_PROVIDER,
        help=f"analysis provider (default {DEFAULT_LLM_PROVIDER})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LLM_MODEL,
        help=f"analysis model (default {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--prompt-version",
        default=PROMPT_VERSION,
        help=f"prompt template version (default {PROMPT_VERSION})",
    )
    parser.add_argument("--criteria", default="", help="search criteria text")
    parser.add_argument(
        "--budget-limit",
        type=int,
        default=DEFAULT_LLM_MAX_API_REQUESTS_PER_DAY,
        help="daily API request limit shown in the report",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="retry limit used to decide what is still pending",
    )
    parser.add_argument(
        "--chats",
        default=None,
        help=(
            "comma-separated chat ids to report (default: chats that already "
            "have a bootstrap state)"
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="print the report as JSON"
    )
    return parser


def run(
    argv: Optional[Sequence[str]] = None,
    *,
    store: Optional[MessageStore] = None,
    clock=None,
) -> int:
    """Bootstrap-only read-only report CLI. Returns 0/2 (success/operational error)."""
    args = _build_parser().parse_args(argv)
    if args.budget_limit < 1:
        print("error=--budget-limit must be at least 1", file=sys.stderr)
        return 2
    if args.max_attempts < 1:
        print("error=--max-attempts must be at least 1", file=sys.stderr)
        return 2

    chat_ids = None
    if args.chats:
        try:
            chat_ids = [
                int(part.strip()) for part in args.chats.split(",") if part.strip()
            ]
        except ValueError:
            print(
                "error=--chats must be a comma-separated list of integer ids",
                file=sys.stderr,
            )
            return 2

    own_store = store is None
    if own_store:
        db_path = Path(args.db).expanduser()
        if not db_path.exists():
            print(
                "error=report database not found (no report; nothing was created)",
                file=sys.stderr,
            )
            return 2
        try:
            # Never use MessageStore here: its constructor creates tables and
            # runs migrations. The report must only read an existing database.
            store = ReadOnlyMessageStore(db_path)
        except FileNotFoundError:
            print(
                "error=report database not found (no report; nothing was created)",
                file=sys.stderr,
            )
            return 2
        except IncompatibleSchemaError:
            print(
                "error=report database has an incompatible schema "
                "(no report; nothing was changed)",
                file=sys.stderr,
            )
            return 2
        except (sqlite3.Error, OSError):
            print(
                "error=report database could not be opened read-only "
                "(no report; nothing was changed)",
                file=sys.stderr,
            )
            return 2

    try:
        if chat_ids is None:
            chat_ids = store.bootstrap_chat_ids()
        report = build_report(
            store,
            chat_ids=chat_ids,
            provider=args.provider,
            model=args.model,
            prompt_version=args.prompt_version,
            criteria=args.criteria,
            budget_limit=args.budget_limit,
            max_attempts=args.max_attempts,
            clock=clock,
        )
        if args.json:
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        else:
            print(format_report(report))
        return 0
    except (IncompatibleSchemaError, sqlite3.Error):
        print(
            "error=report database could not be read read-only "
            "(no report; nothing was changed)",
            file=sys.stderr,
        )
        return 2
    finally:
        if own_store:
            store.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
