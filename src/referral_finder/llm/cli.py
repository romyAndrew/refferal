"""Offline pipeline check and opt-in historical backfill CLI.

Without ``--live`` it runs ``prefilter → scrub → provider → validation →
storage`` against an explicit SQLite database using only
:class:`~referral_finder.llm.fake.FakeProvider`. It never opens a network
connection, never talks to Telegram, and never uses a real DeepSeek key. It does
not read ``.env`` and requires an explicit ``--db``.

Historical backfill of an existing database is a separate, explicitly confirmed
operation: it requires ``--live --include-history --limit N`` *and*
``LLM_ENABLED=true`` and ``DEEPSEEK_API_KEY`` in the environment. It reuses the
application's persistent daily :class:`~referral_finder.llm.budget.ApiRequestBudget`
(via the provider), so it cannot exceed the same limit as the running worker;
once the daily limit is reached it stops and leaves the remaining messages
queued. There is no automatic historical backfill at startup.

Output is aggregate-only: no message text, usernames, links, identifiers, or
provider responses are printed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ..companies import CompanyCatalogError, load_default_catalog
from ..config import (
    ConfigError,
    load_config,
    resolve_environment,
    resolve_llm_max_api_requests_per_day,
)
from ..prefilter import effective_prefilter_version
from ..storage import DEFAULT_MAX_ATTEMPTS, MessageStore
from .base import LLMConfigError
from .factory import DEEPSEEK_API_KEY_ENV, provider_from_config
from .fake import FakeProvider
from .service import (
    STATUS_BUDGET_EXHAUSTED,
    STATUS_DUPLICATE,
    STATUS_PREFILTER_SKIPPED,
    STATUS_SAVED,
    STATUS_STORAGE_ERROR,
    MessageAnalysisService,
)
from .versioning import effective_analysis_version

DEFAULT_BATCH_SIZE = 500
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_enabled(env: Mapping[str, str]) -> bool:
    return (env.get("LLM_ENABLED") or "").strip().lower() in _TRUE_VALUES


def _live_env_error(env: Mapping[str, str]) -> Optional[str]:
    """Return a reason the live backfill may not run, or ``None`` when allowed."""
    if not _env_enabled(env):
        return "LLM_ENABLED is not true"
    if not env.get(DEEPSEEK_API_KEY_ENV):
        return f"{DEEPSEEK_API_KEY_ENV} is not set"
    provider = (env.get("LLM_PROVIDER") or "").strip().lower()
    if provider in ("fake", "none"):
        return (
            "LLM_PROVIDER must name a real provider for --live "
            "(fake/none are not allowed)"
        )
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m referral_finder.llm.cli",
        description=(
            "Offline AI pipeline check on a test SQLite DB (FakeProvider only, "
            "no network) or an explicitly confirmed live historical backfill."
        ),
    )
    parser.add_argument(
        "--db",
        required=True,
        help="path to a SQLite DB (required; the user DB is never used by default)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="process at most this many messages"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"messages per keyset page (default {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="only count not-yet-analysed messages"
    )
    parser.add_argument(
        "--criteria", default="", help="search criteria text passed to the provider"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="use the real provider (only together with --include-history)",
    )
    parser.add_argument(
        "--include-history",
        action="store_true",
        help=(
            "explicit historical backfill of all pending messages "
            "(requires --live and a positive --limit)"
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=(
            "how many times a failed message is retried before it is left "
            f"alone (default {DEFAULT_MAX_ATTEMPTS})"
        ),
    )
    parser.add_argument(
        "--companies",
        default=None,
        help=(
            "company reference catalog (default: the repository "
            "config/companies.yaml, or TG_COMPANIES_FILE if set)"
        ),
    )
    return parser


def _empty_counts() -> dict:
    return {
        "checked": 0,
        "prefilter_skipped": 0,
        "saved": 0,
        "duplicate": 0,
        "invalid": 0,
        "errors": 0,
        "budget_exhausted": 0,
    }


async def _run_pipeline(
    store, provider, enabled, criteria, limit, max_attempts, catalog, batch_size
) -> dict:
    service = MessageAnalysisService(
        store, provider, enabled=enabled, criteria=criteria, catalog=catalog
    )
    version = effective_analysis_version(provider.prompt_version, criteria)
    counts = _empty_counts()
    for batch in store.iter_ai_unprocessed(
        provider.provider_name,
        provider.model,
        version,
        max_attempts=max_attempts,
        batch_size=batch_size,
        prefilter_version=effective_prefilter_version(catalog),
    ):
        for chat_id, message_id, text in batch:
            if limit is not None and counts["checked"] >= limit:
                return counts
            counts["checked"] += 1
            outcome = await service.analyze(chat_id, message_id, text)
            _tally(counts, outcome.status)
            if outcome.status == STATUS_BUDGET_EXHAUSTED:
                # Daily budget reached: stop and leave the rest queued. Nothing
                # is deleted and no attempt is recorded for the skipped messages.
                return counts
    return counts


def _tally(counts: dict, status: str) -> None:
    if status == STATUS_PREFILTER_SKIPPED:
        counts["prefilter_skipped"] += 1
    elif status == STATUS_SAVED:
        counts["saved"] += 1
    elif status == STATUS_DUPLICATE:
        counts["duplicate"] += 1
    elif status == STATUS_BUDGET_EXHAUSTED:
        counts["budget_exhausted"] += 1
    elif status == "invalid":
        counts["invalid"] += 1
    elif status in ("error", STATUS_STORAGE_ERROR):
        counts["errors"] += 1


def _format(counts: dict, *, enabled: bool) -> str:
    prefilter_passed = counts["checked"] - counts["prefilter_skipped"]
    return (
        f"enabled={int(enabled)} checked={counts['checked']} "
        f"prefilter_passed={prefilter_passed} saved={counts['saved']} "
        f"prefilter_skipped={counts['prefilter_skipped']} "
        f"duplicate={counts['duplicate']} invalid={counts['invalid']} "
        f"errors={counts['errors']} budget_exhausted={counts['budget_exhausted']}"
    )


def _llm_config_from_env(env: Mapping[str, str]):
    """Build a ``Config`` for the provider factory without Telegram credentials.

    Only the LLM-related variables are read from ``env``; harmless placeholders
    stand in for the Telegram-required fields, which the LLM path never uses.
    ``load_config`` is called with an explicit mapping, so ``.env`` is not read.
    """
    merged = dict(env)
    merged.setdefault("TG_API_ID", "1")
    merged.setdefault("TG_API_HASH", "unused")
    merged.setdefault("TG_SESSION_PATH", "unused.session")
    return load_config(merged, chats_file=Path("__no_chats_configured__.txt"))


def run(
    argv: Optional[Sequence[str]] = None,
    *,
    enabled: Optional[bool] = None,
    provider=None,
    env: Optional[Mapping[str, str]] = None,
    catalog=None,
    clock=None,
) -> int:
    args = _build_parser().parse_args(argv)
    # Live commands use the same .env/environment source as the application;
    # offline checks keep their explicit mapping (no .env read).
    if args.live:
        env_map = resolve_environment(env)
    else:
        env_map = os.environ if env is None else env

    if args.include_history and not args.live:
        print("error=--include-history requires --live", file=sys.stderr)
        return 2
    if args.live and not args.include_history:
        print(
            "error=--live is only supported together with --include-history",
            file=sys.stderr,
        )
        return 2
    if args.include_history and (args.limit is None or args.limit < 1):
        print(
            "error=--include-history requires --limit with a positive integer",
            file=sys.stderr,
        )
        return 2
    if args.batch_size < 1:
        print("error=--batch-size must be at least 1", file=sys.stderr)
        return 2
    if args.live and isinstance(provider, FakeProvider):
        print(
            "error=--live does not accept the offline FakeProvider",
            file=sys.stderr,
        )
        return 2

    db_path = Path(args.db).expanduser()
    if args.live and provider is None:
        reason = _live_env_error(env_map)
        if reason is not None:
            print(f"error={reason}", file=sys.stderr)
            return 2
        try:
            resolve_llm_max_api_requests_per_day(env_map)
        except ConfigError as exc:
            print(f"error={exc}", file=sys.stderr)
            return 2
    if args.live and not db_path.exists():
        # Never silently create a fresh database (and thus a fresh budget).
        print("error=backfill database not found; nothing was sent", file=sys.stderr)
        return 2

    try:
        store = MessageStore(db_path)
    except (sqlite3.Error, OSError) as exc:
        print(
            f"error=backfill database could not be opened ({type(exc).__name__})",
            file=sys.stderr,
        )
        return 2

    try:
        if provider is None:
            if args.live:
                try:
                    config = _llm_config_from_env(env_map)
                    provider = provider_from_config(
                        config,
                        store=store,
                        environ=env_map,
                        clock=clock,
                    )
                except (LLMConfigError, ConfigError) as exc:
                    print(f"error={exc}", file=sys.stderr)
                    return 2
            else:
                provider = FakeProvider()

        if catalog is None:
            try:
                catalog = load_default_catalog(
                    Path(args.companies).expanduser() if args.companies else None
                )
            except (CompanyCatalogError, OSError) as exc:
                print(
                    "error=company catalog could not be loaded "
                    f"({type(exc).__name__})",
                    file=sys.stderr,
                )
                return 2

        if args.live:
            enabled = True
        elif enabled is None:
            enabled = _env_enabled(env_map)

        if args.dry_run:
            version = effective_analysis_version(provider.prompt_version, args.criteria)
            pending = store.count_ai_unprocessed(
                provider.provider_name,
                provider.model,
                version,
                max_attempts=args.max_attempts,
                prefilter_version=effective_prefilter_version(catalog),
            )
            print(f"dry_run=1 pending={pending} version={version}")
            return 0

        if not enabled:
            print(_format(_empty_counts(), enabled=False))
            return 0

        counts = asyncio.run(
            _run_pipeline(
                store,
                provider,
                enabled,
                args.criteria,
                args.limit,
                args.max_attempts,
                catalog,
                args.batch_size,
            )
        )
        print(_format(counts, enabled=True))
        return 0
    except sqlite3.Error as exc:
        print(
            f"error=backfill database operation failed ({type(exc).__name__})",
            file=sys.stderr,
        )
        return 2
    finally:
        store.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
