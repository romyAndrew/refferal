"""Offline pipeline check CLI.

Runs ``prefilter → scrub → provider → validation → storage`` against an explicit
SQLite database using only :class:`~referral_finder.llm.fake.FakeProvider`. It
never opens a network connection, never talks to Telegram, and never uses a real
DeepSeek key. It does not read ``.env`` and requires an explicit ``--db``.

The single enable switch is ``LLM_ENABLED`` (read from the process environment,
default off); with it disabled the command reports aggregates only. Output is
aggregate-only: no message text, usernames, links, identifiers, or provider
responses are printed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ..companies import load_default_catalog
from ..prefilter import effective_prefilter_version
from ..storage import DEFAULT_MAX_ATTEMPTS, MessageStore
from .fake import FakeProvider
from .service import (
    STATUS_DUPLICATE,
    STATUS_PREFILTER_SKIPPED,
    STATUS_SAVED,
    STATUS_STORAGE_ERROR,
    MessageAnalysisService,
)
from .versioning import effective_analysis_version

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_enabled(env: Mapping[str, str]) -> bool:
    return (env.get("LLM_ENABLED") or "").strip().lower() in _TRUE_VALUES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m referral_finder.llm.cli",
        description=(
            "Offline AI pipeline check on a test SQLite DB (FakeProvider only, "
            "no network)."
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
        "--dry-run", action="store_true", help="only count not-yet-analysed messages"
    )
    parser.add_argument(
        "--criteria", default="", help="search criteria text passed to the provider"
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
    }


async def _run_pipeline(
    store, provider, enabled, criteria, limit, max_attempts, catalog
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
        prefilter_version=effective_prefilter_version(catalog),
    ):
        for chat_id, message_id, text in batch:
            if limit is not None and counts["checked"] >= limit:
                return counts
            counts["checked"] += 1
            outcome = await service.analyze(chat_id, message_id, text)
            _tally(counts, outcome.status)
    return counts


def _tally(counts: dict, status: str) -> None:
    if status == STATUS_PREFILTER_SKIPPED:
        counts["prefilter_skipped"] += 1
    elif status == STATUS_SAVED:
        counts["saved"] += 1
    elif status == STATUS_DUPLICATE:
        counts["duplicate"] += 1
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
        f"errors={counts['errors']}"
    )


def run(
    argv: Optional[Sequence[str]] = None,
    *,
    enabled: Optional[bool] = None,
    provider=None,
    env: Optional[Mapping[str, str]] = None,
    catalog=None,
) -> int:
    args = _build_parser().parse_args(argv)
    provider = provider or FakeProvider()
    if enabled is None:
        enabled = _env_enabled(env if env is not None else os.environ)
    if catalog is None:
        catalog = load_default_catalog(
            Path(args.companies).expanduser() if args.companies else None
        )

    store = MessageStore(Path(args.db).expanduser())
    try:
        if args.dry_run:
            version = effective_analysis_version(provider.prompt_version, args.criteria)
            pending = store.count_ai_unprocessed(
                provider.provider_name,
                provider.model,
                version,
                max_attempts=args.max_attempts,
                prefilter_version=effective_prefilter_version(catalog),
            )
            print(
                f"dry_run=1 pending={pending} version={version}"
            )
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
            )
        )
        print(_format(counts, enabled=True))
        return 0
    finally:
        store.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
