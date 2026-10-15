"""Message classification: post-persist service and offline backfill CLI.

``ClassifierService`` is the post-persist hook used by the single writer when
``CLASSIFY_ON_INGEST`` is enabled. ``python -m referral_finder.classify`` runs an
offline backfill over an existing SQLite database (no Telegram connection).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

from .classifier import CATEGORIES, CLASSIFIER_VERSION, Classification, classify
from .models import IncomingMessage
from .storage import MessageStore

logger = logging.getLogger(__name__)


def _classification_tuple(
    chat_id: int, message_id: int, result: Classification, classified_at_utc: int
) -> tuple:
    return (
        int(chat_id),
        int(message_id),
        result.classifier_version,
        result.category,
        result.score,
        json.dumps(list(result.matched_rules)),
        int(classified_at_utc),
    )


class ClassifierService:
    """Classify saved messages and persist results (single-writer friendly).

    Runs inside the :class:`~referral_finder.ingest.MessageWriter` task, so it
    shares the SQLite connection; it never opens a second writer. Any error is
    raised to the caller (the writer logs and continues).
    """

    def __init__(self, store: MessageStore):
        self._store = store

    def classify_saved(self, messages) -> int:
        """Classify ``messages`` and store results; returns inserted rows."""
        now = int(time.time())
        rows = [
            _classification_tuple(m.chat_id, m.message_id, classify(m.text), now)
            for m in messages
            if getattr(m, "text", None)
        ]
        return self._store.save_classifications(rows) if rows else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m referral_finder.classify",
        description="Classify stored messages and store results (no Telegram).",
    )
    parser.add_argument(
        "--db",
        help="SQLite path (default: DB_PATH from the environment/.env)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process at most this many messages (trial run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only report how many messages are not yet classified",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="messages per read/write batch (default 500)",
    )
    return parser


def _resolve_db_path(db_arg: Optional[str]) -> Path:
    if db_arg:
        return Path(db_arg).expanduser()
    # Imported lazily so --db works without Telegram credentials in the env.
    from .config import load_config

    return load_config().db_path


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    version = CLASSIFIER_VERSION
    store = MessageStore(_resolve_db_path(args.db))
    try:
        if args.dry_run:
            pending = store.count_unclassified(version)
            print(f"version={version} unclassified={pending} dry_run=1")
            return 0

        processed = 0
        stored = 0
        errors = 0
        categories: Counter = Counter()

        for batch in store.iter_unclassified(version, batch_size=args.batch_size):
            if args.limit is not None:
                remaining = args.limit - processed
                if remaining <= 0:
                    break
                batch = batch[:remaining]

            now = int(time.time())
            rows = []
            for chat_id, message_id, text in batch:
                processed += 1
                try:
                    result = classify(text)
                except Exception:
                    errors += 1
                    logger.exception(
                        "Classification failed chat_id=%s message_id=%s",
                        chat_id,
                        message_id,
                    )
                    continue
                categories[result.category] += 1
                rows.append(
                    _classification_tuple(chat_id, message_id, result, now)
                )
            if rows:
                stored += store.save_classifications(rows)

            if args.limit is not None and processed >= args.limit:
                break

        summary = " ".join(
            f"{name}={categories.get(name, 0)}" for name in CATEGORIES
        )
        print(
            f"version={version} processed={processed} stored={stored} "
            f"errors={errors} {summary}"
        )
        return 0
    finally:
        store.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
