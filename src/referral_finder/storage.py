"""SQLite storage for incoming messages (stage 3).

Uses only the standard-library :mod:`sqlite3` module. The database directory is
created on demand and the ``messages`` table is created on first use.

Deduplication relies on the composite primary key ``(chat_id, message_id)``
together with ``INSERT ... ON CONFLICT(chat_id, message_id) DO NOTHING``: a
repeated message is neither inserted twice nor used to overwrite the original
row.

No extra profile data is collected and no media is downloaded. This module only
maps the storage-ready DTO onto rows; converting Telethon objects to that DTO
happens elsewhere (see :mod:`referral_finder.telegram_client`).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Tuple

from .classifier import CANDIDATE_CATEGORIES as _CANDIDATE_CATEGORIES
from .models import IncomingMessage

# A fetched row: (chat_id, message_id, date_utc, received_at_utc, text, sender_id).
Row = Tuple[int, int, int, int, Optional[str], Optional[int]]

# Smallest SQLite INTEGER, used as the initial keyset cursor for pagination.
_INT64_MIN = -(2 ** 63)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    chat_id             INTEGER NOT NULL,
    message_id          INTEGER NOT NULL,
    date_utc            INTEGER NOT NULL,
    received_at_utc     INTEGER NOT NULL,
    text                TEXT,
    sender_id           INTEGER,
    topic_id            INTEGER,
    topic_title         TEXT,
    sender_username     TEXT,
    sender_first_name   TEXT,
    sender_last_name    TEXT,
    sender_display_name TEXT,
    message_url         TEXT,
    is_protected        INTEGER,
    sender_kind         TEXT,
    PRIMARY KEY (chat_id, message_id)
)
"""

# Columns added after the initial schema. Existing databases are migrated with
# ALTER TABLE ADD COLUMN (no data loss, primary key unchanged).
_MIGRATION_COLUMNS = (
    ("topic_id", "INTEGER"),
    ("topic_title", "TEXT"),
    ("sender_username", "TEXT"),
    ("sender_first_name", "TEXT"),
    ("sender_last_name", "TEXT"),
    ("sender_display_name", "TEXT"),
    ("message_url", "TEXT"),
    ("is_protected", "INTEGER"),
    ("sender_kind", "TEXT"),
)

_INSERT = """
INSERT INTO messages (
    chat_id, message_id, date_utc, received_at_utc, text, sender_id,
    topic_id, topic_title, sender_username, sender_first_name, sender_last_name,
    sender_display_name, message_url, is_protected, sender_kind
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(chat_id, message_id) DO NOTHING
"""

# On a repeated message, only fill metadata that is currently NULL; never
# overwrite an existing (non-NULL) value. Text, timestamps and ids are untouched.
_FILL_NULLS = """
UPDATE messages SET
    topic_id            = COALESCE(topic_id, ?),
    topic_title         = COALESCE(topic_title, ?),
    sender_username     = COALESCE(sender_username, ?),
    sender_first_name   = COALESCE(sender_first_name, ?),
    sender_last_name    = COALESCE(sender_last_name, ?),
    sender_display_name = COALESCE(sender_display_name, ?),
    message_url         = COALESCE(message_url, ?),
    is_protected        = COALESCE(is_protected, ?),
    sender_kind         = COALESCE(sender_kind, ?)
WHERE chat_id = ? AND message_id = ?
"""

_SELECT_ONE = """
SELECT chat_id, message_id, date_utc, received_at_utc, text, sender_id
FROM messages
WHERE chat_id = ? AND message_id = ?
"""

_SELECT_ALL = """
SELECT chat_id, message_id, date_utc, received_at_utc, text, sender_id
FROM messages
ORDER BY chat_id, message_id
"""

_METADATA_COLUMNS = (
    "topic_id",
    "topic_title",
    "sender_username",
    "sender_first_name",
    "sender_last_name",
    "sender_display_name",
    "message_url",
    "is_protected",
    "sender_kind",
)

_SELECT_METADATA = """
SELECT topic_id, topic_title, sender_username, sender_first_name,
       sender_last_name, sender_display_name, message_url, is_protected,
       sender_kind
FROM messages
WHERE chat_id = ? AND message_id = ?
"""

# Classifier results live in their own table so the source message/text is never
# duplicated or modified. One row per (message, classifier_version): a new
# version is written alongside older ones, and re-processing a version is
# idempotent.
_CLASSIFICATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS message_classifications (
    chat_id            INTEGER NOT NULL,
    message_id         INTEGER NOT NULL,
    classifier_version TEXT    NOT NULL,
    category           TEXT    NOT NULL CHECK (category IN
        ('referral_request','job_interest','referral_offer',
         'recruitment_offer','irrelevant','uncertain')),
    score              REAL    NOT NULL DEFAULT 0,
    matched_rules      TEXT    NOT NULL DEFAULT '[]',
    classified_at_utc  INTEGER NOT NULL,
    PRIMARY KEY (chat_id, message_id, classifier_version),
    FOREIGN KEY (chat_id, message_id) REFERENCES messages(chat_id, message_id)
)
"""

_CLASSIFICATION_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_class_version_category "
    "ON message_classifications(classifier_version, category)"
)

_INSERT_CLASSIFICATION = """
INSERT INTO message_classifications (
    chat_id, message_id, classifier_version, category, score,
    matched_rules, classified_at_utc
)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(chat_id, message_id, classifier_version) DO NOTHING
"""

_SELECT_UNCLASSIFIED = """
SELECT m.chat_id, m.message_id, m.text
FROM messages m
LEFT JOIN message_classifications c
    ON c.chat_id = m.chat_id
   AND c.message_id = m.message_id
   AND c.classifier_version = ?
WHERE m.text IS NOT NULL
  AND c.chat_id IS NULL
  AND (m.chat_id, m.message_id) > (?, ?)
ORDER BY m.chat_id, m.message_id
LIMIT ?
"""

_COUNT_UNCLASSIFIED = """
SELECT COUNT(*)
FROM messages m
LEFT JOIN message_classifications c
    ON c.chat_id = m.chat_id
   AND c.message_id = m.message_id
   AND c.classifier_version = ?
WHERE m.text IS NOT NULL AND c.chat_id IS NULL
"""

_SELECT_CANDIDATES = """
SELECT m.chat_id, m.message_id, m.date_utc, m.sender_id, c.category, c.score
FROM message_classifications c
JOIN messages m
    ON m.chat_id = c.chat_id AND m.message_id = c.message_id
WHERE c.classifier_version = ?
  AND c.category IN ({placeholders})
ORDER BY m.date_utc, m.chat_id, m.message_id
LIMIT ?
"""

# Per-chat history reconciliation checkpoint. ``last_message_id`` is the newest
# message ID whose entire history range has been fully examined (and its text
# messages persisted or deliberately skipped). It is separate from the
# ``messages`` table on purpose: ``MAX(message_id)`` cannot be used as a
# checkpoint because it would hide gaps in the stored history.
_SYNC_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_sync_state (
    chat_id         INTEGER PRIMARY KEY,
    last_message_id INTEGER NOT NULL,
    updated_at_utc  INTEGER NOT NULL
)
"""

_SELECT_CHECKPOINT = "SELECT last_message_id FROM chat_sync_state WHERE chat_id = ?"

_UPSERT_CHECKPOINT = """
INSERT INTO chat_sync_state (chat_id, last_message_id, updated_at_utc)
VALUES (?, ?, ?)
ON CONFLICT(chat_id) DO UPDATE SET
    last_message_id = excluded.last_message_id,
    updated_at_utc = excluded.updated_at_utc
"""


class MessageStore:
    """Thin SQLite wrapper around the ``messages`` table.

    The store opens its own connection and is intended to be used by a single
    writer at a time. Call :meth:`close` (or use it as a context manager) to
    release the connection.
    """

    def __init__(self, db_path):
        self._db_path = Path(db_path)
        parent = self._db_path.parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute(_SCHEMA)
        # Migrate the message table (add missing columns) and create the
        # checkpoint table on existing databases too (no data loss).
        self._migrate_messages()
        self._conn.execute(_SYNC_SCHEMA)
        self._conn.execute(_CLASSIFICATION_SCHEMA)
        self._conn.execute(_CLASSIFICATION_INDEX)
        self._conn.commit()

    def _migrate_messages(self) -> None:
        existing = {
            row[1] for row in self._conn.execute("PRAGMA table_info(messages)")
        }
        for name, declaration in _MIGRATION_COLUMNS:
            if name not in existing:
                self._conn.execute(
                    f"ALTER TABLE messages ADD COLUMN {name} {declaration}"
                )

    @property
    def db_path(self) -> Path:
        return self._db_path

    @staticmethod
    def _row(message: IncomingMessage) -> tuple:
        return (
            message.chat_id,
            message.message_id,
            message.date_utc,
            message.received_at_utc,
            message.text,
            message.sender_id,
            message.topic_id,
            message.topic_title,
            message.sender_username,
            message.sender_first_name,
            message.sender_last_name,
            message.sender_display_name,
            message.message_url,
            message.is_protected,
            message.sender_kind,
        )

    @staticmethod
    def _metadata_values(row: tuple) -> tuple:
        """The metadata columns (indices 6..14) of an insert row."""
        return row[6:15]

    def save(self, message: IncomingMessage) -> bool:
        """Insert ``message``, returning ``True`` if a new row was written.

        A message whose ``(chat_id, message_id)`` already exists is not
        duplicated: ``False`` is returned, and only currently-NULL metadata
        (topic, sender, URL, protection flag) may be filled in — existing text,
        timestamps and ids are never overwritten. A write failure raises
        :class:`sqlite3.Error`.
        """
        row = self._row(message)
        cursor = self._conn.execute(_INSERT, row)
        self._conn.commit()
        inserted = cursor.rowcount > 0
        if not inserted:
            values = self._metadata_values(row)
            if any(value is not None for value in values):
                self._conn.execute(_FILL_NULLS, (*values, row[0], row[1]))
                self._conn.commit()
        return inserted

    def save_many(self, messages: Iterable[IncomingMessage]) -> Tuple[int, int]:
        """Insert many messages in a single transaction.

        Returns ``(inserted, duplicates)``. Used for bulk history
        reconciliation to avoid one commit per message. Deduplication is the
        same ``ON CONFLICT(chat_id, message_id) DO NOTHING`` rule; for duplicate
        keys only missing metadata is filled in (never overwritten).
        """
        rows = [self._row(message) for message in messages]
        if not rows:
            return 0, 0
        before = self._conn.total_changes
        self._conn.executemany(_INSERT, rows)
        self._conn.commit()
        inserted = int(self._conn.total_changes - before)

        fill_rows = [
            (*self._metadata_values(row), row[0], row[1])
            for row in rows
            if any(value is not None for value in self._metadata_values(row))
        ]
        if fill_rows:
            self._conn.executemany(_FILL_NULLS, fill_rows)
            self._conn.commit()
        return inserted, len(rows) - inserted

    def count(self) -> int:
        """Return the number of stored messages."""
        return int(self._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])

    def fetch(self, chat_id: int, message_id: int) -> Optional[Row]:
        """Return a single stored row or ``None``."""
        return self._conn.execute(_SELECT_ONE, (chat_id, message_id)).fetchone()

    def fetch_all(self) -> list:
        """Return all stored rows ordered by ``(chat_id, message_id)``."""
        return self._conn.execute(_SELECT_ALL).fetchall()

    def fetch_metadata(self, chat_id: int, message_id: int) -> Optional[dict]:
        """Return the topic/sender/url/protection metadata for a stored message.

        Returns ``None`` when the message is not present. Missing metadata is
        reported as ``None`` (for example for messages stored before these
        columns existed and never re-processed).
        """
        row = self._conn.execute(
            _SELECT_METADATA, (int(chat_id), int(message_id))
        ).fetchone()
        if row is None:
            return None
        return dict(zip(_METADATA_COLUMNS, row))

    def get_checkpoint(self, chat_id: int) -> Optional[int]:
        """Return the last fully reconciled message ID for ``chat_id``.

        ``None`` means the chat has never been history-synced, which triggers
        the bounded initial reconciliation.
        """
        row = self._conn.execute(_SELECT_CHECKPOINT, (int(chat_id),)).fetchone()
        return int(row[0]) if row is not None else None

    def set_checkpoint(
        self, chat_id: int, last_message_id: int, updated_at_utc: int
    ) -> None:
        """Store (or replace) the reconciliation checkpoint for ``chat_id``.

        Only call this after the range up to ``last_message_id`` has been fully
        processed, so the checkpoint never skips unprocessed messages.
        """
        self._conn.execute(
            _UPSERT_CHECKPOINT,
            (int(chat_id), int(last_message_id), int(updated_at_utc)),
        )
        self._conn.commit()

    # --- classifier results ------------------------------------------------

    def save_classifications(self, rows) -> int:
        """Persist classifier results (idempotent per ``classifier_version``).

        Each row is ``(chat_id, message_id, classifier_version, category, score,
        matched_rules_json, classified_at_utc)``. Returns the number of new rows
        written; existing (chat_id, message_id, version) rows are left as is.
        """
        rows = list(rows)
        if not rows:
            return 0
        before = self._conn.total_changes
        self._conn.executemany(_INSERT_CLASSIFICATION, rows)
        self._conn.commit()
        return int(self._conn.total_changes - before)

    def get_classification(self, chat_id, message_id, version):
        """Return a stored classification as a dict, or ``None``."""
        row = self._conn.execute(
            "SELECT category, score, matched_rules, classified_at_utc "
            "FROM message_classifications "
            "WHERE chat_id = ? AND message_id = ? AND classifier_version = ?",
            (int(chat_id), int(message_id), version),
        ).fetchone()
        if row is None:
            return None
        return {
            "category": row[0],
            "score": float(row[1]),
            "matched_rules": row[2],
            "classified_at_utc": int(row[3]),
        }

    def count_unclassified(self, version) -> int:
        """Number of text messages that have no result for ``version``."""
        return int(
            self._conn.execute(_COUNT_UNCLASSIFIED, (version,)).fetchone()[0]
        )

    def iter_unclassified(self, version, batch_size: int = 500):
        """Yield batches of ``(chat_id, message_id, text)`` needing ``version``.

        Keyset-paginated by ``(chat_id, message_id)`` so classification can be
        committed per batch and resumed after interruption without re-reading
        already-classified rows.
        """
        last_chat = _INT64_MIN
        last_message = _INT64_MIN
        while True:
            rows = self._conn.execute(
                _SELECT_UNCLASSIFIED,
                (version, last_chat, last_message, int(batch_size)),
            ).fetchall()
            if not rows:
                return
            yield [(int(r[0]), int(r[1]), r[2]) for r in rows]
            last_chat = int(rows[-1][0])
            last_message = int(rows[-1][1])
            if len(rows) < batch_size:
                return

    def iter_candidates(self, version, categories=None, limit=None):
        """Yield candidate messages of ``version`` for the given categories.

        Defaults to the candidate categories (``referral_request``,
        ``job_interest``). Yields ``(chat_id, message_id, date_utc, sender_id,
        category, score)``. The version is passed explicitly (no SQL view with a
        dangling parameter).
        """
        cats = tuple(categories) if categories is not None else _CANDIDATE_CATEGORIES
        if not cats:
            return
        placeholders = ",".join("?" for _ in cats)
        sql = _SELECT_CANDIDATES.format(placeholders=placeholders)
        params = [version, *cats, -1 if limit is None else int(limit)]
        for row in self._conn.execute(sql, params):
            yield (int(row[0]), int(row[1]), int(row[2]), row[3], row[4], float(row[5]))

    def close(self) -> None:
        """Close the connection. Safe to call more than once."""
        if self._closed:
            return
        self._conn.close()
        self._closed = True

    def __enter__(self) -> "MessageStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
