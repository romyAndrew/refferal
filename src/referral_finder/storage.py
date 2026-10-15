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

from .models import IncomingMessage

# A fetched row: (chat_id, message_id, date_utc, received_at_utc, text, sender_id).
Row = Tuple[int, int, int, int, Optional[str], Optional[int]]

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
)

_INSERT = """
INSERT INTO messages (
    chat_id, message_id, date_utc, received_at_utc, text, sender_id,
    topic_id, topic_title, sender_username, sender_first_name, sender_last_name,
    sender_display_name, message_url, is_protected
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    is_protected        = COALESCE(is_protected, ?)
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
)

_SELECT_METADATA = """
SELECT topic_id, topic_title, sender_username, sender_first_name,
       sender_last_name, sender_display_name, message_url, is_protected
FROM messages
WHERE chat_id = ? AND message_id = ?
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
        )

    @staticmethod
    def _metadata_values(row: tuple) -> tuple:
        """The new metadata columns (indices 6..13) of an insert row."""
        return row[6:14]

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
