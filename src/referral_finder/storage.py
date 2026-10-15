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
from typing import Optional, Tuple

from .models import IncomingMessage

# A fetched row: (chat_id, message_id, date_utc, received_at_utc, text, sender_id).
Row = Tuple[int, int, int, int, Optional[str], Optional[int]]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    chat_id         INTEGER NOT NULL,
    message_id      INTEGER NOT NULL,
    date_utc        INTEGER NOT NULL,
    received_at_utc INTEGER NOT NULL,
    text            TEXT,
    sender_id       INTEGER,
    PRIMARY KEY (chat_id, message_id)
)
"""

_INSERT = """
INSERT INTO messages (chat_id, message_id, date_utc, received_at_utc, text, sender_id)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(chat_id, message_id) DO NOTHING
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
        self._conn.commit()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def save(self, message: IncomingMessage) -> bool:
        """Insert ``message``, returning ``True`` if a new row was written.

        A message whose ``(chat_id, message_id)`` already exists is left
        untouched and ``False`` is returned. A write failure raises
        :class:`sqlite3.Error`.
        """
        cursor = self._conn.execute(
            _INSERT,
            (
                message.chat_id,
                message.message_id,
                message.date_utc,
                message.received_at_utc,
                message.text,
                message.sender_id,
            ),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of stored messages."""
        return int(self._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])

    def fetch(self, chat_id: int, message_id: int) -> Optional[Row]:
        """Return a single stored row or ``None``."""
        return self._conn.execute(_SELECT_ONE, (chat_id, message_id)).fetchone()

    def fetch_all(self) -> list:
        """Return all stored rows ordered by ``(chat_id, message_id)``."""
        return self._conn.execute(_SELECT_ALL).fetchall()

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
