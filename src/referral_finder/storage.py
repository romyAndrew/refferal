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
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

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

# AI (LLM) analysis results, stored separately from the source message. One row
# per (message, provider, model, prompt_version): a new model/prompt revision is
# written alongside older ones, and re-running a revision is idempotent. The
# source `messages` table is never modified by this feature.
_AI_SCHEMA = """
CREATE TABLE IF NOT EXISTS message_ai_analysis (
    chat_id            INTEGER NOT NULL,
    message_id         INTEGER NOT NULL,
    provider           TEXT    NOT NULL,
    model              TEXT    NOT NULL,
    prompt_version     TEXT    NOT NULL,
    prompt_hash        TEXT    NOT NULL,
    status             TEXT    NOT NULL,
    relevant           INTEGER,
    category           TEXT,
    confidence         REAL,
    explanation        TEXT,
    error              TEXT,
    analyzed_at_utc    INTEGER NOT NULL,
    attempts           INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (chat_id, message_id, provider, model, prompt_version),
    FOREIGN KEY (chat_id, message_id) REFERENCES messages(chat_id, message_id)
)
"""

# Columns added after the initial AI schema (existing DBs are migrated in place
# with ALTER TABLE ADD COLUMN; no data loss, primary key unchanged).
_AI_MIGRATION_COLUMNS = (("attempts", "INTEGER NOT NULL DEFAULT 0"),)

# Retry policy default: a temporary failure is retried until this many attempts
# have been recorded for the same analysis version, then it is left alone.
DEFAULT_MAX_ATTEMPTS = 3

_AI_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_ai_model_category "
    "ON message_ai_analysis(provider, model, prompt_version, category)"
)

# Prefilter decisions, stored separately from the AI result and from the source
# message. Keyed by (chat_id, message_id, prefilter_version) so bumping
# PREFILTER_VERSION re-checks every message while a decision stays idempotent
# for a given version. ``matched=0`` lets the AI selection skip rejected
# messages instead of re-reading them every cycle (no starvation).
_PREFILTER_SCHEMA = """
CREATE TABLE IF NOT EXISTS message_prefilter (
    chat_id            INTEGER NOT NULL,
    message_id         INTEGER NOT NULL,
    prefilter_version  TEXT    NOT NULL,
    matched            INTEGER NOT NULL,
    matched_rules      TEXT    NOT NULL DEFAULT '[]',
    checked_at_utc     INTEGER NOT NULL,
    PRIMARY KEY (chat_id, message_id, prefilter_version),
    FOREIGN KEY (chat_id, message_id) REFERENCES messages(chat_id, message_id)
)
"""

_PREFILTER_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_prefilter_version_matched "
    "ON message_prefilter(prefilter_version, matched)"
)

_INSERT_PREFILTER = """
INSERT INTO message_prefilter (
    chat_id, message_id, prefilter_version, matched, matched_rules, checked_at_utc
)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(chat_id, message_id, prefilter_version) DO NOTHING
"""

_SELECT_PREFILTER = """
SELECT prefilter_version, matched, matched_rules, checked_at_utc
FROM message_prefilter
WHERE chat_id = ? AND message_id = ? AND prefilter_version = ?
"""

_INSERT_AI = """
INSERT INTO message_ai_analysis (
    chat_id, message_id, provider, model, prompt_version, prompt_hash,
    status, relevant, category, confidence, explanation, error, analyzed_at_utc
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(chat_id, message_id, provider, model, prompt_version) DO NOTHING
"""

# Upsert used by the analysis service: records an attempt (attempts+1 on retry)
# and updates the outcome, but never overwrites an already-successful row. This
# keeps a successful classification idempotent while allowing a temporary
# failure to be retried (bounded by DEFAULT_MAX_ATTEMPTS in the selection query).
_AI_UPSERT = """
INSERT INTO message_ai_analysis (
    chat_id, message_id, provider, model, prompt_version, prompt_hash,
    status, relevant, category, confidence, explanation, error, analyzed_at_utc,
    attempts
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
ON CONFLICT(chat_id, message_id, provider, model, prompt_version) DO UPDATE SET
    status = excluded.status,
    relevant = excluded.relevant,
    category = excluded.category,
    confidence = excluded.confidence,
    explanation = excluded.explanation,
    error = excluded.error,
    analyzed_at_utc = excluded.analyzed_at_utc,
    attempts = message_ai_analysis.attempts + 1
WHERE message_ai_analysis.status != 'ok'
"""

_SELECT_AI = """
SELECT status, relevant, category, confidence, explanation, error,
       prompt_hash, analyzed_at_utc, attempts
FROM message_ai_analysis
WHERE chat_id = ? AND message_id = ? AND provider = ? AND model = ?
  AND prompt_version = ?
"""

_SELECT_AI_UNPROCESSED = """
SELECT m.chat_id, m.message_id, m.text
FROM messages m
LEFT JOIN message_ai_analysis a
    ON a.chat_id = m.chat_id
   AND a.message_id = m.message_id
   AND a.provider = ?
   AND a.model = ?
   AND a.prompt_version = ?
LEFT JOIN message_prefilter p
    ON p.chat_id = m.chat_id
   AND p.message_id = m.message_id
   AND p.prefilter_version = ?
WHERE m.text IS NOT NULL
  AND (a.chat_id IS NULL OR (a.status != 'ok' AND a.attempts < ?))
  AND (p.chat_id IS NULL OR p.matched = 1)
  AND (m.chat_id, m.message_id) > (?, ?)
ORDER BY m.chat_id, m.message_id
LIMIT ?
"""

_COUNT_AI_UNPROCESSED = """
SELECT COUNT(*)
FROM messages m
LEFT JOIN message_ai_analysis a
    ON a.chat_id = m.chat_id
   AND a.message_id = m.message_id
   AND a.provider = ?
   AND a.model = ?
   AND a.prompt_version = ?
LEFT JOIN message_prefilter p
    ON p.chat_id = m.chat_id
   AND p.message_id = m.message_id
   AND p.prefilter_version = ?
WHERE m.text IS NOT NULL
  AND (a.chat_id IS NULL OR (a.status != 'ok' AND a.attempts < ?))
  AND (p.chat_id IS NULL OR p.matched = 1)
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

# --- bootstrap state (first-seen chats) ----------------------------------
# One row per configured chat: the fixed initial snapshot (up to the bootstrap
# limit) and the per-chat steady high-water mark. Messages with
# ``message_id > steady_after_message_id`` are the steady stream; older messages
# (including history beyond the snapshot window) are never auto-analysed.
# ``steady_after_message_id`` is NOT NULL and is 0 for a chat that has no
# messages yet, so any future message (id >= 1) is correctly treated as steady
# instead of being excluded by a NULL comparison.
_BOOTSTRAP_SCHEMA = """
CREATE TABLE IF NOT EXISTS ai_bootstrap_state (
    chat_id                 INTEGER PRIMARY KEY,
    status                  TEXT    NOT NULL,
    available_count         INTEGER NOT NULL,
    selected_count          INTEGER NOT NULL,
    steady_after_message_id INTEGER NOT NULL,
    created_at_utc          INTEGER NOT NULL,
    updated_at_utc          INTEGER NOT NULL
)
"""

# The fixed snapshot membership: exactly the messages chosen at snapshot time,
# so a restart reuses the same selection instead of re-picking a different one.
_BOOTSTRAP_ITEM_SCHEMA = """
CREATE TABLE IF NOT EXISTS ai_bootstrap_item (
    chat_id      INTEGER NOT NULL,
    message_id   INTEGER NOT NULL,
    added_at_utc INTEGER NOT NULL,
    PRIMARY KEY (chat_id, message_id),
    FOREIGN KEY (chat_id, message_id) REFERENCES messages(chat_id, message_id)
)
"""

_BOOTSTRAP_ITEM_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_bootstrap_item_chat "
    "ON ai_bootstrap_item(chat_id, message_id)"
)

# --- daily API request budget --------------------------------------------
# One row per UTC day; ``used`` counts every outgoing HTTP attempt (including
# retries). Reserved atomically with a single upsert so concurrent callers can
# never exceed ``limit``.
_API_BUDGET_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_request_budget (
    day_utc        TEXT    PRIMARY KEY,
    used           INTEGER NOT NULL,
    updated_at_utc INTEGER NOT NULL
)
"""

# Atomic reserve: the INSERT seeds a new day at 1; the DO UPDATE only fires while
# used < limit. ``rowcount`` is 0 when the limit is exhausted (no row changed),
# which the caller interprets as "denied".
_RESERVE_API_REQUEST = """
INSERT INTO api_request_budget (day_utc, used, updated_at_utc)
VALUES (?, 1, ?)
ON CONFLICT(day_utc) DO UPDATE SET
    used = used + 1,
    updated_at_utc = excluded.updated_at_utc
WHERE used < ?
"""

_SELECT_API_USED = "SELECT used FROM api_request_budget WHERE day_utc = ?"

_INSERT_BOOTSTRAP_STATE = """
INSERT INTO ai_bootstrap_state (
    chat_id, status, available_count, selected_count, steady_after_message_id,
    created_at_utc, updated_at_utc
)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(chat_id) DO NOTHING
"""

_SELECT_BOOTSTRAP_STATE = """
SELECT chat_id, status, available_count, selected_count,
       steady_after_message_id, created_at_utc, updated_at_utc
FROM ai_bootstrap_state
WHERE chat_id = ?
"""

_SELECT_BOOTSTRAP_CHAT_IDS = "SELECT chat_id FROM ai_bootstrap_state ORDER BY chat_id"

_UPDATE_BOOTSTRAP_STATUS = (
    "UPDATE ai_bootstrap_state SET status = ?, updated_at_utc = ? WHERE chat_id = ?"
)

# COALESCE keeps the watermark a number (0) for an empty chat; a NULL would make
# ``message_id > NULL`` false and silently drop every future message.
_SELECT_MAX_MESSAGE_ID = "SELECT COALESCE(MAX(message_id), 0) FROM messages WHERE chat_id = ?"

_COUNT_TEXT_MESSAGES = (
    "SELECT COUNT(*) FROM messages WHERE chat_id = ? AND text IS NOT NULL"
)

_SELECT_LAST_TEXT_MESSAGES = """
SELECT message_id FROM messages
WHERE chat_id = ? AND text IS NOT NULL
ORDER BY message_id DESC
LIMIT ?
"""

_INSERT_BOOTSTRAP_ITEM = """
INSERT INTO ai_bootstrap_item (chat_id, message_id, added_at_utc)
VALUES (?, ?, ?)
ON CONFLICT(chat_id, message_id) DO NOTHING
"""

# Bootstrap selection: the fixed snapshot items, no prefilter join (bootstrap
# bypasses prefilter); AI success/attempts still bound repeats.
_SELECT_AI_BOOTSTRAP = """
SELECT m.chat_id, m.message_id, m.text
FROM ai_bootstrap_item b
JOIN messages m
    ON m.chat_id = b.chat_id AND m.message_id = b.message_id
LEFT JOIN message_ai_analysis a
    ON a.chat_id = m.chat_id
   AND a.message_id = m.message_id
   AND a.provider = ?
   AND a.model = ?
   AND a.prompt_version = ?
WHERE m.text IS NOT NULL
  AND (a.chat_id IS NULL OR (a.status != 'ok' AND a.attempts < ?))
  AND (m.chat_id, m.message_id) > (?, ?)
ORDER BY m.chat_id, m.message_id
LIMIT ?
"""

_COUNT_AI_BOOTSTRAP = """
SELECT COUNT(*)
FROM ai_bootstrap_item b
JOIN messages m
    ON m.chat_id = b.chat_id AND m.message_id = b.message_id
LEFT JOIN message_ai_analysis a
    ON a.chat_id = m.chat_id
   AND a.message_id = m.message_id
   AND a.provider = ?
   AND a.model = ?
   AND a.prompt_version = ?
WHERE m.text IS NOT NULL
  AND (a.chat_id IS NULL OR (a.status != 'ok' AND a.attempts < ?))
"""

# Steady selection: only messages strictly newer than the per-chat high-water
# mark. Old history (including beyond the bootstrap window) is never selected.
_SELECT_AI_UNPROCESSED_STEADY = """
SELECT m.chat_id, m.message_id, m.text
FROM messages m
JOIN ai_bootstrap_state s
    ON s.chat_id = m.chat_id
LEFT JOIN message_ai_analysis a
    ON a.chat_id = m.chat_id
   AND a.message_id = m.message_id
   AND a.provider = ?
   AND a.model = ?
   AND a.prompt_version = ?
LEFT JOIN message_prefilter p
    ON p.chat_id = m.chat_id
   AND p.message_id = m.message_id
   AND p.prefilter_version = ?
WHERE m.text IS NOT NULL
  AND m.message_id > s.steady_after_message_id
  AND (a.chat_id IS NULL OR (a.status != 'ok' AND a.attempts < ?))
  AND (p.chat_id IS NULL OR p.matched = 1)
  AND (m.chat_id, m.message_id) > (?, ?)
ORDER BY m.chat_id, m.message_id
LIMIT ?
"""

_COUNT_AI_UNPROCESSED_STEADY = """
SELECT COUNT(*)
FROM messages m
JOIN ai_bootstrap_state s
    ON s.chat_id = m.chat_id
LEFT JOIN message_ai_analysis a
    ON a.chat_id = m.chat_id
   AND a.message_id = m.message_id
   AND a.provider = ?
   AND a.model = ?
   AND a.prompt_version = ?
LEFT JOIN message_prefilter p
    ON p.chat_id = m.chat_id
   AND p.message_id = m.message_id
   AND p.prefilter_version = ?
WHERE m.text IS NOT NULL
  AND m.message_id > s.steady_after_message_id
  AND (a.chat_id IS NULL OR (a.status != 'ok' AND a.attempts < ?))
  AND (p.chat_id IS NULL OR p.matched = 1)
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
        self._conn.execute(_AI_SCHEMA)
        self._conn.execute(_AI_INDEX)
        self._migrate_ai_analysis()
        self._conn.execute(_PREFILTER_SCHEMA)
        self._conn.execute(_PREFILTER_INDEX)
        self._conn.execute(_BOOTSTRAP_SCHEMA)
        self._conn.execute(_BOOTSTRAP_ITEM_SCHEMA)
        self._conn.execute(_BOOTSTRAP_ITEM_INDEX)
        self._conn.execute(_API_BUDGET_SCHEMA)
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

    def _migrate_ai_analysis(self) -> None:
        existing = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(message_ai_analysis)")
        }
        for name, declaration in _AI_MIGRATION_COLUMNS:
            if name not in existing:
                self._conn.execute(
                    f"ALTER TABLE message_ai_analysis ADD COLUMN {name} {declaration}"
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

    # --- AI (LLM) analysis results ----------------------------------------

    def save_ai_analysis(self, rows) -> int:
        """Persist AI-analysis rows, idempotent per (message, model, prompt).

        Each row is ``(chat_id, message_id, provider, model, prompt_version,
        prompt_hash, status, relevant, category, confidence, explanation, error,
        analyzed_at_utc)`` (see :func:`referral_finder.llm.schema.ai_analysis_row`).
        Returns the number of new rows written; an existing
        ``(chat_id, message_id, provider, model, prompt_version)`` row is left
        unchanged.
        """
        rows = list(rows)
        if not rows:
            return 0
        before = self._conn.total_changes
        self._conn.executemany(_INSERT_AI, rows)
        self._conn.commit()
        return int(self._conn.total_changes - before)

    def upsert_ai_analysis(self, rows) -> int:
        """Record analysis attempts and outcomes; keep a success idempotent.

        Each row has the same shape as :meth:`save_ai_analysis`. On conflict the
        row is updated and its ``attempts`` counter is incremented, **unless** an
        existing row is already successful (``status='ok'``), which is never
        overwritten. Returns the number of inserted or updated rows.
        """
        rows = list(rows)
        if not rows:
            return 0
        before = self._conn.total_changes
        self._conn.executemany(_AI_UPSERT, rows)
        self._conn.commit()
        return int(self._conn.total_changes - before)

    def get_ai_analysis(
        self, chat_id, message_id, provider, model, prompt_version
    ) -> Optional[dict]:
        """Return a stored AI-analysis dict, or ``None``."""
        row = self._conn.execute(
            _SELECT_AI,
            (
                int(chat_id),
                int(message_id),
                str(provider),
                str(model),
                str(prompt_version),
            ),
        ).fetchone()
        if row is None:
            return None
        return {
            "status": row[0],
            "relevant": None if row[1] is None else bool(row[1]),
            "category": row[2],
            "confidence": None if row[3] is None else float(row[3]),
            "explanation": row[4],
            "error": row[5],
            "prompt_hash": row[6],
            "analyzed_at_utc": int(row[7]),
            "attempts": int(row[8]),
        }

    # --- bootstrap state (first-seen chats) --------------------------------

    def snapshot_bootstrap(
        self, chat_ids: Iterable[int], limit: int = 200, clock: Optional[Callable[[], float]] = None
    ) -> list:
        """Fix the initial snapshot for chats that have no bootstrap state yet.

        For each chat, records the per-chat ``steady_after_message_id`` (the
        maximum id of every message already stored) and stores the most recent
        up-to-``limit`` text messages as fixed bootstrap items. Idempotent: a
        chat that already has state is left untouched, so a restart never
        re-picks a different snapshot. Chats without messages still get a state
        row with zero counts.

        Returns a list of summary dicts (one per chat).
        """
        now = int((clock or time.time)())
        summaries = []
        for chat_id in chat_ids:
            chat_id = int(chat_id)
            existing = self._conn.execute(_SELECT_BOOTSTRAP_STATE, (chat_id,)).fetchone()
            if existing is not None:
                summaries.append(self._bootstrap_state_dict(existing, created=False))
                continue

            available = int(
                self._conn.execute(_COUNT_TEXT_MESSAGES, (chat_id,)).fetchone()[0]
            )
            steady_after = int(
                self._conn.execute(_SELECT_MAX_MESSAGE_ID, (chat_id,)).fetchone()[0]
            )
            message_ids = [
                int(row[0])
                for row in self._conn.execute(
                    _SELECT_LAST_TEXT_MESSAGES, (chat_id, int(limit))
                ).fetchall()
            ]
            self._conn.execute(
                _INSERT_BOOTSTRAP_STATE,
                (chat_id, "pending", available, len(message_ids), steady_after, now, now),
            )
            self._conn.executemany(
                _INSERT_BOOTSTRAP_ITEM,
                [(chat_id, message_id, now) for message_id in message_ids],
            )
            self._conn.commit()
            summaries.append(
                {
                    "chat_id": chat_id,
                    "status": "pending",
                    "available_count": available,
                    "selected_count": len(message_ids),
                    "steady_after_message_id": steady_after,
                    "created_at_utc": now,
                    "updated_at_utc": now,
                    "created": True,
                }
            )
        return summaries

    @staticmethod
    def _bootstrap_state_dict(row, *, created: bool = False) -> dict:
        return {
            "chat_id": int(row[0]),
            "status": row[1],
            "available_count": int(row[2]),
            "selected_count": int(row[3]),
            "steady_after_message_id": int(row[4]),
            "created_at_utc": int(row[5]),
            "updated_at_utc": int(row[6]),
            "created": created,
        }

    def get_bootstrap_state(self, chat_id: int) -> Optional[dict]:
        """Return the bootstrap state for ``chat_id``, or ``None``."""
        row = self._conn.execute(
            _SELECT_BOOTSTRAP_STATE, (int(chat_id),)
        ).fetchone()
        if row is None:
            return None
        return self._bootstrap_state_dict(row, created=False)

    def bootstrap_chat_ids(self) -> list:
        """Return chat ids that already have a bootstrap state."""
        return [int(row[0]) for row in self._conn.execute(_SELECT_BOOTSTRAP_CHAT_IDS)]

    def set_bootstrap_status(
        self, chat_id: int, status: str, clock: Optional[Callable[[], float]] = None
    ) -> None:
        """Update the bootstrap status for ``chat_id`` (pending/running/done)."""
        now = int((clock or time.time)())
        self._conn.execute(_UPDATE_BOOTSTRAP_STATUS, (str(status), now, int(chat_id)))
        self._conn.commit()

    def iter_ai_bootstrap(
        self,
        provider,
        model,
        prompt_version,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        batch_size: int = 500,
    ):
        """Yield batches of fixed bootstrap ``(chat_id, message_id, text)``.

        No prefilter join: the bootstrap set bypasses prefilter by design. A
        successful AI result is skipped; failures are retried up to
        ``max_attempts``. Keyset-paginated by ``(chat_id, message_id)``.
        """
        last_chat = _INT64_MIN
        last_message = _INT64_MIN
        while True:
            rows = self._conn.execute(
                _SELECT_AI_BOOTSTRAP,
                (
                    str(provider),
                    str(model),
                    str(prompt_version),
                    int(max_attempts),
                    last_chat,
                    last_message,
                    int(batch_size),
                ),
            ).fetchall()
            if not rows:
                return
            yield [(int(r[0]), int(r[1]), r[2]) for r in rows]
            last_chat = int(rows[-1][0])
            last_message = int(rows[-1][1])
            if len(rows) < batch_size:
                return

    def count_ai_bootstrap(
        self,
        provider,
        model,
        prompt_version,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> int:
        """Number of bootstrap items still needing analysis for this version."""
        return int(
            self._conn.execute(
                _COUNT_AI_BOOTSTRAP,
                (str(provider), str(model), str(prompt_version), int(max_attempts)),
            ).fetchone()[0]
        )

    # --- daily API request budget ------------------------------------------

    def reserve_api_request(
        self, day_utc: str, limit: int, updated_at_utc: int
    ) -> bool:
        """Atomically reserve one outgoing HTTP request for ``day_utc``.

        Returns ``True`` when the request is allowed (and ``used`` was
        incremented), ``False`` when the daily limit is already reached. The
        single upsert is atomic, so concurrent callers/connections cannot jointly
        exceed ``limit``.
        """
        if int(limit) < 1:
            raise ValueError("limit must be at least 1")
        cursor = self._conn.execute(
            _RESERVE_API_REQUEST, (str(day_utc), int(updated_at_utc), int(limit))
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def api_requests_used(self, day_utc: str) -> int:
        """Number of requests already consumed for ``day_utc`` (0 if unseen)."""
        row = self._conn.execute(_SELECT_API_USED, (str(day_utc),)).fetchone()
        return int(row[0]) if row is not None else 0

    def save_prefilter_results(self, rows) -> int:
        """Persist prefilter decisions (idempotent per prefilter version).

        Each row is ``(chat_id, message_id, prefilter_version, matched,
        matched_rules_json, checked_at_utc)``. Returns the number of new rows.
        An existing ``(chat_id, message_id, prefilter_version)`` row is kept, so
        a decision is stable for a version; bumping the version re-checks.
        """
        rows = list(rows)
        if not rows:
            return 0
        before = self._conn.total_changes
        self._conn.executemany(_INSERT_PREFILTER, rows)
        self._conn.commit()
        return int(self._conn.total_changes - before)

    def get_prefilter_result(
        self, chat_id, message_id, prefilter_version
    ) -> Optional[dict]:
        """Return a stored prefilter decision, or ``None``."""
        row = self._conn.execute(
            _SELECT_PREFILTER,
            (int(chat_id), int(message_id), str(prefilter_version)),
        ).fetchone()
        if row is None:
            return None
        return {
            "prefilter_version": row[0],
            "matched": bool(row[1]),
            "matched_rules": row[2],
            "checked_at_utc": int(row[3]),
        }

    def count_ai_unprocessed(
        self,
        provider,
        model,
        prompt_version,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        prefilter_version: str = "",
        steady_only: bool = False,
    ) -> int:
        """Number of messages still needing AI analysis for this version.

        A message counts when it has no AI row yet, or a failed row whose
        ``attempts`` is still below ``max_attempts`` (temporary errors stay
        retryable; successful rows never count). Messages recorded as rejected
        by the given ``prefilter_version`` never count. When
        ``prefilter_version`` is empty no prefilter filtering is applied.

        With ``steady_only=True`` only messages strictly newer than each chat's
        bootstrap high-water mark are counted (no old history, no bootstrap
        snapshot); chats without a bootstrap state are excluded entirely.
        """
        if steady_only:
            return int(
                self._conn.execute(
                    _COUNT_AI_UNPROCESSED_STEADY,
                    (
                        str(provider),
                        str(model),
                        str(prompt_version),
                        str(prefilter_version),
                        int(max_attempts),
                    ),
                ).fetchone()[0]
            )
        return int(
            self._conn.execute(
                _COUNT_AI_UNPROCESSED,
                (
                    str(provider),
                    str(model),
                    str(prompt_version),
                    str(prefilter_version),
                    int(max_attempts),
                ),
            ).fetchone()[0]
        )

    def iter_ai_unprocessed(
        self,
        provider,
        model,
        prompt_version,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        batch_size: int = 500,
        prefilter_version: str = "",
        steady_only: bool = False,
    ):
        """Yield batches of ``(chat_id, message_id, text)`` needing AI analysis.

        Successful rows are skipped; failed rows are retried until their
        ``attempts`` reach ``max_attempts``. Messages recorded as rejected by the
        given ``prefilter_version`` are skipped so they cannot repeatedly occupy
        a batch. Keyset-paginated by ``(chat_id, message_id)``.

        With ``steady_only=True`` only messages strictly newer than each chat's
        bootstrap high-water mark are yielded, so historical messages (including
        those beyond the bootstrap snapshot) can never enter the steady stream.
        """
        select = _SELECT_AI_UNPROCESSED_STEADY if steady_only else _SELECT_AI_UNPROCESSED
        last_chat = _INT64_MIN
        last_message = _INT64_MIN
        while True:
            rows = self._conn.execute(
                select,
                (
                    str(provider),
                    str(model),
                    str(prompt_version),
                    str(prefilter_version),
                    int(max_attempts),
                    last_chat,
                    last_message,
                    int(batch_size),
                ),
            ).fetchall()
            if not rows:
                return
            yield [(int(r[0]), int(r[1]), r[2]) for r in rows]
            last_chat = int(rows[-1][0])
            last_message = int(rows[-1][1])
            if len(rows) < batch_size:
                return

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
