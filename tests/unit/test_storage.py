"""Unit tests for the SQLite message store.

Everything uses temporary databases under ``tmp_path``; the developer's real
database and settings are never touched.
"""

import sqlite3

from referral_finder.models import IncomingMessage
from referral_finder.storage import MessageStore


def make_message(**overrides) -> IncomingMessage:
    data = {
        "chat_id": -1001234567890,
        "message_id": 100,
        "date_utc": 1767224000,
        "received_at_utc": 1767224005,
        "text": "Hello courier, referral please",
        "sender_id": 777,
    }
    data.update(overrides)
    return IncomingMessage(**data)


# --- schema / creation ---------------------------------------------------


def test_creates_database_and_table_on_temp_path(tmp_path):
    db_path = tmp_path / "nested" / "data" / "messages.db"
    assert not db_path.exists()

    with MessageStore(db_path) as store:
        assert db_path.exists()
        assert store.count() == 0
        row = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
        ).fetchone()
        assert row == ("messages",)


def test_primary_key_is_chat_id_message_id(tmp_path):
    with MessageStore(tmp_path / "messages.db") as store:
        info = store._conn.execute("PRAGMA table_info(messages)").fetchall()
        columns = {name: (notnull, pk) for (_, name, _type, notnull, _dflt, pk) in info}
        assert columns["chat_id"] == (1, 1)
        assert columns["message_id"] == (1, 2)
        assert columns["date_utc"] == (1, 0)
        assert columns["received_at_utc"] == (1, 0)
        assert columns["text"] == (0, 0)
        assert columns["sender_id"] == (0, 0)


# --- persistence ---------------------------------------------------------


def test_save_stores_all_available_fields(tmp_path):
    with MessageStore(tmp_path / "messages.db") as store:
        message = make_message()
        assert store.save(message) is True
        assert store.count() == 1
        assert store.fetch(message.chat_id, message.message_id) == (
            message.chat_id,
            message.message_id,
            message.date_utc,
            message.received_at_utc,
            message.text,
            message.sender_id,
        )


def test_save_without_sender_id(tmp_path):
    with MessageStore(tmp_path / "messages.db") as store:
        message = make_message(sender_id=None)
        assert store.save(message) is True
        stored = store.fetch(message.chat_id, message.message_id)
        assert stored is not None
        assert stored[5] is None


def test_duplicate_insert_is_ignored_and_does_not_overwrite(tmp_path):
    with MessageStore(tmp_path / "messages.db") as store:
        original = make_message(text="first version")
        assert store.save(original) is True

        duplicate = make_message(text="second version")
        assert store.save(duplicate) is False

        assert store.count() == 1
        assert store.fetch(original.chat_id, original.message_id)[4] == "first version"


def test_same_message_id_in_different_chats_are_distinct(tmp_path):
    with MessageStore(tmp_path / "messages.db") as store:
        first = make_message(chat_id=-1001, message_id=42, text="chat one")
        second = make_message(chat_id=-1002, message_id=42, text="chat two")

        assert store.save(first) is True
        assert store.save(second) is True

        assert store.count() == 2
        assert store.fetch(-1001, 42)[4] == "chat one"
        assert store.fetch(-1002, 42)[4] == "chat two"


def test_stored_timestamps_round_trip(tmp_path):
    with MessageStore(tmp_path / "messages.db") as store:
        message = make_message(date_utc=1700000000, received_at_utc=1700000123)
        store.save(message)
        stored = store.fetch(message.chat_id, message.message_id)
        assert stored[2] == 1700000000
        assert stored[3] == 1700000123


# --- failure mode --------------------------------------------------------


def test_save_after_close_raises_sqlite_error(tmp_path):
    store = MessageStore(tmp_path / "messages.db")
    store.close()
    try:
        store.save(make_message())
    except sqlite3.Error:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected a sqlite3 error after close")


def test_close_is_idempotent(tmp_path):
    store = MessageStore(tmp_path / "messages.db")
    store.close()
    store.close()
