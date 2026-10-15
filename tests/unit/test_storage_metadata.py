"""Tests for the extended message metadata columns and migration."""

import sqlite3

from referral_finder.models import IncomingMessage
from referral_finder.storage import MessageStore

CHAT = -1001234567890
NEW_COLUMNS = (
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


def make_message(**overrides) -> IncomingMessage:
    data = {
        "chat_id": CHAT,
        "message_id": 1,
        "date_utc": 100,
        "received_at_utc": 101,
        "text": "hello",
        "sender_id": 777,
        "topic_id": 5,
        "topic_title": "Работа",
        "sender_username": "ivan",
        "sender_first_name": "Иван",
        "sender_last_name": "Петров",
        "sender_display_name": "Иван Петров",
        "message_url": "https://t.me/grp/5/1",
        "is_protected": 0,
    }
    data.update(overrides)
    return IncomingMessage(**data)


def test_migration_adds_columns_and_preserves_rows(tmp_path):
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE messages ("
        "chat_id INTEGER NOT NULL, message_id INTEGER NOT NULL, "
        "date_utc INTEGER NOT NULL, received_at_utc INTEGER NOT NULL, "
        "text TEXT, sender_id INTEGER, PRIMARY KEY (chat_id, message_id))"
    )
    conn.execute("INSERT INTO messages VALUES (-1, 5, 1, 1, 'old text', 9)")
    conn.commit()
    conn.close()

    store = MessageStore(db_path)
    columns = {row[1] for row in store._conn.execute("PRAGMA table_info(messages)")}
    for name in NEW_COLUMNS:
        assert name in columns
    assert store.count() == 1
    assert store.fetch(-1, 5)[4] == "old text"
    assert store.fetch_metadata(-1, 5) == {name: None for name in NEW_COLUMNS}
    store.close()


def test_save_stores_topic_and_sender_metadata(tmp_path):
    with MessageStore(tmp_path / "messages.db") as store:
        message = make_message()
        assert store.save(message) is True
        metadata = store.fetch_metadata(CHAT, 1)
        assert metadata["topic_id"] == 5
        assert metadata["topic_title"] == "Работа"
        assert metadata["sender_username"] == "ivan"
        assert metadata["sender_first_name"] == "Иван"
        assert metadata["sender_last_name"] == "Петров"
        assert metadata["sender_display_name"] == "Иван Петров"
        assert metadata["message_url"] == "https://t.me/grp/5/1"
        assert metadata["is_protected"] == 0


def test_duplicate_fills_missing_metadata_without_touching_text(tmp_path):
    with MessageStore(tmp_path / "messages.db") as store:
        store.save(make_message(text="original", topic_id=None, topic_title=None))
        inserted = store.save(make_message(text="conflicting", topic_id=7, topic_title="New"))
        assert inserted is False
        assert store.count() == 1
        assert store.fetch(CHAT, 1)[4] == "original"  # text never overwritten
        assert store.fetch_metadata(CHAT, 1)["topic_id"] == 7  # NULL filled


def test_duplicate_does_not_overwrite_known_metadata(tmp_path):
    with MessageStore(tmp_path / "messages.db") as store:
        store.save(make_message(topic_id=7, topic_title="First"))
        store.save(make_message(topic_id=9, topic_title="Second"))
        metadata = store.fetch_metadata(CHAT, 1)
        assert metadata["topic_id"] == 7
        assert metadata["topic_title"] == "First"


def test_save_many_stores_metadata_and_reports_counts(tmp_path):
    with MessageStore(tmp_path / "messages.db") as store:
        store.save(make_message(message_id=1, topic_id=1, topic_title="General"))
        inserted, duplicates = store.save_many(
            [
                make_message(message_id=1, topic_id=1, topic_title="General"),
                make_message(message_id=2, topic_id=2, topic_title="Second"),
                make_message(message_id=3, topic_id=3, topic_title="Third"),
            ]
        )
        assert (inserted, duplicates) == (2, 1)
        assert store.count() == 3
        assert store.fetch_metadata(CHAT, 2)["topic_title"] == "Second"


def test_save_many_fills_missing_metadata_of_existing_row(tmp_path):
    with MessageStore(tmp_path / "messages.db") as store:
        store.save(make_message(message_id=1, topic_id=None, topic_title=None))
        store.save_many([make_message(message_id=1, topic_id=42, topic_title="Filled")])
        metadata = store.fetch_metadata(CHAT, 1)
        assert metadata["topic_id"] == 42
        assert metadata["topic_title"] == "Filled"


def test_full_text_with_unicode_and_newlines_round_trips(tmp_path):
    text = "Привет 👋\nвторая строка\tтаб"
    with MessageStore(tmp_path / "messages.db") as store:
        store.save(make_message(text=text))
        assert store.fetch(CHAT, 1)[4] == text


def test_repeated_save_is_idempotent(tmp_path):
    with MessageStore(tmp_path / "messages.db") as store:
        first = store.save(make_message())
        second = store.save(make_message())
        assert first is True
        assert second is False
        assert store.count() == 1
