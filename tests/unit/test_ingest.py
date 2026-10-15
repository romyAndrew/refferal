"""Tests for the queue + single-writer ingestion pipeline.

Covers the writer in isolation (draining, error handling, deduplication
counting) and the end-to-end path
``fake Telethon event -> handler -> queue -> SQLite``. No network access and no
real Telegram account are used.
"""

import logging
import sqlite3
from datetime import datetime, timezone

from referral_finder.config import Config
from referral_finder.ingest import MessageWriter
from referral_finder.models import IncomingMessage
from referral_finder.storage import MessageStore
from referral_finder.telegram_client import TelegramRunner, extract_message

SAMPLE_DATE = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def make_message(**overrides) -> IncomingMessage:
    data = {
        "chat_id": -1001234567890,
        "message_id": 100,
        "date_utc": 1767224000,
        "received_at_utc": 1767224005,
        "text": "Hello courier",
        "sender_id": 777,
    }
    data.update(overrides)
    return IncomingMessage(**data)


def make_config(tmp_path, chats=("-1001234567890",)) -> Config:
    return Config(
        api_id=123456,
        api_hash="fake_api_hash_value",
        session_path=tmp_path / "referral.session",
        db_path=tmp_path / "messages.db",
        log_level="INFO",
        chats_file=tmp_path / "chats.txt",
        chats=tuple(chats),
    )


class FakeEvent:
    def __init__(
        self,
        chat_id=-1001234567890,
        message_id=100,
        date=SAMPLE_DATE,
        sender_id=None,
        raw_text=None,
    ):
        self.chat_id = chat_id
        self.id = message_id
        self.date = date
        self.sender_id = sender_id
        self.raw_text = raw_text


class FakeClient:
    """Minimal structural stand-in for TelegramClient."""

    def __init__(self):
        self.handlers = []
        self.calls = []

    def add_event_handler(self, handler, builder):
        self.calls.append("add_event_handler")
        self.handlers.append((handler, builder))

    async def start(self, *args, **kwargs):
        self.calls.append("start")

    async def catch_up(self):
        self.calls.append("catch_up")

    async def run_until_disconnected(self):
        self.calls.append("run_until_disconnected")

    def is_connected(self):
        return True

    async def disconnect(self):
        self.calls.append("disconnect")


class BrokenStore:
    """Store whose writes always fail, for error-path coverage."""

    def save(self, message):
        raise sqlite3.OperationalError("disk I/O error")


# --- writer in isolation -------------------------------------------------


async def test_writer_persists_submitted_messages(tmp_path):
    store = MessageStore(tmp_path / "messages.db")
    writer = MessageWriter(store)
    await writer.start()
    try:
        await writer.submit(make_message(message_id=1))
        await writer.submit(make_message(message_id=2))
    finally:
        await writer.stop()

    assert writer.saved == 2
    assert writer.errors == 0
    assert store.count() == 2
    store.close()


async def test_writer_shutdown_drains_all_accepted_items(tmp_path):
    store = MessageStore(tmp_path / "messages.db")
    writer = MessageWriter(store)
    await writer.start()
    for message_id in range(1, 6):
        await writer.submit(make_message(message_id=message_id))

    await writer.stop()

    assert writer.running is False
    assert writer.pending == 0
    assert writer.saved == 5
    assert store.count() == 5
    store.close()


async def test_writer_counts_duplicates_without_second_row(tmp_path):
    store = MessageStore(tmp_path / "messages.db")
    writer = MessageWriter(store)
    await writer.start()
    try:
        await writer.submit(make_message(message_id=7))
        await writer.submit(make_message(message_id=7))
    finally:
        await writer.stop()

    assert writer.saved == 1
    assert writer.duplicates == 1
    assert store.count() == 1
    store.close()


async def test_writer_persists_batch_and_counts_duplicates(tmp_path):
    store = MessageStore(tmp_path / "messages.db")
    writer = MessageWriter(store)
    await writer.start()
    try:
        await writer.submit_batch(
            [
                make_message(message_id=1),
                make_message(message_id=2),
                make_message(message_id=2),  # duplicate within the same batch
            ]
        )
        await writer.flush()
    finally:
        await writer.stop()

    assert writer.saved == 2
    assert writer.duplicates == 1
    assert writer.errors == 0
    assert store.count() == 2
    store.close()


async def test_writer_calls_on_saved_hook(tmp_path):
    store = MessageStore(tmp_path / "messages.db")
    seen = []
    writer = MessageWriter(store, on_saved=seen.extend)
    await writer.start()
    try:
        await writer.submit(make_message(message_id=1))
        await writer.submit(make_message(message_id=2))
        await writer.flush()
    finally:
        await writer.stop()

    assert sorted(m.message_id for m in seen) == [1, 2]
    store.close()


async def test_batch_submit_calls_on_saved_hook(tmp_path):
    store = MessageStore(tmp_path / "messages.db")
    seen = []
    writer = MessageWriter(store, on_saved=seen.extend)
    await writer.start()
    try:
        await writer.submit_batch(
            [make_message(message_id=1), make_message(message_id=2)]
        )
        await writer.flush()
    finally:
        await writer.stop()

    assert sorted(m.message_id for m in seen) == [1, 2]
    store.close()


async def test_on_saved_hook_failure_does_not_break_ingestion(tmp_path, caplog):
    store = MessageStore(tmp_path / "messages.db")

    def boom(messages):
        raise RuntimeError("classifier boom")

    writer = MessageWriter(store, on_saved=boom)
    await writer.start()
    with caplog.at_level(logging.ERROR):
        try:
            await writer.submit(make_message(message_id=1))
            await writer.submit(make_message(message_id=2))
            await writer.flush()
        finally:
            await writer.stop()

    assert store.count() == 2
    assert writer.errors == 0
    assert "Post-persist hook failed" in caplog.text
    store.close()


async def test_on_saved_hook_not_called_when_store_fails():
    seen = []
    writer = MessageWriter(BrokenStore(), on_saved=seen.extend)
    await writer.start()
    try:
        await writer.submit(make_message(message_id=1))
        await writer.flush()
    finally:
        await writer.stop()

    assert writer.errors == 1
    assert seen == []


async def test_writer_stop_without_start_is_noop():
    writer = MessageWriter(BrokenStore())
    await writer.stop()
    assert writer.running is False


async def test_writer_logs_persistence_errors_and_keeps_running(caplog):
    writer = MessageWriter(BrokenStore())
    await writer.start()
    with caplog.at_level(logging.INFO):
        try:
            await writer.submit(make_message(message_id=1, text="SECRET TEXT ONE"))
            await writer.submit(make_message(message_id=2, text="SECRET TEXT TWO"))
        finally:
            await writer.stop()

    assert writer.errors == 2
    assert writer.saved == 0
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 2
    assert "Failed to store message chat_id=-1001234567890 message_id=1" in caplog.text
    # Full text and secrets must never reach the logs.
    assert "SECRET TEXT ONE" not in caplog.text
    assert "SECRET TEXT TWO" not in caplog.text


# --- end-to-end: event -> handler -> queue -> SQLite ----------------------


async def test_event_flows_from_handler_through_queue_to_sqlite(tmp_path, caplog):
    db_path = tmp_path / "messages.db"
    store = MessageStore(db_path)
    writer = MessageWriter(store)
    client = FakeClient()
    runner = TelegramRunner(make_config(tmp_path), client=client, writer=writer, store=store)

    with caplog.at_level(logging.INFO):
        await runner.start()
        handler, builder = client.handlers[0]
        assert builder.incoming is True
        await handler(FakeEvent(message_id=321, sender_id=55, raw_text="Need a referral"))
        await runner.shutdown()

    assert writer.saved == 1
    reopened = MessageStore(db_path)
    stored = reopened.fetch(-1001234567890, 321)
    assert stored is not None
    assert stored[4] == "Need a referral"
    assert stored[5] == 55
    assert "Need a referral" not in caplog.text
    reopened.close()


async def test_message_without_text_is_not_stored(tmp_path):
    db_path = tmp_path / "messages.db"
    store = MessageStore(db_path)
    writer = MessageWriter(store)
    client = FakeClient()
    runner = TelegramRunner(make_config(tmp_path), client=client, writer=writer, store=store)

    await runner.start()
    handler, _ = client.handlers[0]
    await handler(FakeEvent(raw_text=None))
    await runner.shutdown()

    assert writer.saved == 0
    reopened = MessageStore(db_path)
    assert reopened.count() == 0
    reopened.close()


async def test_logs_exclude_message_text_and_secrets(tmp_path, caplog):
    text = "SUPER SECRET COURIER MESSAGE"
    config = make_config(tmp_path)
    store = MessageStore(tmp_path / "messages.db")
    writer = MessageWriter(store)
    client = FakeClient()
    runner = TelegramRunner(config, client=client, writer=writer, store=store)

    with caplog.at_level(logging.DEBUG):
        await runner.start()
        handler, _ = client.handlers[0]
        await handler(FakeEvent(raw_text=text))
        await handler(FakeEvent(message_id=101, raw_text=text))
        await runner.shutdown()

    assert text not in caplog.text
    assert config.api_hash not in caplog.text
    assert str(config.session_path) not in caplog.text
    store.close()


# --- date/time conversion ------------------------------------------------


def test_extract_message_converts_date_and_received_time():
    before = int(datetime.now(timezone.utc).timestamp())
    message = extract_message(FakeEvent(date=SAMPLE_DATE))
    after = int(datetime.now(timezone.utc).timestamp())

    assert message.date_utc == int(SAMPLE_DATE.timestamp())
    assert before <= message.received_at_utc <= after
