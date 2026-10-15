"""Tests for bounded history reconciliation.

All tests use a fake Telethon-like client, a temporary SQLite database, and the
real :class:`MessageWriter`. No network access and no real Telegram account are
used.
"""

import sqlite3
from datetime import datetime, timezone

from referral_finder.config import Config
from referral_finder.history import sync_history
from referral_finder.ingest import MessageWriter
from referral_finder.models import IncomingMessage
from referral_finder.storage import MessageStore
from referral_finder.telegram_client import TelegramRunner, handle_new_message

SAMPLE_DATE = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
CHAT = -1001234567890


def make_message(**overrides) -> IncomingMessage:
    data = {
        "chat_id": CHAT,
        "message_id": 1,
        "date_utc": int(SAMPLE_DATE.timestamp()),
        "received_at_utc": 1767224005,
        "text": "history text",
        "sender_id": 777,
    }
    data.update(overrides)
    return IncomingMessage(**data)


class FakeMessage:
    """Structural stand-in for a Telethon ``Message``."""

    def __init__(
        self,
        chat_id=CHAT,
        message_id=1,
        text="hi",
        out=False,
        sender_id=5,
        date=SAMPLE_DATE,
    ):
        self.chat_id = chat_id
        self.id = message_id
        self.raw_text = text
        self.out = out
        self.sender_id = sender_id
        self.date = date


class FakeHistoryClient:
    """Fake client exposing an async ``iter_messages`` with Telethon semantics."""

    def __init__(self, histories, error=None, error_at_index=None):
        self._histories = {
            int(chat): sorted(messages, key=lambda m: m.id)
            for chat, messages in histories.items()
        }
        self._error = error
        self._error_at_index = error_at_index
        self.calls = []

    def iter_messages(self, entity, limit=None, *, min_id=0, reverse=False, **kwargs):
        self.calls.append(
            {"entity": int(entity), "limit": limit, "min_id": min_id, "reverse": reverse}
        )
        messages = list(self._histories.get(int(entity), []))
        if reverse:
            seq = [m for m in messages if m.id > min_id]
        else:
            seq = list(reversed(messages))
        if limit is not None:
            seq = seq[:limit]
        return self._generate(seq)

    async def _generate(self, seq):
        if self._error is not None and self._error_at_index is None:
            raise self._error
        for index, message in enumerate(seq):
            if self._error is not None and index >= self._error_at_index:
                raise self._error
            yield message


class FailingWriteStore:
    """Delegating store whose ``save`` always fails (for error-path tests)."""

    def __init__(self, real):
        self._real = real

    def get_checkpoint(self, chat_id):
        return self._real.get_checkpoint(chat_id)

    def set_checkpoint(self, chat_id, last_message_id, updated_at_utc):
        self._real.set_checkpoint(chat_id, last_message_id, updated_at_utc)

    def save(self, message):
        raise sqlite3.OperationalError("disk I/O error")

    def save_many(self, messages):
        raise sqlite3.OperationalError("disk I/O error")


def make_config(tmp_path, chats=(str(CHAT),)) -> Config:
    return Config(
        api_id=123456,
        api_hash="fake_api_hash_value",
        session_path=tmp_path / "referral.session",
        db_path=tmp_path / "messages.db",
        log_level="INFO",
        chats_file=tmp_path / "chats.txt",
        chats=tuple(chats),
    )


# --- initial bounded reconciliation --------------------------------------


async def test_initial_sync_fills_gap_even_when_higher_id_present(tmp_path):
    db_path = tmp_path / "messages.db"
    store = MessageStore(db_path)
    # A previously stored message with a higher ID; 26 and 27 are missing.
    store.save(make_message(message_id=28, text="existing 28"))
    writer = MessageWriter(store)
    await writer.start()

    client = FakeHistoryClient(
        {CHAT: [FakeMessage(message_id=i) for i in (26, 27, 28)]}
    )
    result = await sync_history(client, [CHAT], store, writer, clock=lambda: 1000.0)
    await writer.stop()

    assert result[0].initial is True
    assert result[0].ok
    assert result[0].stored == 2  # 26, 27; 28 is a duplicate
    assert result[0].checkpoint_before is None
    assert result[0].checkpoint_after == 28
    assert store.get_checkpoint(CHAT) == 28
    assert store.count() == 3
    assert store.fetch(CHAT, 26) is not None
    assert store.fetch(CHAT, 27) is not None
    assert store.fetch(CHAT, 28)[4] == "existing 28"  # not overwritten
    store.close()


async def test_initial_sync_is_bounded(tmp_path):
    store = MessageStore(tmp_path / "messages.db")
    writer = MessageWriter(store)
    await writer.start()

    history = [FakeMessage(message_id=i) for i in range(1, 301)]
    client = FakeHistoryClient({CHAT: history})
    result = await sync_history(
        client, [CHAT], store, writer, initial_limit=200, clock=lambda: 1.0
    )
    await writer.stop()

    # Only the newest 200 (101..300) are examined, not the whole archive.
    assert result[0].examined == 200
    assert store.fetch(CHAT, 100) is None
    assert store.fetch(CHAT, 101) is not None
    assert store.get_checkpoint(CHAT) == 300
    assert client.calls[0]["limit"] == 200
    store.close()


# --- subsequent reconciliation -------------------------------------------


async def test_subsequent_sync_saves_everything_after_checkpoint(tmp_path):
    store = MessageStore(tmp_path / "messages.db")
    store.set_checkpoint(CHAT, 30, 1)
    writer = MessageWriter(store)
    await writer.start()

    history = [FakeMessage(message_id=i) for i in range(28, 34)]  # 28..33
    client = FakeHistoryClient({CHAT: history})
    result = await sync_history(client, [CHAT], store, writer, clock=lambda: 2.0)
    await writer.stop()

    assert result[0].initial is False
    assert result[0].examined == 3  # 31, 32, 33
    assert result[0].stored == 3
    assert store.get_checkpoint(CHAT) == 33
    assert store.fetch(CHAT, 28) is None  # not refetched (<= checkpoint)
    assert store.fetch(CHAT, 31) is not None
    store.close()


async def test_repeated_sync_creates_no_duplicates(tmp_path):
    store = MessageStore(tmp_path / "messages.db")
    writer = MessageWriter(store)
    await writer.start()

    client = FakeHistoryClient({CHAT: [FakeMessage(message_id=i) for i in (1, 2, 3)]})
    await sync_history(client, [CHAT], store, writer, clock=lambda: 1.0)
    count_after_first = store.count()

    second = await sync_history(client, [CHAT], store, writer, clock=lambda: 1.0)
    await writer.stop()

    assert count_after_first == 3
    assert store.count() == 3
    assert second[0].stored == 0  # already present, no new rows
    store.close()


async def test_checkpoints_are_isolated_per_chat(tmp_path):
    store = MessageStore(tmp_path / "messages.db")
    writer = MessageWriter(store)
    await writer.start()

    client = FakeHistoryClient(
        {
            -1001: [FakeMessage(chat_id=-1001, message_id=5)],
            -1002: [FakeMessage(chat_id=-1002, message_id=9)],
        }
    )
    await sync_history(client, [-1001, -1002], store, writer, clock=lambda: 1.0)
    await writer.stop()

    assert store.get_checkpoint(-1001) == 5
    assert store.get_checkpoint(-1002) == 9
    store.close()


# --- reliability ---------------------------------------------------------


async def test_fetch_error_does_not_advance_checkpoint(tmp_path):
    store = MessageStore(tmp_path / "messages.db")
    store.set_checkpoint(CHAT, 30, 1)
    writer = MessageWriter(store)
    await writer.start()

    client = FakeHistoryClient({CHAT: []}, error=RuntimeError("network down"))
    result = await sync_history(client, [CHAT], store, writer, clock=lambda: 1.0)
    await writer.stop()

    assert result[0].error == "RuntimeError"
    assert result[0].checkpoint_after == 30
    assert store.get_checkpoint(CHAT) == 30
    store.close()


async def test_partial_failure_commits_only_processed_pages(tmp_path):
    store = MessageStore(tmp_path / "messages.db")
    store.set_checkpoint(CHAT, 0, 1)
    writer = MessageWriter(store)
    await writer.start()

    history = [FakeMessage(message_id=i) for i in range(1, 6)]  # 1..5
    # First page (1, 2) is yielded, then the iterator raises.
    client = FakeHistoryClient(
        {CHAT: history}, error=RuntimeError("boom"), error_at_index=2
    )
    result = await sync_history(
        client, [CHAT], store, writer, page_size=2, clock=lambda: 1.0
    )
    await writer.stop()

    assert result[0].error == "RuntimeError"
    assert store.get_checkpoint(CHAT) == 2  # committed first page only
    assert store.fetch(CHAT, 1) is not None
    assert store.fetch(CHAT, 2) is not None
    assert store.fetch(CHAT, 3) is None
    store.close()


async def test_persistence_error_does_not_advance_checkpoint(tmp_path):
    real = MessageStore(tmp_path / "messages.db")
    real.set_checkpoint(CHAT, 0, 1)
    writer = MessageWriter(FailingWriteStore(real))
    await writer.start()

    client = FakeHistoryClient(
        {CHAT: [FakeMessage(message_id=1), FakeMessage(message_id=2)]}
    )
    result = await sync_history(client, [CHAT], real, writer, clock=lambda: 1.0)
    await writer.stop()

    assert result[0].error is not None
    assert real.get_checkpoint(CHAT) == 0  # not advanced over unpersisted range
    assert real.count() == 0
    real.close()


# --- filtering -----------------------------------------------------------


async def test_outgoing_and_textless_messages_are_skipped(tmp_path):
    store = MessageStore(tmp_path / "messages.db")
    writer = MessageWriter(store)
    await writer.start()

    client = FakeHistoryClient(
        {
            CHAT: [
                FakeMessage(message_id=10, text="mine", out=True),
                FakeMessage(message_id=11, text=None),
                FakeMessage(message_id=12, text="real incoming"),
            ]
        }
    )
    result = await sync_history(client, [CHAT], store, writer, clock=lambda: 1.0)
    await writer.stop()

    assert result[0].examined == 3
    assert result[0].submitted == 1
    assert result[0].skipped == 2
    assert result[0].stored == 1
    assert store.fetch(CHAT, 10) is None
    assert store.fetch(CHAT, 11) is None
    assert store.fetch(CHAT, 12)[4] == "real incoming"
    assert store.get_checkpoint(CHAT) == 12  # skipped ids are still "seen"
    store.close()


# --- live handler interplay ----------------------------------------------


async def test_history_and_live_delivery_do_not_duplicate(tmp_path):
    db_path = tmp_path / "messages.db"
    store = MessageStore(db_path)
    writer = MessageWriter(store)
    await writer.start()

    client = FakeHistoryClient({CHAT: [FakeMessage(message_id=40, text="once")]})
    await sync_history(client, [CHAT], store, writer, clock=lambda: 1.0)

    # A live re-delivery of the same message must not add a row.
    await handle_new_message(FakeMessage(message_id=40, text="once"), writer=writer)
    await writer.flush()
    await writer.stop()

    assert store.count() == 1
    store.close()


# --- runner integration --------------------------------------------------


class FakeRunnerClient:
    def __init__(self, histories):
        self._history = FakeHistoryClient(histories)
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

    def iter_messages(self, entity, **kwargs):
        return self._history.iter_messages(entity, **kwargs)


async def test_runner_runs_history_sync_when_enabled(tmp_path):
    db_path = tmp_path / "messages.db"
    store = MessageStore(db_path)
    writer = MessageWriter(store)
    client = FakeRunnerClient(
        {CHAT: [FakeMessage(message_id=1), FakeMessage(message_id=2)]}
    )
    runner = TelegramRunner(
        make_config(tmp_path), client=client, writer=writer, store=store, history_sync=True
    )

    await runner.start()

    assert store.count() == 2
    assert store.get_checkpoint(CHAT) == 2
    assert "catch_up" in client.calls

    await runner.shutdown()
    # Closed by shutdown; reopening proves the data was committed.
    reopened = MessageStore(db_path)
    assert reopened.count() == 2
    reopened.close()


async def test_runner_skips_history_sync_when_disabled(tmp_path):
    db_path = tmp_path / "messages.db"
    store = MessageStore(db_path)
    writer = MessageWriter(store)
    client = FakeRunnerClient({CHAT: [FakeMessage(message_id=1)]})
    runner = TelegramRunner(
        make_config(tmp_path), client=client, writer=writer, store=store
    )

    await runner.start()

    assert store.count() == 0
    assert store.get_checkpoint(CHAT) is None
    assert client._history.calls == []

    await runner.shutdown()


# --- writer flush --------------------------------------------------------


async def test_writer_flush_waits_for_persistence(tmp_path):
    store = MessageStore(tmp_path / "messages.db")
    writer = MessageWriter(store)
    await writer.start()
    for message_id in range(1, 6):
        await writer.submit(make_message(message_id=message_id))
    await writer.flush()
    assert store.count() == 5
    await writer.stop()
    store.close()
