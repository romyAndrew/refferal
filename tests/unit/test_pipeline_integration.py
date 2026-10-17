"""Stage 4 integration tests: handler -> writer -> SQLite robustness.

These tests exercise the whole local pipeline with fake Telethon events and
temporary SQLite databases. No network access and no real Telegram account are
used. They also cover failure handling: writer crashes must be surfaced, DB
write errors must not hang the writer or vanish silently, backpressure must
block predictably, and start/stop must be re-entrant without leaking tasks.
"""

import asyncio
import logging
import sqlite3
from datetime import datetime, timezone

import pytest

from referral_finder.config import Config
from referral_finder.ingest import MessageWriter, MessageWriterError
from referral_finder.models import IncomingMessage
from referral_finder.storage import MessageStore
from referral_finder.telegram_client import TelegramRunner, handle_new_message

SAMPLE_DATE = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
DEFAULT_CHAT = -1001234567890


def make_message(**overrides) -> IncomingMessage:
    data = {
        "chat_id": DEFAULT_CHAT,
        "message_id": 100,
        "date_utc": 1767224000,
        "received_at_utc": 1767224005,
        "text": "Hello courier",
        "sender_id": 777,
    }
    data.update(overrides)
    return IncomingMessage(**data)


def make_config(tmp_path, chats=(str(DEFAULT_CHAT),)) -> Config:
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
        chat_id=DEFAULT_CHAT,
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


class FlakyStore:
    """Delegates to a real store but fails the first write."""

    def __init__(self, real: MessageStore):
        self.real = real
        self.calls = 0

    def save(self, message):
        self.calls += 1
        if self.calls == 1:
            raise sqlite3.OperationalError("transient disk error")
        return self.real.save(message)


class WriterBoom(BaseException):
    """Non-Exception failure used to kill the writer task."""


class FatalStore:
    """Every write raises a non-Exception error, killing the writer task."""

    def save(self, message):
        raise WriterBoom("fatal")


class FailAfterFirstStore:
    """First write succeeds, later writes raise a non-Exception error."""

    def __init__(self):
        self.calls = 0

    def save(self, message):
        self.calls += 1
        if self.calls >= 2:
            raise WriterBoom("fatal after first")
        return True


async def wait_until(predicate, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("condition was not met before the timeout")


# --- happy path: event -> handler -> queue -> SQLite ----------------------


async def test_text_message_reaches_sqlite_with_all_fields(tmp_path):
    db_path = tmp_path / "messages.db"
    store = MessageStore(db_path)
    writer = MessageWriter(store)
    client = FakeClient()
    runner = TelegramRunner(make_config(tmp_path), client=client, writer=writer, store=store)

    await runner.start()
    handler, _ = client.handlers[0]
    await handler(FakeEvent(message_id=321, sender_id=55, raw_text="Need a referral"))
    await runner.shutdown()

    reopened = MessageStore(db_path)
    stored = reopened.fetch(DEFAULT_CHAT, 321)
    assert stored[0] == DEFAULT_CHAT
    assert stored[1] == 321
    assert stored[2] == int(SAMPLE_DATE.timestamp())
    assert isinstance(stored[3], int)
    assert stored[4] == "Need a referral"
    assert stored[5] == 55
    reopened.close()


async def test_message_without_text_is_not_stored(tmp_path):
    db_path = tmp_path / "messages.db"
    store = MessageStore(db_path)
    writer = MessageWriter(store)
    client = FakeClient()
    runner = TelegramRunner(make_config(tmp_path), client=client, writer=writer, store=store)

    await runner.start()
    handler, _ = client.handlers[0]
    await handler(FakeEvent(message_id=1, raw_text=None))
    await handler(FakeEvent(message_id=2, raw_text=""))
    await runner.shutdown()

    reopened = MessageStore(db_path)
    assert reopened.count() == 0
    reopened.close()


async def test_duplicate_delivery_and_same_id_in_other_chat(tmp_path):
    db_path = tmp_path / "messages.db"
    store = MessageStore(db_path)
    writer = MessageWriter(store)
    client = FakeClient()
    runner = TelegramRunner(make_config(tmp_path), client=client, writer=writer, store=store)

    await runner.start()
    handler, _ = client.handlers[0]
    await handler(FakeEvent(chat_id=-1001, message_id=5, raw_text="first version"))
    await handler(FakeEvent(chat_id=-1001, message_id=5, raw_text="second version"))  # dup
    await handler(FakeEvent(chat_id=-1002, message_id=5, raw_text="other chat"))
    await runner.shutdown()

    reopened = MessageStore(db_path)
    assert reopened.count() == 2
    assert reopened.fetch(-1001, 5)[4] == "first version"
    assert reopened.fetch(-1002, 5)[4] == "other chat"
    reopened.close()


# --- shutdown / lifecycle ------------------------------------------------


async def test_shutdown_drains_queue_and_closes_store(tmp_path):
    db_path = tmp_path / "messages.db"
    store = MessageStore(db_path)
    writer = MessageWriter(store)
    client = FakeClient()
    runner = TelegramRunner(make_config(tmp_path), client=client, writer=writer, store=store)

    await runner.start()
    handler, _ = client.handlers[0]
    for i in range(25):
        await handler(FakeEvent(message_id=i, raw_text=f"message {i}"))

    await runner.shutdown()

    assert writer.running is False
    assert writer.pending == 0
    assert writer.saved == 25
    # The store must be closed by shutdown (a write now raises a DB error).
    with pytest.raises(sqlite3.ProgrammingError):
        store.save(make_message(message_id=999))

    reopened = MessageStore(db_path)
    assert reopened.count() == 25
    reopened.close()


async def test_runner_lifecycle_order_and_resource_teardown(tmp_path):
    order = []

    class RecordingWriter:
        async def start(self):
            order.append("writer.start")

        async def submit(self, message):
            order.append("writer.submit")

        async def stop(self):
            order.append("writer.stop")

    class RecordingStore:
        def close(self):
            order.append("store.close")

    class RecordingClient(FakeClient):
        def add_event_handler(self, handler, builder):
            order.append("register")
            super().add_event_handler(handler, builder)

        async def start(self, *args, **kwargs):
            order.append("client.start")
            await super().start(*args, **kwargs)

        async def catch_up(self):
            order.append("client.catch_up")
            await super().catch_up()

        async def disconnect(self):
            order.append("client.disconnect")
            await super().disconnect()

    runner = TelegramRunner(
        make_config(tmp_path),
        client=RecordingClient(),
        writer=RecordingWriter(),
        store=RecordingStore(),
    )

    await runner.start()
    assert order == ["writer.start", "register", "client.start", "client.catch_up"]

    await runner.shutdown()
    # Teardown order: stop event intake first, drain the writer, close SQLite
    # last. This keeps Telethon from cancelling handlers mid-write and keeps the
    # store open until the writer is done.
    assert order[4:] == ["client.disconnect", "writer.stop", "store.close"]


# --- failure handling ----------------------------------------------------


async def test_db_write_error_is_logged_and_writer_continues(tmp_path, caplog):
    store = MessageStore(tmp_path / "messages.db")
    writer = MessageWriter(FlakyStore(store))
    await writer.start()
    with caplog.at_level(logging.ERROR):
        await writer.submit(make_message(message_id=1, text="FIRST TEXT"))
        await writer.submit(make_message(message_id=2, text="SECOND TEXT"))
        await writer.stop()

    assert writer.errors == 1
    assert writer.saved == 1
    assert writer.running is False
    assert store.count() == 1
    assert store.fetch(DEFAULT_CHAT, 2) is not None
    assert "Failed to store message" in caplog.text
    assert "chat_id=" not in caplog.text
    assert "message_id=" not in caplog.text
    assert "FIRST TEXT" not in caplog.text
    assert "SECOND TEXT" not in caplog.text
    store.close()


async def test_fatal_writer_exception_is_surfaced_to_caller(tmp_path):
    writer = MessageWriter(FailAfterFirstStore())
    await writer.start()
    # First write succeeds deterministically...
    await writer.submit(make_message(message_id=1))
    # ...the second kills the writer task. submit may observe the death already
    # (and raise) or the death may happen just after; both are acceptable.
    try:
        await writer.submit(make_message(message_id=2))
    except MessageWriterError:
        pass
    await wait_until(lambda: writer.failure is not None)

    assert isinstance(writer.failure, WriterBoom)
    assert writer.running is False

    # stop() surfaces the failure as MessageWriterError (chained to the cause).
    with pytest.raises(MessageWriterError) as stop_exc:
        await writer.stop()
    assert isinstance(stop_exc.value.__cause__, WriterBoom)

    # A subsequent submit must not block forever or silently accept the message.
    with pytest.raises(MessageWriterError):
        await writer.submit(make_message(message_id=2))


async def test_shutdown_logs_writer_failure(tmp_path, caplog):
    writer = MessageWriter(FatalStore())
    client = FakeClient()
    runner = TelegramRunner(
        make_config(tmp_path), client=client, writer=writer, store=None
    )

    await runner.start()
    handler, _ = client.handlers[0]
    await handler(FakeEvent(message_id=1, raw_text="will fail"))
    await wait_until(lambda: writer.failure is not None)

    with caplog.at_level(logging.ERROR):
        await runner.shutdown()

    assert "Error while stopping the message writer" in caplog.text


async def test_submit_raises_after_clean_stop(tmp_path):
    store = MessageStore(tmp_path / "messages.db")
    writer = MessageWriter(store)
    await writer.start()
    await writer.stop()

    with pytest.raises(MessageWriterError):
        await writer.submit(make_message())
    store.close()


# --- backpressure --------------------------------------------------------


async def test_bounded_queue_caps_pending_and_loses_nothing(tmp_path):
    store = MessageStore(tmp_path / "messages.db")
    writer = MessageWriter(store, maxsize=4)

    tasks = [
        asyncio.create_task(writer.submit(make_message(message_id=i)))
        for i in range(20)
    ]
    await asyncio.sleep(0.01)
    # No consumer yet, so at most maxsize items may have been accepted.
    assert writer.pending <= 4
    assert sum(task.done() for task in tasks) <= 4

    await writer.start()
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
    await writer.stop()

    assert writer.saved == 20
    assert writer.pending == 0
    assert store.count() == 20
    store.close()


async def test_backpressure_blocks_until_consumer_drains(tmp_path):
    store = MessageStore(tmp_path / "messages.db")
    writer = MessageWriter(store, maxsize=3)

    for i in range(3):
        await writer.submit(make_message(message_id=i))
    assert writer.pending == 3

    blocked = asyncio.create_task(writer.submit(make_message(message_id=99)))
    await asyncio.sleep(0)
    assert blocked.done() is False
    assert writer.pending == 3

    await writer.start()
    await asyncio.wait_for(blocked, timeout=2)
    await writer.stop()

    assert writer.saved == 4
    assert writer.pending == 0
    assert store.count() == 4
    store.close()


# --- restart hygiene -----------------------------------------------------


async def test_writer_restart_does_not_leak_tasks_or_connections(tmp_path):
    db_path = tmp_path / "messages.db"
    store = MessageStore(db_path)
    writer = MessageWriter(store)

    for cycle in range(3):
        await writer.start()
        await writer.submit(make_message(message_id=cycle))
        await writer.stop()

    assert writer.running is False
    leaked = [task for task in asyncio.all_tasks() if task.get_name() == "message-writer"]
    assert leaked == []
    assert store.count() == 3

    store.close()
    with pytest.raises(sqlite3.ProgrammingError):
        store.save(make_message(message_id=404))


# --- shutdown ordering / drain under pressure ----------------------------


async def test_shutdown_with_empty_queue(tmp_path):
    db_path = tmp_path / "messages.db"
    store = MessageStore(db_path)
    writer = MessageWriter(store)
    client = FakeClient()
    runner = TelegramRunner(make_config(tmp_path), client=client, writer=writer, store=store)

    await runner.start()
    assert writer.pending == 0
    await asyncio.wait_for(runner.shutdown(), timeout=2)

    assert writer.running is False
    assert writer.saved == 0
    assert "disconnect" in client.calls
    # Shutdown must have closed the store.
    with pytest.raises(sqlite3.ProgrammingError):
        store.save(make_message(message_id=1))

    reopened = MessageStore(db_path)
    assert reopened.count() == 0
    reopened.close()


async def test_shutdown_with_nonempty_queue_drains_everything(tmp_path):
    db_path = tmp_path / "messages.db"
    store = MessageStore(db_path)
    writer = MessageWriter(store)
    client = FakeClient()
    runner = TelegramRunner(make_config(tmp_path), client=client, writer=writer, store=store)

    await runner.start()
    for i in range(10):
        await writer.submit(make_message(message_id=i))

    await asyncio.wait_for(runner.shutdown(), timeout=2)

    assert writer.saved == 10
    assert writer.pending == 0
    reopened = MessageStore(db_path)
    assert reopened.count() == 10
    reopened.close()


async def test_shutdown_with_full_queue_and_waiting_submit_drains_without_hang(tmp_path):
    db_path = tmp_path / "messages.db"
    store = MessageStore(db_path)
    writer = MessageWriter(store, maxsize=2)
    client = FakeClient()
    runner = TelegramRunner(make_config(tmp_path), client=client, writer=writer, store=store)

    # Fill the bounded queue before any consumer exists, then block a third
    # submit waiting for space.
    await writer.submit(make_message(message_id=1))
    await writer.submit(make_message(message_id=2))
    blocked = asyncio.create_task(writer.submit(make_message(message_id=3)))
    await asyncio.sleep(0)
    assert blocked.done() is False
    assert writer.pending == 2

    # Start the pipeline and immediately shut it down. Nothing accepted may be
    # lost and shutdown must not hang while a submit is waiting.
    await writer.start()
    await asyncio.wait_for(runner.shutdown(), timeout=2)
    await asyncio.wait_for(blocked, timeout=2)

    assert writer.saved == 3
    assert writer.pending == 0
    reopened = MessageStore(db_path)
    assert reopened.count() == 3
    reopened.close()


async def test_event_after_shutdown_is_not_persisted(tmp_path, caplog):
    db_path = tmp_path / "messages.db"
    store = MessageStore(db_path)
    writer = MessageWriter(store)
    client = FakeClient()
    runner = TelegramRunner(make_config(tmp_path), client=client, writer=writer, store=store)

    await runner.start()
    handler, _ = client.handlers[0]
    await runner.shutdown()

    # A late event (Telegram still dispatching, or a race during shutdown) must
    # not be persisted and must be logged, not swallowed.
    with caplog.at_level(logging.ERROR):
        await handler(FakeEvent(message_id=7, raw_text="late message"))

    assert writer.saved == 0
    assert "Message writer unavailable; message not persisted" in caplog.text
    assert "late message" not in caplog.text

    reopened = MessageStore(db_path)
    assert reopened.count() == 0
    reopened.close()


async def test_handler_cancelled_while_waiting_for_queue_space_cleans_up(tmp_path, caplog):
    store = MessageStore(tmp_path / "messages.db")
    writer = MessageWriter(store, maxsize=1)

    # Fill the queue with no consumer, so the handler's submit() blocks.
    await writer.submit(make_message(message_id=1))
    assert writer.pending == 1

    handler_task = asyncio.create_task(
        handle_new_message(FakeEvent(message_id=2, raw_text="blocked"), writer=writer)
    )
    await asyncio.sleep(0)
    assert handler_task.done() is False

    # Simulate Telethon cancelling the in-flight handler on disconnect.
    with caplog.at_level(logging.WARNING):
        handler_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await handler_task

    # The pending put must have been cancelled (no orphan), so the queue is
    # unchanged and only the first message can ever be persisted.
    assert writer.pending == 1
    assert len(writer._queue._putters) == 0
    assert "message not persisted" in caplog.text
    assert "blocked" not in caplog.text

    await writer.start()
    await writer.stop()
    assert writer.saved == 1
    assert store.count() == 1
    store.close()


async def test_shutdown_writer_error_still_closes_store_and_disconnects(tmp_path, caplog):
    class SpyStore:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    writer = MessageWriter(FatalStore())
    store = SpyStore()
    client = FakeClient()
    runner = TelegramRunner(make_config(tmp_path), client=client, writer=writer, store=store)

    await runner.start()
    handler, _ = client.handlers[0]
    await handler(FakeEvent(message_id=1, raw_text="will fail"))
    await wait_until(lambda: writer.failure is not None)

    with caplog.at_level(logging.ERROR):
        await asyncio.wait_for(runner.shutdown(), timeout=2)

    # A failing writer must not prevent the client disconnect or store close.
    assert "disconnect" in client.calls
    assert store.closed is True
    assert "Error while stopping the message writer" in caplog.text


async def test_disconnect_happens_before_writer_stop_and_store_close(tmp_path):
    order = []

    class RecordingWriter:
        async def start(self):
            order.append("writer.start")

        async def stop(self):
            order.append("writer.stop")

    class RecordingStore:
        def close(self):
            order.append("store.close")

    class RecordingClient(FakeClient):
        async def disconnect(self):
            order.append("client.disconnect")
            await super().disconnect()

    writer = RecordingWriter()
    runner = TelegramRunner(
        make_config(tmp_path),
        client=RecordingClient(),
        writer=writer,
        store=RecordingStore(),
    )

    await runner.start()
    order.clear()
    await runner.shutdown()

    assert order == ["client.disconnect", "writer.stop", "store.close"]


async def test_shutdown_continues_cleanup_when_disconnect_raises(tmp_path, caplog):
    class BoomDisconnectClient(FakeClient):
        async def disconnect(self):
            self.calls.append("disconnect")
            raise RuntimeError("network gone")

    db_path = tmp_path / "messages.db"
    store = MessageStore(db_path)
    writer = MessageWriter(store)
    runner = TelegramRunner(
        make_config(tmp_path), client=BoomDisconnectClient(), writer=writer, store=store
    )

    await runner.start()
    await writer.submit(make_message(message_id=1))
    with caplog.at_level(logging.ERROR):
        await asyncio.wait_for(runner.shutdown(), timeout=2)

    # A failing disconnect must not prevent the writer from draining or the
    # store from closing (or hang shutdown).
    assert writer.running is False
    assert writer.saved == 1
    assert "Error while disconnecting the Telegram client" in caplog.text
    with pytest.raises(sqlite3.ProgrammingError):
        store.save(make_message(message_id=2))

    reopened = MessageStore(db_path)
    assert reopened.count() == 1
    reopened.close()


async def test_shutdown_finishes_cleanup_then_reraises_cancellation(tmp_path):
    class CancelDisconnectClient(FakeClient):
        async def disconnect(self):
            self.calls.append("disconnect")
            raise asyncio.CancelledError()

    db_path = tmp_path / "messages.db"
    store = MessageStore(db_path)
    writer = MessageWriter(store)
    runner = TelegramRunner(
        make_config(tmp_path), client=CancelDisconnectClient(), writer=writer, store=store
    )

    await runner.start()
    await writer.submit(make_message(message_id=1))

    with pytest.raises(asyncio.CancelledError):
        await runner.shutdown()

    # Cancellation is remembered and re-raised, but the writer was still
    # stopped and the store still closed first.
    assert writer.running is False
    assert writer.saved == 1
    with pytest.raises(sqlite3.ProgrammingError):
        store.save(make_message(message_id=2))

    reopened = MessageStore(db_path)
    assert reopened.count() == 1
    reopened.close()
