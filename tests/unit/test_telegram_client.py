"""Tests for the Telethon client wrapper, handler, and lifecycle.

No network access and no real Telegram account are used: the client is
constructed offline or replaced by fakes.
"""

import logging
from datetime import datetime, timezone

import pytest

from referral_finder.config import Config, ConfigError
from referral_finder.models import IncomingMessage
from referral_finder.telegram_client import (
    MessageLogRecord,
    TelegramRunner,
    create_client,
    extract_message,
    handle_new_message,
    new_message_builder,
    parse_chat_ids,
    to_log_record,
)

SAMPLE_DATE = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


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
    """Structural stand-in for TelegramClient, recording call order."""

    def __init__(self):
        self.calls = []
        self.handlers = []

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


# --- chat id parsing -----------------------------------------------------


def test_parse_chat_ids_converts_numeric_strings():
    assert parse_chat_ids(("-1001234567890", "42")) == (-1001234567890, 42)


def test_parse_chat_ids_rejects_non_numeric():
    with pytest.raises(ConfigError):
        parse_chat_ids(("@username",))


def test_parse_chat_ids_rejects_empty_string():
    with pytest.raises(ConfigError):
        parse_chat_ids(("",))


# --- event filter --------------------------------------------------------


def test_new_message_builder_filters_incoming_and_chats():
    builder = new_message_builder((-1001, -1002))
    assert builder.incoming is True
    assert list(builder.chats) == [-1001, -1002]


def test_new_message_builder_requires_chats():
    with pytest.raises(ConfigError):
        new_message_builder(())


# --- client construction -------------------------------------------------


def test_create_client_does_not_connect(tmp_path):
    config = make_config(tmp_path)
    client = create_client(config)
    assert client.api_id == 123456
    assert client.api_hash == "fake_api_hash_value"
    assert client.is_connected() is False


# --- event -> record mapping --------------------------------------------


def test_extract_message_maps_fields():
    event = FakeEvent(
        chat_id=-1001234567890,
        message_id=555,
        sender_id=777,
        raw_text="Hello courier",
    )
    message = extract_message(event)
    assert isinstance(message, IncomingMessage)
    assert message.chat_id == -1001234567890
    assert message.message_id == 555
    assert message.sender_id == 777
    assert message.text == "Hello courier"
    assert message.date_utc == int(SAMPLE_DATE.timestamp())
    assert isinstance(message.received_at_utc, int)


def test_extract_message_without_sender_id():
    event = FakeEvent(sender_id=None, raw_text="text")
    message = extract_message(event)
    assert message.sender_id is None


def test_extract_message_without_text():
    event = FakeEvent(raw_text=None)
    message = extract_message(event)
    assert message.text is None
    record = to_log_record(message)
    assert isinstance(record, MessageLogRecord)
    assert record.text_length == 0


# --- handler -------------------------------------------------------------


async def test_handler_logs_technical_record_without_full_text(caplog):
    event = FakeEvent(chat_id=-1001234567890, message_id=42, raw_text="SECRET MESSAGE TEXT")
    with caplog.at_level(logging.INFO, logger="referral_finder.telegram_client"):
        await handle_new_message(event)

    assert "chat_id=-1001234567890" in caplog.text
    assert "message_id=42" in caplog.text
    assert "text_length=19" in caplog.text
    assert "SECRET MESSAGE TEXT" not in caplog.text


async def test_handler_logs_extraction_errors(caplog):
    class BrokenEvent:
        pass

    with caplog.at_level(logging.ERROR, logger="referral_finder.telegram_client"):
        await handle_new_message(BrokenEvent())

    assert any(record.levelno == logging.ERROR for record in caplog.records)
    assert "Failed to extract" in caplog.text


# --- lifecycle -----------------------------------------------------------


async def test_runner_registers_before_start_and_catch_up(tmp_path):
    client = FakeClient()
    runner = TelegramRunner(make_config(tmp_path), client=client)

    await runner.run_forever()

    assert client.calls == [
        "add_event_handler",
        "start",
        "catch_up",
        "run_until_disconnected",
    ]


async def test_runner_shutdown_disconnects(tmp_path):
    client = FakeClient()
    runner = TelegramRunner(make_config(tmp_path), client=client)

    await runner.shutdown()

    assert "disconnect" in client.calls


async def test_runner_start_registers_handler_when_not_registered(tmp_path):
    client = FakeClient()
    runner = TelegramRunner(make_config(tmp_path), client=client)

    await runner.start()

    assert client.calls[0] == "add_event_handler"
    assert client.handlers
    _, builder = client.handlers[0]
    assert builder.incoming is True
    assert list(builder.chats) == [-1001234567890]


def test_runner_register_requires_chats(tmp_path):
    runner = TelegramRunner(make_config(tmp_path, chats=()), client=FakeClient())
    with pytest.raises(ConfigError):
        runner.register()


async def test_lifecycle_logs_exclude_secrets_and_session_path(tmp_path, caplog):
    config = make_config(tmp_path)
    client = FakeClient()
    runner = TelegramRunner(config, client=client)

    with caplog.at_level(logging.INFO, logger="referral_finder.telegram_client"):
        await runner.start()

    assert config.api_hash not in caplog.text
    assert str(config.session_path) not in caplog.text
