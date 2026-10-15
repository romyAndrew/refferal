"""Tests for ``title:`` configuration parsing and supergroup resolution."""

import pytest

from referral_finder.config import Config, ConfigError, parse_chat_entries
from referral_finder.telegram_client import TelegramRunner


class FakeEntity:
    def __init__(
        self,
        id,
        title,
        megagroup=True,
        forum=True,
        broadcast=False,
        username=None,
    ):
        self.id = id
        self.title = title
        self.megagroup = megagroup
        self.forum = forum
        self.broadcast = broadcast
        self.username = username


class FakeDialog:
    def __init__(self, name, entity):
        self.name = name
        self.entity = entity


class FakeDialogClient:
    def __init__(self, dialogs):
        self._dialogs = dialogs
        self.handlers = []
        self.calls = []

    async def get_dialogs(self):
        self.calls.append("get_dialogs")
        return self._dialogs

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


def make_config(tmp_path, chats) -> Config:
    return Config(
        api_id=123456,
        api_hash="fake_api_hash_value",
        session_path=tmp_path / "referral.session",
        db_path=tmp_path / "messages.db",
        log_level="INFO",
        chats_file=tmp_path / "chats.txt",
        chats=tuple(chats),
    )


# --- parsing -------------------------------------------------------------


def test_parse_chat_entries_splits_ids_and_titles():
    entries = parse_chat_entries(["-1001", "title:My Forum", "42"])
    assert entries.ids == (-1001, 42)
    assert entries.titles == ("My Forum",)
    assert entries.has_titles is True


def test_parse_chat_entries_numeric_only():
    entries = parse_chat_entries(["-1001"])
    assert entries.ids == (-1001,)
    assert entries.titles == ()
    assert entries.has_titles is False


def test_parse_chat_entries_trims_title():
    assert parse_chat_entries(["  title:  My  Forum  "]).titles == ("My  Forum",)


def test_parse_chat_entries_rejects_non_numeric():
    with pytest.raises(ConfigError):
        parse_chat_entries(["@username"])


def test_parse_chat_entries_rejects_empty_title():
    with pytest.raises(ConfigError):
        parse_chat_entries(["title:   "])


# --- runner resolution ---------------------------------------------------


async def test_title_resolves_to_single_forum_supergroup(tmp_path):
    entity = FakeEntity(id=-1001234567890, title="My Forum")
    client = FakeDialogClient([FakeDialog("My Forum", entity)])
    runner = TelegramRunner(make_config(tmp_path, ["title:My Forum"]), client=client)

    await runner.start()

    assert client.calls == ["start", "get_dialogs", "add_event_handler", "catch_up"]
    _, builder = client.handlers[0]
    assert list(builder.chats) == [-1001234567890]
    assert builder.incoming is True
    assert runner._entities_by_chat_id[-1001234567890] is entity


async def test_title_match_normalizes_whitespace_and_case(tmp_path):
    entity = FakeEntity(id=-1005, title="My   Forum")
    client = FakeDialogClient([FakeDialog("MY  FORUM", entity)])
    runner = TelegramRunner(make_config(tmp_path, ["title:my forum"]), client=client)

    await runner.start()

    assert runner._entities_by_chat_id[-1005] is entity


async def test_missing_title_raises_clear_error(tmp_path):
    client = FakeDialogClient([FakeDialog("Other", FakeEntity(-1, "Other"))])
    runner = TelegramRunner(make_config(tmp_path, ["title:Missing"]), client=client)

    with pytest.raises(ConfigError) as exc:
        await runner.start()
    assert "Missing" in str(exc.value)


async def test_ambiguous_title_is_rejected(tmp_path):
    dialogs = [
        FakeDialog("Same", FakeEntity(-1, "Same")),
        FakeDialog("Same", FakeEntity(-2, "Same")),
    ]
    client = FakeDialogClient(dialogs)
    runner = TelegramRunner(make_config(tmp_path, ["title:Same"]), client=client)

    with pytest.raises(ConfigError) as exc:
        await runner.start()
    assert "Multiple" in str(exc.value) or "numeric" in str(exc.value)


async def test_non_forum_group_is_rejected(tmp_path):
    entity = FakeEntity(id=-1, title="No Topics", megagroup=True, forum=False)
    client = FakeDialogClient([FakeDialog("No Topics", entity)])
    runner = TelegramRunner(make_config(tmp_path, ["title:No Topics"]), client=client)

    with pytest.raises(ConfigError) as exc:
        await runner.start()
    assert "forum" in str(exc.value).lower()


async def test_broadcast_channel_is_rejected(tmp_path):
    entity = FakeEntity(
        id=-1, title="News", megagroup=False, forum=False, broadcast=True
    )
    client = FakeDialogClient([FakeDialog("News", entity)])
    runner = TelegramRunner(make_config(tmp_path, ["title:News"]), client=client)

    with pytest.raises(ConfigError):
        await runner.start()


async def test_numeric_config_still_registers_before_start(tmp_path):
    client = FakeDialogClient([])
    runner = TelegramRunner(make_config(tmp_path, ["-1001234567890"]), client=client)

    await runner.start()

    # Numeric-only path preserves the original order.
    assert client.calls[0] == "add_event_handler"
    assert "get_dialogs" not in client.calls
