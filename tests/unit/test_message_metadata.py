"""Tests for message text / sender / topic / url / protection extraction.

These tests exercise the shared mapping used by both the live handler and
history reconciliation. No network access.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

from referral_finder.models import (
    GENERAL_TOPIC_ID,
    build_incoming,
    incoming_from_object,
)
from referral_finder.telegram_client import extract_message

SAMPLE_DATE = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


class MessageActionTopicCreate:
    def __init__(self, title="New topic"):
        self.title = title


class FakeMessage:
    def __init__(
        self,
        chat_id=-1001234567890,
        message_id=100,
        text="hello",
        date=SAMPLE_DATE,
        sender=None,
        sender_id=777,
        reply_to=None,
        action=None,
        noforwards=None,
        post_author=None,
    ):
        self.chat_id = chat_id
        self.id = message_id
        self.raw_text = text
        self.date = date
        self.sender = sender
        self.sender_id = sender_id
        self.reply_to = reply_to
        self.action = action
        self.noforwards = noforwards
        self.post_author = post_author


class FakeUser:
    def __init__(self, username=None, first_name=None, last_name=None):
        self.username = username
        self.first_name = first_name
        self.last_name = last_name


class FakeChannelSender:
    def __init__(self, username=None, title="Channel"):
        self.username = username
        self.title = title


class FakeChannelEntity:
    def __init__(
        self,
        id=-1001234567890,
        username=None,
        megagroup=True,
        forum=True,
        broadcast=False,
        noforwards=None,
    ):
        self.id = id
        self.username = username
        self.megagroup = megagroup
        self.forum = forum
        self.broadcast = broadcast
        self.noforwards = noforwards


class FakeBasicGroupEntity:
    def __init__(self, id=-555, noforwards=None):
        self.id = id
        self.noforwards = noforwards


# --- message text --------------------------------------------------------


def test_full_text_is_preserved_with_unicode_newlines_and_caption():
    text = "Привет 👋\nВторая строка\tтаб"
    message = incoming_from_object(FakeMessage(text=text), received_at_utc=1)
    assert message.text == text


def test_empty_text_is_normalised_to_none():
    assert incoming_from_object(FakeMessage(text=""), received_at_utc=1).text is None
    assert incoming_from_object(FakeMessage(text=None), received_at_utc=1).text is None


# --- topic id ------------------------------------------------------------


def test_topic_id_from_forum_reply_header():
    msg = FakeMessage(
        reply_to=SimpleNamespace(
            forum_topic=True, reply_to_top_id=555, reply_to_msg_id=600
        )
    )
    assert incoming_from_object(msg, received_at_utc=1).topic_id == 555


def test_topic_id_falls_back_to_reply_to_msg_id():
    msg = FakeMessage(
        reply_to=SimpleNamespace(
            forum_topic=True, reply_to_top_id=None, reply_to_msg_id=600
        )
    )
    assert incoming_from_object(msg, received_at_utc=1).topic_id == 600


def test_topic_creation_message_uses_its_own_id():
    msg = FakeMessage(message_id=777, action=MessageActionTopicCreate())
    assert incoming_from_object(msg, received_at_utc=1).topic_id == 777


def test_general_topic_is_one_for_forum_chats():
    entity = FakeChannelEntity(forum=True)
    msg = FakeMessage(reply_to=None)
    assert incoming_from_object(msg, received_at_utc=1, chat_entity=entity).topic_id == (
        GENERAL_TOPIC_ID
    )


def test_plain_chat_has_no_topic_id():
    entity = FakeChannelEntity(forum=False)
    msg = FakeMessage(reply_to=SimpleNamespace(forum_topic=False))
    assert incoming_from_object(msg, received_at_utc=1, chat_entity=entity).topic_id is None


# --- sender --------------------------------------------------------------


def test_sender_user_fields():
    sender = FakeUser(username="ivan", first_name="Иван", last_name="Петров")
    message = incoming_from_object(FakeMessage(sender=sender), received_at_utc=1)
    assert message.sender_username == "ivan"
    assert message.sender_first_name == "Иван"
    assert message.sender_last_name == "Петров"
    assert message.sender_display_name == "Иван Петров"


def test_sender_without_username():
    sender = FakeUser(first_name="Иван", last_name="Петров")
    message = incoming_from_object(FakeMessage(sender=sender), received_at_utc=1)
    assert message.sender_username is None
    assert message.sender_display_name == "Иван Петров"


def test_channel_sender_has_title_not_user_names():
    sender = FakeChannelSender(username="news", title="Новости")
    message = incoming_from_object(FakeMessage(sender=sender), received_at_utc=1)
    assert message.sender_username == "news"
    assert message.sender_first_name is None
    assert message.sender_last_name is None
    assert message.sender_display_name == "Новости"


def test_anonymous_admin_uses_post_author():
    message = incoming_from_object(
        FakeMessage(sender=None, sender_id=None, post_author="Аноним"),
        received_at_utc=1,
    )
    assert message.sender_username is None
    assert message.sender_display_name == "Аноним"


def test_missing_sender_information_is_all_none():
    message = incoming_from_object(
        FakeMessage(sender=None, sender_id=None, post_author=None), received_at_utc=1
    )
    assert message.sender_username is None
    assert message.sender_first_name is None
    assert message.sender_last_name is None
    assert message.sender_display_name is None


# --- message url ---------------------------------------------------------


def test_message_url_public_forum_includes_topic():
    entity = FakeChannelEntity(id=-100123, username="grp", forum=True)
    msg = FakeMessage(message_id=100, reply_to=SimpleNamespace(
        forum_topic=True, reply_to_top_id=5, reply_to_msg_id=5
    ))
    message = incoming_from_object(msg, received_at_utc=1, chat_entity=entity)
    assert message.message_url == "https://t.me/grp/5/100"


def test_message_url_public_general_topic_has_no_topic_part():
    entity = FakeChannelEntity(id=-100123, username="grp", forum=True)
    msg = FakeMessage(message_id=100, reply_to=None)
    message = incoming_from_object(msg, received_at_utc=1, chat_entity=entity)
    assert message.message_url == "https://t.me/grp/100"


def test_message_url_private_supergroup_uses_c_form():
    entity = FakeChannelEntity(id=-1001234567890, username=None, forum=True)
    msg = FakeMessage(message_id=100, reply_to=None)
    message = incoming_from_object(msg, received_at_utc=1, chat_entity=entity)
    assert message.message_url == "https://t.me/c/1234567890/100"


def test_message_url_is_none_for_basic_group_and_unknown_entity():
    msg = FakeMessage(message_id=100)
    assert incoming_from_object(msg, received_at_utc=1, chat_entity=FakeBasicGroupEntity()).message_url is None
    assert incoming_from_object(msg, received_at_utc=1, chat_entity=None).message_url is None


# --- protection ----------------------------------------------------------


def test_protection_from_message_flag():
    assert incoming_from_object(FakeMessage(noforwards=True), received_at_utc=1).is_protected == 1


def test_protection_from_chat_flag():
    entity = FakeChannelEntity(noforwards=True)
    msg = FakeMessage(noforwards=False)
    assert incoming_from_object(msg, received_at_utc=1, chat_entity=entity).is_protected == 1


def test_not_protected_when_both_flags_false():
    entity = FakeChannelEntity(noforwards=False)
    msg = FakeMessage(noforwards=False)
    assert incoming_from_object(msg, received_at_utc=1, chat_entity=entity).is_protected == 0


def test_protection_unknown_is_none():
    entity = FakeChannelEntity(noforwards=None)
    msg = FakeMessage(noforwards=None)
    assert incoming_from_object(msg, received_at_utc=1, chat_entity=entity).is_protected is None


def test_message_without_flag_does_not_prove_chat_is_unprotected():
    entity = FakeChannelEntity(noforwards=None)
    msg = FakeMessage(noforwards=False)
    assert incoming_from_object(msg, received_at_utc=1, chat_entity=entity).is_protected is None


# --- sender kind ---------------------------------------------------------
# Classes are named like the real Telethon types, because sender_kind() uses the
# entity class name (entity type), not username presence.


class User:
    def __init__(self, username=None, first_name=None, last_name=None):
        self.username = username
        self.first_name = first_name
        self.last_name = last_name


class Channel:
    def __init__(self, username=None, title="Channel"):
        self.username = username
        self.title = title


class Chat:
    def __init__(self, title="Chat"):
        self.title = title


def test_sender_kind_user():
    message = incoming_from_object(FakeMessage(sender=User(first_name="Иван")), received_at_utc=1)
    assert message.sender_kind == "user"


def test_sender_kind_channel():
    message = incoming_from_object(FakeMessage(sender=Channel(title="Новости")), received_at_utc=1)
    assert message.sender_kind == "channel"


def test_sender_kind_chat():
    message = incoming_from_object(FakeMessage(sender=Chat()), received_at_utc=1)
    assert message.sender_kind == "chat"


def test_sender_kind_anonymous_from_post_author():
    message = incoming_from_object(
        FakeMessage(sender=None, post_author="Аноним"), received_at_utc=1
    )
    assert message.sender_kind == "anonymous"


def test_sender_kind_unknown_without_entity():
    message = incoming_from_object(FakeMessage(sender=None), received_at_utc=1)
    assert message.sender_kind == "unknown"


# --- resolver + consistency ---------------------------------------------


class FakeResolver:
    def __init__(self, title="Работа"):
        self.title = title

    async def title_for(self, chat_id, topic_id, entity=None):
        return self.title


async def test_build_incoming_resolves_topic_title():
    entity = FakeChannelEntity(forum=True)
    msg = FakeMessage(reply_to=SimpleNamespace(
        forum_topic=True, reply_to_top_id=5, reply_to_msg_id=5
    ))
    message = await build_incoming(
        msg, received_at_utc=1, chat_entity=entity, topic_resolver=FakeResolver("Работа")
    )
    assert message.topic_id == 5
    assert message.topic_title == "Работа"


async def test_title_failure_does_not_raise():
    class BrokenResolver:
        async def title_for(self, chat_id, topic_id, entity=None):
            raise RuntimeError("boom")

    entity = FakeChannelEntity(forum=True)
    msg = FakeMessage(reply_to=SimpleNamespace(
        forum_topic=True, reply_to_top_id=5, reply_to_msg_id=5
    ))
    # A resolver failure must not raise or drop the message: topic_id is kept,
    # the title is simply absent.
    message = await build_incoming(
        msg, received_at_utc=1, chat_entity=entity, topic_resolver=BrokenResolver()
    )
    assert message.topic_id == 5
    assert message.topic_title is None
    assert message.text == "hello"


def test_extract_message_matches_history_mapping():
    entity = FakeChannelEntity(forum=True)
    msg = FakeMessage(text="same", reply_to=None)
    live = extract_message(msg)
    history = incoming_from_object(msg, received_at_utc=live.received_at_utc, chat_entity=entity)
    # extract_message has no chat_entity, so topic_id differs for General; other
    # core fields must match exactly.
    assert live.chat_id == history.chat_id
    assert live.message_id == history.message_id
    assert live.text == history.text
    assert live.sender_id == history.sender_id
    assert live.date_utc == history.date_utc
