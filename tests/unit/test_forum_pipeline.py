"""Live + history forum pipeline: topic/sender metadata end to end.

No network: a fake Telethon client supplies messages, forum topics and the
parent entity. The real MessageWriter and MessageStore are used.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

from referral_finder.history import sync_history
from referral_finder.ingest import MessageWriter
from referral_finder.storage import MessageStore
from referral_finder.telegram_client import handle_new_message
from referral_finder.topics import TopicResolver

SAMPLE_DATE = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
CHAT = -1001234567890


class FakeMessage:
    def __init__(self, message_id, topic_id=None, text="hi", chat_id=CHAT, title_missing=False):
        self.chat_id = chat_id
        self.id = message_id
        self.raw_text = text
        self.date = SAMPLE_DATE
        self.sender = None
        self.sender_id = 7
        self.action = None
        self.noforwards = False
        self.post_author = None
        if topic_id is None:
            self.reply_to = None
        else:
            self.reply_to = SimpleNamespace(
                forum_topic=True, reply_to_top_id=topic_id, reply_to_msg_id=topic_id
            )


class FakeChannelEntity:
    def __init__(self, id=CHAT, forum=True, megagroup=True, username=None):
        self.id = id
        self.forum = forum
        self.megagroup = megagroup
        self.broadcast = False
        self.username = username
        self.noforwards = None


class FakeForumClient:
    def __init__(self, entity, messages, topics):
        self._entity = entity
        self._messages = list(messages)
        self._topics = list(topics)
        self.topic_calls = 0

    async def get_input_entity(self, target):
        return target

    async def __call__(self, request):
        self.topic_calls += 1
        objs = [
            SimpleNamespace(id=t[0], title=t[1], title_missing=(len(t) > 2 and t[2]))
            for t in self._topics
        ]
        return SimpleNamespace(topics=objs)

    def iter_messages(self, entity, limit=None, *, min_id=0, reverse=False, **kwargs):
        return self._generate(entity, limit, min_id, reverse)

    async def _generate(self, entity, limit, min_id, reverse):
        messages = [m for m in self._messages if m.chat_id == int(entity)]
        if reverse:
            seq = [m for m in messages if m.id > min_id]
        else:
            seq = list(reversed(messages))
        if limit is not None:
            seq = seq[:limit]
        for message in seq:
            yield message


def make_store_and_writer(tmp_path):
    store = MessageStore(tmp_path / "messages.db")
    writer = MessageWriter(store)
    return store, writer


async def test_history_stores_topic_metadata_for_all_topics(tmp_path):
    store, writer = make_store_and_writer(tmp_path)
    await writer.start()

    entity = FakeChannelEntity()
    messages = [
        FakeMessage(10, topic_id=5),
        FakeMessage(11, topic_id=6),
        FakeMessage(12, topic_id=None),  # General
        FakeMessage(13, topic_id=7),  # title missing
    ]
    client = FakeForumClient(entity, messages, topics=[(5, "Работа"), (6, "Офферы"), (7, None, True)])
    resolver = TopicResolver(client)

    await sync_history(
        client, [CHAT], store, writer, entities_by_chat_id={CHAT: entity}, topic_resolver=resolver
    )
    await writer.flush()
    await writer.stop()

    assert store.fetch_metadata(CHAT, 10)["topic_id"] == 5
    assert store.fetch_metadata(CHAT, 10)["topic_title"] == "Работа"
    assert store.fetch_metadata(CHAT, 11)["topic_title"] == "Офферы"
    # General gets id 1 and the fixed General title.
    assert store.fetch_metadata(CHAT, 12)["topic_id"] == 1
    assert store.fetch_metadata(CHAT, 12)["topic_title"] == "General"
    # Title unavailable: message is still stored, title stays empty.
    assert store.fetch_metadata(CHAT, 13)["topic_id"] == 7
    assert store.fetch_metadata(CHAT, 13)["topic_title"] is None
    assert store.count() == 4
    store.close()


async def test_live_handler_stores_topic_metadata(tmp_path):
    store, writer = make_store_and_writer(tmp_path)
    await writer.start()

    entity = FakeChannelEntity()
    client = FakeForumClient(entity, [], topics=[(5, "Работа")])
    resolver = TopicResolver(client)

    await handle_new_message(
        FakeMessage(20, topic_id=5, text="нужна работа"),
        writer=writer,
        chat_entity=entity,
        topic_resolver=resolver,
    )
    await writer.flush()
    await writer.stop()

    metadata = store.fetch_metadata(CHAT, 20)
    assert metadata["topic_id"] == 5
    assert metadata["topic_title"] == "Работа"
    store.close()


async def test_live_and_history_produce_consistent_metadata(tmp_path):
    store, writer = make_store_and_writer(tmp_path)
    await writer.start()

    entity = FakeChannelEntity()
    message = FakeMessage(30, topic_id=6, text="consistent")
    client = FakeForumClient(entity, [message], topics=[(6, "Офферы")])
    resolver = TopicResolver(client)

    await sync_history(
        client, [CHAT], store, writer, entities_by_chat_id={CHAT: entity}, topic_resolver=resolver
    )
    # Re-deliver the same message through the live handler: no duplicate, and the
    # stored metadata stays consistent.
    await handle_new_message(
        message, writer=writer, chat_entity=entity, topic_resolver=resolver
    )
    await writer.flush()
    await writer.stop()

    assert store.count() == 1
    metadata = store.fetch_metadata(CHAT, 30)
    assert metadata["topic_id"] == 6
    assert metadata["topic_title"] == "Офферы"
    store.close()


async def test_topic_titles_are_cached_across_messages(tmp_path):
    store, writer = make_store_and_writer(tmp_path)
    await writer.start()

    entity = FakeChannelEntity()
    client = FakeForumClient(entity, [], topics=[(5, "Работа")])
    resolver = TopicResolver(client)

    for message_id in range(100, 105):
        await handle_new_message(
            FakeMessage(message_id, topic_id=5, text="x"),
            writer=writer,
            chat_entity=entity,
            topic_resolver=resolver,
        )
    await writer.flush()
    await writer.stop()

    # One forum-topics fetch for all messages of the chat.
    assert client.topic_calls == 1
    store.close()
