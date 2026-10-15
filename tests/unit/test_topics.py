"""Tests for forum topic title resolution and caching."""

from types import SimpleNamespace

from referral_finder.models import GENERAL_TOPIC_ID, GENERAL_TOPIC_TITLE
from referral_finder.topics import TopicResolver


class FakeForumClient:
    """Fake Telethon client for ``messages.getForumTopics``.

    ``topics_by_chat`` maps chat_id -> list of ``(id, title)`` or
    ``(id, title, title_missing)``. Pagination follows ``offset_topic``.
    """

    def __init__(self, topics_by_chat, fail=False):
        self._topics = {int(k): list(v) for k, v in topics_by_chat.items()}
        self._fail = fail
        self.calls = 0

    async def get_input_entity(self, target):
        return target

    async def __call__(self, request):
        self.calls += 1
        if self._fail:
            raise RuntimeError("topic fetch failed")
        chat_id = int(request.peer)
        all_topics = self._topics.get(chat_id, [])
        start = 0
        if request.offset_topic:
            ids = [t[0] for t in all_topics]
            start = ids.index(request.offset_topic) + 1
        page = all_topics[start:start + request.limit]
        objs = [
            SimpleNamespace(
                id=t[0], title=t[1], title_missing=(len(t) > 2 and t[2])
            )
            for t in page
        ]
        return SimpleNamespace(topics=objs)


async def test_general_topic_has_fixed_title_without_fetch():
    client = FakeForumClient({-1001: [(5, "Work")]})
    resolver = TopicResolver(client)
    assert await resolver.title_for(-1001, GENERAL_TOPIC_ID) == GENERAL_TOPIC_TITLE
    assert client.calls == 0


async def test_titles_are_resolved_and_cached():
    client = FakeForumClient({-1001: [(5, "Работа"), (6, "Офферы")]})
    resolver = TopicResolver(client)

    assert await resolver.title_for(-1001, 5) == "Работа"
    assert await resolver.title_for(-1001, 6) == "Офферы"
    assert client.calls == 1  # one fetch for all topics


async def test_title_missing_is_reported_as_none():
    client = FakeForumClient({-1001: [(5, None, True)]})
    resolver = TopicResolver(client)
    assert await resolver.title_for(-1001, 5) is None


async def test_unknown_topic_triggers_single_refresh_then_stops():
    client = FakeForumClient({-1001: [(5, "Работа")]})
    resolver = TopicResolver(client)

    assert await resolver.title_for(-1001, 99) is None
    assert client.calls == 2  # initial load + one refresh
    assert await resolver.title_for(-1001, 99) is None
    assert client.calls == 2  # no further requests per message


async def test_refresh_picks_up_new_topic():
    class GrowingClient(FakeForumClient):
        async def __call__(self, request):
            if self.calls == 1:
                self._topics.setdefault(-1001, []).append((99, "Новая"))
            return await super().__call__(request)

    client = GrowingClient({-1001: [(5, "Работа")]})
    resolver = TopicResolver(client)

    assert await resolver.title_for(-1001, 99) == "Новая"
    assert client.calls == 2


async def test_fetch_failure_is_swallowed_and_not_retried_per_message():
    client = FakeForumClient({-1001: [(5, "Работа")]}, fail=True)
    resolver = TopicResolver(client)

    assert await resolver.title_for(-1001, 5) is None
    calls_after_first = client.calls
    assert await resolver.title_for(-1001, 5) is None
    assert client.calls == calls_after_first  # no per-message retry storm


async def test_large_topic_list_is_paginated():
    topics = [(i, f"t{i}") for i in range(1, 151)]
    client = FakeForumClient({-1001: topics})
    resolver = TopicResolver(client)

    assert await resolver.title_for(-1001, 150) == "t150"
    assert client.calls == 2  # 100 + 50
