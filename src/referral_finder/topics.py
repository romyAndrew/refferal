"""Forum topic metadata resolution and caching.

Telegram forums (supergroups with topics) expose topics through
``messages.getForumTopics``. The General topic is implicit: it has id 1, is not
returned by that call, and its title is not localized by the API. A message in a
non-General topic carries a ``MessageReplyHeader`` with ``forum_topic=True``
(this is handled in :mod:`referral_finder.models`).

This resolver fetches topic titles once per chat and caches them. If an unknown
topic id appears later (a newly created or renamed topic), it refreshes at most
once per chat and per process. There is no periodic polling and no per-message
request. A fetch failure never blocks message storage: the caller keeps the
known ``topic_id`` and stores an empty title.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Set

from telethon.tl.functions.messages import GetForumTopicsRequest

from .models import GENERAL_TOPIC_ID, GENERAL_TOPIC_TITLE

logger = logging.getLogger(__name__)

_TOPICS_PAGE_SIZE = 100


class TopicResolver:
    """Resolve ``topic_id`` -> ``topic_title`` with a per-chat cache."""

    def __init__(self, client):
        self._client = client
        self._titles: Dict[int, Dict[int, Optional[str]]] = {}
        self._peers: Dict[int, object] = {}
        self._loaded: Set[int] = set()
        self._refreshed: Set[int] = set()

    async def title_for(
        self, chat_id: int, topic_id: Optional[int], entity=None
    ) -> Optional[str]:
        """Return the topic title, or ``None`` if it cannot be resolved.

        ``None`` is a valid stored value (e.g. ``title_missing``), so callers
        must treat "unknown" and "known to be empty" the same way: keep the
        message, leave the title empty.
        """
        if topic_id is None:
            return None
        chat_id = int(chat_id)
        topic_id = int(topic_id)
        if topic_id == GENERAL_TOPIC_ID:
            return GENERAL_TOPIC_TITLE

        cache = self._titles.setdefault(chat_id, {})
        if topic_id in cache:
            return cache[topic_id]

        await self._load(chat_id, entity)
        if topic_id in cache:
            return cache[topic_id]

        # Unknown after the first load: a new or renamed topic may have appeared
        # during this run. Refresh once per chat (never per message).
        if chat_id not in self._refreshed:
            self._refreshed.add(chat_id)
            await self._load(chat_id, entity, force=True)
        return cache.get(topic_id)

    async def _peer_for(self, chat_id: int, entity):
        peer = self._peers.get(chat_id)
        if peer is not None:
            return peer
        target = entity if entity is not None else chat_id
        peer = await self._client.get_input_entity(target)
        self._peers[chat_id] = peer
        return peer

    async def _load(self, chat_id: int, entity=None, force: bool = False) -> None:
        if chat_id in self._loaded and not force:
            return
        try:
            peer = await self._peer_for(chat_id, entity)
            cache = self._titles.setdefault(chat_id, {})
            offset_topic = 0
            while True:
                result = await self._client(
                    GetForumTopicsRequest(
                        peer=peer,
                        offset_date=0,
                        offset_id=0,
                        offset_topic=offset_topic,
                        limit=_TOPICS_PAGE_SIZE,
                        q=None,
                    )
                )
                topics = list(getattr(result, "topics", []) or [])
                for topic in topics:
                    topic_id = getattr(topic, "id", None)
                    if topic_id is None:
                        continue
                    if getattr(topic, "title_missing", False):
                        cache[int(topic_id)] = None
                    else:
                        cache[int(topic_id)] = getattr(topic, "title", None)
                if len(topics) < _TOPICS_PAGE_SIZE:
                    break
                offset_topic = int(topics[-1].id)
            logger.info(
                "Resolved %d forum topic title(s) for chat_id=%s",
                len(cache),
                chat_id,
            )
        except Exception:
            # Never fail the message pipeline because topic metadata is missing.
            logger.exception("Failed to fetch forum topics chat_id=%s", chat_id)
        finally:
            # Avoid a request per message when topic fetch keeps failing.
            self._loaded.add(chat_id)
