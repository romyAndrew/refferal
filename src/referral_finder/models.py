"""Data transfer objects.

Defines the storage-ready shape of a message, decoupled from Telethon: the
conversion from a Telethon event to :class:`IncomingMessage` happens here (via
duck typing, no Telethon import), and mapping the DTO onto SQL rows happens in
:mod:`referral_finder.storage`. Keeping the mapping here means live events and
history reconciliation always apply the same rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Optional

logger = logging.getLogger(__name__)

# Telegram's General topic is implicit: it is not returned by getForumTopics and
# its messages have no ``forum_topic`` reply header. Its topic id is 1.
GENERAL_TOPIC_ID = 1
GENERAL_TOPIC_TITLE = "General"


@dataclass(frozen=True)
class IncomingMessage:
    """Minimal representation of a received message.

    Fields follow the data-minimisation decision: no phone numbers or other
    private profile data are stored. New metadata fields are optional and stay
    ``None`` for plain (non-forum) chats or when Telegram does not expose them.
    """

    chat_id: int
    message_id: int
    date_utc: int
    received_at_utc: int
    text: Optional[str] = None
    sender_id: Optional[int] = None
    topic_id: Optional[int] = None
    topic_title: Optional[str] = None
    sender_username: Optional[str] = None
    sender_first_name: Optional[str] = None
    sender_last_name: Optional[str] = None
    sender_display_name: Optional[str] = None
    message_url: Optional[str] = None
    is_protected: Optional[int] = None


def _sender_fields(obj) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Return (username, first_name, last_name, display_name) from Telegram data.

    Uses only the already-available ``message.sender`` object (no extra network
    request). For channel/anonymous senders no user name is invented; the
    channel title or ``post_author`` is used only as a display name.
    """
    sender = getattr(obj, "sender", None)
    if sender is None:
        return (None, None, None, getattr(obj, "post_author", None) or None)

    username = getattr(sender, "username", None)
    first_name = getattr(sender, "first_name", None)
    last_name = getattr(sender, "last_name", None)
    if first_name or last_name:
        display = " ".join(part for part in (first_name, last_name) if part) or None
    else:
        # Channel/Chat sender: real title, but no user first/last names.
        display = getattr(sender, "title", None)
        first_name = None
        last_name = None
    return (username, first_name, last_name, display)


def extract_topic_id(obj, chat_entity=None) -> Optional[int]:
    """Resolve the forum topic id from a Telethon-like message.

    Official semantics: a message in a non-General topic carries a
    ``MessageReplyHeader`` with ``forum_topic=True`` and ``reply_to_top_id``
    (the topic root). A topic-creation service message uses its own id. For a
    forum chat, a message without such a header belongs to General (id 1).
    Non-forum chats yield ``None``.
    """
    reply = getattr(obj, "reply_to", None)
    if reply is not None and getattr(reply, "forum_topic", False):
        topic_id = getattr(reply, "reply_to_top_id", None) or getattr(
            reply, "reply_to_msg_id", None
        )
        if topic_id is not None:
            return int(topic_id)

    action = getattr(obj, "action", None)
    if action is not None and type(action).__name__ == "MessageActionTopicCreate":
        message_id = getattr(obj, "id", None)
        return int(message_id) if message_id is not None else None

    if chat_entity is not None and getattr(chat_entity, "forum", False):
        return GENERAL_TOPIC_ID
    return None


def _internal_channel_id(marked_id: int) -> Optional[int]:
    """Convert a Telethon marked channel id (-100...) to the t.me/c internal id."""
    if marked_id is None or marked_id >= 0:
        return None
    digits = str(-marked_id)
    if not digits.startswith("100") or len(digits) <= 3:
        return None
    return int(digits[3:])


def build_message_url(obj, chat_entity, topic_id: Optional[int]) -> Optional[str]:
    """Build a reliable message link, or ``None`` when it cannot be formed.

    Public channels/supergroups use the username form; private channels and
    supergroups use the documented ``t.me/c/<internal>/<msg>`` form for members.
    Basic groups and users yield ``None`` (no fabricated links).
    """
    if chat_entity is None:
        return None
    message_id = getattr(obj, "id", None)
    if message_id is None:
        return None

    username = getattr(chat_entity, "username", None)
    is_channel_like = hasattr(chat_entity, "broadcast") or hasattr(
        chat_entity, "megagroup"
    )
    if username:
        base = f"https://t.me/{username}"
    elif is_channel_like:
        internal = _internal_channel_id(getattr(chat_entity, "id", None))
        if internal is None:
            return None
        base = f"https://t.me/c/{internal}"
    else:
        return None

    if (
        getattr(chat_entity, "forum", False)
        and topic_id is not None
        and topic_id != GENERAL_TOPIC_ID
    ):
        return f"{base}/{topic_id}/{message_id}"
    return f"{base}/{message_id}"


def compute_protected(obj, chat_entity) -> Optional[int]:
    """Return 1 if content is protected, 0 if known not protected, else None.

    Combines message-level ``noforwards`` with chat-level ``noforwards``: the
    absence of the flag on a single message is not proof that the whole chat is
    unprotected, so either flag marks the message as protected.
    """
    message_flag = getattr(obj, "noforwards", None)
    chat_flag = getattr(chat_entity, "noforwards", None) if chat_entity is not None else None
    if message_flag is True or chat_flag is True:
        return 1
    # Only an explicit chat-level "not protected" allows concluding 0. A message
    # without the flag (False/None) is not proof about the whole chat.
    if chat_flag is False and message_flag is not True:
        return 0
    return None


def incoming_from_object(
    obj, *, received_at_utc: int, chat_entity=None
) -> IncomingMessage:
    """Map a Telethon-like object to an :class:`IncomingMessage`.

    Duck-typed on purpose (no Telethon import). Live events and history
    reconciliation both call this, so message text, sender info, topic id,
    message URL and the protection flag are extracted identically.
    """
    text = getattr(obj, "raw_text", None)
    sender_id = getattr(obj, "sender_id", None)
    date = getattr(obj, "date", None)
    date_utc = int(date.timestamp()) if date is not None else 0

    topic_id = extract_topic_id(obj, chat_entity)
    username, first_name, last_name, display_name = _sender_fields(obj)

    return IncomingMessage(
        chat_id=int(obj.chat_id),
        message_id=int(obj.id),
        date_utc=date_utc,
        received_at_utc=int(received_at_utc),
        text=text if text else None,
        sender_id=int(sender_id) if sender_id is not None else None,
        topic_id=topic_id,
        topic_title=None,
        sender_username=username,
        sender_first_name=first_name,
        sender_last_name=last_name,
        sender_display_name=display_name,
        message_url=build_message_url(obj, chat_entity, topic_id),
        is_protected=compute_protected(obj, chat_entity),
    )


async def build_incoming(
    obj, *, received_at_utc: int, chat_entity=None, topic_resolver=None
) -> IncomingMessage:
    """Like :func:`incoming_from_object`, but also resolves the topic title.

    The title comes from a (cached) :class:`~referral_finder.topics.TopicResolver`;
    if it cannot be resolved, the message is still returned with the known
    ``topic_id`` and an empty ``topic_title``.
    """
    message = incoming_from_object(
        obj, received_at_utc=received_at_utc, chat_entity=chat_entity
    )
    if message.topic_id is not None and topic_resolver is not None:
        try:
            title = await topic_resolver.title_for(
                message.chat_id, message.topic_id, chat_entity
            )
        except Exception:
            # A missing title must never drop or fail the message.
            logger.debug(
                "Topic title resolution failed chat_id=%s topic_id=%s",
                message.chat_id,
                message.topic_id,
                exc_info=True,
            )
            title = None
        if title is not None:
            message = replace(message, topic_title=title)
    return message
