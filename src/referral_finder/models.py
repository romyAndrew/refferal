"""Data transfer objects.

Defines the storage-ready shape of a message, decoupled from Telethon: the
conversion from a Telethon event to :class:`IncomingMessage` happens in
:mod:`referral_finder.telegram_client`, and mapping the DTO onto SQL rows
happens in :mod:`referral_finder.storage`. These types carry no Telethon imports
so they can be tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class IncomingMessage:
    """Minimal representation of a received message.

    Fields follow the data-minimisation decision: no profile data (names,
    phone numbers, photos) is stored.
    """

    chat_id: int
    message_id: int
    date_utc: int
    received_at_utc: int
    text: Optional[str] = None
    sender_id: Optional[int] = None


def incoming_from_object(obj, *, received_at_utc: int) -> IncomingMessage:
    """Map a Telethon-like object to an :class:`IncomingMessage`.

    Duck-typed on purpose (no Telethon import) so the same rule is used for live
    events and for history reconciliation: text/caption only, ``sender_id`` when
    available, empty text normalised to ``None``. Callers are responsible for
    filtering out outgoing messages.
    """
    text = getattr(obj, "raw_text", None)
    sender_id = getattr(obj, "sender_id", None)
    date = getattr(obj, "date", None)
    date_utc = int(date.timestamp()) if date is not None else 0
    return IncomingMessage(
        chat_id=int(obj.chat_id),
        message_id=int(obj.id),
        date_utc=date_utc,
        received_at_utc=int(received_at_utc),
        text=text if text else None,
        sender_id=int(sender_id) if sender_id is not None else None,
    )
