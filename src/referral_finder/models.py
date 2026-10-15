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
