"""Data transfer objects.

Stage 1 defines the storage-ready shape of a message only. Nothing produces or
persists these objects yet; that happens on later stages. These types carry no
Telethon imports so they can be tested in isolation.
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
