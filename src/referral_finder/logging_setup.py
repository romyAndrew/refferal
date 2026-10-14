"""Logging setup with secret masking.

A :class:`SecretFilter` replaces known secret values with ``***`` before a
record is emitted, so credentials never end up in the log output.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

LOG_FORMAT = "[%(levelname)s %(asctime)s] %(name)s: %(message)s"


class SecretFilter(logging.Filter):
    """Mask known secret values in log records."""

    def __init__(self, secrets: Iterable[str] = ()):
        super().__init__()
        self._secrets = tuple(secret for secret in secrets if secret)

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._secrets:
            return True
        message = record.getMessage()
        for secret in self._secrets:
            if secret in message:
                message = message.replace(secret, "***")
        record.msg = message
        record.args = ()
        return True


def setup_logging(level: str = "INFO", secrets: Sequence[str] = ()) -> None:
    """Configure root logging to stderr with a secret-masking filter."""
    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(SecretFilter(secrets))
    root.addHandler(handler)
