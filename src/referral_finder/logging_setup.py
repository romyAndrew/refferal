"""Logging setup with secret masking.

A :class:`SecretFilter` replaces known secret values with ``***`` before a
record is emitted, so credentials never end up in the log output.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Sequence

LOG_FORMAT = "[%(levelname)s %(asctime)s] %(name)s: %(message)s"

# Secrets registered before/after logging setup is configured (for example an
# API key read lazily just before its first use). Merged into every handler.
_REGISTERED_SECRETS: List[str] = []


class SecretFilter(logging.Filter):
    """Mask known secret values in log records."""

    def __init__(self, secrets: Iterable[str] = ()):
        super().__init__()
        self._secrets = tuple(secret for secret in secrets if secret)

    def add_secret(self, secret: str) -> None:
        """Add a secret to mask (idempotent, keeps existing secrets)."""
        if secret and secret not in self._secrets:
            self._secrets = self._secrets + (secret,)

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


def register_secret(secret: str) -> None:
    """Register ``secret`` so it is masked from now on.

    Safe to call before or after :func:`setup_logging`; the value is applied to
    all current handlers and remembered for handlers created later.
    """
    if not secret:
        return
    if secret not in _REGISTERED_SECRETS:
        _REGISTERED_SECRETS.append(secret)
    for handler in logging.getLogger().handlers:
        for log_filter in handler.filters:
            if isinstance(log_filter, SecretFilter):
                log_filter.add_secret(secret)


def setup_logging(level: str = "INFO", secrets: Sequence[str] = ()) -> None:
    """Configure root logging to stderr with a secret-masking filter."""
    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    all_secrets = tuple(secrets) + tuple(_REGISTERED_SECRETS)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(SecretFilter(all_secrets))
    root.addHandler(handler)
