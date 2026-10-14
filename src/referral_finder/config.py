"""Application configuration.

Loads settings from environment variables (optionally from a local ``.env``
file via python-dotenv) and from a plain-text chat list. No Telegram or network
access happens here.

Stage 1 note: the required Telegram variables are validated, but nothing is
connected. Real credentials are only needed from stage 2 onward.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

try:  # python-dotenv is a declared dependency; guard keeps imports robust.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


DEFAULT_DB_PATH = "data/messages.db"
DEFAULT_CHATS_FILE = "config/chats.txt"
DEFAULT_LOG_LEVEL = "INFO"

REQUIRED_VARS = ("TG_API_ID", "TG_API_HASH", "TG_SESSION_PATH")


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    """Validated application configuration."""

    api_id: int
    api_hash: str
    session_path: Path
    db_path: Path
    log_level: str
    chats_file: Path
    chats: Tuple[str, ...]

    def safe_repr(self) -> str:
        """Representation that never exposes the API hash."""
        return (
            "Config("
            f"api_id={self.api_id}, "
            "api_hash='***', "
            f"session_path='{self.session_path}', "
            f"db_path='{self.db_path}', "
            f"log_level='{self.log_level}', "
            f"chats_file='{self.chats_file}', "
            f"chats={len(self.chats)} entries)"
        )


def _get(env: Mapping[str, str], key: str) -> Optional[str]:
    value = env.get(key)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _require(env: Mapping[str, str], key: str) -> str:
    value = _get(env, key)
    if value is None:
        raise ConfigError(f"Missing required environment variable: {key}")
    return value


def parse_chats_file(path: Path) -> Tuple[str, ...]:
    """Read chat identifiers, one per line.

    Blank lines and lines starting with ``#`` are ignored. A missing file
    yields an empty tuple (no chats configured yet).
    """
    if not path.exists():
        return ()
    entries = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return tuple(entries)


def load_config(
    env: Optional[Mapping[str, str]] = None,
    *,
    env_file: Optional[str] = None,
    chats_file: Optional[Path] = None,
) -> Config:
    """Build a :class:`Config` from environment variables.

    When ``env`` is ``None`` a local ``.env`` file is loaded (if present) and
    ``os.environ`` is used. Passing an explicit mapping (used by tests) skips
    ``.env`` loading entirely, so tests never touch real local settings.
    """
    if env is None:
        if load_dotenv is not None:
            load_dotenv(env_file)
        env = os.environ

    raw_api_id = _require(env, "TG_API_ID")
    try:
        api_id = int(raw_api_id)
    except ValueError as exc:
        raise ConfigError("TG_API_ID must be an integer") from exc
    if api_id <= 0:
        raise ConfigError("TG_API_ID must be a positive integer")

    api_hash = _require(env, "TG_API_HASH")
    session_path = Path(_require(env, "TG_SESSION_PATH")).expanduser()

    db_path = Path(_get(env, "DB_PATH") or DEFAULT_DB_PATH).expanduser()
    log_level = (_get(env, "LOG_LEVEL") or DEFAULT_LOG_LEVEL).upper()

    if chats_file is not None:
        chats_path = Path(chats_file).expanduser()
    else:
        chats_path = Path(_get(env, "TG_CHATS_FILE") or DEFAULT_CHATS_FILE).expanduser()

    chats = parse_chats_file(chats_path)

    return Config(
        api_id=api_id,
        api_hash=api_hash,
        session_path=session_path,
        db_path=db_path,
        log_level=log_level,
        chats_file=chats_path,
        chats=chats,
    )
