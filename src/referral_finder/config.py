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
from typing import Iterable, Mapping, Optional, Tuple

try:  # python-dotenv is a declared dependency; guard keeps imports robust.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


DEFAULT_DB_PATH = "data/messages.db"
DEFAULT_CHATS_FILE = "config/chats.txt"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_CLASSIFY_ON_INGEST = False

# LLM (DeepSeek) integration. Offline by default: online calls must be enabled
# explicitly. The API key is read from the environment only and never logged.
DEFAULT_LLM_ENABLED = False
DEFAULT_LLM_PROVIDER = "deepseek"
DEFAULT_LLM_MODEL = "deepseek-flash"
DEFAULT_LLM_BASE_URL = "https://api.deepseek.com"
DEFAULT_LLM_TIMEOUT = 30.0
DEFAULT_LLM_MAX_RETRIES = 2
DEFAULT_LLM_MAX_INPUT_CHARS = 4000
DEFAULT_LLM_MAX_OUTPUT_TOKENS = 400

REQUIRED_VARS = ("TG_API_ID", "TG_API_HASH", "TG_SESSION_PATH")

TITLE_PREFIX = "title:"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ConfigError(f"Invalid boolean value: {value!r}")


def _as_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid integer value: {value!r}") from exc


def _as_float(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid number value: {value!r}") from exc


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class ChatEntries:
    """Configured chats split into numeric ids and exact-title entries."""

    ids: Tuple[int, ...]
    titles: Tuple[str, ...]

    @property
    def has_titles(self) -> bool:
        return bool(self.titles)


def parse_chat_entries(chats: Iterable[str]) -> ChatEntries:
    """Parse ``config/chats.txt`` entries.

    Each entry is either a numeric chat id or ``title:<exact supergroup name>``.
    Non-numeric entries without the ``title:`` prefix are rejected.
    """
    ids = []
    titles = []
    for raw in chats:
        entry = raw.strip()
        if not entry:
            raise ConfigError(
                "Invalid chat entry in config/chats.txt: empty line "
                "(numeric ID or 'title:Name' expected)"
            )
        if entry.lower().startswith(TITLE_PREFIX):
            title = entry[len(TITLE_PREFIX):].strip()
            if not title:
                raise ConfigError(
                    "Invalid chat entry in config/chats.txt: 'title:' without a name"
                )
            titles.append(title)
        else:
            try:
                ids.append(int(entry))
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"Invalid chat id in config/chats.txt: {raw!r} "
                    "(numeric ID or 'title:Name' expected)"
                ) from exc
    return ChatEntries(ids=tuple(ids), titles=tuple(titles))


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
    classify_on_ingest: bool = DEFAULT_CLASSIFY_ON_INGEST
    llm_enabled: bool = DEFAULT_LLM_ENABLED
    llm_provider: str = DEFAULT_LLM_PROVIDER
    llm_model: str = DEFAULT_LLM_MODEL
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_api_key: Optional[str] = None
    llm_timeout: float = DEFAULT_LLM_TIMEOUT
    llm_max_retries: int = DEFAULT_LLM_MAX_RETRIES
    llm_max_input_chars: int = DEFAULT_LLM_MAX_INPUT_CHARS
    llm_max_output_tokens: int = DEFAULT_LLM_MAX_OUTPUT_TOKENS

    def safe_repr(self) -> str:
        """Representation that never exposes the API hash or the LLM key."""
        key_state = "'***'" if self.llm_api_key else "None"
        return (
            "Config("
            f"api_id={self.api_id}, "
            "api_hash='***', "
            f"session_path='{self.session_path}', "
            f"db_path='{self.db_path}', "
            f"log_level='{self.log_level}', "
            f"chats_file='{self.chats_file}', "
            f"chats={len(self.chats)} entries, "
            f"classify_on_ingest={self.classify_on_ingest}, "
            f"llm_enabled={self.llm_enabled}, "
            f"llm_provider='{self.llm_provider}', "
            f"llm_model='{self.llm_model}', "
            f"llm_api_key={key_state})"
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
    classify_on_ingest = _as_bool(
        _get(env, "CLASSIFY_ON_INGEST"), DEFAULT_CLASSIFY_ON_INGEST
    )

    llm_enabled = _as_bool(_get(env, "LLM_ENABLED"), DEFAULT_LLM_ENABLED)
    llm_timeout = _as_float(_get(env, "LLM_TIMEOUT"), DEFAULT_LLM_TIMEOUT)
    if llm_timeout <= 0:
        raise ConfigError("LLM_TIMEOUT must be positive")
    llm_max_retries = _as_int(_get(env, "LLM_MAX_RETRIES"), DEFAULT_LLM_MAX_RETRIES)
    if llm_max_retries < 0:
        raise ConfigError("LLM_MAX_RETRIES must not be negative")
    llm_max_input_chars = _as_int(
        _get(env, "LLM_MAX_INPUT_CHARS"), DEFAULT_LLM_MAX_INPUT_CHARS
    )
    if llm_max_input_chars <= 0:
        raise ConfigError("LLM_MAX_INPUT_CHARS must be positive")
    llm_max_output_tokens = _as_int(
        _get(env, "LLM_MAX_OUTPUT_TOKENS"), DEFAULT_LLM_MAX_OUTPUT_TOKENS
    )
    if llm_max_output_tokens <= 0:
        raise ConfigError("LLM_MAX_OUTPUT_TOKENS must be positive")

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
        classify_on_ingest=classify_on_ingest,
        llm_enabled=llm_enabled,
        llm_provider=(_get(env, "LLM_PROVIDER") or DEFAULT_LLM_PROVIDER),
        llm_model=(_get(env, "LLM_MODEL") or DEFAULT_LLM_MODEL),
        llm_base_url=(_get(env, "LLM_BASE_URL") or DEFAULT_LLM_BASE_URL),
        llm_api_key=_get(env, "LLM_API_KEY"),
        llm_timeout=llm_timeout,
        llm_max_retries=llm_max_retries,
        llm_max_input_chars=llm_max_input_chars,
        llm_max_output_tokens=llm_max_output_tokens,
    )
