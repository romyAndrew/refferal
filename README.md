# Telegram Referral Lead Finder

A local tool that (in later stages) will monitor selected Telegram chats via
the official MTProto API (Telethon) and store new messages locally so they can
be reviewed for courier-job interest.

**Current status: stage 2 — Telethon client and message handler.**
The client wrapper, the incoming-message handler, and the application lifecycle
are implemented and unit-tested. Messages are **not** stored yet (no SQLite),
and a **real Telegram connection has not been run** — that live check is a
separate, explicitly approved step. The handler only emits a minimal technical
log record (chat id, message id, timestamp, text length); the message text is
never logged.

## Requirements

- Python 3.11+ (developed on 3.12)
- Windows + VS Code (local development)
- Telethon `1.45.x` (installed automatically as a project dependency; the
  range is patch-only to keep behavior stable)

## Project layout

```
config/            chat list (example tracked, real file ignored)
src/referral_finder/
  __init__.py
  __main__.py          entry point (runs the client lifecycle)
  config.py            env + chats.txt loading and validation
  logging_setup.py     logging with secret masking
  models.py            message DTO (shape only)
  telegram_client.py   Telethon client, incoming handler, lifecycle
tests/unit/        unit tests (no Telegram, no network)
```

Modules for storage and ingestion are intentionally **not** created yet; they
will be added on their own stages.

## Setup (PowerShell)

```
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

Editable install uses the `src` layout, so `python -m referral_finder` resolves
the package predictably.

## Configuration

Configuration is loaded from environment variables. For local development the
single supported way to load a `.env` file is **python-dotenv** (declared as a
project dependency): `load_config()` calls `load_dotenv()` automatically when
no explicit environment mapping is provided.

1. Copy the example file:
   ```
   Copy-Item .env.example .env
   ```
2. Fill in real values in `.env`. Real credentials are only needed from
   stage 2 (Telegram connection); stage 1 tests and the skeleton run without
   them.

Variables:

| Variable | Required | Description |
|---|---|---|
| `TG_API_ID` | yes | Telegram API id (https://my.telegram.org) |
| `TG_API_HASH` | yes | Telegram API hash |
| `TG_SESSION_PATH` | yes | Session file path, **outside the repository** |
| `DB_PATH` | no | SQLite path (default `data/messages.db`) |
| `LOG_LEVEL` | no | `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL` (default `INFO`) |
| `TG_CHATS_FILE` | no | Chat list path (default `config/chats.txt`) |

### Chat list

Copy `config/chats.example.txt` to `config/chats.txt` and add one chat
identifier per line (numeric IDs preferred). Blank lines and `#` comments are
ignored. The real `config/chats.txt` is git-ignored; only the example is
tracked.

## Run

```
.\.venv\Scripts\python -m referral_finder
```

Without required settings the command exits with a clear message and a non-zero
code (it does not print a traceback).

With a valid `.env` and a non-empty `config/chats.txt`, the command starts the
Telethon client, registers the incoming-message handler, runs the standard
recovery (`client.catch_up()`), and waits until disconnected.

> **Live connection is not run as part of stage 2.** Running the command above
> opens a real Telegram connection and may prompt for the login code on the
> first use. That live check is performed separately, after explicit approval.
> First-time login is interactive; subsequent runs reuse the session file.

The handler logs only `chat_id`, `message_id`, `date_utc`, and `text_length`.
It never logs the message text, and it does not store anything yet.

## Tests

```
.\.venv\Scripts\python -m pytest -q
```

Unit tests use fake environment mappings and temporary files; they never read
real local settings and require no Telegram credentials.

Live tests (marked `live`) are excluded by default and will only be added and
run on later stages, with explicit approval.

## Security

- Never commit `.env`, session files, the real `config/chats.txt`, or the
  database. `.gitignore` enforces this as a safety net.
- The session file grants full access to the account; keep it outside the
  repository.
- Logs mask known secret values (API hash, session path).
