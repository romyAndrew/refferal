# Telegram Referral Lead Finder

A local tool that (in later stages) will monitor selected Telegram chats via
the official MTProto API (Telethon) and store new messages locally so they can
be reviewed for courier-job interest.

**Current status: stages 3–4 — SQLite storage, deduplication, and pipeline
robustness.**
New incoming text messages are persisted to SQLite through a bounded queue with
a single writer; repeated delivery of the same `(chat_id, message_id)` is
ignored. Messages without text or a caption are skipped. Writer failures,
DB-write errors, backpressure, and start/stop lifecycle are covered by offline
integration tests against temporary databases. These stages add no live checks:
a **real Telegram connection is not run here**. The message text is never
written to the logs.

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
  __main__.py          entry point (runs the client + storage lifecycle)
  config.py            env + chats.txt loading and validation
  logging_setup.py     logging with secret masking
  models.py            message DTO (Telethon-independent shape)
  storage.py           SQLite store for messages + deduplication
  ingest.py            bounded queue + single writer pipeline
  telegram_client.py   Telethon client, incoming handler, lifecycle
tests/unit/        unit + integration tests (no Telegram, no network)
```

### Message storage

The `messages` table has the composite primary key `(chat_id, message_id)`.
Writes use `INSERT ... ON CONFLICT(chat_id, message_id) DO NOTHING`, so a
message delivered twice is stored once and its original row is not overwritten.
`date_utc` comes from the Telegram message and `received_at_utc` is set by the
application when the event is handled; both are Unix timestamps in UTC.

The inbound handler queues text messages into a bounded `asyncio.Queue`
(default size 1000) drained by a single background writer, so only one
coroutine ever writes to SQLite. Persistence errors are logged (with the
identifiers only, never the text) and are never reported as success.

Shutdown order is deliberate: the Telegram client is **disconnected first**
(Telethon cancels and awaits in-flight handler tasks, so no new message can be
submitted), then the writer is stopped and the already-accepted queue is drained,
and only then is the SQLite connection closed. Messages that were still in a
handler when the client disconnected are dropped and logged (identifiers only).
Each shutdown step is isolated so one failing component does not block the
others.

If the background writer task dies unexpectedly, the failure is recorded and
surfaced: `submit()` raises `MessageWriterError` instead of blocking forever on
a queue nobody drains, `stop()` raises it after the task ends, and the runner
logs it during shutdown. `start()`/`stop()` are re-entrant, so the writer can be
restarted without leaking tasks.

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

With a valid `.env` and a non-empty `config/chats.txt`, the command opens the
SQLite database at `DB_PATH`, starts the Telethon client, registers the
incoming-message handler, runs the standard recovery (`client.catch_up()`), and
waits until disconnected. New text messages are written to SQLite; repeated
delivery of the same message is ignored.

The database (and its parent directory) is created only when the command runs,
i.e. only with a configured `DB_PATH`. Set `DB_PATH` to a path **outside the
repository** (the default `data/messages.db` is git-ignored only as a safety
net).

> **Live connection is not run as part of stage 3.** Running the command above
> opens a real Telegram connection and may prompt for the login code on the
> first use. That live check is performed separately, after explicit approval.
> First-time login is interactive; subsequent runs reuse the session file.

The handler logs only `chat_id`, `message_id`, `date_utc`, and `text_length`.
It never logs the message text.

## Tests

```
.\.venv\Scripts\python -m pytest -q
```

All tests are offline: they use fake environment mappings, fake Telethon events,
and temporary SQLite databases under `tmp_path`. They never read or write the
developer's real `.env`, `config/chats.txt`, session file, or database, and they
require no Telegram credentials.

Live tests (marked `live`) are excluded by default and will only be added and
run on later stages, with explicit approval.

## Security

- Never commit `.env`, session files, the real `config/chats.txt`, or the
  database. `.gitignore` enforces this as a safety net.
- The session file grants full access to the account; keep it outside the
  repository.
- Logs mask known secret values (API hash, session path).
