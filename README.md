# Telegram Referral Lead Finder

A local tool that (in later stages) will monitor selected Telegram chats via
the official MTProto API (Telethon) and store new messages locally so they can
be reviewed for courier-job interest.

**Current status: stages 3–8 — SQLite storage, deduplication, pipeline
robustness, bounded startup history reconciliation, forum-topic support, and a
rule-based message classifier (offline backfill + optional online hook).**
New incoming text messages are persisted to SQLite through a bounded queue with
a single writer; repeated delivery of the same `(chat_id, message_id)` is
ignored. Messages without text or a caption are skipped. On startup the client
reconciles recent history for each configured chat to close gaps left while it
was offline. Chats can be configured by numeric id or by an exact supergroup
title (`title:`), and messages from forum supergroups store the topic id/title,
sender information and (when reliably derivable) the message link and a content
protection flag. Writer failures, DB-write errors, backpressure, lifecycle,
reconciliation and topic resolution are covered by offline integration tests
against temporary databases. No live checks are run here: a **real Telegram
connection is not run by the test suite**. The message text is never written to
the logs.

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
  storage.py           SQLite store for messages + deduplication + checkpoints
  ingest.py            bounded queue + single writer pipeline
  history.py           bounded startup history reconciliation
  topics.py            forum topic title resolution and caching
  classifier.py        rule-based message classifier (pure)
  classify.py          post-persist service + offline backfill CLI
  telegram_client.py   Telethon client, incoming handler, lifecycle
tests/unit/        unit + integration tests (no Telegram, no network)
tests/data/        synthetic classifier examples (no real messages)
```

### Message storage

The `messages` table has the composite primary key `(chat_id, message_id)`.
Writes use `INSERT ... ON CONFLICT(chat_id, message_id) DO NOTHING`, so a
message delivered twice is stored once and its original row is not overwritten.
`date_utc` comes from the Telegram message and `received_at_utc` is set by the
application when the event is handled; both are Unix timestamps in UTC.

Additional nullable columns store the forum topic (`topic_id`, `topic_title`),
the author (`sender_id`, `sender_username`, `sender_first_name`,
`sender_last_name`, `sender_display_name`), `message_url` and `is_protected`
(`1`/`0`, `NULL` when unknown). For non-forum chats these stay `NULL`. Existing
databases are migrated in place with `ALTER TABLE ADD COLUMN` (no data loss, the
primary key is unchanged). On a repeated message only currently-`NULL` metadata
is filled in; existing non-NULL values, the text and the timestamps are never
overwritten. Old rows keep `NULL` metadata until they are re-processed.

`sender_id` is the primary author identifier. For channel/anonymous senders no
user names are invented (the real channel title or `post_author` may be used as
a display name). `message_url` is only produced when it can be derived reliably
(public username links or the documented `t.me/c/` form for private
channels/supergroups). The protection flag combines message- and chat-level
`noforwards`, so a single message without the flag is never treated as proof that
the chat is unprotected.

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

### Startup history reconciliation

`client.catch_up()` is not a delivery guarantee, so on startup (after the client
connects and catches up) the app scans the configured chats' history and fills
gaps through the same single-writer pipeline. Deduplication by
`(chat_id, message_id)` makes overlap with live events harmless.

- **First scan of a chat** (no checkpoint yet) is bounded to the most recent
  `200` messages. This finds small gaps (for example a stored id 28 with 26 and
  27 missing) without importing the whole archive, and is why `MAX(message_id)`
  is *not* used as a checkpoint.
- **Later scans** read everything after the stored checkpoint, paginated, so a
  long offline period is not silently truncated.
- The checkpoint is stored **per chat** in the `chat_sync_state` table of the
  same SQLite database (created on existing databases too, without data loss)
  and is advanced only after a page has been fully persisted.
- Pages are written as a single batched transaction; on any fetch or write
  error the checkpoint stays at the last fully processed page, so no message is
  skipped. Errors are logged (chat id and range/counts only, never text).

This is a one-shot startup scan, not a periodic poller, and it never downloads
media.

### Message classification

A pure, deterministic rule-based classifier (`classifier.py`) labels each stored
message into one of six categories:

| Category | Meaning |
|---|---|
| `referral_request` | asks for a referral link/code |
| `job_interest` | looking for a courier job / asking about conditions |
| `referral_offer` | shares/gives a referral link or code |
| `recruitment_offer` | recruits couriers / offers the job |
| `irrelevant` | not related to the target task |
| `uncertain` | not enough (or conflicting) evidence |

`referral_offer` and `recruitment_offer` are distinct. When strong signals of
different intents tie, the result is `uncertain` rather than a guess. `score` is
a heuristic weight, **not** a calibrated probability.

Results are stored in the separate `message_classifications` table (the `messages`
table is never modified by classification). The primary key is
`(chat_id, message_id, classifier_version)`, so re-processing a version is
idempotent and a new rules version is stored alongside the old one. The active
version is the explicit `CLASSIFIER_VERSION` constant (`classifier.py`), not a
`MAX(version)` query. `matched_rules` stores rule ids as a JSON array; the
message text is never copied into this table or into the logs.

**Candidate queue:** `MessageStore.iter_candidates(version)` returns the active
version's `referral_request` and `job_interest` messages; `referral_offer` /
`recruitment_offer` are available separately (pass explicit `categories`).

**Message classification vs author analytics (future stage).** Classification is
per message. Grouping all messages of one person by the stable `sender_id`,
keeping per-message historical usernames, and showing everything one author
posted is a separate, later feature. This stage only prepares the basis: each
message already stores `sender_id`, `sender_kind` (`user`/`channel`/`chat`/
`anonymous`/`unknown`), and the username/name snapshot from processing time.
Channel/anonymous senders are never treated as people.

Online classification is **off by default**:

| Variable | Required | Description |
|---|---|---|
| `CLASSIFY_ON_INGEST` | no | `true`/`1`/`on` to classify each successfully saved message (default `false`) |

When enabled, classification runs as a post-persist hook inside the single
writer; a classifier error is logged and does **not** stop collection. Backfill
existing rows without Telegram:

```
.\.venv\Scripts\python -m referral_finder.classify --dry-run   # count only
.\.venv\Scripts\python -m referral_finder.classify --limit 100 # trial run
.\.venv\Scripts\python -m referral_finder.classify             # all pending
```

The backfill reads/writes in batches, is resumable, idempotent, uses
`--db PATH` (default `DB_PATH`), and prints only counts/categories (never
message text).

Limitations: this is a transparent keyword/rule classifier (Russian, common
English terms, transliteration). It does not understand arbitrary phrasing and
its `score` is not calibrated. The synthetic examples under `tests/data/` are for
functional and regression testing only; real precision/recall must be measured
on a separate hand-labeled sample and is **not** claimed here.

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

Copy `config/chats.example.txt` to `config/chats.txt` and add one identifier per
line. Blank lines and `#` comments are ignored. The real `config/chats.txt` is
git-ignored; only the example is tracked. Two forms are supported:

- **Numeric id** (recommended), e.g. `-1001234567890`.
- **Exact supergroup title**, prefixed with `title:`, e.g.
  `title:My Courier Community`. The title is matched exactly (after collapsing
  whitespace and case-folding) against the dialogs visible to the account. No
  fuzzy matching: if there is no match, or several chats share the title, the app
  stops with a clear error and asks for a numeric id. The matched chat must be a
  forum supergroup with topics enabled.

Both forms may be mixed in one file. Title resolution needs a connected client,
so it happens right after connecting and before `catch_up()`.

## Run

```
.\.venv\Scripts\python -m referral_finder
```

Without required settings the command exits with a clear message and a non-zero
code (it does not print a traceback).

With a valid `.env` and a non-empty `config/chats.txt`, the command opens the
SQLite database at `DB_PATH`, starts the Telethon client, resolves any
`title:` entries (for those, it connects first), registers the incoming-message
handler, runs the standard recovery (`client.catch_up()`), reconciles recent
history for each configured chat, and waits until disconnected. New text
messages are written to SQLite; repeated delivery of the same message is
ignored. Forum supergroups are subscribed by parent chat id (all topics at
once), and each message keeps its topic id/title.

The database (and its parent directory) is created only when the command runs,
i.e. only with a configured `DB_PATH`. Set `DB_PATH` to a path **outside the
repository** (the default `data/messages.db` is git-ignored only as a safety
net).

> **Live connection is not run by the test suite.** Running the command above
> opens a real Telegram connection and may prompt for the login code on the
> first use. That live check is performed separately, after explicit approval.
> First-time login is interactive; subsequent runs reuse the session file.

The handler logs only technical identifiers (`chat_id`, `message_id`,
`date_utc`, `text_length`, `topic_id`, and whether the topic title is known). It
never logs the message text.

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
