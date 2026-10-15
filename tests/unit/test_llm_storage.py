"""Tests for AI-analysis storage (temporary DBs only, no message changes)."""

import sqlite3

from referral_finder.classifier import JOB_INTEREST, UNCERTAIN
from referral_finder.llm import AnalysisResult, ai_analysis_row
from referral_finder.models import IncomingMessage
from referral_finder.storage import MessageStore

PROVIDER = "deepseek"
MODEL = "deepseek-flash"
PROMPT_VERSION = "prompt-test"
PROMPT_HASH = "hash-test"


def make_message(message_id, text="hello"):
    return IncomingMessage(
        chat_id=-1,
        message_id=message_id,
        date_utc=message_id,
        received_at_utc=message_id,
        text=text,
        sender_id=7,
    )


def row(message_id, result, *, version=PROMPT_VERSION, status="ok", error=None):
    return ai_analysis_row(
        -1,
        message_id,
        result,
        provider=PROVIDER,
        model=MODEL,
        prompt_version=version,
        prompt_hash=PROMPT_HASH,
        analyzed_at_utc=1,
        status=status,
        error=error,
    )


def test_ai_table_exists(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        found = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='message_ai_analysis'"
        ).fetchone()
        assert found == ("message_ai_analysis",)


def test_save_is_idempotent_per_model_and_prompt(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        store.save(make_message(1))
        result = AnalysisResult(JOB_INTEREST, True, 0.9, "first")
        assert store.save_ai_analysis([row(1, result)]) == 1
        # Same key: ignored, the first result is kept.
        other = AnalysisResult(UNCERTAIN, None, 0.1, "second")
        assert store.save_ai_analysis([row(1, other)]) == 0
        stored = store.get_ai_analysis(-1, 1, PROVIDER, MODEL, PROMPT_VERSION)
        assert stored["category"] == JOB_INTEREST
        assert stored["explanation"] == "first"


def test_prompt_versions_are_stored_separately(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        store.save(make_message(1))
        store.save_ai_analysis([row(1, AnalysisResult("irrelevant", False, 0.0, "a"), version="v1")])
        store.save_ai_analysis([row(1, AnalysisResult(JOB_INTEREST, True, 0.9, "b"), version="v2")])
        assert store.get_ai_analysis(-1, 1, PROVIDER, MODEL, "v1")["category"] == "irrelevant"
        assert store.get_ai_analysis(-1, 1, PROVIDER, MODEL, "v2")["category"] == JOB_INTEREST


def test_uncertain_relevant_stays_null(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        store.save(make_message(1))
        store.save_ai_analysis([row(1, AnalysisResult(UNCERTAIN, None, 0.2, "?"))])
        stored = store.get_ai_analysis(-1, 1, PROVIDER, MODEL, PROMPT_VERSION)
        assert stored["category"] == UNCERTAIN
        assert stored["relevant"] is None


def test_error_row_has_no_analysis_fields(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        store.save(make_message(1))
        store.save_ai_analysis([row(1, None, status="error", error="timeout")])
        stored = store.get_ai_analysis(-1, 1, PROVIDER, MODEL, PROMPT_VERSION)
        assert stored["status"] == "error"
        assert stored["category"] is None
        assert stored["error"] == "timeout"


def test_get_missing_returns_none(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        assert store.get_ai_analysis(-1, 99, PROVIDER, MODEL, PROMPT_VERSION) is None


def test_ai_analysis_does_not_modify_source_message(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        store.save(make_message(1, text="original"))
        store.save_ai_analysis([row(1, AnalysisResult(JOB_INTEREST, True, 0.5, "x"))])
        assert store.fetch(-1, 1)[4] == "original"
        assert store.count() == 1


def test_schema_migrates_existing_database(tmp_path):
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE messages ("
        "chat_id INTEGER NOT NULL, message_id INTEGER NOT NULL, "
        "date_utc INTEGER NOT NULL, received_at_utc INTEGER NOT NULL, "
        "text TEXT, sender_id INTEGER, PRIMARY KEY (chat_id, message_id))"
    )
    conn.execute("INSERT INTO messages VALUES (-1, 5, 1, 1, 'old text', 9)")
    conn.commit()
    conn.close()

    with MessageStore(db_path) as store:
        found = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='message_ai_analysis'"
        ).fetchone()
        assert found == ("message_ai_analysis",)
        assert store.count() == 1  # existing data preserved


# --- retry policy ----------------------------------------------------------


def test_upsert_increments_attempts_on_failure(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        store.save(make_message(1))
        store.upsert_ai_analysis([row(1, None, status="error", error="timeout")])
        store.upsert_ai_analysis([row(1, None, status="error", error="timeout")])
        stored = store.get_ai_analysis(-1, 1, PROVIDER, MODEL, PROMPT_VERSION)
        assert stored["status"] == "error"
        assert stored["attempts"] == 2


def test_upsert_never_overwrites_a_success(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        store.save(make_message(1))
        store.upsert_ai_analysis([row(1, AnalysisResult(JOB_INTEREST, True, 0.9, "ok"))])
        store.upsert_ai_analysis([row(1, None, status="error", error="boom")])
        stored = store.get_ai_analysis(-1, 1, PROVIDER, MODEL, PROMPT_VERSION)
        assert stored["status"] == "ok"
        assert stored["attempts"] == 1


def test_iter_ai_unprocessed_skips_success_and_retries_failed(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        store.save(make_message(1))
        store.save(make_message(2))
        store.save(make_message(3))
        store.upsert_ai_analysis([row(1, AnalysisResult(JOB_INTEREST, True, 0.9, "ok"))])
        store.upsert_ai_analysis([row(2, None, status="error", error="x")])

        def ids(max_attempts):
            batches = list(
                store.iter_ai_unprocessed(
                    PROVIDER, MODEL, PROMPT_VERSION, max_attempts=max_attempts
                )
            )
            return sorted(mid for batch in batches for _, mid, _ in batch)

        assert ids(max_attempts=3) == [2, 3]  # success skipped, failure retryable
        assert ids(max_attempts=1) == [3]  # failure exhausted, not retried


def test_count_ai_unprocessed_respects_max_attempts(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        store.save(make_message(1))
        store.upsert_ai_analysis([row(1, None, status="error", error="x")])
        assert store.count_ai_unprocessed(PROVIDER, MODEL, PROMPT_VERSION, max_attempts=3) == 1
        assert store.count_ai_unprocessed(PROVIDER, MODEL, PROMPT_VERSION, max_attempts=1) == 0


def test_migration_adds_attempts_to_existing_ai_table(tmp_path):
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE messages ("
        "chat_id INTEGER NOT NULL, message_id INTEGER NOT NULL, "
        "date_utc INTEGER NOT NULL, received_at_utc INTEGER NOT NULL, "
        "text TEXT, sender_id INTEGER, PRIMARY KEY (chat_id, message_id))"
    )
    conn.execute(
        "CREATE TABLE message_ai_analysis ("
        "chat_id INTEGER NOT NULL, message_id INTEGER NOT NULL, provider TEXT NOT NULL, "
        "model TEXT NOT NULL, prompt_version TEXT NOT NULL, prompt_hash TEXT NOT NULL, "
        "status TEXT NOT NULL, relevant INTEGER, category TEXT, confidence REAL, "
        "explanation TEXT, error TEXT, analyzed_at_utc INTEGER NOT NULL, "
        "PRIMARY KEY (chat_id, message_id, provider, model, prompt_version))"
    )
    conn.execute("INSERT INTO messages VALUES (-1, 5, 1, 1, 'old text', 9)")
    conn.execute(
        "INSERT INTO message_ai_analysis VALUES "
        "(-1, 5, 'deepseek', 'deepseek-flash', 'v1', 'h', 'ok', 1, 'job_interest', "
        "0.5, 'e', NULL, 1)"
    )
    conn.commit()
    conn.close()

    with MessageStore(db_path) as store:
        columns = {
            r[1] for r in store._conn.execute("PRAGMA table_info(message_ai_analysis)")
        }
        assert "attempts" in columns
        stored = store.get_ai_analysis(-1, 5, "deepseek", "deepseek-flash", "v1")
        assert stored["status"] == "ok"  # existing row preserved
        assert stored["attempts"] == 0  # backfilled default
