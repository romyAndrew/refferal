"""Tests for bootstrap storage and selection (temporary DBs, no network)."""

import sqlite3

from referral_finder.classifier import JOB_INTEREST
from referral_finder.llm import AnalysisResult, ai_analysis_row
from referral_finder.llm.prompt import PROMPT_VERSION
from referral_finder.models import IncomingMessage
from referral_finder.storage import MessageStore

CHAT_A = -1001
CHAT_B = -1002
PROVIDER = "fake"
MODEL = "fake-model"
UNRELATED = "Сегодня отличная погода, идём гулять"


def seed(store, chat_id, message_id, text="hello"):
    store.save(
        IncomingMessage(
            chat_id=chat_id,
            message_id=message_id,
            date_utc=message_id,
            received_at_utc=message_id,
            text=text,
            sender_id=7,
        )
    )


def ai_ok(chat_id, message_id):
    return ai_analysis_row(
        chat_id,
        message_id,
        AnalysisResult(JOB_INTEREST, True, 0.9, "ok"),
        provider=PROVIDER,
        model=MODEL,
        prompt_version=PROMPT_VERSION,
        prompt_hash="h",
        analyzed_at_utc=1,
        status="ok",
    )


def ai_error(chat_id, message_id):
    return ai_analysis_row(
        chat_id,
        message_id,
        None,
        provider=PROVIDER,
        model=MODEL,
        prompt_version=PROMPT_VERSION,
        prompt_hash="h",
        analyzed_at_utc=1,
        status="error",
        error="server_error",
    )


def ids(rows):
    return sorted(mid for batch in rows for _, mid, _ in batch)


# --- snapshot ---------------------------------------------------------------


def test_snapshot_keeps_last_200_and_sets_high_water_mark(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        for message_id in range(1, 251):
            seed(store, CHAT_A, message_id, text=f"msg {message_id}")

        summary = store.snapshot_bootstrap([CHAT_A], limit=200)[0]

        assert summary["available_count"] == 250
        assert summary["selected_count"] == 200
        assert summary["steady_after_message_id"] == 250
        assert ids(store.iter_ai_bootstrap(PROVIDER, MODEL, PROMPT_VERSION)) == list(
            range(51, 251)
        )


def test_fewer_than_limit_selects_all(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        for message_id in range(1, 6):
            seed(store, CHAT_A, message_id)
        summary = store.snapshot_bootstrap([CHAT_A], limit=200)[0]
        assert summary["available_count"] == 5
        assert summary["selected_count"] == 5
        assert summary["steady_after_message_id"] == 5


def test_chat_without_messages_gets_zero_state(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        summary = store.snapshot_bootstrap([CHAT_A], limit=200)[0]
        assert summary["available_count"] == 0
        assert summary["selected_count"] == 0
        assert summary["steady_after_message_id"] == 0
        state = store.get_bootstrap_state(CHAT_A)
        assert state["status"] == "pending"


def test_empty_chat_message_goes_to_steady_not_bootstrap(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        # 1) empty chat gets a bootstrap state.
        store.snapshot_bootstrap([CHAT_A], limit=200)
        state = store.get_bootstrap_state(CHAT_A)
        # 2) zero counts and a numeric (non-NULL) watermark of 0.
        assert state["available_count"] == 0
        assert state["selected_count"] == 0
        assert state["steady_after_message_id"] == 0

        # 3) a new message must become a steady candidate (0 < 1), not bootstrap.
        seed(store, CHAT_A, 1, text="первое сообщение")
        assert ids(store.iter_ai_bootstrap(PROVIDER, MODEL, PROMPT_VERSION)) == []
        assert ids(
            store.iter_ai_unprocessed(PROVIDER, MODEL, PROMPT_VERSION, steady_only=True)
        ) == [1]


def test_history_boundary_not_weakened_by_empty_chat_fix(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        for message_id in (1, 2, 3):
            seed(store, CHAT_B, message_id)
        store.snapshot_bootstrap([CHAT_B], limit=200)
        assert store.get_bootstrap_state(CHAT_B)["steady_after_message_id"] == 3
        # Old messages (id <= watermark) stay out of steady.
        assert list(
            store.iter_ai_unprocessed(PROVIDER, MODEL, PROMPT_VERSION, steady_only=True)
        ) == []
        seed(store, CHAT_B, 4)
        assert ids(
            store.iter_ai_unprocessed(PROVIDER, MODEL, PROMPT_VERSION, steady_only=True)
        ) == [4]


def test_two_chats_have_independent_state(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        for message_id in range(1, 8):
            seed(store, CHAT_A, message_id)
        for message_id in range(1, 4):
            seed(store, CHAT_B, message_id)
        store.snapshot_bootstrap([CHAT_A, CHAT_B], limit=200)
        assert store.get_bootstrap_state(CHAT_A)["selected_count"] == 7
        assert store.get_bootstrap_state(CHAT_B)["selected_count"] == 3
        assert store.get_bootstrap_state(CHAT_A)["steady_after_message_id"] == 7
        assert store.get_bootstrap_state(CHAT_B)["steady_after_message_id"] == 3


def test_snapshot_is_idempotent_across_restart(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        for message_id in range(1, 11):
            seed(store, CHAT_A, message_id)
        first = store.snapshot_bootstrap([CHAT_A], limit=5)[0]
        assert first["created"] is True
        # Simulate a restart after more messages arrived.
        seed(store, CHAT_A, 11)
        second = store.snapshot_bootstrap([CHAT_A], limit=5)[0]
        assert second["created"] is False
        assert second["selected_count"] == first["selected_count"]
        assert second["steady_after_message_id"] == first["steady_after_message_id"]


def test_bootstrap_status_can_be_updated(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, CHAT_A, 1)
        store.snapshot_bootstrap([CHAT_A], limit=200)
        store.set_bootstrap_status(CHAT_A, "done")
        assert store.get_bootstrap_state(CHAT_A)["status"] == "done"
        assert store.bootstrap_chat_ids() == [CHAT_A]


# --- bootstrap iteration ----------------------------------------------------


def test_bootstrap_bypasses_prefilter(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, CHAT_A, 1, text=UNRELATED)  # prefilter would reject this
        store.snapshot_bootstrap([CHAT_A], limit=200)
        assert ids(store.iter_ai_bootstrap(PROVIDER, MODEL, PROMPT_VERSION)) == [1]


def test_bootstrap_skips_success_and_bounds_attempts(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        for message_id in (1, 2, 3):
            seed(store, CHAT_A, message_id)
        store.snapshot_bootstrap([CHAT_A], limit=200)
        store.upsert_ai_analysis([ai_ok(CHAT_A, 1)])
        store.upsert_ai_analysis([ai_error(CHAT_A, 2)])

        # success skipped; failure retried while attempts < max.
        assert ids(store.iter_ai_bootstrap(PROVIDER, MODEL, PROMPT_VERSION, max_attempts=3)) == [2, 3]
        # failure exhausted at attempts=1.
        assert ids(store.iter_ai_bootstrap(PROVIDER, MODEL, PROMPT_VERSION, max_attempts=1)) == [3]
        assert store.count_ai_bootstrap(PROVIDER, MODEL, PROMPT_VERSION, max_attempts=3) == 2


# --- steady stream isolation ------------------------------------------------


def test_history_beyond_snapshot_never_enters_steady(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        for message_id in range(1, 251):
            seed(store, CHAT_A, message_id, text=f"msg {message_id}")
        store.snapshot_bootstrap([CHAT_A], limit=200)

        # Nothing newer than the high-water mark yet.
        assert list(store.iter_ai_unprocessed(PROVIDER, MODEL, PROMPT_VERSION, steady_only=True)) == []
        assert store.count_ai_unprocessed(PROVIDER, MODEL, PROMPT_VERSION, steady_only=True) == 0

        # The 50 oldest messages are neither bootstrap items nor steady.
        bootstrap = set(ids(store.iter_ai_bootstrap(PROVIDER, MODEL, PROMPT_VERSION)))
        assert 1 not in bootstrap and 50 not in bootstrap

        # A genuinely new message is the only steady candidate.
        seed(store, CHAT_A, 251, text="brand new")
        assert ids(
            store.iter_ai_unprocessed(PROVIDER, MODEL, PROMPT_VERSION, steady_only=True)
        ) == [251]


def test_steady_only_excludes_chats_without_bootstrap_state(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, CHAT_A, 1)  # no bootstrap state for CHAT_A
        assert list(store.iter_ai_unprocessed(PROVIDER, MODEL, PROMPT_VERSION, steady_only=True)) == []
        # Non-steady selection still sees it (manual backfill path).
        assert ids(store.iter_ai_unprocessed(PROVIDER, MODEL, PROMPT_VERSION)) == [1]


# --- migration --------------------------------------------------------------


def test_bootstrap_tables_are_created_on_existing_db(tmp_path):
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
        names = {
            row[0]
            for row in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"ai_bootstrap_state", "ai_bootstrap_item"} <= names
        assert store.count() == 1  # existing data preserved
        store.snapshot_bootstrap([-1], limit=200)
        assert store.get_bootstrap_state(-1)["selected_count"] == 1
