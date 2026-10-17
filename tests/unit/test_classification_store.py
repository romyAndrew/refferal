"""Tests for the message_classifications store and candidate queries."""

import json

import pytest

from referral_finder.classifier import (
    CLASSIFIER_VERSION,
    COMPANY_CONTACT_SEARCH,
    RULE_CATEGORIES,
    classify,
)
from referral_finder.models import IncomingMessage
from referral_finder.storage import MessageStore

CHAT = -1001234567890


def make_message(message_id, text="hello", **overrides):
    data = {
        "chat_id": CHAT,
        "message_id": message_id,
        "date_utc": 1000 + message_id,
        "received_at_utc": 2000 + message_id,
        "text": text,
        "sender_id": 7,
    }
    data.update(overrides)
    return IncomingMessage(**data)


def row(chat_id, message_id, version, category, score=0.5, rules=None, ts=1):
    return (
        int(chat_id),
        int(message_id),
        version,
        category,
        score,
        json.dumps(rules or []),
        ts,
    )


def test_classification_table_exists(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        found = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='message_classifications'"
        ).fetchone()
        assert found == ("message_classifications",)


def test_rule_classifier_never_emits_llm_only_category():
    assert COMPANY_CONTACT_SEARCH not in RULE_CATEGORIES
    for text in (
        "Кто работает в Яндексе?",
        "Есть знакомые в Самокате?",
        "Ищу работу курьером, дайте рефку",
        "Предлагаю вакансию курьера",
    ):
        result = classify(text)
        assert result.category in RULE_CATEGORIES
        assert result.category != COMPANY_CONTACT_SEARCH


def test_save_classifications_rejects_llm_only_category_before_sqlite(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        store.save(make_message(1))
        rows = [
            row(CHAT, 1, "v1", "job_interest"),
            row(CHAT, 1, "v2", COMPANY_CONTACT_SEARCH),
        ]

        with pytest.raises(ValueError, match="company_contact_search"):
            store.save_classifications(rows)

        count = store._conn.execute(
            "SELECT COUNT(*) FROM message_classifications"
        ).fetchone()[0]
        assert count == 0


def test_save_is_idempotent_per_version(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        store.save(make_message(1))
        assert store.save_classifications([row(CHAT, 1, "v1", "uncertain")]) == 1
        assert store.save_classifications([row(CHAT, 1, "v1", "job_interest")]) == 0
        # The first (idempotent) result is kept.
        assert store.get_classification(CHAT, 1, "v1")["category"] == "uncertain"


def test_versions_are_stored_separately(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        store.save(make_message(1))
        store.save_classifications([row(CHAT, 1, "v1", "uncertain")])
        store.save_classifications([row(CHAT, 1, "v2", "job_interest")])
        assert store.get_classification(CHAT, 1, "v1")["category"] == "uncertain"
        assert store.get_classification(CHAT, 1, "v2")["category"] == "job_interest"


def test_get_classification_missing_returns_none(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        assert store.get_classification(CHAT, 99, "v1") is None


def test_count_and_iter_unclassified_skip_textless_and_classified(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        store.save(make_message(1))
        store.save(make_message(2))
        store.save(make_message(3))
        store.save(make_message(4, text=None))  # no text -> never classified
        store.save_classifications([row(CHAT, 1, "v1", "irrelevant")])

        assert store.count_unclassified("v1") == 2
        batches = list(store.iter_unclassified("v1", batch_size=10))
        ids = sorted(message_id for batch in batches for _, message_id, _ in batch)
        assert ids == [2, 3]


def test_iter_unclassified_paginates(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        for message_id in range(1, 11):
            store.save(make_message(message_id))
        batches = list(store.iter_unclassified("v1", batch_size=3))
        assert [len(b) for b in batches] == [3, 3, 3, 1]
        seen = [mid for batch in batches for _, mid, _ in batch]
        assert seen == list(range(1, 11))


def test_iter_candidates_default_and_custom(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        store.save(make_message(1))
        store.save(make_message(2))
        store.save(make_message(3))
        store.save(make_message(4))
        store.save_classifications(
            [
                row(CHAT, 1, "v1", "referral_request", score=0.6),
                row(CHAT, 2, "v1", "job_interest", score=0.4),
                row(CHAT, 3, "v1", "recruitment_offer"),
                row(CHAT, 4, "v1", "irrelevant"),
            ]
        )
        default = [c[4] for c in store.iter_candidates("v1")]
        assert set(default) == {"referral_request", "job_interest"}

        offers = [c[4] for c in store.iter_candidates("v1", categories=("recruitment_offer",))]
        assert offers == ["recruitment_offer"]

        # Only the active version is returned; a different version sees nothing.
        assert list(store.iter_candidates("v2")) == []


def test_classification_does_not_modify_source_message(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        store.save(make_message(1, text="original text"))
        store.save_classifications([row(CHAT, 1, "v1", "job_interest")])
        assert store.fetch(CHAT, 1)[4] == "original text"
        assert store.count() == 1


def test_default_version_constant_is_explicit():
    assert isinstance(CLASSIFIER_VERSION, str) and CLASSIFIER_VERSION
