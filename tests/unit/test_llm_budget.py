"""Tests for the persistent daily API request budget (no network)."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from referral_finder.classifier import JOB_INTEREST
from referral_finder.llm import (
    STATUS_BUDGET_EXHAUSTED,
    AnalysisResult,
    ApiRequestBudget,
    DeepSeekProvider,
    FakeProvider,
    LLMBudgetExceededError,
    LLMRequest,
    MessageAnalysisService,
)
from referral_finder.llm.budget import utc_day
from referral_finder.llm.deepseek import HttpResponse
from referral_finder.llm.versioning import effective_analysis_version
from referral_finder.models import IncomingMessage
from referral_finder.storage import MessageStore

DAY = "2026-01-01"
CHAT = -1001234567890
MATCHING = "Ищу подработку курьером, подскажите, есть ли реферальная программа?"


class FakeTransport:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    async def send(self, request):
        self.requests.append(request)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def ok_response():
    body = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "category": "job_interest",
                                "relevant": True,
                                "confidence": 0.8,
                                "explanation": "ok",
                            }
                        )
                    }
                }
            ]
        }
    ).encode("utf-8")
    return HttpResponse(200, body)


def seed(store, message_id, text=MATCHING):
    store.save(
        IncomingMessage(
            chat_id=CHAT,
            message_id=message_id,
            date_utc=message_id,
            received_at_utc=message_id,
            text=text,
            sender_id=7,
        )
    )


# --- storage / budget object ------------------------------------------------


def test_budget_table_created_on_old_db(tmp_path):
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
        assert "api_request_budget" in names
        assert store.count() == 1  # existing data preserved
        assert store.reserve_api_request(DAY, 5, 0) is True


def test_first_reserve_and_read_counter(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        budget = ApiRequestBudget(store, limit=5, clock=lambda: 0)
        assert budget.day() == utc_day(0)
        assert budget.used() == 0
        assert budget.reserve() is True
        assert budget.used() == 1
        assert budget.remaining() == 4


def test_reaching_the_limit(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        budget = ApiRequestBudget(store, limit=3, clock=lambda: 0)
        assert [budget.reserve() for _ in range(4)] == [True, True, True, False]
        assert budget.used() == 3
        assert budget.remaining() == 0


def test_request_10000_allowed_10001_denied(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        budget = ApiRequestBudget(store, limit=10000, clock=lambda: 0)
        # Seed 9999 used directly to keep the test fast; the boundary behaviour
        # is what matters (the 10000th is allowed, the 10001st is denied).
        store._conn.execute(
            "INSERT INTO api_request_budget(day_utc, used, updated_at_utc) "
            "VALUES (?, ?, ?)",
            (budget.day(), 9999, 0),
        )
        store._conn.commit()
        assert budget.reserve() is True  # 10000th
        assert budget.reserve() is False  # 10001st
        assert budget.reserve() is False
        assert budget.used() == 10000


def test_parallel_reservations_cannot_exceed_limit(tmp_path):
    path = tmp_path / "m.db"
    with MessageStore(path):
        pass
    limit = 50

    def worker(_):
        store = MessageStore(path)
        try:
            return sum(1 for _ in range(20) if store.reserve_api_request(DAY, limit, 0))
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        granted = list(pool.map(worker, range(8)))

    assert sum(granted) == limit
    with MessageStore(path) as store:
        assert store.api_requests_used(DAY) == limit


def test_multiple_connections_share_the_limit(tmp_path):
    path = tmp_path / "m.db"
    with MessageStore(path):
        pass
    a = MessageStore(path)
    b = MessageStore(path)
    try:
        granted = [a.reserve_api_request(DAY, 2, 0), b.reserve_api_request(DAY, 2, 0)]
        assert granted == [True, True]
        assert a.reserve_api_request(DAY, 2, 0) is False
        assert b.reserve_api_request(DAY, 2, 0) is False
        assert a.api_requests_used(DAY) == 2
    finally:
        a.close()
        b.close()


def test_new_utc_day_resets_quota_and_keeps_history(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        now = {"ts": datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc).timestamp()}
        budget = ApiRequestBudget(store, limit=2, clock=lambda: now["ts"])
        assert budget.day() == DAY
        assert [budget.reserve() for _ in range(3)] == [True, True, False]
        now["ts"] += 86400  # next UTC day
        assert budget.reserve() is True
        assert store.api_requests_used(DAY) == 2  # previous day kept


def test_utc_day_ignores_local_timezone():
    # 2026-01-01T23:30:00Z: a local timezone east of UTC would already be Jan 2.
    epoch = datetime(2026, 1, 1, 23, 30, tzinfo=timezone.utc).timestamp()
    assert utc_day(epoch) == "2026-01-01"
    assert utc_day(0) == "1970-01-01"


# --- provider / transport integration ---------------------------------------


async def test_retries_consume_separate_budget_units(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        budget = ApiRequestBudget(store, limit=10, clock=lambda: 0)
        transport = FakeTransport(
            [HttpResponse(500, b"boom"), HttpResponse(429, b"slow"), ok_response()]
        )
        provider = DeepSeekProvider(
            api_key="k", transport=transport, max_retries=2, backoff_seconds=0.0,
            budget=budget,
        )
        result = await provider.analyze(LLMRequest(text="x"))
        assert result.category == JOB_INTEREST
        assert len(transport.requests) == 3  # each retry reserved one unit
        assert budget.used() == 3


async def test_timeout_retry_consumes_units_and_hits_budget(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        budget = ApiRequestBudget(store, limit=2, clock=lambda: 0)
        transport = FakeTransport(
            [HttpResponse(500, b"boom"), HttpResponse(500, b"boom"), HttpResponse(500, b"boom")]
        )
        provider = DeepSeekProvider(
            api_key="k", transport=transport, max_retries=5, backoff_seconds=0.0,
            budget=budget,
        )
        with pytest.raises(LLMBudgetExceededError):
            await provider.analyze(LLMRequest(text="x"))
        # Only two HTTP attempts happened; the third was blocked before sending.
        assert len(transport.requests) == 2
        assert budget.used() == 2


async def test_fake_provider_does_not_use_budget(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        provider = FakeProvider(handler=lambda request: AnalysisResult(JOB_INTEREST, True, 0.9, "ok"))
        await provider.analyze(LLMRequest(text="x"))
        assert store.api_requests_used(utc_day(0)) == 0


# --- service contract -------------------------------------------------------


async def test_budget_exhausted_is_not_a_success_or_attempt(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, 1)
        budget = ApiRequestBudget(store, limit=1, clock=lambda: 0)
        assert budget.reserve() is True  # consume the only unit
        transport = FakeTransport([ok_response()])
        provider = DeepSeekProvider(api_key="k", transport=transport, budget=budget)
        service = MessageAnalysisService(store, provider, enabled=True)

        outcome = await service.analyze(CHAT, 1, MATCHING)

        assert outcome.status == STATUS_BUDGET_EXHAUSTED
        assert outcome.error == "budget_exhausted"
        assert transport.requests == []  # no HTTP request was made
        version = effective_analysis_version(provider.prompt_version, "")
        # No AI row written (no success, no failure, attempts unchanged).
        assert (
            store.get_ai_analysis(
                CHAT, 1, provider.provider_name, provider.model, version
            )
            is None
        )
        assert store.count() == 1  # source message untouched
