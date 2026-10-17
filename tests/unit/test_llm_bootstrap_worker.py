"""Tests for the bootstrap + steady worker (offline, temporary SQLite only).

No Telegram, no DeepSeek and no real database are touched: every test uses a
temporary SQLite file, the deterministic :class:`FakeProvider` (or a small
budget-backed fake) and injected clocks/waits.
"""

import asyncio

import pytest

from referral_finder.classifier import JOB_INTEREST
from referral_finder.llm import (
    AnalysisResult,
    ApiRequestBudget,
    BootstrapSteadyWorker,
    FakeProvider,
    LLMBudgetExceededError,
)
from referral_finder.llm.prompt import PROMPT_HASH, PROMPT_VERSION
from referral_finder.llm.versioning import effective_analysis_version
from referral_finder.models import IncomingMessage
from referral_finder.prefilter import PREFILTER_VERSION
from referral_finder.storage import MessageStore

CHAT_A = -1001
CHAT_B = -1002
CHAT_C = -1003
MATCHING = "Ищу подработку курьером, подскажите про реферальную программу"
UNRELATED = "Сегодня отличная погода, идём гулять"
WITH_PHONE = "Отличная погода сегодня, звоните +7 999 123-45-67"


def seed(store, chat_id, message_id, text=MATCHING):
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


def ai_count(store) -> int:
    return store._conn.execute("SELECT COUNT(*) FROM message_ai_analysis").fetchone()[0]


def bootstrap_item_ids(store, provider):
    version = effective_analysis_version(provider.prompt_version, "")
    return sorted(
        mid
        for batch in store.iter_ai_bootstrap("fake", "fake-model", version)
        for _, mid, _ in batch
    )


async def wait_until(predicate, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("condition was not met before the timeout")


class CycleWait:
    """Deterministic poll replacement: yields and counts cycles."""

    def __init__(self):
        self.calls = 0

    async def __call__(self) -> None:
        self.calls += 1
        await asyncio.sleep(0)


def provider_ok():
    return FakeProvider(
        handler=lambda request: AnalysisResult(JOB_INTEREST, True, 0.9, "ok")
    )


class BudgetProvider:
    """Offline provider that reserves from the shared daily budget."""

    provider_name = "budget"
    model = "budget-model"
    prompt_version = PROMPT_VERSION
    prompt_hash = PROMPT_HASH

    def __init__(self, budget):
        self._budget = budget
        self.calls = 0

    async def analyze(self, request) -> AnalysisResult:
        if not self._budget.reserve():
            raise LLMBudgetExceededError("daily budget exhausted")
        self.calls += 1
        return AnalysisResult(JOB_INTEREST, True, 0.9, "ok")


def build(store, provider, chat_ids=(CHAT_A,), **kwargs):
    kwargs.setdefault("enabled", True)
    kwargs.setdefault("wait", CycleWait())
    return BootstrapSteadyWorker(store, provider, list(chat_ids), **kwargs)


class GatedProvider:
    """Offline provider whose first call blocks until ``release`` is set."""

    provider_name = "gated"
    model = "gated-model"
    prompt_version = PROMPT_VERSION
    prompt_hash = PROMPT_HASH

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def analyze(self, request) -> AnalysisResult:
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        return AnalysisResult(JOB_INTEREST, True, 0.9, "ok")


# --- bootstrap phase --------------------------------------------------------


async def test_snapshot_and_bootstrap_analysis(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        for message_id in (1, 2, 3):
            seed(store, CHAT_A, message_id)
        provider = provider_ok()
        worker = build(store, provider)

        await worker.start()
        try:
            await wait_until(lambda: worker.bootstrap_processed >= 3)
            await wait_until(lambda: worker.bootstrap_done)
        finally:
            await worker.stop()

        assert len(provider.calls) == 3
        assert ai_count(store) == 3
        assert store.get_bootstrap_state(CHAT_A)["status"] == "done"
        assert worker.phase == "steady"
    finally:
        store.close()


async def test_bootstrap_bypasses_prefilter_but_scrubs(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, CHAT_A, 1, text=WITH_PHONE)
        provider = provider_ok()
        worker = build(store, provider)

        await worker.start()
        try:
            await wait_until(lambda: len(provider.calls) == 1)
        finally:
            await worker.stop()

        # Bypassed the prefilter (an unrelated text was analysed), but the text
        # reached the provider scrubbed and no prefilter decision was stored.
        assert "+7 999 123-45-67" not in provider.calls[0].text
        assert "[phone]" in provider.calls[0].text
        assert store.get_prefilter_result(CHAT_A, 1, PREFILTER_VERSION) is None
        assert store.fetch(CHAT_A, 1)[4] == WITH_PHONE  # source untouched
    finally:
        store.close()


async def test_restart_resumes_without_reanalyzing_success(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        for message_id in (1, 2, 3):
            seed(store, CHAT_A, message_id)

        first = provider_ok()
        worker = build(store, first)
        await worker.start()
        try:
            await wait_until(lambda: worker.bootstrap_done and len(first.calls) == 3)
        finally:
            await worker.stop()

        state_before = store.get_bootstrap_state(CHAT_A)

        second = provider_ok()
        restarted = build(store, second)
        await restarted.start()
        try:
            await wait_until(lambda: restarted.bootstrap_done)
            await wait_until(lambda: restarted.cycles >= 3)
        finally:
            await restarted.stop()

        assert second.calls == []  # no success is re-analysed
        assert ai_count(store) == 3
        state_after = store.get_bootstrap_state(CHAT_A)
        assert state_after["selected_count"] == state_before["selected_count"]
        assert (
            state_after["steady_after_message_id"]
            == state_before["steady_after_message_id"]
        )
    finally:
        store.close()


async def test_snapshot_is_fixed_across_restart(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        for message_id in range(1, 6):
            seed(store, CHAT_A, message_id)

        first = provider_ok()
        worker = build(store, first, bootstrap_limit=2)
        worker.snapshot()
        assert worker._store.get_bootstrap_state(CHAT_A)["selected_count"] == 2

        # A message that arrived after the snapshot must not join it.
        seed(store, CHAT_A, 6)
        worker.snapshot()  # idempotent
        state = store.get_bootstrap_state(CHAT_A)
        assert state["selected_count"] == 2
        assert state["steady_after_message_id"] == 5

        await worker.start()
        try:
            await wait_until(lambda: worker.bootstrap_done)
            await wait_until(lambda: worker.steady_processed >= 1)
        finally:
            await worker.stop()

        # Bootstrap items are exactly ids 4 and 5 (the last two before the
        # snapshot); id 6 is steady, ids 1..3 never auto-analysed.
        assert bootstrap_item_ids(store, first) == []
        version = effective_analysis_version(first.prompt_version, "")
        assert store.get_ai_analysis(CHAT_A, 6, "fake", "fake-model", version) is not None
        for message_id in (1, 2, 3):
            assert (
                store.get_ai_analysis(
                    CHAT_A, message_id, "fake", "fake-model", version
                )
                is None
            )
    finally:
        store.close()


async def test_multiple_chats_and_empty_chat(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        for message_id in (1, 2, 3):
            seed(store, CHAT_A, message_id)
        for message_id in (1, 2):
            seed(store, CHAT_C, message_id)

        provider = provider_ok()
        worker = build(store, provider, chat_ids=(CHAT_A, CHAT_B, CHAT_C))
        await worker.start()
        try:
            await wait_until(lambda: worker.bootstrap_done)
        finally:
            await worker.stop()

        for chat_id in (CHAT_A, CHAT_B, CHAT_C):
            assert store.get_bootstrap_state(chat_id)["status"] == "done"
        assert len(provider.calls) == 5  # CHAT_B contributed nothing

        # A message in the previously empty chat is a steady candidate (> 0).
        seed(store, CHAT_B, 1)
        second = provider_ok()
        restarted = build(
            store, second, chat_ids=(CHAT_A, CHAT_B, CHAT_C), wait=CycleWait()
        )
        await restarted.start()
        try:
            await wait_until(lambda: restarted.steady_processed >= 1)
        finally:
            await restarted.stop()

        version = effective_analysis_version(second.prompt_version, "")
        assert store.get_ai_analysis(CHAT_B, 1, "fake", "fake-model", version) is not None
    finally:
        store.close()


async def test_worker_ignores_other_chats_bootstrap_and_steady(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, CHAT_A, 1)
        seed(store, CHAT_B, 1)
        store.snapshot_bootstrap([CHAT_A, CHAT_B], limit=200)
        state_b_before = store.get_bootstrap_state(CHAT_B)
        version = effective_analysis_version(PROMPT_VERSION, "")

        first = provider_ok()
        worker = build(store, first, chat_ids=(CHAT_A,))
        await worker.start()
        try:
            await wait_until(lambda: worker.bootstrap_done)
        finally:
            await worker.stop()

        # A was analysed; B's bootstrap item and state were left untouched.
        assert store.get_ai_analysis(CHAT_A, 1, "fake", "fake-model", version) is not None
        assert store.get_ai_analysis(CHAT_B, 1, "fake", "fake-model", version) is None
        assert store.get_bootstrap_state(CHAT_B) == state_b_before
        assert [
            mid
            for batch in store.iter_ai_bootstrap(
                "fake", "fake-model", version, chat_ids=[CHAT_B]
            )
            for _, mid, _ in batch
        ] == [1]

        # B also stays out of the steady stream of a worker configured for A.
        seed(store, CHAT_B, 2)
        second = provider_ok()
        restarted = build(store, second, chat_ids=(CHAT_A,))
        await restarted.start()
        try:
            await wait_until(lambda: restarted.bootstrap_done)
            await wait_until(lambda: restarted.cycles >= 2)
        finally:
            await restarted.stop()

        assert second.calls == []
        assert store.get_ai_analysis(CHAT_B, 2, "fake", "fake-model", version) is None
        assert store.get_bootstrap_state(CHAT_B) == state_b_before
    finally:
        store.close()


async def test_new_message_during_bootstrap_is_steady(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        for message_id in (1, 2, 3):
            seed(store, CHAT_A, message_id)
        provider = provider_ok()
        worker = build(store, provider)

        worker.snapshot()  # fix the snapshot at ids 1..3
        seed(store, CHAT_A, 4)  # arrives after the snapshot
        assert store.get_bootstrap_state(CHAT_A)["steady_after_message_id"] == 3

        await worker.start()
        try:
            await wait_until(lambda: worker.bootstrap_done)
            await wait_until(lambda: worker.steady_processed >= 1)
        finally:
            await worker.stop()

        assert worker.bootstrap_processed >= 3
        assert 4 not in bootstrap_item_ids(store, provider)
        version = effective_analysis_version(provider.prompt_version, "")
        assert store.get_ai_analysis(CHAT_A, 4, "fake", "fake-model", version) is not None
    finally:
        store.close()


async def test_history_beyond_limit_never_enters_steady(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        for message_id in range(1, 251):
            seed(store, CHAT_A, message_id)

        provider = provider_ok()
        worker = build(store, provider, bootstrap_limit=200, batch_size=500)
        await worker.start()
        try:
            await wait_until(lambda: worker.bootstrap_done, timeout=5.0)
        finally:
            await worker.stop()

        version = effective_analysis_version(provider.prompt_version, "")
        assert len(provider.calls) == 200
        # The 50 oldest messages (outside the snapshot window) are never analysed.
        for message_id in range(1, 51):
            assert (
                store.get_ai_analysis(CHAT_A, message_id, "fake", "fake-model", version)
                is None
            )
        assert bootstrap_item_ids(store, provider) == []

        # Only a genuinely new message passes into the steady stream.
        seed(store, CHAT_A, 251)
        second = provider_ok()
        restarted = build(store, second, bootstrap_limit=200, batch_size=500)
        await restarted.start()
        try:
            await wait_until(lambda: restarted.steady_processed >= 1)
        finally:
            await restarted.stop()

        assert len(second.calls) == 1
        version2 = effective_analysis_version(second.prompt_version, "")
        assert (
            store.get_ai_analysis(CHAT_A, 251, "fake", "fake-model", version2)
            is not None
        )
        assert (
            store.get_ai_analysis(CHAT_A, 1, "fake", "fake-model", version2) is None
        )
    finally:
        store.close()


async def test_steady_applies_prefilter(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        store.snapshot_bootstrap([CHAT_A], limit=200)  # empty snapshot
        seed(store, CHAT_A, 1, text=UNRELATED)  # steady (1 > 0), prefilter rejects

        provider = provider_ok()
        worker = build(store, provider)
        await worker.start()
        try:
            await wait_until(lambda: worker.steady_processed >= 1)
            await wait_until(
                lambda: store.get_prefilter_result(CHAT_A, 1, PREFILTER_VERSION)
                is not None
            )
        finally:
            await worker.stop()

        assert provider.calls == []
        assert ai_count(store) == 0
        assert store.get_prefilter_result(CHAT_A, 1, PREFILTER_VERSION)["matched"] is False
    finally:
        store.close()


# --- daily budget -----------------------------------------------------------


async def test_budget_shared_across_bootstrap_and_steady(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        budget = ApiRequestBudget(store, limit=3, clock=lambda: 0)
        provider = BudgetProvider(budget)
        worker = build(store, provider)

        for message_id in (1, 2, 3):
            seed(store, CHAT_A, message_id)
        worker.snapshot()
        seed(store, CHAT_A, 4)
        seed(store, CHAT_A, 5)

        await worker.start()
        try:
            await wait_until(lambda: worker.bootstrap_done)
            await wait_until(lambda: worker.cycles >= 5)
        finally:
            await worker.stop()

        # The single shared budget caps the total across both phases.
        assert provider.calls == 3
        assert budget.used() == 3
        assert worker.bootstrap_processed == 3
        assert worker.steady_processed == 0

        version = effective_analysis_version(provider.prompt_version, "")
        for message_id in (4, 5):
            # Blocked messages are neither deleted nor marked as analysed; no
            # attempt was recorded, so they stay queued for the next UTC day.
            assert (
                store.get_ai_analysis(CHAT_A, message_id, "budget", "budget-model", version)
                is None
            )
        queued = [
            message_id
            for batch in store.iter_ai_unprocessed(
                "budget", "budget-model", version, steady_only=True
            )
            for _, message_id, _ in batch
        ]
        assert queued == [4, 5]
    finally:
        store.close()


async def test_budget_exhaustion_stops_phase_without_attempts(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        budget = ApiRequestBudget(store, limit=1, clock=lambda: 0)
        provider = BudgetProvider(budget)
        worker = build(store, provider)

        for message_id in (1, 2):
            seed(store, CHAT_A, message_id)

        await worker.start()
        try:
            await wait_until(lambda: provider.calls == 1)
            await wait_until(lambda: worker.cycles >= 4)
        finally:
            await worker.stop()

        assert provider.calls == 1
        assert budget.used() == 1
        version = effective_analysis_version(provider.prompt_version, "")
        assert (
            store.get_ai_analysis(CHAT_A, 1, "budget", "budget-model", version)["attempts"]
            == 1
        )
        assert (
            store.get_ai_analysis(CHAT_A, 2, "budget", "budget-model", version) is None
        )
        # The queue still has the blocked message.
        assert store.count_ai_bootstrap("budget", "budget-model", version) == 1
    finally:
        store.close()


async def test_budget_recovers_on_next_utc_day(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        now = {"ts": 0.0}
        budget = ApiRequestBudget(store, limit=1, clock=lambda: now["ts"])
        provider = BudgetProvider(budget)
        worker = build(store, provider)

        seed(store, CHAT_A, 1)
        worker.snapshot()
        seed(store, CHAT_A, 2)  # steady

        await worker.start()
        try:
            await wait_until(lambda: worker.bootstrap_done and provider.calls == 1)
            await wait_until(lambda: worker.cycles >= 3)
            assert worker.steady_processed == 0
            now["ts"] += 86400  # next UTC day: budget resets
            await wait_until(lambda: worker.steady_processed >= 1)
        finally:
            await worker.stop()

        assert provider.calls == 2
        assert store.api_requests_used("1970-01-01") == 1
    finally:
        store.close()


# --- lifecycle / disable ----------------------------------------------------


async def test_disabled_worker_takes_no_snapshot(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, CHAT_A, 1)
        provider = provider_ok()
        worker = build(store, provider, enabled=False)

        await worker.start()
        await worker.stop()

        assert provider.calls == []
        assert store.get_bootstrap_state(CHAT_A) is None
        assert worker.running is False
    finally:
        store.close()


async def test_stop_leaves_no_tasks(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, CHAT_A, 1)
        worker = build(store, provider_ok())
        await worker.start()
        await worker.stop()
        await worker.stop()  # re-entrant

        assert worker.running is False
        leaked = [
            task
            for task in asyncio.all_tasks()
            if task.get_name() == "ai-bootstrap-worker"
        ]
        assert leaked == []
    finally:
        store.close()


async def test_worker_does_not_close_the_store(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, CHAT_A, 1)
        worker = build(store, provider_ok())
        await worker.start()
        try:
            await wait_until(lambda: worker.bootstrap_done)
        finally:
            await worker.stop()

        assert store.count() == 1  # still usable by the owner
    finally:
        store.close()


def test_rejects_invalid_parameters(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        provider = provider_ok()
        with pytest.raises(ValueError):
            BootstrapSteadyWorker(store, provider, [CHAT_A], enabled=True, poll_interval=0)
        with pytest.raises(ValueError):
            BootstrapSteadyWorker(store, provider, [CHAT_A], enabled=True, batch_size=0)
        with pytest.raises(ValueError):
            BootstrapSteadyWorker(store, provider, [CHAT_A], enabled=True, max_attempts=0)
        with pytest.raises(ValueError):
            BootstrapSteadyWorker(
                store, provider, [CHAT_A], enabled=True, bootstrap_limit=0
            )
    finally:
        store.close()


# --- B3 coverage gaps -------------------------------------------------------


async def test_steady_waits_for_bootstrap_without_loss(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        for message_id in (1, 2):
            seed(store, CHAT_A, message_id)
        provider = GatedProvider()
        worker = build(store, provider)

        worker.snapshot()  # snapshot fixed at ids 1..2
        seed(store, CHAT_A, 3)  # arrives "during" bootstrap; steady candidate

        await worker.start()
        try:
            await asyncio.wait_for(provider.entered.wait(), timeout=2)
            await asyncio.sleep(0)
            # Bootstrap is mid-flight: steady must not have started, and the new
            # message must not be lost.
            assert worker.bootstrap_done is False
            assert worker.steady_processed == 0
            version = effective_analysis_version(provider.prompt_version, "")
            assert (
                store.get_ai_analysis(CHAT_A, 3, "gated", "gated-model", version)
                is None
            )
            # Release bootstrap; steady then processes the new message.
            provider.release.set()
            await wait_until(lambda: worker.steady_processed >= 1)
        finally:
            provider.release.set()
            await worker.stop()

        assert worker.bootstrap_processed >= 2
        version = effective_analysis_version(provider.prompt_version, "")
        assert (
            store.get_ai_analysis(CHAT_A, 3, "gated", "gated-model", version)
            is not None
        )
    finally:
        store.close()


async def test_cancellation_mid_analysis_leaves_no_false_success(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, CHAT_A, 1)
        provider = GatedProvider()
        worker = build(store, provider)
        await worker.start()
        try:
            await asyncio.wait_for(provider.entered.wait(), timeout=2)

            # Cancel while the provider is in flight: never a success, no leak.
            worker._task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker.stop()

            assert ai_count(store) == 0
            assert store.count() == 1
            leaked = [
                task
                for task in asyncio.all_tasks()
                if task.get_name() == "ai-bootstrap-worker"
            ]
            assert leaked == []

            # The store is still owned by the caller and remains usable.
            seed(store, CHAT_A, 2)
            assert store.count() == 2
        finally:
            provider.release.set()
    finally:
        store.close()
