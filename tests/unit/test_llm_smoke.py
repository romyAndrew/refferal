"""Tests for the one-message live smoke command (no real network)."""

import contextlib
import io
import json
import logging
import sqlite3

from referral_finder.classifier import JOB_INTEREST
from referral_finder.llm import AnalysisResult, FakeProvider, HttpResponse, LLMRequest
from referral_finder.llm import smoke as smoke_module
from referral_finder.llm.factory import DEEPSEEK_API_KEY_ENV
from referral_finder.llm.smoke import SMOKE_MESSAGE, run
from referral_finder.logging_setup import setup_logging
from referral_finder.scrubbing import scrub
from referral_finder.storage import MessageStore


def test_without_live_flag_sends_nothing(tmp_path):
    provider = FakeProvider()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert run([], env={}, provider=provider) == 0
    assert provider.calls == []
    text = out.getvalue()
    assert "live=0" in text
    assert "requests=0" in text


def test_live_without_enabled_returns_error():
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert run(["--live"], env={}) == 2
    assert "LLM_ENABLED" in err.getvalue()


def test_live_without_key_returns_error():
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert run(["--live"], env={"LLM_ENABLED": "true"}) == 2
    assert "DEEPSEEK_API_KEY" in err.getvalue()


def test_live_with_injected_provider_sends_scrubbed_message_only():
    provider = FakeProvider(
        handler=lambda request: AnalysisResult(JOB_INTEREST, True, 0.9, "ok")
    )
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = run(["--live", "--criteria", "курьер"], env={}, provider=provider)

    assert rc == 0
    # The provider saw the scrubbed synthetic message, and nothing else.
    assert len(provider.calls) == 1
    sent = provider.calls[0]
    assert sent.text == scrub(SMOKE_MESSAGE).text
    assert sent.criteria == "курьер"
    assert isinstance(sent, LLMRequest)
    assert not hasattr(sent, "chat_id")

    text = out.getvalue()
    assert "status=ok" in text
    assert "category=job_interest" in text
    assert "relevant=True" in text
    assert "requests_planned=1" in text
    assert "requests=1" in text
    # Never print the raw request or the message body.
    assert SMOKE_MESSAGE not in text


def test_live_summary_shows_model_version_and_request_count():
    provider = FakeProvider(
        handler=lambda request: AnalysisResult(JOB_INTEREST, True, 0.5, "ok")
    )
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        run(["--live"], env={}, provider=provider)
    text = out.getvalue()
    assert "model=fake-model" in text
    assert "criteria_version=" in text
    assert "requests_planned=1" in text
    assert "requests=1" in text


def test_prefilter_rejection_skips_provider(monkeypatch):
    import referral_finder.llm.smoke as smoke_module
    from referral_finder.prefilter import PrefilterResult

    monkeypatch.setattr(smoke_module, "prefilter", lambda text: PrefilterResult(False, ()))
    provider = FakeProvider(
        handler=lambda request: AnalysisResult(JOB_INTEREST, True, 0.9, "ok")
    )
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = run(["--live"], env={}, provider=provider)

    assert rc == 1
    assert provider.calls == []  # provider was never called
    text = out.getvalue()
    assert "status=prefilter_skipped" in text
    assert "requests=0" in text
    assert "requests=1" not in text


class FakeTransport:
    """Captures the HTTP request; returns a canned response. No network."""

    def __init__(self, response):
        self._response = response
        self.requests = []

    async def send(self, request):
        self.requests.append(request)
        return self._response


def _chat_response(content, usage=None):
    payload = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        payload["usage"] = usage
    return HttpResponse(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def test_sqlite_error_after_transport_reports_one_request():
    transport = smoke_module._CountingTransport(
        FakeTransport(_chat_response("{}"))
    )

    class AfterTransportSQLiteError:
        prompt_version = "test-prompt"
        model = "test-model"

        async def analyze(self, request):
            await transport.send(request)
            raise sqlite3.Error("store failed")

    args = smoke_module._build_parser().parse_args(["--live"])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = smoke_module._analyze(
            AfterTransportSQLiteError(), args, requests_counter=transport
        )

    text = out.getvalue()
    assert rc == 2
    assert transport.calls == 1
    assert "status=error" in text
    assert "requests=1" in text
    assert "requests=0" not in text


def test_live_path_uses_real_provider_and_sends_no_metadata(tmp_path):
    content = json.dumps(
        {
            "category": "job_interest",
            "relevant": True,
            "confidence": 0.8,
            "explanation": "ok",
        },
        ensure_ascii=False,
    )
    db = tmp_path / "budget.db"
    with MessageStore(db):
        pass  # create the persistent budget database
    transport = FakeTransport(_chat_response(content, {"total_tokens": 42}))
    env = {"LLM_ENABLED": "true", DEEPSEEK_API_KEY_ENV: "SMOKEKEY123"}

    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            setup_logging(level="INFO", secrets=[])
            rc = run(["--live", "--db", str(db)], env=env, transport=transport)
            logging.getLogger("unit.smoke").info("using key SMOKEKEY123 now")
    finally:
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)

    assert rc == 0
    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request.url == "https://api.deepseek.com/chat/completions"

    body = json.loads(request.body)
    assert body["model"] == "deepseek-flash"
    assert body["response_format"] == {"type": "json_object"}
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 200
    user_content = body["messages"][1]["content"]
    assert scrub(SMOKE_MESSAGE).text in user_content

    body_text = request.body.decode("utf-8")
    assert "SMOKEKEY123" not in body_text  # the key never goes into the body
    for leaked in ("chat_id", "sender_id", "message_id", "topic_id", "t.me", "@"):
        assert leaked not in body_text

    text = out.getvalue()
    assert "status=ok" in text
    assert "category=job_interest" in text
    assert "usage=" in text
    assert SMOKE_MESSAGE not in text  # raw message never printed

    # The API key is masked before any use.
    assert "SMOKEKEY123" not in err.getvalue()
    assert "***" in err.getvalue()


def test_live_without_db_sends_nothing():
    transport = FakeTransport(_chat_response("{}"))
    env = {"LLM_ENABLED": "true", DEEPSEEK_API_KEY_ENV: "SMOKEKEY123"}
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = run(["--live"], env=env, transport=transport)
    assert rc == 2
    assert transport.requests == []
    assert "--db" in err.getvalue()


def test_live_with_missing_db_file_sends_nothing(tmp_path):
    missing = tmp_path / "nope" / "budget.db"
    transport = FakeTransport(_chat_response("{}"))
    env = {"LLM_ENABLED": "true", DEEPSEEK_API_KEY_ENV: "SMOKEKEY123"}
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = run(["--live", "--db", str(missing)], env=env, transport=transport)
    assert rc == 2
    assert transport.requests == []
    assert not missing.exists()  # never silently creates a fresh budget DB


def test_live_unreadable_database_sends_nothing(tmp_path):
    db = tmp_path / "corrupt.db"
    db.write_bytes(b"not a sqlite database")
    transport = FakeTransport(_chat_response("{}"))
    env = {"LLM_ENABLED": "true", DEEPSEEK_API_KEY_ENV: "SMOKEKEY123"}
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert run(["--live", "--db", str(db)], env=env, transport=transport) == 2
    assert "budget database could not be opened" in err.getvalue()
    assert transport.requests == []


def test_live_invalid_daily_limit_sends_nothing(tmp_path):
    db = tmp_path / "budget.db"
    with MessageStore(db):
        pass
    transport = FakeTransport(_chat_response("{}"))
    env = {
        "LLM_ENABLED": "true",
        DEEPSEEK_API_KEY_ENV: "SMOKEKEY123",
        "LLM_MAX_API_REQUESTS_PER_DAY": "0",
    }
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert run(["--live", "--db", str(db)], env=env, transport=transport) == 2
    assert "LLM_MAX_API_REQUESTS_PER_DAY" in err.getvalue()
    assert transport.requests == []


def test_live_respects_daily_budget(tmp_path, capsys):
    db = tmp_path / "budget.db"
    with MessageStore(db):
        pass
    content = json.dumps(
        {"category": "job_interest", "relevant": True, "confidence": 0.8, "explanation": "ok"},
        ensure_ascii=False,
    )
    transport = FakeTransport(_chat_response(content))
    env = {
        "LLM_ENABLED": "true",
        DEEPSEEK_API_KEY_ENV: "SMOKEKEY123",
        "LLM_MAX_API_REQUESTS_PER_DAY": "1",
    }
    assert run(["--live", "--db", str(db)], env=env, transport=transport) == 0
    capsys.readouterr()

    # The second live run cannot reserve the daily budget: no HTTP request.
    assert run(["--live", "--db", str(db)], env=env, transport=transport) == 1
    out = capsys.readouterr().out
    assert "status=budget_exhausted" in out
    assert "error=LLMBudgetExceededError" in out
    assert "requests=0" in out
    assert "requests=1" not in out
    assert len(transport.requests) == 1
