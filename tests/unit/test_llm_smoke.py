"""Tests for the one-message live smoke command (no real network)."""

import contextlib
import io

from referral_finder.classifier import JOB_INTEREST
from referral_finder.llm import AnalysisResult, FakeProvider, LLMRequest
from referral_finder.llm.smoke import SMOKE_MESSAGE, run
from referral_finder.scrubbing import scrub


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
