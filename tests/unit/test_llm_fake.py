"""Tests for the deterministic offline FakeProvider (no network)."""

import pytest

from referral_finder.llm import (
    AnalysisResult,
    FakeProvider,
    LLMRequest,
    PROMPT_HASH,
    PROMPT_VERSION,
)
from referral_finder.classifier import UNCERTAIN


async def test_default_returns_uncertain_and_records_call():
    provider = FakeProvider()
    result = await provider.analyze(LLMRequest(text="привет"))
    assert result.category == UNCERTAIN
    assert result.relevant is None
    assert provider.calls == [LLMRequest(text="привет")]


async def test_from_mapping_returns_configured_result():
    expected = AnalysisResult("job_interest", True, 0.7, "ok")
    provider = FakeProvider.from_mapping({"ищу работу": expected})
    assert await provider.analyze(LLMRequest(text="ищу работу")) == expected
    # Unmapped text falls back to uncertain.
    fallback = await provider.analyze(LLMRequest(text="что-то другое"))
    assert fallback.category == UNCERTAIN


async def test_handler_can_raise_an_error():
    def handler(request):
        raise RuntimeError("provider down")

    provider = FakeProvider(handler)
    with pytest.raises(RuntimeError):
        await provider.analyze(LLMRequest(text="x"))
    assert len(provider.calls) == 1


async def test_only_receives_what_caller_passes():
    provider = FakeProvider()
    request = LLMRequest(text="scrubbed text", criteria="курьер, доставка")
    await provider.analyze(request)
    assert provider.calls[0].text == "scrubbed text"
    assert provider.calls[0].criteria == "курьер, доставка"


def test_prompt_metadata_present():
    provider = FakeProvider()
    assert provider.provider_name == "fake"
    assert provider.prompt_version == PROMPT_VERSION
    assert provider.prompt_hash == PROMPT_HASH
