"""Tests for the DeepSeek adapter (injected transport, no network)."""

import json

import pytest

from referral_finder.classifier import JOB_INTEREST, UNCERTAIN
from referral_finder.llm import (
    DeepSeekProvider,
    HttpRequest,
    HttpResponse,
    LLMConfigError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponseError,
    LLMServerError,
    LLMTimeoutError,
    UrllibHttpTransport,
)
from referral_finder.scrubbing import scrub

API_KEY = "test-key-placeholder"


class AllowAllBudget:
    """Permissive test budget: always allows a request, never touches storage."""

    limit = 1_000_000

    def reserve(self) -> bool:
        return True

    def used(self) -> int:
        return 0


class FakeTransport:
    """Records requests and returns/raises canned responses; never connects."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    async def send(self, request: HttpRequest) -> HttpResponse:
        self.requests.append(request)
        outcome = self._responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def chat_response(content) -> HttpResponse:
    body = json.dumps(
        {"choices": [{"message": {"content": content}}]}, ensure_ascii=False
    ).encode("utf-8")
    return HttpResponse(200, body)


def analysis_content(category=JOB_INTEREST, relevant=True, confidence=0.8) -> str:
    return json.dumps(
        {
            "category": category,
            "relevant": relevant,
            "confidence": confidence,
            "explanation": "по смыслу ищет работу",
        },
        ensure_ascii=False,
    )


def provider(transport, **kwargs) -> DeepSeekProvider:
    kwargs.setdefault("api_key", API_KEY)
    kwargs.setdefault("backoff_seconds", 0.0)
    kwargs.setdefault("budget", AllowAllBudget())
    return DeepSeekProvider(transport=transport, **kwargs)


async def test_analyze_parses_valid_response():
    transport = FakeTransport([chat_response(analysis_content())])
    result = await provider(transport).analyze(LLMRequest(text="ищу работу курьером"))
    assert result.category == JOB_INTEREST
    assert result.relevant is True
    assert result.confidence == 0.8


def test_provider_requires_a_budget():
    # No production path may send an HTTP request without reserving the budget.
    with pytest.raises(TypeError):
        DeepSeekProvider(api_key=API_KEY, transport=FakeTransport([]))
    with pytest.raises(LLMConfigError):
        DeepSeekProvider(api_key=API_KEY, transport=FakeTransport([]), budget=None)


def test_api_key_is_required():
    with pytest.raises(LLMConfigError):
        DeepSeekProvider(api_key="", budget=AllowAllBudget())


def test_default_transport_is_std_lib_and_constructs_offline():
    # Constructing the provider must not open a socket or resolve DNS.
    p = DeepSeekProvider(api_key=API_KEY, budget=AllowAllBudget())
    assert isinstance(p._transport, UrllibHttpTransport)


async def test_default_model_is_deepseek_flash_and_overridable():
    transport = FakeTransport([chat_response(analysis_content())])
    await provider(transport).analyze(LLMRequest(text="x"))
    assert json.loads(transport.requests[0].body)["model"] == "deepseek-flash"

    transport2 = FakeTransport([chat_response(analysis_content())])
    await provider(transport2, model="deepseek-chat").analyze(LLMRequest(text="x"))
    assert json.loads(transport2.requests[0].body)["model"] == "deepseek-chat"


async def test_request_shape_and_headers():
    transport = FakeTransport([chat_response(analysis_content())])
    await provider(transport).analyze(
        LLMRequest(text="очищенный текст", criteria="курьер, доставка")
    )
    request = transport.requests[0]
    assert request.method == "POST"
    assert request.url == "https://api.deepseek.com/chat/completions"
    headers = dict(request.headers)
    assert headers["Content-Type"] == "application/json"
    assert headers["Authorization"] == f"Bearer {API_KEY}"

    payload = json.loads(request.body)
    assert payload["messages"][0]["role"] == "system"
    user_content = payload["messages"][1]["content"]
    assert "очищенный текст" in user_content
    assert "курьер, доставка" in user_content
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["stream"] is False


async def test_body_contains_only_scrubbed_text_not_metadata():
    raw = (
        "Пиши @helper, mail@test.ru, тел +7 999 111-22-33, "
        "https://t.me/helper/10, чат -1001234567890"
    )
    cleaned = scrub(raw).text
    transport = FakeTransport([chat_response(analysis_content())])
    await provider(transport).analyze(LLMRequest(text=cleaned, criteria="курьер"))

    body_text = transport.requests[0].body.decode("utf-8")
    for leaked in ("@helper", "mail@test.ru", "111-22-33", "t.me", "-1001234567890"):
        assert leaked not in body_text, leaked


async def test_input_is_bounded():
    transport = FakeTransport([chat_response(analysis_content())])
    await provider(transport, max_input_chars=10).analyze(LLMRequest(text="a" * 500))
    payload = json.loads(transport.requests[0].body)
    content = payload["messages"][1]["content"]
    assert content.split("Message:\n", 1)[1] == "a" * 10


async def test_server_error_retries_then_raises():
    transport = FakeTransport([HttpResponse(500, b"boom"), HttpResponse(500, b"boom")])
    ppt = provider(transport, max_retries=1)
    with pytest.raises(LLMServerError):
        await ppt.analyze(LLMRequest(text="x"))
    assert len(transport.requests) == 2  # 1 + max_retries, not infinite


async def test_rate_limit_raises():
    transport = FakeTransport([HttpResponse(429, b"slow down")])
    with pytest.raises(LLMRateLimitError):
        await provider(transport, max_retries=0).analyze(LLMRequest(text="x"))


async def test_timeout_raises_after_retries():
    transport = FakeTransport([LLMTimeoutError("t"), LLMTimeoutError("t")])
    with pytest.raises(LLMTimeoutError):
        await provider(transport, max_retries=1).analyze(LLMRequest(text="x"))
    assert len(transport.requests) == 2


async def test_auth_error_is_not_retried():
    transport = FakeTransport([HttpResponse(401, b"nope")])
    with pytest.raises(LLMConfigError):
        await provider(transport, max_retries=2).analyze(LLMRequest(text="x"))
    assert len(transport.requests) == 1


async def test_invalid_json_body_raises_response_error():
    transport = FakeTransport([HttpResponse(200, b"<html>not json</html>")])
    with pytest.raises(LLMResponseError):
        await provider(transport).analyze(LLMRequest(text="x"))


async def test_missing_choices_raises_response_error():
    transport = FakeTransport([HttpResponse(200, b'{"error": "nope"}')])
    with pytest.raises(LLMResponseError):
        await provider(transport).analyze(LLMRequest(text="x"))


async def test_invalid_schema_content_raises_response_error():
    bad = json.dumps({"category": "nonsense", "relevant": True, "confidence": 0.5,
                      "explanation": ""})
    transport = FakeTransport([chat_response(bad)])
    with pytest.raises(LLMResponseError):
        await provider(transport).analyze(LLMRequest(text="x"))


async def test_uncertain_abstains_through_adapter():
    content = analysis_content(category=UNCERTAIN, relevant=None, confidence=0.3)
    transport = FakeTransport([chat_response(content)])
    result = await provider(transport).analyze(LLMRequest(text="x"))
    assert result.category == UNCERTAIN
    assert result.relevant is None


async def test_api_key_is_not_placed_in_the_body():
    transport = FakeTransport([chat_response(analysis_content())])
    await provider(transport).analyze(LLMRequest(text="x"))
    assert API_KEY not in transport.requests[0].body.decode("utf-8")


async def test_thinking_mode_is_disabled_in_request():
    transport = FakeTransport([chat_response(analysis_content())])
    await provider(transport).analyze(LLMRequest(text="x"))
    payload = json.loads(transport.requests[0].body)
    assert payload["thinking"] == {"type": "disabled"}


async def test_usage_is_captured_when_present():
    body = json.dumps(
        {
            "choices": [{"message": {"content": analysis_content()}}],
            "usage": {"prompt_tokens": 16, "completion_tokens": 10, "total_tokens": 26},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    transport = FakeTransport([HttpResponse(200, body)])
    p = provider(transport)
    await p.analyze(LLMRequest(text="x"))
    assert p.last_usage == {"prompt_tokens": 16, "completion_tokens": 10, "total_tokens": 26}


async def test_usage_is_none_when_absent():
    transport = FakeTransport([chat_response(analysis_content())])
    p = provider(transport)
    await p.analyze(LLMRequest(text="x"))
    assert p.last_usage is None
