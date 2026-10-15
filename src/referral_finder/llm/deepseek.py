"""DeepSeek API adapter (no real calls in tests).

Implements the documented DeepSeek chat-completions call behind an injectable
HTTP transport, so tests can exercise request building, headers, error mapping,
retries and JSON parsing without any DNS or network access. The default
transport uses the standard-library :mod:`urllib` (no extra dependency).

Privacy: the adapter only ever sends the already-scrubbed message text and the
configured search criteria. It never receives or forwards Telegram identifiers
or source links. Logs never contain the text, the API key, or message content.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple

from .base import (
    LLMConfigError,
    LLMError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponseError,
    LLMServerError,
    LLMTimeoutError,
)
from .prompt import PROMPT_HASH, PROMPT_VERSION, build_messages
from .schema import AnalysisResult, ValidationError, validate_analysis

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
_CHAT_PATH = "/chat/completions"


@dataclass(frozen=True)
class HttpRequest:
    method: str
    url: str
    headers: Tuple[Tuple[str, str], ...]
    body: bytes


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes


class HttpTransport(Protocol):
    """Minimal async HTTP surface; substitutable in tests."""

    async def send(self, request: HttpRequest) -> HttpResponse:
        ...


class UrllibHttpTransport:
    """Default transport using the standard library (no extra dependency)."""

    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout

    async def send(self, request: HttpRequest) -> HttpResponse:
        return await asyncio.to_thread(self._send_sync, request)

    def _send_sync(self, request: HttpRequest) -> HttpResponse:
        urllib_request = urllib.request.Request(
            request.url,
            data=request.body,
            headers=dict(request.headers),
            method=request.method,
        )
        try:
            with urllib.request.urlopen(urllib_request, timeout=self._timeout) as resp:
                return HttpResponse(int(resp.status), resp.read())
        except urllib.error.HTTPError as exc:
            # HTTP errors still carry a body/status; let the caller map them.
            return HttpResponse(int(exc.code), exc.read())
        except (TimeoutError, socket.timeout) as exc:
            raise LLMTimeoutError("provider request timed out") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise LLMTimeoutError("provider request timed out") from exc
            raise LLMError("provider connection failed") from exc


class DeepSeekProvider:
    """Analysis provider for the DeepSeek chat-completions API.

    Online by construction, but performs no call until :meth:`analyze` is
    awaited with a real transport. Bounded input, bounded output, and a limited
    (never infinite) retry policy for timeouts/rate limits/server errors.
    """

    provider_name = "deepseek"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        transport: Optional[HttpTransport] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        max_input_chars: int = 4000,
        max_output_tokens: int = 400,
        temperature: float = 0.0,
        backoff_seconds: float = 0.5,
        sleep: Optional[Callable[[float], "asyncio.Future"]] = None,
    ):
        if not api_key:
            raise LLMConfigError("DeepSeek API key is required")
        if max_retries < 0:
            raise LLMConfigError("max_retries must not be negative")
        self._api_key = api_key
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._transport: HttpTransport = transport or UrllibHttpTransport(timeout)
        self._max_retries = max_retries
        self._max_input_chars = max_input_chars
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep or asyncio.sleep
        self.prompt_version = PROMPT_VERSION
        self.prompt_hash = PROMPT_HASH

    async def analyze(self, request: LLMRequest) -> AnalysisResult:
        bounded = LLMRequest(
            text=(request.text or "")[: self._max_input_chars],
            criteria=request.criteria,
        )
        http_request = self._build_http_request(build_messages(bounded))

        attempts = self._max_retries + 1
        last_error: Optional[LLMError] = None
        for attempt in range(attempts):
            try:
                response = await self._transport.send(http_request)
            except LLMTimeoutError as exc:
                last_error = exc
                if attempt < self._max_retries:
                    await self._backoff(attempt)
                    continue
                raise

            retryable: Optional[LLMError] = None
            if response.status == 429:
                retryable = LLMRateLimitError("provider rate limit")
            elif response.status >= 500:
                retryable = LLMServerError(f"provider server error ({response.status})")

            if retryable is not None:
                last_error = retryable
                if attempt < self._max_retries:
                    await self._backoff(attempt)
                    continue
                raise retryable

            if response.status in (401, 403):
                raise LLMConfigError("provider authentication failed")
            if not 200 <= response.status < 300:
                raise LLMResponseError(f"unexpected provider status ({response.status})")

            return self._parse_response(response.body)

        # Unreachable: the loop either returns or raises. Kept for clarity.
        raise last_error or LLMError("provider analysis failed")

    async def _backoff(self, attempt: int) -> None:
        await self._sleep(self._backoff_seconds * (2 ** attempt))

    def _build_http_request(self, messages) -> HttpRequest:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = (
            ("Content-Type", "application/json"),
            ("Authorization", f"Bearer {self._api_key}"),
        )
        return HttpRequest(
            method="POST",
            url=f"{self._base_url}{_CHAT_PATH}",
            headers=headers,
            body=body,
        )

    @staticmethod
    def _parse_response(body: bytes) -> AnalysisResult:
        try:
            data = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise LLMResponseError("provider response is not valid JSON") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("provider response is missing choices") from exc

        try:
            payload = json.loads(content)
        except (ValueError, TypeError) as exc:
            raise LLMResponseError("model content is not valid JSON") from exc

        try:
            return validate_analysis(payload)
        except ValidationError as exc:
            raise LLMResponseError(f"model content failed validation: {exc}") from exc
