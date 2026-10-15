"""Provider-agnostic LLM integration (offline-first).

The package defines the structured analysis contract, the prompt definition and
its version/hash, a provider interface, and a deterministic ``FakeProvider`` for
offline tests. Concrete network adapters (for example DeepSeek) live alongside
and are disabled by default; nothing here performs any network access.
"""

from .base import (
    LLMConfigError,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
    LLMRequest,
    LLMResponseError,
    LLMServerError,
    LLMTimeoutError,
)
from .fake import FakeProvider
from .deepseek import (
    DeepSeekProvider,
    HttpRequest,
    HttpResponse,
    HttpTransport,
    UrllibHttpTransport,
)
from .prompt import PROMPT_HASH, PROMPT_VERSION, build_messages
from .service import (
    STATUS_DISABLED,
    STATUS_DUPLICATE,
    STATUS_PREFILTER_SKIPPED,
    STATUS_SAVED,
    STATUS_STORAGE_ERROR,
    AnalysisOutcome,
    MessageAnalysisService,
)
from .schema import (
    STATUS_ERROR,
    STATUS_INVALID,
    STATUS_OK,
    STATUS_REFUSED,
    STATUSES,
    RELEVANT_CATEGORIES,
    AnalysisResult,
    ValidationError,
    ai_analysis_row,
    expected_relevant,
    validate_analysis,
)

__all__ = [
    "AnalysisOutcome",
    "AnalysisResult",
    "DeepSeekProvider",
    "FakeProvider",
    "HttpRequest",
    "HttpResponse",
    "HttpTransport",
    "LLMConfigError",
    "LLMError",
    "LLMProvider",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponseError",
    "LLMServerError",
    "LLMTimeoutError",
    "MessageAnalysisService",
    "PROMPT_HASH",
    "PROMPT_VERSION",
    "RELEVANT_CATEGORIES",
    "STATUS_DISABLED",
    "STATUS_DUPLICATE",
    "STATUS_ERROR",
    "STATUS_INVALID",
    "STATUS_OK",
    "STATUS_PREFILTER_SKIPPED",
    "STATUS_REFUSED",
    "STATUS_SAVED",
    "STATUS_STORAGE_ERROR",
    "STATUSES",
    "UrllibHttpTransport",
    "ValidationError",
    "ai_analysis_row",
    "build_messages",
    "expected_relevant",
    "validate_analysis",
]
