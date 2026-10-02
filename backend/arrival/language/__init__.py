"""Public language gateway API."""
from .contracts import (
    GatewayConfigurationError,
    GatewayError,
    LanguageGateway,
    LanguageRequest,
    NarrationResult,
    QueryPlan,
    ToolCall,
)
from .factory import build_language_gateway
from .fake import FakeLanguageGateway
from .local import LocalLanguageGateway
from .openai_gateway import OpenAILanguageGateway
from .narration import NarrationGenerationError, NarrationService, evidence_guard

__all__ = [
    "FakeLanguageGateway",
    "GatewayConfigurationError",
    "GatewayError",
    "LanguageGateway",
    "LanguageRequest",
    "LocalLanguageGateway",
    "NarrationResult",
    "NarrationGenerationError",
    "NarrationService",
    "OpenAILanguageGateway",
    "QueryPlan",
    "ToolCall",
    "build_language_gateway",
    "evidence_guard",
]
