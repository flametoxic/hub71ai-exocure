"""Language adapter construction without importing optional SDKs in local mode."""
from __future__ import annotations

from arrival.config import Settings

from .contracts import GatewayConfigurationError, LanguageGateway
from .local import LocalLanguageGateway
from .openai_gateway import OpenAILanguageGateway


def build_language_gateway(settings: Settings) -> LanguageGateway:
    if settings.language_mode == "local":
        return LocalLanguageGateway()
    if not settings.openai_api_key:
        raise GatewayConfigurationError("OPENAI_API_KEY is required in openai mode")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise GatewayConfigurationError("The openai package is required in openai mode") from error
    return OpenAILanguageGateway(settings, client=OpenAI(api_key=settings.openai_api_key, timeout=60, max_retries=0))
