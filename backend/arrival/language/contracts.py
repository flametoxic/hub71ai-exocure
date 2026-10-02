"""Typed language boundary shared by local, fake, and OpenAI adapters."""
from __future__ import annotations

from typing import Any, Mapping, Protocol

from pydantic import BaseModel, Field


class LanguageRequest(BaseModel):
    text: str
    lang: str = "en"
    resident_id: str
    context: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None


class QueryPlan(BaseModel):
    source: str
    operations: list[ToolCall] = Field(default_factory=list)
    response_id: str | None = None
    status: str = "planned"


class NarrationResult(BaseModel):
    text: str
    source: str
    model: str | None = None
    response_id: str | None = None


BackendResultLike = Mapping[str, Any]


class LanguageGateway(Protocol):
    def plan(self, request: LanguageRequest, catalog: list[dict[str, Any]]) -> QueryPlan: ...

    def narrate(
        self,
        request: LanguageRequest,
        result: BackendResultLike,
        feedback: dict[str, Any] | None = None,
    ) -> NarrationResult: ...


class GatewayError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        self.fallback_used = False
        super().__init__(message)


class GatewayConfigurationError(GatewayError):
    def __init__(self, message: str):
        super().__init__("configuration_error", message)
