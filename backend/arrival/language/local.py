"""Deterministic language adapter used for local development and tests."""
from __future__ import annotations

from typing import Any

from .contracts import BackendResultLike, LanguageRequest, NarrationResult, QueryPlan, ToolCall


class LocalLanguageGateway:
    def plan(self, request: LanguageRequest, catalog: list[dict[str, Any]]) -> QueryPlan:
        from arrival.orchestrator import route

        return QueryPlan(
            source="local",
            operations=[ToolCall(name="legacy_route", arguments={"intent": route(request.text), "text": request.text})],
        )

    def narrate(
        self,
        request: LanguageRequest,
        result: BackendResultLike,
        feedback: dict[str, Any] | None = None,
    ) -> NarrationResult:
        text = str(result.get("local_text") or result.get("message") or "")
        return NarrationResult(text=text, source="local")
