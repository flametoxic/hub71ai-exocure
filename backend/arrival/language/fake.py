"""Scripted language adapter with call recording for deterministic tests."""
from __future__ import annotations

from typing import Any

from .contracts import BackendResultLike, LanguageRequest, NarrationResult, QueryPlan


class FakeLanguageGateway:
    def __init__(self, *, plans: list[QueryPlan] | None = None, narrations: list[NarrationResult] | None = None):
        self.plans = list(plans or [])
        self.narrations = list(narrations or [])
        self.calls: list[dict[str, Any]] = []

    def plan(self, request: LanguageRequest, catalog: list[dict[str, Any]]) -> QueryPlan:
        self.calls.append({"method": "plan", "request": request, "catalog": catalog})
        if not self.plans:
            return QueryPlan(source="fake", operations=[])
        return self.plans.pop(0)

    def narrate(
        self,
        request: LanguageRequest,
        result: BackendResultLike,
        feedback: dict[str, Any] | None = None,
    ) -> NarrationResult:
        self.calls.append({"method": "narrate", "request": request, "result": dict(result), "feedback": feedback})
        if not self.narrations:
            return NarrationResult(text=str(result.get("local_text") or result.get("message") or ""), source="fake")
        return self.narrations.pop(0)
