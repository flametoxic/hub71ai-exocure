"""Structured backend results passed to language gateways and API adapters."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ResultStatus = Literal[
    "computed", "clarification_required", "missing_data", "insufficient_model", "unsupported_request",
    "consent_required", "approval_required", "policy_blocked", "generation_failed", "internal_error",
]


class BackendResult(BaseModel):
    status: ResultStatus
    operation: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    facts: dict[str, Any] = Field(default_factory=dict)
    intervals: dict[str, Any] = Field(default_factory=dict)
    cards: list[dict[str, Any]] = Field(default_factory=list)
    pending: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    data_mode: str = "synthetic"
    calculation_status: str = "computed"
    missing: list[str] = Field(default_factory=list)
    clarifications: list[str] = Field(default_factory=list)
    external_action: bool = False
    message: str = ""
    local_text: str = ""
    plan_response_id: str | None = None
    tool_call_id: str | None = None
    tool_outputs: list[dict[str, Any]] = Field(default_factory=list)


class ValidatedPlan(BaseModel):
    source: str
    operations: list[Any]
    response_id: str | None = None
