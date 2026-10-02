"""Immutable schema objects for registered CURE domains and models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Role = Literal["observation", "intervention", "objective", "output"]


class FrozenSpec(BaseModel):
    model_config = ConfigDict(frozen=True)


class DomainSpec(FrozenSpec):
    id: str
    title: str


class VariableSpec(FrozenSpec):
    id: str
    domain: str
    title: str
    description: str = ""
    value_type: Literal["number", "integer", "boolean", "text", "date", "object"]
    unit: str
    roles: list[Role]
    privacy: Literal["public", "personal", "sealed", "aggregate"]
    models: list[str] = Field(default_factory=list)
    missing_policy: Literal["required", "optional", "computed", "declinable"]
    mutable: bool = True
    minimum: float | None = None
    maximum: float | None = None
    grounding_terms: list[str] = Field(default_factory=list)
    version: str = "1"


class ProfileFieldSpec(FrozenSpec):
    variable_id: str
    prompts: dict[str, str]
    optional: bool = True
    order: int = 100


class MechanismSpec(FrozenSpec):
    id: str
    model_id: str
    model_version: str
    inputs: list[str]
    output: str
    adapter: str
    formula: str
    version: str


class ModelSpec(FrozenSpec):
    id: str
    domain: str
    version: str
    title: str
    objectives: list[str] = Field(default_factory=list)


class OperationSpec(FrozenSpec):
    id: str
    description: str
    parameters: list[str] = Field(default_factory=list)
