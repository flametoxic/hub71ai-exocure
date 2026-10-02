"""Validated, versioned catalog of backend capabilities."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from .specs import DomainSpec, MechanismSpec, ModelSpec, OperationSpec, ProfileFieldSpec, VariableSpec


KNOWN_UNITS = {
    "unitless", "days", "date", "AED/month", "AED", "hours", "minutes", "degC", "ug/m3",
    "boolean", "count", "text", "percent", "kW", "minutes/month", "h/month",
}

KNOWN_ADAPTERS = {
    "relocation.simulate", "housing.optimize", "apartment.simulate", "mobility.optimize",
    "schedule.optimize", "consent.propose", "city.read", "cohort.compare", "learning.explain",
    "profile.read", "profile.record",
}


class RegistryError(ValueError):
    pass


def _unique(items: Iterable[Any], kind: str) -> dict[str, Any]:
    result = {}
    for item in items:
        if item.id in result:
            raise RegistryError(f"duplicate {kind}: {item.id}")
        result[item.id] = item
    return result


class ModelRegistry:
    def __init__(
        self,
        *,
        schema_version: str,
        domains: list[DomainSpec],
        variables: list[VariableSpec],
        models: list[ModelSpec],
        mechanisms: list[MechanismSpec],
        operations: list[OperationSpec],
        profile_fields: list[ProfileFieldSpec],
        available_adapters: set[str] | None = None,
    ):
        self.schema_version = schema_version
        self.domains = _unique(domains, "domain")
        self.variables = _unique(variables, "variable")
        self.models = _unique(models, "model")
        self.mechanisms = _unique(mechanisms, "mechanism")
        self.operations = _unique(operations, "operation")
        self._profile_fields = sorted(profile_fields, key=lambda item: (item.order, item.variable_id))
        self._available_adapters = available_adapters or KNOWN_ADAPTERS
        self._validate()

    @classmethod
    def from_documents(
        cls,
        registry: dict[str, Any],
        profile: dict[str, Any] | None = None,
        *,
        available_adapters: set[str] | None = None,
    ) -> "ModelRegistry":
        try:
            return cls(
                schema_version=str(registry.get("schema_version", "1")),
                domains=[DomainSpec.model_validate(item) for item in registry.get("domains", [])],
                variables=[VariableSpec.model_validate(item) for item in registry.get("variables", [])],
                models=[ModelSpec.model_validate(item) for item in registry.get("models", [])],
                mechanisms=[MechanismSpec.model_validate(item) for item in registry.get("mechanisms", [])],
                operations=[OperationSpec.model_validate(item) for item in registry.get("operations", [])],
                profile_fields=[ProfileFieldSpec.model_validate(item) for item in (profile or {}).get("fields", [])],
                available_adapters=available_adapters,
            )
        except RegistryError:
            raise
        except Exception as error:
            raise RegistryError(str(error)) from error

    @classmethod
    def load(cls, path: Path, profile_path: Path | None = None) -> "ModelRegistry":
        registry = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        profile = yaml.safe_load(profile_path.read_text(encoding="utf-8")) if profile_path else {}
        return cls.from_documents(registry, profile or {})

    @classmethod
    def load_default(cls) -> "ModelRegistry":
        demo = Path(__file__).parents[1] / "demo"
        return cls.load(demo / "model_registry.yaml", demo / "profile_schema.yaml")

    def _validate(self) -> None:
        for variable in self.variables.values():
            if variable.domain not in self.domains:
                raise RegistryError(f"variable {variable.id} has unknown domain {variable.domain}")
            if variable.unit not in KNOWN_UNITS:
                raise RegistryError(f"variable {variable.id} has unknown unit {variable.unit}")
            for model_id in variable.models:
                if model_id not in self.models:
                    raise RegistryError(f"variable {variable.id} references unknown model {model_id}")
                if self.models[model_id].domain != variable.domain:
                    raise RegistryError(f"variable {variable.id} and model {model_id} have different domains")
            if variable.minimum is not None and variable.maximum is not None and variable.minimum > variable.maximum:
                raise RegistryError(f"variable {variable.id} has invalid numeric bounds")
        for model in self.models.values():
            if model.domain not in self.domains:
                raise RegistryError(f"model {model.id} has unknown domain {model.domain}")
        for mechanism in self.mechanisms.values():
            if mechanism.model_id not in self.models:
                raise RegistryError(f"mechanism {mechanism.id} references unknown model {mechanism.model_id}")
            if self.models[mechanism.model_id].version != mechanism.model_version:
                raise RegistryError(f"mechanism {mechanism.id} model version does not match")
            if mechanism.adapter not in self._available_adapters:
                raise RegistryError(f"mechanism {mechanism.id} uses unknown adapter {mechanism.adapter}")
            for item in mechanism.inputs:
                if item not in self.variables:
                    raise RegistryError(f"mechanism {mechanism.id} has unknown input {item}")
            if mechanism.output not in self.variables:
                raise RegistryError(f"mechanism {mechanism.id} has unknown output {mechanism.output}")
            involved = [self.variables[item] for item in mechanism.inputs + [mechanism.output]]
            if any(mechanism.model_id not in item.models for item in involved):
                raise RegistryError(f"mechanism {mechanism.id} uses a variable outside model {mechanism.model_id}")
        for field in self._profile_fields:
            if field.variable_id not in self.variables:
                raise RegistryError(f"profile field references unknown variable {field.variable_id}")
            if "observation" not in self.variables[field.variable_id].roles:
                raise RegistryError(f"profile field {field.variable_id} is not an observation")
        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        graph: dict[str, set[str]] = defaultdict(set)
        for mechanism in self.mechanisms.values():
            for item in mechanism.inputs:
                graph[item].add(mechanism.output)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise RegistryError(f"mechanism dependency cycle at {node}")
            if node in visited:
                return
            visiting.add(node)
            for child in graph[node]:
                visit(child)
            visiting.remove(node)
            visited.add(node)

        for node in set(graph):
            visit(node)

    def variable(self, variable_id: str) -> VariableSpec:
        try:
            return self.variables[variable_id]
        except KeyError as error:
            raise RegistryError(f"unknown variable {variable_id}") from error

    def is_active(self, variable_id: str) -> bool:
        self.variable(variable_id)
        return any(variable_id == mechanism.output or variable_id in mechanism.inputs for mechanism in self.mechanisms.values())

    def has_path(self, source: str, target: str) -> bool:
        self.variable(source)
        self.variable(target)
        graph: dict[str, set[str]] = defaultdict(set)
        for mechanism in self.mechanisms.values():
            for item in mechanism.inputs:
                graph[item].add(mechanism.output)
        pending = [source]
        seen = set()
        while pending:
            node = pending.pop()
            if node == target:
                return True
            if node not in seen:
                seen.add(node)
                pending.extend(graph[node])
        return False

    def profile_schema(self) -> list[ProfileFieldSpec]:
        return list(self._profile_fields)

    def tool_catalog(self) -> list[dict[str, Any]]:
        scalar_value = {
            "anyOf": [
                {"type": "number"},
                {"type": "string"},
                {"type": "boolean"},
                {"type": "null"},
            ]
        }
        intervention_ids = sorted(
            variable.id for variable in self.variables.values() if "intervention" in variable.roles
        )
        observation_ids = sorted(field.variable_id for field in self._profile_fields)
        assignment = {
            "type": "object",
            "properties": {
                "variable_id": {"type": "string", "enum": intervention_ids},
                "value": scalar_value,
                "unit": {"type": "string", "enum": sorted(KNOWN_UNITS)},
            },
            "required": ["variable_id", "value", "unit"],
            "additionalProperties": False,
        }
        argument = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "value": scalar_value},
            "required": ["name", "value"],
            "additionalProperties": False,
        }
        properties = {
            "model_id": {"type": "string", "enum": sorted(self.models)},
            "variable_id": {"type": "string", "enum": observation_ids},
            "output_id": {"type": "string", "enum": sorted(
                variable.id for variable in self.variables.values() if "output" in variable.roles
            )},
            "domain": {"type": "string", "enum": sorted(self.domains)},
            "interventions": {"type": "array", "items": assignment},
            "observations": {
                "type": "object",
                "properties": {
                    "value": scalar_value,
                    "status": {"type": "string", "enum": ["known", "unknown", "skipped", "declined"]},
                    "source": {"type": "string"},
                },
                "required": ["value", "status", "source"],
                "additionalProperties": False,
            },
            "objectives": {"type": "array", "items": {"type": "string", "enum": sorted({
                objective for model in self.models.values() for objective in model.objectives
            })}},
            "action": {"type": "string"},
            "arguments": {"type": "array", "items": argument},
        }
        tools = []
        for operation in self.operations.values():
            selected = {name: properties.get(name, {"type": "string"}) for name in operation.parameters}
            tools.append({
                "type": "function",
                "name": operation.id,
                "description": operation.description,
                "parameters": {
                    "type": "object",
                    "properties": selected,
                    "required": list(operation.parameters),
                    "additionalProperties": False,
                },
                "strict": True,
            })
        return tools
