"""Validation of model-selected operations before any session mutation."""
from __future__ import annotations

import math
import re
from typing import Any

from arrival.guard import _tokens
from arrival.language import QueryPlan, ToolCall

from .registry import ModelRegistry
from .results import BackendResult, ValidatedPlan


def _failure(status: str, operation: str, message: str, *, missing: list[str] | None = None) -> BackendResult:
    return BackendResult(
        status=status,
        operation=operation,
        calculation_status="not_computed",
        missing=missing or [],
        message=message,
        local_text=message,
    )


class PlanValidator:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def validate(self, plan: QueryPlan, session, *, user_text: str | None = None) -> ValidatedPlan | BackendResult:
        if len(plan.operations) > 3:
            return _failure("policy_blocked", "plan", "At most three operations are allowed.", missing=["operation_limit:3"])
        if not plan.operations:
            return _failure("unsupported_request", "plan", "No registered backend capability matches this request.")
        normalized_operations = []
        for call in plan.operations:
            normalized = self._normalize_call(call)
            if isinstance(normalized, BackendResult):
                return normalized
            failure = self._validate_call(normalized, session, source=plan.source, user_text=user_text)
            if failure:
                return failure
            normalized_operations.append(normalized)
        return ValidatedPlan(source=plan.source, operations=normalized_operations, response_id=plan.response_id)

    @staticmethod
    def _normalize_call(call: ToolCall) -> ToolCall | BackendResult:
        args = dict(call.arguments)
        assignments = args.get("interventions")
        if isinstance(assignments, list):
            normalized = {}
            for item in assignments:
                if not isinstance(item, dict) or set(item) != {"variable_id", "value", "unit"}:
                    return _failure("unsupported_request", call.name, "Malformed intervention assignment.")
                variable_id = str(item["variable_id"])
                if variable_id in normalized:
                    return _failure("unsupported_request", call.name, f"Duplicate intervention: {variable_id}")
                normalized[variable_id] = {"value": item["value"], "unit": item["unit"]}
            args["interventions"] = normalized
        action_arguments = args.get("arguments")
        if isinstance(action_arguments, list):
            normalized_arguments = {}
            for item in action_arguments:
                if not isinstance(item, dict) or set(item) != {"name", "value"}:
                    return _failure("unsupported_request", call.name, "Malformed action argument.")
                name = str(item["name"])
                if name in normalized_arguments:
                    return _failure("unsupported_request", call.name, f"Duplicate action argument: {name}")
                normalized_arguments[name] = item["value"]
            args["arguments"] = normalized_arguments
        return call.model_copy(update={"arguments": args})

    def _validate_call(self, call: ToolCall, session, *, source: str, user_text: str | None) -> BackendResult | None:
        if call.name not in self.registry.operations:
            return _failure("unsupported_request", call.name, f"Unknown operation: {call.name}")
        args = call.arguments
        if not self._finite(args):
            return _failure("unsupported_request", call.name, "All numeric values must be finite.")
        domain = args.get("domain")
        if domain is not None and domain not in self.registry.domains:
            return _failure("unsupported_request", call.name, f"Unsupported domain: {domain}")
        model_id = args.get("model_id")
        if model_id is not None and model_id not in self.registry.models:
            return _failure("unsupported_request", call.name, f"Unknown model: {model_id}")
        variable_id = args.get("variable_id")
        if variable_id is not None and variable_id not in self.registry.variables:
            return _failure("unsupported_request", call.name, f"Unknown variable: {variable_id}")
        output_id = args.get("output_id")
        if output_id is not None:
            if output_id not in self.registry.variables or "output" not in self.registry.variables[output_id].roles:
                return _failure("unsupported_request", call.name, f"Unknown output: {output_id}")
        if call.name == "record_event" and variable_id not in {field.variable_id for field in self.registry.profile_schema()}:
            return _failure("unsupported_request", call.name, f"Variable {variable_id} is not a writable profile field.")
        if call.name == "optimize" and model_id:
            requested = list(args.get("objectives") or [])
            registered = list(self.registry.models[model_id].objectives)
            if not registered or set(requested) != set(registered):
                return _failure("unsupported_request", call.name, f"Objectives do not match model {model_id}.")
        for collection_name, role in (("interventions", "intervention"), ("observations", "observation")):
            if call.name == "record_event" and collection_name == "observations":
                continue
            for key, raw in (args.get(collection_name) or {}).items():
                failure = self._validate_variable(call.name, key, raw, role, model_id)
                if failure:
                    return failure
                if role == "intervention" and source == "openai" and not self._grounded(key, raw, user_text, session):
                    return _failure(
                        "clarification_required",
                        call.name,
                        f"Intervention {key} must come from the resident or known backend state.",
                        missing=[f"grounding:{key}"],
                    )
        if call.name == "record_event" and variable_id:
            raw = (args.get("observations") or {}).get("value")
            failure = self._validate_variable(call.name, variable_id, raw, "observation", None)
            if failure:
                return failure
            if source == 'openai' and raw is not None and not self._grounded(variable_id, raw, user_text, session):
                return _failure('clarification_required', call.name,
                                'A stored user fact must be grounded in the resident message or confirmed state.',
                                missing=[f'grounding:{variable_id}'])
        if model_id and self.registry.models[model_id].domain == "city" and "arrival_vision" in session.c.revoked:
            return _failure("consent_required", call.name, "Consent is required.", missing=["consent:arrival_vision"])
        return None

    def _grounded(self, variable_id: str, raw: Any, user_text: str | None, session) -> bool:
        value = raw.get("value") if isinstance(raw, dict) and "value" in raw else raw
        stored = session.c.extra.get("profile_fields", {}).get(variable_id, {})
        if stored.get("status") in {"known", "derived"} and stored.get("value") == value:
            return True
        text = user_text or ""
        lowered = text.casefold()
        spec = self.registry.variables[variable_id]
        ignored = {"day", "days", "hour", "hours", "minute", "minutes", "value", "count"}
        terms = list(spec.grounding_terms) or [
            token for token in variable_id.replace("_", " ").split() if token not in ignored and len(token) > 2
        ]
        if not any(re.search(rf"(?<!\w){re.escape(term.casefold())}(?!\w)", lowered) for term in terms):
            return False
        if isinstance(value, bool):
            return any(word in lowered.split() for word in {"true", "false", "yes", "no"})
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return any(float(candidate) == float(value) for candidate, _ in _tokens(text))
        if isinstance(value, str):
            return bool(value.strip()) and value.strip().casefold() in lowered
        return False

    def _validate_variable(self, operation: str, variable_id: str, raw: Any, role: str, model_id: str | None) -> BackendResult | None:
        if variable_id not in self.registry.variables:
            return _failure("unsupported_request", operation, f"Unknown variable: {variable_id}")
        spec = self.registry.variables[variable_id]
        if role not in spec.roles:
            return _failure("unsupported_request", operation, f"Variable {variable_id} cannot be used as {role}.")
        if model_id and model_id not in spec.models:
            return _failure("unsupported_request", operation, f"Variable {variable_id} does not apply to {model_id}.")
        value = raw
        if isinstance(raw, dict) and "value" in raw:
            if raw.get("unit") != spec.unit:
                return _failure("unsupported_request", operation, f"Unit mismatch for {variable_id}.")
            value = raw["value"]
        if spec.value_type == "boolean" and not isinstance(value, bool):
            return _failure("unsupported_request", operation, f"Variable {variable_id} requires a boolean.")
        if spec.value_type in {"number", "integer"} and (isinstance(value, bool) or not isinstance(value, (int, float))):
            return _failure("unsupported_request", operation, f"Variable {variable_id} requires a number.")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if spec.minimum is not None and value < spec.minimum:
                return _failure("unsupported_request", operation, f"Variable {variable_id} is below its minimum.")
            if spec.maximum is not None and value > spec.maximum:
                return _failure("unsupported_request", operation, f"Variable {variable_id} is above its maximum.")
        return None

    @staticmethod
    def _finite(value: Any) -> bool:
        if isinstance(value, bool) or value is None:
            return True
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(PlanValidator._finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(PlanValidator._finite(item) for item in value)
        return True
