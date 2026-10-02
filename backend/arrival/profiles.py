"""Domain-driven partial resident profiles stored inside the encrypted contour."""
from __future__ import annotations

from datetime import date
import math
from typing import Any

from .modeling.registry import ModelRegistry, RegistryError


PROFILE_KEY = "profile_fields"
VALID_STATUSES = {"known", "unknown", "skipped", "declined", "derived"}


class ProfileValueError(ValueError):
    pass


class ProfileService:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def initialize(self, session, values: dict[str, Any], *, source: str) -> dict[str, dict[str, Any]]:
        fields = {
            field.variable_id: {"status": "unknown", "value": None, "source": None}
            for field in self.registry.profile_schema()
        }
        session.c.extra[PROFILE_KEY] = fields
        for variable_id, value in values.items():
            if value is not None and variable_id in fields:
                self.record(session, variable_id, value=value, status="known", source=source, save=False)
        session.save()
        return self.snapshot(session)

    def _state(self, session) -> dict[str, dict[str, Any]]:
        if PROFILE_KEY not in session.c.extra:
            self.initialize(session, {}, source="unknown")
        return session.c.extra[PROFILE_KEY]

    def _coerce(self, variable_id: str, value: Any) -> Any:
        spec = self.registry.variable(variable_id)
        if spec.value_type == "boolean":
            if not isinstance(value, bool):
                raise ProfileValueError(f"{variable_id} requires a boolean")
            return value
        if spec.value_type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ProfileValueError(f"{variable_id} requires an integer")
            return value
        if spec.value_type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ProfileValueError(f"{variable_id} requires a finite number")
            return float(value)
        if spec.value_type == "date":
            try:
                return date.fromisoformat(str(value)).isoformat()
            except ValueError as error:
                raise ProfileValueError(f"{variable_id} requires an ISO date") from error
        if spec.value_type == "text" and not isinstance(value, str):
            raise ProfileValueError(f"{variable_id} requires text")
        return value

    def record(
        self,
        session,
        variable_id: str,
        *,
        value: Any = None,
        status: str = "known",
        source: str = "user",
        save: bool = True,
    ) -> dict[str, Any]:
        if status not in VALID_STATUSES:
            raise ProfileValueError(f"unknown profile status {status}")
        state = self._state(session)
        if variable_id not in state:
            raise RegistryError(f"{variable_id} is not a registered profile field")
        actual = self._coerce(variable_id, value) if status in {"known", "derived"} else None
        state[variable_id] = {"status": status, "value": actual, "source": source if actual is not None else None}
        self._apply_to_contour(session, variable_id, actual, status)
        if save:
            session.save()
        return dict(state[variable_id])

    @staticmethod
    def _apply_to_contour(session, variable_id: str, value: Any, status: str) -> None:
        if status not in {"known", "derived"}:
            return
        if variable_id == "housing_budget_aed_month":
            session.c.profile_doc.setdefault("params", {})["budget_aed_month"] = {"value": value, "decided_by": "user"}
            session.cache.clear()
        elif variable_id == "preferred_setpoint_c":
            session.c.profile_doc.setdefault("params", {})["setpoint_c"] = {"value": value, "decided_by": "user"}
            session.cache.clear()
        elif variable_id == "arrival_date":
            session.c.extra["arrival"] = value

    def snapshot(self, session) -> dict[str, dict[str, Any]]:
        return {key: dict(value) for key, value in self._state(session).items()}

    def next_question(self, session, lang: str) -> dict[str, Any] | None:
        state = self._state(session)
        for field in self.registry.profile_schema():
            if state[field.variable_id]["status"] == "unknown":
                return {"variable_id": field.variable_id, "text": field.prompts.get(lang) or field.prompts.get("en"), "optional": field.optional}
        return None

    def missing_for_model(self, session, model_id: str) -> list[str]:
        state = self._state(session)
        required = {
            item
            for mechanism in self.registry.mechanisms.values()
            if mechanism.model_id == model_id
            for item in mechanism.inputs
            if self.registry.variables[item].missing_policy == "required"
        }
        return sorted(item for item in required if state.get(item, {}).get("status") not in {"known", "derived"})
