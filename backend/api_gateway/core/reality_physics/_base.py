"""Общие проверки входов пакета reality_physics (мастер-ТЗ, блок B)."""
from __future__ import annotations

import math
from typing import Any, Mapping

from ..world_model.reality_formulas import FormulaReference

MASTER = "CURE-REALITY-ENGINE-MASTER-TZ"


class PhysicsError(ValueError):
    pass


def ref(formula_id: str, page: int, expression: str) -> FormulaReference:
    return FormulaReference(formula_id, MASTER, page, expression)


def num(name: str, v: Any, *, low: float | None = None, high: float | None = None, strict_low: bool = False) -> float:
    if isinstance(v, bool):
        raise PhysicsError(f"{name} must be a number")
    try:
        x = float(v)
    except (TypeError, ValueError) as exc:
        raise PhysicsError(f"{name} must be a number") from exc
    if not math.isfinite(x):
        raise PhysicsError(f"{name} must be finite")
    if low is not None and (x < low or (strict_low and x == low)):
        raise PhysicsError(f"{name}={x} below the allowed bound {low}")
    if high is not None and x > high:
        raise PhysicsError(f"{name}={x} above the allowed bound {high}")
    return x


def prob(name: str, v: Any) -> float:
    return num(name, v, low=0.0, high=1.0)


def need(m: Mapping, key: str, where: str) -> Any:
    if key not in m or m[key] is None:
        raise PhysicsError(f"{where}: '{key}' is required (policy / site data, no default)")
    return m[key]
