"""Общее для ТЗ «EXO Economic Optimizer, Multi-Agent Reasoning & Cross-Domain Generalization»."""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from ..world_model.reality_formulas import FormulaReference, FormulaResult

DOCUMENT = "EXO-economic-optimizer-multiagent-generalization"


class EconomicsError(ValueError):
    """Нарушен контракт экономического слоя (fail-closed)."""


def ref(formula_id: str, page: int, expression: str) -> FormulaReference:
    return FormulaReference(formula_id, DOCUMENT, page, expression)


F_STATE = ref("ECO-1-STATE", 1, "x_t = [x_physical, x_asset_health, x_service, x_economic, x_risk]")
F_LIFECYCLE = ref("ECO-1.1-LIFECYCLE", 1, "J = E[sum_t gamma^t (C_energy + C_maintenance + C_downtime + C_comfort/SLA + "
                  "C_risk + C_carbon)] + C_CA; u* = argmin_{u in U_safe} J(u)")
F_NPV = ref("ECO-1.2-NPV", 1, "NPV = sum_t (Benefits_t - Costs_t)/(1+r)^t; RiskAdjustedNPV = NPV - lambda CVaR_a(Loss)")
F_MAINTENANCE = ref("ECO-1.3-MAINTENANCE", 1, "Q_a(u) = C_action(u) + E[C_failure|u] + E[C_energy_degradation|u] + "
                    "C_service_impact(u)")
F_FEASIBLE = ref("ECO-2.1-FEASIBLE", 2, "F = {u : safety(u), law(u), contract(u), capacity(u)}")
F_PARETO = ref("ECO-2.2-PARETO", 2, "u* in F is Pareto-optimal if no u' in F improves some J_i without worsening another")
F_WEIGHTED = ref("ECO-2.3-POLICY-WEIGHTED", 2, "min_{u in F} sum_i w_i J_i(u) + lambda_q Inequity(u)")
F_SCORE = ref("ECO-3-SCORE", 3, "Score(q) = sum_i w_i Support_i(q) - sum_j v_j Contradiction_j(q)")
F_MAPPING = ref("ECO-4-MAPPING", 3, "(u,v) in E_S => (phi(u),phi(v)) in E_T; M_T(phi(u),phi(v)) ~ Psi(M_S(u,v))")
F_PROOF = ref("ECO-6-PROOF", 4, "A |- phi (follows from axioms A, not empirical truth)")
F_UTILITY = ref("ECO-7-UTILITY", 5, "Utility(g) = V_human + V_knowledge + V_service - lambda_r Risk - lambda_p PolicyViolation "
                "- lambda_c C... (the last term is cut off at the page edge)")
F_CAPABILITY = ref("ECO-8-CAPABILITY", 5, "dCapability = Capability(v_new) - Capability(v_base); promote iff "
                   "dCapability > tau_gain AND dSafety >= 0 AND dReliability >= -eps")


def number(name: str, value: Any, *, low: float | None = None, high: float | None = None, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise EconomicsError(f"{name} must be a finite number")
    v = float(value)
    if not math.isfinite(v):
        raise EconomicsError(f"{name} must be finite")
    if low is not None and (v < low or (strict and v == low)):
        raise EconomicsError(f"{name} must be {'>' if strict else '>='} {low}")
    if high is not None and v > high:
        raise EconomicsError(f"{name} must be <= {high}")
    return v


def text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EconomicsError(f"{name} is required")
    return value.strip()


def texts(name: str, values: Iterable[Any]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise EconomicsError(f"{name} must be a sequence")
    return tuple(text(name, v) for v in values)


def cvar(losses, alpha: float) -> tuple[float, float]:
    """(VaR_α, CVaR_α) по выборке: CVaR_α(L) = E[L | L ≥ VaR_α(L)]."""
    L = np.asarray(losses, float).reshape(-1)
    if L.size < 2 or not np.all(np.isfinite(L)):
        raise EconomicsError("CVaR needs at least two finite loss samples")
    a = number("alpha", alpha, low=0.0, high=1.0)
    if not 0.0 < a < 1.0:
        raise EconomicsError("alpha must be in (0, 1)")
    var = float(np.quantile(L, a))
    return var, float(L[L >= var].mean())


def result(value: Any, formula: FormulaReference, **inter: Any) -> FormulaResult:
    return FormulaResult(value, formula, dict(inter))


__all__ = ["DOCUMENT", "EconomicsError", "F_CAPABILITY", "F_FEASIBLE", "F_LIFECYCLE", "F_MAINTENANCE", "F_MAPPING",
           "F_NPV", "F_PARETO", "F_PROOF", "F_SCORE", "F_STATE", "F_UTILITY", "F_WEIGHTED", "cvar", "number", "ref",
           "result", "text", "texts"]
