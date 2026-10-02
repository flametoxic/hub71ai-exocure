from __future__ import annotations

from math import isfinite

from .reality_formulas import FormulaReference, FormulaResult


# Source: CCIA addendum section 2.2 (Confounder registry), page 3 -- not CEOS p.9, which only holds
# epsilon_outcome and DeltaY_hat. The formula_id is kept stable because the formula verifier pins it.
SENSITIVITY_INTERVAL_FORMULA = FormulaReference(
    "CEOS-16-SENSITIVITY-INTERVAL",
    "EXO-causal-reasoning-memory-planning-integration-addendum",
    3,
    "Effect_reported=Effect_estimated+/-Sensitivity_unobserved_confounding",
)


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def outcome_error(*, observed: float, expected: float) -> FormulaResult:
    return FormulaResult(
        value=_finite("observed", observed) - _finite("expected", expected),
        formula=FormulaReference(
            "CEOS-16-OUTCOME-ERROR",
            "EXO-causal-engine-operational-specification",
            9,
            "epsilon_outcome = Y_observed - Y_expected",
        ),
    )


def counterfactual_delta(*, observed: float, no_action: float) -> FormulaResult:
    return FormulaResult(
        value=_finite("observed", observed) - _finite("no_action", no_action),
        formula=FormulaReference(
            "CEOS-16-COUNTERFACTUAL-DELTA",
            "EXO-causal-engine-operational-specification",
            9,
            "DeltaY_hat = Y_observed - Y_no_action",
        ),
    )


def sensitivity_interval(*, estimated_effect: float, unobserved_confounding_sensitivity: float) -> tuple[float, float]:
    effect = _finite("estimated_effect", estimated_effect)
    sensitivity = _finite("unobserved_confounding_sensitivity", unobserved_confounding_sensitivity)
    if sensitivity < 0.0:
        raise ValueError("unobserved_confounding_sensitivity must be non-negative")
    return effect - sensitivity, effect + sensitivity


__all__ = ["SENSITIVITY_INTERVAL_FORMULA", "counterfactual_delta", "outcome_error", "sensitivity_interval"]

