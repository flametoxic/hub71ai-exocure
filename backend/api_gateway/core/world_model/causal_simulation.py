from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from statistics import mean
from typing import Any, Mapping

from .reality_formulas import FormulaReference
from .structural_causal_model import StructuralCausalModel


CEOS = "EXO-causal-engine-operational-specification"
INTERVENTION_FORMULA = FormulaReference(
    "CEOS-10-INTERVENTION",
    CEOS,
    7,
    "do(X_j=x*) => X_j := x*; effect = Y_do(X=x*)(U_e) - Y(U_e), U_e abducted from the current state",
)
COUNTERFACTUAL_FORMULA = FormulaReference(
    "CEOS-11-COUNTERFACTUAL",
    CEOS,
    7,
    "Y_a'(e) = Predict(M_do(A=a'), U_e)",
)
_ABDUCTION_TOLERANCE = 1e-9


def abduct_disturbances(scm: StructuralCausalModel, factual_state: Mapping[str, float]) -> dict[str, float]:
    """Abduction step: infer U_i for every equation whose value and parents are observed in the trace.

    U_i = x_i - f_i(PA_i, 0; theta_i), then verified by re-evaluation (additive-noise mechanisms only).
    A mechanism that cannot reproduce the observed value with the inferred U fails closed.
    """

    values = {str(name): float(value) for name, value in factual_state.items()}
    if any(not isfinite(value) for value in values.values()):
        raise ValueError("factual trace must be finite")
    disturbances: dict[str, float] = {}
    for variable, equation in scm.equations.items():
        if variable not in values or any(parent not in values for parent in equation.parents):
            continue
        parent_values = {parent: values[parent] for parent in equation.parents}
        base = scm.evaluate_equation(variable, parent_values=parent_values, disturbance=0.0)
        disturbance = values[variable] - base
        reproduced = scm.evaluate_equation(variable, parent_values=parent_values, disturbance=disturbance)
        if not isclose(reproduced, values[variable], rel_tol=0.0, abs_tol=_ABDUCTION_TOLERANCE * max(1.0, abs(values[variable]))):
            raise ValueError(f"abduction failed for {variable}: mechanism is not additive in its disturbance")
        disturbances[variable] = disturbance
    return disturbances


def _propagate(
    scm: StructuralCausalModel,
    *,
    exogenous_state: Mapping[str, float],
    target_variables: tuple[str, ...],
    disturbances: Mapping[str, float],
) -> dict[str, float]:
    """Prediction step: evaluate targets through the (possibly intervened) SCM with the abducted U.

    Variables without an equation are exogenous and read from the state; an endogenous variable
    whose U could not be abducted fails closed instead of silently using U = 0.
    """

    values: dict[str, float] = {}
    visiting: set[str] = set()

    def evaluate(variable: str) -> float:
        if variable in values:
            return values[variable]
        equation = scm.equations.get(variable)
        if equation is None:
            if variable not in exogenous_state:
                raise ValueError(f"missing factual value for exogenous variable {variable}")
            values[variable] = float(exogenous_state[variable])
            return values[variable]
        if not equation.parents and "|do:" in equation.version:
            values[variable] = scm.evaluate_equation(variable, parent_values={}, disturbance=0.0)
            return values[variable]
        if variable not in disturbances:
            raise ValueError(f"no abducted disturbance for {variable}: factual value or parents are missing")
        if variable in visiting:
            raise ValueError("structural equation cycle detected")
        visiting.add(variable)
        parent_values = {parent: evaluate(parent) for parent in equation.parents}
        values[variable] = scm.evaluate_equation(
            variable,
            parent_values=parent_values,
            disturbance=float(disturbances[variable]),
        )
        visiting.remove(variable)
        return values[variable]

    return {target: evaluate(target) for target in target_variables}


@dataclass(frozen=True)
class InterventionSimulationResult:
    intervention: Mapping[str, Any]
    predicted_state: Mapping[str, float]
    expected_effects: Mapping[str, float]
    side_effects: tuple[str, ...]
    risk_distribution: Mapping[str, float]
    blast_radius: tuple[str, ...]
    constraint_margins: Mapping[str, float]
    uncertainty: Mapping[str, Any]
    counterfactual_baseline: Mapping[str, float]
    model_versions: tuple[str, ...]
    policy_versions: tuple[str, ...]
    scm_version: str
    scope: tuple[str, ...]
    abduction: Mapping[str, float]
    epistemic_status: str = "simulated"
    execution_status: str = "not_executed_simulation"
    real_action: bool = False


class InterventionSimulator:
    """Evaluates do(X:=x) on the current world: abduct U from the baseline, replace the equation,
    propagate the same U. do(x) equal to the current value therefore yields a zero effect."""

    def __init__(self, scm: StructuralCausalModel) -> None:
        self.scm = scm

    def simulate(
        self,
        *,
        intervention: Mapping[str, Any],
        baseline_state: Mapping[str, float],
        target_variables: tuple[str, ...],
        scope: tuple[str, ...],
        affected_entities: Mapping[str, float],
        blast_threshold: float,
        constraint_margins: Mapping[str, float],
        side_effects: tuple[str, ...],
        model_versions: tuple[str, ...],
        policy_versions: tuple[str, ...],
    ) -> InterventionSimulationResult:
        variable = str(intervention.get("variable") or "").strip()
        if not variable or "value" not in intervention:
            raise ValueError("intervention requires variable and value")
        if not scope or any(not str(item).strip() for item in scope):
            raise ValueError("intervention simulation requires a non-empty scope")
        if not target_variables:
            raise ValueError("target_variables are required")
        value = float(intervention["value"])
        threshold = float(blast_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("blast_threshold must be in [0, 1]")
        risks = {str(entity): float(probability) for entity, probability in affected_entities.items()}
        if any(not 0.0 <= probability <= 1.0 for probability in risks.values()):
            raise ValueError("affected entity probabilities must be in [0, 1]")
        missing_targets = [target for target in target_variables if target not in baseline_state]
        if missing_targets:
            raise ValueError(f"baseline state lacks factual values for targets: {missing_targets}")

        abducted = abduct_disturbances(self.scm, baseline_state)
        factual = _propagate(
            self.scm,
            exogenous_state=baseline_state,
            target_variables=target_variables,
            disturbances=abducted,
        )
        intervened = self.scm.do_intervention(variable, value=value)
        predicted = _propagate(
            intervened,
            exogenous_state={**baseline_state, variable: value},
            target_variables=target_variables,
            disturbances=abducted,
        )
        expected_effects = {target: float(predicted[target]) - float(factual[target]) for target in target_variables}
        return InterventionSimulationResult(
            intervention={"do": variable, "value": value},
            predicted_state=predicted,
            expected_effects=expected_effects,
            side_effects=tuple(side_effects),
            risk_distribution={
                "maximum": max(risks.values(), default=0.0),
                "mean": mean(risks.values()) if risks else 0.0,
            },
            blast_radius=tuple(sorted(entity for entity, probability in risks.items() if probability > threshold)),
            constraint_margins={str(name): float(margin) for name, margin in constraint_margins.items()},
            uncertainty={
                "affected_entity_probabilities": risks,
                "abduction": "point_estimate_from_current_state",
            },
            counterfactual_baseline=factual,
            model_versions=tuple(model_versions),
            policy_versions=tuple(policy_versions),
            scm_version=intervened.model_version,
            scope=tuple(str(item) for item in scope),
            abduction=abducted,
        )


@dataclass(frozen=True)
class CounterfactualResult:
    abduction: Mapping[str, float]
    action: Mapping[str, Any]
    prediction: Mapping[str, float]
    factual_outcome: Mapping[str, float]
    difference: Mapping[str, float]
    scm_version: str
    scope: tuple[str, ...]
    epistemic_status: str = "estimated"
    execution_status: str = "not_executed_simulation"
    real_action: bool = False


class CounterfactualEngine:
    """Abduction -> action -> prediction (CEOS p.7). U_e is inferred from the factual trace, never
    supplied by the caller, so the factual action a' = a reproduces the fact exactly."""

    def __init__(self, scm: StructuralCausalModel) -> None:
        self.scm = scm

    def evaluate(
        self,
        *,
        factual_trace: Mapping[str, float],
        factual_intervention: Mapping[str, Any],
        alternate_intervention: Mapping[str, Any],
        target_variables: tuple[str, ...],
        scope: tuple[str, ...],
    ) -> CounterfactualResult:
        factual_variable = str(factual_intervention.get("variable") or "")
        alternate_variable = str(alternate_intervention.get("variable") or "")
        if not factual_variable or factual_variable != alternate_variable:
            raise ValueError("factual and alternate interventions must target the same variable")
        if "value" not in factual_intervention or "value" not in alternate_intervention:
            raise ValueError("factual and alternate interventions require values")
        if not scope or any(not str(item).strip() for item in scope):
            raise ValueError("counterfactual requires a non-empty scope")
        if factual_variable not in factual_trace:
            raise ValueError("factual trace must contain the intervened variable")
        if not isclose(float(factual_trace[factual_variable]), float(factual_intervention["value"]), abs_tol=1e-12):
            raise ValueError("factual intervention is inconsistent with the factual trace")
        missing = [target for target in target_variables if target not in factual_trace]
        if not target_variables or missing:
            raise ValueError(f"factual trace must contain every target variable: {missing}")

        abducted = abduct_disturbances(self.scm, factual_trace)
        alternate_value = float(alternate_intervention["value"])
        intervened = self.scm.do_intervention(alternate_variable, alternate_value)
        prediction = _propagate(
            intervened,
            exogenous_state={**factual_trace, alternate_variable: alternate_value},
            target_variables=target_variables,
            disturbances=abducted,
        )
        factual_outcome = {target: float(factual_trace[target]) for target in target_variables}
        return CounterfactualResult(
            abduction=abducted,
            action={"do": alternate_variable, "value": alternate_value},
            prediction=prediction,
            factual_outcome=factual_outcome,
            difference={target: prediction[target] - factual_outcome[target] for target in target_variables},
            scm_version=intervened.model_version,
            scope=tuple(str(item) for item in scope),
        )

    def evaluate_distribution(self, **kwargs: Any) -> "CounterfactualDistribution":
        """UMRM p.6: {mean, p05, p95} with U sampled from P(U | evidence) (see counterfactual_distribution)."""

        return counterfactual_distribution(self.scm, **kwargs)


@dataclass(frozen=True)
class CounterfactualDistribution:
    """UMRM p.6: Y_do(A=a) -> {mean, p05, p95} with U drawn from P(U | evidence)."""

    distribution: Mapping[str, Mapping[str, float]]
    difference: Mapping[str, Mapping[str, float]]
    factual_outcome: Mapping[str, float | None]
    action: Mapping[str, Any]
    abduction_complete: bool
    posterior_sampled: tuple[str, ...]
    effective_sample_size: float
    samples: int
    seed: int
    scm_version: str
    scope: tuple[str, ...]
    epistemic_status: str = "estimated"
    execution_status: str = "not_executed_simulation"
    real_action: bool = False


def _topological(scm: StructuralCausalModel) -> list[str]:
    pending = {v: set(eq.parents) & set(scm.equations) for v, eq in scm.equations.items()}
    order: list[str] = []
    while pending:
        ready = sorted(v for v, deps in pending.items() if not deps - set(order))
        if not ready:
            raise ValueError("structural equation cycle detected")
        for v in ready:
            order.append(v)
            del pending[v]
    return order


def counterfactual_distribution(
    scm: StructuralCausalModel,
    *,
    factual_trace: Mapping[str, float],
    factual_intervention: Mapping[str, Any],
    alternate_intervention: Mapping[str, Any],
    target_variables: tuple[str, ...],
    scope: tuple[str, ...],
    noise_std: Mapping[str, float],
    exogenous_prior: Mapping[str, tuple[float, float]],
    samples: int,
    seed: int,
) -> CounterfactualDistribution:
    """Abduction -> action -> prediction with the posterior over U (UMRM p.6, CEOS p.7).

    Observed endogenous X_i: U_i = x_i - f_i(PA_i) per posterior sample, weight *= N(U_i; 0, σ_i²)
    unless every parent is fixed by evidence (then U_i is exact). Unobserved endogenous X_i:
    U_i ~ N(0, σ_i²). Unobserved exogenous roots: drawn from ``exogenous_prior`` (mean, std).
    σ_i and the priors are calibrated model inputs (no defaults): missing ones fail closed.
    """

    from .structural_equations import _weighted_summary
    import numpy as np

    variable = str(factual_intervention.get("variable") or "")
    if not variable or variable != str(alternate_intervention.get("variable") or ""):
        raise ValueError("factual and alternate interventions must target the same variable")
    if "value" not in factual_intervention or "value" not in alternate_intervention:
        raise ValueError("factual and alternate interventions require values")
    if not scope or any(not str(item).strip() for item in scope):
        raise ValueError("counterfactual requires a non-empty scope")
    if not target_variables:
        raise ValueError("target_variables are required")
    if int(samples) <= 0:
        raise ValueError("samples must be positive")
    trace = {str(k): float(v) for k, v in factual_trace.items()}
    if any(not isfinite(v) for v in trace.values()):
        raise ValueError("factual trace must be finite")
    if variable in trace and not isclose(trace[variable], float(factual_intervention["value"]), abs_tol=1e-12):
        raise ValueError("factual intervention is inconsistent with the factual trace")
    trace[variable] = float(factual_intervention["value"])
    n = int(samples)
    rng = np.random.default_rng(int(seed))
    order = _topological(scm)
    exogenous = sorted({p for eq in scm.equations.values() for p in eq.parents} - set(scm.equations))

    def sigma(name: str) -> float:
        if name not in noise_std:
            raise ValueError(f"noise_std for {name} is required: its disturbance is not fixed by evidence")
        value = float(noise_std[name])
        if not isfinite(value) or value < 0.0:
            raise ValueError("noise_std must be finite and non-negative")
        return value

    factual: dict[str, np.ndarray] = {}
    fixed: set[str] = set()
    for name in exogenous:
        if name in trace:
            factual[name] = np.full(n, trace[name])
            fixed.add(name)
        else:
            if name not in exogenous_prior:
                raise ValueError(f"exogenous_prior for unobserved root {name} is required")
            mean_value, std_value = (float(v) for v in exogenous_prior[name])
            if not isfinite(mean_value) or not isfinite(std_value) or std_value < 0.0:
                raise ValueError("exogenous prior must be finite with non-negative std")
            factual[name] = mean_value + std_value * rng.standard_normal(n)
    disturbances: dict[str, np.ndarray] = {}
    log_weight = np.zeros(n)
    for name in order:
        equation = scm.equations[name]
        base = np.array([
            scm.evaluate_equation(name, parent_values={p: float(factual[p][i]) for p in equation.parents},
                                  disturbance=0.0)
            for i in range(n)
        ])
        if name in trace:
            u = trace[name] - base
            exact = all(p in fixed for p in equation.parents)
            if exact:
                fixed.add(name)
            else:
                std = sigma(name)
                if std > 0.0:
                    log_weight += -0.5 * (u / std) ** 2
                else:
                    log_weight += np.where(np.abs(u) <= _ABDUCTION_TOLERANCE * max(1.0, abs(trace[name])), 0.0, -np.inf)
            probe = scm.evaluate_equation(name, parent_values={p: float(factual[p][0]) for p in equation.parents},
                                          disturbance=float(u[0]))
            if not isclose(probe, trace[name], rel_tol=0.0, abs_tol=_ABDUCTION_TOLERANCE * max(1.0, abs(trace[name]))):
                raise ValueError(f"abduction failed for {name}: mechanism is not additive in its disturbance")
            factual[name] = np.full(n, trace[name])
        else:
            std = sigma(name)
            u = std * rng.standard_normal(n) if std > 0.0 else np.zeros(n)
            factual[name] = np.array([
                scm.evaluate_equation(name, parent_values={p: float(factual[p][i]) for p in equation.parents},
                                      disturbance=float(u[i]))
                for i in range(n)
            ])
        disturbances[name] = np.asarray(u, float)
    if not np.any(np.isfinite(log_weight)):
        raise ValueError("evidence has zero likelihood under the SCM: abduction impossible")
    weights = np.exp(log_weight - np.max(log_weight))
    weights /= weights.sum()

    alternate_value = float(alternate_intervention["value"])
    intervened = scm.do_intervention(variable, alternate_value)
    predicted: dict[str, np.ndarray] = {name: factual[name] for name in exogenous if name != variable}
    predicted[variable] = np.full(n, alternate_value)
    for name in order:
        if name == variable:
            continue
        equation = intervened.equations[name]
        predicted[name] = np.array([
            intervened.evaluate_equation(name, parent_values={p: float(predicted[p][i]) for p in equation.parents},
                                         disturbance=float(disturbances[name][i]))
            for i in range(n)
        ])
    unknown = [t for t in target_variables if t not in predicted]
    if unknown:
        raise ValueError(f"unknown target variables {unknown}")
    distribution = {t: _weighted_summary(predicted[t], weights) for t in target_variables}
    difference = {t: _weighted_summary(predicted[t] - factual[t], weights) for t in target_variables}
    sampled = tuple(v for v in (*exogenous, *order) if v not in fixed and v != variable)
    return CounterfactualDistribution(
        distribution=distribution,
        difference=difference,
        factual_outcome={t: (trace[t] if t in trace else None) for t in target_variables},
        action={"do": variable, "value": alternate_value},
        abduction_complete=not sampled,
        posterior_sampled=sampled,
        effective_sample_size=float(1.0 / np.sum(weights ** 2)),
        samples=n,
        seed=int(seed),
        scm_version=intervened.model_version,
        scope=tuple(str(item) for item in scope),
    )


__all__ = [
    "COUNTERFACTUAL_FORMULA",
    "CounterfactualDistribution",
    "counterfactual_distribution",
    "CounterfactualEngine",
    "CounterfactualResult",
    "INTERVENTION_FORMULA",
    "InterventionSimulationResult",
    "InterventionSimulator",
    "abduct_disturbances",
]

