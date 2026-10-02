"""Единый путь идентификации и оценки эффекта X -> Y (CCIA 2.1-2.4 + аппарат Перла).

Порядок: backdoor -> frontdoor -> общий алгоритм ID (Shpitser & Pearl 2006) -> инструментальная
переменная. ID полон для непараметрической идентификации: если он отказал, эффект по графу
неидентифицируем, и инструмент даёт число только при явно объявленном параметрическом допущении
(линейность с однородным эффектом -> ATE; монотонность -> LATE compliers, не ATE). Такой ответ —
``partial`` (CCIA 2.1 «partially») с правом не выше ``informational``; без допущения — non_identified.
Граф отвечает «идентифицируем ли эффект и каким способом», данные — «какой он».
Ни один наблюдательный способ не даёт больше ``review`` (CEOS стр. 9: ActionAuthority != CausalConfidence).
Выбор «максимума эвристик» отсутствует: берётся первый способ, чей критерий доказан на графе.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .causal_estimators import (
    backdoor_effect_from_data,
    frontdoor_criterion,
    frontdoor_effect,
    iv_criterion,
    two_stage_least_squares,
)
from .causal_graph_identification import EXHAUSTIVE_SEARCH_LIMIT, _closure, _nodes, minimal_backdoor_set
from .causal_id_algorithm import ADMG, EmpiricalDistribution, _variables, identify as id_identify, interventional_mean
from .causal_identification import (
    NEXT_ACTION_REQUEST_OBSERVATION,
    NON_IDENTIFIED_PLANNING_USE,
    IdentificationMethod,
    IdentificationResult,
    IdentificationStatus,
    IVAssumption,
)
from .reality_formulas import FormulaReference, FormulaResult

F_PIPELINE = FormulaReference(
    "CCIA-2.1-IDENTIFICATION-PIPELINE",
    "EXO-causal-reasoning-memory-planning-integration-addendum",
    2,
    "identifiable yes|no|partially; method; required confounders; unobserved risk; assumptions; allowed use",
)
F_CONTRAST = FormulaReference(
    "CAUSAL-CONTRAST",
    "Pearl, Causality (2009), §3.2",
    0,
    "Effect = E[Y|do(X=x)] - E[Y|do(X=x_ref)]",
)
F_IV_LATE = FormulaReference(
    "CAUSAL-IV-LATE",
    "Imbens & Angrist (1994), Econometrica 62(2):467-475, Theorem 1",
    0,
    "beta_IV = Cov(Z,Y)/Cov(Z,X) = E[Y1-Y0 | complier] under independence, exclusion, monotonicity",
)
F_IV_LINEAR = FormulaReference(
    "CAUSAL-IV-LINEAR-SCM",
    "Pearl, Causality (2009), §5.4.3 instrumental variables in linear models",
    0,
    "Y = beta X + U_Y, Cov(Z,U_Y)=0 => beta = Cov(Z,Y)/Cov(Z,X) (ATE only when the effect is homogeneous)",
)
BASE_ASSUMPTIONS = ("causal graph is correct", "consistency", "positivity")
#: estimand and assumption text per declared IV assumption (never the ATE without linearity)
IV_ESTIMANDS = {
    IVAssumption.LINEAR_CONSTANT_EFFECT: (
        "ATE_under_linear_homogeneous_SCM",
        "linear structural equations with a homogeneous X->Y effect (parametric; not testable from P(V))",
        F_IV_LINEAR,
    ),
    IVAssumption.MONOTONICITY: (
        "LATE_compliers",
        "monotone compliance, no defiers (Imbens-Angrist); estimand is the complier LATE, not the ATE",
        F_IV_LATE,
    ),
}
IV_PLANNING_USE = "informational"
METHOD_LABEL = {
    IdentificationMethod.ADJUSTMENT: "backdoor_adjustment",
    IdentificationMethod.FRONTDOOR: "frontdoor_adjustment",
    IdentificationMethod.INSTRUMENTAL_VARIABLE: "instrumental_variable",
    IdentificationMethod.ID_ALGORITHM: "id_algorithm",
}


def _graph(edges: Iterable[tuple[str, str]], treatment: str, outcome: str) -> list[tuple[str, str]]:
    edges = [(str(a), str(b)) for a, b in edges]
    nodes = _nodes(edges)
    if treatment not in nodes or outcome not in nodes:
        raise ValueError("treatment and outcome must be nodes of the causal graph")
    if treatment == outcome:
        raise ValueError("treatment and outcome must differ")
    return edges


def find_frontdoor_set(edges, treatment: str, outcome: str, *, latent: Iterable[str] = ()) -> tuple[str, ...] | None:
    """Smallest observed mediator set satisfying the front-door criterion (None if none exists)."""
    edges, latent = list(edges), set(latent)
    candidates = sorted(
        (_closure(edges, {treatment}, forward=True) & _closure(edges, {outcome}, forward=False))
        - {treatment, outcome} - latent
    )
    limit = min(len(candidates), EXHAUSTIVE_SEARCH_LIMIT)
    for size in range(1, limit + 1):
        for subset in combinations(candidates, size):
            if frontdoor_criterion(edges, treatment, outcome, subset, latent=latent).value:
                return tuple(subset)
    return None


def find_instrument(edges, treatment: str, outcome: str, *, latent: Iterable[str] = ()) -> str | None:
    """First (sorted) observed node satisfying the instrument criterion with no covariates."""
    edges, latent = list(edges), set(latent)
    descendants = _closure(edges, {treatment}, forward=True)
    for node in sorted(_nodes(edges) - descendants - latent - {treatment, outcome}):
        if iv_criterion(edges, node, treatment, outcome).value:
            return node
    return None


def identify_causal_effect(
    *,
    edges: Iterable[tuple[str, str]],
    treatment: str,
    outcome: str,
    latent: Iterable[str] = (),
    unobserved_confounder_risk: float | None = None,
    iv_assumption: IVAssumption | str | None = None,
) -> IdentificationResult:
    """Backdoor -> frontdoor -> ID -> (IV as ``partial``). Returns the first method whose criterion holds.

    ``unobserved_confounder_risk`` is the caller's (policy / ConfounderRegistry) risk that the graph
    misses a confounder; None means "not assessed" and is reported as such (never as 0).
    ``iv_assumption`` is the parametric assumption the caller's policy declares for instruments; an
    instrument is used only after ID failed and only with it (status ``partial``, informational use).
    """

    treatment, outcome = str(treatment), str(outcome)
    edges = _graph(edges, treatment, outcome)
    latent = {str(v) for v in latent}
    if {treatment, outcome} & latent:
        raise ValueError("treatment and outcome must be observed")
    if unobserved_confounder_risk is not None:
        risk = float(unobserved_confounder_risk)
        if not 0.0 <= risk <= 1.0:
            raise ValueError("unobserved_confounder_risk must be in [0, 1]")
    else:
        risk = None
    assumption = IVAssumption(iv_assumption) if iv_assumption is not None else None
    reasons: list[str] = []

    adjustment = minimal_backdoor_set(edges, treatment, outcome, latent=latent)
    if adjustment is not None:
        z = ",".join(adjustment)
        expression = f"Σ_{{{z}}} P({outcome}|{treatment},{z})P({z})" if adjustment else f"P({outcome}|{treatment})"
        return IdentificationResult(
            treatment, outcome, IdentificationStatus.IDENTIFIED, IdentificationMethod.ADJUSTMENT, adjustment, risk,
            (*BASE_ASSUMPTIONS, "no undeclared confounders"), "review",
            expression=expression, formula=F_PIPELINE,
        )
    reasons.append("backdoor:no_observed_adjustment_set")

    mediators = find_frontdoor_set(edges, treatment, outcome, latent=latent)
    if mediators is not None:
        m = ",".join(mediators)
        return IdentificationResult(
            treatment, outcome, IdentificationStatus.IDENTIFIED, IdentificationMethod.FRONTDOOR, (), risk,
            (*BASE_ASSUMPTIONS, "mediators intercept every directed path"), "review",
            reasons=tuple(reasons), mediators=mediators, formula=F_PIPELINE,
            expression=f"Σ_{{{m}}} P({m}|{treatment}) Σ_{{{treatment}'}} P({outcome}|{m},{treatment}')P({treatment}')",
        )
    reasons.append("frontdoor:no_valid_mediator_set")

    graph = ADMG.from_latent_dag(edges, latent)
    id_result = id_identify(graph, y=[outcome], x=[treatment])
    if id_result.identified:
        return IdentificationResult(
            treatment, outcome, IdentificationStatus.IDENTIFIED, IdentificationMethod.ID_ALGORITHM, (), risk,
            BASE_ASSUMPTIONS, "review", reasons=tuple(reasons), expression=id_result.rendered, formula=id_result.formula,
        )
    hedge = id_result.hedge.describe()
    reasons.append(f"id:{hedge}")
    measure = tuple(minimal_backdoor_set(edges, treatment, outcome, latent=latent, measurable_latent=True) or ())

    instrument = find_instrument(edges, treatment, outcome, latent=latent)
    if instrument is not None and assumption is not None:
        estimand, assumption_text, formula = IV_ESTIMANDS[assumption]
        return IdentificationResult(
            treatment, outcome, IdentificationStatus.PARTIAL, IdentificationMethod.INSTRUMENTAL_VARIABLE, measure, risk,
            (*BASE_ASSUMPTIONS, assumption_text, "instrument relevance and exclusion"),
            IV_PLANNING_USE, reasons=(*reasons, "iv:non_parametrically_non_identified_parametric_estimand_only"),
            instrument=instrument, hedge=hedge, formula=formula, estimand=estimand,
            next_action=NEXT_ACTION_REQUEST_OBSERVATION,
            expression=f"{estimand} = Cov({instrument},{outcome}) / Cov({instrument},{treatment})",
        )
    if instrument is not None:
        reasons.append(f"iv:instrument_{instrument}_needs_declared_parametric_assumption")
    else:
        reasons.append("iv:no_valid_instrument")
    return IdentificationResult(
        treatment, outcome, IdentificationStatus.NON_IDENTIFIED, None, measure, 1.0, BASE_ASSUMPTIONS,
        NON_IDENTIFIED_PLANNING_USE, None, reasons=tuple(reasons), next_action=NEXT_ACTION_REQUEST_OBSERVATION,
        hedge=hedge, instrument=instrument, formula=id_result.formula,
    )


def _column(data: Mapping[str, Sequence[Any]], name: str) -> np.ndarray:
    if name not in data:
        raise ValueError(f"data lacks the column {name!r} required by the identification")
    return np.asarray(data[name])


def estimate_identified_effect(
    result: IdentificationResult,
    *,
    edges: Iterable[tuple[str, str]],
    latent: Iterable[str] = (),
    data: Mapping[str, Sequence[Any]],
    treatment_value: Any,
    reference_value: Any,
) -> FormulaResult:
    """Effect = E[Y|do(x)] - E[Y|do(x_ref)] by the estimator matching the identification method.

    Raises for a non-identified effect (CCIA 2.1: no number only because a graph exists), for missing
    columns and for positivity violations. IV runs only for a ``partial`` result with a declared
    parametric assumption (2SLS); its number is the declared estimand (LATE / linear-SCM slope), never
    more than informational use.
    """

    partial_iv = (result.status is IdentificationStatus.PARTIAL
                  and result.method is IdentificationMethod.INSTRUMENTAL_VARIABLE and bool(result.estimand))
    if (result.status is not IdentificationStatus.IDENTIFIED and not partial_iv) or result.method is None:
        raise ValueError("effect is not identified: no numerical causal effect may be produced")
    x, y = result.treatment, result.outcome
    method = result.method
    intermediates: dict[str, Any] = {"method": METHOD_LABEL[method], "allowed_planning_use": "review"}
    if method is IdentificationMethod.ADJUSTMENT:
        columns = {name: _column(data, name) for name in result.required_confounders}
        at = backdoor_effect_from_data(x=_column(data, x), y=_column(data, y), z=columns, x_value=treatment_value)
        ref = backdoor_effect_from_data(x=_column(data, x), y=_column(data, y), z=columns, x_value=reference_value)
        effect = float(at.value) - float(ref.value)
        intermediates.update({"at": float(at.value), "reference": float(ref.value)})
    elif method is IdentificationMethod.FRONTDOOR:
        mediators = [_column(data, name) for name in result.mediators]
        m = mediators[0] if len(mediators) == 1 else np.array([repr(tuple(row)) for row in zip(*mediators)])
        at = frontdoor_effect(x=_column(data, x), m=m, y=_column(data, y), x_value=treatment_value)
        ref = frontdoor_effect(x=_column(data, x), m=m, y=_column(data, y), x_value=reference_value)
        effect = float(at.value) - float(ref.value)
        intermediates.update({"at": float(at.value), "reference": float(ref.value)})
    elif method is IdentificationMethod.INSTRUMENTAL_VARIABLE:
        fit = two_stage_least_squares(y=_column(data, y), x=_column(data, x), z=_column(data, str(result.instrument)))
        delta = float(treatment_value) - float(reference_value)
        effect = float(fit.value) * delta
        lower, upper = (float(v) * delta for v in fit.intermediates["ci95"])
        intermediates.update({
            "slope": float(fit.value), "ci95": (min(lower, upper), max(lower, upper)),
            "first_stage_f": fit.intermediates["first_stage_f"], "weak_instrument": fit.intermediates["weak_instrument"],
            # partially identified: never above informational, whatever the instrument strength
            "allowed_planning_use": IV_PLANNING_USE, "estimand": result.estimand,
        })
    else:
        graph = ADMG.from_latent_dag(list(edges), latent)
        id_result = id_identify(graph, y=[y], x=[x])
        if not id_result.identified:
            raise ValueError("graph changed: effect is no longer identifiable")
        needed = set(_variables(id_result.expression)) | {x, y} | set(id_result.invariant_variables)
        distribution = EmpiricalDistribution({name: _column(data, name) for name in sorted(needed)})
        at = interventional_mean(id_result, outcome=y, x_values={x: treatment_value}, distribution=distribution)
        ref = interventional_mean(id_result, outcome=y, x_values={x: reference_value}, distribution=distribution)
        effect = float(at.value) - float(ref.value)
        intermediates.update({"at": float(at.value), "reference": float(ref.value), "expression": id_result.rendered})
    if not np.isfinite(effect):
        raise ValueError("estimated effect is not finite")
    return FormulaResult(effect, F_CONTRAST, intermediates)


__all__ = [
    "F_IV_LATE",
    "F_IV_LINEAR",
    "F_PIPELINE",
    "IV_ESTIMANDS",
    "IV_PLANNING_USE",
    "METHOD_LABEL",
    "estimate_identified_effect",
    "find_frontdoor_set",
    "find_instrument",
    "identify_causal_effect",
]
