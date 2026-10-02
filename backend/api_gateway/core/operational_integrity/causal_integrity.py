"""WM-05 §5: причинная целостность.

    P(Y | X = x) ≠ P(Y | do(X = x))  — наблюдательный результат нельзя слить, показать или передать как эффект do()
    Coverage_causal = Σ_{p ∈ Grounded ∪ Validated} w_p / Σ_{p ∈ DecisionPaths} w_p   (минимум — политика по классу)
    max_hypotheses_per_incident = 5; каждая отвергнутая гипотеза — с машинной причиной
    P(ω) = f(U_state, H_sensor, T_mechanism, N_evidence, E_model): слабые свидетельства расширяют интервал симуляции
    и могут снизить полномочия (вид f в ТЗ не задан — здесь линейное расширение с весами политики, см. сверку).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from ..world_model.reality_formulas import FormulaReference
from .context import DOC, DataMode, IntegrityError

F_COVERAGE = FormulaReference("WM05-5.2-COVERAGE", DOC, 6,
                              "Coverage_causal = Σ_{p∈Grounded∪Validated} w_p / Σ_{p∈DecisionPaths} w_p")
F_TRUST_MC = FormulaReference("WM05-5.4-TRUST-MC", DOC, 7, "P(ω) = f(U_state, H_sensor, T_mechanism, N_evidence, E_model)")
MAX_HYPOTHESES_PER_INCIDENT = 5
ELIMINATION_CAUSES = ("constraint_violation", "topology_no_path", "d_separated_given", "contradicted_by",
                      "simulation_gap", "insufficient_evidence")
LIFECYCLE_ORDER = ("structural_possible", "observational_candidate", "physics_grounded", "intervention_validated")
_ALIASES = {"mechanism_grounded": "physics_grounded"}


class ObservationalAsInterventionalError(TypeError):
    """§5.1."""


class CausalMethod(str, Enum):
    OBSERVATIONAL = "observational"
    INTERVENTIONAL = "interventional"
    COUNTERFACTUAL = "counterfactual"


class IdentificationStatus(str, Enum):
    IDENTIFIED = "identified"
    PARTIAL = "partial"
    NON_IDENTIFIED = "non_identified"


@dataclass(frozen=True)
class CausalResultEnvelope:
    query: str
    method: CausalMethod
    identification_status: IdentificationStatus
    estimate: Optional[float]
    interval: Optional[tuple]
    data_mode: DataMode
    evidence_refs: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", CausalMethod(self.method))
        object.__setattr__(self, "identification_status", IdentificationStatus(self.identification_status))
        object.__setattr__(self, "data_mode", DataMode(self.data_mode))
        if self.identification_status is IdentificationStatus.NON_IDENTIFIED and self.estimate is not None:
            raise IntegrityError("a non-identified query carries no point estimate")

    def as_do_effect(self) -> float:
        """Единственный путь получить число как P(Y | do(X)): только интервенционный/контрфактический и
        идентифицированный результат."""
        if self.method is CausalMethod.OBSERVATIONAL:
            raise ObservationalAsInterventionalError("observational result cannot be passed as P(Y | do(X = x))")
        if self.identification_status is not IdentificationStatus.IDENTIFIED or self.estimate is None:
            raise ObservationalAsInterventionalError(f"effect is {self.identification_status.value}: no do() value")
        return float(self.estimate)


def merge_results(a: CausalResultEnvelope, b: CausalResultEnvelope) -> tuple:
    if a.method is not b.method:
        raise ObservationalAsInterventionalError(f"cannot merge {a.method.value} with {b.method.value} results")
    return (a, b)


def _level(x: str) -> int:
    x = _ALIASES.get(x, x)
    if x not in LIFECYCLE_ORDER:
        raise IntegrityError(f"unknown causal lifecycle {x!r}")
    return LIFECYCLE_ORDER.index(x)


@dataclass(frozen=True)
class DecisionPath:
    path_id: str
    edge_lifecycles: tuple                # уровни всех рёбер пути
    weight: float                         # важность пути для решения (политика/влияние)


@dataclass(frozen=True)
class CausalCoverageReport:
    site_scope: str
    body_class: str
    action_class: str
    risk_class: str
    paths_total: int
    structural_possible: int
    observational_candidate: int
    physics_grounded: int
    intervention_validated: int
    weighted_coverage: float
    formula: Any = F_COVERAGE


def causal_coverage(paths: Sequence[DecisionPath], *, site_scope: str, body_class: str, action_class: str,
                    risk_class: str) -> CausalCoverageReport:
    """Путь считается по самому слабому ребру; «обоснован» — все рёбра не ниже physics_grounded."""
    counts = dict.fromkeys(LIFECYCLE_ORDER, 0)
    num = den = 0.0
    for p in paths:
        if not p.edge_lifecycles or p.weight < 0:
            raise IntegrityError(f"path {p.path_id} needs edges and a non-negative weight")
        weakest = min(_level(x) for x in p.edge_lifecycles)
        counts[LIFECYCLE_ORDER[weakest]] += 1
        den += p.weight
        if weakest >= LIFECYCLE_ORDER.index("physics_grounded"):
            num += p.weight
    return CausalCoverageReport(site_scope, body_class, action_class, risk_class, len(paths),
                                counts["structural_possible"], counts["observational_candidate"],
                                counts["physics_grounded"], counts["intervention_validated"],
                                0.0 if den == 0 else num / den)


def coverage_gate(report: CausalCoverageReport, minimum: Mapping[tuple, float]) -> dict:
    key = (report.action_class, report.risk_class)
    if key not in minimum:
        return {"allowed": False, "reason": f"no minimum causal coverage defined for {key} (policy)"}
    ok = report.weighted_coverage >= float(minimum[key])
    return {"allowed": ok, "coverage": report.weighted_coverage, "minimum": float(minimum[key]),
            "reason": None if ok else "causal_coverage_below_policy_minimum"}


def validate_elimination(hypotheses: Sequence[str], accepted: Optional[str],
                         rejected: Mapping[str, Sequence[str]]) -> dict:
    """§5.3: не больше 5 гипотез; каждая отвергнутая — с одной или несколькими машинными причинами."""
    if len(hypotheses) > MAX_HYPOTHESES_PER_INCIDENT:
        raise IntegrityError(f"{len(hypotheses)} hypotheses > {MAX_HYPOTHESES_PER_INCIDENT} per incident")
    problems = []
    for h in hypotheses:
        if h == accepted:
            continue
        causes = rejected.get(h)
        if not causes:
            problems.append(f"{h}: eliminated without trace")
            continue
        bad = [c for c in causes if str(c).split(":")[0] not in ELIMINATION_CAUSES]
        if bad:
            problems.append(f"{h}: non-machine-readable causes {bad}")
    if problems:
        raise IntegrityError("; ".join(problems))
    return {"ok": True, "hypotheses": len(hypotheses)}


def trust_weighted_sigma(sigma: float, *, u_state: float, h_sensor: float, t_mechanism: float, n_evidence: int,
                         e_model: float, weights: Mapping[str, float], authority_max_sigma: float) -> dict:
    """σ_eff = σ · (1 + a_U·U + a_H·H + a_T·(1 − T) + a_N/(1 + N) + a_E·E); веса — политика (ТЗ f не задаёт).
    Все входы нормированы: U, H, E ∈ [0, 1] (доли), T ∈ [0, 1] — доверие механизму, N — число свидетельств."""
    for n, v in (("u_state", u_state), ("h_sensor", h_sensor), ("t_mechanism", t_mechanism), ("e_model", e_model)):
        if not 0.0 <= float(v) <= 1.0:
            raise IntegrityError(f"{n} must be normalised to [0, 1]")
    need = ("u_state", "h_sensor", "t_mechanism", "n_evidence", "e_model")
    if set(weights) != set(need) or any(float(w) < 0 for w in weights.values()):
        raise IntegrityError(f"weights for {need} are required and non-negative (policy)")
    factor = 1 + weights["u_state"] * u_state + weights["h_sensor"] * h_sensor + \
        weights["t_mechanism"] * (1 - t_mechanism) + weights["n_evidence"] / (1 + max(0, int(n_evidence))) + \
        weights["e_model"] * e_model
    s = float(sigma) * factor
    return {"sigma": s, "widening_factor": factor, "authority_reduced": s > authority_max_sigma or not math.isfinite(s),
            "formula": F_TRUST_MC}


__all__ = ["CausalCoverageReport", "CausalMethod", "CausalResultEnvelope", "DecisionPath", "ELIMINATION_CAUSES",
           "F_COVERAGE", "F_TRUST_MC", "IdentificationStatus", "MAX_HYPOTHESES_PER_INCIDENT",
           "ObservationalAsInterventionalError", "causal_coverage", "coverage_gate", "merge_results",
           "trust_weighted_sigma", "validate_elimination"]
