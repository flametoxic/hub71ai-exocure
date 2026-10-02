"""Арбитраж между сторонами (§2; мастер-ТЗ E2): сначала допустимое множество, потом Парето, потом веса политики.

  1. F = {u : safety(u), law(u), contract(u), capacity(u)} — ни Парето, ни Нэш не выбирают вне F.
  2. Парето-фронт на F по целям сторон J_i (минимизация).
  3. min_{u∈F} Σ w_i J_i(u) + λ_q·Inequity(u); веса — из полномочий/политики, не из предпочтений модели;
     аварийная политика заменяет обычные веса.
  4. Предложения сторон ограничены (цель, ограничения, бюджет/приоритет, свидетельства, коридор действий);
     решает центральный решатель по законности, безопасности и полномочиям, а не большинство голосов.
     Несогласие и его основания записываются (как в ArgumentationFramework).
Неравенство: J_i нормируются по диапазону на F (иначе рубли и минуты несравнимы), мера — из политики:
"gini" (коэффициент Джини нормированных потерь) или "max_gap" (разница худшей и лучшей стороны).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

import numpy as np

from ._base import F_FEASIBLE, F_PARETO, F_WEIGHTED, EconomicsError, number, result, text, texts

FEASIBILITY = ("safety", "law", "contract", "capacity")
INEQUITY_MEASURES = ("gini", "max_gap")


@dataclass(frozen=True)
class Stakeholder:
    stakeholder_id: str
    objectives: tuple[str, ...]            # цели J_i, которые сторона вправе объявлять
    action_scope: tuple[str, ...]          # действия, которые сторона вправе запрашивать


@dataclass(frozen=True)
class ArbitrationPolicy:
    policy_version: str
    weights: Mapping[str, float]           # цель → w_i (из полномочий)
    emergency_weights: Mapping[str, float]
    inequity_lambda: float
    inequity_measure: str
    stakeholders: tuple[Stakeholder, ...]

    def __post_init__(self) -> None:
        text("policy_version", self.policy_version)
        for name in ("weights", "emergency_weights"):
            w = {text("objective", k): number(f"{name}[{k}]", v, low=0.0) for k, v in dict(getattr(self, name)).items()}
            if not w or sum(w.values()) <= 0:
                raise EconomicsError(f"{name} must be non-empty with a positive total")
            object.__setattr__(self, name, MappingProxyType(w))
        number("inequity_lambda", self.inequity_lambda, low=0.0)
        if self.inequity_measure not in INEQUITY_MEASURES:
            raise EconomicsError(f"inequity_measure must be one of {INEQUITY_MEASURES}")
        ids = [s.stakeholder_id for s in self.stakeholders]
        if not ids or len(set(ids)) != len(ids):
            raise EconomicsError("stakeholders must be named once each")


@dataclass(frozen=True)
class Candidate:
    action_id: str
    objectives: Mapping[str, float]         # J_i(u) — чем меньше, тем лучше
    feasibility: Mapping[str, bool]         # safety/law/contract/capacity — проверенные предикаты
    actions: tuple[str, ...] = ()           # какие действия содержит вариант


@dataclass(frozen=True)
class Proposal:
    """§2.4 ограниченное предложение стороны."""

    stakeholder_id: str
    objective: str
    constraints: Mapping[str, float]        # цель → максимум, приемлемый для стороны
    priority: float
    evidence_refs: tuple[str, ...]
    action_envelope: tuple[str, ...]        # какие действия сторона просит допустить

    def __post_init__(self) -> None:
        text("stakeholder_id", self.stakeholder_id)
        text("objective", self.objective)
        number("priority", self.priority, low=0.0)
        if not texts("evidence_refs", self.evidence_refs):
            raise EconomicsError("a proposal must carry supporting evidence (§2.4)")


@dataclass
class ArbitrationResult:
    chosen: Optional[str]
    feasible: tuple[str, ...]
    infeasible: Mapping[str, tuple[str, ...]]
    pareto: tuple[str, ...]
    scores: Mapping[str, float]
    inequity: Mapping[str, float]
    weights_used: Mapping[str, float]
    emergency: bool
    proposals: tuple[dict, ...] = ()
    disagreement: tuple[dict, ...] = ()
    decided_by: str = "central_policy_solver"
    formula: object = field(default=F_WEIGHTED)


def _inequity(norm: np.ndarray, measure: str) -> float:
    if norm.size < 2:
        return 0.0
    if measure == "max_gap":
        return float(norm.max() - norm.min())
    x = np.sort(norm)
    if x.sum() == 0:
        return 0.0
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def arbitrate(candidates: Sequence[Candidate], policy: ArbitrationPolicy, *, proposals: Sequence[Proposal] = (),
              emergency: bool = False) -> ArbitrationResult:
    if not isinstance(policy, ArbitrationPolicy):
        raise EconomicsError("ArbitrationPolicy required")
    weights = dict(policy.emergency_weights if emergency else policy.weights)
    objectives = sorted(weights)
    ids = [c.action_id for c in candidates]
    if len(set(ids)) != len(ids):
        raise EconomicsError("candidate ids must be unique")
    for c in candidates:
        missing = [o for o in objectives if o not in c.objectives]
        if missing:
            raise EconomicsError(f"{c.action_id}: objectives {missing} not evaluated (unknown is not zero)")
    # 1. допустимое множество
    feasible, infeasible = [], {}
    for c in candidates:
        failed = tuple(k for k in FEASIBILITY if c.feasibility.get(k) is not True)
        (infeasible.__setitem__(c.action_id, failed) if failed else feasible.append(c))
    # 4a. предложения: полномочия и коридор, без голосования
    by_id = {s.stakeholder_id: s for s in policy.stakeholders}
    prop_log, accepted = [], []
    for p in proposals:
        s = by_id.get(p.stakeholder_id)
        reasons = []
        if s is None:
            reasons.append("unknown_stakeholder")
        else:
            if p.objective not in s.objectives:
                reasons.append(f"objective_outside_authority:{p.objective}")
            bad = [a for a in p.action_envelope if a not in s.action_scope]
            if bad:
                reasons.append(f"actions_outside_authority:{bad}")
        prop_log.append({"stakeholder": p.stakeholder_id, "objective": p.objective, "accepted": not reasons,
                         "reasons": reasons, "evidence_refs": list(p.evidence_refs), "priority": p.priority})
        if not reasons:
            accepted.append(p)
    # ограничения принятых предложений сужают F (они законны и в пределах полномочий)
    constrained = []
    for c in feasible:
        viol = [f"{p.stakeholder_id}:{k}>{v}" for p in accepted for k, v in p.constraints.items()
                if k in c.objectives and float(c.objectives[k]) > float(v)]
        if viol:
            infeasible[c.action_id] = tuple(viol)
        else:
            constrained.append(c)
    feasible = constrained
    if not feasible:
        return ArbitrationResult(None, (), infeasible, (), {}, {}, weights, emergency, tuple(prop_log),
                                 _disagreement(accepted, candidates, objectives))
    # 2. Парето на F
    J = np.array([[float(c.objectives[o]) for o in objectives] for c in feasible])
    pareto = [feasible[i].action_id for i in range(len(feasible))
              if not any(np.all(J[j] <= J[i]) and np.any(J[j] < J[i]) for j in range(len(feasible)) if j != i)]
    # 3. взвешенный выбор с неравенством
    lo, hi = J.min(0), J.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    norm = (J - lo) / span
    w = np.array([weights[o] for o in objectives])
    ineq = {feasible[i].action_id: _inequity(norm[i], policy.inequity_measure) for i in range(len(feasible))}
    scores = {feasible[i].action_id: float(w @ J[i]) + policy.inequity_lambda * ineq[feasible[i].action_id]
              for i in range(len(feasible))}
    chosen = min(pareto, key=lambda a: (scores[a], a))            # выбор только среди Парето-оптимальных (§2.2)
    return ArbitrationResult(chosen, tuple(c.action_id for c in feasible), infeasible, tuple(pareto), scores, ineq,
                             weights, emergency, tuple(prop_log), _disagreement(accepted, feasible, objectives))


def _disagreement(accepted: Sequence[Proposal], candidates: Sequence[Candidate], objectives) -> tuple[dict, ...]:
    """Кто какой вариант предпочитает по своей цели — запись несогласия, не голосование."""
    out = []
    for p in accepted:
        if p.objective not in objectives or not candidates:
            continue
        pref = min(candidates, key=lambda c: (float(c.objectives[p.objective]), c.action_id)).action_id
        out.append({"stakeholder": p.stakeholder_id, "objective": p.objective, "prefers": pref,
                    "evidence_refs": list(p.evidence_refs)})
    return tuple(out)


def feasible_set(candidates: Sequence[Candidate]):
    ok = tuple(c.action_id for c in candidates if all(c.feasibility.get(k) is True for k in FEASIBILITY))
    return result(ok, F_FEASIBLE)


def is_pareto_optimal(chosen: Candidate, feasible: Sequence[Candidate], objectives: Sequence[str]) -> bool:
    j = np.array([float(chosen.objectives[o]) for o in objectives])
    for c in feasible:
        k = np.array([float(c.objectives[o]) for o in objectives])
        if np.all(k <= j) and np.any(k < j):
            return False
    return True


__all__ = ["ArbitrationPolicy", "ArbitrationResult", "Candidate", "FEASIBILITY", "F_PARETO", "INEQUITY_MEASURES",
           "Proposal", "Stakeholder", "arbitrate", "feasible_set", "is_pareto_optimal"]
