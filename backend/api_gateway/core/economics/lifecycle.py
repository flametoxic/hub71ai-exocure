"""Долгосрочная экономика (§1): жизненный цикл, риск-скорректированная NPV, выбор плана в безопасном множестве.

J = E[Σ_t γ^t (C_energy + C_maintenance + C_downtime + C_comfort/SLA + C_risk + C_carbon)] + C_CA
u* = argmin_{u ∈ U_safe} J(u) — безопасность, регуляторика и минимальный сервис — жёсткие ограничения:
план вне U_safe не участвует в минимизации вообще (его J не сравнивается).
NPV = Σ_t (Benefits_t − Costs_t)/(1+r)^t;  RiskAdjustedNPV = NPV − λ·CVaR_α(Loss) — план, экономящий «в среднем»,
но с редким дорогим отказом сервиса, наказывается хвостом.
Ожидания — по выборкам сценариев (например, прогонам Simulation Proof Runtime), а не по одному прогнозу.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from ._base import F_LIFECYCLE, F_NPV, EconomicsError, cvar, number, result, text

COST_COMPONENTS = ("energy", "maintenance", "downtime", "comfort_sla", "risk", "carbon")


def discounted_cost(stream: Sequence[Mapping[str, float]], *, gamma: float, capital: float) -> float:
    """Σ_t γ^t Σ_k C_k,t + C_CA для одной траектории затрат; каждая компонента обязана быть задана."""
    g = number("gamma", gamma, low=0.0, high=1.0)
    total = 0.0
    for t, costs in enumerate(stream):
        missing = [k for k in COST_COMPONENTS if k not in costs]
        if missing:
            raise EconomicsError(f"cost step {t} lacks components {missing} (a missing cost is not zero)")
        extra = sorted(set(costs) - set(COST_COMPONENTS))
        if extra:
            raise EconomicsError(f"unknown cost components {extra}")
        total += (g ** t) * sum(number(f"C_{k},{t}", costs[k]) for k in COST_COMPONENTS)
    return total + number("C_CA", capital)


@dataclass(frozen=True)
class PlanEconomics:
    plan_id: str
    samples: tuple[float, ...]            # J по сценариям
    expected: float
    safe: bool
    violations: tuple[str, ...]


def lifecycle_objective(plan_id: str, streams: Sequence[Sequence[Mapping[str, float]]], *, gamma: float,
                        capital: float, violations: Sequence[str] = ()) -> PlanEconomics:
    if len(streams) < 1:
        raise EconomicsError("at least one cost scenario is required")
    js = tuple(discounted_cost(s, gamma=gamma, capital=capital) for s in streams)
    return PlanEconomics(text("plan_id", plan_id), js, float(np.mean(js)), not violations, tuple(violations))


def choose_plan(plans: Sequence[PlanEconomics]):
    """u* = argmin_{u ∈ U_safe} J(u); небезопасные планы исключены до сравнения."""
    safe = [p for p in plans if p.safe]
    excluded = {p.plan_id: p.violations for p in plans if not p.safe}
    if not safe:
        return result(None, F_LIFECYCLE, excluded=excluded, reason="no plan in U_safe")
    best = min(safe, key=lambda p: (p.expected, p.plan_id))
    return result(best.plan_id, F_LIFECYCLE, objective={p.plan_id: p.expected for p in safe}, excluded=excluded)


def npv(benefits: Sequence[float], costs: Sequence[float], *, rate: float) -> float:
    r = number("discount_rate", rate, low=-1.0, strict=True)
    if len(benefits) != len(costs) or not benefits:
        raise EconomicsError("benefits and costs must be non-empty series of one length")
    return float(sum((number("benefit", b) - number("cost", c)) / (1.0 + r) ** t
                     for t, (b, c) in enumerate(zip(benefits, costs))))


def risk_adjusted_npv(npv_samples: Sequence[float], loss_samples: Sequence[float], *, risk_lambda: float,
                      alpha: float, scenario_ids: Optional[Sequence[str]] = None):
    """RiskAdjustedNPV = E[NPV] − λ·CVaR_α(Loss); NPV и потери — по одним и тем же сценариям."""
    n = np.asarray(npv_samples, float)
    if n.size != len(loss_samples) or n.size < 2:
        raise EconomicsError("NPV and loss samples must come from the same ≥ 2 scenarios")
    lam = number("risk_lambda", risk_lambda, low=0.0)
    var, cv = cvar(loss_samples, alpha)
    value = float(n.mean()) - lam * cv
    return result(value, F_NPV, expected_npv=float(n.mean()), var=var, cvar=cv, risk_lambda=lam, alpha=alpha)


__all__ = ["COST_COMPONENTS", "PlanEconomics", "choose_plan", "discounted_cost", "lifecycle_objective", "npv",
           "risk_adjusted_npv"]
