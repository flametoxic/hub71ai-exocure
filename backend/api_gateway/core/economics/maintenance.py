"""Оптимизация обслуживания (§1.3): не «насос нездоров», а конкретное решение с ценой, уверенностью и допущениями.

Q_a(u) = C_action(u) + E[C_failure | u] + E[C_energy_degradation | u] + C_service_impact(u),
u ∈ {maintain_now, defer, derate, replace, inspect}.
Ожидания — по выборкам сценариев (одинаковые сценарии для всех вариантов — общие случайные числа), поэтому
уверенность = доля сценариев, где выбранный вариант дешевле отсрочки, а «если отложить» — распределение.
Вероятность отказа можно брать из пакета AssetDegradationDynamics UMRM (Вейбулл с восстановлением):
weibull_failure_probability ниже повторяет его формулу для окна.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ._base import F_MAINTENANCE, EconomicsError, number, result, text, texts

ACTIONS = ("maintain_now", "defer", "derate", "replace", "inspect")


@dataclass(frozen=True)
class MaintenanceOption:
    action: str
    when_days: float
    action_cost: float
    failure_probability: tuple[float, ...]        # P(отказ в окне | u) по сценариям
    failure_cost: tuple[float, ...]               # стоимость отказа по тем же сценариям
    energy_degradation_cost: tuple[float, ...]
    service_impact_cost: float

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise EconomicsError(f"action must be one of {ACTIONS}")
        number("when_days", self.when_days, low=0.0)
        number("action_cost", self.action_cost, low=0.0)
        number("service_impact_cost", self.service_impact_cost, low=0.0)
        n = len(self.failure_probability)
        if n == 0 or len(self.failure_cost) != n or len(self.energy_degradation_cost) != n:
            raise EconomicsError("scenario samples must be non-empty and of one length")
        for p in self.failure_probability:
            number("failure_probability", p, low=0.0, high=1.0)

    def q_samples(self) -> np.ndarray:
        p, cf, ce = (np.asarray(x, float) for x in (self.failure_probability, self.failure_cost,
                                                     self.energy_degradation_cost))
        return self.action_cost + p * cf + ce + self.service_impact_cost


def weibull_failure_probability(*, virtual_age_h: float, window_h: float, load_ratio: float, restoration: float,
                                shape: float, scale_h: float, acceleration_exponent: float) -> float:
    """Та же модель, что asset_degradation_step (UMRM): возраст после восстановления (1−q)·age, ускорение load^n,
    P(отказ в окне) = 1 − exp(−[(a1/η)^β − (a0/η)^β])."""
    age = number("virtual_age_h", virtual_age_h, low=0.0)
    q = number("restoration", restoration, low=0.0, high=1.0)
    beta = number("shape", shape, low=0.0, strict=True)
    eta = number("scale_h", scale_h, low=0.0, strict=True)
    a0 = (1.0 - q) * age
    a1 = a0 + number("load_ratio", load_ratio, low=0.0) ** number("acceleration_exponent", acceleration_exponent,
                                                                    low=0.0) * number("window_h", window_h, low=0.0)
    return float(1.0 - np.exp(-((a1 / eta) ** beta - (a0 / eta) ** beta)))


@dataclass(frozen=True)
class MaintenanceRecommendation:
    asset: str
    action: str
    when_days: float
    expected_q: float
    expected_avoided_failure_cost: float
    confidence: float
    assumptions: tuple[str, ...]
    deferral_impact: Mapping[str, float]
    ranking: tuple[tuple[str, float], ...]
    evidence_refs: tuple[str, ...]
    epistemic_status: str = "predicted"
    action_authority: str = "none"


def recommend(asset: str, options: Sequence[MaintenanceOption], *, assumptions: Sequence[str],
              evidence_refs: Sequence[str], quantiles: Sequence[float]) -> MaintenanceRecommendation:
    """Выбор u с минимальным E[Q_a(u)]; уверенность и последствия отсрочки — по сценариям."""
    asset = text("asset", asset)
    if not assumptions:
        raise EconomicsError("a maintenance recommendation must state its assumptions (§1.3)")
    refs = texts("evidence_refs", evidence_refs)
    if not refs:
        raise EconomicsError("a maintenance recommendation must cite its evidence")
    by = {o.action: o for o in options}
    if len(by) != len(options):
        raise EconomicsError("one option per action")
    if "defer" not in by:
        raise EconomicsError("the deferral option is required: its consequences are part of the answer")
    n = {len(o.failure_probability) for o in options}
    if len(n) != 1:
        raise EconomicsError("all options must be evaluated on the same scenarios")
    qs = {a: o.q_samples() for a, o in by.items()}
    ranking = tuple(sorted(((a, float(v.mean())) for a, v in qs.items()), key=lambda kv: (kv[1], kv[0])))
    best = ranking[0][0]
    o, d = by[best], by["defer"]
    avoided = float(np.mean(np.asarray(d.failure_probability) * np.asarray(d.failure_cost)
                            - np.asarray(o.failure_probability) * np.asarray(o.failure_cost)))
    confidence = float(np.mean(qs[best] < qs["defer"])) if best != "defer" else float(
        np.mean(np.all([qs["defer"] <= v for a, v in qs.items() if a != "defer"], axis=0)))
    diff = qs["defer"] - qs[best]
    impact = {f"q{int(round(float(q) * 100))}": float(np.quantile(diff, float(q))) for q in quantiles}
    impact["mean"] = float(diff.mean())
    return MaintenanceRecommendation(asset, best, o.when_days, float(qs[best].mean()), avoided, confidence,
                                     tuple(assumptions), impact, ranking, refs)


def q_value(option: MaintenanceOption):
    q = option.q_samples()
    return result(float(q.mean()), F_MAINTENANCE, action=option.action, samples=len(q))


__all__ = ["ACTIONS", "MaintenanceOption", "MaintenanceRecommendation", "q_value", "recommend",
           "weibull_failure_probability"]
