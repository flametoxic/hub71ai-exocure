"""StateFusionService (мастер-ТЗ A4; WM Spec §2.5): слитое состояние принадлежит сервису слияния, сырое
наблюдение никогда не становится истиной напрямую.

    p(x_t | z_1:t) ∝ p(z_t | x_t) ∫ p(x_t | x_{t−1}, u_t) p(x_{t−1} | z_1:t−1) dx_{t−1}
    x̂ = Σ z_i/σ_i² / Σ 1/σ_i²,  σ̂² = 1/Σ 1/σ_i²,  σ_i² = σ²_sensor,i + σ²_trust,i
    Kalman: x̂⁻ = F x̂ + B u;  P⁻ = F P F + Q;  K = P⁻H/(H P⁻ H + R);  x̂ = x̂⁻ + K(z − H x̂⁻);  P = (1 − K H) P⁻
    Дрейф: r_t = z_t − H x̂⁻,  S_t = H P⁻ H + R;  дрейф, если > 30 % из последних 20 невязок превышают 3√S_t →
    дисперсия датчика ×10, sensor_health = degraded, источник не удаляется.
σ²_trust = базовая дисперсия доверия источника (политика) + скорость роста × возраст наблюдения (устаревший
датчик весит меньше). Результат — утверждение INFERRED, которое пишет ProjectionPipeline.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Sequence

from .policy import F_DRIFT, F_IVW, F_KALMAN, SpatialWorldError, SpatialWorldPolicy, aware, finite, text


@dataclass(frozen=True)
class Reading:
    sensor_id: str
    value: float
    sensor_variance: float           # σ²_sensor из паспорта
    observed_at: datetime


@dataclass
class FusedEstimate:
    entity_id: str
    attribute: str
    value: float
    variance: float
    method: str
    contributing_sources: list
    effective_variance: dict
    sensor_health: dict
    at: datetime
    events: list = field(default_factory=list)
    formula: tuple = (F_IVW,)


def inverse_variance_weighted(z: Sequence[float], sigma2: Sequence[float]) -> tuple[float, float]:
    if not z or len(z) != len(sigma2) or any(s <= 0 for s in sigma2):
        raise SpatialWorldError("IVW needs equal non-empty lists and positive variances")
    w = [1.0 / s for s in sigma2]
    return sum(wi * zi for wi, zi in zip(w, z)) / sum(w), 1.0 / sum(w)


def categorical_fusion(votes: Mapping[str, str], weights: Mapping[str, float]) -> dict:
    tally: dict[str, float] = {}
    for src, v in votes.items():
        tally[v] = tally.get(v, 0.0) + float(weights[src])
    total = sum(tally.values())
    best = max(tally, key=tally.get)
    tied = [k for k, w in tally.items() if w == tally[best]]
    return {"value": best if len(tied) == 1 else None, "confidence": tally[best] / total, "tally": tally,
            "status": "fused" if len(tied) == 1 else "conflicted"}


@dataclass
class _Track:
    x: float
    p: float
    q_per_s: float
    t: datetime
    residuals: dict = field(default_factory=dict)     # sensor → deque[bool] (превышение 3√S)
    inflation: dict = field(default_factory=dict)     # sensor → накопленный множитель (10, 100, …)

    @property
    def inflated(self) -> set:
        return set(self.inflation)


class StateFusionService:
    def __init__(self, policy: SpatialWorldPolicy) -> None:
        self.policy = policy
        self._tracks: dict[tuple, _Track] = {}

    def sigma2(self, r: Reading, now: datetime) -> float:
        base = self.policy.source_trust_variance.get(r.sensor_id)
        if base is None:
            raise SpatialWorldError(f"no trust variance for source {r.sensor_id} in policy (fail-closed)")
        age = max(0.0, (now - aware("observed_at", r.observed_at)).total_seconds())
        return finite("sensor_variance", r.sensor_variance) + float(base) + self.policy.staleness_variance_rate * age

    def fuse_static(self, entity_id: str, attribute: str, readings: Sequence[Reading], *, now: datetime) -> FusedEstimate:
        now = aware("now", now)
        s2 = {r.sensor_id: self.sigma2(r, now) for r in readings}
        x, v = inverse_variance_weighted([r.value for r in readings], [s2[r.sensor_id] for r in readings])
        return FusedEstimate(entity_id, attribute, x, v, "inverse_variance", [r.sensor_id for r in readings], s2,
                             {r.sensor_id: "ok" for r in readings}, now)

    def start_track(self, entity_id: str, attribute: str, *, x0: float, p0: float, process_noise_per_s: float,
                    at: datetime) -> None:
        if finite("p0", p0) <= 0 or finite("process_noise_per_s", process_noise_per_s) < 0:
            raise SpatialWorldError("p0 > 0 and process noise ≥ 0 are required (dynamics pack or site policy)")
        self._tracks[(text("entity_id", entity_id), text("attribute", attribute))] = _Track(
            finite("x0", x0), float(p0), float(process_noise_per_s), aware("at", at))

    def step(self, entity_id: str, attribute: str, readings: Sequence[Reading], *, now: datetime,
             f: float = 1.0, bu: float = 0.0) -> FusedEstimate:
        """Один шаг фильтра: прогноз (F, B·u — из пакета динамики, иначе случайное блуждание) и
        последовательное обновление по всем датчикам с проверкой дрейфа."""
        now = aware("now", now)
        tr = self._tracks.get((entity_id, attribute))
        if tr is None:
            raise SpatialWorldError("start_track first")
        dt = max(0.0, (now - tr.t).total_seconds())
        x_prior = f * tr.x + bu
        p_prior = f * tr.p * f + tr.q_per_s * dt
        dr = self.policy.drift
        window, ratio, k, infl = int(dr["window"]), float(dr["ratio"]), float(dr["k_sigma"]), float(dr["inflation"])
        events, eff, health = [], {}, {}
        # проверка дрейфа по априорному прогнозу (до обновлений этого шага)
        for r in readings:
            rv = self.sigma2(r, now) * tr.inflation.get(r.sensor_id, 1.0)
            s = p_prior + rv
            resid = r.value - x_prior
            q = tr.residuals.setdefault(r.sensor_id, deque(maxlen=window))
            q.append(abs(resid) > k * math.sqrt(s))
            if len(q) == window and sum(q) / window > ratio:
                # правило ТЗ применяется каждый раз, когда окно снова показывает дрейф: ×10 к уже раздутой
                # дисперсии (дрейф растёт — вес падает дальше); источник не удаляется
                first = r.sensor_id not in tr.inflation
                tr.inflation[r.sensor_id] = tr.inflation.get(r.sensor_id, 1.0) * infl
                q.clear()
                events.append({"event": "sensor_drift" if first else "sensor_drift_persisting",
                               "sensor_id": r.sensor_id, "entity_id": entity_id, "attribute": attribute,
                               "at": now.isoformat(), "sensor_health": "degraded",
                               "variance_inflation": tr.inflation[r.sensor_id], "source_deleted": False,
                               "formula": F_DRIFT.formula_id})
        x, p = x_prior, p_prior
        for r in readings:
            rv = self.sigma2(r, now) * tr.inflation.get(r.sensor_id, 1.0)
            eff[r.sensor_id] = rv
            health[r.sensor_id] = "degraded" if r.sensor_id in tr.inflated else "ok"
            kg = p / (p + rv)
            x, p = x + kg * (r.value - x), (1 - kg) * p
        tr.x, tr.p, tr.t = x, p, now
        return FusedEstimate(entity_id, attribute, x, p, "kalman", [r.sensor_id for r in readings], eff, health, now,
                             events, (F_KALMAN, F_DRIFT))

    def degraded_sensors(self, entity_id: str, attribute: str) -> set:
        tr = self._tracks.get((entity_id, attribute))
        return set() if tr is None else set(tr.inflated)


__all__ = ["FusedEstimate", "Reading", "StateFusionService", "categorical_fusion", "inverse_variance_weighted"]
