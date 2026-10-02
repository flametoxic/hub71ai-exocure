"""Мастер-ТЗ B1a (стр. 8–9): тепловая зона 1R1C — теплопритоки, шаг Эйлера, калибровка с ограничениями.

    C dT/dt = (T_out − T)/R_env + ṁ c_p (T_supply − T) + Q_solar + Q_occ + Q_equip
    T(t+Δt) = T(t) + Δt/C · [(T_out − T)/R_env + ṁ c_p (T_supply − T) + Q_solar + Q_occ]
    Q_solar = k_solar · GHI · A_window,   Q_occ = k_occ · occupancy
    θ* = argmin_θ Σ_t (T_obs(t+Δt) − T_model(t+Δt | θ))²
    Ограничения: C > 0, R_env > 0, k_solar ≥ 0, k_occ ≥ 0. История ≥ 72 ч. Хранить RMSE и версию.
    Приёмка B1a: RMSE прогноза на час ≤ порога политики (ТЗ: 0.8 °C) на отложенной телеметрии.

Q_equip не входит в шаг Эйлера в тексте ТЗ; если площадка его передаёт, он добавляется так же, как в
производной (передаётся явно, по умолчанию 0 — отсутствие данных, а не придуманное значение).
Результат калибровки — версия-кандидат, в продакшен сама не попадает.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Optional, Sequence

import numpy as np

from ._base import PhysicsError, need, num, ref

F_SOLAR = ref("MASTER-B1-Q-SOLAR", 8, "Q_solar = k_solar·GHI·A_window")
F_OCC = ref("MASTER-B1-Q-OCC", 8, "Q_occ = k_occ·occupancy")
F_EULER = ref("MASTER-B1-EULER", 8, "T(t+Δt) = T(t) + Δt/C·[(T_out−T)/R_env + ṁc_p(T_supply−T) + Q_solar + Q_occ]")
F_CALIB = ref("MASTER-B1-CALIBRATION", 9, "θ* = argmin_θ Σ_t (T_obs(t+Δt) − T_model(t+Δt|θ))²; C>0, R_env>0, "
                                          "k_solar≥0, k_occ≥0; ≥72h history; store RMSE and version")
MIN_HISTORY_H = 72.0          # мастер-ТЗ стр. 9: «Minimum 72h history»
PARAMS = ("thermal_capacitance", "envelope_resistance", "k_solar", "k_occ")


def solar_gain(*, k_solar: float, ghi_w_m2: float, window_area_m2: float) -> float:
    return num("k_solar", k_solar, low=0.0) * num("GHI", ghi_w_m2, low=0.0) * num("A_window", window_area_m2, low=0.0)


def occupancy_gain(*, k_occ: float, occupancy: float) -> float:
    return num("k_occ", k_occ, low=0.0) * num("occupancy", occupancy, low=0.0)


def euler_step(*, temperature: float, dt_s: float, thermal_capacitance: float, envelope_resistance: float,
               outdoor_temperature: float, mass_flow_kg_s: float, specific_heat: float, supply_temperature: float,
               q_solar: float, q_occ: float, q_equip: float = 0.0) -> float:
    c = num("C", thermal_capacitance, low=0.0, strict_low=True)
    r = num("R_env", envelope_resistance, low=0.0, strict_low=True)
    t = num("T", temperature)
    flux = (num("T_out", outdoor_temperature) - t) / r \
        + num("m_dot", mass_flow_kg_s, low=0.0) * num("c_p", specific_heat, low=0.0, strict_low=True) \
        * (num("T_supply", supply_temperature) - t) + num("Q_solar", q_solar) + num("Q_occ", q_occ) + num("Q_equip", q_equip)
    return t + num("dt", dt_s, low=0.0, strict_low=True) / c * flux


@dataclass(frozen=True)
class ThermalSample:
    """Один шаг телеметрии зоны: состояние в t и наблюдение в t + Δt."""
    at: datetime
    dt_s: float
    temperature: float
    next_temperature: float
    outdoor_temperature: float
    mass_flow_kg_s: float
    supply_temperature: float
    ghi_w_m2: float
    occupancy: float
    q_equip: float = 0.0


@dataclass
class CalibrationResult:
    zone_id: str
    theta: dict
    theta_initial: dict
    rmse_fit: float
    rmse_initial: float
    rmse_holdout: Optional[float]
    b1a_threshold: Optional[float]
    b1a_pass: Optional[bool]
    history_hours: float
    n_samples: int
    version: str
    status: str = "candidate"                    # в продакшен — только через гейт обучения и одобрение
    formula: str = F_CALIB.formula_id
    notes: list = field(default_factory=list)


def _predict(theta: np.ndarray, s: ThermalSample, cp: float, area: float) -> float:
    c, r, ks, ko = theta
    return euler_step(temperature=s.temperature, dt_s=s.dt_s, thermal_capacitance=c, envelope_resistance=r,
                      outdoor_temperature=s.outdoor_temperature, mass_flow_kg_s=s.mass_flow_kg_s, specific_heat=cp,
                      supply_temperature=s.supply_temperature,
                      q_solar=solar_gain(k_solar=ks, ghi_w_m2=s.ghi_w_m2, window_area_m2=area),
                      q_occ=occupancy_gain(k_occ=ko, occupancy=s.occupancy), q_equip=s.q_equip)


def _rmse(theta, samples, cp, area) -> float:
    e = [_predict(theta, s, cp, area) - s.next_temperature for s in samples]
    return float(np.sqrt(np.mean(np.square(e))))


def calibrate_zone(zone_id: str, samples: Sequence[ThermalSample], *, site: Mapping, initial: Mapping,
                   holdout: Sequence[ThermalSample] = (), b1a_rmse_threshold: Optional[float] = None) -> CalibrationResult:
    """site: {specific_heat, window_area_m2} (паспорт зоны); initial: стартовые C, R_env, k_solar, k_occ."""
    from scipy.optimize import least_squares

    samples = sorted(samples, key=lambda s: s.at)
    if len(samples) < len(PARAMS) + 1:
        raise PhysicsError("not enough samples to identify C, R_env, k_solar, k_occ")
    span_h = (samples[-1].at - samples[0].at).total_seconds() / 3600.0 + samples[-1].dt_s / 3600.0
    if span_h < MIN_HISTORY_H:
        raise PhysicsError(f"calibration needs ≥ {MIN_HISTORY_H:.0f} h of history, got {span_h:.1f} h (master TZ B1a)")
    cp = num("specific_heat", need(site, "specific_heat", "site"), low=0.0, strict_low=True)
    area = num("window_area_m2", need(site, "window_area_m2", "site"), low=0.0)
    x0 = np.array([num(p, need(initial, p, "initial")) for p in PARAMS], float)
    if x0[0] <= 0 or x0[1] <= 0 or x0[2] < 0 or x0[3] < 0:
        raise PhysicsError("initial values violate the bounds C>0, R_env>0, k_solar≥0, k_occ≥0")
    tiny = np.finfo(float).tiny
    lo, hi = np.array([tiny, tiny, 0.0, 0.0]), np.full(4, np.inf)
    scale = np.maximum(np.abs(x0), 1e-12)

    def resid(z):
        th = z * scale
        return np.array([_predict(th, s, cp, area) - s.next_temperature for s in samples])

    fit = least_squares(resid, x0 / scale, bounds=(lo / scale, hi / scale), x_scale="jac")
    theta = fit.x * scale
    rh = _rmse(theta, holdout, cp, area) if holdout else None
    thr = None if b1a_rmse_threshold is None else num("b1a_rmse_threshold", b1a_rmse_threshold, low=0.0, strict_low=True)
    th = dict(zip(PARAMS, map(float, theta)))
    body = {"zone": zone_id, "theta": th, "n": len(samples), "first": samples[0].at.isoformat(),
            "last": samples[-1].at.isoformat()}
    version = "thermal1r1c:" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    notes = [] if fit.success else [f"optimizer: {fit.message}"]
    if holdout == ():
        notes.append("no holdout: B1a acceptance not evaluated")
    return CalibrationResult(zone_id, th, dict(zip(PARAMS, map(float, x0))), _rmse(theta, samples, cp, area),
                             _rmse(x0, samples, cp, area), rh, thr,
                             None if (rh is None or thr is None) else rh <= thr, span_h, len(samples), version,
                             notes=notes)


__all__ = ["CalibrationResult", "F_CALIB", "F_EULER", "F_OCC", "F_SOLAR", "MIN_HISTORY_H", "ThermalSample",
           "calibrate_zone", "euler_step", "occupancy_gain", "solar_gain"]
