"""Формулы City WM (§3–§10, §14). Все пороги и веса — из политики площадки/города; коэффициентов по умолчанию нет.

    d_t = [weather_t, event_t, holiday_t, tariff_t, VIP_t];  x_{t+1} = f_θ(x_t, u_t, d_t)  (d_t — конфаундеры)
    P_building = P_HVAC + P_lighting + P_elevators + P_retail + P_other
    Q_cooling ≈ ṁ c_p (T_return − T_supply);  cooling_anomaly = 1[Q_expected − Q_observed > τ_Q]
    comfort_i = g(|T_i − T_i^pref|, CO2_i, humidity_i, noise_i)  (g задаёт площадка; энергосбережение не нарушает минимумы)
    ρ_e(t+Δt) = ρ_e(t) + Δt/L_e (q_e^in − q_e^out),  q_e = ρ_e v_e
    AV_allowed = 1[v2x_latency_ms ≤ τ_v2x ∧ perception_confidence ≥ τ_c]  (без фиксированного «20 мс» — только из safety case/ODD)
    T_walk = T_min + ΔT_crossing + ΔT_mobility + ΔT_crowd;  ΔT_mobility = α·P(mobility_aid) + β·P(heavy_items)
    entrance_admission ≤ platform_safe_capacity − platform_occupancy
    A = MTBF/(MTBF + MTTR);  L_e2e = L_sensor + L_network + L_queue + L_inference + L_actuation
    r_t = λ_E E_saved + λ_C CO2_saved + λ_M Δflow + λ_H comfort − λ_S S_risk − λ_D degradation − λ_P privacy…
          (последний член обрезан в PDF; безопасность и приватность — прежде всего жёсткие ограничения)
    TL_1 = Green ⇒ TL_2 ≠ Green;   T_pedestrian ≥ T_crossing^min + T_clearance
    send_to_LLM(d) = 1 ⇒ mask(d) = 1 ∧ purpose_authorized(d) = 1
    u_t ∈ U_allowed ⟺ physical_safe(u_t) ∧ policy_valid(u_t) ∧ consent_valid(u_t) ∧ authorized(u_t)
"""
from __future__ import annotations

import math
from typing import Callable, Mapping

from ..world_model.reality_formulas import FormulaReference
from .schema import DOC, CitySchemaError

C_P_WATER_J_PER_KG_K = 4186.0          # теплоёмкость воды — физическая константа, не настройка


def _f(fid: str, page: int, expr: str) -> FormulaReference:
    return FormulaReference(fid, DOC, page, expr)


F_EXOGENOUS = _f("CITY-3-EXOGENOUS", 4, "d_t = [weather, event, holiday, tariff, VIP]; x_{t+1} = f_θ(x_t, u_t, d_t)")
F_COOLING = _f("CITY-4.2-COOLING", 5, "Q_cooling ≈ ṁ c_p (T_return − T_supply); anomaly = 1[Q_exp − Q_obs > τ_Q]")
F_TRAFFIC = _f("CITY-5.2-TRAFFIC", 7, "ρ(t+Δt) = ρ(t) + Δt/L (q_in − q_out); q = ρ v")
F_AV = _f("CITY-5.3-AV", 7, "AV_allowed = 1[v2x_latency_ms ≤ τ_v2x ∧ perception_confidence ≥ τ_c]")
F_WALK = _f("CITY-5.4-WALK", 8, "T_walk = T_min + ΔT_crossing + ΔT_mobility + ΔT_crowd; ΔT_mobility = αP(aid) + βP(items)")
F_CROWD = _f("CITY-5.5-CROWD", 8, "entrance_admission ≤ platform_safe_capacity − platform_occupancy")
F_REWARD = _f("CITY-9.4-REWARD", 13, "r_t = λ_E E + λ_C CO2 + λ_M Δflow + λ_H comfort − λ_S S_risk − λ_D degradation − λ_P priva… (cut off)")
F_ALLOWED = _f("CITY-14-ALLOWED", 16, "u ∈ U_allowed ⟺ physical_safe ∧ policy_valid ∧ consent_valid ∧ authorized")


def exogenous_vector(weather: float, event: float, holiday: float, tariff: float, vip: float) -> dict:
    """d_t логируется как конфаундеры (влияют и на действие оператора, и на исход), а не как простые признаки."""
    return {"d_t": [weather, event, holiday, tariff, vip], "role": "confounders", "formula": F_EXOGENOUS}


def building_power(p_hvac: float, p_lighting: float, p_elevators: float, p_retail: float, p_other: float) -> float:
    return p_hvac + p_lighting + p_elevators + p_retail + p_other


def cooling_load_w(m_dot_kg_s: float, t_return_c: float, t_supply_c: float) -> float:
    if m_dot_kg_s < 0:
        raise CitySchemaError("mass flow must be ≥ 0")
    return m_dot_kg_s * C_P_WATER_J_PER_KG_K * (t_return_c - t_supply_c)


def cooling_anomaly(q_expected_w: float, q_observed_w: float, *, tau_q_w: float) -> dict:
    return {"anomaly": (q_expected_w - q_observed_w) > tau_q_w, "gap_w": q_expected_w - q_observed_w, "formula": F_COOLING}


def comfort(g: Callable[..., float], *, t: float, t_pref: float, co2: float, humidity: float, noise: float,
            minimums: Mapping[str, tuple]) -> dict:
    """g — функция площадки; минимально допустимые параметры комфорта/здоровья — жёсткие пределы."""
    violations = [k for k, (lo, hi) in minimums.items()
                  if (v := {"temperature": t, "co2": co2, "humidity": humidity, "noise": noise}.get(k)) is not None
                  and not (lo <= v <= hi)]
    return {"comfort": float(g(abs(t - t_pref), co2, humidity, noise)), "hard_limit_violations": violations,
            "energy_saving_allowed": not violations}


def traffic_step(rho: float, q_in: float, q_out: float, *, dt_s: float, length_m: float) -> float:
    if length_m <= 0 or dt_s <= 0:
        raise CitySchemaError("segment length and Δt must be positive")
    return max(0.0, rho + dt_s / length_m * (q_in - q_out))


def flow(rho: float, v: float) -> float:
    return rho * v


def av_allowed(v2x_latency_ms: float, perception_confidence: float, *, tau_v2x_ms: float, tau_c: float,
               safety_case_ref: str) -> dict:
    if not str(safety_case_ref).strip():
        raise CitySchemaError("τ thresholds require a domain safety case / ODD reference (no fixed 20 ms rule)")
    return {"allowed": v2x_latency_ms <= tau_v2x_ms and perception_confidence >= tau_c, "formula": F_AV,
            "safety_case": safety_case_ref}


def walk_time(*, t_min: float, dt_crossing: float, p_mobility_aid: float, p_heavy_items: float, alpha: float,
              beta: float, dt_crowd: float) -> dict:
    values = {
        "t_min": t_min,
        "dt_crossing": dt_crossing,
        "p_mobility_aid": p_mobility_aid,
        "p_heavy_items": p_heavy_items,
        "alpha": alpha,
        "beta": beta,
        "dt_crowd": dt_crowd,
    }
    checked = {}
    for name, value in values.items():
        try:
            checked[name] = float(value)
        except (TypeError, ValueError) as exc:
            raise CitySchemaError(f"{name} must be a finite number") from exc
        if not math.isfinite(checked[name]):
            raise CitySchemaError(f"{name} must be a finite number")
    dm = checked["alpha"] * checked["p_mobility_aid"] + checked["beta"] * checked["p_heavy_items"]
    total = checked["t_min"] + checked["dt_crossing"] + dm + checked["dt_crowd"]
    return {"t_walk_s": total, "dt_mobility_s": dm, "formula": F_WALK,
            "privacy": "anonymous short-lived perception token only; identity not stored to extend a phase"}


def crowd_admission_limit(platform_safe_capacity: int, platform_occupancy: int) -> int:
    return max(0, int(platform_safe_capacity) - int(platform_occupancy))


def e2e_latency(*, sensor: float, network: float, queue: float, inference: float, actuation: float) -> float:
    return sensor + network + queue + inference + actuation


def reward(terms: Mapping[str, float], lambdas: Mapping[str, float], *, hard_constraints_ok: bool) -> dict:
    """Безопасность и приватность — сначала жёсткие ограничения: при нарушении награда не считается вовсе."""
    pos = ("energy_saved", "co2_saved", "delta_flow", "comfort")
    neg = ("safety_risk", "degradation", "privacy")
    if set(lambdas) != set(pos + neg) or set(terms) != set(pos + neg):
        raise CitySchemaError(f"reward needs terms and λ for {pos + neg}")
    if not hard_constraints_ok:
        return {"reward": None, "status": "hard_constraint_violated", "formula": F_REWARD}
    r = sum(lambdas[k] * terms[k] for k in pos) - sum(lambdas[k] * terms[k] for k in neg)
    return {"reward": r, "status": "ok", "formula": F_REWARD}


def u_allowed(*, physical_safe: bool, policy_valid: bool, consent_valid: bool, authorized: bool) -> dict:
    checks = {"physical_safe": physical_safe, "policy_valid": policy_valid, "consent_valid": consent_valid,
              "authorized": authorized}
    return {"allowed": all(v is True for v in checks.values()), "failed": [k for k, v in checks.items() if v is not True],
            "formula": F_ALLOWED}


def hard_rule_signal_conflict(tl1_color: str, tl2_color: str) -> bool:
    """TL_1 = Green ⇒ TL_2 ≠ Green (для конфликтующих подходов)."""
    return not (tl1_color == "green" and tl2_color == "green")


def hard_rule_pedestrian(t_pedestrian_s: float, *, t_crossing_min_s: float, t_clearance_s: float) -> bool:
    return t_pedestrian_s >= t_crossing_min_s + t_clearance_s


def privacy_gate_llm(*, masked: bool, purpose_authorized: bool) -> bool:
    return bool(masked) and bool(purpose_authorized)


__all__ = [n for n in dir() if n.startswith("F_") or n in (
    "C_P_WATER_J_PER_KG_K", "av_allowed", "building_power", "comfort", "cooling_anomaly", "cooling_load_w",
    "crowd_admission_limit", "e2e_latency", "exogenous_vector", "flow", "hard_rule_pedestrian",
    "hard_rule_signal_conflict", "privacy_gate_llm", "reward", "traffic_step", "u_allowed", "walk_time")]
