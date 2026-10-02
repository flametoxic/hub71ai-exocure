"""Flat twin → monthly cooling bill; pre-cooling for arrival.

CURE formulas: reality_physics.thermal_1r1c.solar_gain / occupancy_gain / euler_step.
Required cooling is the same 1R1C heat balance at dT/dt = 0:  Q = (T_out − T_set)/R + Q_solar + Q_occ + Q_equip.
"""
from __future__ import annotations

import numpy as np

from api_gateway.core.reality_physics.thermal_1r1c import euler_step, occupancy_gain, solar_gain

from .core import MissingData, Store, answer, interval, refused, sample

PARAMS = ("R", "k_solar", "k_occ", "COP")


def calendar_months(climate: Store) -> list[str]:
    return [m for m in climate.raw("months") if "_" not in m]  # "aug_dust" is a scenario, not a calendar month


def _priors(policy: Store, answers: dict) -> dict:
    pri = {p: dict(policy.raw(f"thermal_priors.{p}")) for p in PARAMS}
    for param, choice in (answers or {}).items():  # the person's answer narrows the parameter range
        pri[param] = dict(policy.raw(f"thermal_answers.{param}.{choice}"))
    return pri


def _solar(policy: Store, listing: dict, ghi_h: float) -> float:
    return sum(solar_gain(k_solar=1.0, ghi_w_m2=ghi_h, window_area_m2=w["area_m2"])
               * float(policy.get(f"orientation_factor.{w['orientation']}")) for w in listing["windows"])


def bill_samples(*, listing: dict, profile: Store, climate: Store, tariffs: Store, policy: Store, seed: int,
                 answers: dict | None = None, setpoint_c: float | None = None) -> dict:
    runs = int(policy.get("mc.runs"))
    t_set = profile.get("params.setpoint_c") if setpoint_c is None else setpoint_c
    tariff = tariffs.get(f"electricity_per_kwh.{listing['tariff_class']}")
    occ = profile.raw("occupancy_by_hour")
    rng = np.random.default_rng(seed)
    pri = _priors(policy, answers or {})
    s = {p: sample(pri[p], rng, runs) for p in PARAMS}
    months = {}
    for m in calendar_months(climate):
        t_out, ghi = climate.get(f"months.{m}.temperature_c"), climate.get(f"months.{m}.ghi_wm2")
        q_day = np.zeros(runs)
        for h in range(24):
            q_occ = np.array([occupancy_gain(k_occ=k, occupancy=occ[h]) for k in s["k_occ"]])
            q_need = (t_out[h] - t_set) / s["R"] + s["k_solar"] * _solar(policy, listing, ghi[h]) + q_occ + listing["equip_w"]
            q_day += np.maximum(q_need, 0.0) * 3600.0
        months[m] = q_day / s["COP"] / 3.6e6 * climate.get(f"months.{m}.days") * tariff
    return {"months": months, "total": sum(months.values()), "params": s}


def cooling_bill(*, listing: dict, profile: Store, climate: Store, tariffs: Store, policy: Store, seed: int,
                 answers: dict | None = None) -> dict:
    try:
        q = tuple(policy.get("mc.interval_q"))
        b = bill_samples(listing=listing, profile=profile, climate=climate, tariffs=tariffs, policy=policy, seed=seed,
                         answers=answers)
        s, total = b["params"], b["total"]
        # the parameter that moves the bill most → ask about it (variance sensitivity)
        sens = {p: float(np.corrcoef(s[p], total)[0, 1] ** 2) for p in PARAMS if np.std(s[p]) > 0}
        open_ = {p: v for p, v in sens.items() if p not in (answers or {})}
        ask = max(open_, key=open_.get) if open_ else None
        question = policy.get(f"questions.{ask}") if ask else None
        return answer({"months": {m: interval(v, q) for m, v in b["months"].items()},
                       "total_months_in_data": interval(total, q), "months_in_data": list(b["months"]),
                       "ask": ask, "question": question, "options": list(policy.raw(f"thermal_answers.{ask}")) if ask else [],
                       "sensitivity": sens, "status": "prior" if not answers else "narrowed"},
                      unit="AED", stores=[profile, climate, tariffs, policy],
                      formulas=["RP-1R1C solar_gain", "RP-1R1C occupancy_gain", "RP-1R1C balance dT/dt=0"],
                      label="Cooling bill — flat physics forecast")
    except MissingData as e:
        return refused(e, what="apartment")


def preload(*, listing: dict, profile: Store, climate: Store, policy: Store, month: str, arrival_h: int,
            seed: int) -> dict:
    """Latest cooling start such that T ≤ T_set + band on arrival with the AC off during the grid peak.
    If that is not reachable — how many kWh of cooling must be drawn in peak hours."""
    try:
        runs, q = int(policy.get("mc.vision_runs")), tuple(policy.get("mc.interval_q"))
        band, dt = policy.get("comfort.arrival_band_c"), policy.get("hvac.dt_s")
        m_dot, cp, t_sup = policy.get("hvac.mass_flow_kg_s"), policy.get("hvac.specific_heat"), policy.get("hvac.supply_c")
        peak = set(policy.get("grid.peak_hours"))
        t_set, t_out = profile.get("params.setpoint_c"), climate.get(f"months.{month}.temperature_c")
        ghi = climate.get(f"months.{month}.ghi_wm2")
        rng = np.random.default_rng(seed)
        R, C = sample(policy.raw("thermal_priors.R"), rng, runs), sample(policy.raw("thermal_priors.C"), rng, runs)
        ks = sample(policy.raw("thermal_priors.k_solar"), rng, runs)
        hours = [h % 24 for h in range(arrival_h)]

        def run_day(i: int, start_h: int | None, peak_off: bool) -> tuple[float, float]:
            t, peak_kwh = t_out[0], 0.0
            for k, h in enumerate(hours):
                on = start_h is not None and start_h <= k and not (peak_off and h in peak)
                q_sol = ks[i] * _solar(policy, listing, ghi[h])
                for _ in range(int(3600 / dt)):
                    if on and h in peak:
                        peak_kwh += m_dot * cp * max(t - t_sup, 0.0) * dt / 3.6e6
                    t = euler_step(temperature=t, dt_s=dt, thermal_capacitance=C[i], envelope_resistance=R[i],
                                   outdoor_temperature=t_out[h], mass_flow_kg_s=m_dot if on else 0.0,
                                   specific_heat=cp, supply_temperature=t_sup, q_solar=q_sol, q_occ=0.0)
            return t, peak_kwh

        target = t_set + band
        off_start, with_plan, need_peak, without = [], [], [], []
        for i in range(runs):
            without.append(run_day(i, None, True)[0])
            ok = [s for s in range(len(hours)) if run_day(i, s, True)[0] <= target]
            off_start.append(max(ok) if ok else np.nan)
            if ok:
                with_plan.append(run_day(i, max(ok), True)[0]); need_peak.append(0.0)
            else:
                ok2 = [s for s in range(len(hours)) if run_day(i, s, False)[0] <= target]
                need_peak.append(run_day(i, max(ok2), False)[1] if ok2 else np.nan)
                if ok2:
                    with_plan.append(run_day(i, max(ok2), False)[0])
        off_start, need_peak = np.asarray(off_start), np.asarray(need_peak)
        feasible, reachable = float(np.mean(~np.isnan(off_start))), float(np.mean(~np.isnan(need_peak)))
        return answer({"p_feasible_off_peak": feasible, "p_reachable": reachable, "arrival_h": arrival_h,
                       "latest_start_hour_off_peak": interval(off_start[~np.isnan(off_start)], q) if feasible else None,
                       "peak_kwh_needed": interval(need_peak[~np.isnan(need_peak)], q) if reachable else None,
                       "indoor_c_at_arrival_without_cooling": interval(without, q),
                       "indoor_c_at_arrival_with_plan": interval(with_plan, q) if with_plan else None},
                      unit="hour", stores=[profile, climate, policy], formulas=["RP-1R1C euler_step"],
                      label="Pre-cooling — forecast")
    except MissingData as e:
        return refused(e, what="preload")
