"""V5 "20 different families": in one week twenty different families land — a wheelchair, a newborn, no dust,
a night flight, heavy luggage. Each has its own intelligence; the city receives only physical constraints:
pm10_ugm3_max, outdoor_harsh_min_max, accessible_vehicle and the mobility of the slowest member.

Without coordination and with CURE — on the same random numbers:
  way home: minutes outside and minutes in heat/dust above EACH family's limit
     (city_world.walk_time with P(mobility_aid)/P(heavy_items), comfort hard limits, PM10);
     without CURE — walk to the rank and wait for a taxi outside (ramp cars take longer);
     with CURE — the car waits at the covered exit; waiting only if cars run short that hour;
  centres (medical, ICP): daily capacity minus other residents; admission — crowd_admission_limit;
     without coordination — wasted trips and crowds, with CURE — booking the nearest free day;
  home: without coordination the flat is hot on arrival; with CURE — pre-cooling (apartment.preload by landing hour),
     grid-peak load only where off-peak is not enough.
All demo synthetic, labelled "vision", nothing is executed.
"""
from __future__ import annotations

import numpy as np

from api_gateway.core.city_world.formulas import crowd_admission_limit

from .core import MissingData, Store, answer, interval, refused, sample
from .day import harsh_mask, walk_minutes


def _centers(arr: np.ndarray, centers: list[str], cap: dict, bg: dict, rng, coordinated: bool, horizon: int) -> dict:
    """Families passing through centres in order. Returns finish day, wasted trips, peak queue per day."""
    n = arr.size
    ready = np.floor(arr).astype(int) + 1               # the day after landing
    wasted = np.zeros(n, int)
    peak_crowd = 0
    for c in centers:
        booked = np.zeros(horizon, int)
        done = np.full(n, -1)
        if coordinated:
            for i in np.argsort(ready, kind="stable"):  # book the nearest free day, in readiness order
                d = ready[i]
                while crowd_admission_limit(cap[c], bg[c][d] + booked[d]) == 0:
                    d += 1
                booked[d] += 1
                done[i] = d
            peak_crowd = max(peak_crowd, int(booked.max()))
        else:
            waiting = set()
            for d in range(horizon):
                waiting |= {i for i in range(n) if ready[i] == d}
                if not waiting:
                    continue
                today = rng.permutation(sorted(waiting))  # who queues first — random
                peak_crowd = max(peak_crowd, len(today))
                k = crowd_admission_limit(cap[c], bg[c][d])
                for i in today[:k]:
                    done[i] = d
                for i in today[k:]:
                    wasted[i] += 1                        # came and did not get in
                waiting -= set(today[:k].tolist())
        if (done < 0).any():
            raise MissingData([f"cohort:horizon_too_short_for_{c}"])
        ready = done + 1
    return {"finish_day": ready - 1, "wasted_trips": wasted, "peak_crowd": peak_crowd}


def _overflow_wait(slots: np.ndarray, cap_per_hour: float) -> np.ndarray:
    """Family's order within its hour / capacity → hours to wait for the next free car."""
    order = np.zeros(slots.size, int)
    for s_ in np.unique(slots):
        idx = np.where(slots == s_)[0]
        order[idx] = np.arange(idx.size)
    return np.floor(order / cap_per_hour) * 60.0


def cohort(*, policy: Store, scenario: Store, climate: Store, queues: Store, cohort_store: Store,
           preload_by_hour: dict, extra_families: list | None = None, stress: bool = False,
           extra_ramp_cars: int = 0) -> dict:
    """preload_by_hour: landing hour → apartment.preload (1R1C) answer for that hour.
    extra_families — families added live (e.g. a judge's family); stress — a second wheelchair lands on the same
    day and hour as f-02; extra_ramp_cars — approved additional ramp cars."""
    try:
        q, runs = tuple(policy.get("mc.interval_q")), int(policy.get("mc.runs"))
        c = lambda k: scenario.get(f"cohort.{k}")
        fams = [dict(f) for f in scenario.raw("cohort.families")] + [dict(f) for f in (extra_families or [])]
        if stress:
            ref = next(f for f in fams if f["id"] == "f-02")
            for f in fams:
                if f["id"] == "f-13":
                    f["arrival_day"], f["arrival_h"] = ref["arrival_day"], ref["arrival_h"]
        n = len(fams)
        rng = np.random.default_rng(int(cohort_store.get("seed")))
        arr = np.asarray([f["arrival_day"] for f in fams], float)
        hours = np.asarray([f["arrival_h"] for f in fams], int)

        # ---- centres
        centers = list(c("centers"))
        cap = {k: int(queues.get(f"centers.{k}.capacity_per_day")) for k in centers}
        horizon = int(c("arrival_window_days")) + 60
        bg = {k: np.round(sample(queues.raw(f"centers.{k}.background_per_day"), rng, horizon)).astype(int)
              for k in centers}                                               # the same background load in both worlds
        seed_rng = int(rng.integers(1 << 31))
        base = _centers(arr, centers, cap, bg, np.random.default_rng(seed_rng), False, horizon)
        twin = _centers(arr, centers, cap, bg, np.random.default_rng(seed_rng), True, horizon)

        # ---- fleet: regular and ramp cars, requests by (day, hour)
        per_car = float(c("rides_per_hour_per_car"))
        access = np.asarray([bool(f["limits"]["accessible_vehicle"]) for f in fams])
        slot = arr.astype(int) * 24 + hours
        over = np.zeros(n)
        ramp_cars = float(c("accessible_fleet_size")) + extra_ramp_cars
        ramp_rate = float(c("ramp_rides_per_hour_per_car"))
        for mask, cap in ((access, ramp_cars * ramp_rate), (~access, float(c("fleet_size")) * per_car)):
            if mask.any():
                over[mask] = _overflow_wait(slot[mask], cap)

        # ---- home: pre-cooling by landing hour
        pv = {int(h): a["value"] for h, a in preload_by_hour.items() if a.get("calculation_status") == "computed"}
        missing = sorted({int(h) for h in hours} - set(pv))
        if missing:
            raise MissingData([f"preload:arrival_h={h}" for h in missing])
        peak = set(policy.get("grid.peak_hours"))
        kw = sample(scenario.raw("cohort.apartment_cooling_kw"), rng, n)
        u = rng.random(n)
        p_off = np.asarray([pv[h]["p_feasible_off_peak"] for h in hours])
        p_reach = np.asarray([pv[h]["p_reachable"] for h in hours])
        in_peak_base = np.asarray([h in peak for h in hours])                # switched on when entering peak hours
        in_peak_twin = (u >= p_off) & (u < p_reach)                         # not reachable off-peak, reachable with peak
        hot_twin = u >= p_reach                                             # not reachable at all
        day = arr.astype(int)
        peak_kw = lambda m: float(np.bincount(day[m], weights=kw[m]).max()) if m.any() else 0.0

        # ---- way home: minutes outside and in heat/dust above each family's limit
        month = scenario.get("arrival.month")
        t_out = climate.get(f"months.{month}.temperature_c")
        t_sd = climate.get(f"months.{month}.temperature_sd_c")
        hum, pm = climate.get(f"months.{month}.humidity_pct"), climate.get(f"months.{month}.pm10_ugm3")
        w = lambda world, k: scenario.get(f"arrival.{world}.{k}")
        rows = []
        for i, f in enumerate(fams):
            h, lim, who = int(f["arrival_h"]), f["limits"], f["slowest_member"]
            temp = t_out[h] + rng.normal(0.0, t_sd, runs)
            heat = harsh_mask(policy, temp, hum[h], 0.0, pm_limit=np.inf)
            dust = np.full(runs, lim["pm10_ugm3_max"] is not None and pm[h] > lim["pm10_ugm3_max"])
            bad = heat | dust
            res = {}
            for world in ("without", "with"):
                walk = walk_minutes(policy, who, w(world, "walk_to_taxi_m")) + walk_minutes(policy, who, w(world, "walk_home_m"))
                if world == "without":
                    spec = scenario.raw("cohort.accessible_taxi_wait_min_without") if lim["accessible_vehicle"] \
                        else scenario.raw("arrival.without.taxi_wait_min")
                    wait = sample(spec, rng, runs)
                else:
                    wait = sample(scenario.raw("arrival.with.taxi_wait_min"), rng, runs) + over[i]
                outdoor = walk + wait
                res[world] = {"outdoor": outdoor, "bad": np.where(bad, outdoor, 0.0)}
            limit = lim["outdoor_harsh_min_max"]
            hi = lambda x: float(np.quantile(x, q[1]))
            rows.append({
                "id": f["id"], "who": f["who"], "who_ru": f.get("who_ru", f["who"]), "arrival_day": int(f["arrival_day"]), "arrival_h": h,
                "sent_to_city": {k: v for k, v in lim.items() if v not in (None, False)},
                "outdoor_min": {k: float(np.median(res[k]["outdoor"])) for k in res},
                "harsh_min": {k: float(np.median(res[k]["bad"])) for k in res},
                "limit_min": limit,
                "limit_kept": ({k: hi(res[k]["bad"]) <= limit for k in res} if limit is not None else None),
                "hot_home": {"without": True, "with": bool(hot_twin[i])},
                "grid_peak": {"without": bool(in_peak_base[i]), "with": bool(in_peak_twin[i])},
                "_bad": {k: res[k]["bad"] for k in res}})
        with_limit = [r for r in rows if r["limit_kept"] is not None]
        tot = {k: np.sum([r["_bad"][k] for r in rows], axis=0) for k in ("without", "with")}
        for r in rows:
            r.pop("_bad")
        return answer({
            "families": n, "rows": rows,
            "limits": {"families_with_personal_limit": len(with_limit),
                       "kept": {k: sum(r["limit_kept"][k] for r in with_limit) for k in ("without", "with")}},
            "harsh_minutes_total": {k: interval(tot[k], q) for k in tot},
            "centers": {"wasted_trips": {"without": int(base["wasted_trips"].sum()), "with": int(twin["wasted_trips"].sum())},
                        "peak_crowd_per_day": {"without": base["peak_crowd"], "with": twin["peak_crowd"]},
                        "worst_family_days": {"without": float((base["finish_day"] - arr.astype(int)).max()),
                                              "with": float((twin["finish_day"] - arr.astype(int)).max())},
                        "note": "coordination creates no centre capacity: it removes wasted trips and crowds"},
            "fleet": {"accessible_families": int(access.sum()), "accessible_fleet_size": int(ramp_cars),
                      "families_waiting_for_a_car_with_cure": int((over > 0).sum())},
            "grid": {"hot_home_arrivals": {"without": n, "with": int(hot_twin.sum())},
                     "arrival_day_peak_kw": {"without": peak_kw(in_peak_base), "with": peak_kw(in_peak_twin)},
                     "chiller_peak_capacity_kw": float(c("chiller_peak_capacity_kw"))},
            "privacy": "the city receives only sent_to_city constraints; the \"who\" label is for demo viewers",
            "stress": stress, "ramp_cars": ramp_cars,
            "proposal": ({"action": "dispatch_ramp_car", "title": "second ramp car from the depot",
                          "families_waiting": [r["who"] for r, w in zip(rows, over) if w > 0],
                          "families_waiting_ru": [r["who_ru"] for r, w in zip(rows, over) if w > 0]}
                         if (over > 0).any() and access[over > 0].any() else None),
            "executed": False},
            unit="families", stores=[policy, scenario, climate, queues, cohort_store],
            formulas=["CITY walk_time (mobility aid, heavy items)", "CITY comfort hard limits", "PM10 limit per family",
                      "CITY crowd_admission_limit", "fleet capacity by hour", "RP-1R1C preload by arrival hour"],
            label="20 different families — vision")
    except MissingData as e:
        return refused(e, what="cohort")
