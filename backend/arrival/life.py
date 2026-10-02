"""Week of Leila: every scene is computed by CURE engines on the personal model and the synthetic week data.

Engines: city_world.walk_time / comfort (via day.py), reality_physics 1R1C (via apartment.py),
cost map (total_cost.py), centre queues (queues). Nothing is executed: actions are proposed via executor.py.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from .apartment import bill_samples, calendar_months, preload
from .core import MissingData, Store, answer, interval, refused, sample
from .day import day_samples, harsh_mask, walk_minutes
from .plan import _mean
from .total_cost import district_samples

WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def hhmm(minute: float) -> str:
    m = int(round(minute))
    return f"{m // 60:02d}:{m % 60:02d}"


def in_peak(minute: float, windows) -> bool:
    return any(a <= minute < b for a, b in windows)


def _forecast_day(week: Store, day: str) -> dict:
    for d in week.raw("forecast.days"):
        if d["date"] == day:
            return d
    raise MissingData([f"week:forecast.{day}"])


def _limits(profile: Store, policy: Store) -> dict:
    """Each person's PM10 limit: personal (derived) or the general policy limit."""
    general = float(policy.get("outdoor_limits.pm10_ugm3_max"))
    out = {}
    for p in profile.raw("household"):
        c = profile.raw("constraints").get(p["id"], {}) if "constraints" in profile.doc else {}
        pm = c.get("pm10_ugm3_max")
        out[p["id"]] = float(pm["value"] if isinstance(pm, dict) else pm) if pm is not None else general
    return out


# ---------------------------------------------------------------- morning departure
def best_departure(*, trip: dict, district: str, city: Store, policy: Store, seed: int) -> dict:
    runs, q = int(policy.get("mc.runs")), tuple(policy.get("mc.interval_q"))
    need, step = float(policy.get("life.on_time_probability")), int(policy.get("life.step_min"))
    back = int(policy.get("life.search_back_min"))
    windows = policy.get("traffic_windows.peak_windows_min")
    rng = np.random.default_rng(seed)
    od = f"{trip['from']}|{trip['to']}|{trip['mode']}"
    by = int(trip["arrive_by_min"])
    opts = []
    cand = sorted(set(range(by - back, by - 5, step)) | {int(trip["usual_depart_min"])})
    for dep in cand:
        band = "peak" if in_peak(dep, windows) else "offpeak"
        drive = sample(city.raw(f"od.{district}.{od}.{band}"), rng, runs)
        opts.append({"depart": hhmm(dep), "depart_min": dep, "band": band,
                     "p_on_time": float(np.mean(dep + drive <= by)), "drive_min": interval(drive, q)})
    ok = [o for o in opts if o["p_on_time"] >= need]
    best = max(ok, key=lambda o: o["depart_min"]) if ok else None
    usual = next((o for o in opts if o["depart_min"] == int(trip["usual_depart_min"])), None)
    return {"trip": trip["id"], "arrive_by": hhmm(by), "best": best, "usual": usual, "options": opts,
            "threshold": need}


def outdoor_windows(*, profile: Store, policy: Store, week: Store, days: list[str] | None = None) -> dict:
    """Outdoor windows: waking hours with no heat hard-limit violation (comfort) and PM10 ≤ the person's limit."""
    lims = _limits(profile, policy)
    awake = policy.get("life.awake_hours")
    min_len = int(policy.get("life.min_window_h"))
    out = {pid: [] for pid in lims}
    fam = []
    for d in week.raw("forecast.days"):
        if days and d["date"] not in days:
            continue
        ok_by = {}
        for pid, lim in lims.items():
            ok = []
            for h in range(awake[0], awake[1]):
                bad = bool(harsh_mask(policy, np.array([d["temperature_c"][h]]), d["humidity_pct"][h],
                                      d["pm10_ugm3"][h], pm_limit=lim)[0])
                ok.append(not bad)
            ok_by[pid] = ok
            out[pid].append({"date": d["date"], "windows": _merge(ok, awake[0], min_len)})
        together = [all(ok_by[p][i] for p in ok_by) for i in range(awake[1] - awake[0])]
        fam.append({"date": d["date"], "windows": _merge(together, awake[0], min_len)})
    return {"people": out, "family": fam, "limits_pm10": lims}


def _merge(ok: list[bool], h0: int, min_len: int) -> list[dict]:
    res, start = [], None
    for i, v in enumerate(ok + [False]):
        if v and start is None:
            start = i
        if not v and start is not None:
            if i - start >= min_len:
                res.append({"from": f"{h0 + start:02d}:00", "to": f"{h0 + i:02d}:00"})
            start = None
    return res


def morning_plan(*, profile: Store, city: Store, policy: Store, week: Store, day: str, district: str,
                 seed: int) -> dict:
    try:
        trips = [t for t in profile.raw("trips") if "arrive_by_min" in t]
        deps = [best_departure(trip=t, district=district, city=city, policy=policy, seed=seed) for t in trips]
        f = _forecast_day(week, day)
        general = float(policy.get("outdoor_limits.pm10_ugm3_max"))
        rise = next((h for h in range(6, 21) if f["pm10_ugm3"][h] > general), None)
        win = outdoor_windows(profile=profile, policy=policy, week=week, days=[day])
        today = {pid: v[0]["windows"] for pid, v in win["people"].items()}
        return answer({"date": day, "departures": deps, "dust_rises_at": f"{rise:02d}:00" if rise is not None else None,
                       "pm10_by_hour": f["pm10_ugm3"], "outdoor_today": today, "limits_pm10": win["limits_pm10"]},
                      unit="plan", stores=[profile, city, policy, week],
                      formulas=["travel time peak/off-peak by minute", "P(on time) ≥ policy threshold",
                                "CITY comfort hard limits", "PM10 per-person limit"],
                      label="Morning plan — computed")
    except MissingData as e:
        return refused(e, what="morning")


def week_windows(*, profile: Store, policy: Store, week: Store) -> dict:
    try:
        w = outdoor_windows(profile=profile, policy=policy, week=week)
        return answer(w, unit="windows", stores=[profile, policy, week],
                      formulas=["CITY comfort hard limits", "PM10 per-person limit"], label="Outdoor windows for the week")
    except MissingData as e:
        return refused(e, what="outdoor")


# ---------------------------------------------------------------- home on return
def return_hours(week: Store) -> dict:
    by = {}
    for r in week.raw("returns_log"):
        by.setdefault(r["weekday"], []).append(float(r["hour"]))
    return {wd: float(np.mean(v)) for wd, v in by.items()}


def home_schedule(*, listing: dict, profile: Store, climate: Store, policy: Store, month: str, return_h: float,
                  seed: int) -> dict:
    try:
        h = int(return_h)                     # cool for the start of the arrival hour — the home is ready early
        p = preload(listing=listing, profile=profile, climate=climate, policy=policy, month=month, arrival_h=h,
                    seed=seed)
        if p["calculation_status"] != "computed":
            return p
        v = p["value"]
        start = v["latest_start_hour_off_peak"]
        return answer({"return_at": hhmm(return_h * 60), "start_at": f"{int(start['mid']):02d}:00" if start else None,
                       "p_off_peak": v["p_feasible_off_peak"], "p_reachable": v["p_reachable"],
                       "target_c": profile.get("params.setpoint_c") + policy.get("comfort.arrival_band_c"),
                       "without_cooling_c": v["indoor_c_at_arrival_without_cooling"],
                       "with_plan_c": v["indoor_c_at_arrival_with_plan"], "preload_trace": p["trace_id"]},
                      unit="schedule", stores=[profile, climate, policy], formulas=["RP-1R1C euler_step preload"],
                      label="Home on return — computed")
    except MissingData as e:
        return refused(e, what="home")


# ---------------------------------------------------------------- errands in one trip
def errands(*, week: Store, queues: Store, city: Store, policy: Store) -> dict:
    try:
        tasks = week.raw("errands")
        center = next(t for t in tasks if t.get("center"))
        hours = queues.raw(f"centers.{center['center']}.wait_days_by_hour")
        best_h = min(hours, key=lambda k: _mean(hours[k]))
        leg = {t["id"]: float(city.get(f"errands.{t['id']}.from_home_min")) for t in tasks}
        hop = float(city.get("errands.hop_min"))
        separate = sum(2 * m for m in leg.values())
        ordered = sorted(tasks, key=lambda t: (0 if t.get("center") else 1, leg[t["id"]]))
        chain = leg[ordered[0]["id"]] + hop * (len(ordered) - 1) + leg[ordered[-1]["id"]]
        return answer({"date": center["date"], "start_at": f"{int(best_h):02d}:00", "order": [t["title"] for t in ordered], "order_ids": [t["id"] for t in ordered],
                       "separate_min": separate, "chain_min": chain, "saved_min": separate - chain,
                       "trips": {"separate": len(tasks), "chain": 1}},
                      unit="minutes", stores=[week, queues, city],
                      formulas=["center queue by hour (best hour)", "chain vs separate round trips"],
                      label="Errands in one trip — computed")
    except MissingData as e:
        return refused(e, what="errands")


# ---------------------------------------------------------------- car conflict
def car_conflict(*, week: Store, city: Store, policy: Store, district: str, lam: dict, seed: int) -> dict:
    try:
        runs, q = int(policy.get("mc.runs")), tuple(policy.get("mc.interval_q"))
        windows = policy.get("traffic_windows.peak_windows_min")
        ev = [e for e in week.raw("calendar") if e.get("needs_car")]
        ev.sort(key=lambda e: (e["date"], e["start_min"]))
        pair = next(((a, b) for a, b in zip(ev, ev[1:]) if a["date"] == b["date"] and b["start_min"] - a["start_min"] < 90),
                    None)
        if pair is None:
            return answer({"conflict": None}, unit="conflict", stores=[week], formulas=[], label="No conflicts")
        fixed = next(e for e in pair if not e.get("movable"))
        movable = next(e for e in pair if e.get("movable"))
        rng = np.random.default_rng(seed)
        band = lambda m: "peak" if in_peak(m, windows) else "offpeak"
        drv = lambda e, m: sample(city.raw(f"od.{district}.{e['od']}.{band(m)}"), rng, runs)
        # option 1: a taxi for the person whose meeting cannot move
        d_fixed = drv(fixed, fixed["start_min"] - 30)
        wait = sample(city.raw("taxi.wait_min"), rng, runs)
        taxi_cost = float(city.get("taxi.base_aed")) + float(city.get("taxi.aed_per_min")) * d_fixed
        # option 2: move the movable meeting to the slot with the shortest drive
        slots = policy.get("life.reschedule_slots_min")
        now_drive = drv(movable, movable["start_min"] - 30)
        by_slot = {s: drv(movable, s - 30) for s in slots}
        best_slot = min(by_slot, key=lambda s: float(np.median(by_slot[s])))
        opt = {"taxi": {"money": float(np.median(taxi_cost)), "time": float(np.median(wait)), "comfort": float(np.median(wait))},
               "move": {"money": 0.0, "time": 0.0, "comfort": 0.0}}
        names = list(policy.get("personal_params.lambda_names"))
        # normalise across options, weight by the person's weights
        score = {}
        for k in opt:
            score[k] = sum(lam[n] * (opt[k][n] / max(max(o[n] for o in opt.values()), 1e-9)) for n in names)
        rec = min(score, key=score.get)
        return answer({"conflict": [fixed["title"], movable["title"]], "date": fixed["date"],
                       "at": hhmm(fixed["start_min"]),
                       "taxi": {"who": fixed["who"], "cost_aed": interval(taxi_cost, q), "wait_outside_min": interval(wait, q)},
                       "fixed_id": fixed["id"], "movable_id": movable["id"],
                       "move": {"who": movable["who"], "from": hhmm(movable["start_min"]), "to": hhmm(best_slot),
                                "drive_now_min": interval(now_drive, q), "drive_new_min": interval(by_slot[best_slot], q)},
                       "recommended": rec, "scores": score,
                       "draft": "investor_reschedule" if rec == "move" else None},
                      unit="options", stores=[week, city, policy],
                      formulas=["travel time peak/off-peak", "taxi fare", "λ-weighted choice"],
                      label="Both need the car — computed")
    except MissingData as e:
        return refused(e, what="car_conflict")


# ---------------------------------------------------------------- second car or taxi (year)
def car_vs_taxi(*, profile: Store, city: Store, climate: Store, policy: Store, district: str, lam: dict, seed: int,
                money_first: bool = False) -> dict:
    try:
        runs, q = int(policy.get("mc.runs")), tuple(policy.get("mc.interval_q"))
        wd = float(policy.get("future.workdays_per_month"))
        lims = {pid: v for pid, v in _limits(profile, policy).items()
                if v != float(policy.get("outdoor_limits.pm10_ugm3_max"))}
        car_trips = [t for t in profile.raw("trips") if t["mode"] == "car"]
        rng = np.random.default_rng(seed + 3)
        own = float(city.get("car.ownership_aed_month"))
        fuel = float(city.get("car.fuel_aed_per_min"))
        base, per_min = float(city.get("taxi.base_aed")), float(city.get("taxi.aed_per_min"))
        tot = {k: {"money": np.zeros(runs), "wait_h": np.zeros(runs), "adam": np.zeros(runs)} for k in ("car", "taxi", "taxi_rules")}
        for m in calendar_months(climate):
            kw = dict(profile=profile, city=city, climate=climate, policy=policy, district=district, month=m, seed=seed,
                      person_limits=lims)
            car = day_samples(**kw)
            waits = {t["id"]: sample(city.raw("taxi.wait_min"), rng, runs) for t in car_trips}
            taxi = day_samples(**kw, overrides={t["id"]: {"extra_outdoor_min": waits[t["id"]]} for t in car_trips})
            indoor = {t["id"]: (sample(city.raw("taxi.wait_indoor_booked_min"), rng, runs) if "son" in t["who"]
                                else waits[t["id"]]) for t in car_trips}
            rules = day_samples(**kw, overrides={t["id"]: {"extra_outdoor_min": indoor[t["id"]]} for t in car_trips})
            drives = {tl["trip"]: tl["drive"] for tl in car["timeline"] if "drive" in tl and tl["trip"] in waits}
            tot["car"]["money"] += own + fuel * wd * sum(drives.values())
            fare = wd * sum(base + per_min * d for d in drives.values())
            for k, ds, w in (("taxi", taxi, waits), ("taxi_rules", rules, waits)):
                tot[k]["money"] += fare
                tot[k]["wait_h"] += wd * sum(w.values()) / 60.0
                tot[k]["adam"] += wd * ds["harsh"].get("son", 0.0)
            tot["car"]["adam"] += wd * car["harsh"].get("son", 0.0)
        axes = {k: [float(np.median(v["money"])), float(np.median(v["wait_h"])), float(np.median(v["adam"]))]
                for k, v in tot.items()}
        names = list(policy.get("personal_params.lambda_names"))
        w = dict(lam)
        if money_first:
            w = {n: (0.7 if n == "money" else 0.15) for n in names}
        X = np.asarray([axes[k] for k in axes])
        N = (X - X.min(0)) / np.where(X.max(0) > X.min(0), X.max(0) - X.min(0), 1.0)
        score = dict(zip(axes, (N @ np.asarray([w[n] for n in names])).tolist()))
        pool = ["car", "taxi_rules"] if money_first else ["car", "taxi"]
        rec = min(pool, key=score.get)
        return answer({"year": {k: {"money_aed": interval(v["money"], q), "waiting_h": interval(v["wait_h"], q),
                                    "adam_harsh_min": interval(v["adam"], q)} for k, v in tot.items()},
                       "diff_car_minus_taxi": {"money_aed": axes["car"][0] - axes["taxi"][0],
                                               "waiting_h": axes["car"][1] - axes["taxi"][1],
                                               "adam_harsh_min": axes["car"][2] - axes["taxi"][2]},
                       "recommended": rec, "money_first": money_first,
                       "rules": ["in dusty hours the child waits for the car indoors", "book ahead"] if money_first else []},
                      unit="year", stores=[profile, city, climate, policy],
                      formulas=["CITY walk_time + comfort + PM10 (per month)", "car ownership + fuel", "taxi fare + wait",
                                "λ-weighted choice"],
                      label="Second car or taxi — one year")
    except MissingData as e:
        return refused(e, what="car_vs_taxi")


# ---------------------------------------------------------------- guest
def guest_arrival(*, week: Store, profile: Store, policy: Store, climate: Store, scenario: Store, listing: dict,
                  seed: int, heat_limit_min: float = 0.0) -> dict:
    try:
        q, runs = tuple(policy.get("mc.interval_q")), int(policy.get("mc.runs"))
        g = week.raw("guest")
        m = g["month"]
        t_out, t_sd = climate.get(f"months.{m}.temperature_c"), climate.get(f"months.{m}.temperature_sd_c")
        hum = climate.get(f"months.{m}.humidity_pct")
        rng = np.random.default_rng(seed)
        rows = []
        for h in g["candidate_hours"]:
            temp = t_out[h] + rng.normal(0, t_sd, runs)
            heat = harsh_mask(policy, temp, hum[h], 0.0, pm_limit=np.inf)
            walk = walk_minutes(policy, g["slowest_member"], scenario.get("arrival.without.walk_to_taxi_m")) + \
                walk_minutes(policy, g["slowest_member"], scenario.get("arrival.without.walk_home_m"))
            outdoor = walk + sample(scenario.raw("arrival.without.taxi_wait_min"), rng, runs)
            pre = preload(listing=listing, profile=profile, climate=climate, policy=policy, month=m, arrival_h=h, seed=seed)
            rows.append({"hour": f"{h:02d}:00", "heat_min_without_cure": interval(np.where(heat, outdoor, 0.0), q),
                         "heat_min_with_cure": 0.0,
                         "room_ready_p": pre["value"]["p_reachable"] if pre["calculation_status"] == "computed" else None})
        best = min(rows, key=lambda r: (r["heat_min_without_cure"]["mid"], -(r["room_ready_p"] or 0), r["hour"]))
        return answer({"guest": g["title"], "guest_ru": g.get("title_ru", g["title"]), "month": m, "rows": rows, "best_hour": best["hour"],
                       "limit": {"outdoor_heat_min_max": heat_limit_min, "stored": "private contour"}},
                      unit="minutes", stores=[week, policy, climate, scenario],
                      formulas=["CITY walk_time (mobility aid)", "CITY comfort hard limits", "RP-1R1C preload"],
                      label="Guest arrival — computed")
    except MissingData as e:
        return refused(e, what="guest")


# ---------------------------------------------------------------- budget plan / fact
def budget_month(*, month_key: str, profile: Store, city: Store, climate: Store, tariffs: Store, policy: Store,
                 listings: dict, district: str, week: Store, seed: int) -> dict:
    try:
        ds = district_samples(district=district, profile=profile, city=city, climate=climate, tariffs=tariffs,
                              policy=policy, listings=listings, seed=seed)
        plan = {"rent": float(np.median(ds["rent"])), "cooling": float(np.median(ds["cooling"])),
                "commute": float(np.median(ds["commute_money"])), "school": float(np.median(ds["school"]))}
        fact = {}
        for t in week.raw("transactions"):
            if t["month"] == month_key:
                fact[t["category"]] = fact.get(t["category"], 0.0) + float(t["aed"])
        diff = {k: fact.get(k, 0.0) - v for k, v in plan.items()}
        listing = listings[city.get(f"typical_listing.{district}")]
        bill = bill_samples(listing=listing, profile=profile, climate=climate, tariffs=tariffs, policy=policy,
                            seed=seed)["months"]
        avg = float(np.median(np.mean([bill[m] for m in calendar_months(climate)], axis=0)))
        extra = {m: float(np.median(bill[m])) - avg for m in week.raw("summer_months")}
        return answer({"month": month_key, "plan": plan, "fact": fact, "diff": diff, "total_diff": sum(diff.values()),
                       "main_saving": min(diff, key=diff.get), "summer_extra_per_month": extra,
                       "summer_reserve": sum(max(v, 0.0) for v in extra.values())},
                      unit="AED", stores=[profile, city, climate, tariffs, policy, week],
                      formulas=["district monthly cost (1R1C + commute + rent + school)", "plan vs fact"],
                      label="Month budget — plan vs fact")
    except MissingData as e:
        return refused(e, what="budget")


# ---------------------------------------------------------------- habits
def habits(*, week: Store, policy: Store) -> dict:
    try:
        log = week.raw("returns_log")
        base = float(np.mean([r["hour"] for r in log]))
        found = []
        for wd, h in return_hours(week).items():
            xs = [r["hour"] for r in log if r["weekday"] == wd]
            if len(xs) >= int(policy.get("habits.min_n")) and abs(h - base) >= float(policy.get("habits.min_shift_h")) \
                    and float(np.std(xs)) <= float(policy.get("habits.max_sd_h")):
                found.append({"kind": "return_shift", "weekday": wd, "at": f"{int(h):02d}:{int(round((h % 1) * 60)):02d}",
                              "usual": f"{int(base):02d}:{int(round((base % 1) * 60)):02d}"})
        taxi_peak = [c for c in week.raw("choices_log") if c["chosen"] == "taxi" and c["peak"]]
        if len(taxi_peak) >= int(policy.get("habits.min_choices")):
            found.append({"kind": "time_weight_up", "observations": len(taxi_peak)})
        return answer({"found": found, "applied": False, "note": "the model changes only after a yes"},
                      unit="habits", stores=[week, policy], formulas=["weekday mean shift ≥ threshold, low spread"],
                      label="Noticed habits")
    except MissingData as e:
        return refused(e, what="habits")
