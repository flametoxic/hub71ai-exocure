"""One family day by the hour: road time, heat and dust outside.

CURE formulas: city_world.formulas.walk_time (walking leg), city_world.formulas.comfort (hard limits).
"""
from __future__ import annotations

import numpy as np

from api_gateway.core.city_world.formulas import comfort, walk_time

from .core import MissingData, Store, answer, interval, refused, sample


def outdoor_g(policy: Store):
    """Site comfort function g for city_world.comfort: weights come from the policy."""
    w_t, w_h = policy.get("comfort.weight_temperature"), policy.get("comfort.weight_humidity")
    return lambda dt, co2, humidity, noise: w_t * dt + w_h * humidity


def walk_minutes(policy: Store, person: dict, meters: float) -> float:
    wt = {k: policy.get(f"walk.{k}") for k in ("alpha", "beta", "dt_crowd_s", "dt_crossing_s")}
    if meters <= 0:
        return 0.0
    return walk_time(t_min=meters / policy.get("walk.speed_mps"), dt_crossing=wt["dt_crossing_s"],
                     p_mobility_aid=float(person.get("mobility_aid", 0)), p_heavy_items=float(person.get("heavy_items", 0)),
                     alpha=wt["alpha"], beta=wt["beta"], dt_crowd=wt["dt_crowd_s"])["t_walk_s"] / 60.0


def harsh_mask(policy: Store, temps: np.ndarray, humidity: float, pm10: float, pm_limit: float | None = None) -> np.ndarray:
    """CURE comfort hard limits for temperature/humidity OR dust above the limit."""
    limits = {"temperature": tuple(policy.get("outdoor_limits.temperature_c")),
              "humidity": tuple(policy.get("outdoor_limits.humidity_pct"))}
    g, t_pref = outdoor_g(policy), policy.get("comfort.t_pref_c")
    lim = policy.get("outdoor_limits.pm10_ugm3_max") if pm_limit is None else pm_limit
    bad_t = np.array([bool(comfort(g, t=float(t), t_pref=t_pref, co2=0.0, humidity=humidity, noise=0.0,
                                   minimums=limits)["hard_limit_violations"]) for t in temps])
    return bad_t | (pm10 > lim)


def day_samples(*, profile: Store, city: Store, climate: Store, policy: Store, district: str, month: str, seed: int,
                overrides: dict | None = None, person_limits: dict | None = None) -> dict:
    """Monte Carlo samples of one day.
    overrides: {trip_id: {"walk_m": x} | {"skip": True} | {"extra_outdoor_min": samples or a number}} — for "what if"
    (extra_outdoor_min — waiting outside, e.g. for a taxi). person_limits: {person_id: PM10 limit} — personal
    constraints (derived, no diagnosis); without them the general policy limit applies."""
    runs = int(policy.get("mc.runs"))
    peak = set(policy.get("traffic.peak_hours"))
    hours_t, hours_h = climate.get(f"months.{month}.temperature_c"), climate.get(f"months.{month}.humidity_pct")
    hours_pm, t_sd = climate.get(f"months.{month}.pm10_ugm3"), climate.get(f"months.{month}.temperature_sd_c")
    people = {p["id"]: p for p in profile.raw("household")}
    rng = np.random.default_rng(seed)  # the same random numbers for all districts and options
    transit, harsh, timeline = np.zeros(runs), {pid: np.zeros(runs) for pid in people}, []
    for trip in profile.raw("trips"):
        ov = (overrides or {}).get(trip["id"], {})
        h = int(trip["depart_h"])
        drive = sample(city.raw(f"od.{district}.{trip['from']}|{trip['to']}|{trip['mode']}."
                                f"{'peak' if h in peak else 'offpeak'}"), rng, runs)
        temp = hours_t[h] + rng.normal(0.0, t_sd, runs)
        if ov.get("skip"):
            timeline.append({"trip": trip["id"], "skipped": True}); continue
        meters = ov.get("walk_m", city.get(f"walk_m.{district}.{trip['id']}"))
        bad = harsh_mask(policy, temp, hours_h[h], hours_pm[h])
        extra = np.asarray(ov.get("extra_outdoor_min", 0.0), float) * np.ones(runs)
        walk = 0.0
        for who in trip["who"]:
            w = walk_minutes(policy, people[who], meters)
            walk = max(walk, w)
            lim = (person_limits or {}).get(who)
            bad_who = bad if lim is None else harsh_mask(policy, temp, hours_h[h], hours_pm[h], pm_limit=lim)
            harsh[who] += np.where(bad_who, w + extra, 0.0)
        transit += drive + walk + extra
        timeline.append({"trip": trip["id"], "depart_h": h, "who": trip["who"], "drive": drive, "walk_min": walk,
                         "temp": temp, "pm10_ugm3": hours_pm[h], "walk_m": meters})
    return {"transit": transit, "harsh": harsh, "timeline": timeline}


def simulate_day(*, profile: Store, city: Store, climate: Store, policy: Store, districts: list[str], month: str,
                 seed: int, overrides: dict | None = None) -> dict:
    try:
        q = tuple(policy.get("mc.interval_q"))
        result = {}
        for d in districts:
            s = day_samples(profile=profile, city=city, climate=climate, policy=policy, district=d, month=month,
                            seed=seed, overrides=overrides)
            result[d] = {"timeline": [{**{k: v for k, v in t.items() if k not in ("drive", "temp")},
                                       **({"drive_min": interval(t["drive"], q), "outdoor_temp_c": interval(t["temp"], q)}
                                          if "drive" in t else {})} for t in s["timeline"]],
                         "transit_min": interval(s["transit"], q),
                         "harsh_outdoor_min": {pid: interval(v, q) for pid, v in s["harsh"].items()}}
        return answer(result, unit="minutes", stores=[profile, city, climate, policy],
                      formulas=["CITY walk_time", "CITY comfort hard limits"], label="Family day — computed")
    except MissingData as e:
        return refused(e, what="day")
