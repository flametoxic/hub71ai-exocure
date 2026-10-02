"""Dusty morning: the day is rebuilt around the dust limit; "what if +1° on the AC".

Engines: day.day_samples (walk_time, comfort — on the aug_dust month), apartment.bill_samples (1R1C, the same sample
of flat parameters → paired difference), city_world.comfort indoors (indoor_limits).
The diagnosis is never used and never leaves the contour: only the constraint "no walking legs in dust" is known.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from api_gateway.core.city_world.formulas import comfort

from . import agents as ag
from .apartment import bill_samples
from .core import MissingData, Store, answer, interval, refused
from .day import day_samples, outdoor_g

DUST_MONTH = "aug_dust"


def adapted_overrides(profile: Store, who: str = "son") -> dict:
    """Plan: trips with the son are door to door; walks with him are moved indoors."""
    out = {}
    for t in profile.raw("trips"):
        if who in t["who"]:
            out[t["id"]] = {"skip": True} if t["mode"] == "walk" else {"walk_m": 0.0}
    return out


def dusty_morning(*, profile: Store, city: Store, climate: Store, policy: Store, rates: Store, district: str,
                  seed: int, agents: list[dict], now: datetime) -> dict:
    try:
        q = tuple(policy.get("mc.interval_q"))
        kw = dict(profile=profile, city=city, climate=climate, policy=policy, district=district, month=DUST_MONTH,
                  seed=seed)
        base = day_samples(**kw)
        ov = adapted_overrides(profile)
        adapted = day_samples(**kw, overrides=ov)
        saved = base["harsh"]["son"] - adapted["harsh"]["son"]
        cards = ag.on_event(agents, event="dust_forecast", rates=rates, now=now,
                            payload={"son_harsh_min_saved": interval(saved, q)})
        return answer({"district": district, "month_data": DUST_MONTH, "changes": ov,
                       "son_harsh_min": {"usual": interval(base["harsh"]["son"], q),
                                         "adapted": interval(adapted["harsh"]["son"], q), "saved": interval(saved, q)},
                       "family_transit_min": {"usual": interval(base["transit"], q),
                                              "adapted": interval(adapted["transit"], q)},
                       "agent_cards": cards,
                       "assumptions": ["the car pulls up to the school and home doors (walk_m = 0)",
                                       "the walk moves indoors — its duration is not modelled"]},
                      unit="minutes", stores=[profile, city, climate, policy],
                      formulas=["CITY walk_time", "CITY comfort hard limits", "PM10 limit"],
                      label="Dusty morning — forecast")
    except MissingData as e:
        return refused(e, what="dusty_morning")


def setpoint_whatif(*, listing: dict, profile: Store, climate: Store, tariffs: Store, policy: Store, seed: int,
                    delta_c: float, personal_setpoint: float, month: str = "aug") -> dict:
    try:
        q = tuple(policy.get("mc.interval_q"))
        sp = float(profile.get("params.setpoint_c"))
        kw = dict(listing=listing, profile=profile, climate=climate, tariffs=tariffs, policy=policy, seed=seed)
        a, b = bill_samples(**kw, setpoint_c=sp), bill_samples(**kw, setpoint_c=sp + delta_c)   # the same flat parameters
        g = outdoor_g(policy)
        lim = {"temperature": tuple(policy.get("indoor_limits.temperature")),
               "humidity": tuple(policy.get("indoor_limits.humidity"))}
        air = {k: float(policy.get(f"indoor_air.{k}")) for k in ("co2_ppm", "humidity_pct", "noise_db")}
        c = lambda t: comfort(g, t=t, t_pref=personal_setpoint, co2=air["co2_ppm"], humidity=air["humidity_pct"],
                              noise=air["noise_db"], minimums=lim)
        now_c, new_c = c(sp), c(sp + delta_c)
        return answer({"setpoint_c": {"now": sp, "new": sp + delta_c},
                       "bill_delta_aed": {"month": month, "value": interval(b["months"][month] - a["months"][month], q),
                                          "year": interval(b["total"] - a["total"], q)},
                       "discomfort_index": {"now": now_c["comfort"], "new": new_c["comfort"],
                                            "t_pref_from_personal_model": personal_setpoint},
                       "hard_limit_violations": {"now": now_c["hard_limit_violations"],
                                                 "new": new_c["hard_limit_violations"]},
                       "not_modeled": ["indoor air quality (PM2.5) — needs a sensor in the flat"]},
                      unit="AED", stores=[profile, climate, tariffs, policy],
                      formulas=["RP-1R1C bill (paired)", "CITY comfort indoor limits"],
                      label="Setpoint +1° — what if")
    except MissingData as e:
        return refused(e, what="setpoint_whatif")
