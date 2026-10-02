"""The next 12 months card. Every number comes from the same samples as the other scenes.

Engines: plan.plan_samples (step graph + teacher facts), apartment.bill_samples (1R1C by month),
day.day_samples (road by month). Every row references the trace_id of its source answer.
"""
from __future__ import annotations

import calendar
from datetime import date

import numpy as np

from .apartment import bill_samples, calendar_months
from .core import MissingData, Store, answer, interval, refused, sample
from .day import day_samples
from .plan import plan_samples


def _month_windows(start: date, n: int) -> list[tuple[str, int, int, int]]:
    """(climate month key, start day from plan start, end day, days in month)."""
    out, y, m = [], start.year, start.month
    for _ in range(n):
        d0 = (date(y, m, 1) - start).days
        days = calendar.monthrange(y, m)[1]
        out.append((f"{y}-{m:02d}", max(d0, 0), d0 + days, days))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def future_card(*, profile: Store, steps: Store, queues: Store, policy: Store, city: Store, climate: Store,
                tariffs: Store, rates: Store, listing: dict, district: str, start: date, seed: int,
                learned: dict | None = None) -> dict:
    try:
        q = tuple(policy.get("mc.interval_q"))
        H = int(policy.get("future.horizon_months"))
        wd = float(policy.get("future.workdays_per_month"))
        per_min = float(city.get("trip_cost_aed_per_min"))
        ps = plan_samples(profile=profile, steps=steps, queues=queues, policy=policy, seed=seed, learned=learned)
        runs = ps["finish"].size
        rng = np.random.default_rng(seed + 1)
        tf = ps["twin_f"]
        move_in = tf["utilities"]                                               # they live in the flat once utilities are on
        school_from = tf.get("school", np.full(runs, np.inf))
        license_done = tf.get("company_license")
        hire = license_done + sample(policy.raw("business.hire_lag_after_license_days"), rng, runs) \
            if license_done is not None else None
        cal = calendar_months(climate)
        bill = bill_samples(listing=listing, profile=profile, climate=climate, tariffs=tariffs, policy=policy,
                            seed=seed)["months"]
        transit = {m: day_samples(profile=profile, city=city, climate=climate, policy=policy, district=district,
                                  month=m, seed=seed)["transit"] for m in cal}
        rent = float(city.get(f"rent_aed_month.{district}"))
        school_fee = float(city.get(f"school_fees_aed_month.{district}"))
        temp_rate = float(rates.get("temporary_housing_per_day"))
        company = float(rates.get("company_monthly_cost_aed"))
        months, cum, commute_h = [], np.zeros(runs), np.zeros(runs)
        for key, d0, d1, days in _month_windows(start, H):
            m = cal[int(key[-2:]) - 1]
            frac_home = np.clip((d1 - np.maximum(move_in, d0)) / days, 0.0, 1.0)
            frac_temp = 1.0 - frac_home if d0 >= 0 else np.zeros(runs)
            frac_school = np.clip((d1 - np.maximum(school_from, d0)) / days, 0.0, 1.0)
            spend = {"temp_housing": frac_temp * days * temp_rate, "rent": frac_home * rent,
                     "cooling": frac_home * bill[m], "commute": frac_home * transit[m] * wd * per_min,
                     "school": frac_school * school_fee}
            if license_done is not None:
                spend["company"] = np.clip((d1 - np.maximum(license_done, d0)) / days, 0.0, 1.0) * company
            total = sum(spend.values())
            cum = cum + total
            commute_h = commute_h + frac_home * transit[m] * wd / 60.0
            months.append({"month": key, "p_plan_done": float(np.mean(ps["finish"] <= d1)),
                           "spend": interval(total, q), "cumulative": interval(cum, q),
                           "parts_mid": {k: float(np.median(v)) for k, v in spend.items()}})
        # main risk: the step whose duration moves the ready date most
        share = {k: float(np.corrcoef(v, ps["finish"])[0, 1] ** 2) for k, v in ps["dur"].items() if np.std(v) > 0}
        top = max(share, key=share.get)
        to_date = lambda d: (start.toordinal() + int(round(d)))
        iv_date = lambda x: {k: date.fromordinal(to_date(v)).isoformat() for k, v in interval(x, q).items()
                             if k in ("low", "mid", "high")}
        return answer({"horizon_months": H, "start": start.isoformat(), "district": district,
                       "months": months, "ready_date": iv_date(ps["finish"]),
                       "move_in_date": iv_date(move_in), "first_hire_date": iv_date(hire) if hire is not None else None,
                       "spend_12m": interval(cum, q), "commute_hours_12m": interval(commute_h, q),
                       "main_risk": {"step": top, "variance_share": share[top], "all": share},
                       "assumptions": ["until utilities are connected the family lives in temporary housing",
                                       "road time counts after move-in", "hire = licence + lag from the policy"]},
                      unit="AED", stores=[profile, steps, queues, policy, city, climate, tariffs, rates],
                      formulas=["CPR MC plan", "RP-1R1C bill by month", "CITY walk_time", "variance share"],
                      label="12 months ahead — forecast")
    except MissingData as e:
        return refused(e, what="future")
