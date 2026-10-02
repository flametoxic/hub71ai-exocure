"""Full cost map: district = rent + cooling + commute + school, plus time and the focus person's minutes in heat and dust.

Nothing is silently merged into one price: three axes (money, time, comfort) → Pareto; the recommendation uses
the personal weights λ. Choosing a district updates λ (person.PersonalModel.observe_choice).
Engines: day.day_samples (walk_time, comfort), apartment.bill_samples (1R1C).
"""
from __future__ import annotations

import numpy as np

from .apartment import bill_samples, calendar_months
from .core import MissingData, Store, answer, interval, refused
from .day import day_samples


def district_samples(*, district: str, profile: Store, city: Store, climate: Store, tariffs: Store, policy: Store,
                     listings: dict, seed: int, focus: str | None = None) -> dict:
    """Average month (over all calendar months in the data): money AED/month, road h/month, focus person heat minutes/month."""
    wd = float(policy.get("future.workdays_per_month"))
    per_min = float(city.get("trip_cost_aed_per_min"))
    months = calendar_months(climate)
    days = [day_samples(profile=profile, city=city, climate=climate, policy=policy, district=district, month=m,
                        seed=seed) for m in months]
    transit = np.mean([d["transit"] for d in days], axis=0)
    focus = focus or focus_person(profile)
    harsh = np.mean([d["harsh"][focus] for d in days], axis=0)
    listing = listings[city.get(f"typical_listing.{district}")]
    cooling = bill_samples(listing=listing, profile=profile, climate=climate, tariffs=tariffs, policy=policy,
                           seed=seed)["total"] / len(months)
    rent = float(city.get(f"rent_aed_month.{district}"))
    school = float(city.get(f"school_fees_aed_month.{district}")) if profile.raw("flags").get("has_school_children") else 0.0
    commute_money = transit * wd * per_min
    return {"money": rent + cooling + commute_money + school, "rent": np.full_like(cooling, rent), "cooling": cooling,
            "commute_money": commute_money, "school": np.full_like(cooling, school),
            "time_h": transit * wd / 60.0, "son_harsh_min": harsh * wd}


def focus_person(profile: Store) -> str:
    """Whose minutes in heat and dust form the third axis: the person with constraints, else the first child, else the first person."""
    hh = profile.raw("household")
    cons = profile.doc.get("constraints") or {}
    for p in hh:
        if p["id"] in cons:
            return p["id"]
    return next((p["id"] for p in hh if p.get("role") == "child"), hh[0]["id"])


def pareto(points: dict[str, list[float]]) -> list[str]:
    keys = list(points)
    J = np.asarray([points[k] for k in keys])
    return [keys[i] for i in range(len(keys))
            if not any(np.all(J[j] <= J[i]) and np.any(J[j] < J[i]) for j in range(len(keys)) if j != i)]


def why_and_switch(axes: dict, lam: dict, names: list) -> dict:
    """Why this district (which axis carried it) and at what money weight the advice changes."""
    ds = list(axes)
    X = np.asarray([axes[d] for d in ds], float)
    N = (X - X.min(0)) / np.where(X.max(0) > X.min(0), X.max(0) - X.min(0), 1.0)
    vec = lambda w: np.asarray([w[n] for n in names])
    sc = N @ vec(lam)
    order = np.argsort(sc)
    rec, runner = ds[order[0]], ds[order[1]] if len(ds) > 1 else None
    adv = {n: float((N[ds.index(runner)][i] - N[ds.index(rec)][i]) * lam[n]) for i, n in enumerate(names)} if runner else {}
    why = max(adv, key=adv.get) if adv else None
    switch = None
    for t in np.linspace(0, 1, 201):
        lt = {n: (1 - t) * lam[n] + t * (1.0 if n == "money" else 0.0) for n in names}
        r = ds[int(np.argmin(N @ vec(lt)))]
        if r != rec:
            switch = {"money_weight": round(float(lt["money"]), 2), "to": r}
            break
    return {"recommended": rec, "runner_up": runner, "why": why, "advantage_by_axis": adv, "switch": switch,
            "margin": float(sc[order[1]] - sc[order[0]]) if runner else None}


def cost_map(*, profile: Store, city: Store, climate: Store, tariffs: Store, policy: Store, listings: dict, seed: int,
             lam: dict) -> dict:
    try:
        q = tuple(policy.get("mc.interval_q"))
        names = list(policy.get("personal_params.lambda_names"))       # money, time, comfort
        focus = focus_person(profile)
        districts = list(city.raw("rent_aed_month"))
        rows, axes = {}, {}
        for d in districts:
            s = district_samples(district=d, profile=profile, city=city, climate=climate, tariffs=tariffs,
                                 policy=policy, listings=listings, seed=seed, focus=focus)
            rows[d] = {k: interval(v, q) for k, v in s.items()}
            axes[d] = [float(np.median(s["money"])), float(np.median(s["time_h"])), float(np.median(s["son_harsh_min"]))]
        budget = float(profile.get("params.budget_aed_month"))
        housing_cost = {d: rows[d]["rent"]["mid"] + rows[d]["cooling"]["mid"] for d in districts}
        over_budget = [d for d in districts if housing_cost[d] > budget]
        feasible = [d for d in districts if d not in over_budget]
        common = {"districts": rows,
                  "axes": {"names": names, "units": ["AED/month", "h/month", "min/month"], "values": axes},
                  "housing_budget_aed_month": budget, "housing_cost_aed_month": housing_cost,
                  "feasible_districts": feasible, "over_housing_budget": over_budget,
                  "lambda_used": lam, "focus_person": focus}
        if not feasible:
            out = answer({**common, "pareto": [], "score_by_personal_lambda": {}, "recommended": None,
                          "why": None, "runner_up": None, "switch": None, "margin": None},
                         unit="per_month", stores=[profile, city, climate, tariffs, policy],
                         formulas=["RP-1R1C bill", "CITY walk_time", "CITY comfort hard limits", "housing budget"],
                         label="Full cost map — infeasible")
            out["calculation_status"] = "infeasible"
            out["missing"] = ["housing_budget:no_feasible_district"]
            return out
        front = pareto({d: axes[d] for d in feasible})
        # Recommendation normalizes only budget-feasible candidates, then minimizes within their Pareto front.
        X = np.asarray([axes[d] for d in feasible])
        lo, hi = X.min(0), X.max(0)
        N = (X - lo) / np.where(hi > lo, hi - lo, 1.0)
        w = np.asarray([lam[n] for n in names])
        score = dict(zip(feasible, (N @ w).tolist()))
        rec = min(front, key=lambda d: score[d])
        ws = why_and_switch({d: axes[d] for d in front}, lam, names) if len(front) > 1 else {"recommended": rec}
        return answer({**common, "pareto": front, "score_by_personal_lambda": score, "recommended": rec,
                       "why": ws.get("why"), "runner_up": ws.get("runner_up"), "switch": ws.get("switch"),
                       "margin": ws.get("margin")},
                      unit="per_month", stores=[profile, city, climate, tariffs, policy],
                      formulas=["RP-1R1C bill", "CITY walk_time", "CITY comfort hard limits", "Pareto", "λ-weighted choice"],
                      label="Full cost map — forecast")
    except MissingData as e:
        return refused(e, what="cost_map")
