"""Move-in plan: step graph, critical path, date with an interval, "what if" and the final number.

One Monte Carlo run computes two worlds on the same random numbers:
  twin     — steps in parallel by dependencies, visits at the hour with the shortest wait;
  baseline — steps in list order, visits at a random hour.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from .core import MissingData, Store, answer, interval, refused, sample


def _steps(steps: Store, profile: Store) -> list[dict]:
    flags = profile.raw("flags")
    chosen = [s for s in steps.raw("steps") if all(flags.get(k) == v for k, v in s.get("applies_if", {}).items())]
    ids = {s["id"] for s in chosen}
    for s in chosen:
        bad = [b for b in s.get("blocks_on", []) if b not in ids]
        if bad:
            raise MissingData([f"steps:{s['id']}.blocks_on:{b}" for b in bad])
    seen, order = set(), []

    def visit(s, stack=()):
        if s["id"] in stack:
            raise ValueError(f"cycle in steps catalog at {s['id']}")
        if s["id"] in seen:
            return
        for b in s.get("blocks_on", []):
            visit(next(x for x in chosen if x["id"] == b), stack + (s["id"],))
        seen.add(s["id"]); order.append(s)

    for s in chosen:
        visit(s)
    return order


def _mean(spec) -> float:
    d = spec.get("dist")
    return {"fixed": lambda: spec["value"], "uniform": lambda: (spec["low"] + spec["high"]) / 2,
            "triangular": lambda: (spec["low"] + spec["mode"] + spec["high"]) / 3}[d]()


def _duration(s: dict, rng, runs: int, learned: dict, k: float, overrides: dict | None = None) -> np.ndarray:
    """Step duration: the catalogue prior mixed with the teacher's facts.
    Share of facts = n / (n + k), k — prior_pseudo_count from the policy (prior weight in "number of facts")."""
    prior = sample((overrides or {}).get(s["id"], s["duration_days"]), rng, runs)   # confirmed fact card
    obs = np.asarray(learned.get(s["id"], []), float)
    if obs.size == 0:
        return prior
    use = rng.random(runs) < obs.size / (obs.size + k)
    return np.where(use, rng.choice(obs, size=runs), prior)


def _run(order, steps: Store, queues: Store, rng, runs, delays, learned=None, k=0.0, overrides=None, completed=None):
    dur = {s["id"]: _duration(s, rng, runs, learned or {}, k, overrides) + float(delays.get(s["id"], 0.0)) for s in order}
    twin_f, base_f, base_prev = {}, {}, np.zeros(runs)
    for s in order:
        if s['id'] in (completed or {}):
            dur[s['id']] = np.zeros(runs)
            twin_f[s['id']] = base_f[s['id']] = np.full(runs, float(completed[s['id']]))
            base_prev = base_f[s['id']]
            continue
        wait_t = wait_b = np.zeros(runs)
        if s.get("requires_visit"):
            hours = queues.raw(f"centers.{s['center']}.wait_days_by_hour")
            keys = list(hours)
            by_hour = np.stack([sample(hours[k], rng, runs) for k in keys])      # wait at each hour
            best = min(range(len(keys)), key=lambda i: _mean(hours[keys[i]]))
            wait_t = by_hour[best]                                                # twin: best hour
            wait_b = by_hour[rng.integers(len(keys), size=runs), np.arange(runs)] # baseline: random hour
        dep_t = np.max([twin_f[b] for b in s.get("blocks_on", [])], axis=0) if s.get("blocks_on") else np.zeros(runs)
        dep_b = np.max([base_f[b] for b in s.get("blocks_on", [])] + [base_prev], axis=0)
        twin_f[s["id"]] = dep_t + wait_t + dur[s["id"]]
        base_f[s["id"]] = dep_b + wait_b + dur[s["id"]]
        base_prev = base_f[s["id"]]
    return dur, twin_f, base_f


def _critical_path(order, dur):
    med = {k: float(np.median(v)) for k, v in dur.items()}
    ef = {}
    for s in order:
        ef[s["id"]] = max([ef[b] for b in s.get("blocks_on", [])], default=0.0) + med[s["id"]]
    node, path = max(ef, key=ef.get), []
    while node:
        path.append(node)
        deps = next(s for s in order if s["id"] == node).get("blocks_on", [])
        node = max(deps, key=lambda b: ef[b]) if deps else None
    return list(reversed(path)), ef


def plan_samples(*, profile: Store, steps: Store, queues: Store, policy: Store, seed: int, delays: dict | None = None,
                 learned: dict | None = None, overrides: dict | None = None, completed: dict | None = None) -> dict:
    """Raw plan samples for other scenes (future, vision). learned: {step_id: [facts in days]} from the teacher."""
    runs = int(policy.get("mc.runs"))
    k = float(policy.get("sync.prior_pseudo_count")) if learned else 0.0
    order = _steps(steps, profile)
    dur, twin_f, base_f = _run(order, steps, queues, np.random.default_rng(seed), runs, delays or {}, learned, k,
                               overrides, completed)
    return {"order": order, "dur": dur, "twin_f": twin_f, "base_f": base_f,
            "finish": np.max(list(twin_f.values()), axis=0), "finish_base": np.max(list(base_f.values()), axis=0)}


def build_plan(*, profile: Store, steps: Store, queues: Store, policy: Store, start: date, target: date, seed: int,
               delays: dict | None = None, learned: dict | None = None, overrides: dict | None = None, completed: dict | None = None) -> dict:
    try:
        q = tuple(policy.get("mc.interval_q"))
        ps = plan_samples(profile=profile, steps=steps, queues=queues, policy=policy, seed=seed, delays=delays,
                          learned=learned, overrides=overrides, completed=completed)
        order, dur, twin_f, finish = ps["order"], ps["dur"], ps["twin_f"], ps["finish"]
        cp, ef = _critical_path(order, dur)
        iv = interval(finish, q)
        to_date = lambda d: (start + timedelta(days=float(d))).isoformat()
        return answer({"steps": [{"id": s["id"], "title": s["title"], "blocks_on": s.get("blocks_on", []),
                                  "finish_day": interval(twin_f[s["id"]], q), "critical": s["id"] in cp,
                                  "source_url": s.get("source_url"),
                                  "facts_from_teacher": len((learned or {}).get(s["id"], [])),
                                  "source_card": (overrides or {}).get(s["id"], {}).get("source")} for s in order],
                       "critical_path": cp, "ready_days": iv,
                       "ready_date": {k: to_date(iv[k]) for k in ("low", "mid", "high")},
                       "p_by_target": float(np.mean(finish <= (target - start).days))},
                      unit="days", stores=[profile, steps, queues, policy], formulas=["CPR critical path", "MC"],
                      label="Move-in plan — forecast")
    except MissingData as e:
        return refused(e, what="plan")


def impact(*, profile: Store, steps: Store, queues: Store, policy: Store, seed: int, learned: dict | None = None) -> dict:
    """Final number: how many days earlier the family is ready with CURE than with the usual checklist."""
    try:
        q = tuple(policy.get("mc.interval_q"))
        ps = plan_samples(profile=profile, steps=steps, queues=queues, policy=policy, seed=seed, learned=learned)
        saved = ps["finish_base"] - ps["finish"]
        return answer({"days_saved": interval(saved, q),
                       "baseline_rule": "steps in list order, visits at a random hour",
                       "twin_rule": "steps in parallel by dependencies, visits at the hour with the shortest wait"},
                      unit="days", stores=[profile, steps, queues, policy], formulas=["MC compare, common random numbers"],
                      label="Days saved — model comparison")
    except MissingData as e:
        return refused(e, what="impact")
