"""Why the date moved, what would have happened without the delay and what helps most — the CURE causal engine.

A structural causal model (core SCM: world_model.structural_causal_model) of the move-in steps is built:
    finish(step) = max(finish(predecessors)) + wait(step) + dur(step) [+ visa_delay for the entry permit] + U
Root variables (durations, waits, visa delay) are fixed from the plan's factual trace (plan engine medians).
"What if" questions are answered by CausalMetaEngine.counterfactual: abduction → action → prediction.
The result is an estimate (simulated, real_action=False), not a measured fact.
"""
from __future__ import annotations

import asyncio

import numpy as np

from api_gateway.core.world_model.causal_meta_engine import CausalMetaEngine
from api_gateway.core.world_model.structural_causal_model import StructuralCausalModel

from .core import MissingData, Store, answer, refused
from .plan import _mean, _steps, plan_samples

ENTRY = "entry_permit"


def _factual_roots(order, steps: Store, queues: Store, ps: dict, visa_delay: float) -> dict:
    roots = {"visa_delay": float(visa_delay)}
    for s in order:
        roots[f"{s['id']}_dur"] = float(np.median(ps["dur"][s["id"]])) - (visa_delay if s["id"] == ENTRY else 0.0)
        if s.get("requires_visit"):
            hours = queues.raw(f"centers.{s['center']}.wait_days_by_hour")
            roots[f"{s['id']}_wait"] = float(min(_mean(hours[h]) for h in hours))   # CURE picks the best hour
    return roots


def build_scm(order, version: str) -> StructuralCausalModel:
    scm = StructuralCausalModel()
    for s in order:
        sid = s["id"]
        preds = tuple(s.get("blocks_on", []))
        extra = tuple(x for x in (f"{sid}_wait" if s.get("requires_visit") else None,
                                  "visa_delay" if sid == ENTRY else None) if x)
        parents = preds + (f"{sid}_dur",) + extra

        def f(p, u, _params, preds=preds, sid=sid, extra=extra):
            start = max((p[b] for b in preds), default=0.0)
            return start + p[f"{sid}_dur"] + sum(p[x] for x in extra) + u
        scm.register_equation(variable=sid, parents=parents, function=f, parameters={}, version=version)
    ids = tuple(s["id"] for s in order)
    scm.register_equation(variable="ready", parents=ids, function=lambda p, u, _t: max(p.values()) + u,
                          parameters={}, version=version)
    return scm


def _forward(order, roots: dict) -> dict:
    vals = dict(roots)
    for s in order:
        sid = s["id"]
        start = max((vals[b] for b in s.get("blocks_on", [])), default=0.0)
        vals[sid] = start + roots[f"{sid}_dur"] + roots.get(f"{sid}_wait", 0.0) + (roots["visa_delay"] if sid == ENTRY else 0.0)
    vals["ready"] = max(vals[s["id"]] for s in order)
    return vals


def _cf(engine, trace, variable, actual, alternate, targets, policy: Store):
    res = asyncio.run(engine.counterfactual(
        factual_trace=trace, actual_action={"variable": variable, "value": actual},
        alternate_action={"variable": variable, "value": alternate}, target_variables=tuple(targets),
        scope=("resident_plan",), samples=int(policy.get("causal.samples")), seed=int(policy.get("causal.seed"))))
    d = res.to_contract_dict()
    return {t: float(d["simulation"]["distribution"][t]["mean"]) for t in targets}, d


def explain_delay(*, profile: Store, steps: Store, queues: Store, policy: Store, seed: int, visa_delay_days: float,
                  move_in_step: str = "utilities") -> dict:
    """Why (path in the causal graph), what if there were no delay, what helps most (levers)."""
    try:
        version = str(policy.get("causal.scm_version"))
        order = _steps(steps, profile)
        ps = plan_samples(profile=profile, steps=steps, queues=queues, policy=policy, seed=seed,
                          delays={ENTRY: visa_delay_days})
        roots = _factual_roots(order, steps, queues, ps, visa_delay_days)
        trace = _forward(order, roots)
        scm = build_scm(order, version)
        engine = CausalMetaEngine(scm=scm)
        ids = [s["id"] for s in order]
        targets = ids + ["ready"]
        alt, raw = _cf(engine, trace, "visa_delay", visa_delay_days, 0.0, targets, policy)
        shift = {k: trace[k] - alt[k] for k in targets}
        chain = [k for k in ids if shift[k] > 0.5]
        untouched = [k for k in ids if shift[k] <= 0.5]
        # levers: each intervention is a separate counterfactual on the same model
        lev = []
        for lv in policy.get("plan_levers"):
            var = lv["variable"]
            if var not in trace:
                continue
            new = float(lv["value"]) if "value" in lv else float(trace[var]) * float(lv["scale"])
            out, _ = _cf(engine, trace, var, trace[var], new, [move_in_step], policy)
            lev.append({"id": lv["id"], "title": lv["title"], "variable": var,
                        "days_saved": trace[move_in_step] - out[move_in_step]})
        lev.sort(key=lambda x: -x["days_saved"])
        titles = {s["id"]: s["title"] for s in order}
        return answer({"visa_delay_days": visa_delay_days,
                       "move_in_step": move_in_step,
                       "why": {"chain": chain, "chain_titles": [titles[c] for c in chain],
                               "untouched": untouched, "untouched_titles": [titles[c] for c in untouched],
                               "shift_days": {k: round(v, 2) for k, v in shift.items()}},
                       "counterfactual": {"with_delay_day": trace[move_in_step], "without_delay_day": alt[move_in_step],
                                          "days_lost": shift[move_in_step]},
                       "levers": lev,
                       "engine": {"facade": "CausalMetaEngine.counterfactual", "scm_version": version,
                                  "process": list(raw["simulation"]["counterfactual_process"]),
                                  "epistemic_status": raw.get("epistemic_status"), "real_action": False}},
                      unit="days", stores=[profile, steps, queues, policy],
                      formulas=["CEOS counterfactual: abduction → action → prediction (CausalMetaEngine)",
                                "SCM: finish = max(predecessors) + wait + duration + U"],
                      label="Why the date moved — causal computation")
    except MissingData as e:
        return refused(e, what="explain_delay")
