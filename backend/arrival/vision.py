"""Vision: CURE runs the city; the resident asks the city to act for her. NOTHING is executed.

V1 city_request      — pseudonym + physical constraints, no diagnosis (checked).
V2 run_chain         — links "plane door → bed": u_allowed + RealTimeEnvelope for each; av_allowed for cars;
   arrival_counterfactual — the child's minutes outside / in dust with and without CURE (walk_time, comfort, PM10).
V3 neighbors         — economics.arbitrate: the "just for us" option is cut, a fair one is chosen.
V4 flight_delay      — +3 h: preload and links recomputed; storm: av_allowed = no, links degrade,
   agents go to shadow, arbitration uses emergency weights, the child's constraint is kept.
Labelled "vision", data_mode synthetic, executed: False.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from api_gateway.core.city_world.formulas import av_allowed, u_allowed
from api_gateway.core.economics.arbitration import ArbitrationPolicy, Candidate, Proposal, Stakeholder, arbitrate
from api_gateway.core.operational_integrity.realtime_envelope import RealTimeEnvelope, check_envelope

from . import agents as ag
from .core import MissingData, Store, answer, interval, refused, sample
from .day import harsh_mask, walk_minutes

FORBIDDEN_KEYS = ("health", "diagnos", "asthma", "астм", "диагноз", "medical", "condition")


def city_request(*, profile: Store, policy: Store) -> dict:
    """V1. The diagnosis stays sealed. Only physical constraints from the policy go out."""
    req = {"pseudonym": profile.get("pseudonym"), "objective": "arrival_home",
           "constraints": {"pm10_ugm3_max": policy.get("vision.pm10_ugm3_max"),
                           "outdoor_min_max": policy.get("vision.outdoor_min_max"),
                           "indoor_c_max_on_arrival": profile.get("params.setpoint_c") + policy.get("comfort.arrival_band_c")},
           "evidence_refs": ["flight:arrival_time"], "action_envelope": [l["action"] for l in policy.raw("vision.links")]}
    flat = repr(req).lower()
    if any(k in flat for k in FORBIDDEN_KEYS):
        raise ValueError("health information must never leave the device")
    return req


def _av(policy: Store, telemetry: dict) -> dict:
    return av_allowed(float(telemetry["v2x_latency_ms"]), float(telemetry["perception_confidence"]),
                      tau_v2x_ms=float(policy.get("vision.av.tau_v2x_ms")), tau_c=float(policy.get("vision.av.tau_c")),
                      safety_case_ref=policy.get("vision.av.safety_case_ref"))


def run_chain(*, profile: Store, policy: Store, scenario: Store, preload_answer: dict, weather: str = "normal",
              label: str = "Vision — how it will work", consent_valid: bool | None = None) -> dict:
    """consent_valid — result of ConsentLedger.allow (purpose arrival_vision); None — profile flag (offline tests)."""
    try:
        req = city_request(profile=profile, policy=policy)
        tel_key = "storm" if weather == "storm" else "av_normal"
        tel = {k: scenario.get(f"{tel_key}.{k}") for k in ("v2x_latency_ms", "perception_confidence")}
        av = _av(policy, tel)
        links = []
        for link in policy.raw("vision.links"):
            env_cfg = policy.raw(f"vision.envelopes.{link['action_class']}")
            env = RealTimeEnvelope(link["action_class"], env_cfg["max_latency_s"], env_cfg["max_staleness_s"],
                                   env_cfg["min_confidence"], env_cfg["degraded"], env_cfg["human_approval"])
            sc = dict(scenario.raw(f"links.{link['id']}"))   # demo synthetic: link latencies, data age
            physical = bool(sc["physical_safe"])
            if link.get("uses_av"):
                physical = physical and av["allowed"]
                sc["confidence"] = min(sc["confidence"], tel["perception_confidence"])
            if link["id"] == "home" and preload_answer.get("calculation_status") == "computed":
                physical = preload_answer["value"]["p_reachable"] >= policy.get("vision.min_p_feasible")
            gate = u_allowed(physical_safe=physical, policy_valid=True,
                             consent_valid=(bool(profile.raw("grants").get("vision_consent")) if consent_valid is None
                                            else consent_valid),
                             authorized=bool(sc["authorized"]))
            envr = check_envelope(env, stages_s=sc["stages_s"], data_age_s=sc["data_age_s"],
                                  confidence=sc["confidence"], human_approved=bool(sc["human_approved"]))
            status = ("ok" if gate["allowed"] and envr["within_envelope"]
                      else "blocked: " + ", ".join(gate["failed"]) if not gate["allowed"] else envr["behavior"])
            links.append({"id": link["id"], "title": link["title"], "action": link["action"],
                          "uses_av": bool(link.get("uses_av")), "u_allowed": gate, "envelope": envr, "status": status})
        return answer({"request_sent_to_city": req, "weather": weather, "av_allowed": av, "links": links,
                       "executed": False},
                      unit="chain", stores=[profile, policy, scenario],
                      formulas=["CITY u_allowed", "CITY av_allowed", "GAP1 L_e2e envelope"], label=label)
    except MissingData as e:
        return refused(e, what="vision")


def arrival_counterfactual(*, profile: Store, policy: Store, climate: Store, scenario: Store, seed: int,
                           delay_h: int = 0, pm_override: float | None = None) -> dict:
    """The child's minutes outside, in heat and in dust above the limit: without CURE (as usual) and with CURE (the chain)."""
    try:
        q = tuple(policy.get("mc.interval_q"))
        runs = int(policy.get("mc.runs"))
        month = scenario.get("arrival.month")
        h = (int(scenario.get("arrival.arrival_h")) + delay_h) % 24
        son = next(p for p in profile.raw("household") if p["id"] == "son")
        rng = np.random.default_rng(seed)
        temp = climate.get(f"months.{month}.temperature_c")[h] + rng.normal(0, climate.get(f"months.{month}.temperature_sd_c"), runs)
        hum = climate.get(f"months.{month}.humidity_pct")[h]
        pm = climate.get(f"months.{month}.pm10_ugm3")[h] if pm_override is None else pm_override
        heat = harsh_mask(policy, temp, hum, 0.0, pm_limit=np.inf)                     # heat/humidity only
        dust = np.full(runs, pm > float(policy.get("vision.pm10_ugm3_max")))            # limit from the V1 request
        out = {}
        for world in ("without", "with"):
            wait = sample(scenario.raw(f"arrival.{world}.taxi_wait_min"), rng, runs)
            outdoor = (walk_minutes(policy, son, scenario.get(f"arrival.{world}.walk_to_taxi_m")) + wait
                       + walk_minutes(policy, son, scenario.get(f"arrival.{world}.walk_home_m")))
            out[world] = {"outdoor_min": interval(outdoor, q), "heat_min": interval(np.where(heat, outdoor, 0.0), q),
                          "dust_min": interval(np.where(dust, outdoor, 0.0), q),
                          "_dust": np.where(dust, outdoor, 0.0), "_out": outdoor}
        saved_out = out["without"].pop("_out") - out["with"].pop("_out")
        saved_dust = out["without"].pop("_dust") - out["with"].pop("_dust")
        return answer({"arrival_hour": h, "month": month, "pm10_at_arrival": pm, **out,
                       "saved": {"outdoor_min": interval(saved_out, q), "dust_min": interval(saved_dust, q)}},
                      unit="minutes", stores=[profile, policy, climate, scenario],
                      formulas=["CITY walk_time", "CITY comfort hard limits", "PM10 limit from V1"],
                      label="Landing with and without CURE — vision")
    except MissingData as e:
        return refused(e, what="arrival_counterfactual")


def neighbors(*, policy: Store, scenario: Store, emergency: bool = False) -> dict:
    """V3. Lobby candidates; "max purge just for us" breaks the neighbours' capacity → cut."""
    try:
        cfg = policy.raw("vision.arbitration")
        stake = tuple(Stakeholder(s["stakeholder_id"], tuple(s["objectives"]), tuple(s["action_scope"]))
                      for s in scenario.raw("neighbors.stakeholders"))
        pol = ArbitrationPolicy(policy_version=cfg["policy_version"], weights=cfg["weights"],
                                emergency_weights=cfg["emergency_weights"], inequity_lambda=cfg["inequity_lambda"],
                                inequity_measure=cfg["inequity_measure"], stakeholders=stake)
        objs = list(cfg["weights"])
        cands = [Candidate(c["action_id"], {o: c[o] for o in objs}, c["feasibility"], ("building.lobby_ventilation",))
                 for c in scenario.raw("neighbors.candidates")]
        props = [Proposal(p["stakeholder_id"], p["objective"], p["constraints"], p["priority"], tuple(p["evidence_refs"]),
                          tuple(p["action_envelope"])) for p in scenario.raw("neighbors.proposals")]
        r = arbitrate(cands, pol, proposals=props, emergency=emergency)
        notes = {c["action_id"]: c["note"] for c in scenario.raw("neighbors.candidates")}
        return answer({"chosen": r.chosen, "chosen_note": notes.get(r.chosen),
                       "chosen_note_ru": {c["action_id"]: c.get("note_ru", c["note"]) for c in scenario.raw("neighbors.candidates")}.get(r.chosen), "feasible": list(r.feasible),
                       "infeasible": {k: list(v) for k, v in r.infeasible.items()}, "pareto": list(r.pareto),
                       "scores": dict(r.scores), "weights_used": dict(r.weights_used), "emergency": r.emergency,
                       "proposals": list(r.proposals), "notes": notes, "decided_by": r.decided_by, "executed": False},
                      unit="decision", stores=[policy, scenario], formulas=["ECON arbitrate: F → Pareto → weighted+inequity"],
                      label="Neighbours: arbitration — vision")
    except MissingData as e:
        return refused(e, what="neighbors")


def storm(*, profile: Store, policy: Store, climate: Store, scenario: Store, rates: Store, preload_answer: dict,
          agents: list[dict], seed: int, now: datetime, consent_valid: bool | None = None) -> dict:
    """V4b. Dust storm: cars fail av_allowed, links degrade per their envelopes,
    agents go to shadow, arbitration uses emergency weights. The child's constraint in the request does not change."""
    try:
        chain = run_chain(profile=profile, policy=policy, scenario=scenario, preload_answer=preload_answer,
                          weather="storm", label="Storm — vision", consent_valid=consent_valid)
        cf = arrival_counterfactual(profile=profile, policy=policy, climate=climate, scenario=scenario, seed=seed,
                                    pm_override=float(scenario.get("storm.pm10_ugm3")))
        arb = neighbors(policy=policy, scenario=scenario, emergency=True)
        shadow = ag.on_event(agents, event="dust_forecast", rates=rates, now=now, mode="shadow",
                             payload={"reason": "storm"}) if agents else []
        return {"chain": chain, "arrival": cf, "arbitration": arb, "agents_shadow": shadow,
                "agents_mode": {a["id"]: "shadow" for a in agents},
                "constraint_kept": city_request(profile=profile, policy=policy)["constraints"]}
    except MissingData as e:
        return refused(e, what="storm")
