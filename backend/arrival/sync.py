"""Fact → teacher, and "the city learns from residents".

The student (the resident's personal layer) reports "the step took N days". The teacher (shared CURE core) accepts it
only through the CURE learning boundary:
  1. ResidualRecord: residual |fact − forecast|, noise σ from the forecast interval;
  2. ResidualFilter: noise / signal / anomaly / refusal (unsigned source);
  3. sanitize: the pseudonym and any personal fields are stripped from the payload;
  4. make_candidate (threshold): candidate θ − η∇L — a PROPOSAL, never written to production;
  5. CompatibilityMatrix.check(federation_import): student and teacher versions are compatible.
The teacher stores the fact as data (not model weights); the next plan mixes it with the catalogue: share n/(n+k).
"""
from __future__ import annotations

import uuid

import numpy as np

from api_gateway.core.learning_boundary.boundary import (BoundaryPolicy, CandidateKind, FilterVerdict, ResidualFilter,
                                                          ResidualRecord, make_candidate, sanitize)
from api_gateway.core.operational_integrity.compatibility import CompatibilityMatrix, VersionVector

from .core import MissingData, Store, answer, refused, sample

EVIDENCE_VERDICTS = (FilterVerdict.NOISE, FilterVerdict.LEARNING_SIGNAL)   # what the teacher stores as an observation


class Teacher:
    """Shared core in the demo: stores only anonymised per-step facts and learning candidates."""

    def __init__(self, *, policy: Store, cohort: Store):
        s = lambda k: policy.get(f"sync.{k}")
        self.policy = BoundaryPolicy(policy_version=policy.meta.get("version", "unknown"),
                                     noise_sigma_multiple=float(s("noise_sigma_multiple")),
                                     anomaly_robust_z=float(s("anomaly_robust_z")), min_window=int(s("min_window")),
                                     pii_keys=tuple(s("pii_keys")), pii_patterns=tuple(s("pii_patterns")))
        self.filter = ResidualFilter(self.policy)
        self.matrix = CompatibilityMatrix()
        # what the teacher knew before — other residents' facts (synthetic, labelled)
        self.facts: dict[str, list[float]] = {k: list(map(float, cohort.get(f"teacher_facts.{k}")))
                                              for k in cohort.raw("teacher_facts")}
        self.origin = {k: "synthetic_residents" for k in self.facts}
        self.log: list[dict] = []

    def learned(self) -> dict:
        return {k: list(v) for k, v in self.facts.items()}


def _prior_stats(steps: Store, step_id: str, policy: Store, seed: int) -> tuple[float, float]:
    spec = next((s for s in steps.raw("steps") if s["id"] == step_id), None)
    if spec is None:
        raise MissingData([f"steps:{step_id}"])
    x = sample(spec["duration_days"], np.random.default_rng(seed), int(policy.get("mc.runs")))
    q = tuple(policy.get("mc.interval_q"))
    lo, hi = np.quantile(x, q)
    return float(np.median(x)), float((hi - lo) / (2 * float(policy.get("sync.sigma_from_interval_z"))))


def submit_fact(teacher: Teacher, *, step_id: str, observed_days: float, source_integrity: str, pseudonym: str,
                steps: Store, policy: Store, versions: Store, seed: int) -> dict:
    try:
        pred, sigma = _prior_stats(steps, step_id, policy, seed)
        ref = f"fact:{uuid.uuid4().hex[:10]}"
        rec = ResidualRecord(subject=f"step:{step_id}", observed=(float(observed_days),), predicted=(pred,),
                             noise_sigma=sigma, source_integrity=source_integrity, evidence_ref=ref)
        fr = teacher.filter.assess(rec)
        # the payload the student would like to send — with a pseudonym; the boundary strips it
        payload = {"step_id": step_id, "observed_days": float(observed_days), "predicted_days": pred,
                   "pseudonym": pseudonym}
        compat = teacher.matrix.check(VersionVector(**versions.raw("student")), VersionVector(**versions.raw("teacher")),
                                      operation="federation_import")
        candidate, sent = None, None
        if compat["result"] != "compatible":
            status = f"blocked_by_versions:{compat['action']}"
        elif fr.verdict in EVIDENCE_VERDICTS:
            n, k = len(teacher.facts.get(step_id, [])), float(policy.get("sync.prior_pseudo_count"))
            if fr.verdict is FilterVerdict.LEARNING_SIGNAL:
                c = make_candidate(CandidateKind.THRESHOLD, subject=f"step:{step_id}", payload=payload,
                                   from_results=[fr], policy=teacher.policy, proposed_by="arrival_student",
                                   theta=[pred], learning_rate=1.0 / (n + 1 + k), loss_gradient=[pred - observed_days],
                                   parameter_names=[f"{step_id}.median_days"])
                candidate = {"kind": c.kind.value, "status": c.status, "real_action": c.real_action,
                             "theta_current": float(c.learning_candidate.theta_current[0]),
                             "theta_candidate": float(c.learning_candidate.theta_candidate[0]),
                             "next_stages": list(c.next_stages), "pii_removed": c.pii_removed}
                sent = dict(c.payload)
            else:
                sent, _ = sanitize(payload, teacher.policy)
            teacher.facts.setdefault(step_id, []).append(float(observed_days))
            teacher.origin[step_id] = "observed"
            status = "stored_as_fact"
        else:
            status = "not_stored"
        entry = {"step_id": step_id, "verdict": fr.verdict.value, "reason": fr.reason, "delta_days": fr.delta,
                 "sigma_days": sigma, "status": status, "sent_to_teacher": sent, "candidate": candidate,
                 "compatibility": {"result": compat["result"], "action": compat["action"]}}
        teacher.log.append(entry)
        return answer(entry, unit="fact", stores=[steps, policy, versions],
                      formulas=["LB ResidualFilter", "LB sanitize", "UMRM θ−η∇L candidate", "OI compatibility"],
                      label="Fact → teacher")
    except MissingData as e:
        return refused(e, what="sync")


def learning_view(teacher: Teacher, *, steps: Store, policy: Store) -> dict:
    """Per step: how many facts the teacher has and where the knowledge comes from (catalogue / synthetic residents / observations)."""
    k = float(policy.get("sync.prior_pseudo_count"))
    rows = []
    for s in steps.raw("steps"):
        n = len(teacher.facts.get(s["id"], []))
        rows.append({"step_id": s["id"], "title": s["title"], "facts": n, "share_from_facts": n / (n + k) if n else 0.0,
                     "origin": teacher.origin.get(s["id"], "catalog_prior")})
    return answer({"steps": rows, "log": teacher.log[-10:]}, unit="steps", stores=[steps, policy],
                  formulas=["mixture n/(n+k)"], label="What the teacher learned — vision")
