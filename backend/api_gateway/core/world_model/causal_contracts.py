from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from math import isfinite
from typing import Any, Mapping

from .assertions import require_aware


class MechanismLifecycle(str, Enum):
    STRUCTURAL_POSSIBLE = "structural_possible"
    OBSERVATIONAL_CANDIDATE = "observational_candidate"
    MECHANISM_GROUNDED = "mechanism_grounded"
    INTERVENTION_VALIDATED = "intervention_validated"


class CausalQueryLevel(str, Enum):
    OBSERVATION = "observation"
    #: factual rollout ("predict" query of the CEOS p.8 contract); never action authority by itself
    PREDICTION = "prediction"
    INTERVENTION = "intervention"
    COUNTERFACTUAL = "counterfactual"


class MechanismOrigin(str, Enum):
    """Who proposed a mechanism. Registration rung is capped by origin (fail-closed for unknown)."""

    UNSPECIFIED = "unspecified"
    SITE_COMPILER = "site_compiler"
    DOMAIN_PACK = "domain_pack"
    OPERATOR = "operator"
    FEDERATION = "federation"
    DISCOVERY = "discovery"
    LLM = "llm"


class PlanningUse(str, Enum):
    INFORMATIONAL = "informational"
    REVIEW = "review"
    BOUNDED = "bounded"


@dataclass(frozen=True)
class CausalMechanism:
    edge_id: str
    source_variable: str
    target_variable: str
    mechanism_type: str
    functional_form: str
    conditions: Mapping[str, Any]
    lag_distribution: Mapping[str, Any]
    scope: tuple[str, ...]
    physical_topological_basis: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    counterexamples: tuple[str, ...]
    causal_confidence: float
    trust_level: float
    scm_version: str
    valid_from: datetime
    valid_to: datetime | None
    lifecycle: MechanismLifecycle
    version: int = 1
    # Verified OutcomeLedger ids attached by the registry on promotion to intervention_validated.
    outcome_refs: tuple[str, ...] = ()
    origin: MechanismOrigin = MechanismOrigin.UNSPECIFIED

    def __post_init__(self) -> None:
        required = (
            self.edge_id,
            self.source_variable,
            self.target_variable,
            self.mechanism_type,
            self.functional_form,
            self.scm_version,
        )
        if any(not value.strip() for value in required) or not self.scope:
            raise ValueError("causal mechanism identity, form, scope, and SCM version are required")
        if not 0.0 <= float(self.causal_confidence) <= 1.0:
            raise ValueError("causal_confidence must be in [0, 1]")
        if not 0.0 <= float(self.trust_level) <= 1.0:
            raise ValueError("trust_level must be in [0, 1]")
        if self.version < 1:
            raise ValueError("version must be positive")
        object.__setattr__(self, "origin", MechanismOrigin(self.origin))
        object.__setattr__(self, "valid_from", require_aware(self.valid_from, "valid_from"))
        if self.valid_to is not None:
            object.__setattr__(self, "valid_to", require_aware(self.valid_to, "valid_to"))
            if self.valid_to <= self.valid_from:
                raise ValueError("valid_to must be after valid_from")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(float(value))


def has_quantitative_uncertainty(uncertainty: Mapping[str, Any]) -> bool:
    """CEOS p.7: Monte Carlo Ŷ with CI = [Q_α/2(Y), Q_1−α/2(Y)].

    Required shape: {"alpha": α in (0,1), "samples": N > 0, "min_tail_samples": k >= 1,
    "intervals": {var: {"mean","lower","upper"}}}.  The empirical quantile Q_α/2 is estimable only when
    each tail holds at least k draws: N·α/2 >= k (k is policy; CEOS fixes none, k >= 1 is the floor).
    Every interval must be finite with lower <= mean <= upper and a non-degenerate width lower < upper:
    a zero-width CI (e.g. N = 1) is a point estimate, not a quantified uncertainty (fail-closed).
    """
    if not isinstance(uncertainty, Mapping):
        return False
    alpha, samples, intervals = uncertainty.get("alpha"), uncertainty.get("samples"), uncertainty.get("intervals")
    tail = uncertainty.get("min_tail_samples")
    if not _finite_number(alpha) or not 0.0 < float(alpha) < 1.0:
        return False
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        return False
    if not isinstance(tail, int) or isinstance(tail, bool) or tail < 1 or samples * float(alpha) / 2.0 < tail:
        return False
    if not isinstance(intervals, Mapping) or not intervals:
        return False
    for row in intervals.values():
        if not isinstance(row, Mapping) or not all(_finite_number(row.get(k)) for k in ("mean", "lower", "upper")):
            return False
        if not float(row["lower"]) <= float(row["mean"]) <= float(row["upper"]):
            return False
        if not float(row["lower"]) < float(row["upper"]):
            return False
    return True


# BOUNDED (action-usable) results are sealed by the facade after it re-verified every accepted mechanism
# against the CausalMechanismRegistry (seal_bounded_result).  The key lives only in this process; a
# CausalResult built directly with a hand-written ``mechanism_checks`` dict carries no valid seal.
_SEAL_KEY = secrets.token_bytes(32)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _bounded_digest(result: "CausalResult") -> str:
    body = _canonical_json({
        "id": result.causal_result_id, "trace": result.trace_id, "level": result.query_level.value,
        "scope": list(result.scope), "snapshot": result.world_snapshot_id, "scm": result.scm_version,
        "accepted": list(result.accepted_mechanism_ids), "assumptions": list(result.assumptions),
        "uncertainty": result.uncertainty, "evidence": list(result.evidence_refs), "body": result.body_id,
        "checks": dict(result.metadata or {}).get("mechanism_checks"),
    })
    return hmac.new(_SEAL_KEY, body.encode("utf-8"), hashlib.sha256).hexdigest()


def seal_bounded_result(result: "CausalResult", *, registry: Any, at: datetime | None) -> "CausalResult":
    """Grant ``bounded`` to an intervention result after re-verifying, against the mechanism registry,
    that every accepted mechanism is known, planning-eligible (intervention_validated via the
    OutcomeLedger) at ``at`` and of the result's SCM version.  The seal binds the verified fields; any
    later change (or a result never passed through here) fails the BOUNDED check in ``__post_init__``."""
    if result.query_level is not CausalQueryLevel.INTERVENTION or not result.accepted_mechanism_ids:
        raise ValueError("only an intervention result with accepted mechanisms can be sealed as bounded")
    for edge_id in result.accepted_mechanism_ids:
        if not registry.contains(edge_id):
            raise ValueError(f"mechanism {edge_id} is not in the registry")
        if not registry.planning_eligible(edge_id, at=at):
            raise ValueError(f"mechanism {edge_id} is not planning-eligible (intervention_validated) now")
        if registry.current(edge_id).scm_version != result.scm_version:
            raise ValueError(f"mechanism {edge_id} belongs to another SCM version")
    metadata = {k: v for k, v in dict(result.metadata or {}).items() if k != "bounded_seal"}
    unsealed = replace(result, metadata=metadata)
    metadata["bounded_seal"] = _bounded_digest(unsealed)
    return replace(unsealed, allowed_planning_use=PlanningUse.BOUNDED, metadata=metadata)


def _passed_check(check: Any) -> bool:
    return (
        isinstance(check, Mapping)
        and isinstance(check.get("failures"), (list, tuple))
        and len(check["failures"]) == 0
        and check.get("lifecycle") == MechanismLifecycle.INTERVENTION_VALIDATED.value
        and check.get("planning_eligible") is True
    )


@dataclass(frozen=True)
class CausalResult:
    causal_result_id: str
    trace_id: str
    query_level: CausalQueryLevel
    scope: tuple[str, ...]
    world_snapshot_id: str
    scm_version: str
    hypotheses: tuple[Any, ...]
    accepted_mechanism_ids: tuple[str, ...]
    rejected_mechanism_ids: tuple[str, ...]
    assumptions: tuple[str, ...]
    unknowns: tuple[str, ...]
    simulation: Mapping[str, Any]
    uncertainty: Mapping[str, Any]
    recommended_next_observations: tuple[str, ...]
    allowed_planning_use: PlanningUse
    evidence_refs: tuple[str, ...]
    epistemic_status: str
    execution_status: str = "not_executed_simulation"
    real_action: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    #: CEOS p.8 contract: "scope": {"body_id": ..., "entity_ids": [...]}
    body_id: str = ""

    def __post_init__(self) -> None:
        if not self.causal_result_id.strip() or not self.trace_id.strip() or not self.scope:
            raise ValueError("causal_result_id, trace_id, and scope are required")
        if not isinstance(self.query_level, CausalQueryLevel) or not isinstance(self.allowed_planning_use, PlanningUse):
            raise ValueError("query_level and allowed_planning_use must be contract enums")
        if self.query_level is CausalQueryLevel.COUNTERFACTUAL and self.epistemic_status not in {"simulated", "estimated"}:
            raise ValueError("counterfactual result must remain simulated or estimated")
        if self.query_level is CausalQueryLevel.INTERVENTION and self.epistemic_status not in {"simulated", "estimated"}:
            raise ValueError("intervention result is a simulation and must remain simulated or estimated")
        if self.query_level is CausalQueryLevel.PREDICTION and self.epistemic_status != "predicted":
            raise ValueError("prediction result must remain predicted (never observed)")
        if self.epistemic_status in {"observed", "manual_verified"}:
            raise ValueError("a causal engine result is never a measurement")
        if self.allowed_planning_use is PlanningUse.BOUNDED:
            # CEOS p.2/p.9: observation (association) and counterfactual (post-incident) levels never
            # authorize bounded action; only a validated intervention simulation may.
            if self.query_level is not CausalQueryLevel.INTERVENTION:
                raise ValueError(f"{self.query_level.value}-level causal result cannot be bounded (action-usable)")
            if not self.accepted_mechanism_ids:
                raise ValueError("action-usable causal result requires accepted intervention-validated mechanisms")
            if not self.world_snapshot_id or not self.scm_version or not self.evidence_refs or not self.uncertainty:
                raise ValueError("action-usable causal result requires snapshot, SCM version, evidence, and uncertainty")
            if not str(self.body_id).strip():
                raise ValueError("action-usable causal result requires scope.body_id (CEOS p.8-9)")
            if not any(str(item).strip() for item in self.assumptions):
                raise ValueError("action-usable causal result must state its assumptions (CEOS p.10, test 6)")
            if not has_quantitative_uncertainty(self.uncertainty):
                # CEOS p.7/p.10: a distribution / CI = [Q_a/2, Q_1-a/2], not a label such as "point_estimate".
                raise ValueError("action-usable causal result requires quantitative uncertainty (Monte Carlo CI)")
            # Every accepted mechanism must carry a passed registry check produced by the facade that
            # verified its lifecycle status; bare ids (e.g. invented by a caller) never qualify.
            checks = dict(self.metadata or {}).get("mechanism_checks")
            if not isinstance(checks, Mapping):
                raise ValueError("action-usable causal result requires registry mechanism_checks")
            unchecked = [mid for mid in self.accepted_mechanism_ids if not _passed_check(checks.get(mid))]
            if unchecked:
                raise ValueError(f"accepted mechanisms without a passed registry check: {unchecked}")
            seal = dict(self.metadata or {}).get("bounded_seal")
            if not isinstance(seal, str) or not hmac.compare_digest(seal, _bounded_digest(self)):
                raise ValueError("bounded causal result must be sealed by the facade after registry re-verification")
        if self.real_action or self.execution_status != "not_executed_simulation":
            raise ValueError("causal result cannot authorize or report physical execution")

    def to_contract_dict(self) -> dict[str, Any]:
        query = {
            CausalQueryLevel.OBSERVATION: "why",
            CausalQueryLevel.PREDICTION: "predict",
            CausalQueryLevel.INTERVENTION: "simulate",
            CausalQueryLevel.COUNTERFACTUAL: "counterfactual",
        }[self.query_level]
        return {
            "causal_result_id": self.causal_result_id,
            "trace_id": self.trace_id,
            "query": query,
            "scope": {"body_id": self.body_id, "entity_ids": list(self.scope)},
            "world_snapshot": self.world_snapshot_id,
            "scm_version": self.scm_version,
            "hypotheses": list(self.hypotheses),
            "accepted_mechanisms": list(self.accepted_mechanism_ids),
            "rejected_mechanisms": list(self.rejected_mechanism_ids),
            "assumptions": list(self.assumptions),
            "unknowns": list(self.unknowns),
            "simulation": dict(self.simulation),
            "uncertainty": dict(self.uncertainty),
            "recommended_next_observation": list(self.recommended_next_observations),
            "allowed_planning_use": self.allowed_planning_use.value,
            "evidence_refs": list(self.evidence_refs),
            "epistemic_status": self.epistemic_status,
            "execution_status": self.execution_status,
            "real_action": self.real_action,
        }


__all__ = [
    "MechanismOrigin",
    "has_quantitative_uncertainty",
    "CausalMechanism",
    "CausalQueryLevel",
    "CausalResult",
    "MechanismLifecycle",
    "PlanningUse",
]

