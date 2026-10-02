from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from math import isfinite
from typing import Any, Callable, Iterable, Mapping, Sequence

from .assertions import require_aware
from .causal_contracts import CausalMechanism, MechanismLifecycle, MechanismOrigin
from .causal_mechanisms import NON_OBSERVATIONAL_PREFIXES, admissible_causal_pairs


_ORDER = (
    MechanismLifecycle.STRUCTURAL_POSSIBLE,
    MechanismLifecycle.OBSERVATIONAL_CANDIDATE,
    MechanismLifecycle.MECHANISM_GROUNDED,
    MechanismLifecycle.INTERVENTION_VALIDATED,
)
_NEXT = {current: following for current, following in zip(_ORDER, _ORDER[1:])}
_PREVIOUS = {following: current for current, following in zip(_ORDER, _ORDER[1:])}
# Registration never exceeds observational_candidate; higher rungs only via promote() (CEOS p.4-5,
# engineering spec F23).
REGISTRATION_CEILING = MechanismLifecycle.OBSERVATIONAL_CANDIDATE
MIN_VERIFIED_OUTCOMES = 3  # engineering spec F23: len(verified outcomes) >= 3

# CEOS p.4-5, column "May affect action?" — one table for the one registry.
ALLOWED_USE = {
    MechanismLifecycle.STRUCTURAL_POSSIBLE: "no",
    MechanismLifecycle.OBSERVATIONAL_CANDIDATE: "investigation_only",
    MechanismLifecycle.MECHANISM_GROUNDED: "recommendation_or_human_review",
    MechanismLifecycle.INTERVENTION_VALIDATED: "limited_autonomy_if_policy_permits",
}
# Registration rung is capped by origin; unknown origin -> structural_possible (fail-closed).
# A discovery algorithm reaches observational_candidate only through register_discovery_candidate(),
# i.e. only inside H_causal (CEOS p.5).
ORIGIN_CAP = {
    MechanismOrigin.UNSPECIFIED: MechanismLifecycle.STRUCTURAL_POSSIBLE,
    MechanismOrigin.SITE_COMPILER: MechanismLifecycle.STRUCTURAL_POSSIBLE,
    MechanismOrigin.LLM: MechanismLifecycle.STRUCTURAL_POSSIBLE,
    MechanismOrigin.DISCOVERY: MechanismLifecycle.STRUCTURAL_POSSIBLE,
    MechanismOrigin.DOMAIN_PACK: MechanismLifecycle.OBSERVATIONAL_CANDIDATE,
    MechanismOrigin.OPERATOR: MechanismLifecycle.OBSERVATIONAL_CANDIDATE,
    MechanismOrigin.FEDERATION: MechanismLifecycle.OBSERVATIONAL_CANDIDATE,
}


def _rank(level: MechanismLifecycle) -> int:
    return _ORDER.index(level)


def _require_observational(refs: Iterable[str]) -> tuple[str, ...]:
    values = tuple(str(ref) for ref in refs)
    bad = [ref for ref in values if not ref.strip() or ref.strip().lower().startswith(NON_OBSERVATIONAL_PREFIXES)]
    if bad:
        raise ValueError(f"not observational evidence (LLM/simulation/empty refs cannot support a mechanism): {bad}")
    return values


class CausalMechanismRegistry:
    """Append-only in-process registry with a fail-closed trust ladder.

    structural_possible -> observational_candidate -> mechanism_grounded -> intervention_validated,
    one rung per promote(), each rung with its own evidence rule; counterexamples demote and block;
    only a current (non-expired) intervention_validated mechanism without counterexamples is
    planning-eligible.
    """

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._history: dict[str, list[CausalMechanism]] = {}
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._validations: dict[str, Mapping[str, Any]] = {}

    def register(self, mechanism: CausalMechanism) -> CausalMechanism:
        return self._register(mechanism, via_discovery_boundary=False)

    def _register(self, mechanism: CausalMechanism, *, via_discovery_boundary: bool) -> CausalMechanism:
        if mechanism.edge_id in self._history:
            raise ValueError(f"mechanism already registered: {mechanism.edge_id}")
        if _rank(mechanism.lifecycle) > _rank(REGISTRATION_CEILING):
            raise ValueError(
                f"registration cannot exceed {REGISTRATION_CEILING.value}; "
                f"{mechanism.lifecycle.value} is reachable only through promote()"
            )
        if not any(str(item).strip() for item in mechanism.physical_topological_basis):
            raise ValueError("mechanism needs a physical/topological basis")
        _require_observational(mechanism.evidence_refs)
        if mechanism.lifecycle is not MechanismLifecycle.STRUCTURAL_POSSIBLE and not mechanism.evidence_refs:
            raise ValueError("evidence_refs required above structural_possible")
        if mechanism.outcome_refs:
            raise ValueError("outcome_refs are attached by promote(), not at registration")
        cap = ORIGIN_CAP[mechanism.origin]
        if via_discovery_boundary and mechanism.origin is MechanismOrigin.DISCOVERY:
            cap = MechanismLifecycle.OBSERVATIONAL_CANDIDATE
        if _rank(mechanism.lifecycle) > _rank(cap):
            if mechanism.origin is MechanismOrigin.DISCOVERY:
                raise ValueError(
                    "a discovered edge becomes observational_candidate only via register_discovery_candidate() "
                    "(inside H_causal, CEOS p.5)"
                )
            raise ValueError(
                f"origin {mechanism.origin.value!r} cannot register above {cap.value}; higher rungs only via promote()"
            )
        self._history[mechanism.edge_id] = [mechanism]
        return mechanism

    def register_discovery_candidate(
        self,
        mechanism: CausalMechanism,
        *,
        topology: Iterable[tuple[str, str]],
        spatial: Iterable[tuple[str, str]],
        temporal: Iterable[tuple[str, str]],
        ontology: Iterable[tuple[str, str]],
        constraints: Iterable[tuple[str, str]],
    ) -> CausalMechanism:
        """CEOS p.5: H_causal = G_topology ∩ G_spatial ∩ G_temporal ∩ G_ontology ∩ G_constraints.

        Each argument is the set of admissible (cause, effect) pairs of that layer; the admissible
        set is computed here, never asserted by the caller.
        """

        admissible = admissible_causal_pairs(
            topology={tuple(pair) for pair in topology},
            spatial={tuple(pair) for pair in spatial},
            temporal={tuple(pair) for pair in temporal},
            ontology={tuple(pair) for pair in ontology},
            constraints={tuple(pair) for pair in constraints},
        ).value
        if (mechanism.source_variable, mechanism.target_variable) not in admissible:
            raise ValueError("discovered edge is outside admissible causal boundary H_causal")
        candidate = replace(
            mechanism, lifecycle=MechanismLifecycle.OBSERVATIONAL_CANDIDATE, origin=MechanismOrigin.DISCOVERY
        )
        return self._register(candidate, via_discovery_boundary=True)

    def current(self, edge_id: str) -> CausalMechanism:
        try:
            return self._history[edge_id][-1]
        except (KeyError, IndexError) as exc:
            raise KeyError(f"unknown mechanism: {edge_id}") from exc

    def contains(self, edge_id: str) -> bool:
        return str(edge_id) in self._history

    def is_current(self, edge_id: str, *, at: datetime | None = None) -> bool:
        mechanism = self.current(edge_id)
        moment = require_aware(at, "at") if at is not None else self._clock()
        if moment < mechanism.valid_from:
            return False
        return mechanism.valid_to is None or moment < mechanism.valid_to

    def planning_eligible(self, edge_id: str, *, at: datetime | None = None) -> bool:
        if not self.contains(edge_id):
            return False
        mechanism = self.current(edge_id)
        return (
            mechanism.lifecycle is MechanismLifecycle.INTERVENTION_VALIDATED
            and not mechanism.counterexamples
            and self.is_current(edge_id, at=at)
        )

    def history(self, edge_id: str) -> tuple[CausalMechanism, ...]:
        return tuple(self._history.get(edge_id, ()))

    def allowed_use(self, edge_id: str, *, at: datetime | None = None) -> str:
        """CEOS p.4-5 "May affect action?" for the current version; outside validity -> revalidate."""
        if not self.contains(edge_id):
            return "no"
        if not self.is_current(edge_id, at=at):
            return "expired_revalidate"
        return ALLOWED_USE[self.current(edge_id).lifecycle]

    def mechanisms(self) -> tuple[CausalMechanism, ...]:
        return tuple(history[-1] for history in self._history.values())

    def validation_record(self, edge_id: str) -> Mapping[str, Any] | None:
        return self._validations.get(edge_id)

    def promote(
        self,
        *,
        edge_id: str,
        target: MechanismLifecycle,
        evidence_refs: tuple[str, ...] = (),
        dynamics_rmse: float | None = None,
        rmse_limit: float | None = None,
        outcome_refs: tuple[str, ...] = (),
        mean_abs_error: float | None = None,
        error_limit: float | None = None,
        at: datetime | None = None,
    ) -> CausalMechanism:
        """One rung per call, by the engineering spec F23 rules (same as causal_mechanisms registry)."""

        current = self.current(edge_id)
        if _NEXT.get(current.lifecycle) is not target:
            raise ValueError(f"invalid lifecycle transition: {current.lifecycle.value} -> {target.value}")
        if current.counterexamples:
            raise ValueError("open counterexamples block promotion")
        if not self.is_current(edge_id, at=at):
            raise ValueError("mechanism is outside its validity window; revalidate before promotion")
        evidence = _require_observational(evidence_refs)
        outcomes = tuple(dict.fromkeys(str(ref).strip() for ref in outcome_refs))
        if any(not ref for ref in outcomes):
            raise ValueError("outcome_refs must be non-empty verified outcome ids")
        if target is MechanismLifecycle.OBSERVATIONAL_CANDIDATE:
            if not evidence:
                raise ValueError("observational_candidate requires observational evidence_refs")
        elif target is MechanismLifecycle.MECHANISM_GROUNDED:
            if dynamics_rmse is None or rmse_limit is None or not float(dynamics_rmse) < float(rmse_limit):
                raise ValueError("mechanism_grounded needs calibrated dynamics with RMSE below the class limit")
        elif target is MechanismLifecycle.INTERVENTION_VALIDATED:
            # CEOS p.9 / test 1 & 10: caller-supplied outcome ids and errors are never trusted.
            raise ValueError(
                "intervention_validated is reachable only via promote_from_outcome_ledger(): outcome ids must be "
                "verified in the OutcomeLedger and the error computed from it, through an approved LearningCandidate"
            )
        promoted = replace(
            current,
            lifecycle=target,
            evidence_refs=tuple(dict.fromkeys((*current.evidence_refs, *evidence))),
            outcome_refs=tuple(dict.fromkeys((*current.outcome_refs, *outcomes))),
            version=current.version + 1,
        )
        self._history[edge_id].append(promoted)
        return promoted

    async def promote_from_outcome_ledger(
        self,
        *,
        edge_id: str,
        outcome_ledger: Any,
        outcome_ids: Sequence[str],
        error_limit: float,
        learning_candidate: Any,
        at: datetime | None = None,
    ) -> CausalMechanism:
        """mechanism_grounded -> intervention_validated (CEOS p.4-5, p.9; engineering spec F23).

        * the update is a causal_calibration LearningCandidate accepted by governance review (EBM §11 status
          "accepted"), which references
          this edge in ``causal_refs`` and every outcome id in ``outcome_refs`` -- nothing self-promotes;
        * >= MIN_VERIFIED_OUTCOMES distinct outcome ids, each learning-eligible in the OutcomeLedger
          (verified, really executed, evidence, baseline, unconfounded);
        * each outcome measures this mechanism: metric == target variable, same SCM version;
        * mean |epsilon_outcome| is computed from the ledger records and must be below the policy limit.
        """

        current = self.current(edge_id)
        if current.lifecycle is not MechanismLifecycle.MECHANISM_GROUNDED:
            raise ValueError(f"invalid lifecycle transition: {current.lifecycle.value} -> intervention_validated")
        if current.counterexamples:
            raise ValueError("open counterexamples block promotion")
        if not self.is_current(edge_id, at=at):
            raise ValueError("mechanism is outside its validity window; revalidate before promotion")
        limit = float(error_limit)
        if not isfinite(limit) or limit <= 0.0:
            raise ValueError("error_limit must be a positive finite policy value")
        ids = tuple(dict.fromkeys(str(ref).strip() for ref in outcome_ids))
        if any(not ref for ref in ids) or len(ids) < MIN_VERIFIED_OUTCOMES:
            raise ValueError(f"intervention_validated needs >= {MIN_VERIFIED_OUTCOMES} distinct verified outcome ids")
        if learning_candidate is None or str(getattr(learning_candidate, "status", "")) != "accepted":
            raise ValueError("promotion requires a governance-accepted LearningCandidate (no self-promotion)")
        if str(getattr(learning_candidate, "candidate_type", "")) != "causal_calibration":
            raise ValueError("promotion requires a causal_calibration LearningCandidate")
        if edge_id not in tuple(getattr(learning_candidate, "causal_refs", ()) or ()):
            raise ValueError("the LearningCandidate does not reference this mechanism")
        if not set(ids) <= set(getattr(learning_candidate, "outcome_refs", ()) or ()):
            raise ValueError("outcome ids are not covered by the approved LearningCandidate")
        if outcome_ledger is None:
            raise ValueError("OutcomeLedger is required to verify outcomes")
        eligible = tuple(await outcome_ledger.learning_candidates(ids))
        eligible_ids = {str(getattr(record, "outcome_id", "")) for record in eligible}
        rejected = [ref for ref in ids if ref not in eligible_ids]
        if rejected:
            raise ValueError(f"outcomes not verified/learning-eligible in the OutcomeLedger: {rejected}")
        errors: list[float] = []
        for record in eligible:
            if str(record.metric) != current.target_variable:
                raise ValueError(f"outcome {record.outcome_id} measures {record.metric}, not {current.target_variable}")
            if str(record.causal_model_version) != current.scm_version:
                raise ValueError(f"outcome {record.outcome_id} was predicted by another SCM version")
            error = record.outcome_error
            if error is None or not isfinite(float(error)):
                raise ValueError(f"outcome {record.outcome_id} has no finite outcome error")
            errors.append(abs(float(error)))
        mean_abs_error = sum(errors) / len(errors)
        if not mean_abs_error < limit:
            raise ValueError(f"mean |outcome error| {mean_abs_error:.6g} is not below the limit {limit:.6g}")
        promoted = replace(
            current,
            lifecycle=MechanismLifecycle.INTERVENTION_VALIDATED,
            outcome_refs=tuple(dict.fromkeys((*current.outcome_refs, *ids))),
            version=current.version + 1,
        )
        self._history[edge_id].append(promoted)
        self._validations[edge_id] = {
            "mean_abs_error": mean_abs_error,
            "error_limit": limit,
            "outcome_ids": ids,
            "learning_candidate_id": str(getattr(learning_candidate, "candidate_id", "")),
            "source": "OutcomeLedger",
        }
        return promoted

    def record_counterexample(
        self,
        *,
        edge_id: str,
        evidence_ref: str,
        confidence_penalty: float,
    ) -> CausalMechanism:
        """A verified counterexample demotes one rung (never promotes) and blocks further promotion."""

        reference = str(evidence_ref).strip()
        penalty = float(confidence_penalty)
        if not reference or not 0.0 <= penalty <= 1.0:
            raise ValueError("counterexample requires evidence_ref and confidence_penalty in [0, 1]")
        current = self.current(edge_id)
        revised = replace(
            current,
            lifecycle=_PREVIOUS.get(current.lifecycle, MechanismLifecycle.STRUCTURAL_POSSIBLE),
            counterexamples=tuple(dict.fromkeys((*current.counterexamples, reference))),
            causal_confidence=max(0.0, current.causal_confidence - penalty),
            trust_level=max(0.0, current.trust_level - penalty),
            version=current.version + 1,
        )
        self._history[edge_id].append(revised)
        return revised


__all__ = ["ALLOWED_USE", "CausalMechanismRegistry", "MIN_VERIFIED_OUTCOMES", "ORIGIN_CAP", "REGISTRATION_CEILING"]

