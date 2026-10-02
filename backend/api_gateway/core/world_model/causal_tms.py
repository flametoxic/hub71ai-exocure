from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping

from .assertions import require_aware
from .causal_mechanisms import belief_support


class CausalBeliefStatus(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    PREDICTED = "predicted"
    SIMULATED = "simulated"
    MANUAL_VERIFIED = "manual_verified"


class EvidenceCondition(str, Enum):
    """CEOS p.6: an observation can be retracted, become stale or fail integrity."""

    ACTIVE = "active"
    RETRACTED = "retracted"
    STALE = "stale"
    INTEGRITY_FAILED = "integrity_failed"


BELIEF_WEIGHT_NAMES = ("independent_evidence", "source_quality", "freshness", "consistency", "alternatives")
# E_independent: only statuses that are evidence about the world. Predicted / simulated are model
# outputs (CCIA §1.2: "only plan support"): they are reported in their own channel, never counted.
INDEPENDENT_STATUSES = frozenset({CausalBeliefStatus.OBSERVED, CausalBeliefStatus.MANUAL_VERIFIED,
                                  CausalBeliefStatus.INFERRED})
MODEL_OUTPUT_STATUSES = frozenset({CausalBeliefStatus.PREDICTED, CausalBeliefStatus.SIMULATED})


@dataclass(frozen=True)
class CausalBelief:
    belief_id: str
    status: CausalBeliefStatus
    confidence: float
    supporting_refs: tuple[str, ...]
    contradicting_refs: tuple[str, ...]
    active: bool
    # Per-status evidence support, kept apart (CEOS p.6: statuses are never merged into one score).
    support_by_status: Mapping[str, float] = field(default_factory=dict)
    independent_sources: tuple[str, ...] = ()
    source_id: str = ""
    version: int = 1
    condition: EvidenceCondition = EvidenceCondition.ACTIVE
    #: Belief f(E_status, Q, F, C, A) per status channel; ``confidence`` uses independent statuses only.
    belief_by_status: Mapping[str, float] = field(default_factory=dict)
    valid_at: datetime | None = None
    recorded_at: datetime | None = None
    reason: str = ""


@dataclass(frozen=True)
class BeliefVersion:
    """Bitemporal history row: what was believed (valid_at) as recorded by the system at recorded_at."""

    belief: CausalBelief
    valid_at: datetime | None
    recorded_at: datetime
    reason: str


@dataclass(frozen=True)
class _HypothesisInput:
    supporting_refs: tuple[str, ...]
    contradicting_refs: tuple[str, ...]
    source_quality: float
    freshness: float
    consistency: float
    alternatives_penalty: float


class CausalTruthMaintenance:
    """Justification-based TMS for causal beliefs (CEOS p.6).

    Belief(q) = f(E_independent, Q_source, F_freshness, C_consistency, A_alternatives). CEOS gives no
    numeric weights, so both the Belief weights and the per-status reliabilities are required
    configuration -- nothing is invented here.

    * E_independent counts only asserted, currently ACTIVE evidence of independent statuses
      (observed / manual_verified / inferred); a source counts once (strongest item); sources combine
      by noisy-OR, E = 1 - prod(1 - r_status * c). Predicted / simulated evidence has its own channel
      in ``support_by_status`` / ``belief_by_status`` and never enters ``confidence``.
    * Active contradicting evidence X lowers consistency: C_eff = C * (1 - X).
    * Retraction, staleness and integrity failure each recompute every dependent belief.
    * No silent overwrite: re-asserting an id is an error; ``revise_*`` appends a new version.
    * Every change is appended to a bitemporal history; ``replay(as_of)`` reconstructs what was
      believed at any past recording time (late evidence changes belief, not history).
    """

    def __init__(
        self,
        *,
        belief_weights: Mapping[str, float] | None = None,
        status_reliability: Mapping[CausalBeliefStatus, float] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if belief_weights is not None:
            weights = {str(key): float(value) for key, value in belief_weights.items()}
            if set(weights) != set(BELIEF_WEIGHT_NAMES):
                raise ValueError(f"belief_weights must define exactly {BELIEF_WEIGHT_NAMES}")
            if any(not 0.0 <= value <= 1.0 for value in weights.values()):
                raise ValueError("belief weights must be in [0, 1]")
            self._weights: dict[str, float] | None = weights
        else:
            self._weights = None
        if status_reliability is not None:
            reliability = {CausalBeliefStatus(key): float(value) for key, value in status_reliability.items()}
            if set(reliability) != set(CausalBeliefStatus):
                raise ValueError("status_reliability must define every CausalBeliefStatus")
            if any(not 0.0 <= value <= 1.0 for value in reliability.values()):
                raise ValueError("status reliabilities must be in [0, 1]")
            self._reliability: dict[CausalBeliefStatus, float] | None = reliability
        else:
            self._reliability = None
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._beliefs: dict[str, CausalBelief] = {}
        self._hypotheses: dict[str, _HypothesisInput] = {}
        self._history: dict[str, list[BeliefVersion]] = {}
        self._last_recorded: datetime | None = None

    # ------------------------------------------------------------------ evidence
    def assert_evidence(
        self,
        belief_id: str,
        *,
        confidence: float,
        status: CausalBeliefStatus,
        source_id: str | None = None,
        valid_at: datetime | None = None,
        recorded_at: datetime | None = None,
    ) -> CausalBelief:
        token = self._new_token(belief_id, "revise_evidence")
        return self._store_evidence(token, confidence=confidence, status=status, source_id=source_id,
                                    valid_at=valid_at, recorded_at=recorded_at, version=1, reason="asserted")

    def revise_evidence(
        self,
        belief_id: str,
        *,
        expected_version: int,
        confidence: float,
        status: CausalBeliefStatus | None = None,
        valid_at: datetime | None = None,
        recorded_at: datetime | None = None,
        reason: str,
    ) -> CausalBelief:
        """Explicit, versioned revision (optimistic lock): the old version stays in history."""
        current = self._current_evidence(belief_id, expected_version)
        if not str(reason).strip():
            raise ValueError("a revision needs a reason")
        return self._store_evidence(
            current.belief_id, confidence=confidence, status=status or current.status, source_id=current.source_id,
            valid_at=valid_at if valid_at is not None else current.valid_at, recorded_at=recorded_at,
            version=current.version + 1, reason=f"revised:{reason}",
        )

    def retract(self, belief_id: str, *, recorded_at: datetime | None = None, reason: str = "retracted") -> tuple[str, ...]:
        return self._set_condition(belief_id, EvidenceCondition.RETRACTED, recorded_at, reason)

    def mark_stale(self, belief_id: str, *, recorded_at: datetime | None = None, reason: str = "stale") -> tuple[str, ...]:
        return self._set_condition(belief_id, EvidenceCondition.STALE, recorded_at, reason)

    def fail_integrity(self, belief_id: str, *, reason: str, recorded_at: datetime | None = None) -> tuple[str, ...]:
        if not str(reason).strip():
            raise ValueError("integrity failure needs a reason")
        return self._set_condition(belief_id, EvidenceCondition.INTEGRITY_FAILED, recorded_at, reason)

    # ------------------------------------------------------------------ hypotheses
    def assert_hypothesis(
        self,
        belief_id: str,
        *,
        supporting_refs: tuple[str, ...],
        contradicting_refs: tuple[str, ...],
        source_quality: float,
        freshness: float,
        consistency: float,
        alternatives_penalty: float,
        recorded_at: datetime | None = None,
    ) -> CausalBelief:
        token = self._new_token(belief_id, "revise_hypothesis")
        inputs = self._hypothesis_input(token, supporting_refs, contradicting_refs, source_quality, freshness,
                                        consistency, alternatives_penalty)
        self._hypotheses[token] = inputs
        moment = self._record_time(recorded_at)
        return self._recompute(token, moment, reason="asserted")

    def revise_hypothesis(
        self,
        belief_id: str,
        *,
        expected_version: int,
        reason: str,
        recorded_at: datetime | None = None,
        **changes,
    ) -> CausalBelief:
        token = str(belief_id).strip()
        if token not in self._hypotheses:
            raise KeyError(f"unknown causal hypothesis: {belief_id}")
        if self._beliefs[token].version != int(expected_version):
            raise ValueError("stale revision: the hypothesis changed since the expected version")
        if not str(reason).strip():
            raise ValueError("a revision needs a reason")
        old = self._hypotheses[token]
        allowed = {"supporting_refs", "contradicting_refs", "source_quality", "freshness", "consistency",
                   "alternatives_penalty"}
        if set(changes) - allowed:
            raise ValueError(f"unknown hypothesis fields {sorted(set(changes) - allowed)}")
        merged = {name: changes.get(name, getattr(old, name)) for name in allowed}
        self._hypotheses[token] = self._hypothesis_input(token, **merged)
        moment = self._record_time(recorded_at)
        belief = self._recompute(token, moment, reason=f"revised:{reason}")
        self._recompute_dependents({token}, moment, reason=f"dependency_revised:{token}")
        return belief

    # ------------------------------------------------------------------ read side
    def get(self, belief_id: str) -> CausalBelief:
        try:
            return self._beliefs[str(belief_id)]
        except KeyError as exc:
            raise KeyError(f"unknown causal belief: {belief_id}") from exc

    def history(self, belief_id: str) -> tuple[BeliefVersion, ...]:
        return tuple(self._history.get(str(belief_id), ()))

    def replay(self, as_of: datetime, *, valid_as_of: datetime | None = None) -> Mapping[str, CausalBelief]:
        """Beliefs as recorded up to ``as_of`` (transaction time); optionally only versions whose valid
        time is <= ``valid_as_of``. History is never rewritten by later or late-arriving evidence."""
        moment = require_aware(as_of, "as_of")
        valid_limit = require_aware(valid_as_of, "valid_as_of") if valid_as_of is not None else None
        snapshot: dict[str, CausalBelief] = {}
        for belief_id, rows in self._history.items():
            chosen = None
            for row in rows:
                if row.recorded_at > moment:
                    break
                if valid_limit is not None and row.valid_at is not None and row.valid_at > valid_limit:
                    continue
                chosen = row
            if chosen is not None:
                snapshot[belief_id] = chosen.belief
        return MappingProxyType(snapshot)

    def justification(self, belief_id: str) -> Mapping[str, tuple[str, ...]]:
        """CEOS p.6: hypothesis <- supported by O1, T1, M1 <- contradicted by O7."""
        token = str(belief_id)
        if token not in self._hypotheses:
            raise KeyError(f"unknown causal hypothesis: {belief_id}")
        inputs = self._hypotheses[token]
        return MappingProxyType({"supported_by": inputs.supporting_refs, "contradicted_by": inputs.contradicting_refs})

    @staticmethod
    def belief_update(
        *,
        independent_evidence: float,
        source_quality: float,
        freshness: float,
        consistency: float,
        alternatives_penalty: float,
        weights: Mapping[str, float],
    ) -> float:
        """Belief(q) with explicit weights (form of f: causal_mechanisms.belief_support)."""

        return float(
            belief_support(
                independent_evidence=independent_evidence,
                source_quality=source_quality,
                freshness=freshness,
                consistency=consistency,
                alternatives=alternatives_penalty,
                weights=weights,
            ).value
        )

    # ------------------------------------------------------------------ internals
    def _new_token(self, belief_id: str, revise_name: str) -> str:
        token = str(belief_id).strip()
        if not token:
            raise ValueError("belief_id is required")
        if token in self._beliefs:
            raise ValueError(f"belief {token!r} already exists: no silent overwrite, use {revise_name}()")
        return token

    def _current_evidence(self, belief_id: str, expected_version: int) -> CausalBelief:
        token = str(belief_id).strip()
        current = self.get(token)
        if token in self._hypotheses:
            raise ValueError(f"{token!r} is a hypothesis, not evidence")
        if current.version != int(expected_version):
            raise ValueError("stale revision: the evidence changed since the expected version")
        return current

    def _record_time(self, recorded_at: datetime | None) -> datetime:
        moment = require_aware(recorded_at, "recorded_at") if recorded_at is not None else require_aware(
            self._clock(), "clock")
        if self._last_recorded is not None and moment < self._last_recorded:
            raise ValueError("recorded_at precedes an already recorded change: history is append-only")
        self._last_recorded = moment
        return moment

    def _append(self, belief: CausalBelief, moment: datetime, reason: str) -> CausalBelief:
        belief = replace(belief, recorded_at=moment, reason=reason)
        self._beliefs[belief.belief_id] = belief
        self._history.setdefault(belief.belief_id, []).append(BeliefVersion(belief, belief.valid_at, moment, reason))
        return belief

    def _store_evidence(self, token, *, confidence, status, source_id, valid_at, recorded_at, version, reason):
        value = self._probability("confidence", confidence)
        status = CausalBeliefStatus(status)
        moment = self._record_time(recorded_at)
        valid = require_aware(valid_at, "valid_at") if valid_at is not None else moment
        belief = CausalBelief(
            token, status, value, (), (), True,
            support_by_status={status.value: value},
            independent_sources=(str(source_id or token),),
            source_id=str(source_id or token),
            version=version,
            valid_at=valid,
        )
        belief = self._append(belief, moment, reason)
        self._recompute_dependents({token}, moment, reason=f"evidence_{reason}:{token}")
        return belief

    def _set_condition(self, belief_id: str, condition: EvidenceCondition, recorded_at, reason: str) -> tuple[str, ...]:
        token = str(belief_id).strip()
        current = self.get(token)
        if token in self._hypotheses:
            raise ValueError(f"{token!r} is a hypothesis; change its evidence instead")
        if current.condition is not EvidenceCondition.ACTIVE:
            raise ValueError(f"evidence {token!r} is already {current.condition.value}")
        moment = self._record_time(recorded_at)
        self._append(replace(current, active=False, condition=condition, version=current.version + 1), moment,
                     f"{condition.value}:{reason}")
        return self._recompute_dependents({token}, moment, reason=f"dependency_{condition.value}:{token}")

    def _hypothesis_input(self, token, supporting_refs, contradicting_refs, source_quality, freshness, consistency,
                          alternatives_penalty) -> _HypothesisInput:
        supporting = tuple(dict.fromkeys(str(ref).strip() for ref in supporting_refs))
        contradicting = tuple(dict.fromkeys(str(ref).strip() for ref in contradicting_refs))
        if not supporting or any(not ref for ref in (*supporting, *contradicting)):
            raise ValueError("hypothesis requires identity and supporting evidence")
        if token in supporting or token in contradicting:
            raise ValueError("hypothesis cannot justify itself")
        self._require_configuration()
        return _HypothesisInput(
            supporting_refs=supporting,
            contradicting_refs=contradicting,
            source_quality=self._probability("source_quality", source_quality),
            freshness=self._probability("freshness", freshness),
            consistency=self._probability("consistency", consistency),
            alternatives_penalty=self._probability("alternatives_penalty", alternatives_penalty),
        )

    def _recompute_dependents(self, changed_ids: set[str], moment: datetime, *, reason: str) -> tuple[str, ...]:
        changed = set(changed_ids)
        dependents: list[str] = []
        pending = True
        while pending:
            pending = False
            for hypothesis_id, inputs in self._hypotheses.items():
                if hypothesis_id in changed:
                    continue
                if changed.intersection(inputs.supporting_refs) or changed.intersection(inputs.contradicting_refs):
                    changed.add(hypothesis_id)
                    dependents.append(hypothesis_id)
                    pending = True
        pending_set, ordered, visiting = set(dependents), [], set()

        def visit(hypothesis_id: str) -> None:  # parents before children (a hypothesis may support another)
            if hypothesis_id in ordered or hypothesis_id in visiting:
                return
            visiting.add(hypothesis_id)
            inputs = self._hypotheses[hypothesis_id]
            for ref in (*inputs.supporting_refs, *inputs.contradicting_refs):
                if ref in pending_set:
                    visit(ref)
            ordered.append(hypothesis_id)

        for hypothesis_id in dependents:
            visit(hypothesis_id)
        for hypothesis_id in ordered:
            self._recompute(hypothesis_id, moment, reason=reason)
        return tuple(dependents)

    def _require_configuration(self) -> None:
        if self._weights is None or self._reliability is None:
            raise ValueError(
                "belief_weights and status_reliability must be configured explicitly; CEOS defines no numeric weights"
            )

    def _channels(self, refs: tuple[str, ...]) -> tuple[float, dict[str, float], tuple[str, ...]]:
        """Noisy-OR per status channel and over independent statuses; a source counts once per channel."""

        assert self._reliability is not None
        independent: dict[str, float] = {}
        by_status: dict[str, dict[str, float]] = {}
        for ref in refs:
            belief = self._beliefs.get(ref)
            if belief is None or not belief.active:
                continue
            strength = self._reliability[belief.status] * belief.confidence
            source = belief.source_id or ref
            channel = by_status.setdefault(belief.status.value, {})
            channel[source] = max(channel.get(source, 0.0), strength)
            if belief.status in INDEPENDENT_STATUSES:
                independent[source] = max(independent.get(source, 0.0), strength)

        def noisy_or(values) -> float:
            remaining = 1.0
            for value in values:
                remaining *= 1.0 - value
            return 1.0 - remaining

        return (noisy_or(independent.values()),
                {status: noisy_or(sources.values()) for status, sources in by_status.items()},
                tuple(sorted(independent)))

    def _recompute(self, token: str, moment: datetime, *, reason: str) -> CausalBelief:
        assert self._weights is not None
        inputs = self._hypotheses[token]
        support, by_status, sources = self._channels(inputs.supporting_refs)
        contradiction, _, _ = self._channels(inputs.contradicting_refs)
        consistency = inputs.consistency * (1.0 - contradiction)

        def belief(evidence: float) -> float:
            if evidence <= 0.0:
                return 0.0  # no asserted, active evidence -> no belief, whatever the other factors say
            return self.belief_update(independent_evidence=evidence, source_quality=inputs.source_quality,
                                      freshness=inputs.freshness, consistency=consistency,
                                      alternatives_penalty=inputs.alternatives_penalty, weights=self._weights)

        previous = self._beliefs.get(token)
        valid_times = [self._beliefs[r].valid_at for r in inputs.supporting_refs
                       if r in self._beliefs and self._beliefs[r].valid_at is not None]
        result = CausalBelief(
            token,
            CausalBeliefStatus.INFERRED,
            belief(support),
            inputs.supporting_refs,
            inputs.contradicting_refs,
            support > 0.0,
            support_by_status=by_status,
            independent_sources=sources,
            source_id=token,
            version=1 if previous is None else previous.version + 1,
            belief_by_status={status: belief(value) for status, value in by_status.items()},
            valid_at=max(valid_times) if valid_times else moment,
        )
        return self._append(result, moment, reason)

    @staticmethod
    def _probability(name: str, value: float) -> float:
        parsed = float(value)
        if not 0.0 <= parsed <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
        return parsed



# совместимость: прежний модуль реэкспортировал эти имена
from api_gateway.core.reasoning.truth_maintenance import Justification, TruthMaintenanceSystem  # noqa: E402,F401

__all__ = [
    "BELIEF_WEIGHT_NAMES",
    "Justification",
    "TruthMaintenanceSystem",
    "BeliefVersion",
    "CausalBelief",
    "CausalBeliefStatus",
    "CausalTruthMaintenance",
    "EvidenceCondition",
    "INDEPENDENT_STATUSES",
    "MODEL_OUTPUT_STATUSES",
]
