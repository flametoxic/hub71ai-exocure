from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from math import isfinite
from typing import Any, Iterable, Iterator, Mapping, Optional, Protocol


def _aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


class FrozenMap(Mapping):
    """Immutable, hashable mapping; values are frozen recursively (CCIA §1.1: a claim is a value).

    ``content``/``metadata`` of an :class:`EpistemicClaim` are stored as ``FrozenMap`` so that
    ``claim.content["x"] = ...`` or mutation of a nested list cannot change a sealed claim.
    ``thaw`` returns plain ``dict``/``list`` copies for serialization.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Any = ()) -> None:
        object.__setattr__(self, "_data", {key: deep_freeze(value) for key, value in dict(data).items()})

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover - defensive
        raise AttributeError("FrozenMap is immutable")

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return NotImplemented

    def __hash__(self) -> int:
        return hash(frozenset(self._data.items()))

    def __repr__(self) -> str:
        return f"FrozenMap({self._data!r})"

    def __reduce__(self):
        return (FrozenMap, (thaw(self),))


def deep_freeze(value: Any) -> Any:
    """Recursively convert mutable containers to immutable ones (dict→FrozenMap, list→tuple, set→frozenset)."""
    if isinstance(value, FrozenMap):
        return value
    if isinstance(value, Mapping):
        return FrozenMap(value)
    if type(value) in (list, tuple):
        return tuple(deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(deep_freeze(item) for item in value)
    if isinstance(value, bytearray):
        return bytes(value)
    return value


def thaw(value: Any) -> Any:
    """Plain, JSON-friendly copy of a frozen value (FrozenMap→dict, tuple→list, frozenset→sorted list)."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [thaw(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((thaw(item) for item in value), key=repr)
    return value


class ClaimType(str, Enum):
    OBSERVATION = "observation"
    FACT = "fact"
    HYPOTHESIS = "hypothesis"
    MECHANISM = "mechanism"
    PREDICTION = "prediction"
    SIMULATION = "simulation"
    PLAN = "plan"


class EpistemicStatus(str, Enum):
    OBSERVED = "observed"
    MANUAL_VERIFIED = "manual_verified"
    INFERRED = "inferred"
    PREDICTED = "predicted"
    SIMULATED = "simulated"
    HYPOTHESIS = "hypothesis"
    CONFLICTED = "conflicted"
    RETRACTED = "retracted"


class PlanningUse(str, Enum):
    """CCIA §1.2 table (p.2), column «Может влиять на plan?»."""

    #: observed — «Да, в пределах quality/freshness»
    DIRECT = "direct_with_quality_and_freshness"
    #: manual_verified — «Да, с scope/authority»
    DIRECT_SCOPED = "direct_with_scope_and_authority"
    #: inferred — «Да, но с uncertainty» (the grant carries the claim's uncertainty)
    WITH_UNCERTAINTY = "with_uncertainty"
    #: conflicted — «Authority reduced/blocked»; also the downgrade for an observation that fails
    #: quality/freshness or a verification outside the verifier's scope/authority
    REVIEW = "review"
    #: predicted / simulated — «Только plan support»
    SUPPORT = "plan_support"
    #: hypothesis — «Investigation/review»
    INVESTIGATION = "investigation"
    #: retracted — «Не использовать»
    BLOCKED = "blocked"


#: Weakest → strongest planning authority (used to take the weakest use of a set of claims).
PLANNING_USE_ORDER: tuple[PlanningUse, ...] = (
    PlanningUse.BLOCKED,
    PlanningUse.INVESTIGATION,
    PlanningUse.SUPPORT,
    PlanningUse.REVIEW,
    PlanningUse.WITH_UNCERTAINTY,
    PlanningUse.DIRECT_SCOPED,
    PlanningUse.DIRECT,
)


def weakest_planning_use(uses: "Iterable[PlanningUse]") -> PlanningUse:
    """Weakest use of a collection; an empty collection carries no authority (BLOCKED)."""
    items = [PlanningUse(item) for item in uses]
    if not items:
        return PlanningUse.BLOCKED
    return min(items, key=PLANNING_USE_ORDER.index)


# CCIA §1.2 (p.2): statuses must not be mixed.
_DEFAULT_PLANNING_USE = {
    EpistemicStatus.OBSERVED: PlanningUse.DIRECT,
    EpistemicStatus.MANUAL_VERIFIED: PlanningUse.DIRECT_SCOPED,
    EpistemicStatus.INFERRED: PlanningUse.WITH_UNCERTAINTY,
    EpistemicStatus.PREDICTED: PlanningUse.SUPPORT,
    EpistemicStatus.SIMULATED: PlanningUse.SUPPORT,
    EpistemicStatus.HYPOTHESIS: PlanningUse.INVESTIGATION,
    EpistemicStatus.CONFLICTED: PlanningUse.REVIEW,
    EpistemicStatus.RETRACTED: PlanningUse.BLOCKED,
}

#: Statuses that denote model output (future rollout / scenario output). They can never
#: become an observation or a human-verified fact by any path (CCIA §1.2, §8.7).
MODEL_OUTPUT_STATUSES = frozenset({EpistemicStatus.PREDICTED, EpistemicStatus.SIMULATED})
#: Statuses that carry direct planning authority.
OBSERVATIONAL_STATUSES = frozenset({EpistemicStatus.OBSERVED, EpistemicStatus.MANUAL_VERIFIED})
#: Claim types whose content is model output by construction.
MODEL_OUTPUT_CLAIM_TYPES = frozenset({ClaimType.PREDICTION, ClaimType.SIMULATION, ClaimType.PLAN})

_COMMON = frozenset({EpistemicStatus.CONFLICTED, EpistemicStatus.RETRACTED})

#: Type/status consistency: which epistemic statuses a claim type may ever carry.
ALLOWED_STATUSES_BY_TYPE: Mapping[ClaimType, frozenset[EpistemicStatus]] = {
    ClaimType.OBSERVATION: _COMMON | {EpistemicStatus.OBSERVED, EpistemicStatus.MANUAL_VERIFIED},
    ClaimType.FACT: _COMMON | {EpistemicStatus.OBSERVED, EpistemicStatus.MANUAL_VERIFIED, EpistemicStatus.INFERRED},
    ClaimType.HYPOTHESIS: _COMMON | {EpistemicStatus.HYPOTHESIS},
    ClaimType.MECHANISM: _COMMON | {
        EpistemicStatus.INFERRED,
        EpistemicStatus.HYPOTHESIS,
        EpistemicStatus.MANUAL_VERIFIED,
    },
    ClaimType.PREDICTION: _COMMON | {EpistemicStatus.PREDICTED},
    ClaimType.SIMULATION: _COMMON | {EpistemicStatus.SIMULATED},
    ClaimType.PLAN: _COMMON | {EpistemicStatus.PREDICTED, EpistemicStatus.SIMULATED, EpistemicStatus.HYPOTHESIS},
}


class EpistemicTransitionError(ValueError):
    """Raised for any forbidden epistemic status change (laundering, revival, etc.)."""


def check_status_history(claim_type: "ClaimType", history: tuple["EpistemicStatus", ...]) -> None:
    """Every step of a status lineage must be an allowed transition (CCIA §1.2, §3.2).

    Allowed steps (exactly what :meth:`EpistemicClaim.transition_to` / ``mark_conflicted`` /
    ``resolve_conflict`` can produce):

    * ``X -> retracted`` for any non-retracted X (retracted is terminal);
    * ``X -> conflicted`` (entered only with both evidence paths);
    * ``X -> manual_verified`` when neither the claim type nor any earlier status is model output;
    * ``conflicted -> Y`` where Y is the status held right before the conflict.

    Anything else (``inferred -> observed``, ``simulated -> inferred``, ``hypothesis -> inferred``,
    repeated statuses, revival after retraction) is rejected, independently of the lineage seal.
    """
    model_output_type = claim_type in MODEL_OUTPUT_CLAIM_TYPES
    for index in range(1, len(history)):
        previous, current = history[index - 1], history[index]
        if previous is EpistemicStatus.RETRACTED:
            raise ValueError("retracted is terminal: a retracted claim cannot be revived")
        if previous is current:
            raise ValueError(f"status lineage repeats {current.value}: not a transition")
        if current in (EpistemicStatus.RETRACTED, EpistemicStatus.CONFLICTED):
            continue
        if current is EpistemicStatus.MANUAL_VERIFIED:
            if model_output_type or any(item in MODEL_OUTPUT_STATUSES for item in history[:index]):
                raise ValueError("status laundering from model output to observation is forbidden")
            continue
        if previous is EpistemicStatus.CONFLICTED and index >= 2 and current is history[index - 2]:
            continue
        if current in OBSERVATIONAL_STATUSES and any(item in MODEL_OUTPUT_STATUSES for item in history[:index]):
            raise ValueError("status laundering from model output to observation is forbidden")
        raise ValueError(f"illegal status transition in lineage: {previous.value} -> {current.value}")


# Process-local key for the claim seal. The seal travels with the claim and covers EVERY field of the
# claim (identity, type, status lineage, verifier, version, content, scopes, valid/knowledge time,
# provenance/evidence, uncertainty, world snapshot, metadata ...). Any ``dataclasses.replace`` of a sealed
# claim keeps the old seal, which no longer matches, so the rewrite is rejected; ``object.__setattr__`` /
# pickle / copy forgeries bypass ``__post_init__`` but are caught by :meth:`EpistemicClaim.verify_integrity`,
# which every :class:`ClaimLedger` runs. Only transition_to / mark_conflicted / resolve_conflict / revise
# re-seal, and each of them raises the version (CCIA §1.1 version, §1.2 statuses).
_SEAL_KEY = os.urandom(32)
_RESEAL = "\x00reseal"


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat() if value.tzinfo is not None else value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=repr)
    if isinstance(value, float):
        return repr(value)
    return value


def _claim_payload(claim: "EpistemicClaim") -> bytes:
    payload = {
        name: _canonical(getattr(claim, name))
        for name in claim.__dataclass_fields__ if name != "lineage_seal"
    }
    payload["confidence"] = repr(float(claim.confidence))
    payload["uncertainty"] = repr(float(claim.uncertainty))
    return json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")


def _claim_seal(claim: "EpistemicClaim") -> str:
    return hmac.new(_SEAL_KEY, _claim_payload(claim), hashlib.sha256).hexdigest()


def claim_content_digest(claim: "EpistemicClaim") -> str:
    """Key-independent SHA-256 of every sealed field: identifies a claim version's content across
    processes and restarts (the HMAC seal is process-local)."""
    return hashlib.sha256(_claim_payload(claim)).hexdigest()


#: provenance of model output (simulation, prediction, LLM/MoE, synthetic, counterfactual, plan): a claim
#: carrying such provenance is never an observation, whatever status it declares (CCIA §1.2).
MODEL_PROVENANCE_PREFIXES = ("llm:", "moe:", "sim:", "simulation:", "simulator:", "synthetic:",
                             "counterfactual:", "prediction:", "predicted:", "forecast:", "plan:",
                             "ifm:")      # нейронная Infrastructure Foundation Model: только совет, не наблюдение


@dataclass(frozen=True)
class EpistemicClaim:
    """CCIA §1.1 (p.1-2): Claim = {content, scope, time, source, evidence, assumptions, uncertainty,
    status, version}. ``content``/``metadata`` are deep-frozen; ``version`` grows with every change
    made through the claim's own methods (transition, conflict, resolution, revision)."""

    claim_id: str
    claim_type: ClaimType
    content: Mapping[str, Any]
    epistemic_status: EpistemicStatus
    subject_scope: tuple[str, ...]
    boundary_scope: tuple[str, ...]
    valid_time: datetime
    knowledge_time: datetime
    provenance_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    assumptions: tuple[str, ...]
    counterexamples: tuple[str, ...]
    confidence: float
    uncertainty: float
    causal_mechanism_refs: tuple[str, ...]
    world_snapshot_id: str
    world_version: str
    policy_scope: tuple[str, ...]
    privacy_scope: tuple[str, ...]
    supersedes: tuple[str, ...]
    retracts: tuple[str, ...]
    trace_id: str
    supporting_evidence: tuple[str, ...] = ()
    contradicting_evidence: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    #: Status lineage of this claim, oldest first, ending with the current status. Carried through
    #: every transition so that a conflicted/retracted claim remembers what it was and cannot be
    #: laundered; a status changed with ``dataclasses.replace`` no longer matches it and is rejected.
    status_history: tuple[EpistemicStatus, ...] = ()
    #: Authorized human that confirmed the claim (required for MANUAL_VERIFIED).
    verified_by: str = ""
    #: CCIA §1.1: version of this claim (1 for a new claim, +1 for every change of status/content).
    version: int = 1
    #: HMAC integrity seal over every other field of the claim; set automatically.
    lineage_seal: str = field(default="", compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.claim_id.strip() or not self.subject_scope or not self.boundary_scope:
            raise ValueError("claim_id, subject_scope, and boundary_scope are required")
        if not self.world_snapshot_id.strip() or not self.world_version.strip() or not self.trace_id.strip():
            raise ValueError("world snapshot/version and trace_id are required")
        confidence = float(self.confidence)
        uncertainty = float(self.uncertainty)
        if not (isfinite(confidence) and isfinite(uncertainty)):
            raise ValueError("confidence and uncertainty must be finite")
        if not 0.0 <= confidence <= 1.0 or not 0.0 <= uncertainty <= 1.0:
            raise ValueError("confidence and uncertainty must be in [0, 1]")
        claim_type = ClaimType(self.claim_type)
        status = EpistemicStatus(self.epistemic_status)
        history = tuple(EpistemicStatus(item) for item in self.status_history) or (status,)
        object.__setattr__(self, "claim_type", claim_type)
        object.__setattr__(self, "epistemic_status", status)
        object.__setattr__(self, "status_history", history)
        if not isinstance(self.content, Mapping) or not isinstance(self.metadata, Mapping):
            raise ValueError("content and metadata must be mappings (never a bare string)")
        object.__setattr__(self, "content", deep_freeze(self.content))
        object.__setattr__(self, "metadata", deep_freeze(self.metadata))
        for name in ("subject_scope", "boundary_scope", "provenance_refs", "evidence_refs", "assumptions",
                     "counterexamples", "causal_mechanism_refs", "policy_scope", "privacy_scope", "supersedes",
                     "retracts", "supporting_evidence", "contradicting_evidence"):
            object.__setattr__(self, name, tuple(str(item) for item in getattr(self, name)))
        allowed = ALLOWED_STATUSES_BY_TYPE[claim_type]
        for item in (status, *history):
            if item not in allowed:
                raise ValueError(
                    f"claim_type={claim_type.value} cannot carry epistemic_status={item.value}"
                )
        check_status_history(claim_type, history)
        if history[-1] is not status:
            raise ValueError(
                "epistemic status changes only through transition_to/mark_conflicted "
                f"(lineage ends with {history[-1].value}, got {status.value})"
            )
        prior = history[:-1]
        if status in OBSERVATIONAL_STATUSES and any(item in MODEL_OUTPUT_STATUSES for item in prior):
            raise ValueError("status laundering from model output to observation is forbidden")
        if status is EpistemicStatus.MANUAL_VERIFIED and not str(self.verified_by or "").strip():
            raise ValueError("manual_verified requires verified_by (authorized human)")
        if status is EpistemicStatus.CONFLICTED and (not self.supporting_evidence or not self.contradicting_evidence):
            raise ValueError("conflicted claim must preserve both support and contradiction paths")
        object.__setattr__(self, "valid_time", _aware(self.valid_time, "valid_time"))
        object.__setattr__(self, "knowledge_time", _aware(self.knowledge_time, "knowledge_time"))
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise ValueError("version must be an integer")
        if self.version < len(history):
            raise ValueError("claim version cannot be lower than the number of status transitions it went through")
        seal = _claim_seal(self)
        provided = str(self.lineage_seal or "")
        if provided and provided != _RESEAL and not hmac.compare_digest(provided, seal):
            raise EpistemicTransitionError(
                "status lineage, version, content, time, scope or metadata altered outside the claim's own "
                "methods (transition_to/mark_conflicted/resolve_conflict/revise); laundering is forbidden"
            )
        object.__setattr__(self, "lineage_seal", seal)

    def verify_integrity(self) -> "EpistemicClaim":
        """Re-check a claim that may have bypassed ``__post_init__`` (``object.__setattr__``, pickle,
        ``copy``): the seal must match every current field and the status lineage must be legal.
        Raises :class:`EpistemicTransitionError`; returns the claim when intact (CCIA §1.1, §1.2, §8.7)."""
        try:
            claim_type = ClaimType(self.claim_type)
            status = EpistemicStatus(self.epistemic_status)
            history = tuple(EpistemicStatus(item) for item in self.status_history)
            if not history or history[-1] is not status:
                raise ValueError("status lineage does not end with the current status")
            allowed = ALLOWED_STATUSES_BY_TYPE[claim_type]
            if any(item not in allowed for item in history):
                raise ValueError(f"claim_type={claim_type.value} cannot carry its status lineage")
            check_status_history(claim_type, history)
            if status in OBSERVATIONAL_STATUSES and any(item in MODEL_OUTPUT_STATUSES for item in history[:-1]):
                raise ValueError("status laundering from model output to observation is forbidden")
            if status is EpistemicStatus.MANUAL_VERIFIED and not str(self.verified_by or "").strip():
                raise ValueError("manual_verified requires verified_by")
            if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < len(history):
                raise ValueError("claim version is inconsistent with its status lineage")
        except (ValueError, KeyError, TypeError) as exc:
            raise EpistemicTransitionError(f"claim {self.claim_id}: integrity check failed: {exc}") from exc
        provided = str(self.lineage_seal or "")
        if not provided or not hmac.compare_digest(provided, _claim_seal(self)):
            raise EpistemicTransitionError(
                f"claim {self.claim_id}: seal does not match its fields (altered outside the claim's own methods)"
            )
        return self

    @property
    def planning_use(self) -> PlanningUse:
        """Status-level use from the CCIA §1.2 table. The *checked* use (quality/freshness for
        observed, scope/authority for manual_verified) comes from :func:`grant_planning_use`."""
        return _DEFAULT_PLANNING_USE[self.epistemic_status]

    @property
    def is_model_output(self) -> bool:
        return is_model_output(self)

    def _with_status(self, target: EpistemicStatus, **changes: Any) -> "EpistemicClaim":
        return replace(
            self,
            epistemic_status=target,
            status_history=(*self.status_history, target),
            version=self.version + 1,
            lineage_seal=_RESEAL,
            **changes,
        )

    def revise(self, **changes: Any) -> "EpistemicClaim":
        """New version of the same claim with changed content/evidence/uncertainty (status unchanged).

        Status, lineage, identity and verifier cannot be changed here — use transition_to /
        mark_conflicted / resolve_conflict.
        """
        forbidden = {"claim_id", "claim_type", "epistemic_status", "status_history", "verified_by", "version",
                     "lineage_seal"} & set(changes)
        if forbidden:
            raise EpistemicTransitionError(f"revise cannot change {sorted(forbidden)}")
        if self.epistemic_status is EpistemicStatus.RETRACTED:
            raise EpistemicTransitionError("retracted claim cannot be revised")
        return replace(self, version=self.version + 1, lineage_seal=_RESEAL, **changes)

    def transition_to(
        self,
        target: EpistemicStatus,
        *,
        verified_by: str = "",
    ) -> "EpistemicClaim":
        """Explicit status transitions (CCIA §1.2, §3.2).

        * retracted is terminal;
        * model output (prediction/simulation/plan) can never become observed/manual_verified,
          directly or through conflicted;
        * conflicted is entered only via :meth:`mark_conflicted` (both evidence paths required);
        * manual_verified requires ``verified_by`` and a claim type that admits it;
        * a conflicted claim may be resolved back to the status it had before the conflict.
        """
        target = EpistemicStatus(target)
        if target is self.epistemic_status:
            return self
        if self.epistemic_status is EpistemicStatus.RETRACTED:
            raise EpistemicTransitionError("retracted claim cannot transition")
        if target in OBSERVATIONAL_STATUSES and is_model_output(self):
            raise EpistemicTransitionError("status laundering from model output to observation is forbidden")
        if target is EpistemicStatus.RETRACTED:
            return self._with_status(target)
        if target is EpistemicStatus.CONFLICTED:
            raise EpistemicTransitionError("use mark_conflicted: conflict must preserve both evidence paths")
        if target not in ALLOWED_STATUSES_BY_TYPE[self.claim_type]:
            raise EpistemicTransitionError(
                f"claim_type={self.claim_type.value} cannot carry epistemic_status={target.value}"
            )
        if target is EpistemicStatus.MANUAL_VERIFIED:
            if not str(verified_by or "").strip():
                raise EpistemicTransitionError("manual_verified requires verified_by (authorized human)")
            return self._with_status(target, verified_by=str(verified_by).strip())
        if (
            self.epistemic_status is EpistemicStatus.CONFLICTED
            and len(self.status_history) >= 2
            and target is self.status_history[-2]
        ):
            return self._with_status(target)
        raise EpistemicTransitionError(
            f"unsupported epistemic transition: {self.epistemic_status.value} -> {target.value}"
        )

    def mark_conflicted(
        self,
        *,
        supporting_evidence: tuple[str, ...],
        contradicting_evidence: tuple[str, ...],
    ) -> "EpistemicClaim":
        if self.epistemic_status is EpistemicStatus.RETRACTED:
            raise EpistemicTransitionError("retracted claim cannot transition")
        if not supporting_evidence or not contradicting_evidence:
            raise ValueError("conflict requires both support and contradiction paths")
        if self.epistemic_status is EpistemicStatus.CONFLICTED:
            return replace(
                self,
                supporting_evidence=tuple(dict.fromkeys((*self.supporting_evidence, *supporting_evidence))),
                contradicting_evidence=tuple(
                    dict.fromkeys((*self.contradicting_evidence, *contradicting_evidence))
                ),
                version=self.version + 1,
                lineage_seal=_RESEAL,
            )
        return self._with_status(
            EpistemicStatus.CONFLICTED,
            supporting_evidence=tuple(supporting_evidence),
            contradicting_evidence=tuple(contradicting_evidence),
        )

    def resolve_conflict(
        self,
        target: EpistemicStatus,
        *,
        resolution_evidence: tuple[str, ...],
        resolved_by: str,
    ) -> "EpistemicClaim":
        """CCIA §3.2 «request resolution»: the only way out of ``conflicted``.

        ``target`` is the status held before the conflict (the conflict is explained away),
        ``retracted`` (the claim lost) or ``manual_verified`` (an authorized human decided; never for
        model output). Resolution evidence and the resolver are mandatory and are kept in the
        claim (``evidence_refs`` + ``metadata['conflict_resolution']``); both evidence paths stay.
        """
        if self.epistemic_status is not EpistemicStatus.CONFLICTED:
            raise EpistemicTransitionError("only a conflicted claim can be resolved")
        evidence = tuple(str(item) for item in resolution_evidence if str(item).strip())
        resolver = str(resolved_by or "").strip()
        if not evidence or not resolver:
            raise EpistemicTransitionError("conflict resolution requires resolution evidence and resolved_by")
        target = EpistemicStatus(target)
        previous = self.status_history[-2] if len(self.status_history) >= 2 else None
        if target not in {EpistemicStatus.RETRACTED, EpistemicStatus.MANUAL_VERIFIED} and target is not previous:
            raise EpistemicTransitionError(
                "a conflict resolves only to the pre-conflict status, retracted or manual_verified"
            )
        if target is EpistemicStatus.MANUAL_VERIFIED and is_model_output(self):
            raise EpistemicTransitionError("status laundering from model output to observation is forbidden")
        if target not in ALLOWED_STATUSES_BY_TYPE[self.claim_type]:
            raise EpistemicTransitionError(
                f"claim_type={self.claim_type.value} cannot carry epistemic_status={target.value}"
            )
        metadata = dict(thaw(self.metadata))
        metadata["conflict_resolution"] = {
            "resolved_to": target.value,
            "resolved_by": resolver,
            "resolution_evidence": list(evidence),
            "supporting_evidence": list(self.supporting_evidence),
            "contradicting_evidence": list(self.contradicting_evidence),
        }
        changes: dict[str, Any] = {
            "evidence_refs": tuple(dict.fromkeys((*self.evidence_refs, *evidence))),
            "metadata": metadata,
        }
        if target is EpistemicStatus.MANUAL_VERIFIED:
            changes["verified_by"] = resolver
        return self._with_status(target, **changes)


# --------------------------------------------------------------------------------------
# CCIA §1.2: one status vocabulary for all organs
# --------------------------------------------------------------------------------------

_MEMORY_STATUS_MAP: Mapping[str, EpistemicStatus] = {
    "observed": EpistemicStatus.OBSERVED,
    "inferred": EpistemicStatus.INFERRED,
    "predicted": EpistemicStatus.PREDICTED,
    "simulated": EpistemicStatus.SIMULATED,
    "manual_verified": EpistemicStatus.MANUAL_VERIFIED,
    # EBM: a rejected candidate and a retracted record are both «Не использовать».
    "rejected": EpistemicStatus.RETRACTED,
    "retracted": EpistemicStatus.RETRACTED,
}
_CAUSAL_BELIEF_STATUS_MAP: Mapping[str, EpistemicStatus] = {
    "observed": EpistemicStatus.OBSERVED,
    "inferred": EpistemicStatus.INFERRED,
    "predicted": EpistemicStatus.PREDICTED,
    "simulated": EpistemicStatus.SIMULATED,
    "manual_verified": EpistemicStatus.MANUAL_VERIFIED,
}
#: CCIA §2.1: identified → inferred mechanism; partial / non_identified → hypothesis
#: («allowed planning use = informational/human review»).
_IDENTIFICATION_STATUS_MAP: Mapping[str, EpistemicStatus] = {
    "identified": EpistemicStatus.INFERRED,
    "partial": EpistemicStatus.HYPOTHESIS,
    "non_identified": EpistemicStatus.HYPOTHESIS,
}
#: Causal-contract PlanningUse (informational / review / bounded) → shared PlanningUse.
_CAUSAL_PLANNING_USE_MAP: Mapping[str, PlanningUse] = {
    "informational": PlanningUse.INVESTIGATION,
    "review": PlanningUse.REVIEW,
    "bounded": PlanningUse.WITH_UNCERTAINTY,
}


def _raw(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip().lower()


def _mapped(value: Any, table: Mapping[str, Any], name: str) -> Any:
    key = _raw(value)
    if key not in table:
        raise ValueError(f"unknown {name}: {value!r} (fail-closed: no implicit status)")
    return table[key]


def memory_status_to_epistemic(status: Any) -> EpistemicStatus:
    """EBM ``MemoryEpistemicStatus`` → shared :class:`EpistemicStatus` (unknown → ValueError)."""
    return _mapped(status, _MEMORY_STATUS_MAP, "memory epistemic status")


def causal_belief_status_to_epistemic(status: Any) -> EpistemicStatus:
    """Causal TMS ``CausalBeliefStatus`` → shared :class:`EpistemicStatus` (unknown → ValueError)."""
    return _mapped(status, _CAUSAL_BELIEF_STATUS_MAP, "causal belief status")


def identification_status_to_epistemic(status: Any) -> EpistemicStatus:
    """``IdentificationStatus`` → shared status of the mechanism claim (unknown → ValueError)."""
    return _mapped(status, _IDENTIFICATION_STATUS_MAP, "identification status")


def causal_planning_use_to_shared(use: Any) -> PlanningUse:
    """Causal-contract ``PlanningUse`` → shared :class:`PlanningUse` (unknown → ValueError)."""
    return _mapped(use, _CAUSAL_PLANNING_USE_MAP, "causal planning use")


# --------------------------------------------------------------------------------------
# CCIA §1.2: checked planning use (quality/freshness, scope/authority, uncertainty)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanningUsePolicy:
    """Deployment policy for the conditions of the §1.2 table. The addendum names the conditions
    («quality/freshness», «scope/authority») but no values, so every field is mandatory."""

    #: an observation older than this (``at - valid_time``) is not fresh
    max_observation_age: timedelta
    #: minimum measurement quality in [0, 1]
    min_observation_quality: float
    #: authorized verifier -> scopes (subject/boundary) the verifier may confirm
    verifier_scopes: Mapping[str, frozenset[str]]

    def __post_init__(self) -> None:
        if not isinstance(self.max_observation_age, timedelta) or self.max_observation_age <= timedelta(0):
            raise ValueError("max_observation_age must be a positive timedelta")
        quality = float(self.min_observation_quality)
        if not isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("min_observation_quality must be finite and in [0, 1]")
        object.__setattr__(self, "min_observation_quality", quality)
        object.__setattr__(self, "verifier_scopes", FrozenMap({
            str(name).strip(): frozenset(str(item) for item in scopes)
            for name, scopes in dict(self.verifier_scopes).items() if str(name).strip()
        }))


@dataclass(frozen=True)
class PlanningUseGrant:
    claim_id: str
    status: EpistemicStatus
    use: PlanningUse
    #: carried for inferred claims («Да, но с uncertainty»)
    uncertainty: Optional[float] = None
    reasons: tuple[str, ...] = ()


def grant_planning_use(
    claim: EpistemicClaim,
    policy: PlanningUsePolicy,
    *,
    at: datetime,
    observation_quality: Optional[float] = None,
) -> PlanningUseGrant:
    """Checked planning use of one claim (CCIA §1.2 table, p.2).

    * observed / manual_verified without ``provenance_refs`` → REVIEW (no provenance, no direct use);
    * observed: DIRECT only within quality/freshness; quality comes from ``observation_quality`` or
      ``claim.metadata['measurement_quality']``; unknown / low quality or stale → REVIEW;
      ``valid_time`` in the future → BLOCKED (an observation of the future is impossible);
    * manual_verified: DIRECT_SCOPED only when ``verified_by`` is an authorized verifier whose scopes
      cover the claim's subject and boundary scope, else REVIEW;
    * inferred: WITH_UNCERTAINTY, the grant carries the claim's uncertainty;
    * every other status: the table value (plan support / investigation / review / blocked).
    """
    if not isinstance(policy, PlanningUsePolicy):
        raise TypeError("policy must be a PlanningUsePolicy")
    moment = _aware(at, "at")
    status = claim.epistemic_status
    if status in OBSERVATIONAL_STATUSES and not claim.provenance_refs:
        # CCIA §1.2: «observed — факт наблюдения с provenance»; without provenance nothing is direct.
        return PlanningUseGrant(claim.claim_id, status, PlanningUse.REVIEW, reasons=("provenance_missing",))
    if status is EpistemicStatus.OBSERVED and not is_model_output(claim):
        raw = observation_quality if observation_quality is not None else claim.metadata.get("measurement_quality")
        try:
            quality = float(raw) if raw is not None and not isinstance(raw, bool) else None
        except (TypeError, ValueError):
            quality = None
        if claim.valid_time > moment:
            return PlanningUseGrant(claim.claim_id, status, PlanningUse.BLOCKED, reasons=("valid_time_in_future",))
        reasons = []
        if quality is None or not isfinite(quality):
            reasons.append("observation_quality_unknown")
        elif quality < policy.min_observation_quality:
            reasons.append("observation_quality_below_policy")
        if moment - claim.valid_time > policy.max_observation_age:
            reasons.append("observation_stale")
        if reasons:
            return PlanningUseGrant(claim.claim_id, status, PlanningUse.REVIEW, reasons=tuple(reasons))
        return PlanningUseGrant(claim.claim_id, status, PlanningUse.DIRECT)
    if status is EpistemicStatus.MANUAL_VERIFIED:
        scopes = policy.verifier_scopes.get(str(claim.verified_by).strip())
        if scopes is None:
            return PlanningUseGrant(claim.claim_id, status, PlanningUse.REVIEW, reasons=("verifier_not_authorized",))
        outside = sorted((set(claim.subject_scope) | set(claim.boundary_scope)) - set(scopes))
        if outside:
            return PlanningUseGrant(claim.claim_id, status, PlanningUse.REVIEW,
                                    reasons=tuple(f"verification_outside_authority_scope:{item}" for item in outside))
        return PlanningUseGrant(claim.claim_id, status, PlanningUse.DIRECT_SCOPED)
    if status is EpistemicStatus.INFERRED:
        return PlanningUseGrant(claim.claim_id, status, PlanningUse.WITH_UNCERTAINTY, uncertainty=claim.uncertainty)
    return PlanningUseGrant(claim.claim_id, status, _DEFAULT_PLANNING_USE[status])


class ClaimLedgerRepository(Protocol):
    """Durable append-only journal of claim versions (claim_id, version, lineage, digest, model_output)."""

    def load(self) -> Iterable[Mapping[str, Any]]: ...

    def append(self, row: Mapping[str, Any]) -> None: ...


class InMemoryClaimLedgerRepository:
    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def load(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(dict(r) for r in self._rows)

    def append(self, row: Mapping[str, Any]) -> None:
        self._rows.append(dict(row))


class JsonlClaimLedgerRepository:
    """File journal: one JSON line per recorded claim version, fsync'd (survives restarts)."""

    def __init__(self, path: str) -> None:
        self._path = str(path)

    def load(self) -> tuple[Mapping[str, Any], ...]:
        if not os.path.exists(self._path):
            return ()
        rows = []
        with open(self._path, "r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"corrupt claim ledger line {number}: {exc}") from exc
        return tuple(rows)

    def append(self, row: Mapping[str, Any]) -> None:
        directory = os.path.dirname(self._path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class ClaimLedger:
    """Append-only view of claim versions by claim_id (CCIA §1.1 version, §1.2 statuses never mixed).

    Every recorded claim is re-verified (:meth:`EpistemicClaim.verify_integrity`). A new version of a
    known claim must extend the recorded status lineage and must not go back in version; the same
    version must carry the same content (key-independent digest). A claim that re-appears with a
    rewritten lineage (e.g. inferred -> observed via a fresh object) is rejected, and so is an
    "observation" whose provenance is model output (simulator, prediction, LLM, ...).

    One ledger is shared by all traces of a runtime (and, with a durable ``repository``, across
    restarts): a claim id once seen as a simulation can never re-appear as an observation in a later
    trace.
    """

    def __init__(self, repository: Optional[ClaimLedgerRepository] = None) -> None:
        import threading

        self._history: dict[str, tuple[EpistemicStatus, ...]] = {}
        self._model_output: dict[str, bool] = {}
        self._version: dict[str, tuple[int, str]] = {}
        self._lock = threading.Lock()
        self._repository = repository
        for row in (repository.load() if repository is not None else ()):
            claim_id = str(row["claim_id"])
            self._history[claim_id] = tuple(EpistemicStatus(v) for v in row["lineage"])
            self._version[claim_id] = (int(row["version"]), str(row["digest"]))
            self._model_output[claim_id] = self._model_output.get(claim_id, False) or bool(row["model_output"])

    def record(self, claim: EpistemicClaim) -> EpistemicClaim:
        if not isinstance(claim, EpistemicClaim):
            raise EpistemicTransitionError(f"not an EpistemicClaim: {type(claim).__name__}")
        # A claim may have bypassed __post_init__ (object.__setattr__, pickle, copy): re-verify the seal.
        claim.verify_integrity()
        if claim.epistemic_status in OBSERVATIONAL_STATUSES and any(
                str(ref).lower().startswith(MODEL_PROVENANCE_PREFIXES) for ref in claim.provenance_refs):
            raise EpistemicTransitionError(
                f"claim {claim.claim_id}: model-output provenance cannot be an observation")
        digest = claim_content_digest(claim)
        history = tuple(claim.status_history)
        with self._lock:
            known = self._history.get(claim.claim_id)
            if known is not None:
                if history[: len(known)] != known:
                    raise EpistemicTransitionError(
                        f"claim {claim.claim_id}: status lineage {[s.value for s in history]} does not extend "
                        f"the recorded lineage {[s.value for s in known]}"
                    )
                if self._model_output.get(claim.claim_id) and claim.epistemic_status in OBSERVATIONAL_STATUSES:
                    raise EpistemicTransitionError("status laundering from model output to observation is forbidden")
                known_version, known_digest = self._version[claim.claim_id]
                if claim.version < known_version:
                    raise EpistemicTransitionError(
                        f"claim {claim.claim_id}: version {claim.version} is older than recorded {known_version}"
                    )
                if claim.version == known_version:
                    if digest != known_digest:
                        raise EpistemicTransitionError(
                            f"claim {claim.claim_id}: version {claim.version} re-used with different content"
                        )
                    return claim  # the same version seen again (e.g. the same observation in a later trace)
            model_output = self._model_output.get(claim.claim_id, False) or is_model_output(claim)
            if self._repository is not None:
                self._repository.append({"claim_id": claim.claim_id, "version": int(claim.version), "digest": digest,
                                         "lineage": [s.value for s in history], "model_output": model_output})
            self._history[claim.claim_id] = history
            self._version[claim.claim_id] = (claim.version, digest)
            self._model_output[claim.claim_id] = model_output
        return claim

    def lineage(self, claim_id: str) -> tuple[EpistemicStatus, ...]:
        return self._history.get(str(claim_id), ())

    def version(self, claim_id: str) -> Optional[int]:
        entry = self._version.get(str(claim_id))
        return entry[0] if entry else None


def is_model_output(claim: EpistemicClaim) -> bool:
    """True when the claim is (or ever was) a prediction/simulation/plan output."""
    return (
        claim.claim_type in MODEL_OUTPUT_CLAIM_TYPES
        or claim.epistemic_status in MODEL_OUTPUT_STATUSES
        or any(item in MODEL_OUTPUT_STATUSES for item in claim.status_history)
    )


__all__ = [
    "ALLOWED_STATUSES_BY_TYPE",
    "ClaimLedger",
    "ClaimLedgerRepository",
    "InMemoryClaimLedgerRepository",
    "JsonlClaimLedgerRepository",
    "MODEL_PROVENANCE_PREFIXES",
    "claim_content_digest",
    "ClaimType",
    "EpistemicClaim",
    "EpistemicStatus",
    "EpistemicTransitionError",
    "FrozenMap",
    "MODEL_OUTPUT_CLAIM_TYPES",
    "MODEL_OUTPUT_STATUSES",
    "OBSERVATIONAL_STATUSES",
    "PLANNING_USE_ORDER",
    "PlanningUse",
    "PlanningUseGrant",
    "PlanningUsePolicy",
    "causal_belief_status_to_epistemic",
    "causal_planning_use_to_shared",
    "check_status_history",
    "deep_freeze",
    "grant_planning_use",
    "identification_status_to_epistemic",
    "is_model_output",
    "memory_status_to_epistemic",
    "thaw",
    "weakest_planning_use",
]
