"""UMRM section 10 (page 8): LearningCandidate and its governed path to production.

    candidate
    -> schema/constraint validation
    -> simulation
    -> golden replay
    -> shadow
    -> human/policy approval
    -> canary
    -> versioned deployment or rollback

"This never changes production directly."  The pipeline is a state machine whose stage is *computed*
by replaying the evidence history from the start; a stage cannot be declared.  Every transition
requires evidence of the type prescribed for that step, bound to the candidate id.  Deployment is
performed only by :class:`ParameterStore`, only from a record whose replayed history contains a passed
approval and a passed canary, and every deployed version can be rolled back.

"Self-development ... does not silently rewrite itself": the proposer of a candidate can never approve
it.  Identities are compared after normalisation (NFKC, trimmed, case-folded), both the approver and the
producer of the approval must differ from the proposer, and :class:`ApprovalEvidence` is minted only by
an :class:`ApproverRegistry` of authorised human/policy approvers (governance configuration); the
:class:`ParameterStore` deploys only approvals minted by *its* registry.

Deployment / rollback evidence is minted only by a :class:`ParameterStore`; every minted evidence object
is kept in the issuer's own journal and the history replay checks it there, so evidence constructed
elsewhere (even with the issuer object) is rejected.  A candidate that was deployed or rolled back by a
store can never be deployed again by that store -- a new candidate with a new approval is required.
"""
from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

import numpy as np

from .assertions import require_aware
from .reality_formulas import (
    DOCUMENT,
    LEARNING_SIGN_NOTE,
    FormulaReference,
    strict_bool,
)

PIPELINE_REFERENCE = FormulaReference(
    "UMRM-10-LEARNING-CANDIDATE-PIPELINE",
    DOCUMENT,
    8,
    "candidate -> schema/constraint validation -> simulation -> golden replay -> shadow -> "
    "human/policy approval -> canary -> versioned deployment or rollback",
)


class LearningTransitionError(ValueError):
    """A pipeline step was attempted without the evidence it requires (fail-closed)."""


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value


def normalize_identity(value: Any) -> str:
    """Canonical form of an actor id for the no-self-approval check (NFKC, trimmed, case-folded,
    inner whitespace collapsed): ``"learner "``, ``"LEARNER"`` and ``"learner"`` are one actor."""
    text = _text("identity", value)
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _readonly_vector(name: str, value: Any) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class LearningCandidate:
    """theta_candidate for ``target`` -- a proposal, never a production value (UMRM 10, page 8)."""

    target: str
    parameter_names: tuple[str, ...]
    theta_current: Any
    theta_candidate: Any
    learning_rate: float
    loss_gradient: Any
    proposed_by: str
    evidence_refs: tuple[str, ...]
    candidate_id: str = field(init=False)
    status: str = field(init=False, default="candidate")
    real_action: bool = field(init=False, default=False)
    sign_note: str = field(init=False, default=LEARNING_SIGN_NOTE)

    def __post_init__(self) -> None:
        _text("target", self.target)
        _text("proposed_by", self.proposed_by)
        names = tuple(self.parameter_names)
        if not names or any(not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != len(names):
            raise ValueError("parameter_names must be unique non-empty names")
        current = _readonly_vector("theta_current", self.theta_current)
        candidate = _readonly_vector("theta_candidate", self.theta_candidate)
        gradient = _readonly_vector("loss_gradient", self.loss_gradient)
        if not (current.shape == candidate.shape == gradient.shape == (len(names),)):
            raise ValueError("theta_current, theta_candidate, loss_gradient and parameter_names must have one length")
        refs = tuple(self.evidence_refs)
        if not refs or any(not isinstance(r, str) or not r.strip() for r in refs):
            raise ValueError("a LearningCandidate must cite the outcome/residual evidence it was derived from")
        rate = float(self.learning_rate)
        if isinstance(self.learning_rate, (bool, np.bool_)) or not np.isfinite(rate) or rate < 0.0:
            raise ValueError("learning_rate must be a finite non-negative number")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "theta_current", current)
        object.__setattr__(self, "theta_candidate", candidate)
        object.__setattr__(self, "loss_gradient", gradient)
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(self, "learning_rate", rate)
        digest = hashlib.blake2b(digest_size=12)
        for part in (self.target, *names, current.tobytes(), candidate.tobytes(), self.proposed_by, *refs):
            digest.update(part if isinstance(part, bytes) else part.encode("utf-8"))
            digest.update(b"\x00")
        object.__setattr__(self, "candidate_id", f"learning-candidate:{digest.hexdigest()}")


class LearningStage(str, Enum):
    CANDIDATE = "candidate"
    VALIDATED = "schema_constraint_validated"
    SIMULATED = "simulated"
    GOLDEN_REPLAYED = "golden_replayed"
    SHADOWED = "shadowed"
    APPROVED = "approved"
    CANARY = "canary_passed"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


TERMINAL_STAGES = frozenset({LearningStage.ROLLED_BACK, LearningStage.REJECTED})


@dataclass(frozen=True)
class StageEvidence:
    """Common evidence envelope: bound to one candidate, timestamped, with an auditable reference."""

    candidate_id: str
    evidence_ref: str
    produced_by: str
    recorded_at: datetime

    def __post_init__(self) -> None:
        _text("candidate_id", self.candidate_id)
        _text("evidence_ref", self.evidence_ref)
        _text("produced_by", self.produced_by)
        object.__setattr__(self, "recorded_at", require_aware(self.recorded_at, "recorded_at"))

    @property
    def passed(self) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass(frozen=True)
class ValidationEvidence(StageEvidence):
    schema_valid: bool = False
    constraints_valid: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        strict_bool("schema_valid", self.schema_valid)
        strict_bool("constraints_valid", self.constraints_valid)

    @property
    def passed(self) -> bool:
        return bool(self.schema_valid) and bool(self.constraints_valid)


@dataclass(frozen=True)
class _PassFailEvidence(StageEvidence):
    succeeded: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        strict_bool("succeeded", self.succeeded)

    @property
    def passed(self) -> bool:
        return bool(self.succeeded)


@dataclass(frozen=True)
class SimulationEvidence(_PassFailEvidence):
    simulation_run_id: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        _text("simulation_run_id", self.simulation_run_id)


@dataclass(frozen=True)
class GoldenReplayEvidence(_PassFailEvidence):
    replay_set_id: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        _text("replay_set_id", self.replay_set_id)


@dataclass(frozen=True)
class ShadowEvidence(_PassFailEvidence):
    shadow_run_id: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        _text("shadow_run_id", self.shadow_run_id)


APPROVER_KINDS = ("human", "policy")


class _EvidenceJournal:
    """Identity journal of the evidence objects an issuer minted (checked on every replay)."""

    def __init__(self) -> None:
        self._minted: dict[int, StageEvidence] = {}

    def add(self, evidence: StageEvidence) -> None:
        self._minted[id(evidence)] = evidence

    def contains(self, evidence: StageEvidence) -> bool:
        return self._minted.get(id(evidence)) is evidence


@dataclass(frozen=True)
class ApprovalEvidence(StageEvidence):
    """Human/policy approval -- minted only by :meth:`ApproverRegistry.approve` (``issuer``)."""

    approver_id: str = ""
    approver_kind: str = ""
    policy_version: str = ""
    approved: bool = False
    issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.issuer, ApproverRegistry):
            raise LearningTransitionError("approval evidence is minted only by an ApproverRegistry")
        _text("approver_id", self.approver_id)
        _text("policy_version", self.policy_version)
        if self.approver_kind not in APPROVER_KINDS:
            raise ValueError(f"approver_kind must be one of {APPROVER_KINDS} (UMRM 10: human/policy approval)")
        strict_bool("approved", self.approved)

    @property
    def passed(self) -> bool:
        return bool(self.approved)


@dataclass(frozen=True)
class CanaryEvidence(_PassFailEvidence):
    canary_id: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        _text("canary_id", self.canary_id)


@dataclass(frozen=True)
class DeploymentEvidence(StageEvidence):
    deployment_version: str = ""
    previous_version: str = ""
    issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.issuer, ParameterStore):
            raise LearningTransitionError("deployment evidence is minted only by ParameterStore.deploy")
        _text("deployment_version", self.deployment_version)
        _text("previous_version", self.previous_version)

    @property
    def passed(self) -> bool:
        return True


@dataclass(frozen=True)
class RollbackEvidence(StageEvidence):
    reason: str = ""
    restored_version: str = ""
    issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.issuer, ParameterStore):
            raise LearningTransitionError("rollback evidence is minted only by ParameterStore.rollback")
        _text("reason", self.reason)
        _text("restored_version", self.restored_version)

    @property
    def passed(self) -> bool:
        return True


# stage -> {evidence type: (stage on pass, stage on fail)}
TRANSITIONS: Mapping[LearningStage, Mapping[type, tuple[LearningStage, LearningStage]]] = {
    LearningStage.CANDIDATE: {ValidationEvidence: (LearningStage.VALIDATED, LearningStage.REJECTED)},
    LearningStage.VALIDATED: {SimulationEvidence: (LearningStage.SIMULATED, LearningStage.REJECTED)},
    LearningStage.SIMULATED: {GoldenReplayEvidence: (LearningStage.GOLDEN_REPLAYED, LearningStage.REJECTED)},
    LearningStage.GOLDEN_REPLAYED: {ShadowEvidence: (LearningStage.SHADOWED, LearningStage.REJECTED)},
    LearningStage.SHADOWED: {ApprovalEvidence: (LearningStage.APPROVED, LearningStage.REJECTED)},
    # a failed canary is rolled back (the canary never became the versioned production parameters)
    LearningStage.APPROVED: {CanaryEvidence: (LearningStage.CANARY, LearningStage.ROLLED_BACK)},
    LearningStage.CANARY: {
        DeploymentEvidence: (LearningStage.DEPLOYED, LearningStage.DEPLOYED),
        RollbackEvidence: (LearningStage.ROLLED_BACK, LearningStage.ROLLED_BACK),
    },
    LearningStage.DEPLOYED: {RollbackEvidence: (LearningStage.ROLLED_BACK, LearningStage.ROLLED_BACK)},
}



def _replay(candidate: LearningCandidate, history: tuple[StageEvidence, ...]) -> LearningStage:
    stage = LearningStage.CANDIDATE
    last_time: datetime | None = None
    for evidence in history:
        if not isinstance(evidence, StageEvidence):
            raise LearningTransitionError("history must contain StageEvidence objects")
        if evidence.candidate_id != candidate.candidate_id:
            raise LearningTransitionError("evidence is bound to a different candidate")
        if last_time is not None and evidence.recorded_at < last_time:
            raise LearningTransitionError("evidence history must be in time order")
        allowed = TRANSITIONS.get(stage, {})
        rule = allowed.get(type(evidence))
        if rule is None:
            expected = ", ".join(t.__name__ for t in allowed) or "nothing (terminal stage)"
            raise LearningTransitionError(
                f"stage '{stage.value}' requires {expected}; got {type(evidence).__name__}"
            )
        if isinstance(evidence, (ApprovalEvidence, DeploymentEvidence, RollbackEvidence)) \
                and not evidence.issuer._journal.contains(evidence):
            raise LearningTransitionError(
                f"{type(evidence).__name__} was not minted by its issuer (forged evidence is rejected)"
            )
        if isinstance(evidence, ApprovalEvidence):
            proposer = normalize_identity(candidate.proposed_by)
            if proposer in (normalize_identity(evidence.approver_id), normalize_identity(evidence.produced_by)):
                raise LearningTransitionError("a candidate cannot be approved by its own proposer (no self-promotion)")
        stage = rule[0] if evidence.passed else rule[1]
        last_time = evidence.recorded_at
    return stage


@dataclass(frozen=True)
class LearningRecord:
    """A candidate plus its evidence history; ``stage`` is computed by replaying the history."""

    candidate: LearningCandidate
    history: tuple[StageEvidence, ...] = ()
    stage: LearningStage = field(init=False)
    reference: FormulaReference = field(init=False, default=PIPELINE_REFERENCE)
    real_action: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, LearningCandidate):
            raise TypeError("candidate must be a LearningCandidate")
        history = tuple(self.history)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "stage", _replay(self.candidate, history))

    def advance(self, evidence: StageEvidence) -> "LearningRecord":
        return LearningRecord(self.candidate, self.history + (evidence,))

    def evidence_of(self, kind: type) -> tuple[StageEvidence, ...]:
        return tuple(e for e in self.history if isinstance(e, kind))

    @property
    def approved(self) -> bool:
        return any(e.passed for e in self.evidence_of(ApprovalEvidence))


@dataclass(frozen=True)
class DeployedVersion:
    target: str
    version: str
    parameter_names: tuple[str, ...]
    theta: Any
    candidate_id: str | None


class ApproverRegistry:
    """Authorised human/policy approvers (UMRM 10 "human/policy approval").

    The approver ids and kinds come from governance configuration -- the TZ names the step, not the
    people.  :meth:`approve` is the only way to mint :class:`ApprovalEvidence`: the approver must be
    registered, must not be the proposer (normalised ids), and is recorded as the evidence producer.
    """

    def __init__(self) -> None:
        self._approvers: dict[str, tuple[str, str]] = {}  # normalised id -> (id, kind)
        self._journal = _EvidenceJournal()

    def register(self, approver_id: str, approver_kind: str) -> None:
        if approver_kind not in APPROVER_KINDS:
            raise ValueError(f"approver_kind must be one of {APPROVER_KINDS}")
        key = normalize_identity(approver_id)
        if key in self._approvers:
            raise ValueError(f"approver {approver_id!r} is already registered")
        self._approvers[key] = (approver_id.strip(), approver_kind)

    def is_authorised(self, approver_id: str) -> bool:
        return normalize_identity(approver_id) in self._approvers

    def approve(self, record: "LearningRecord", *, approver_id: str, policy_version: str, approved: bool,
                evidence_ref: str, at: datetime) -> "LearningRecord":
        if not isinstance(record, LearningRecord):
            raise TypeError("record must be a LearningRecord")
        key = normalize_identity(approver_id)
        if key == normalize_identity(record.candidate.proposed_by):
            raise LearningTransitionError("a candidate cannot be approved by its own proposer (no self-promotion)")
        if key not in self._approvers:
            raise LearningTransitionError(f"{approver_id!r} is not an authorised approver")
        canonical, kind = self._approvers[key]
        evidence = ApprovalEvidence(record.candidate.candidate_id, evidence_ref, canonical, at,
                                    approver_id=canonical, approver_kind=kind, policy_version=policy_version,
                                    approved=approved, issuer=self)
        self._journal.add(evidence)
        return record.advance(evidence)


class ParameterStore:
    """Versioned production parameters.  The only writer is :meth:`deploy`/:meth:`rollback`.

    ``approvers`` is the trusted :class:`ApproverRegistry`; only approvals it minted count.  The store
    remembers every candidate it deployed or rolled back and refuses to deploy it again."""

    def __init__(self, *, approvers: ApproverRegistry) -> None:
        if not isinstance(approvers, ApproverRegistry):
            raise TypeError("ParameterStore requires the trusted ApproverRegistry")
        self._approvers = approvers
        self._versions: dict[str, list[DeployedVersion]] = {}
        self._active: dict[str, int] = {}
        self._journal = _EvidenceJournal()
        self._retired: dict[str, str] = {}  # candidate_id -> "deployed" / "rolled_back"

    def _check_own_evidence(self, record: "LearningRecord") -> None:
        for evidence in record.evidence_of(ApprovalEvidence):
            if evidence.issuer is not self._approvers:
                raise LearningTransitionError("approval was not minted by this store's ApproverRegistry")
        for evidence in record.evidence_of(DeploymentEvidence) + record.evidence_of(RollbackEvidence):
            if evidence.issuer is not self:
                raise LearningTransitionError("deployment/rollback evidence was not minted by this ParameterStore")

    def register_baseline(self, *, target: str, version: str, parameter_names, theta) -> DeployedVersion:
        _text("target", target)
        _text("version", version)
        if target in self._versions:
            raise ValueError(f"baseline already registered for {target}")
        entry = DeployedVersion(target, version, tuple(parameter_names), _readonly_vector("theta", theta), None)
        self._versions[target] = [entry]
        self._active[target] = 0
        return entry

    def current(self, target: str) -> DeployedVersion:
        if target not in self._versions:
            raise KeyError(f"no deployed parameters for {target}")
        return self._versions[target][self._active[target]]

    def deploy(self, record: LearningRecord, *, version: str, deployed_by: str, at: datetime) -> LearningRecord:
        """Versioned deployment -- only from a replayed history with passed approval AND canary."""
        if not isinstance(record, LearningRecord):
            raise TypeError("record must be a LearningRecord")
        replayed = LearningRecord(record.candidate, record.history)  # never trust a passed-in stage
        if replayed.stage is not LearningStage.CANARY or not replayed.approved:
            raise LearningTransitionError(
                f"deployment requires stage '{LearningStage.CANARY.value}' with approval; got '{replayed.stage.value}'"
            )
        self._check_own_evidence(replayed)
        candidate = replayed.candidate
        if candidate.candidate_id in self._retired:
            raise LearningTransitionError(
                f"candidate {candidate.candidate_id} was already {self._retired[candidate.candidate_id]}; "
                "re-deployment needs a new candidate with a new approval"
            )
        current = self.current(candidate.target)
        if current.parameter_names != candidate.parameter_names or not np.array_equal(current.theta, candidate.theta_current):
            raise LearningTransitionError("stale candidate: it was not derived from the currently deployed parameters")
        if any(v.version == version for v in self._versions[candidate.target]):
            raise ValueError(f"version {version} already exists for {candidate.target}")
        evidence = DeploymentEvidence(candidate.candidate_id, f"deployment:{candidate.target}:{version}", deployed_by,
                                      at, deployment_version=version, previous_version=current.version,
                                      issuer=self)
        self._journal.add(evidence)
        advanced = replayed.advance(evidence)
        self._retired[candidate.candidate_id] = "deployed"
        entry = DeployedVersion(candidate.target, version, candidate.parameter_names, candidate.theta_candidate,
                                candidate.candidate_id)
        self._versions[candidate.target].append(entry)
        self._active[candidate.target] = len(self._versions[candidate.target]) - 1
        return advanced

    def rollback(self, record: LearningRecord, *, reason: str, rolled_back_by: str, at: datetime) -> LearningRecord:
        """Rollback of a canary or a deployed candidate; production returns to the previous version."""
        if not isinstance(record, LearningRecord):
            raise TypeError("record must be a LearningRecord")
        replayed = LearningRecord(record.candidate, record.history)
        self._check_own_evidence(replayed)
        candidate = replayed.candidate
        if replayed.stage is LearningStage.DEPLOYED:
            deployments = replayed.evidence_of(DeploymentEvidence)
            restored = deployments[-1].previous_version
            versions = self._versions[candidate.target]
            if self.current(candidate.target).candidate_id != candidate.candidate_id:
                raise LearningTransitionError("the candidate is no longer the active version; roll back newer versions first")
            index = next(i for i, v in enumerate(versions) if v.version == restored)
            self._active[candidate.target] = index
        elif replayed.stage is LearningStage.CANARY:
            restored = self.current(candidate.target).version  # canary never reached production parameters
        else:
            raise LearningTransitionError(f"nothing to roll back at stage '{replayed.stage.value}'")
        evidence = RollbackEvidence(candidate.candidate_id, f"rollback:{candidate.target}:{restored}", rolled_back_by,
                                    at, reason=reason, restored_version=restored, issuer=self)
        self._journal.add(evidence)
        advanced = replayed.advance(evidence)
        self._retired[candidate.candidate_id] = "rolled_back"
        return advanced


__all__ = [
    "APPROVER_KINDS",
    "ApprovalEvidence",
    "ApproverRegistry",
    "CanaryEvidence",
    "DeployedVersion",
    "DeploymentEvidence",
    "GoldenReplayEvidence",
    "LearningCandidate",
    "LearningRecord",
    "LearningStage",
    "LearningTransitionError",
    "PIPELINE_REFERENCE",
    "ParameterStore",
    "RollbackEvidence",
    "ShadowEvidence",
    "SimulationEvidence",
    "StageEvidence",
    "TRANSITIONS",
    "ValidationEvidence",
    "normalize_identity",
]
