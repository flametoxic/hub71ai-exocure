"""CCIA §5.3 (p.7-8): plan commitments and delegation.

AgentCoordinationController needs a formal delegation contract:
delegation_id, principal, agent/role, subgoal, scope, authority boundary, deadline,
required evidence, escalation path, completion verification.

"Delegation does not transfer unrestricted authority": the delegated boundary must be a
subset of the principal's own boundary, may never be a wildcard, expires at the deadline,
and re-delegation is allowed only when explicitly granted and only narrowing further.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Sequence


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _tokens(values, name: str, *, required: bool) -> frozenset[str]:
    items = frozenset(str(item).strip() for item in values)
    if "" in items:
        raise ValueError(f"{name} contains an empty value")
    if any(item in {"*", "all", "any"} for item in items):
        raise ValueError(f"{name} cannot be a wildcard: delegation never transfers unrestricted authority")
    if required and not items:
        raise ValueError(f"{name} is required")
    return items


@dataclass(frozen=True)
class AuthorityBoundary:
    """Explicit, enumerable authority: allowed actions within allowed scopes."""

    allowed_actions: frozenset[str]
    allowed_scopes: frozenset[str]
    may_redelegate: bool = False
    #: actions that still need human approval even inside the boundary
    approval_required_actions: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_actions", _tokens(self.allowed_actions, "allowed_actions", required=True))
        object.__setattr__(self, "allowed_scopes", _tokens(self.allowed_scopes, "allowed_scopes", required=True))
        approval = _tokens(self.approval_required_actions, "approval_required_actions", required=False)
        if not approval <= self.allowed_actions:
            raise ValueError("approval_required_actions must be allowed actions")
        object.__setattr__(self, "approval_required_actions", approval)
        if not isinstance(self.may_redelegate, bool):
            raise ValueError("may_redelegate must be a bool")

    def contains(self, other: "AuthorityBoundary") -> bool:
        return (
            other.allowed_actions <= self.allowed_actions
            and other.allowed_scopes <= self.allowed_scopes
            and (self.may_redelegate or not other.may_redelegate)
            and self.approval_required_actions & other.allowed_actions <= other.approval_required_actions
        )


class DelegationDecision(str, Enum):
    ALLOWED = "allowed"
    REQUIRES_APPROVAL = "requires_approval"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class AuthorizationResult:
    decision: DelegationDecision
    reason: str
    escalate_to: tuple[str, ...] = ()

    @property
    def allowed(self) -> bool:
        return self.decision is DelegationDecision.ALLOWED


class CompletionStatus(str, Enum):
    VERIFIED = "verified"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"
    LATE = "late"


@dataclass(frozen=True)
class CompletionResult:
    status: CompletionStatus
    missing_evidence: tuple[str, ...] = ()
    escalate_to: tuple[str, ...] = ()


@dataclass(frozen=True)
class DelegationContract:
    delegation_id: str
    principal: str
    agent: str
    role: str
    subgoal: str
    scope: frozenset[str]
    authority: AuthorityBoundary
    issued_at: datetime
    deadline: datetime
    required_evidence: tuple[str, ...]
    escalation_path: tuple[str, ...]
    completion_verification: str
    trace_id: str
    parent_delegation_id: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    # The principal's own boundary. Required: a contract that cannot show it stays inside the
    # principal's authority does not exist (use ``issue()``; direct construction is checked too).
    principal_authority: Optional[AuthorityBoundary] = None

    def __post_init__(self) -> None:
        if not isinstance(self.principal_authority, AuthorityBoundary):
            raise ValueError(
                "DelegationContract requires principal_authority (create it via DelegationContract.issue)"
            )
        for name in ("delegation_id", "principal", "agent", "role", "subgoal", "completion_verification", "trace_id"):
            if not str(getattr(self, name) or "").strip():
                raise ValueError(f"{name} is required")
        if self.agent == self.principal:
            raise ValueError("principal cannot delegate to itself")
        if not isinstance(self.authority, AuthorityBoundary):
            raise ValueError("authority must be an explicit AuthorityBoundary")
        scope = _tokens(self.scope, "scope", required=True)
        if not scope <= self.authority.allowed_scopes:
            raise ValueError("delegation scope must lie inside the authority boundary")
        object.__setattr__(self, "scope", scope)
        issued = _aware(self.issued_at, "issued_at")
        deadline = _aware(self.deadline, "deadline")
        if deadline <= issued:
            raise ValueError("deadline must be after issued_at")
        object.__setattr__(self, "issued_at", issued)
        object.__setattr__(self, "deadline", deadline)
        evidence = tuple(str(item).strip() for item in self.required_evidence)
        if not evidence or "" in evidence:
            raise ValueError("required_evidence is required")
        object.__setattr__(self, "required_evidence", evidence)
        path = tuple(str(item).strip() for item in self.escalation_path)
        if not path or "" in path:
            raise ValueError("escalation_path is required")
        object.__setattr__(self, "escalation_path", path)
        if not self.principal_authority.contains(self.authority):
            raise ValueError("delegated authority exceeds the principal's authority")
        if not scope <= self.principal_authority.allowed_scopes:
            raise ValueError("delegated scope exceeds the principal's authority")

    @classmethod
    def issue(cls, *, principal_authority: AuthorityBoundary, **fields) -> "DelegationContract":
        """Create a delegation that never exceeds the principal's own authority."""
        return cls(principal_authority=principal_authority, **fields)

    def delegate(self, **fields) -> "DelegationContract":
        """Re-delegation: only if granted, only narrower, never beyond this deadline."""
        if not self.authority.may_redelegate:
            raise ValueError("this delegation does not permit re-delegation")
        child = DelegationContract.issue(
            principal_authority=self.authority,
            principal=self.agent,
            parent_delegation_id=self.delegation_id,
            **fields,
        )
        if child.deadline > self.deadline:
            raise ValueError("re-delegation deadline cannot exceed the parent deadline")
        if not child.scope <= self.scope:
            raise ValueError("re-delegation scope must narrow the parent scope")
        return child

    def authorize(self, *, action: str, scope: str, at: datetime) -> AuthorizationResult:
        instant = _aware(at, "at")
        if instant < self.issued_at:
            return AuthorizationResult(DelegationDecision.ESCALATE, "delegation_not_yet_active", self.escalation_path)
        if instant > self.deadline:
            return AuthorizationResult(DelegationDecision.ESCALATE, "deadline_passed", self.escalation_path)
        if action not in self.authority.allowed_actions:
            return AuthorizationResult(DelegationDecision.ESCALATE, "action_outside_authority", self.escalation_path)
        if scope not in self.scope:
            return AuthorizationResult(DelegationDecision.ESCALATE, "scope_outside_delegation", self.escalation_path)
        if action in self.authority.approval_required_actions:
            return AuthorizationResult(DelegationDecision.REQUIRES_APPROVAL, "human_approval_required",
                                       self.escalation_path[:1])
        return AuthorizationResult(DelegationDecision.ALLOWED, "within_delegated_authority")

    def verify_completion(self, *, evidence: Mapping[str, str], verified_by: str, at: datetime) -> CompletionResult:
        """Completion is accepted only with every required evidence item, verified by someone
        other than the delegate, before the deadline; otherwise escalate."""
        instant = _aware(at, "at")
        if not str(verified_by or "").strip() or verified_by == self.agent:
            raise ValueError("completion must be verified by someone other than the delegate")
        missing = tuple(item for item in self.required_evidence if not str(evidence.get(item) or "").strip())
        if missing:
            return CompletionResult(CompletionStatus.INCOMPLETE_EVIDENCE, missing, self.escalation_path)
        if instant > self.deadline:
            return CompletionResult(CompletionStatus.LATE, (), self.escalation_path)
        return CompletionResult(CompletionStatus.VERIFIED)


def plan_step_scopes(contract: Any, step: Any) -> tuple[str, ...]:
    """Scopes an actuating step acts on: its own bodies, else the plan's entities/bodies."""
    scopes = tuple(getattr(step, "body_ids", ()) or ()) or tuple(getattr(contract, "entity_ids", ()) or ()) \
        or tuple(getattr(contract, "body_ids", ()) or ())
    return tuple(str(item) for item in scopes)


_DECISION_ORDER = (DelegationDecision.ALLOWED, DelegationDecision.REQUIRES_APPROVAL, DelegationDecision.ESCALATE)


class AgentCoordinationController:
    """CCIA §5.3 (p.7-8): the controller that owns delegation contracts.

    * ``issue`` registers a delegation and (with commitment memory) records the open obligation
      ``commitment:delegation:<id>`` (kind pending_operator_task, due at the deadline, fulfilled only
      with the contract's required evidence);
    * ``authorize_plan`` — every actuating step of a plan executed by the delegate must be inside the
      delegated actions, scope and time (``DelegationContract.authorize``); the strictest answer wins;
    * ``screen_proposals`` — a coordination proposal (``coordinate_plans``) from a delegated agent is
      accepted only within its delegation scope and deadline;
    * ``complete`` — ``verify_completion``; VERIFIED closes the delegation commitment with the evidence.

    Unknown delegation → ESCALATE (fail-closed). Delegation never transfers unrestricted authority.
    """

    def __init__(self, *, commitments: Any = None, tenant_id: Optional[str] = None) -> None:
        if commitments is not None and not str(tenant_id or "").strip():
            raise ValueError("commitment memory requires tenant_id")
        self.commitments = commitments
        self.tenant_id = tenant_id
        self._delegations: dict[str, DelegationContract] = {}

    @staticmethod
    def commitment_id(delegation_id: str) -> str:
        return f"commitment:delegation:{delegation_id}"

    def get(self, delegation_id: str) -> Optional[DelegationContract]:
        return self._delegations.get(str(delegation_id))

    async def issue(self, contract: DelegationContract, *, recorded_at: Optional[datetime] = None) -> DelegationContract:
        if not isinstance(contract, DelegationContract):
            raise TypeError("a DelegationContract is required")
        if contract.delegation_id in self._delegations:
            raise ValueError(f"delegation already issued: {contract.delegation_id}")
        if contract.parent_delegation_id is not None and contract.parent_delegation_id not in self._delegations:
            raise ValueError("re-delegation requires the parent delegation to be registered")
        if self.commitments is not None:
            from ..memory.commitments import Commitment, CommitmentKind

            await self.commitments.record(Commitment(
                commitment_id=self.commitment_id(contract.delegation_id),
                kind=CommitmentKind.PENDING_OPERATOR_TASK,
                tenant_id=str(self.tenant_id),
                owner=contract.agent,
                source_ref=contract.delegation_id,
                subject_refs=tuple(sorted(contract.scope)),
                description=f"delegated subgoal {contract.subgoal} ({contract.principal} -> {contract.agent})",
                created_at=contract.issued_at,
                due_at=contract.deadline,
                required_evidence=contract.required_evidence,
                trace_id=contract.trace_id,
            ), recorded_at=recorded_at or contract.issued_at)
        self._delegations[contract.delegation_id] = contract
        return contract

    def authorize(self, delegation_id: str, *, action: str, scope: str, at: datetime) -> AuthorizationResult:
        contract = self.get(delegation_id)
        if contract is None:
            return AuthorizationResult(DelegationDecision.ESCALATE, "delegation_unknown")
        return contract.authorize(action=action, scope=scope, at=at)

    def authorize_plan(self, delegation_id: str, plan: Any, *, at: datetime) -> AuthorizationResult:
        contract = self.get(delegation_id)
        if contract is None:
            return AuthorizationResult(DelegationDecision.ESCALATE, "delegation_unknown")
        worst = AuthorizationResult(DelegationDecision.ALLOWED, "within_delegated_authority")
        steps = tuple(plan.actuating_steps())
        if str(getattr(plan, "trace_id", "") or "") and str(plan.trace_id) != contract.trace_id:
            return AuthorizationResult(DelegationDecision.ESCALATE, "trace_outside_delegation", contract.escalation_path)
        for step in steps:
            scopes = plan_step_scopes(plan, step)
            if not scopes:
                return AuthorizationResult(DelegationDecision.ESCALATE, f"step_scope_unknown:{step.step_id}",
                                           contract.escalation_path)
            for scope in scopes:
                result = contract.authorize(action=str(step.action_type), scope=scope, at=at)
                if _DECISION_ORDER.index(result.decision) > _DECISION_ORDER.index(worst.decision):
                    worst = AuthorizationResult(result.decision, f"{result.reason}:{step.step_id}", result.escalate_to)
        return worst

    def screen_proposals(self, proposals: Sequence[Any], *, delegations: Mapping[str, str], at: datetime):
        """Split coordination proposals: agents acting under a delegation (``agent_id → delegation_id``)
        must stay inside its scope and deadline. Returns ``(admissible, rejected{proposal_id: reason})``;
        the admissible ones go to ``coordinate_plans``."""
        admissible, rejected = [], {}
        for proposal in proposals:
            delegation_id = delegations.get(str(proposal.agent_id))
            if delegation_id is None:
                admissible.append(proposal)
                continue
            contract = self.get(delegation_id)
            instant = _aware(at, "at")
            if contract is None or contract.agent != proposal.agent_id:
                rejected[proposal.proposal_id] = "delegation_unknown"
            elif instant < contract.issued_at or instant > contract.deadline:
                rejected[proposal.proposal_id] = "delegation_not_active"
            elif not frozenset(proposal.scope) or not frozenset(proposal.scope) <= contract.scope:
                rejected[proposal.proposal_id] = "scope_outside_delegation"
            else:
                admissible.append(proposal)
        return tuple(admissible), rejected

    async def complete(self, delegation_id: str, *, evidence: Mapping[str, str], verified_by: str,
                       at: datetime) -> CompletionResult:
        contract = self.get(delegation_id)
        if contract is None:
            raise KeyError(f"unknown delegation: {delegation_id}")
        result = contract.verify_completion(evidence=evidence, verified_by=verified_by, at=at)
        if result.status is CompletionStatus.VERIFIED and self.commitments is not None:
            await self.commitments.fulfil(
                self.commitment_id(delegation_id), tenant_id=str(self.tenant_id),
                evidence={item: str(evidence[item]) for item in contract.required_evidence},
                closed_by=verified_by, recorded_at=at,
            )
        return result


__all__ = [
    "AgentCoordinationController",
    "AuthorityBoundary",
    "AuthorizationResult",
    "CompletionResult",
    "CompletionStatus",
    "DelegationContract",
    "DelegationDecision",
    "plan_step_scopes",
]

