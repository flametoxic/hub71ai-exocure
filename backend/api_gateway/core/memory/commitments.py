"""CCIA §4.4 (p.6): commitment memory.

Memory must store open commitments — approved plan, pending operator task, required
verification, unresolved incident, promised notification, scheduled maintenance — so that
NarrativeSelf/Planning do not lose unfinished obligations between sessions or restarts.

Storage is durable and append-only (every change is a new version; history is kept).
A commitment closes only as FULFILLED (with evidence covering every required evidence kind)
or CANCELLED (with reason and authority). Nothing reopens a closed commitment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Sequence
from uuid import uuid4

from sqlalchemy import Column, DateTime, Index, Integer, String, UniqueConstraint, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.types import JSON

from ...models.base import Base


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _parse(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


class CommitmentKind(str, Enum):
    APPROVED_PLAN = "approved_plan"
    PENDING_OPERATOR_TASK = "pending_operator_task"
    REQUIRED_VERIFICATION = "required_verification"
    UNRESOLVED_INCIDENT = "unresolved_incident"
    PROMISED_NOTIFICATION = "promised_notification"
    SCHEDULED_MAINTENANCE = "scheduled_maintenance"


class CommitmentStatus(str, Enum):
    OPEN = "open"
    FULFILLED = "fulfilled"
    CANCELLED = "cancelled"


class CommitmentError(ValueError):
    pass


class CommitmentConcurrencyError(CommitmentError):
    pass


@dataclass(frozen=True)
class Commitment:
    commitment_id: str
    kind: CommitmentKind
    tenant_id: str
    owner: str
    source_ref: str
    subject_refs: tuple[str, ...]
    description: str
    created_at: datetime
    due_at: Optional[datetime]
    required_evidence: tuple[str, ...]
    trace_id: str
    status: CommitmentStatus = CommitmentStatus.OPEN
    closed_at: Optional[datetime] = None
    closure_evidence: Mapping[str, str] = field(default_factory=dict)
    closure_reason: str = ""
    closed_by: str = ""

    def __post_init__(self) -> None:
        for name in ("commitment_id", "tenant_id", "owner", "source_ref", "description", "trace_id"):
            if not str(getattr(self, name) or "").strip():
                raise ValueError(f"{name} is required")
        object.__setattr__(self, "kind", CommitmentKind(self.kind))
        object.__setattr__(self, "status", CommitmentStatus(self.status))
        object.__setattr__(self, "subject_refs", tuple(str(item) for item in self.subject_refs))
        required = tuple(str(item).strip() for item in self.required_evidence)
        if not required or any(not item for item in required):
            raise ValueError("required_evidence must name how the commitment is shown to be fulfilled")
        object.__setattr__(self, "required_evidence", required)
        created = _aware(self.created_at, "created_at")
        object.__setattr__(self, "created_at", created)
        if self.due_at is not None:
            due = _aware(self.due_at, "due_at")
            if due < created:
                raise ValueError("due_at cannot precede created_at")
            object.__setattr__(self, "due_at", due)
        elif self.kind in {CommitmentKind.SCHEDULED_MAINTENANCE, CommitmentKind.PROMISED_NOTIFICATION,
                           CommitmentKind.REQUIRED_VERIFICATION}:
            raise ValueError(f"{self.kind.value} commitment requires due_at")
        object.__setattr__(self, "closure_evidence", {str(k): str(v) for k, v in dict(self.closure_evidence).items()})
        if self.status is CommitmentStatus.OPEN:
            if self.closed_at is not None or self.closure_evidence or self.closure_reason or self.closed_by:
                raise ValueError("an open commitment carries no closure data")
        else:
            if self.closed_at is None or not self.closed_by.strip():
                raise ValueError("a closed commitment requires closed_at and closed_by")
            object.__setattr__(self, "closed_at", _aware(self.closed_at, "closed_at"))
            if self.status is CommitmentStatus.FULFILLED:
                missing = [kind for kind in self.required_evidence if not self.closure_evidence.get(kind)]
                if missing:
                    raise ValueError(f"fulfilment evidence missing for: {missing}")
            if self.status is CommitmentStatus.CANCELLED and not self.closure_reason.strip():
                raise ValueError("cancellation requires a reason")

    @property
    def is_open(self) -> bool:
        return self.status is CommitmentStatus.OPEN

    def is_overdue(self, at: datetime) -> bool:
        return self.is_open and self.due_at is not None and _aware(at, "at") > self.due_at

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            kind=self.kind.value,
            status=self.status.value,
            created_at=self.created_at.isoformat(),
            due_at=self.due_at.isoformat() if self.due_at else None,
            closed_at=self.closed_at.isoformat() if self.closed_at else None,
            subject_refs=list(self.subject_refs),
            required_evidence=list(self.required_evidence),
            closure_evidence=dict(self.closure_evidence),
        )
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Commitment":
        values = dict(payload)
        values["created_at"] = _parse(values["created_at"])
        values["due_at"] = _parse(values.get("due_at"))
        values["closed_at"] = _parse(values.get("closed_at"))
        values["subject_refs"] = tuple(values.get("subject_refs") or ())
        values["required_evidence"] = tuple(values.get("required_evidence") or ())
        return cls(**values)


def new_commitment_id() -> str:
    return f"commitment:{uuid4().hex}"


def verification_commitment_for_outcome(record, *, tenant_id: str, owner: str) -> Commitment:
    """A pending OutcomeLedger record is an open 'required verification' commitment."""
    return Commitment(
        commitment_id=f"commitment:verify:{record.outcome_id}",
        kind=CommitmentKind.REQUIRED_VERIFICATION,
        tenant_id=tenant_id,
        owner=owner,
        source_ref=record.outcome_id,
        subject_refs=tuple(item for item in (record.entity_id, record.decision_id) if item),
        description=f"verify {record.metric} outcome of {record.intervention_id}",
        created_at=record.expected_effect_window[0],
        due_at=record.verification_due_at,
        required_evidence=("outcome_resolution",),
        trace_id=record.trace_id or record.decision_id,
    )


class DbCommitmentVersion(Base):
    __tablename__ = "commitment_memory_versions"

    row_id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    commitment_id = Column(String(200), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    tenant_id = Column(String(160), nullable=False, index=True)
    kind = Column(String(64), nullable=False, index=True)
    status = Column(String(32), nullable=False, index=True)
    due_at = Column(DateTime(timezone=True), nullable=True, index=True)
    recorded_at = Column(DateTime(timezone=True), nullable=False, index=True)
    superseded_at = Column(DateTime(timezone=True), nullable=True, index=True)
    payload_json = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)

    __table_args__ = (
        UniqueConstraint("commitment_id", "version", name="uq_commitment_version"),
        Index(
            "uq_commitment_single_current",
            "commitment_id",
            unique=True,
            sqlite_where=text("superseded_at IS NULL"),
            postgresql_where=text("superseded_at IS NULL"),
        ),
    )


class CommitmentMemory:
    """Durable, append-only store of open obligations (survives sessions and restarts)."""

    def __init__(self, session_factory) -> None:
        if session_factory is None:
            raise ValueError("durable storage is required for commitment memory")
        self._session_factory = session_factory

    async def record(self, commitment: Commitment, *, recorded_at: Optional[datetime] = None) -> Commitment:
        if not commitment.is_open:
            raise CommitmentError("a new commitment must be open")
        instant = _aware(recorded_at or datetime.now(timezone.utc), "recorded_at")
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    exists = await session.scalar(
                        select(DbCommitmentVersion.row_id)
                        .where(DbCommitmentVersion.commitment_id == commitment.commitment_id)
                        .limit(1)
                    )
                    if exists is not None:
                        raise CommitmentError(f"commitment already exists: {commitment.commitment_id}")
                    session.add(self._row(commitment, version=1, recorded_at=instant))
                    await session.flush()
        except IntegrityError as exc:
            raise CommitmentConcurrencyError(f"commitment already exists: {commitment.commitment_id}") from exc
        return commitment

    async def _close(self, commitment_id: str, update, *, tenant_id: str, recorded_at: Optional[datetime]) -> Commitment:
        instant = _aware(recorded_at or datetime.now(timezone.utc), "recorded_at")
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    row = await session.scalar(
                        select(DbCommitmentVersion)
                        .where(
                            DbCommitmentVersion.commitment_id == commitment_id,
                            DbCommitmentVersion.superseded_at.is_(None),
                        )
                        .with_for_update()
                    )
                    if row is None or row.tenant_id != tenant_id:
                        raise KeyError(f"unknown commitment for tenant: {commitment_id}")
                    current = Commitment.from_payload(row.payload_json)
                    if not current.is_open:
                        raise CommitmentError(f"commitment is already {current.status.value}")
                    closed = update(current, instant)
                    row.superseded_at = instant
                    session.add(self._row(closed, version=int(row.version) + 1, recorded_at=instant))
                    await session.flush()
                    return closed
        except IntegrityError as exc:
            raise CommitmentConcurrencyError(f"commitment concurrently updated: {commitment_id}") from exc

    async def fulfil(
        self,
        commitment_id: str,
        *,
        tenant_id: str,
        evidence: Mapping[str, str],
        closed_by: str,
        recorded_at: Optional[datetime] = None,
    ) -> Commitment:
        def update(current: Commitment, instant: datetime) -> Commitment:
            return replace(current, status=CommitmentStatus.FULFILLED, closed_at=instant,
                           closure_evidence=dict(evidence), closed_by=closed_by)

        return await self._close(commitment_id, update, tenant_id=tenant_id, recorded_at=recorded_at)

    async def cancel(
        self,
        commitment_id: str,
        *,
        tenant_id: str,
        reason: str,
        closed_by: str,
        recorded_at: Optional[datetime] = None,
    ) -> Commitment:
        def update(current: Commitment, instant: datetime) -> Commitment:
            return replace(current, status=CommitmentStatus.CANCELLED, closed_at=instant,
                           closure_reason=reason, closed_by=closed_by)

        return await self._close(commitment_id, update, tenant_id=tenant_id, recorded_at=recorded_at)

    async def open_commitments(self, *, tenant_id: str) -> tuple[Commitment, ...]:
        """What NarrativeSelf/Planning reload after a restart: all open obligations of a tenant."""
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(DbCommitmentVersion)
                    .where(
                        DbCommitmentVersion.tenant_id == tenant_id,
                        DbCommitmentVersion.superseded_at.is_(None),
                        DbCommitmentVersion.status == CommitmentStatus.OPEN.value,
                    )
                    .order_by(DbCommitmentVersion.due_at.asc(), DbCommitmentVersion.commitment_id.asc())
                )
            ).all()
            return tuple(Commitment.from_payload(row.payload_json) for row in rows)

    async def overdue(self, *, tenant_id: str, at: datetime) -> tuple[Commitment, ...]:
        return tuple(item for item in await self.open_commitments(tenant_id=tenant_id) if item.is_overdue(at))

    async def history(self, commitment_id: str) -> tuple[Commitment, ...]:
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(DbCommitmentVersion)
                    .where(DbCommitmentVersion.commitment_id == commitment_id)
                    .order_by(DbCommitmentVersion.version.asc())
                )
            ).all()
            return tuple(Commitment.from_payload(row.payload_json) for row in rows)

    @staticmethod
    def _row(commitment: Commitment, *, version: int, recorded_at: datetime) -> DbCommitmentVersion:
        return DbCommitmentVersion(
            commitment_id=commitment.commitment_id,
            version=version,
            tenant_id=commitment.tenant_id,
            kind=commitment.kind.value,
            status=commitment.status.value,
            due_at=commitment.due_at,
            recorded_at=recorded_at,
            superseded_at=None,
            payload_json=commitment.to_payload(),
        )


# --------------------------------------------------------------------------------------
# CCIA §4.4: port for NarrativeSelf / Planning start-up
# --------------------------------------------------------------------------------------


class OpenCommitmentsPort(Protocol):
    """What a consumer needs from commitment memory: the open obligations of a tenant."""

    async def open_commitments(self, *, tenant_id: str) -> tuple[Commitment, ...]: ...


class CommitmentConsumer(Protocol):
    """NarrativeSelf / Planning (or any organ that must not lose obligations across restarts)."""

    def on_open_commitments(self, obligations: "OpenObligations") -> Any: ...


@dataclass(frozen=True)
class OpenObligations:
    tenant_id: str
    as_of: datetime
    open: tuple[Commitment, ...]
    overdue: tuple[Commitment, ...]

    def of_kind(self, kind: CommitmentKind) -> tuple[Commitment, ...]:
        kind = CommitmentKind(kind)
        return tuple(item for item in self.open if item.kind is kind)

    def refs(self) -> tuple[str, ...]:
        return tuple(item.commitment_id for item in self.open)


async def load_open_commitments(port: OpenCommitmentsPort, *, tenant_id: str, at: datetime) -> OpenObligations:
    """``open_commitments`` for the start of Planning / NarrativeSelf: every open obligation of the tenant
    plus the overdue subset at ``at``. Storage failures propagate (the caller must not start "clean")."""
    if not str(tenant_id or "").strip():
        raise ValueError("tenant_id is required")
    instant = _aware(at, "at")
    items = tuple(await port.open_commitments(tenant_id=tenant_id))
    if any(not isinstance(item, Commitment) or not item.is_open or item.tenant_id != tenant_id for item in items):
        raise CommitmentError("commitment port returned a closed or foreign-tenant commitment")
    return OpenObligations(tenant_id, instant, items, tuple(item for item in items if item.is_overdue(instant)))


class CommitmentStartup:
    """Start-up hook: reloads open commitments and hands them to every registered consumer
    (NarrativeSelf, PlanningCore, ...). A consumer that fails to take them over is reported by raising
    after all consumers were notified — an organ must not silently start without its obligations."""

    def __init__(self, port: OpenCommitmentsPort, consumers: Sequence[CommitmentConsumer] = ()) -> None:
        if port is None:
            raise ValueError("commitment port is required")
        self.port = port
        self.consumers = list(consumers)
        self.last: Optional[OpenObligations] = None

    def register(self, consumer: CommitmentConsumer) -> None:
        self.consumers.append(consumer)

    async def restore(self, *, tenant_id: str, at: datetime) -> OpenObligations:
        obligations = await load_open_commitments(self.port, tenant_id=tenant_id, at=at)
        errors = []
        for consumer in self.consumers:
            try:
                out = consumer.on_open_commitments(obligations)
                if hasattr(out, "__await__"):
                    await out
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(consumer).__name__}:{type(exc).__name__}:{exc}")
        self.last = obligations
        if errors:
            raise CommitmentError(f"open commitments not taken over: {errors}")
        return obligations


__all__ = [
    "Commitment",
    "CommitmentConcurrencyError",
    "CommitmentError",
    "CommitmentKind",
    "CommitmentConsumer",
    "CommitmentMemory",
    "CommitmentStartup",
    "CommitmentStatus",
    "OpenCommitmentsPort",
    "OpenObligations",
    "load_open_commitments",
    "DbCommitmentVersion",
    "new_commitment_id",
    "verification_commitment_for_outcome",
]

