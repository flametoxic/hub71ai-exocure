"""Контракты Sovereign Runtime (ТЗ «EXO Sovereign Runtime, Execution Control & Security», §3.2, §3.4).

Контракт действия — единственная форма, в которой система вообще может попросить изменить мир.
Executable(u) = I ∧ C ∧ F ∧ P ∧ A ∧ H (§3.2): личность цели разрешена, устройство умеет действие, факты
свежие, физические пределы соблюдены, полномочия/политика разрешают, здоровье среды/устройства достаточно.
verified(u) = 1[ack ∧ postcondition observed ∧ no safety violation] (§3.4): подтверждение команды — ещё не
физический исход.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional

from ..common.epistemic import deep_freeze
from ..world_model.reality_formulas import FormulaReference

DOCUMENT = "EXO-sovereign-runtime-execution-security-spec"


def ref(formula_id: str, page: int, expression: str) -> FormulaReference:
    return FormulaReference(formula_id, DOCUMENT, page, expression)


F_SOVEREIGNTY = ref("SOV-1-INVARIANT", 1, "critical action availability does not require external network availability")
F_EXECUTABLE = ref("SOV-3.2-EXECUTABLE", 2, "Executable(u) = I ∧ C ∧ F ∧ P ∧ A ∧ H")
F_IDEMPOTENCY = ref("SOV-3.3-IDEMPOTENCY", 3, "execute(k,u) = execute(k,u) for all retries with same idempotency key k")
F_VERIFIED = ref("SOV-3.4-VERIFIED", 3, "verified(u) = 1[ack ∧ postcondition observed ∧ no safety violation]")
F_HARD_GUARD = ref("SOV-5-HARD-GUARD", 4, "x_lo <= x <= x_hi, u_lo <= u <= u_hi, |du/dt| <= r_u")
F_TRIP = ref("SOV-5-TRIP", 4, "trip = 1[x not in X_hard ∨ u not in U_hard ∨ watchdog timeout ∨ command integrity failure]")
F_LATENCY = ref("SOV-6-LATENCY", 4, "L_e2e = L_sensor + L_edge_validation + L_policy + L_network + L_controller")
F_REPLAY = ref("SOV-7-REPLAY", 4, "accept(m) = 1[sig valid ∧ nonce fresh ∧ seq > seq_last ∧ time within window]")


class SovereignError(ValueError):
    """Нарушен контракт исполнения (fail-closed)."""


class Stage(str, Enum):
    INTENT = "1_action_intent"
    CAPABILITY = "2_capability_resolution"
    PRECONDITIONS = "3_preconditions_freshness"
    ENVELOPE = "4_physical_envelope"
    AUTHORITY = "5_policy_consent_authority"
    SIMULATION = "6_simulation_impact"
    APPROVAL = "7_approval_autonomy_tier"
    EXECUTION = "8_idempotent_protocol_execution"
    VERIFICATION = "9_verification_journal_safe_state"


class ExecutionStatus(str, Enum):
    BLOCKED = "blocked"                 # не прошло шаги 1–7 — команды не было
    SHADOW = "shadow"                   # всё прошло, но уровень автономии — только тень: команды не было
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTED_UNVERIFIED = "executed_unverified"
    VERIFIED = "verified"
    FAILED_SAFE_STATE = "failed_safe_state"      # не подтвердилось → компенсация/безопасное состояние
    DEGRADED_BLOCKED = "degraded_blocked"        # деградация среды: физическая запись заблокирована


def canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_default)


def _default(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, (set, frozenset)):
        return sorted(v, key=str)
    if isinstance(v, Mapping):
        return dict(v)
    if isinstance(v, tuple):
        return list(v)
    return str(v)


def digest(payload: Any) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def aware(name: str, v: Any) -> datetime:
    if not isinstance(v, datetime) or v.tzinfo is None or v.utcoffset() is None:
        raise SovereignError(f"{name} must be a timezone-aware datetime")
    return v.astimezone(timezone.utc)


def text(name: str, v: Any) -> str:
    if not isinstance(v, str) or not v.strip():
        raise SovereignError(f"{name} is required")
    return v.strip()


def finite(name: str, v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise SovereignError(f"{name} must be a finite number")
    return float(v)


@dataclass(frozen=True)
class ExpectedEffect:
    """Наблюдаемое постусловие (§3.4): точка телеметрии, направление/интервал, окно проверки."""

    point: str
    lower: Optional[float]
    upper: Optional[float]
    window_s: float

    def __post_init__(self) -> None:
        text("point", self.point)
        if self.lower is None and self.upper is None:
            raise SovereignError("an expected effect needs a lower and/or upper bound")
        if finite("window_s", self.window_s) <= 0:
            raise SovereignError("verification window must be positive")

    def holds(self, value: float) -> bool:
        v = float(value)
        return math.isfinite(v) and (self.lower is None or v >= self.lower) and (self.upper is None or v <= self.upper)


@dataclass(frozen=True)
class ActionContract:
    """§3.2 — все 17 полей обязательны (кроме approval_ref для уровней без одобрения человека)."""

    action_id: str
    trace_id: str
    idempotency_key: str
    target_entity_id: str
    action_type: str
    parameters: Mapping[str, float]           # точка записи → значение
    capability_version: str
    preconditions: tuple[str, ...]
    physical_limits: Mapping[str, tuple[Optional[float], Optional[float]]]
    expected_effect: tuple[ExpectedEffect, ...]
    expiry_at: datetime
    authority_tier: str
    policy_refs: tuple[str, ...]
    approval_ref: Optional[str]
    safe_state: Mapping[str, float]
    compensation_plan: tuple[str, ...]
    tenant: str
    site: str
    risk_tier: str
    action_class: str
    contract_digest: str = ""

    def __post_init__(self) -> None:
        for n in ("action_id", "trace_id", "idempotency_key", "target_entity_id", "action_type", "capability_version",
                  "authority_tier", "tenant", "site", "risk_tier", "action_class"):
            object.__setattr__(self, n, text(n, getattr(self, n)))
        object.__setattr__(self, "expiry_at", aware("expiry_at", self.expiry_at))
        if not self.parameters:
            raise SovereignError("an action needs typed parameters")
        if not self.expected_effect:
            raise SovereignError("an action without an observable expected effect cannot be verified (§3.4)")
        if not self.safe_state:
            raise SovereignError("an action needs a declared safe state (§3.2)")
        if not self.policy_refs:
            raise SovereignError("an action needs policy references")
        for k, v in dict(self.parameters).items():
            finite(f"parameter {k}", v)
        object.__setattr__(self, "parameters", deep_freeze({str(k): float(v) for k, v in dict(self.parameters).items()}))
        object.__setattr__(self, "safe_state", deep_freeze({str(k): finite(k, v) for k, v in dict(self.safe_state).items()}))
        object.__setattr__(self, "physical_limits", deep_freeze(dict(self.physical_limits)))
        for n in ("preconditions", "policy_refs", "compensation_plan", "expected_effect"):
            object.__setattr__(self, n, tuple(getattr(self, n)))
        body = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "contract_digest"}
        d = digest(body)
        if self.contract_digest and self.contract_digest != d:
            raise SovereignError("contract digest mismatch: the action contract was altered")
        object.__setattr__(self, "contract_digest", d)


@dataclass
class ExecutionRecord:
    action_id: str
    idempotency_key: str
    status: ExecutionStatus
    stages: list = field(default_factory=list)          # (stage, passed, reasons)
    command_ref: Optional[str] = None
    ack: Optional[bool] = None
    verified: Optional[bool] = None
    observed: dict = field(default_factory=dict)
    safe_state_applied: bool = False
    real_action: bool = False
    contract_digest: str = ""
    action_class: str = ""
    tier: Optional[str] = None
    executed_at: Optional[str] = None
    in_doubt: bool = False
    timings_s: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def stage(self, s: Stage, ok: bool, *reasons: str) -> bool:
        self.stages.append((s.value, bool(ok), tuple(reasons)))
        return ok

    def to_dict(self) -> dict:
        return json.loads(canonical({k: getattr(self, k) for k in self.__dataclass_fields__}))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ExecutionRecord":
        kw = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        kw["status"] = ExecutionStatus(kw["status"])
        kw["stages"] = [tuple(x[:2]) + (tuple(x[2]),) for x in kw.get("stages", [])]
        return cls(**kw)

    @property
    def final(self) -> bool:
        """Команда уже уходила на устройство: повтор с тем же ключом возвращает этот результат (§3.3)."""
        return self.status in (ExecutionStatus.EXECUTED_UNVERIFIED, ExecutionStatus.VERIFIED,
                               ExecutionStatus.FAILED_SAFE_STATE) or self.real_action


__all__ = ["ActionContract", "DOCUMENT", "ExecutionRecord", "ExecutionStatus", "ExpectedEffect", "F_EXECUTABLE",
           "F_HARD_GUARD", "F_IDEMPOTENCY", "F_LATENCY", "F_REPLAY", "F_SOVEREIGNTY", "F_TRIP", "F_VERIFIED",
           "SovereignError", "Stage", "aware", "canonical", "digest", "finite", "text"]
