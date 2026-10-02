"""Граница протоколов, жёсткий предохранитель и компиляция политик на край (§4, §5, §6; мастер-ТЗ D4).

§4 Адаптеры типизированы и ограничены: чтение/запись по объявленным точкам с типом, единицей и пределами,
   список поддерживаемых команд, лимит частоты, идентичность устройства, семантика подтверждения, безопасное
   состояние, пульс. Интерфейса execute(command_string) нет.
§5 Предохранитель: x_lo ≤ x ≤ x_hi, u_lo ≤ u ≤ u_hi, |du/dt| ≤ r_u;
   trip = 1[x ∉ X_hard ∨ u ∉ U_hard ∨ таймаут сторожевого таймера ∨ нарушение целостности команды].
   Адаптер принимает запись ТОЛЬКО с разрешением, выданным предохранителем (обойти его нельзя: разрешение
   проверяется по журналу выдачи). Программный предохранитель — не замена сертифицированного PLC/SIS: реальный
   контур — порт HardwareInterlock (сухой контакт / safety PLC), и его «запрет» сильнее любого решения ИИ.
§6 Политики: DSL → проверка типов и размерностей → формальная проверка совместности → хэш артефакта →
   подпись → развёртывание на край (только с верной подписью) → аттестация и откат.
   L_e2e = L_sensor + L_edge_validation + L_policy + L_network + L_controller — бюджет p99 на класс действий.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Mapping, Optional, Protocol

import numpy as np

from .contracts import SovereignError, aware, digest, finite, text
from .trust import sign_artifact, verify_artifact


class ProtocolKind(str, Enum):
    BACNET = "bacnet"
    MODBUS = "modbus"
    MQTT = "mqtt"
    REST_VENDOR = "rest_vendor"
    V2X = "v2x"


@dataclass(frozen=True)
class PointSpec:
    point: str
    unit: str
    lower: float
    upper: float
    writable: bool

    def __post_init__(self) -> None:
        text("point", self.point)
        text("unit", self.unit)
        if not (finite("lower", self.lower) <= finite("upper", self.upper)):
            raise SovereignError(f"{self.point}: lower bound above upper bound")


@dataclass(frozen=True)
class AdapterSpec:
    adapter_id: str
    protocol: ProtocolKind
    device_id: str
    certificate_fingerprint: str
    points: tuple[PointSpec, ...]
    supported_commands: tuple[str, ...]
    max_writes_per_minute: int
    ack_semantics: str                   # что значит «подтверждено» у этого протокола
    safe_state: Mapping[str, float]
    heartbeat_timeout_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol", ProtocolKind(self.protocol))
        for n in ("adapter_id", "device_id", "certificate_fingerprint", "ack_semantics"):
            text(n, getattr(self, n))
        if self.max_writes_per_minute < 1 or self.heartbeat_timeout_s <= 0:
            raise SovereignError("rate limit and heartbeat timeout must be positive")
        names = [p.point for p in self.points]
        if len(set(names)) != len(names):
            raise SovereignError("duplicate point")
        bad = [k for k in self.safe_state if k not in names]
        if bad:
            raise SovereignError(f"safe state uses unknown points {bad}")

    def point(self, name: str) -> PointSpec:
        for p in self.points:
            if p.point == name:
                return p
        raise SovereignError(f"{self.adapter_id}: unknown point {name!r}")


class Transport(Protocol):
    """Реальный протокольный стек (BACnet/Modbus/…) — ваш код или вендор. Здесь только интерфейс."""

    def write(self, point: str, value: float) -> bool: ...

    def read(self, point: str) -> float: ...


@dataclass(frozen=True)
class GuardPermit:
    permit_id: str
    adapter_id: str
    point: str
    value: float
    issued_at: datetime
    command_digest: str
    safe_state: bool = False


@dataclass(frozen=True)
class Ack:
    ok: bool
    command_ref: str
    semantics: str
    at: datetime


class TypedAdapter:
    """Типизированный адаптер. Только write(point, value, unit, command, permit) и read(point)."""

    def __init__(self, spec: AdapterSpec, transport: Transport, guard: "EdgeHardGuard") -> None:
        self.spec, self._transport, self._guard = spec, transport, guard
        self._writes: deque = deque()
        self.last_heartbeat: Optional[datetime] = None

    def heartbeat(self, at: datetime) -> None:
        self.last_heartbeat = aware("at", at)

    def healthy(self, now: datetime) -> bool:
        return self.last_heartbeat is not None and \
            aware("now", now) - self.last_heartbeat <= timedelta(seconds=self.spec.heartbeat_timeout_s)

    def read(self, point: str) -> float:
        self.spec.point(point)
        return float(self._transport.read(point))

    def write(self, point: str, value: float, *, unit: str, command: str, permit: GuardPermit, now: datetime,
              safe_state: bool = False) -> Ack:
        """safe_state=True — запись безопасного состояния: без проверки команды и лимита частоты, но только
        с разрешением предохранителя, выданным именно на безопасное состояние."""
        spec = self.spec.point(point)
        if not spec.writable:
            raise SovereignError(f"{point} is read-only on {self.spec.adapter_id}")
        if isinstance(permit, GuardPermit) and permit.safe_state != safe_state:
            raise SovereignError("permit kind does not match the write kind (safe state vs command)")
        if not safe_state and command not in self.spec.supported_commands:
            raise SovereignError(f"command {command!r} is not supported by {self.spec.adapter_id}")
        if unit != spec.unit:
            raise SovereignError(f"unit mismatch for {point}: {unit} != {spec.unit}")
        v = finite("value", value)
        if not spec.lower <= v <= spec.upper:
            raise SovereignError(f"{point}={v} outside device range [{spec.lower}, {spec.upper}]")
        if not self._guard.honours(permit, adapter_id=self.spec.adapter_id, point=point, value=v):
            raise SovereignError("write refused: no valid permit from the edge hard guard")
        now = aware("now", now)
        while self._writes and now - self._writes[0] > timedelta(minutes=1):
            self._writes.popleft()
        if not safe_state and len(self._writes) >= self.spec.max_writes_per_minute:
            raise SovereignError(f"rate limit of {self.spec.adapter_id} exceeded")
        ok = bool(self._transport.write(point, v))
        self._writes.append(now)
        return Ack(ok, f"{self.spec.adapter_id}:{point}:{permit.permit_id}", self.spec.ack_semantics, now)


class HardwareInterlock(Protocol):
    """Независимый аппаратный контур (сухой контакт / сертифицированный safety PLC). True — запрещает."""

    def inhibits(self, point: str, value: float) -> bool: ...


@dataclass(frozen=True)
class HardEnvelope:
    state_bounds: Mapping[str, tuple[float, float]]      # x_lo ≤ x ≤ x_hi
    control_bounds: Mapping[str, tuple[float, float]]    # u_lo ≤ u ≤ u_hi
    rate_limits: Mapping[str, float]                     # |du/dt| ≤ r_u (единиц в секунду)
    watchdog_timeout_s: float


class EdgeHardGuard:
    """Последний программный рубеж перед адаптером. Решение «trip» ИИ отменить не может."""

    def __init__(self, envelope: HardEnvelope, *, command_key: bytes, interlock: Optional[HardwareInterlock] = None) -> None:
        if envelope.watchdog_timeout_s <= 0:
            raise SovereignError("watchdog timeout must be positive")
        self.envelope, self._key, self.interlock = envelope, command_key, interlock
        self._issued: dict[str, GuardPermit] = {}
        self._last_u: dict[str, tuple[float, datetime]] = {}
        self.last_watchdog: Optional[datetime] = None
        self.policy: Optional[CompiledPolicy] = None
        self.trips: list[dict] = []

    def kick(self, at: datetime) -> None:
        self.last_watchdog = aware("at", at)

    def load_policy(self, policy: Optional["CompiledPolicy"]) -> None:
        self.policy = policy

    def sign_command(self, adapter_id: str, point: str, value: float, action_id: str) -> str:
        return sign_artifact({"adapter": adapter_id, "point": point, "value": value, "action": action_id}, self._key)

    def check(self, *, adapter_id: str, point: str, value: float, state: Mapping[str, float], action_id: str,
              action_class: str, command_signature: str, now: datetime,
              safe_state: bool = False) -> tuple[Optional[GuardPermit], tuple[str, ...]]:
        """safe_state=True — переход в объявленное (OEM) безопасное состояние: состояние x, политика класса,
        темп и сторожевой таймер его не блокируют (именно в этих случаях оно и нужно), но U_hard, целостность
        команды и аппаратная блокировка проверяются всегда."""
        now = aware("now", now)
        reasons = []
        if not safe_state:
            for var, (lo, hi) in self.envelope.state_bounds.items():
                x = state.get(var)
                if x is None or not math.isfinite(float(x)) or not lo <= float(x) <= hi:
                    reasons.append(f"x_outside_hard:{var}")
        lo_hi = self.envelope.control_bounds.get(point)
        if lo_hi is None:
            reasons.append(f"no_hard_bounds_for:{point}")
        elif not lo_hi[0] <= value <= lo_hi[1]:
            reasons.append(f"u_outside_hard:{point}")
        if not safe_state:
            if self.policy is not None:
                reasons += list(self.policy.violations(action_class, point, value))
            else:
                reasons.append("no_signed_edge_policy_loaded")
            r = self.envelope.rate_limits.get(point)
            prev = self._last_u.get(point)
            if r is not None and prev is not None:
                dt = max((now - prev[1]).total_seconds(), 1e-9)
                if abs(value - prev[0]) / dt > r:
                    reasons.append(f"rate_limit:{point}")
            if self.last_watchdog is None or \
                    (now - self.last_watchdog).total_seconds() > self.envelope.watchdog_timeout_s:
                reasons.append("watchdog_timeout")
        if not verify_artifact({"adapter": adapter_id, "point": point, "value": value, "action": action_id},
                               command_signature, self._key):
            reasons.append("command_integrity_failure")
        if self.interlock is not None and self.interlock.inhibits(point, value):
            reasons.append("hardware_interlock_inhibits")
        if reasons:
            self.trips.append({"at": now.isoformat(), "point": point, "value": value, "reasons": reasons})
            return None, tuple(reasons)
        self._n = getattr(self, "_n", 0) + 1
        permit = GuardPermit(f"permit:{self._n}:{digest([adapter_id, point, value, action_id])[:10]}",
                             adapter_id, point, float(value), now, digest([adapter_id, point, value, action_id]),
                             bool(safe_state))
        self._issued[permit.permit_id] = permit
        self._last_u[point] = (float(value), now)
        return permit, ()

    def honours(self, permit: GuardPermit, *, adapter_id: str, point: str, value: float) -> bool:
        issued = self._issued.pop(permit.permit_id, None) if isinstance(permit, GuardPermit) else None
        return issued is permit and issued.adapter_id == adapter_id and issued.point == point and issued.value == value


# ------------------------------------------------------------------------------------ компиляция политик
@dataclass(frozen=True)
class CompiledPolicy:
    version: str
    action_class: str
    scope: tuple[str, ...]
    table: Mapping[str, tuple[Optional[float], Optional[float]]]     # точка → [lo, hi] (решающая таблица)
    artifact_hash: str

    def violations(self, action_class: str, point: str, value: float) -> tuple[str, ...]:
        if action_class != self.action_class:
            return (f"edge_policy_not_for_action_class:{action_class}",)
        if point not in self.table:
            return (f"edge_policy_has_no_rule_for:{point}",)
        lo, hi = self.table[point]
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            return (f"edge_policy_bound:{point}",)
        return ()


_OPS = ("<=", ">=", "==")


def compile_policy(dsl: Mapping, *, point_units: Mapping[str, str], version: str, key: bytes) -> tuple[CompiledPolicy, str]:
    """DSL {action_class, scope, rules: [{point, op, value, unit}]} → проверенный, подписанный артефакт."""
    ac, scope = text("action_class", dsl.get("action_class")), tuple(dsl.get("scope") or ())
    if not scope:
        raise SovereignError("policy scope (site/asset) is required")
    table: dict[str, list] = {}
    for r in dsl.get("rules") or ():
        p, op, unit = text("point", r.get("point")), r.get("op"), text("unit", r.get("unit"))
        if op not in _OPS:
            raise SovereignError(f"unknown operator {op!r}")
        if point_units.get(p) != unit:
            raise SovereignError(f"dimensional check failed for {p}: rule unit {unit} vs point unit {point_units.get(p)}")
        v = finite("value", r.get("value"))
        lo, hi = table.setdefault(p, [None, None])
        if op in (">=", "=="):
            lo = v if lo is None else max(lo, v)
        if op in ("<=", "=="):
            hi = v if hi is None else min(hi, v)
        table[p] = [lo, hi]
    if not table:
        raise SovereignError("a policy needs at least one rule")
    for p, (lo, hi) in table.items():
        if lo is not None and hi is not None and lo > hi:
            raise SovereignError(f"formal check failed: constraints on {p} are unsatisfiable ({lo} > {hi})")
    body = {"version": text("version", version), "action_class": ac, "scope": list(scope),
            "table": {p: list(v) for p, v in sorted(table.items())}}
    h = digest(body)
    pol = CompiledPolicy(version, ac, scope, {p: tuple(v) for p, v in table.items()}, h)
    return pol, sign_artifact(body, key)


class EdgePolicyStore:
    """Развёртывание только подписанных артефактов; аттестация (хэш активной версии) и откат.

    Локальный кэш (§1 «cached/signed local policy»): при холодном старте без сети политика читается с диска,
    и каждая версия заново проверяется по подписи; невалидная — отказ, край остаётся без политики
    (остаётся только аппаратно-безопасное поведение, §9)."""

    def __init__(self, *, governance_key: bytes, cache_path: Optional[str] = None) -> None:
        self._key = governance_key
        self._versions: list[tuple[CompiledPolicy, str]] = []
        self.active: Optional[CompiledPolicy] = None
        self.cache_path = cache_path
        self.load_error: Optional[str] = None
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as fh:
                    for item in json.load(fh):
                        pol = CompiledPolicy(item["version"], item["action_class"], tuple(item["scope"]),
                                             {p: tuple(v) for p, v in item["table"].items()}, item["artifact_hash"])
                        self._accept(pol, item["signature"])
            except (OSError, ValueError, KeyError, TypeError, SovereignError) as exc:
                self._versions, self.active = [], None
                self.load_error = f"policy cache rejected: {exc}"

    @staticmethod
    def _body(policy: CompiledPolicy) -> dict:
        return {"version": policy.version, "action_class": policy.action_class, "scope": list(policy.scope),
                "table": {p: list(v) for p, v in sorted(policy.table.items())}}

    def _accept(self, policy: CompiledPolicy, signature: str) -> None:
        body = self._body(policy)
        if digest(body) != policy.artifact_hash or not verify_artifact(body, signature, self._key):
            raise SovereignError("policy signature verification failed: artifact not deployed")
        self._versions.append((policy, signature))
        self.active = policy

    def _persist(self) -> None:
        if not self.cache_path:
            return
        data = [dict(self._body(p), artifact_hash=p.artifact_hash, signature=s) for p, s in self._versions]
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.cache_path)

    def deploy(self, policy: CompiledPolicy, signature: str) -> CompiledPolicy:
        self._accept(policy, signature)
        self._persist()
        return policy

    def attest(self) -> Optional[str]:
        return None if self.active is None else self.active.artifact_hash

    def rollback(self) -> Optional[CompiledPolicy]:
        if len(self._versions) < 2:
            raise SovereignError("no previous policy version to roll back to")
        self._versions.pop()
        self.active = self._versions[-1][0]
        self._persist()
        return self.active


# ------------------------------------------------------------------------------------ задержка
@dataclass
class LatencyBudget:
    """Бюджет p99 по классу действий: измеряется, а не обещается («1 мс» без замера — не требование)."""

    action_class: str
    p99_budget_s: float
    samples: list = field(default_factory=list)

    def record(self, *, sensor: float, edge_validation: float, policy: float, network: float, controller: float) -> float:
        total = sum(finite(n, v) for n, v in (("sensor", sensor), ("edge_validation", edge_validation),
                                              ("policy", policy), ("network", network), ("controller", controller)))
        self.samples.append(total)
        return total

    def p99(self) -> Optional[float]:
        return float(np.quantile(self.samples, 0.99)) if self.samples else None

    def within(self) -> Optional[bool]:
        p = self.p99()
        return None if p is None else p <= self.p99_budget_s


def measure(fn: Callable[[], object]) -> tuple[object, float]:
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


__all__ = ["Ack", "AdapterSpec", "CompiledPolicy", "EdgeHardGuard", "EdgePolicyStore", "GuardPermit", "HardEnvelope",
           "HardwareInterlock", "LatencyBudget", "PointSpec", "ProtocolKind", "Transport", "TypedAdapter",
           "compile_policy", "measure"]
