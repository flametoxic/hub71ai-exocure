"""Execution Control Pipeline (ТЗ Sovereign §3.1–§3.4, §8, §9; мастер-ТЗ E4).

1 ActionIntent → 2 capability → 3 preconditions/freshness → 4 physical envelope → 5 policy/consent/authority →
6 simulation/impact → 7 approval/autonomy tier → 8 idempotent protocol execution → 9 verification, journal,
compensation or safe state.

Executable(u) = I ∧ C ∧ F ∧ P ∧ A ∧ H. Этот слой — ворота перед исполнением: он не заменяет ваш
core/execution и core/safety, а подключает их через порты (SafetyPort, Transport адаптера). Физическую
запись он выполняет только через TypedAdapter с разрешением EdgeHardGuard; аппаратная блокировка сильнее.

Журнал — упреждающая запись: «попытка команды» пишется ДО записи в устройство. Если журнал недоступен —
команды нет (DEGRADED_BLOCKED). После перезапуска попытка без подтверждения считается «исполнено, не
проверено (in doubt)» и НИКОГДА не повторяется автоматически.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping, Optional, Protocol

from .assurance import AssuranceRegistry
from .autonomy import AutonomyGate, AutonomyKey, AutonomyTier, EntropyMonitor, SafetySignal
from .contracts import (ActionContract, ExecutionRecord, ExecutionStatus, SovereignError, Stage, aware, finite,
                        text)
from .edge import EdgeHardGuard, EdgePolicyStore, TypedAdapter
from .sovereign import RuntimeStatus, SovereignState, runtime_status
from .trust import DeviceTrust, ExecutionJournal, SignedMessage, verify_artifact


# ------------------------------------------------------------------------------------ порты
@dataclass(frozen=True)
class FactStatus:
    holds: bool
    observed_at: datetime
    healthy: bool


class IdentityPort(Protocol):
    """I: цель действия разрешена в канонический объект (entity resolution World Model)."""

    def resolved(self, entity_id: str) -> bool: ...


class FactPort(Protocol):
    """F: предусловия — именованные факты локального состояния мира."""

    def fact(self, name: str) -> Optional[FactStatus]: ...


class SafetyPort(Protocol):
    """A: ваш SafetyService / политика / согласие. (разрешено?, причины)."""

    def evaluate(self, contract: ActionContract) -> tuple[bool, tuple[str, ...]]: ...


@dataclass(frozen=True)
class Approval:
    approval_id: str
    approver_id: str
    contract_digest: str
    signed_at: datetime
    signature: str

    def body(self) -> dict:
        return {"approval_id": self.approval_id, "approver": self.approver_id, "contract_digest": self.contract_digest,
                "signed_at": self.signed_at}


@dataclass(frozen=True)
class RuntimePolicy:
    policy_version: str
    max_fact_age_s: float
    max_telemetry_age_s: float

    def __post_init__(self) -> None:
        text("policy_version", self.policy_version)
        if finite("max_fact_age_s", self.max_fact_age_s) <= 0 or finite("max_telemetry_age_s", self.max_telemetry_age_s) <= 0:
            raise SovereignError("freshness limits must be positive (site policy)")


@dataclass
class _Binding:
    adapter: TypedAdapter
    capability_version: str
    action_types: tuple[str, ...]


# ------------------------------------------------------------------------------------ pipeline
class ExecutionControlPipeline:
    def __init__(self, *, policy: RuntimePolicy, journal: ExecutionJournal, device_trust: DeviceTrust,
                 guard: EdgeHardGuard, policy_store: EdgePolicyStore, assurance: AssuranceRegistry,
                 autonomy: AutonomyGate, entropy: EntropyMonitor, sovereign: SovereignState,
                 identity: IdentityPort, facts: FactPort, safety: SafetyPort,
                 approver_keys: Mapping[str, bytes]) -> None:
        self.policy, self.journal, self.trust, self.guard = policy, journal, device_trust, guard
        self.policy_store, self.assurance, self.autonomy, self.entropy = policy_store, assurance, autonomy, entropy
        self.sovereign, self.identity, self.facts, self.safety = sovereign, identity, facts, safety
        self._approvers = dict(approver_keys)
        self._bindings: dict[str, _Binding] = {}
        self._sensors: dict[str, str] = {}                     # точка телеметрии → устройство-источник
        self._telemetry: dict[str, list[tuple[datetime, float]]] = {}
        self._records: dict[str, ExecutionRecord] = {}
        self._contracts: dict[str, ActionContract] = {}
        self.operator_notifications: list[dict] = []
        self._restore_from_journal()

    # ---------------------------------------------------------------- регистрация
    def register_capability(self, entity_id: str, adapter: TypedAdapter, *, capability_version: str,
                            action_types: tuple[str, ...]) -> None:
        self._bindings[text("entity_id", entity_id)] = _Binding(adapter, text("capability_version", capability_version),
                                                                 tuple(action_types))
        for p in adapter.spec.points:
            self._sensors.setdefault(p.point, adapter.spec.device_id)

    def register_sensor(self, point: str, device_id: str) -> None:
        self._sensors[text("point", point)] = text("device_id", device_id)

    # ---------------------------------------------------------------- телеметрия (data plane, локально)
    def ingest(self, msg: SignedMessage, *, now: datetime) -> tuple[bool, tuple[str, ...]]:
        ok, reasons = self.trust.accept(msg, now=now)
        if not ok:
            return False, reasons
        try:
            point = text("point", msg.payload.get("point"))
            value = finite("value", msg.payload.get("value"))
            at = aware("observed_at", datetime.fromisoformat(str(msg.payload.get("observed_at"))))
        except (SovereignError, ValueError) as exc:
            return False, (f"malformed_telemetry:{exc}",)
        if self._sensors.get(point) != msg.device_id:
            return False, (f"device_not_source_of_point:{point}",)
        if at > aware("now", now) + self.trust.max_skew:
            return False, ("observation_in_future",)
        self._telemetry.setdefault(point, []).append((at, value))
        return True, ()

    def latest(self, point: str, *, now: datetime) -> Optional[tuple[datetime, float]]:
        s = self._telemetry.get(point)
        if not s:
            return None
        _, (at, v) = max(enumerate(s), key=lambda iv: (iv[1][0], iv[0]))   # при равном времени — последнее принятое
        return (at, v) if (now - at).total_seconds() <= self.policy.max_telemetry_age_s else None

    def _state(self, now: datetime) -> tuple[dict[str, float], Optional[float]]:
        state, ages = {}, []
        for var in self.guard.envelope.state_bounds:
            lv = self.latest(var, now=now)
            if lv is not None:
                state[var] = lv[1]
                ages.append((now - lv[0]).total_seconds())
            else:
                ages.append(float("inf"))
        return state, (max(ages) if ages else None)

    # ---------------------------------------------------------------- журнал
    def _j(self, kind: str, payload: dict, now: datetime) -> bool:
        try:
            self.journal.append(kind, payload, at=now)
            return True
        except SovereignError:
            return False

    def _save(self, rec: ExecutionRecord, now: datetime) -> bool:
        self._records[rec.idempotency_key] = rec
        return self._j("record", rec.to_dict(), now)

    def _restore_from_journal(self) -> None:
        attempts: dict[str, dict] = {}
        for e in self.journal.entries():
            p = e["payload"]
            if e["kind"] == "record":
                self._records[p["idempotency_key"]] = ExecutionRecord.from_dict(p)
                attempts.pop(p["idempotency_key"], None)
            elif e["kind"] == "command_attempt" and p.get("tag") != ":safe_mode":
                attempts[p["idempotency_key"]] = p
        for key, p in attempts.items():   # попытка записана, итог — нет: сбой между командой и журналом
            rec = self._records.get(key) or ExecutionRecord(p["action_id"], key, ExecutionStatus.EXECUTED_UNVERIFIED,
                                                            contract_digest=p.get("contract_digest", ""),
                                                            action_class=p.get("action_class", ""))
            rec.status, rec.real_action, rec.in_doubt = ExecutionStatus.EXECUTED_UNVERIFIED, True, True
            rec.notes.append("restored_in_doubt:command_attempt_without_recorded_result:never_auto_retried")
            self._records[key] = rec

    # ---------------------------------------------------------------- 9 стадий
    def submit(self, contract: ActionContract, *, evidence: Mapping[str, str], recommendation_source: str,
               now: datetime, simulation_pack=None, approval: Optional[Approval] = None) -> ExecutionRecord:
        now = aware("now", now)
        prior = self._records.get(contract.idempotency_key)
        if prior is not None and prior.contract_digest != contract.contract_digest:
            rec = ExecutionRecord(contract.action_id, contract.idempotency_key, ExecutionStatus.BLOCKED,
                                  contract_digest=contract.contract_digest, action_class=contract.action_class)
            rec.stage(Stage.INTENT, False, "idempotency_key_reused_for_a_different_contract")
            self._j("replay_conflict", {"action_id": contract.action_id, "idempotency_key": contract.idempotency_key}, now)
            return rec
        if prior is not None and prior.final:
            self._j("replay", {"action_id": contract.action_id, "idempotency_key": contract.idempotency_key}, now)
            return prior          # §3.3: исходный результат, вторая физическая команда не выдаётся
        rec = ExecutionRecord(contract.action_id, contract.idempotency_key, ExecutionStatus.BLOCKED,
                              contract_digest=contract.contract_digest, action_class=contract.action_class)
        if runtime_status(self.journal) is RuntimeStatus.DEGRADED:
            rec.status = ExecutionStatus.DEGRADED_BLOCKED
            rec.stage(Stage.INTENT, False, "durable_journal_unavailable:physical_write_authority_blocked")
            self._records[rec.idempotency_key] = rec
            return rec
        self._contracts[contract.action_id] = contract
        t0 = time.perf_counter()
        case = self.assurance.for_action(contract.action_class, contract.capability_version)
        # 1 — намерение
        r1 = []
        if contract.expiry_at <= now:
            r1.append("contract_expired")
        if case is None:
            r1.append("no_assurance_case_for_action_class_and_capability_version")
        else:
            r1 += case.evidence_gaps(evidence)
        ok_src, why_src = self.sovereign.recommendation_admissible(recommendation_source)
        r1 += list(why_src)
        try:
            requested = AutonomyTier(contract.authority_tier)
        except ValueError:
            requested = None
            r1.append(f"unknown_authority_tier:{contract.authority_tier}")
        if not rec.stage(Stage.INTENT, not r1, *r1):
            return self._end(rec, contract, now, evidence)
        # 2 — возможность и личность (I, C)
        b = self._bindings.get(contract.target_entity_id)
        r2 = [] if self.identity.resolved(contract.target_entity_id) else ["target_identity_unresolved"]
        if b is None:
            r2.append("no_adapter_bound_to_target")
        else:
            if b.capability_version != contract.capability_version:
                r2.append(f"capability_version_mismatch:{b.capability_version}")
            if contract.action_type not in b.action_types or contract.action_type not in b.adapter.spec.supported_commands:
                r2.append(f"action_type_not_supported:{contract.action_type}")
            for p in contract.parameters:
                try:
                    if not b.adapter.spec.point(p).writable:
                        r2.append(f"point_not_writable:{p}")
                except SovereignError:
                    r2.append(f"unknown_point:{p}")
        if not rec.stage(Stage.CAPABILITY, not r2, *r2):
            return self._end(rec, contract, now, evidence)
        # 3 — предусловия, свежесть, здоровье (F, H)
        r3, fact_ages = [], []
        for name in contract.preconditions:
            f = self.facts.fact(name)
            if f is None:
                r3.append(f"precondition_unknown:{name}")
                continue
            age = (now - aware("observed_at", f.observed_at)).total_seconds()
            fact_ages.append(age)
            if age > self.policy.max_fact_age_s:
                r3.append(f"precondition_stale:{name}")
            if not f.healthy:
                r3.append(f"precondition_unhealthy:{name}")
            if not f.holds:
                r3.append(f"precondition_false:{name}")
        if not b.adapter.healthy(now):
            r3.append("adapter_heartbeat_missing")
        if self.guard.last_watchdog is None or \
                (now - self.guard.last_watchdog).total_seconds() > self.guard.envelope.watchdog_timeout_s:
            r3.append("edge_watchdog_timeout")
        if not rec.stage(Stage.PRECONDITIONS, not r3, *r3):
            return self._end(rec, contract, now, evidence)
        # 4 — физическая оболочка (P)
        state, state_age = self._state(now)
        r4 = []
        for var, (lo, hi) in self.guard.envelope.state_bounds.items():
            if var not in state:
                r4.append(f"state_unknown_or_stale:{var}")
            elif not lo <= state[var] <= hi:
                r4.append(f"state_outside_hard_envelope:{var}")
        for p, v in contract.parameters.items():
            lim = contract.physical_limits.get(p)
            if lim is None:
                r4.append(f"no_physical_limits_declared:{p}")
            elif (lim[0] is not None and v < lim[0]) or (lim[1] is not None and v > lim[1]):
                r4.append(f"parameter_outside_contract_limits:{p}")
            spec = b.adapter.spec.point(p)
            if not spec.lower <= v <= spec.upper:
                r4.append(f"parameter_outside_device_range:{p}")
            hb = self.guard.envelope.control_bounds.get(p)
            if hb is None or not hb[0] <= v <= hb[1]:
                r4.append(f"parameter_outside_hard_envelope:{p}")
        for p, v in contract.safe_state.items():
            hb = self.guard.envelope.control_bounds.get(p)
            if hb is None or not hb[0] <= v <= hb[1]:
                r4.append(f"safe_state_not_inside_hard_envelope:{p}")
        if not rec.stage(Stage.ENVELOPE, not r4, *r4):
            return self._end(rec, contract, now, evidence)
        # 5 — политика, согласие, полномочия (A)
        r5 = []
        active = self.policy_store.active
        if active is None:
            r5.append("policy_artifact_invalid_or_unavailable:only_hardware_safe_behaviour")
        else:
            if active.artifact_hash not in contract.policy_refs and active.version not in contract.policy_refs:
                r5.append("contract_does_not_reference_active_edge_policy")
            if contract.site not in active.scope and contract.target_entity_id not in active.scope:
                r5.append("edge_policy_scope_excludes_target")
            for p, v in contract.parameters.items():
                r5 += list(active.violations(contract.action_class, p, v))
        allowed, why = self.safety.evaluate(contract)
        if not allowed:
            r5 += list(why) or ["safety_service_denied"]
        sov_ok, sov_cap, sov_why = self.sovereign.authority(
            contract.action_class, active_policy_hash=self.policy_store.attest(), now=now)
        r5 += list(sov_why)
        if not rec.stage(Stage.AUTHORITY, not r5 and sov_ok, *r5):
            return self._end(rec, contract, now, evidence)
        # 6 — симуляция/влияние (пакет — только предусловие, права на действие у него нет)
        r6 = list(case.simulation_gaps(simulation_pack, trace_id=contract.trace_id))
        if not rec.stage(Stage.SIMULATION, not r6, *r6):
            return self._end(rec, contract, now, evidence, simulation_pack)
        # 7 — одобрение / уровень автономии (никогда не глобально)
        key = AutonomyKey(contract.tenant, contract.site, contract.target_entity_id, contract.action_class,
                          contract.risk_tier)
        world_age = max([state_age if state_age is not None else float("inf")] + fact_ages)
        tier, why7 = self.autonomy.effective_tier(key, requested=requested, now=now, world_state_age_s=world_age,
                                                  signal=self.entropy.signal(), cap=sov_cap)
        rec.tier = tier.value
        if tier is AutonomyTier.SHADOW:
            rec.stage(Stage.APPROVAL, True, "shadow_only", *why7)
            rec.status = ExecutionStatus.SHADOW
            return self._end(rec, contract, now, evidence, simulation_pack, approval, state)
        if tier is AutonomyTier.HUMAN_APPROVED:
            if approval is None:
                rec.stage(Stage.APPROVAL, False, "human_approval_required", *why7)
                rec.status = ExecutionStatus.AWAITING_APPROVAL
                return self._end(rec, contract, now, evidence, simulation_pack, approval, state)
            r7 = self._approval_gaps(contract, approval, now)
            if not rec.stage(Stage.APPROVAL, not r7, *r7, *why7):
                return self._end(rec, contract, now, evidence, simulation_pack, approval, state)
        else:
            rec.stage(Stage.APPROVAL, True, "limited_autonomy_grant", *why7)
        rec.timings_s["gates_1_7"] = time.perf_counter() - t0
        # 8 — идемпотентное протокольное исполнение
        if not self._decision(rec, contract, now, evidence, simulation_pack, approval, state):
            rec.status = ExecutionStatus.DEGRADED_BLOCKED
            rec.stage(Stage.EXECUTION, False, "durable_journal_unavailable:physical_write_authority_blocked")
            self._records[rec.idempotency_key] = rec
            return rec
        return self._execute(rec, contract, b.adapter, state, now)

    def _approval_gaps(self, contract: ActionContract, a: Approval, now: datetime) -> list[str]:
        out = []
        key = self._approvers.get(a.approver_id)
        if key is None or not verify_artifact(a.body(), a.signature, key):
            out.append("approval_signature_invalid_or_unknown_approver")
        if a.contract_digest != contract.contract_digest:
            out.append("approval_for_a_different_contract")
        if contract.approval_ref != a.approval_id:
            out.append("contract_approval_ref_mismatch")
        if aware("signed_at", a.signed_at) > now:
            out.append("approval_signed_in_future")
        return out

    def _decision(self, rec, contract, now, evidence, pack=None, approval=None, state=None) -> bool:
        """§8: одна запись восстанавливает вход, проверки, состояние, симуляцию, политику, одобрение, версии."""
        case = self.assurance.for_action(contract.action_class, contract.capability_version)
        return self._j("decision", {
            "action_id": contract.action_id, "trace_id": contract.trace_id, "idempotency_key": contract.idempotency_key,
            "contract": {k: getattr(contract, k) for k in contract.__dataclass_fields__},
            "stages": rec.stages, "status": rec.status.value, "tier": rec.tier, "evidence": dict(evidence),
            "state_used": dict(state or {}),
            "simulation": None if pack is None else {"pack_digest": getattr(pack, "pack_digest", None),
                                                     "status": getattr(getattr(pack, "status", None), "value", None),
                                                     "use": getattr(getattr(pack, "allowed_downstream_use", None),
                                                                    "value", None)},
            "approval": None if approval is None else {"approval_id": approval.approval_id,
                                                       "approver": approval.approver_id},
            "versions": {"capability": contract.capability_version,
                         "edge_policy": self.policy_store.attest(),
                         "assurance_case": None if case is None else case.case_version,
                         "runtime_policy": self.policy.policy_version,
                         "autonomy_policy": self.autonomy.policy.policy_version},
            "connectivity": self.sovereign.connectivity.value}, now)

    def _end(self, rec, contract, now, evidence, pack=None, approval=None, state=None) -> ExecutionRecord:
        if not self._decision(rec, contract, now, evidence, pack, approval, state):
            rec.status = ExecutionStatus.DEGRADED_BLOCKED
            rec.notes.append("durable_journal_unavailable")
        self._save(rec, now)
        return rec

    def _write(self, adapter: TypedAdapter, contract: ActionContract, point: str, value: float, state, now,
               *, tag: str) -> tuple[Optional[bool], tuple[str, ...]]:
        """Одна запись через предохранитель. (ack.ok | None если команды не было, причины)."""
        spec = adapter.spec.point(point)
        action_ref = f"{contract.action_id}{tag}"
        sig = self.guard.sign_command(adapter.spec.adapter_id, point, value, action_ref)
        safe = tag in (":safe_state", ":safe_mode")
        self.guard.load_policy(self.policy_store.active)
        t = time.perf_counter()
        permit, why = self.guard.check(adapter_id=adapter.spec.adapter_id, point=point, value=value, state=state,
                                       action_id=action_ref, action_class=contract.action_class,
                                       command_signature=sig, now=now, safe_state=safe)
        edge_s = time.perf_counter() - t
        if permit is None:
            self._j("guard_trip", {"action_id": contract.action_id, "point": point, "value": value,
                                   "reasons": list(why)}, now)
            return None, tuple(why)
        if not self._j("command_attempt", {"action_id": contract.action_id, "idempotency_key": contract.idempotency_key,
                                           "point": point, "value": value, "permit": permit.permit_id, "tag": tag,
                                           "action_class": contract.action_class,
                                           "contract_digest": getattr(contract, "contract_digest", "")}, now):
            return None, ("durable_journal_unavailable:command_not_sent",)
        t = time.perf_counter()
        try:
            ack = adapter.write(point, value, unit=spec.unit, command=contract.action_type, permit=permit, now=now,
                                safe_state=safe)
        except SovereignError as exc:
            self._j("command_refused", {"action_id": contract.action_id, "point": point, "reason": str(exc)}, now)
            return False, (f"adapter_refused:{exc}",)
        ctrl_s = time.perf_counter() - t
        self._j("command_ack", {"action_id": contract.action_id, "idempotency_key": contract.idempotency_key,
                                "point": point, "ok": ack.ok, "command_ref": ack.command_ref,
                                "semantics": ack.semantics, "edge_validation_s": edge_s, "controller_s": ctrl_s}, now)
        return ack.ok, ()

    def _execute(self, rec: ExecutionRecord, contract: ActionContract, adapter: TypedAdapter, state, now) -> ExecutionRecord:
        written = False
        for point, value in sorted(contract.parameters.items()):
            ok, why = self._write(adapter, contract, point, value, state, now, tag="")
            if ok is None and any(w.startswith("durable_journal_unavailable") for w in why):
                rec.status = ExecutionStatus.DEGRADED_BLOCKED
                rec.stage(Stage.EXECUTION, False, *why)
                if written:
                    self._apply_safe_state(rec, contract, adapter, now, "journal_lost_mid_action")
                self._records[rec.idempotency_key] = rec
                return rec
            if ok is not None:
                written = True
                rec.real_action = True
            if ok is not True:
                rec.ack = False
                rec.stage(Stage.EXECUTION, False, *(why or ("command_not_acknowledged",)))
                if written:
                    self._apply_safe_state(rec, contract, adapter, now, "execution_failed")
                    rec.status = ExecutionStatus.FAILED_SAFE_STATE
                else:
                    rec.status = ExecutionStatus.BLOCKED       # предохранитель не выпустил ни одной команды
                self._save(rec, now)
                return rec
        rec.ack, rec.status, rec.executed_at = True, ExecutionStatus.EXECUTED_UNVERIFIED, now.isoformat()
        rec.command_ref = f"{adapter.spec.adapter_id}:{contract.action_id}"
        rec.stage(Stage.EXECUTION, True)
        self._save(rec, now)
        return rec

    def _apply_safe_state(self, rec, contract, adapter, now, why: str) -> None:
        state, _ = self._state(now)
        results = {}
        for p, v in sorted(contract.safe_state.items()):
            ok, reasons = self._write(adapter, contract, p, v, state, now, tag=":safe_state")
            results[p] = {"ok": ok, "reasons": list(reasons)}
        rec.safe_state_applied = all(r["ok"] is True for r in results.values())
        if not rec.safe_state_applied:
            rec.notes.append("safe_state_not_confirmed:hardware_interlock_and_plc_remain_the_authority")
            self.operator_notifications.append({"at": now.isoformat(), "action_id": contract.action_id,
                                                "event": "safe_state_not_confirmed", "detail": results})
        self._j("safe_state", {"action_id": contract.action_id, "why": why, "results": results,
                               "compensation_plan": list(contract.compensation_plan)}, now)

    # ---------------------------------------------------------------- 9 — проверка по телеметрии
    def verify(self, idempotency_key: str, *, now: datetime) -> ExecutionRecord:
        now = aware("now", now)
        rec = self._records.get(idempotency_key)
        if rec is None:
            raise SovereignError("unknown idempotency key")
        if rec.status is not ExecutionStatus.EXECUTED_UNVERIFIED:
            return rec
        contract = self._contracts.get(rec.action_id)
        if contract is None or rec.executed_at is None:
            rec.notes.append("verification_impossible:contract_not_in_memory_after_restart:operator_review")
            self.operator_notifications.append({"at": now.isoformat(), "action_id": rec.action_id,
                                                "event": "in_doubt_requires_operator"})
            self._save(rec, now)
            return rec
        start = datetime.fromisoformat(rec.executed_at)
        case = self.assurance.for_action(contract.action_class, contract.capability_version)
        pending, failed, observed = [], [], {}
        for eff in contract.expected_effect:
            end = start + timedelta(seconds=eff.window_s)
            samples = sorted(((t, v) for t, v in self._telemetry.get(eff.point, ()) if start <= t <= end),
                             key=lambda x: x[0])        # устойчиво: при равном времени — последнее принятое
            observed[eff.point] = samples[-1][1] if samples else None
            if samples and eff.holds(samples[-1][1]):
                continue
            (pending if now < end else failed).append(eff.point)
        violations = []
        for var, (lo, hi) in self.guard.envelope.state_bounds.items():
            for t, v in self._telemetry.get(var, ()):
                if t >= start and not lo <= v <= hi:
                    violations.append(f"safety_violation:{var}")
                    break
        if case is not None and now > start + timedelta(seconds=case.timeout_s) and pending:
            failed += pending
            pending = []
        rec.observed = observed
        if violations or failed:
            rec.verified = False
            rec.stage(Stage.VERIFICATION, False, *violations, *(f"postcondition_not_observed:{p}" for p in failed))
            b = self._bindings[contract.target_entity_id]
            self._apply_safe_state(rec, contract, b.adapter, now, "verification_failed")
            rec.status = ExecutionStatus.FAILED_SAFE_STATE
            self._save(rec, now)
            return rec
        if pending:
            return rec
        rec.verified = bool(rec.ack)
        rec.status = ExecutionStatus.VERIFIED if rec.verified else ExecutionStatus.FAILED_SAFE_STATE
        rec.stage(Stage.VERIFICATION, rec.verified)
        self._save(rec, now)
        return rec

    # ---------------------------------------------------------------- тень → свидетельство для автономии
    def record_shadow_outcome(self, idempotency_key: str, *, expected_effect_held: bool, evidence_ref: str,
                              now: datetime) -> None:
        rec = self._records.get(idempotency_key)
        contract = self._contracts.get(rec.action_id) if rec else None
        if rec is None or rec.status is not ExecutionStatus.SHADOW or contract is None:
            raise SovereignError("shadow outcome can only be recorded for a shadow decision")
        key = AutonomyKey(contract.tenant, contract.site, contract.target_entity_id, contract.action_class,
                          contract.risk_tier)
        if not self._j("shadow_outcome", {"action_id": rec.action_id, "held": bool(expected_effect_held),
                                          "evidence_ref": text("evidence_ref", evidence_ref)}, aware("now", now)):
            raise SovereignError("durable journal unavailable: shadow evidence not recorded")
        self.autonomy.record_shadow(key, expected_effect_held=expected_effect_held)

    # ---------------------------------------------------------------- энтропия → SAFE_MODE
    def observe_state_distribution(self, p, *, at: datetime) -> SafetySignal:
        was = self.entropy.latched is not None
        sig = self.entropy.observe(p, at=at)
        if sig is SafetySignal.SAFE_MODE and not was:
            revoked = self.autonomy.revoke_all_active()
            results = {}
            state, _ = self._state(aware("at", at))
            for eid, b in self._bindings.items():
                for point, value in sorted(b.adapter.spec.safe_state.items()):
                    probe = _SafeProbe(eid, point, value)
                    ok, why = self._write(b.adapter, probe, point, value, state, aware("at", at), tag=":safe_mode")
                    results[f"{eid}:{point}"] = {"ok": ok, "reasons": list(why)}
            event = {"at": aware("at", at).isoformat(), "event": "SAFE_MODE", "entropy": self.entropy.latched,
                     "autonomy_grants_revoked": revoked, "local_safe_setpoints": results}
            self.operator_notifications.append(event)
            self._j("safe_mode", event, aware("at", at))
        return sig

    def record(self, idempotency_key: str) -> Optional[ExecutionRecord]:
        return self._records.get(idempotency_key)


@dataclass(frozen=True)
class _SafeProbe:
    """Минимальный «контракт» для записи локальных безопасных уставок в SAFE_MODE (через тот же предохранитель)."""

    target_entity_id: str
    point: str
    value: float

    @property
    def action_id(self) -> str:
        return f"safe_mode:{self.target_entity_id}:{self.point}"

    @property
    def idempotency_key(self) -> str:
        return self.action_id

    @property
    def action_class(self) -> str:
        return "safe_state"

    @property
    def action_type(self) -> str:
        return "write_setpoint"


__all__ = ["Approval", "ExecutionControlPipeline", "FactPort", "FactStatus", "IdentityPort", "RuntimePolicy",
           "SafetyPort"]
