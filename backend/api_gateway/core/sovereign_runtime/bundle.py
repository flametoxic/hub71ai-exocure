"""Сборка Sovereign Runtime из настроек площадки (app/config). Все пороги — из политики, ключи — из
отдельного секретного файла (не из config). Конвейер собирается только когда переданы все три порта
(идентичность, факты, SafetyService); без них — не собирается (fail-closed), физических записей нет.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .assurance import AssuranceRegistry
from .autonomy import AutonomyGate, AutonomyPolicy, EntropyMonitor
from .contracts import SovereignError
from .edge import EdgeHardGuard, EdgePolicyStore, HardEnvelope, HardwareInterlock
from .pipeline import ExecutionControlPipeline, FactPort, IdentityPort, RuntimePolicy, SafetyPort
from .sovereign import SovereignState
from .trust import DeviceTrust, ExecutionJournal


def _key(v: Any, name: str) -> bytes:
    try:
        b = bytes.fromhex(str(v))
    except ValueError as exc:
        raise SovereignError(f"key {name} must be hex") from exc
    if len(b) < 16:
        raise SovereignError(f"key {name} must be at least 16 bytes")
    return b


@dataclass
class SovereignRuntimeBundle:
    runtime_policy: RuntimePolicy
    journal: ExecutionJournal
    device_trust: DeviceTrust
    guard: EdgeHardGuard
    policy_store: EdgePolicyStore
    assurance: AssuranceRegistry
    autonomy: AutonomyGate
    entropy: EntropyMonitor
    sovereign: SovereignState
    approver_keys: Mapping[str, bytes]

    @classmethod
    def from_settings(cls, policy: Mapping[str, Any], keys: Mapping[str, Any], *,
                      interlock: Optional[HardwareInterlock] = None) -> "SovereignRuntimeBundle":
        p = dict(policy)
        env = dict(p["envelope"])
        gov, cmd, op = _key(keys["governance"], "governance"), _key(keys["command"], "command"), _key(keys["operator"], "operator")
        approvers = {str(k): _key(v, f"approver:{k}") for k, v in dict(keys["approvers"]).items()}
        trust = DeviceTrust(**dict(p["device_trust"]))
        for d, k in dict(keys.get("devices", {})).items():
            trust.register(str(d), _key(k, f"device:{d}"))
        apol = AutonomyPolicy(**dict(p["autonomy"]))
        return cls(
            runtime_policy=RuntimePolicy(**dict(p["runtime"])),
            journal=ExecutionJournal(str(p["journal_path"])),
            device_trust=trust,
            guard=EdgeHardGuard(HardEnvelope(
                state_bounds={k: (float(v[0]), float(v[1])) for k, v in dict(env["state_bounds"]).items()},
                control_bounds={k: (float(v[0]), float(v[1])) for k, v in dict(env["control_bounds"]).items()},
                rate_limits={k: float(v) for k, v in dict(env.get("rate_limits", {})).items()},
                watchdog_timeout_s=float(env["watchdog_timeout_s"])), command_key=cmd, interlock=interlock),
            policy_store=EdgePolicyStore(governance_key=gov, cache_path=str(p["policy_cache_path"])),
            assurance=AssuranceRegistry(governance_key=gov),
            autonomy=AutonomyGate(apol, approver_keys=approvers),
            entropy=EntropyMonitor(apol, operator_key=op),
            sovereign=SovereignState(),
            approver_keys=approvers)

    @staticmethod
    def read_keys(path: str) -> dict:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def build_pipeline(self, *, identity: Optional[IdentityPort], facts: Optional[FactPort],
                       safety: Optional[SafetyPort]) -> ExecutionControlPipeline:
        missing = [n for n, v in (("identity", identity), ("facts", facts), ("safety", safety)) if v is None]
        if missing:
            raise SovereignError(f"execution pipeline not built: missing ports {missing} (fail-closed)")
        return ExecutionControlPipeline(
            policy=self.runtime_policy, journal=self.journal, device_trust=self.device_trust, guard=self.guard,
            policy_store=self.policy_store, assurance=self.assurance, autonomy=self.autonomy, entropy=self.entropy,
            sovereign=self.sovereign, identity=identity, facts=facts, safety=safety, approver_keys=self.approver_keys)


__all__ = ["SovereignRuntimeBundle"]
