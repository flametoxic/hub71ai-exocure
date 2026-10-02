"""EXO Sovereign Runtime, Execution Control & Security (+ мастер-ТЗ D4, E3, E4).

Ворота перед физическим исполнением: 9 стадий, Executable = I∧C∧F∧P∧A∧H, идемпотентность, проверка по
телеметрии, безопасное состояние, журнал с цепочкой дайджестов и корнем Меркла, суверенный и деградированный
режимы, уровни автономии только по точному ключу (никогда глобально), assurance case на класс действий.
Модель мир не меняет: писать в устройство может только TypedAdapter с разрешением EdgeHardGuard.
"""
from .assurance import F_ASSURANCE, AssuranceCase, AssuranceRegistry
from .autonomy import (F_ENTROPY, AutonomyGate, AutonomyGrant, AutonomyKey, AutonomyPolicy, AutonomyTier,
                       EntropyMonitor, SafetySignal, shannon_entropy)
from .bundle import SovereignRuntimeBundle
from .contracts import (F_EXECUTABLE, F_HARD_GUARD, F_IDEMPOTENCY, F_LATENCY, F_REPLAY, F_SOVEREIGNTY, F_TRIP,
                        F_VERIFIED, ActionContract, ExecutionRecord, ExecutionStatus, ExpectedEffect, SovereignError,
                        Stage)
from .edge import (Ack, AdapterSpec, CompiledPolicy, EdgeHardGuard, EdgePolicyStore, GuardPermit, HardEnvelope,
                   HardwareInterlock, LatencyBudget, PointSpec, ProtocolKind, Transport, TypedAdapter, compile_policy,
                   measure)
from .hvac import HVAC_SETPOINT_ACTION_CLASS, HVAC_SETPOINT_COMMAND, build_hvac_setpoint_action
from .pipeline import Approval, ExecutionControlPipeline, FactPort, FactStatus, IdentityPort, RuntimePolicy, SafetyPort
from .sovereign import (CAUSAL_SOURCES, Connectivity, LocalArtifacts, ReconciliationLedger, RuntimeStatus,
                        SignedSyncBatch, SovereignState, export_sync_batch, runtime_status)
from .trust import DeviceTrust, ExecutionJournal, SignedMessage, merkle_root, sign_artifact, verify_artifact

__all__ = ["Ack", "ActionContract", "AdapterSpec", "Approval", "AssuranceCase", "AssuranceRegistry", "AutonomyGate",
           "AutonomyGrant", "AutonomyKey", "AutonomyPolicy", "AutonomyTier", "CAUSAL_SOURCES", "CompiledPolicy",
           "Connectivity", "DeviceTrust", "EdgeHardGuard", "EdgePolicyStore", "EntropyMonitor",
           "ExecutionControlPipeline", "ExecutionJournal", "ExecutionRecord", "ExecutionStatus", "ExpectedEffect",
           "F_ASSURANCE", "F_ENTROPY", "F_EXECUTABLE", "F_HARD_GUARD", "F_IDEMPOTENCY", "F_LATENCY", "F_REPLAY",
           "F_SOVEREIGNTY", "F_TRIP", "F_VERIFIED", "FactPort", "FactStatus", "GuardPermit",
           "HVAC_SETPOINT_ACTION_CLASS", "HVAC_SETPOINT_COMMAND", "HardEnvelope", "HardwareInterlock",
           "IdentityPort", "LatencyBudget", "LocalArtifacts", "PointSpec", "ProtocolKind", "ReconciliationLedger",
           "RuntimePolicy", "RuntimeStatus", "SafetyPort", "SafetySignal", "SignedMessage", "SignedSyncBatch",
           "SovereignError", "SovereignRuntimeBundle", "SovereignState", "Stage", "Transport", "TypedAdapter",
           "build_hvac_setpoint_action",
           "compile_policy", "export_sync_batch", "measure", "merkle_root", "runtime_status", "shannon_entropy",
           "sign_artifact", "verify_artifact"]
