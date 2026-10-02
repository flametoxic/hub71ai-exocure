"""WM-05 §11 (модель операторов и организации), §12 (надёжность когнитивного рантайма), §13 (цепочка поставки).

§11  U_feasible = U_physical ∩ U_policy ∩ U_human_availability — нельзя запросить проверку/одобрение у роли,
     недоступной в нужное окно, без явной эскалации.
§12  Availability = MTBF / (MTBF + MTTR); у каждого класса когнитивного инцидента — обнаружение, тяжесть,
     владелец, сдерживание, runbook и шаблон разбора.
§13  Valid(a) = SignatureValid(a) ∧ SBOMApproved(a) ∧ VersionAllowed(a) ∧ NotRevoked(a);
     неизвестное/отозванное устройство → нет операционных свидетельств; неподписанное → нет развёртывания;
     битая цепочка OTA → край остаётся на доверенной прежней версии; ключ чужого арендатора → блок + аудит + инцидент.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Mapping, Optional, Sequence

from ..world_model.reality_formulas import FormulaReference
from .context import DOC, IntegrityError

F_FEASIBLE = FormulaReference("WM05-11-FEASIBLE", DOC, 11, "U_feasible = U_physical ∩ U_policy ∩ U_human_availability")
F_AVAILABILITY = FormulaReference("WM05-12-AVAILABILITY", DOC, 11, "Availability = MTBF / (MTBF + MTTR)")
F_ARTIFACT = FormulaReference("WM05-13-VALID", DOC, 12,
                              "Valid(a) = SignatureValid ∧ SBOMApproved ∧ VersionAllowed ∧ NotRevoked")


# ------------------------------------------------------------------------ §11 операторы
@dataclass(frozen=True)
class ShiftWindow:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class Operator:
    operator_id: str
    role: str
    authority_scope: tuple                 # классы действий
    shifts: tuple                          # ShiftWindow
    certifications: tuple
    site_permissions: tuple
    delegation_rights: tuple = ()
    response_sla_s: float = 0.0
    escalates_to: Optional[str] = None     # роль, куда эскалировать


class Organization:
    def __init__(self, operators: Sequence[Operator]) -> None:
        self.operators = {o.operator_id: o for o in operators}
        self.overrides: list[dict] = []

    def available(self, role: str, *, site: str, action_class: str, window: ShiftWindow,
                  certification: Optional[str] = None) -> list[str]:
        out = []
        for o in self.operators.values():
            if o.role != role or site not in o.site_permissions or action_class not in o.authority_scope:
                continue
            if certification and certification not in o.certifications:
                continue
            if any(s.start <= window.start and window.end <= s.end for s in o.shifts):
                out.append(o.operator_id)
        return sorted(out)

    def human_feasible(self, role: str, *, site: str, action_class: str, window: ShiftWindow,
                       certification: Optional[str] = None) -> dict:
        """Доступна ли роль в окно; иначе — явная цепочка эскалации (до цикла); иначе действие недопустимо."""
        path, cur, seen = [], role, set()
        while cur not in seen:
            seen.add(cur)
            who = self.available(cur, site=site, action_class=action_class, window=window, certification=certification)
            path.append({"role": cur, "available": who})
            if who:
                return {"feasible": True, "role": cur, "operators": who, "escalated": cur != role, "path": path,
                        "formula": F_FEASIBLE}
            nxt = next((o.escalates_to for o in self.operators.values() if o.role == cur and o.escalates_to), None)
            if not nxt:
                break
            cur = nxt
        return {"feasible": False, "role": role, "operators": [], "escalated": False, "path": path,
                "reason": "no_authorized_human_available_in_window_and_no_escalation", "formula": F_FEASIBLE}

    def record_override(self, *, operator_id: str, action_id: str, reason: str, at: datetime) -> None:
        if operator_id not in self.operators or not reason.strip():
            raise IntegrityError("override needs a known operator and a reason (post-action accountability)")
        self.overrides.append({"operator_id": operator_id, "action_id": action_id, "reason": reason,
                               "at": at.isoformat()})


def feasible_actions(candidates: Iterable[str], *, physical_ok: Mapping[str, bool], policy_ok: Mapping[str, bool],
                     human_ok: Mapping[str, bool]) -> list[str]:
    return [u for u in candidates if physical_ok.get(u) is True and policy_ok.get(u) is True and human_ok.get(u) is True]


# ------------------------------------------------------------------------ §12 надёжность
SLO_NAMES = ("state_freshness", "world_projection_latency", "causal_result_latency", "plan_generation_latency",
             "execution_verification", "audit_durability", "replay_completion", "sensor_coverage",
             "degraded_mode_recovery")
INCIDENT_CLASSES = ("reality_integrity", "identity_resolution", "sensor_integrity", "causal_model", "unsafe_fallback",
                    "policy_execution_boundary", "audit_replay", "model_calibration")
INCIDENT_FIELDS = ("detection", "severity", "owner", "containment_action", "operator_runbook", "postmortem_template")


def availability(mtbf_s: float, mttr_s: float) -> float:
    if mtbf_s <= 0 or mttr_s < 0:
        raise IntegrityError("MTBF > 0 and MTTR ≥ 0 required")
    return mtbf_s / (mtbf_s + mttr_s)


@dataclass
class ReliabilityRegistry:
    slo_targets: Mapping[str, float]                    # SLO → цель (секунды, доля — как задано площадкой)
    slo_direction: Mapping[str, str]                    # "max" (не больше) | "min" (не меньше)
    incidents: Mapping[str, Mapping[str, str]]
    measurements: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        miss = [s for s in SLO_NAMES if s not in self.slo_targets or self.slo_direction.get(s) not in ("max", "min")]
        if miss:
            raise IntegrityError(f"SLO targets/directions missing for {miss} (WM-05 §12.1)")
        gaps = [f"{c}:{f}" for c in INCIDENT_CLASSES for f in INCIDENT_FIELDS
                if not str(dict(self.incidents.get(c, {})).get(f, "")).strip()]
        if gaps:
            raise IntegrityError(f"incident classes incomplete: {gaps[:6]}… (WM-05 §12.2)")

    def record(self, slo: str, value: float) -> None:
        self.measurements.setdefault(slo, []).append(float(value))

    def report(self) -> dict:
        out = {}
        for s in SLO_NAMES:
            vals = self.measurements.get(s)
            if not vals:
                out[s] = {"status": "not_measured"}
                continue
            worst = max(vals) if self.slo_direction[s] == "max" else min(vals)
            ok = worst <= self.slo_targets[s] if self.slo_direction[s] == "max" else worst >= self.slo_targets[s]
            out[s] = {"status": "met" if ok else "violated", "worst": worst, "target": self.slo_targets[s]}
        return out


# ------------------------------------------------------------------------ §13 цепочка поставки
def _sig(key: bytes, body: Mapping) -> str:
    return hmac.new(key, json.dumps(dict(body), sort_keys=True, default=str).encode(), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    kind: str                                  # model | policy | adapter | data_pack | firmware
    version: str
    tenant: str
    digest: str
    sbom_ref: Optional[str]
    signer_key_id: str
    signature: str

    def body(self) -> dict:
        return {"artifact_id": self.artifact_id, "kind": self.kind, "version": self.version, "tenant": self.tenant,
                "digest": self.digest, "sbom_ref": self.sbom_ref, "signer_key_id": self.signer_key_id}


class SupplyChain:
    def __init__(self) -> None:
        self._keys: dict[str, tuple[str, bytes]] = {}     # key_id → (tenant, key)
        self.revoked_keys: set = set()
        self.revoked_artifacts: set = set()
        self.sbom_approved: set = set()
        self.allowed_versions: dict[tuple, set] = {}      # (kind, artifact_id) → разрешённые версии
        self.devices: dict[str, str] = {}                  # device_id → tenant
        self.revoked_devices: set = set()
        self.edge_version: dict[str, str] = {}             # edge_id → доверенная версия
        self.audit: list[dict] = []
        self.incidents: list[dict] = []

    # ключи
    def generate_key(self, key_id: str, tenant: str, material: bytes) -> None:
        if len(material) < 16 or key_id in self._keys:
            raise IntegrityError("keys ≥ 16 bytes, unique key ids")
        self._keys[key_id] = (tenant, material)

    def rotate_key(self, old_id: str, new_id: str, material: bytes) -> None:
        tenant = self._keys[old_id][0]
        self.generate_key(new_id, tenant, material)
        self.revoked_keys.add(old_id)
        self.audit.append({"event": "key_rotated", "old": old_id, "new": new_id})

    def revoke_key(self, key_id: str) -> None:
        self.revoked_keys.add(key_id)
        self.audit.append({"event": "key_revoked", "key_id": key_id})

    def sign(self, a: Artifact, key_id: str) -> Artifact:
        tenant, key = self._keys[key_id]
        body = dict(a.body(), signer_key_id=key_id)
        return Artifact(**{**body, "signature": _sig(key, body)})

    # проверка
    def valid(self, a: Artifact) -> dict:
        k = self._keys.get(a.signer_key_id)
        sig_ok = k is not None and a.signer_key_id not in self.revoked_keys and \
            hmac.compare_digest(_sig(k[1], a.body()), a.signature)
        cross = k is not None and k[0] != a.tenant
        if cross:
            self.audit.append({"event": "cross_tenant_key_misuse", "artifact": a.artifact_id})
            self.incidents.append({"class": "policy_execution_boundary", "kind": "cross_tenant_key_misuse",
                                   "artifact": a.artifact_id})
        checks = {"SignatureValid": sig_ok and not cross, "SBOMApproved": bool(a.sbom_ref) and a.sbom_ref in self.sbom_approved,
                  "VersionAllowed": a.version in self.allowed_versions.get((a.kind, a.artifact_id), set()),
                  "NotRevoked": a.artifact_id + "@" + a.version not in self.revoked_artifacts}
        return {"valid": all(checks.values()), "checks": checks, "formula": F_ARTIFACT}

    def deploy(self, a: Artifact) -> dict:
        v = self.valid(a)
        if not v["valid"]:
            self.audit.append({"event": "deployment_refused", "artifact": a.artifact_id, "checks": v["checks"]})
            return {"deployed": False, **v}
        return {"deployed": True, **v}

    def ota_update(self, edge_id: str, chain: Sequence[Artifact]) -> dict:
        """Каждое звено цепочки OTA должно быть валидно; иначе край остаётся на прежней доверенной версии."""
        prev = self.edge_version.get(edge_id)
        for a in chain:
            if not self.valid(a)["valid"]:
                self.audit.append({"event": "ota_chain_invalid", "edge": edge_id, "artifact": a.artifact_id})
                return {"updated": False, "edge_version": prev, "reason": f"invalid_link:{a.artifact_id}"}
        self.edge_version[edge_id] = chain[-1].version
        return {"updated": True, "edge_version": chain[-1].version}

    def device_evidence_allowed(self, device_id: str, tenant: str) -> bool:
        ok = self.devices.get(device_id) == tenant and device_id not in self.revoked_devices
        if not ok:
            self.audit.append({"event": "device_evidence_refused", "device": device_id})
        return ok


__all__ = ["Artifact", "F_ARTIFACT", "F_AVAILABILITY", "F_FEASIBLE", "INCIDENT_CLASSES", "INCIDENT_FIELDS", "Operator",
           "Organization", "ReliabilityRegistry", "SLO_NAMES", "ShiftWindow", "SupplyChain", "availability",
           "feasible_actions"]
