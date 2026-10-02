"""Assurance case на класс действий (мастер-ТЗ E3, стр. 15; DoD 13).

Каждый класс действий имеет версионированный assurance case: требования к свидетельствам, пороги
неопределённости, гейт симуляции, таймаут/идемпотентность/откат, политика деградации, тесты. Кейс привязан
к версии возможности (capability_version); смена возможности без повышения версии кейса — провал CI
(``ci_check`` и ``scripts/check_assurance_cases.py``). Кейс подписан ключом управления; неподписанный
не регистрируется. Нет кейса для (action_class, capability_version) — класс не исполняется.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from ..common.epistemic import deep_freeze
from ..world_model.reality_formulas import FormulaReference
from .contracts import SovereignError, digest, finite, text
from .trust import verify_artifact

F_ASSURANCE = FormulaReference("MASTER-E3-ASSURANCE", "CURE-REALITY-ENGINE-MASTER-TZ", 15,
                               "every action class has a versioned assurance case; CI fails if capability changes "
                               "without assurance case version bump")

_FIDELITY = ("L0", "L1", "L2", "L3", "L4", "L5")
_USE = ("none", "informational", "review", "planning_support")


def _version(v: str) -> tuple[int, ...]:
    try:
        parts = tuple(int(x) for x in text("case_version", v).split("."))
    except ValueError as exc:
        raise SovereignError(f"case_version {v!r} must be dotted integers (e.g. 1.2.0)") from exc
    if not parts or any(p < 0 for p in parts):
        raise SovereignError("case_version must be non-negative dotted integers")
    return parts


@dataclass(frozen=True)
class AssuranceCase:
    action_class: str
    case_version: str
    capability_version: str
    evidence_requirements: tuple[str, ...]              # виды свидетельств, которые должен приложить запрос
    simulation_required: bool
    min_simulation_fidelity: Optional[str]             # L0…L5
    min_simulation_use: Optional[str]                  # informational / review / planning_support
    max_uncertainty_std: Mapping[str, float]           # метрика пакета симуляции → предел std
    max_failure_probability: Optional[float]           # предел доли «провальных» прогонов (хвост)
    timeout_s: float                                   # окно верификации эффекта
    idempotency_required: bool
    rollback: tuple[str, ...]                          # шаги отката/компенсации
    degradation_policy: Mapping[str, str]              # состояние деградации → поведение
    tests: tuple[str, ...]                             # идентификаторы тестов, доказывающих кейс
    approved_by: str
    case_digest: str = ""

    def __post_init__(self) -> None:
        for n in ("action_class", "capability_version", "approved_by"):
            object.__setattr__(self, n, text(n, getattr(self, n)))
        _version(self.case_version)
        if not self.evidence_requirements:
            raise SovereignError("an assurance case must name its evidence requirements")
        if not self.tests:
            raise SovereignError("an assurance case must reference the tests that support it")
        if not self.rollback:
            raise SovereignError("an assurance case must define rollback/compensation")
        if not self.idempotency_required:
            raise SovereignError("idempotency is mandatory for physical action classes (§3.3)")
        if finite("timeout_s", self.timeout_s) <= 0:
            raise SovereignError("timeout must be positive")
        for need in ("journal_unavailable", "world_state_stale", "policy_invalid", "model_unavailable"):
            if need not in self.degradation_policy:
                raise SovereignError(f"degradation policy must cover {need!r} (§9)")
        if self.simulation_required:
            if self.min_simulation_fidelity not in _FIDELITY or self.min_simulation_use not in _USE[1:]:
                raise SovereignError("simulation gate needs min fidelity (L0–L5) and min downstream use")
            if not self.max_uncertainty_std:
                raise SovereignError("simulation gate needs explicit uncertainty thresholds")
        for k, v in dict(self.max_uncertainty_std).items():
            if finite(f"max_uncertainty_std[{k}]", v) < 0:
                raise SovereignError("uncertainty thresholds must be non-negative")
        if self.max_failure_probability is not None and not 0 <= finite("max_failure_probability",
                                                                          self.max_failure_probability) <= 1:
            raise SovereignError("max_failure_probability must be in [0, 1]")
        for n in ("evidence_requirements", "rollback", "tests"):
            object.__setattr__(self, n, tuple(text(n, x) for x in getattr(self, n)))
        object.__setattr__(self, "max_uncertainty_std", deep_freeze({str(k): float(v) for k, v in
                                                                     dict(self.max_uncertainty_std).items()}))
        object.__setattr__(self, "degradation_policy", deep_freeze(dict(self.degradation_policy)))
        d = digest(self.body())
        if self.case_digest and self.case_digest != d:
            raise SovereignError("assurance case digest mismatch")
        object.__setattr__(self, "case_digest", d)

    def body(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "case_digest"}

    # ------------------------------------------------------------------ проверка запроса по кейсу
    def evidence_gaps(self, evidence: Mapping[str, str]) -> tuple[str, ...]:
        return tuple(f"missing_evidence:{e}" for e in self.evidence_requirements if not str(evidence.get(e) or "").strip())

    def simulation_gaps(self, pack, *, trace_id: str) -> tuple[str, ...]:
        """Пакет симуляции — только предусловие (у него нет права на действие)."""
        if not self.simulation_required:
            return ()
        if pack is None:
            return ("simulation_pack_missing",)
        out = []
        if getattr(pack, "trace_id", None) != trace_id:
            out.append("simulation_pack_for_other_trace")
        if not getattr(pack, "valid_evidence", False):
            out.append(f"simulation_pack_not_valid_evidence:{getattr(pack, 'status', '?')}")
        fid = getattr(getattr(pack, "fidelity_level", None), "value", None)
        if fid not in _FIDELITY or _FIDELITY.index(fid) < _FIDELITY.index(self.min_simulation_fidelity):
            out.append(f"simulation_fidelity_{fid}_below_{self.min_simulation_fidelity}")
        use = getattr(getattr(pack, "allowed_downstream_use", None), "value", None)
        if use not in _USE or _USE.index(use) < _USE.index(self.min_simulation_use):
            out.append(f"simulation_use_{use}_below_{self.min_simulation_use}")
        unc = dict(getattr(pack, "uncertainty", {}) or {})
        for metric, lim in self.max_uncertainty_std.items():
            u = unc.get(metric)
            std = u.get("std") if isinstance(u, Mapping) else None
            if std is None:
                out.append(f"uncertainty_not_estimated:{metric}")
            elif float(std) > lim:
                out.append(f"uncertainty_above_threshold:{metric}")
        if self.max_failure_probability is not None:
            fp = dict(getattr(pack, "tail_risk", {}) or {}).get("failure_probability")
            if fp is None:
                out.append("tail_risk_not_estimated")
            elif float(fp) > self.max_failure_probability:
                out.append("failure_probability_above_threshold")
        return tuple(out)


class AssuranceRegistry:
    def __init__(self, *, governance_key: bytes) -> None:
        self._key = governance_key
        self._cases: dict[str, list[AssuranceCase]] = {}

    def register(self, case: AssuranceCase, signature: str) -> AssuranceCase:
        if not verify_artifact(case.body(), signature, self._key):
            raise SovereignError("assurance case signature invalid: not registered")
        hist = self._cases.setdefault(case.action_class, [])
        if hist:
            last = hist[-1]
            if _version(case.case_version) <= _version(last.case_version):
                raise SovereignError(f"assurance case for {case.action_class} must bump its version "
                                     f"({case.case_version} ≤ {last.case_version})")
        hist.append(case)
        return case

    def active(self, action_class: str) -> Optional[AssuranceCase]:
        hist = self._cases.get(action_class)
        return hist[-1] if hist else None

    def for_action(self, action_class: str, capability_version: str) -> Optional[AssuranceCase]:
        c = self.active(action_class)
        return c if c is not None and c.capability_version == capability_version else None

    def ci_check(self, capabilities: Mapping[str, str]) -> tuple[str, ...]:
        """capabilities: action_class → текущая capability_version. Любое расхождение — провал CI."""
        fails = []
        for ac, cv in sorted(capabilities.items()):
            c = self.active(ac)
            if c is None:
                fails.append(f"{ac}: no assurance case")
            elif c.capability_version != cv:
                fails.append(f"{ac}: capability {cv} changed without assurance case version bump "
                             f"(case {c.case_version} covers {c.capability_version})")
        return tuple(fails)


__all__ = ["AssuranceCase", "AssuranceRegistry", "F_ASSURANCE"]
