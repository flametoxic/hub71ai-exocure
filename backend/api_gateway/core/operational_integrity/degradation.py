"""DegradationManager (WM Engineering Spec §2.20; Gap Closure v1 §11; WM-05 §8.2; мастер-ТЗ DoD 10).

report(component, reason): 1) статус в состоянии рантайма, 2) событие аудита, 3) оповещение, 4) хук полномочий →
ограничение. Ни один отказ не тихий, полномочия падают при деградации:
    журнал/хранилище в памяти → ALLOW выключен (только тень);
    граф/векторное хранилище/память → нет ALLOW без свидетельств;
    GPU/симулятор → DEFER (действие откладывается, а не исполняется без доказательства);
    модель → никакой подмены LLM под причинную рекомендацию;
    политика → только аппаратно-безопасное поведение.
Правила «компонент → ограничение» задаёт политика; неизвестный компонент ограничивает сильнее всего (fail-closed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Mapping, Optional

from .context import IntegrityError

RESTRICTIONS = ("allow_off", "no_allow_without_evidence", "defer", "no_llm_fallback", "hardware_safe_only")
SEVERITY = {"allow_off": 3, "hardware_safe_only": 4, "defer": 2, "no_allow_without_evidence": 1, "no_llm_fallback": 1}


@dataclass
class DegradationStatus:
    component: str
    reason: str
    restriction: str
    since: str
    active: bool = True


@dataclass
class DegradationManager:
    rules: Mapping[str, str]                                    # компонент → ограничение
    authority_hook: Optional[Callable[[str, str], None]] = None
    alert: Optional[Callable[[dict], None]] = None
    audit: list = field(default_factory=list)
    _status: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        bad = {v for v in self.rules.values() if v not in RESTRICTIONS}
        if bad:
            raise IntegrityError(f"unknown restrictions {bad}")

    def report(self, component: str, reason: str, *, at: datetime) -> DegradationStatus:
        if not str(reason).strip():
            raise IntegrityError("a degradation needs a reason (no silent fallback)")
        restriction = self.rules.get(component, "hardware_safe_only")
        st = DegradationStatus(component, reason, restriction, at.isoformat())
        self._status[component] = st
        ev = {"event": "degradation.activated", "component": component, "reason": reason,
              "restriction": restriction, "at": at.isoformat()}
        self.audit.append(ev)
        if self.alert is not None:
            self.alert(ev)
        if self.authority_hook is not None:
            self.authority_hook(component, restriction)
        return st

    def recover(self, component: str, *, evidence: str, at: datetime) -> None:
        st = self._status.get(component)
        if st is None or not st.active:
            return
        if not str(evidence).strip():
            raise IntegrityError("recovery needs evidence (health check / test reference)")
        st.active = False
        self.audit.append({"event": "degradation.recovered", "component": component, "evidence": evidence,
                           "at": at.isoformat()})

    def current(self) -> dict:
        active = {c: s.restriction for c, s in self._status.items() if s.active}
        worst = max(active.values(), key=lambda r: SEVERITY[r]) if active else None
        return {"degraded": bool(active), "components": active, "strongest_restriction": worst,
                "data_mode_note": "degraded_volatile" if any(r == "allow_off" for r in active.values()) else None}

    def authority(self, *, requested: str, has_evidence: bool, has_simulation_proof: bool,
                  recommendation_source: str = "causal_engine") -> dict:
        """Итоговое право на действие при текущих деградациях: allow | shadow | defer | block."""
        cur = self.current()["components"].values()
        if "hardware_safe_only" in cur:
            return {"authority": "block", "reason": "policy/critical component degraded: hardware-safe behaviour only"}
        if "allow_off" in cur and requested == "allow":
            return {"authority": "shadow", "reason": "durable storage degraded: ALLOW off"}
        if "defer" in cur and not has_simulation_proof:
            return {"authority": "defer", "reason": "simulation/GPU degraded: no proof, action deferred"}
        if "no_allow_without_evidence" in cur and not has_evidence:
            return {"authority": "shadow", "reason": "retrieval degraded: no ALLOW without evidence"}
        if "no_llm_fallback" in cur and recommendation_source == "llm":
            return {"authority": "block", "reason": "model degraded: LLM fallback cannot pose as causal recommendation"}
        return {"authority": requested, "reason": None}


__all__ = ["DegradationManager", "DegradationStatus", "RESTRICTIONS"]
