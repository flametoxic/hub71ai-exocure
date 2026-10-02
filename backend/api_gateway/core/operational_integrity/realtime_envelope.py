"""Gap Closure v1 §11 и формула L_e2e (стр. 13): реальное время по классам действий.

    L_e2e = L_sense + L_validation + L_network + L_state + L_reasoning + L_policy + L_actuation

Для каждого класса действия — конверт: max_end_to_end_latency, max_staleness, minimum_confidence,
degraded_mode_behavior, human_approval_requirement. Решение: измеренная задержка по всем 7 звеньям,
возраст данных и уверенность сверяются с конвертом; нарушение → поведение деградации из конверта
(никогда не «продолжить молча»). Числа — только из политики площадки.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from ..world_model.reality_formulas import FormulaReference
from .context import IntegrityError

F_E2E = FormulaReference("GAP1-13-L-E2E", "EXO_world_model_gap_closure_TZ", 13,
                         "L_e2e = L_sense + L_validation + L_network + L_state + L_reasoning + L_policy + L_actuation")
STAGES = ("sense", "validation", "network", "state", "reasoning", "policy", "actuation")
DEGRADED = ("block", "defer", "shadow_only", "hardware_safe_state")


@dataclass(frozen=True)
class RealTimeEnvelope:
    action_class: str
    max_end_to_end_latency_s: float
    max_staleness_s: float
    minimum_confidence: float
    degraded_mode_behavior: str
    human_approval_required: bool

    def __post_init__(self) -> None:
        if not self.action_class.strip() or self.degraded_mode_behavior not in DEGRADED:
            raise IntegrityError(f"action_class required and degraded_mode_behavior ∈ {DEGRADED}")
        for n in ("max_end_to_end_latency_s", "max_staleness_s"):
            v = float(getattr(self, n))
            if not (math.isfinite(v) and v > 0):
                raise IntegrityError(f"{n} must be a positive number from site policy")
        if not 0.0 <= float(self.minimum_confidence) <= 1.0:
            raise IntegrityError("minimum_confidence must be in [0, 1]")


def end_to_end_latency(stages_s: Mapping[str, float]) -> float:
    missing = [s for s in STAGES if s not in stages_s]
    if missing:
        raise IntegrityError(f"latency of every stage must be measured, missing {missing}")
    vals = [float(stages_s[s]) for s in STAGES]
    if any(not math.isfinite(v) or v < 0 for v in vals):
        raise IntegrityError("stage latencies must be finite and non-negative")
    return sum(vals)


def check_envelope(env: RealTimeEnvelope, *, stages_s: Mapping[str, float], data_age_s: float, confidence: float,
                   human_approved: bool) -> dict:
    l_e2e = end_to_end_latency(stages_s)
    reasons = []
    if l_e2e > env.max_end_to_end_latency_s:
        reasons.append(f"latency {l_e2e:.4f}s > {env.max_end_to_end_latency_s}s")
    if not math.isfinite(float(data_age_s)) or float(data_age_s) > env.max_staleness_s:
        reasons.append(f"data age {data_age_s}s > {env.max_staleness_s}s")
    if not math.isfinite(float(confidence)) or float(confidence) < env.minimum_confidence:
        reasons.append(f"confidence {confidence} < {env.minimum_confidence}")
    if env.human_approval_required and human_approved is not True:
        reasons.append("human approval required")
    return {"action_class": env.action_class, "l_e2e_s": l_e2e, "within_envelope": not reasons, "reasons": reasons,
            "behavior": "proceed_to_safety_gate" if not reasons else env.degraded_mode_behavior,
            "formula": F_E2E.formula_id}


__all__ = ["DEGRADED", "F_E2E", "RealTimeEnvelope", "STAGES", "check_envelope", "end_to_end_latency"]
