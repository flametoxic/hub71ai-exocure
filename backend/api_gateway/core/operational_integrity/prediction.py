"""WM-05 §3–§4: четыре разных представления, контракт невязки, лестница источников прогноза, допуск нейромодели.

    Raw Measurement ≠ Fused Estimate ≠ Dynamic State ≠ Prediction   (у каждого свой владелец)
    r_t = y_t^observed − ŷ_t^predicted;  PredictionResidual → Surprise → LearningSignalCandidate
    (невязка не меняет напрямую параметры динамики, доверие к причинным связям, пороги политики и веса модели)
    Promote(m) = BeatBaseline ∧ PhysicsPass ∧ IntervalCalibrated ∧ DegradedSuitePass ∧ Cana…
    (последний член в PDF обрезан по краю; прочитан как CanaryPass — это отмечено в сверке)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from ..world_model.reality_formulas import FormulaReference
from .context import DOC, DataMode, IntegrityError

F_RESIDUAL = FormulaReference("WM05-4.1-RESIDUAL", DOC, 5, "r_t = y_t^observed − ŷ_t^predicted")
F_PROMOTE = FormulaReference("WM05-4.3-PROMOTE", DOC, 6,
                             "Promote(m) = BeatBaseline ∧ PhysicsPass ∧ IntervalCalibrated ∧ DegradedSuitePass ∧ Cana… "
                             "(cut off; read as CanaryPass)")

OWNERS = {"MeasurementObservation": "perception_adapter", "FusedEstimate": "StateFusionService",
          "DynamicState": "DynamicStateService", "WorldPrediction": "PredictiveWorldModel"}
DYNAMIC_STATE_FIELDS = ("pose", "velocity", "acceleration", "trajectory", "active_track", "occupancy_geometry",
                        "on_lane", "approaching", "moving_toward", "blocking", "trajectory_conflict")


def require_owner(representation: str, writer: str) -> None:
    """Единственный писатель каждого представления (§3.1–3.3): VisionCore, энкодеры, латентные слои не пишут
    слитое физическое состояние."""
    owner = OWNERS.get(representation)
    if owner is None:
        raise IntegrityError(f"unknown representation {representation!r}")
    if writer != owner:
        raise IntegrityError(f"{writer} cannot publish {representation}; owner is {owner} (WM-05 §3)")


class PredictionSource(str, Enum):
    CALIBRATED_DYNAMICS = "approved_calibrated_domain_dynamics"
    NEURAL_PHYSICS = "shadow_approved_neural_physics"
    NEURAL_PREDICTOR = "existing_neural_world_predictor"
    HISTORICAL_PATTERN = "historical_pattern_model"
    LINEAR_EXTRAPOLATION = "linear_extrapolation"

    @property
    def rank(self) -> int:
        return list(PredictionSource).index(self)

    @property
    def physical(self) -> bool:
        return self in (PredictionSource.CALIBRATED_DYNAMICS, PredictionSource.NEURAL_PHYSICS)


@dataclass(frozen=True)
class PredictionEnvelope:
    """§4.2: всё, что обязан показать любой прогноз."""

    value: Any
    source_class: PredictionSource
    model_version: str
    physical: bool
    interval: tuple
    input_coverage: float
    sensor_health: Mapping[str, str]
    horizon_s: float
    validation_scope: str
    data_mode: DataMode
    representation: str = field(init=False, default="WorldPrediction")

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_class", PredictionSource(self.source_class))
        object.__setattr__(self, "data_mode", DataMode(self.data_mode))
        if not str(self.model_version).strip() or not str(self.validation_scope).strip():
            raise IntegrityError("prediction needs model_version and validation_scope")
        if self.physical and not self.source_class.physical:
            raise IntegrityError(f"{self.source_class.value} cannot claim to be a physical simulation (§4.3)")
        lo, hi = self.interval
        if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
            raise IntegrityError("prediction interval must be finite lo ≤ hi")
        if not 0.0 <= float(self.input_coverage) <= 1.0 or float(self.horizon_s) <= 0:
            raise IntegrityError("input_coverage in [0,1] and positive horizon required")


def select_prediction_source(available: Mapping[PredictionSource, bool]) -> PredictionSource:
    """Лестница §4.2: первый доступный сверху вниз; ничего нет — честный отказ, а не выдумка."""
    for s in PredictionSource:
        if available.get(s):
            return s
    raise IntegrityError("no prediction source available: abstain (no fabricated certainty)")


@dataclass(frozen=True)
class PredictionResidual:
    entity_id: str
    attribute: str
    observed: float
    predicted: float
    prediction_model: str
    data_mode: DataMode

    @property
    def r(self) -> float:
        return float(self.observed) - float(self.predicted)


@dataclass(frozen=True)
class LearningSignalCandidate:
    residual: PredictionResidual
    surprise: float
    target: str                         # что предлагается пересмотреть (только как кандидат)
    status: str = "candidate"
    mutates_production: bool = False


FORBIDDEN_DIRECT_TARGETS = ("dynamics_parameters", "causal_edge_trust", "policy_thresholds", "production_model_weights")


def residual_to_signal(res: PredictionResidual, *, sigma: float, target: str) -> LearningSignalCandidate:
    """r → Surprise (−log N(r; 0, σ²)) → кандидат; прямое изменение запрещённых целей невозможно по типу."""
    if sigma <= 0:
        raise IntegrityError("residual σ must be positive")
    surprise = 0.5 * (res.r / sigma) ** 2 + math.log(sigma * math.sqrt(2 * math.pi))
    return LearningSignalCandidate(res, surprise, target)


@dataclass(frozen=True)
class NeuralPromotionEvidence:
    beat_baseline: bool
    physics_pass: bool
    interval_calibrated: bool
    degraded_suite_pass: bool
    canary_pass: bool
    evidence_refs: tuple = ()


def promote_neural_model(ev: NeuralPromotionEvidence) -> dict:
    checks = {"BeatBaseline": ev.beat_baseline, "PhysicsPass": ev.physics_pass,
              "IntervalCalibrated": ev.interval_calibrated, "DegradedSuitePass": ev.degraded_suite_pass,
              "CanaryPass": ev.canary_pass}
    for k, v in checks.items():
        if not isinstance(v, bool):
            raise IntegrityError(f"{k} must be a strict bool")
    ok = all(checks.values())
    return {"promote": ok, "failed": [k for k, v in checks.items() if not v], "may_call_itself_physical": ok,
            "formula": F_PROMOTE, "note": "promotion makes the model a shadow-approved physics source, not production"}


def ladder_report(envs: Sequence[PredictionEnvelope]) -> Optional[dict]:
    if not envs:
        return None
    best = min(envs, key=lambda e: e.source_class.rank)
    return {"used": best.source_class.value, "physical": best.physical, "data_mode": best.data_mode.value,
            "alternatives": [e.source_class.value for e in envs if e is not best]}


__all__ = ["DYNAMIC_STATE_FIELDS", "FORBIDDEN_DIRECT_TARGETS", "F_PROMOTE", "F_RESIDUAL", "LearningSignalCandidate",
           "NeuralPromotionEvidence", "OWNERS", "PredictionEnvelope", "PredictionResidual", "PredictionSource",
           "ladder_report", "promote_neural_model", "require_owner", "residual_to_signal", "select_prediction_source"]
