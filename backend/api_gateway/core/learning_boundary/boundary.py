"""Граница обучения (мастер-ТЗ F1, стр. 15): вход конвейера кандидатов.

    сырая невязка/обратная связь → фильтр шума/аномалий/атак → очистка PII → типизированный кандидат
    (graph / rule / LoRA / threshold / geometry) → симуляция + ограничения + golden replay → человек/политика
    → shadow/canary → версионированное развёртывание / откат        (дальше — world_model.learning_pipeline)

    Δ = |State_observed − State_predicted|         (для 3D/пространства — норма вектора/занятости)
    θ_{t+1} = θ_t − η ∇_θ L(y_obs, y_pred)          (в ТЗ напечатан «+»; для функции потерь это рост ошибки —
                                                    см. LEARNING_SIGN_NOTE, реализован спуск)

Ничто отсюда не меняет прод: результат — кандидат со статусом candidate. Пространственная невязка хранится
как свидетельство и может породить кандидата, но никогда не обновляет прод напрямую.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..world_model.reality_formulas import FormulaReference, learning_parameter_candidate

F_RESIDUAL = FormulaReference("MASTER-F1-RESIDUAL", "CURE-REALITY-ENGINE-MASTER-TZ", 15,
                              "Δ = |State_observed − State_predicted|")
F_CALIBRATION = FormulaReference("MASTER-F1-CALIBRATION", "CURE-REALITY-ENGINE-MASTER-TZ", 15,
                                 "θ_{t+1} = θ_t + η∇θ L(y_obs, y_pred) (printed; implemented as descent θ − η∇L)")


class LearningBoundaryError(ValueError):
    pass


class CandidateKind(str, Enum):
    GRAPH = "graph"
    RULE = "rule"
    LORA = "lora"
    THRESHOLD = "threshold"
    GEOMETRY = "geometry"


class FilterVerdict(str, Enum):
    LEARNING_SIGNAL = "learning_signal"
    NOISE = "noise"
    ANOMALY_QUARANTINED = "anomaly_quarantined"
    ADVERSARIAL_REJECTED = "adversarial_rejected"


@dataclass(frozen=True)
class BoundaryPolicy:
    policy_version: str
    noise_sigma_multiple: float          # |Δ|/σ ≤ k → шум
    anomaly_robust_z: float              # робастный z по MAD окна > z → аномалия (карантин)
    min_window: int                      # минимум невязок для оценки аномалии
    pii_keys: tuple[str, ...]
    pii_patterns: tuple[str, ...]        # регулярные выражения (почта, телефон, номер и т.п.) из data governance

    def __post_init__(self) -> None:
        if not str(self.policy_version).strip() or not self.pii_keys:
            raise LearningBoundaryError("policy_version and PII keys are required (data governance)")
        if float(self.noise_sigma_multiple) < 0 or float(self.anomaly_robust_z) <= 0 or int(self.min_window) < 3:
            raise LearningBoundaryError("noise multiple ≥ 0, anomaly z > 0 and window ≥ 3 are required")
        for p in self.pii_patterns:
            re.compile(p)


@dataclass(frozen=True)
class ResidualRecord:
    subject: str
    observed: tuple[float, ...]
    predicted: tuple[float, ...]
    noise_sigma: float
    source_integrity: str                # "verified" или причина отказа из шлюза
    evidence_ref: str
    spatial: bool = False

    @property
    def delta(self) -> float:
        o, p = np.asarray(self.observed, float), np.asarray(self.predicted, float)
        if o.shape != p.shape or o.size == 0:
            raise LearningBoundaryError("observed and predicted state must have the same non-empty shape")
        return float(np.linalg.norm(o - p))


@dataclass
class FilterResult:
    record: ResidualRecord
    delta: float
    verdict: FilterVerdict
    reason: str
    stored_as_evidence: bool = True       # невязка всегда хранится как свидетельство


class ResidualFilter:
    def __init__(self, policy: BoundaryPolicy) -> None:
        self.policy = policy
        self._window: dict[str, list[float]] = {}

    def assess(self, r: ResidualRecord) -> FilterResult:
        d = r.delta
        if r.source_integrity != "verified":
            return FilterResult(r, d, FilterVerdict.ADVERSARIAL_REJECTED, f"source_integrity:{r.source_integrity}")
        if r.noise_sigma <= 0:
            raise LearningBoundaryError("noise σ must be positive")
        if d / r.noise_sigma <= self.policy.noise_sigma_multiple:
            return FilterResult(r, d, FilterVerdict.NOISE, "within_noise")
        w = self._window.setdefault(r.subject, [])
        verdict, reason = FilterVerdict.LEARNING_SIGNAL, "persistent_residual_above_noise"
        if len(w) >= self.policy.min_window:
            med = float(np.median(w))
            mad = float(np.median(np.abs(np.array(w) - med))) * 1.4826       # согласование MAD с σ нормального
            if mad > 0 and abs(d - med) / mad > self.policy.anomaly_robust_z:
                verdict, reason = FilterVerdict.ANOMALY_QUARANTINED, "isolated_outlier_vs_recent_residuals"
        if verdict is FilterVerdict.LEARNING_SIGNAL:
            w.append(d)
        return FilterResult(r, d, verdict, reason)


def sanitize(payload: Any, policy: BoundaryPolicy) -> tuple[Any, list[str]]:
    removed: list[str] = []
    keys = {k.lower() for k in policy.pii_keys}
    pats = [re.compile(p) for p in policy.pii_patterns]

    def go(v, path):
        if isinstance(v, Mapping):
            out = {}
            for k, x in v.items():
                if str(k).lower() in keys:
                    removed.append(f"{path}.{k}")
                    continue
                out[k] = go(x, f"{path}.{k}")
            return out
        if isinstance(v, (list, tuple)):
            return type(v)(go(x, f"{path}[{i}]") for i, x in enumerate(v))
        if isinstance(v, str):
            s = v
            for p in pats:
                if p.search(s):
                    removed.append(f"{path}:pattern")
                    s = p.sub("[redacted]", s)
            return s
        return v

    return go(payload, "payload"), removed


@dataclass
class TypedCandidate:
    kind: CandidateKind
    subject: str
    payload: Mapping[str, Any]
    evidence_refs: tuple[str, ...]
    pii_removed: list = field(default_factory=list)
    learning_candidate: Any = None        # для threshold/параметров — LearningCandidate (спуск по потерям)
    status: str = field(init=False, default="candidate")
    real_action: bool = field(init=False, default=False)
    next_stages: tuple[str, ...] = field(init=False, default=(
        "simulation+constraints+golden_replay", "human/policy_review", "shadow/canary",
        "versioned_deployment_or_rollback"))


def make_candidate(kind: CandidateKind, *, subject: str, payload: Mapping[str, Any],
                   from_results: Sequence[FilterResult], policy: BoundaryPolicy, proposed_by: str,
                   theta: Optional[Sequence[float]] = None, learning_rate: Optional[float] = None,
                   loss_gradient: Optional[Sequence[float]] = None,
                   parameter_names: Optional[Sequence[str]] = None) -> TypedCandidate:
    kind = CandidateKind(kind)
    signal = [f for f in from_results if f.verdict is FilterVerdict.LEARNING_SIGNAL]
    if not signal:
        raise LearningBoundaryError("no filtered learning signal: noise, anomalies and rejected inputs do not train")
    clean, removed = sanitize(dict(payload), policy)
    refs = tuple(f.record.evidence_ref for f in signal)
    lc = None
    if kind is CandidateKind.THRESHOLD:
        if theta is None or learning_rate is None or loss_gradient is None or not parameter_names:
            raise LearningBoundaryError("a threshold/parameter candidate needs θ, η, ∇L and parameter names")
        lc = learning_parameter_candidate(theta=theta, learning_rate=learning_rate, loss_gradient=loss_gradient,
                                          target=subject, parameter_names=parameter_names, proposed_by=proposed_by,
                                          evidence_refs=refs).value
    return TypedCandidate(kind, subject, clean, refs, removed, lc)


__all__ = ["BoundaryPolicy", "CandidateKind", "F_CALIBRATION", "F_RESIDUAL", "FilterResult", "FilterVerdict",
           "LearningBoundaryError", "ResidualFilter", "ResidualRecord", "TypedCandidate", "make_candidate", "sanitize"]
