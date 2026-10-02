"""Политика пространственной модели мира и ссылки на формулы ТЗ.

Числа, которые заданы в ТЗ (TTL классов, веса и пороги разрешения сущностей, радиусы near, правило
дрейфа, χ²-гейт, порог cross-view), доступны через ``SpatialWorldPolicy.from_tz_defaults(...)``; всё, чего
ТЗ не задаёт (δ конфликта по атрибутам, σ пространственного совпадения по классам, классы атрибутов,
дисперсии доверия источников, ε для TTC, плотность ложных треков), обязано прийти от площадки.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from ..common.epistemic import deep_freeze
from ..world_model.reality_formulas import FormulaReference

MASTER = "CURE-REALITY-ENGINE-MASTER-TZ"
GAP1 = "EXO_world_model_gap_closure_TZ"
SPEC = "TZ-WM-ENGINEERING-SPEC"


def _f(fid: str, doc: str, page: int, expr: str) -> FormulaReference:
    return FormulaReference(fid, doc, page, expr)


F_FRESHNESS = _f("MASTER-A1-FRESHNESS", MASTER, 5, "freshness(a,t) = t − a.knowledge_time; stale ⟺ freshness > τ_attr")
F_STATE_AT = _f("MASTER-A1-STATE-AT", MASTER, 5,
                "State(T) = {a | a.knowledge_time ≤ T ∧ a.valid_from ≤ T ∧ (a.valid_to = ∅ ∨ a.valid_to > T)}")
F_CONFLICT = _f("MASTER-A1-CONFLICT", MASTER, 5,
                "conflict(a1,a2) ⟺ same(entity,attribute) ∧ observed(a1,a2) ∧ |v1 − v2| > δ_attribute")
F_IDENTITY = _f("MASTER-A2-SCORE", MASTER, 6,
                "Score(c|o) = 0.30 S_alias + 0.35 S_det + 0.15 S_spatial + 0.10 S_type + 0.10 S_temporal; "
                "S_spatial = exp(−d²/(2σ²))")
F_FRAME = _f("MASTER-A3-FRAME", MASTER, 7, "^W p = ^W T_S · ^S p; ^W T_sensor = ^W T_site · … · ^floor T_sensor")
F_COV = _f("MASTER-A3-COV", MASTER, 7, "Σ_W = R Σ_S R^T")
F_NEAR = _f("MASTER-A3-NEAR", MASTER, 7, "near(a,b,t,scale) ⟺ ‖p_a(t) − p_b(t)‖ < r_scale")
F_VISIBLE = _f("MASTER-A3-VISIBLE", MASTER, 7, "V(agent,obj,t) = FOV ∩ line_of_sight ∩ range ∩ ¬occlusion")
F_SWITCH = _f("MASTER-A3-SWITCH", MASTER, 7,
              "edge_old.valid_to = t_switch; edge_new.valid_from = t_switch; topology_version ← topology_version + 1")
F_IVW = _f("MASTER-A4-IVW", MASTER, 7, "x̂ = Σ z_i/σ_i² / Σ 1/σ_i²; σ̂² = 1/Σ 1/σ_i²; σ_i² = σ²_sensor,i + σ²_trust,i")
F_KALMAN = _f("MASTER-A4-KALMAN", MASTER, 7, "x̂⁻ = F x̂ + B u; P⁻ = F P F^T + Q; K = P⁻H^T(HP⁻H^T+R)^−1; …")
F_DRIFT = _f("MASTER-A4-DRIFT", MASTER, 8,
             "r_t = z_t − H x̂⁻; drift if > 30% of last 20 residuals exceed 3√S_t, S_t = H P⁻ H^T + R; "
             "on drift inflate variance ×10, sensor_health = degraded, never delete source")
F_GATE = _f("MASTER-A4-GATE", MASTER, 8, "d² = (z − Hx̂⁻)^T S^−1 (z − Hx̂⁻); associate when d² < 5.99")
F_TTC = _f("MASTER-A4-TTC", MASTER, 8, "TTC(a,b) = d(a,b) / max(v_closing, ε)")

#: значения, прямо напечатанные в ТЗ
TZ_TTL_SECONDS = {"telemetry": 300.0, "state": 3600.0, "config": 86400.0}     # topology — версионируется, TTL нет
TZ_IDENTITY_WEIGHTS = {"alias": 0.30, "det": 0.35, "spatial": 0.15, "type": 0.10, "temporal": 0.10}
TZ_IDENTITY_THRESHOLDS = {"resolved": 0.90, "candidate": 0.70, "conflict_delta": 0.05}
TZ_NEAR_RADII_M = {"building": 3.0, "traffic": 30.0, "district": 500.0}
TZ_DRIFT = {"window": 20, "ratio": 0.30, "k_sigma": 3.0, "inflation": 10.0}
TZ_GATE_CHI2_95_DOF2 = 5.99
TZ_CROSS_VIEW_MIN = 0.85


class SpatialWorldError(ValueError):
    pass


def aware(name: str, v: Any) -> datetime:
    if not isinstance(v, datetime) or v.tzinfo is None or v.utcoffset() is None:
        raise SpatialWorldError(f"{name} must be a timezone-aware datetime")
    return v.astimezone(timezone.utc)


def text(name: str, v: Any) -> str:
    if not isinstance(v, str) or not v.strip():
        raise SpatialWorldError(f"{name} is required")
    return v.strip()


def finite(name: str, v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise SpatialWorldError(f"{name} must be a finite number")
    return float(v)


@dataclass(frozen=True)
class SpatialWorldPolicy:
    policy_version: str
    attribute_class: Mapping[str, str]            # атрибут → telemetry | state | config | topology
    ttl_seconds: Mapping[str, float]              # класс → τ
    conflict_delta: Mapping[str, float]           # атрибут → δ
    conflict_window_s: float                      # наблюдения разных источников в пределах окна сравниваются
    identity_weights: Mapping[str, float]
    identity_thresholds: Mapping[str, float]
    spatial_sigma_m: Mapping[str, float]          # класс сущности → σ для S_spatial
    near_radii_m: Mapping[str, float]
    drift: Mapping[str, float]
    gate_chi2: float
    cross_view_min: float
    source_trust_variance: Mapping[str, float]    # источник → σ²_trust (базовая)
    staleness_variance_rate: float                # рост σ²_trust на секунду возраста наблюдения
    ttc_epsilon_mps: float
    clutter_density: float                        # плотность ложных совпадений для LR cross-view
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text("policy_version", self.policy_version)
        for n in ("attribute_class", "ttl_seconds", "conflict_delta", "identity_weights", "identity_thresholds",
                  "spatial_sigma_m", "near_radii_m", "drift", "source_trust_variance"):
            object.__setattr__(self, n, deep_freeze(dict(getattr(self, n))))
        bad = {c for c in self.attribute_class.values() if c not in ("telemetry", "state", "config", "topology")}
        if bad:
            raise SpatialWorldError(f"unknown attribute classes {bad}")
        for c in {c for c in self.attribute_class.values() if c != "topology"}:
            if c not in self.ttl_seconds or finite(f"ttl[{c}]", self.ttl_seconds[c]) <= 0:
                raise SpatialWorldError(f"TTL for class {c!r} is required")
        for k, v in self.conflict_delta.items():
            if finite(f"delta[{k}]", v) < 0:
                raise SpatialWorldError("conflict δ must be ≥ 0")
        if set(self.identity_weights) != {"alias", "det", "spatial", "type", "temporal"} or \
                not math.isclose(sum(self.identity_weights.values()), 1.0, abs_tol=1e-9):
            raise SpatialWorldError("identity weights must cover alias/det/spatial/type/temporal and sum to 1")
        t = self.identity_thresholds
        if not 0 < t["candidate"] < t["resolved"] <= 1 or not 0 < t["conflict_delta"] < 1:
            raise SpatialWorldError("need 0 < candidate < resolved ≤ 1 and 0 < conflict_delta < 1")
        if any(finite("sigma", v) <= 0 for v in self.spatial_sigma_m.values()):
            raise SpatialWorldError("spatial σ must be positive")
        for k in ("window", "ratio", "k_sigma", "inflation"):
            if k not in self.drift:
                raise SpatialWorldError(f"drift rule needs {k}")
        for n in ("conflict_window_s", "gate_chi2", "ttc_epsilon_mps", "clutter_density"):
            if finite(n, getattr(self, n)) <= 0:
                raise SpatialWorldError(f"{n} must be positive")
        if not 0 < finite("cross_view_min", self.cross_view_min) < 1:
            raise SpatialWorldError("cross_view_min must be in (0, 1)")
        if finite("staleness_variance_rate", self.staleness_variance_rate) < 0:
            raise SpatialWorldError("staleness_variance_rate must be ≥ 0")

    @classmethod
    def from_tz_defaults(cls, *, policy_version: str, attribute_class: Mapping[str, str],
                         conflict_delta: Mapping[str, float], conflict_window_s: float,
                         spatial_sigma_m: Mapping[str, float], source_trust_variance: Mapping[str, float],
                         staleness_variance_rate: float, ttc_epsilon_mps: float, clutter_density: float,
                         **overrides: Any) -> "SpatialWorldPolicy":
        """Числа ТЗ + обязательные параметры площадки."""
        kw = dict(policy_version=policy_version, attribute_class=attribute_class, ttl_seconds=TZ_TTL_SECONDS,
                  conflict_delta=conflict_delta, conflict_window_s=conflict_window_s,
                  identity_weights=TZ_IDENTITY_WEIGHTS, identity_thresholds=TZ_IDENTITY_THRESHOLDS,
                  spatial_sigma_m=spatial_sigma_m, near_radii_m=TZ_NEAR_RADII_M, drift=TZ_DRIFT,
                  gate_chi2=TZ_GATE_CHI2_95_DOF2, cross_view_min=TZ_CROSS_VIEW_MIN,
                  source_trust_variance=source_trust_variance, staleness_variance_rate=staleness_variance_rate,
                  ttc_epsilon_mps=ttc_epsilon_mps, clutter_density=clutter_density)
        kw.update(overrides)
        return cls(**kw)

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> "SpatialWorldPolicy":
        m = dict(m)
        if m.pop("use_tz_defaults", False):
            return cls.from_tz_defaults(**m)
        return cls(**m)

    def ttl_for(self, attribute: str) -> float | None:
        c = self.attribute_class.get(attribute)
        if c is None:
            raise SpatialWorldError(f"attribute {attribute!r} has no class in policy (TTL unknown — fail-closed)")
        return None if c == "topology" else float(self.ttl_seconds[c])


__all__ = [n for n in dir() if n.startswith(("F_", "TZ_")) or n in (
    "GAP1", "MASTER", "SPEC", "SpatialWorldError", "SpatialWorldPolicy", "aware", "finite", "text")]
