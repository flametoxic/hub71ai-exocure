from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .spatial_formulas import FrameUnresolvedError as _AnyFrameUnresolved
from .assertions import FrameUnresolvedError, finite_real, reject_non_finite, require_aware
from .reality_formulas import (
    ROTATION_ATOL,
    FormulaResult,
    ScalarFusionResult,
    body_capability,
    epistemic_action_gate,
    strict_bool,
    validate_covariance,
    visibility_state,
)
from .spatial_formulas import FrameRegistry, FrameResolution, VisibilityState, transform_pose_state


def _unit_interval(name: str, value: Any) -> float:
    number = finite_real(name, value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _non_negative_real(name: str, value: Any) -> float:
    number = finite_real(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


_strict_bool = strict_bool  # one strict-bool definition for the whole UMRM layer (reality_formulas)


def _names(name: str, value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of ids, not a single string")
    items = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in items) or len(set(items)) != len(items):
        raise ValueError(f"{name} must contain unique non-empty ids")
    return items


def _bool_map(name: str, value: Any) -> Mapping[str, bool]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping id -> bool")
    return MappingProxyType({_names(name, (k,))[0]: _strict_bool(f"{name}[{k!r}]", v) for k, v in value.items()})


def _unit_map(name: str, value: Any) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping id -> [0, 1]")
    return MappingProxyType({_names(name, (k,))[0]: _unit_interval(f"{name}[{k!r}]", v) for k, v in value.items()})


def _vector(name: str, value: Any, size: int) -> tuple[float, ...]:
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of {size} finite numbers") from exc
    if len(items) != size:
        raise ValueError(f"{name} must contain exactly {size} values")
    return tuple(finite_real(f"{name}[{index}]", item) for index, item in enumerate(items))


@dataclass(frozen=True)
class SpatialPose:
    """UMRM 3.1 (page 2): p = (x,y,z, q_x,q_y,q_z,q_w, v_x,v_y,v_z, w_x,w_y,w_z), Sigma = Cov(p).

    ``frame_resolved`` is COMPUTED (UMRM 3.2): the pose is resolved only if ``frame_registry`` holds a
    transform chain from ``frame_id`` to the registry root.  The caller cannot declare it.  Linear and
    angular velocities are expressed in ``frame_id``.
    """

    frame_id: str
    position: tuple[float, float, float]
    orientation_quaternion: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    covariance: Any
    observed_at: datetime
    frame_registry: FrameRegistry | None = field(default=None, compare=False, repr=False)
    frame_resolution: FrameResolution | None = field(init=False, default=None, compare=False)

    def __post_init__(self) -> None:
        """Fail-closed: named frame, finite components, unit quaternion, 13x13 finite symmetric PSD covariance."""
        object.__setattr__(self, "observed_at", require_aware(self.observed_at, "observed_at"))
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise FrameUnresolvedError("frame_unresolved: a pose must name the frame it is expressed in")
        position = _vector("position", self.position, 3)
        quaternion = _vector("orientation_quaternion", self.orientation_quaternion, 4)
        linear = _vector("linear_velocity", self.linear_velocity, 3)
        angular = _vector("angular_velocity", self.angular_velocity, 3)
        norm = float(np.linalg.norm(quaternion))
        if abs(norm - 1.0) > ROTATION_ATOL:
            raise ValueError(f"orientation_quaternion must be a unit quaternion (norm={norm})")
        covariance = validate_covariance(self.covariance, name="pose covariance", shape=(13, 13)).copy()
        covariance.setflags(write=False)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "orientation_quaternion", quaternion)
        object.__setattr__(self, "linear_velocity", linear)
        object.__setattr__(self, "angular_velocity", angular)
        object.__setattr__(self, "covariance", covariance)
        resolution = None
        if self.frame_registry is not None:
            if not isinstance(self.frame_registry, FrameRegistry):
                raise TypeError("frame_registry must be a FrameRegistry")
            try:
                resolution = self.frame_registry.resolve(self.frame_id, self.frame_registry.root)
            except _AnyFrameUnresolved:
                resolution = None  # honest frame_unresolved knowledge state
        object.__setattr__(self, "frame_resolution", resolution)

    @property
    def frame_resolved(self) -> bool:
        """Computed: a transform chain frame_id -> world root exists in the registry *now*."""
        if self.frame_resolution is None:
            return False
        try:
            self.frame_resolution.refreshed()
        except _AnyFrameUnresolved:
            return False
        return True

    def require_world_resolved(self) -> "SpatialPose":
        """Return self only if a transform chain to the world frame exists (UMRM 3.2, page 3)."""
        if not self.frame_resolved:
            raise FrameUnresolvedError(
                f"frame_unresolved: pose in frame '{self.frame_id}' has no transform chain to the world frame"
            )
        return self

    def as_vector(self) -> np.ndarray:
        return np.array(self.position + self.orientation_quaternion + self.linear_velocity + self.angular_velocity)

    def transformed_to(self, target_frame: str, registry: FrameRegistry | None = None) -> "SpatialPose":
        """The same pose expressed in ``target_frame``: full 13-D state and full 13x13 covariance
        (Sigma_T = J Sigma J^T, J = blockdiag(R, L(q_R), R, R)) through the registry chain."""
        registry = registry if registry is not None else self.frame_registry
        if registry is None:
            raise FrameUnresolvedError(f"frame_unresolved: no FrameRegistry to move pose from '{self.frame_id}'")
        out = registry.transform_pose_state(self.frame_id, target_frame, self.as_vector(), self.covariance).value
        v = out["vector"]
        return SpatialPose(
            frame_id=target_frame,
            position=tuple(v[0:3]),
            orientation_quaternion=tuple(v[3:7]),
            linear_velocity=tuple(v[7:10]),
            angular_velocity=tuple(v[10:13]),
            covariance=out["covariance"],
            observed_at=self.observed_at,
            frame_registry=registry,
        )


@dataclass(frozen=True)
class UniversalEntity:
    entity_id: str
    ontology_class: str
    scope: tuple[str, ...]
    semantic_state: Mapping[str, Any]
    spatial_pose: SpatialPose | None
    geometry_ref: str | None
    capabilities: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    confidence: float

    def __post_init__(self) -> None:
        if not self.entity_id.strip() or not self.ontology_class.strip() or not self.scope:
            raise ValueError("entity_id, ontology_class, and scope are required")
        _unit_interval("confidence", self.confidence)
        reject_non_finite("semantic_state", self.semantic_state)
        if self.spatial_pose is not None:
            if not isinstance(self.spatial_pose, SpatialPose):
                raise TypeError("spatial_pose must be a SpatialPose or None")
            # p_e is the entity's world-model pose: an unresolved frame cannot be stored as one (UMRM 3.2).
            self.spatial_pose.require_world_resolved()


@dataclass(frozen=True)
class InteroceptionState:
    """UMRM 4.4 (page 4): I_t = (energy, thermal load, sensor health, actuator availability, compute load,
    network latency, ...) -- what CURE "feels" about one of its bodies.  Part of the self-model, not an
    external object.

    Representation: ``energy``, ``thermal_load`` and ``compute_load`` are fractions in [0, 1] (of
    capacity / thermal limit / compute capacity); ``sensor_health`` in [0, 1] per sensor;
    ``actuator_availability`` a strict bool per actuator; ``network_latency_ms`` >= 0.  The TZ line is
    cut by the page edge after "network latenc..."; further components are not specified by the TZ
    and may only be carried in ``additional`` (finite numbers), never interpreted by the core.
    """

    body_id: str
    energy: float
    thermal_load: float
    sensor_health: Mapping[str, float]
    actuator_availability: Mapping[str, bool]
    compute_load: float
    network_latency_ms: float
    additional: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.body_id, str) or not self.body_id.strip():
            raise ValueError("body_id is required")
        for name in ("energy", "thermal_load", "compute_load"):
            _unit_interval(name, getattr(self, name))
        _non_negative_real("network_latency_ms", self.network_latency_ms)
        object.__setattr__(self, "sensor_health", _unit_map("sensor_health", self.sensor_health))
        object.__setattr__(self, "actuator_availability", _bool_map("actuator_availability", self.actuator_availability))
        if not isinstance(self.additional, Mapping):
            raise TypeError("additional must be a mapping")
        object.__setattr__(
            self, "additional", MappingProxyType({str(k): finite_real(f"additional[{k!r}]", v) for k, v in self.additional.items()})
        )

    def require_covers(self, body: "BodyState") -> None:
        """I_t of a body must report every sensor and actuator of that body (no silent gaps)."""
        if body.body_id != self.body_id:
            raise ValueError(f"interoception is for body '{self.body_id}', not '{body.body_id}'")
        missing_sensors = set(body.sensors) - set(self.sensor_health)
        missing_actuators = set(body.actuators) - set(self.actuator_availability)
        if missing_sensors or missing_actuators:
            raise ValueError(
                f"interoception misses sensors {sorted(missing_sensors)} / actuators {sorted(missing_actuators)} "
                f"of body '{body.body_id}'"
            )

    def actuators_available(self, body: "BodyState") -> bool:
        self.require_covers(body)
        return all(self.actuator_availability[a] for a in body.actuators)


@dataclass(frozen=True)
class BodyState:
    """UMRM 3.3 (page 3): b = (pose, sensors, actuators, capabilities, health, authority, compute,
    connectivity).  ``body_available`` and ``actuator_health`` are the two separate inputs of the
    UMRM 4.4 capability form (they are not derived from ``compute_available`` / ``health``)."""

    body_id: str
    pose: SpatialPose | None
    sensors: tuple[str, ...]
    actuators: tuple[str, ...]
    capabilities: tuple[str, ...]
    health: float
    authority: str
    compute_available: bool
    connectivity_available: bool
    body_available: bool
    actuator_health: float

    def __post_init__(self) -> None:
        if not self.body_id.strip() or not self.authority.strip():
            raise ValueError("body_id and authority are required")
        _unit_interval("health", self.health)
        _unit_interval("actuator_health", self.actuator_health)
        _strict_bool("compute_available", self.compute_available)
        _strict_bool("connectivity_available", self.connectivity_available)
        _strict_bool("body_available", self.body_available)
        object.__setattr__(self, "sensors", _names("sensors", self.sensors))
        object.__setattr__(self, "actuators", _names("actuators", self.actuators))
        object.__setattr__(self, "capabilities", _names("capabilities", self.capabilities))
        if self.pose is not None and not isinstance(self.pose, SpatialPose):
            raise TypeError("pose must be a SpatialPose or None")

    def capability(
        self,
        action: str,
        *,
        interoception: InteroceptionState,
        policy_allows: bool,
        in_scope: bool,
        network_quality: float,
        minimum_actuator_health: float,
        minimum_network_quality: float,
    ) -> FormulaResult:
        """UMRM 4.4 (page 4): Capability(action,t) = f(body availability, actuator health, network, policy, scope).

        Delegates to the single form :func:`reality_formulas.body_capability`: body availability =
        ``body_available``; actuator factor = ``actuator_health`` >= minimum AND every actuator of the
        body available in I_t (``interoception``, required); network = ``connectivity_available`` AND
        ``network_quality`` >= minimum; policy and scope are explicit caller decisions.  Thresholds
        have no value in the TZ and are required.  ``intermediates["blockers"]`` names every failure.
        """
        if not isinstance(interoception, InteroceptionState):
            raise TypeError("interoception (I_t) is required: it influences capability (UMRM 4.4)")
        return body_capability(
            capability_declared=action in self.capabilities,
            body_available=self.body_available,
            actuator_available=interoception.actuators_available(self),
            actuator_health=self.actuator_health,
            minimum_actuator_health=minimum_actuator_health,
            network_available=self.connectivity_available,
            network_quality=network_quality,
            minimum_network_quality=minimum_network_quality,
            policy_allows=policy_allows,
            in_scope=in_scope,
        )

    def can_support(self, capability: str, **gates: Any) -> bool:
        """Boolean view of :meth:`capability`; all UMRM 4.4 factors, I_t and thresholds are required."""
        return bool(self.capability(capability, **gates).value)


@dataclass(frozen=True)
class CandidateSpatialRelation:
    """UMRM 4.2 (page 4): "candidate spatial relation" of a sensor-bound observation.

    Always a candidate (``status == "candidate"``), never a fact such as "person is at X".  The
    position is expressed in ``frame_id``; it is world-grounded only when ``frame_registry`` resolves
    that frame (computed ``frame_resolved``).
    """

    subject_ref: str
    relation: str
    object_ref: str
    frame_id: str
    position: tuple[float, float, float] | None
    covariance: Any | None
    confidence: float
    status: str = field(init=False, default="candidate")
    real_action: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        for name in ("subject_ref", "relation", "object_ref", "frame_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is required")
        _unit_interval("confidence", self.confidence)
        if (self.position is None) != (self.covariance is None):
            raise ValueError("position and covariance come together (a position without uncertainty is not allowed)")
        if self.position is not None:
            object.__setattr__(self, "position", _vector("position", self.position, 3))
            cov = validate_covariance(self.covariance, name="relation covariance", shape=(3, 3)).copy()
            cov.setflags(write=False)
            object.__setattr__(self, "covariance", cov)


@dataclass(frozen=True)
class ObservationState:
    observation_id: str
    sensor_id: str
    body_id: str
    observed_at: datetime
    visibility: VisibilityState
    value: Any
    calibration_ref: str
    pose: SpatialPose | None
    confidence: float
    evidence_refs: tuple[str, ...]
    candidate_spatial_relation: CandidateSpatialRelation | None = None

    def __post_init__(self) -> None:
        """UMRM 4.2-4.3 (pages 3-4): a sensor-bound observation with visibility state.

        sensor-bound observation + timestamp + calibration + pose + occlusion (visibility) + confidence
        + candidate spatial relation (:class:`CandidateSpatialRelation`, never a fact).

        Consistency (fail-closed): an observed state (low / calibrated confidence) must carry a
        finite value and confidence > 0; a calibrated-confidence state also needs a calibration
        reference.  Any not-observed state (not observed, occluded, out of range, sensor
        unavailable) cannot carry a measured value -- missing detection is not absence.
        """
        object.__setattr__(self, "observed_at", require_aware(self.observed_at, "observed_at"))
        if not self.observation_id.strip() or not self.sensor_id.strip() or not self.body_id.strip():
            raise ValueError("observation_id, sensor_id, and body_id are required")
        object.__setattr__(self, "visibility", VisibilityState(self.visibility))
        confidence = _unit_interval("confidence", self.confidence)
        if not isinstance(self.calibration_ref, str):
            raise ValueError("calibration_ref must be a string")
        if self.pose is not None and not isinstance(self.pose, SpatialPose):
            raise TypeError("pose must be a SpatialPose or None")
        relation = self.candidate_spatial_relation
        if relation is not None:
            if not isinstance(relation, CandidateSpatialRelation):
                raise TypeError("candidate_spatial_relation must be a CandidateSpatialRelation")
            if not self.observed:
                raise ValueError(f"{self.visibility.value}: a not-observed state cannot carry a spatial relation")
        if self.observed:
            if self.value is None:
                raise ValueError(f"{self.visibility.value}: an observed state requires a measured value")
            reject_non_finite("value", self.value)
            if confidence <= 0.0:
                raise ValueError(f"{self.visibility.value}: an observed state requires confidence > 0")
            if self.visibility is VisibilityState.CALIBRATED and not self.calibration_ref.strip():
                raise ValueError(f"{VisibilityState.CALIBRATED.value} requires a calibration_ref")
        elif self.value is not None:
            raise ValueError(f"{self.visibility.value}: a not-observed state cannot carry a measured value")

    @property
    def observed(self) -> bool:
        return self.visibility in {VisibilityState.LOW_CONFIDENCE, VisibilityState.CALIBRATED}

    @property
    def asserts_absence(self) -> bool:
        # The specification explicitly forbids treating missing detection as absence.
        return False


@dataclass(frozen=True)
class SelfModelUncertainty:
    """UMRM 9 (page 7): epistemic and aleatoric uncertainty kept separate.

    ``epistemic`` is the result of ``reality_formulas.epistemic_uncertainty`` (UMRM-9 formula);
    ``aleatoric`` maps a belief variable to its ``weighted_scalar_fusion`` result -- "aleatoric
    uncertainty comes from physical variance / fusion covariance", so its value is the fusion variance.
    """

    epistemic: FormulaResult
    aleatoric: Mapping[str, ScalarFusionResult]

    def __post_init__(self) -> None:
        if not isinstance(self.epistemic, FormulaResult) or self.epistemic.formula.formula_id != "UMRM-9-EPISTEMIC-UNCERTAINTY":
            raise TypeError("epistemic must be the FormulaResult of epistemic_uncertainty (UMRM-9)")
        finite_real("epistemic", self.epistemic.value)
        if not isinstance(self.aleatoric, Mapping):
            raise TypeError("aleatoric must map variable -> ScalarFusionResult")
        for name, fused in self.aleatoric.items():
            if not isinstance(fused, ScalarFusionResult) or fused.formula.formula_id != "UMRM-5-SCALAR-FUSION":
                raise TypeError(f"aleatoric[{name!r}] must be a weighted_scalar_fusion result (fusion variance)")
        object.__setattr__(self, "aleatoric", MappingProxyType(dict(self.aleatoric)))

    @property
    def epistemic_value(self) -> float:
        return float(self.epistemic.value)

    def aleatoric_variance(self, variable: str) -> float:
        if variable not in self.aleatoric:
            raise KeyError(f"no fused aleatoric variance for '{variable}' (unknown is not zero)")
        return float(self.aleatoric[variable].variance)


@dataclass(frozen=True)
class SelfModelState:
    """UMRM 1 (page 1) and 9 (page 7): M_t = (sensor coverage, calibration, model error, capability,
    latency, policy authority, knowledge ..., active body, sensor/actuator availability, uncertainty).

    The M_t line of section 9 is cut by the page edge after "knowle..."; only the visible components
    and those named in section 1 are modelled.  ``actuator_availability`` is read from I_t
    (``interoception``, UMRM 4.4) so the self-model has one source for it.
    """

    sensor_coverage: float
    calibration: float
    model_error: float
    capabilities: tuple[str, ...]
    latency_ms: float
    policy_authority: str
    knowledge_gaps: tuple[str, ...]
    active_body_id: str
    sensor_availability: Mapping[str, bool]
    interoception: InteroceptionState
    uncertainty: SelfModelUncertainty

    def __post_init__(self) -> None:
        for name in ("sensor_coverage", "calibration"):
            _unit_interval(name, getattr(self, name))
        _non_negative_real("model_error", self.model_error)
        _non_negative_real("latency_ms", self.latency_ms)
        if not self.policy_authority.strip():
            raise ValueError("policy_authority is required")
        if any(not isinstance(gap, str) or not gap.strip() for gap in self.knowledge_gaps):
            raise ValueError("knowledge_gaps must contain non-empty names")
        if not isinstance(self.active_body_id, str) or not self.active_body_id.strip():
            raise ValueError("active_body_id is required (UMRM 1: M_t includes the active body)")
        object.__setattr__(self, "sensor_availability", _bool_map("sensor_availability", self.sensor_availability))
        if not isinstance(self.interoception, InteroceptionState):
            raise TypeError("interoception must be an InteroceptionState (UMRM 4.4: part of the self-model)")
        if self.interoception.body_id != self.active_body_id:
            raise ValueError("interoception must describe the active body")
        if not isinstance(self.uncertainty, SelfModelUncertainty):
            raise TypeError("uncertainty must be a SelfModelUncertainty (separate epistemic and aleatoric)")

    @property
    def actuator_availability(self) -> Mapping[str, bool]:
        return self.interoception.actuator_availability

    def require_consistent_with(self, body: BodyState) -> None:
        if body.body_id != self.active_body_id:
            raise ValueError(f"self-model active body is '{self.active_body_id}', not '{body.body_id}'")
        self.interoception.require_covers(body)
        missing = set(body.sensors) - set(self.sensor_availability)
        if missing:
            raise ValueError(f"sensor_availability misses sensors {sorted(missing)} of the active body")

    def capability(self, action: str, body: BodyState, **gates: Any) -> FormulaResult:
        """UMRM 4.4 with the self-model's I_t; the active body must be ``body``."""
        self.require_consistent_with(body)
        return body.capability(action, interoception=self.interoception, **gates)

    def action_gate(self, *, epistemic_uncertainty: float, action_class_threshold: float) -> FormulaResult:
        """UMRM 9 (page 7): U_epistemic > theta_action_class => DEFER.

        The self-model's ``knowledge_gaps`` are the missing observations/capabilities that a DEFER
        must name; a DEFER with no declared knowledge gap raises (fail-closed).
        """
        return epistemic_action_gate(
            epistemic_uncertainty=epistemic_uncertainty,
            action_class_threshold=action_class_threshold,
            missing_observations=self.knowledge_gaps,
        )

    def gate_from_uncertainty(self, *, action_class_threshold: float) -> FormulaResult:
        """The same gate on the self-model's own epistemic uncertainty (never the aleatoric part)."""
        return self.action_gate(epistemic_uncertainty=self.uncertainty.epistemic_value,
                                action_class_threshold=action_class_threshold)


@dataclass(frozen=True)
class RealityState:
    entities: Mapping[str, UniversalEntity]
    physical_state: Mapping[str, Any]
    spatial_topology: Mapping[str, Any]
    belief_state: Mapping[str, Any]
    causal_state: Mapping[str, Any]
    decision_state: Mapping[str, Any]
    self_model: SelfModelState
    snapshot_id: str
    observed_at: datetime
    bodies: Mapping[str, BodyState]
    active_body_id: str

    def __post_init__(self) -> None:
        """UMRM 1 (page 1) + 3.3 (page 3): BODIES_t = {b_1..b_n} with the active body that defines
        the perspective; the self-model must describe that active body."""
        if not self.snapshot_id.strip():
            raise ValueError("snapshot_id is required")
        object.__setattr__(self, "observed_at", require_aware(self.observed_at, "observed_at"))
        if any(key != entity.entity_id for key, entity in self.entities.items()):
            raise ValueError("entity mapping keys must equal canonical entity IDs")
        if not isinstance(self.bodies, Mapping) or not self.bodies:
            raise ValueError("BODIES_t must contain at least one body (UMRM 3.3)")
        for key, body in self.bodies.items():
            if not isinstance(body, BodyState) or key != body.body_id:
                raise ValueError("bodies must map body_id -> BodyState")
        object.__setattr__(self, "bodies", MappingProxyType(dict(self.bodies)))
        if self.active_body_id not in self.bodies:
            raise ValueError(f"active body '{self.active_body_id}' is not in BODIES_t")
        if not isinstance(self.self_model, SelfModelState):
            raise TypeError("self_model must be a SelfModelState")
        self.self_model.require_consistent_with(self.active_body)

    @property
    def active_body(self) -> BodyState:
        return self.bodies[self.active_body_id]

    def perspective(self, *, sensor_set: Any, calibration: Mapping[str, Any]) -> Any:
        """UMRM 3.3: Perspective_t = f(body_t, pose_t, sensor_set_t, calibration_t) of the active body."""
        from .perspective import perspective

        body = self.active_body
        if body.pose is None:
            raise FrameUnresolvedError(f"frame_unresolved: active body '{body.body_id}' has no pose")
        return perspective(body=body, pose=body.pose, sensor_set=sensor_set, calibration=calibration)

    def as_spec_tuple(self) -> tuple[Any, ...]:
        """Return R_t=(E, X_t, G_t, B_t, C_t, D_t, M_t), document 1 page 1."""
        return (
            self.entities,
            self.physical_state,
            self.spatial_topology,
            self.belief_state,
            self.causal_state,
            self.decision_state,
            self.self_model,
        )


def classify_visibility(
    *,
    sensor_available: bool,
    in_field_of_view: bool,
    line_of_sight: bool,
    in_range: bool,
    occluded: bool,
    detection_confidence: float | None,
    confidence_threshold: float,
    sensor_calibrated: bool,
) -> VisibilityState:
    """UMRM 4.3 (page 4): map V = FOV AND LoS AND Range AND NOT Occlusion to a VisibilityState."""
    result = visibility_state(
        sensor_available=sensor_available,
        in_field_of_view=in_field_of_view,
        line_of_sight=line_of_sight,
        in_range=in_range,
        occluded=occluded,
        detection_confidence=detection_confidence,
        confidence_threshold=confidence_threshold,
        sensor_calibrated=sensor_calibrated,
    )
    return VisibilityState(result.value["state"])  # one enum for the whole code base (UMRM 4.3)


__all__ = [
    "BodyState",
    "CandidateSpatialRelation",
    "InteroceptionState",
    "SelfModelUncertainty",
    "FrameUnresolvedError",
    "classify_visibility",
    "ObservationState",
    "RealityState",
    "FrameRegistry",
    "SelfModelState",
    "SpatialPose",
    "UniversalEntity",
    "VisibilityState",
]

