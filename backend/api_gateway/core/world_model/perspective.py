"""UMRM 3.3 (page 3): "Active body defines perspective":

    Perspective_t = f(body_t, pose_t, sensor_set_t, calibration_t)

The TZ does not fix the form of f.  Here f is the purely geometric composition that the other UMRM
formulas need (no model constants are added):

* ``pose_t`` (UMRM 3.1) gives ^F T_body in the pose frame F;
* ``calibration_t`` gives, per sensor, the extrinsics ^body T_sensor and (for cameras) intrinsics K;
* ^F T_sensor = ^F T_body . ^body T_sensor and, when F is resolved in the FrameRegistry,
  ^W T_sensor = ^W T_F . ^F T_sensor (UMRM 3.2 chain);
* the camera model of UMRM 4.2, u~ = K [R | t] X~_W, uses [R | t] = (^W T_sensor)^-1.

A sensor of the set without calibration makes the perspective invalid (fail-closed); a pose without a
chain to the world root keeps the perspective sensor-/pose-local (``world_resolved`` is computed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .assertions import FrameUnresolvedError
from .reality_contracts import BodyState, SpatialPose
from .reality_formulas import DOCUMENT, FormulaReference, FormulaResult
from .spatial_formulas import FrameRegistry, _se3, pose_to_transform, project_point, rigid_inverse

PERSPECTIVE_REFERENCE = FormulaReference(
    "UMRM-3.3-PERSPECTIVE",
    DOCUMENT,
    3,
    "Perspective_t = f(body_t, pose_t, sensor_set_t, calibration_t); "
    "f: ^W T_sensor = ^W T_F . ^F T_body(pose_t) . ^body T_sensor(calibration_t)",
)


@dataclass(frozen=True)
class SensorCalibration:
    """theta_sensor of one sensor: extrinsics ^body T_sensor (+ intrinsics K for a camera)."""

    sensor_id: str
    calibration_ref: str
    body_from_sensor: Any
    intrinsics: Any | None = None

    def __post_init__(self) -> None:
        for name in ("sensor_id", "calibration_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is required")
        extrinsics = _se3(self.body_from_sensor).copy()
        extrinsics.setflags(write=False)
        object.__setattr__(self, "body_from_sensor", extrinsics)
        if self.intrinsics is not None:
            k = np.array(self.intrinsics, dtype=float, copy=True)
            if k.shape != (3, 3) or not np.all(np.isfinite(k)):
                raise ValueError("intrinsics must be a finite 3x3 matrix")
            k.setflags(write=False)
            object.__setattr__(self, "intrinsics", k)


@dataclass(frozen=True)
class Perspective:
    body_id: str
    pose: SpatialPose
    sensor_set: tuple[str, ...]
    calibration: Mapping[str, SensorCalibration]
    frame_id: str = field(init=False)
    world_resolved: bool = field(init=False)
    formula: FormulaReference = field(init=False, default=PERSPECTIVE_REFERENCE)

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", self.pose.frame_id)
        object.__setattr__(self, "world_resolved", self.pose.frame_resolved)

    def _calibration(self, sensor_id: str) -> SensorCalibration:
        if sensor_id not in self.sensor_set:
            raise ValueError(f"sensor '{sensor_id}' is not in the active sensor set {self.sensor_set}")
        return self.calibration[sensor_id]

    def frame_from_sensor(self, sensor_id: str) -> np.ndarray:
        """^F T_sensor in the pose frame F."""
        body = pose_to_transform(self.pose.position, self.pose.orientation_quaternion)
        return body @ self._calibration(sensor_id).body_from_sensor

    def world_from_sensor(self, sensor_id: str) -> FormulaResult:
        """^W T_sensor; frame_unresolved if the pose frame has no chain to the world root."""
        if not self.world_resolved:
            raise FrameUnresolvedError(
                f"frame_unresolved: perspective of body '{self.body_id}' is in frame '{self.frame_id}' without a "
                "transform chain to the world frame"
            )
        resolution = self.pose.frame_resolution.refreshed()
        return FormulaResult(resolution.transform @ self.frame_from_sensor(sensor_id), PERSPECTIVE_REFERENCE,
                             {"chain": resolution.chain + (self.body_id, sensor_id)})

    def sensor_frame_registry(self) -> FrameRegistry:
        """A snapshot of the pose's FrameRegistry extended with this perspective's dynamic frames:
        ``body_id`` under the pose frame (^F T_body from pose_t) and every sensor of the set under the
        body (^body T_sensor from calibration_t).  Chains such as (W, site, building, floor, body,
        sensor) are then *computed* by the registry.  If the pose frame is not grounded, the chain to
        the world root does not exist and every world-frame use stays frame_unresolved."""
        base = self.pose.frame_registry
        registry = FrameRegistry() if base is None else FrameRegistry(root=base.root, _frames=dict(base._frames))
        for frame in (self.body_id, *self.sensor_set):
            if frame in registry._frames or frame == registry.root:
                raise ValueError(f"frame id '{frame}' already exists in the registry; body/sensor ids must be distinct")
        registry.register(self.body_id, self.frame_id,
                          pose_to_transform(self.pose.position, self.pose.orientation_quaternion))
        for sensor in self.sensor_set:
            registry.register(sensor, self.body_id, self.calibration[sensor].body_from_sensor)
        return registry

    def camera_extrinsics(self, sensor_id: str) -> tuple[np.ndarray, np.ndarray]:
        """[R | t] of UMRM 4.2 (world -> camera), from the computed ^W T_sensor."""
        inverse = rigid_inverse(self.world_from_sensor(sensor_id).value)
        return inverse[:3, :3], inverse[:3, 3]

    def project(self, sensor_id: str, world_point: Sequence[float]) -> FormulaResult:
        """u~ = K [R | t] X~_W for a camera of the active sensor set (UMRM 4.2, page 3)."""
        calibration = self._calibration(sensor_id)
        if calibration.intrinsics is None:
            raise ValueError(f"sensor '{sensor_id}' has no intrinsics: it is not a camera")
        rotation, translation = self.camera_extrinsics(sensor_id)
        return project_point(intrinsics=calibration.intrinsics, rotation=rotation, translation=translation,
                             world_point=world_point)


def perspective(
    *,
    body: BodyState,
    pose: SpatialPose,
    sensor_set: Sequence[str],
    calibration: Mapping[str, SensorCalibration],
) -> Perspective:
    """Perspective_t = f(body_t, pose_t, sensor_set_t, calibration_t) (UMRM 3.3, page 3)."""
    if not isinstance(body, BodyState):
        raise TypeError("body must be a BodyState")
    if not isinstance(pose, SpatialPose):
        raise TypeError("pose must be a SpatialPose")
    if isinstance(sensor_set, (str, bytes)):
        raise TypeError("sensor_set must be a sequence of sensor ids")
    sensors = tuple(sensor_set)
    if not sensors or len(set(sensors)) != len(sensors):
        raise ValueError("sensor_set must be a non-empty set of sensor ids")
    foreign = [s for s in sensors if s not in body.sensors]
    if foreign:
        raise ValueError(f"sensors {foreign} are not mounted on body '{body.body_id}'")
    if not isinstance(calibration, Mapping):
        raise TypeError("calibration must map sensor id -> SensorCalibration")
    missing = [s for s in sensors if s not in calibration]
    if missing:
        raise ValueError(f"no calibration for sensors {missing}: a perspective needs calibration_t")
    for sensor in sensors:
        item = calibration[sensor]
        if not isinstance(item, SensorCalibration) or item.sensor_id != sensor:
            raise ValueError(f"calibration[{sensor!r}] must be the SensorCalibration of that sensor")
    return Perspective(body.body_id, pose, sensors, MappingProxyType({s: calibration[s] for s in sensors}))


__all__ = ["PERSPECTIVE_REFERENCE", "Perspective", "SensorCalibration", "perspective"]
