from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence
import numpy as np
from .reality_formulas import FormulaReference, FormulaResult, transform_covariance, validate_covariance, validate_rotation

UMRM = "CURE-Unified-Mathematical-Reality-Model"
F_TRANSFORM = FormulaReference("UMRM-3.2-FRAME-TRANSFORM", UMRM, 2, "^W p = ^W T_S · ^S p")
F_CHAIN = FormulaReference("UMRM-3.2-TRANSFORM-CHAIN", UMRM, 2, "^W T_sensor = ^W T_site · ^site T_building · ^building T_floor · ^floor T_sensor")
F_POSE_TRANSFORM = FormulaReference(
    "UMRM-3.2-POSE-COVARIANCE",
    UMRM,
    2,
    "p = (x,y,z,q_x,q_y,q_z,q_w,v_x,v_y,v_z,w_x,w_y,w_z) (p.2, 3.1); ^W p = ^W T_S · ^S p; "
    "Sigma_W = J Sigma_S J^T with J = blockdiag(R, L(q_R), R, R) -- the 13-D form of Sigma_W = R Sigma_S R^T",
)
F_OBS = FormulaReference("UMRM-4.1-OBSERVATION-MODEL", UMRM, 3, "z_t = h(X_t, G_t, b_t, theta_sensor) + nu_t")
F_CAMERA = FormulaReference("UMRM-4.2-CAMERA-PROJECTION", UMRM, 3, "u~ = K [R|t] X~_W")
F_VISIBILITY = FormulaReference("UMRM-4.3-VISIBILITY", UMRM, 4, "V = FOV ∩ LoS ∩ Range ∩ ¬Occlusion")

# Numerical consistency tolerance for rigid transforms / unit quaternions (not a model constant).
_ATOL = 1e-6


class FrameUnresolvedError(RuntimeError, ValueError):
    """frame_unresolved: мировой assertion не создаётся (UMRM стр. 3).  Also a ValueError: a missing
    transform is a fail-closed input condition everywhere in the UMRM layer."""


def _se3(m) -> np.ndarray:
    t = np.asarray(m, float)
    if t.shape != (4, 4) or not np.all(np.isfinite(t)) or not np.allclose(t[3], (0, 0, 0, 1)):
        raise ValueError("transform must be a finite homogeneous 4x4 matrix")
    r = t[:3, :3]
    if not np.allclose(r @ r.T, np.eye(3), atol=_ATOL) or not np.isclose(np.linalg.det(r), 1.0, atol=_ATOL):
        raise ValueError("rotation block must be orthonormal with det=+1")
    return t


def _vec3(p, name="point") -> np.ndarray:
    v = np.asarray(p, float)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return v


def rigid_inverse(transform) -> np.ndarray:
    """Exact inverse of a rigid transform: [R|t]^-1 = [R^T | -R^T t]."""
    t = _se3(transform)
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return out


def transform_point(transform, point) -> FormulaResult:
    return FormulaResult((_se3(transform) @ np.append(_vec3(point), 1.0))[:3], F_TRANSFORM)


def compose_transform_chain(transforms) -> FormulaResult:
    if not transforms:
        raise ValueError("chain must contain at least one transform")
    out = np.eye(4)
    for t in transforms:
        out = out @ _se3(t)
    return FormulaResult(out, F_CHAIN, {"chain_length": len(transforms)})


# ---------------------------------------------------------------------------------------------
# Quaternions, order (q_x, q_y, q_z, q_w) exactly as in UMRM 3.1 (page 2).  Hamilton product.
# ---------------------------------------------------------------------------------------------

def _unit_quaternion(q, name="quaternion") -> np.ndarray:
    v = np.asarray(q, float)
    if v.shape != (4,) or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be a finite 4-vector (q_x, q_y, q_z, q_w)")
    if abs(float(np.linalg.norm(v)) - 1.0) > _ATOL:
        raise ValueError(f"{name} must be a unit quaternion")
    return v


def quaternion_left_matrix(p) -> np.ndarray:
    """L(p) with p ⊗ q = L(p) q, both in (x, y, z, w) order.  L(p) is orthogonal for a unit p."""
    x, y, z, w = np.asarray(p, float)
    return np.array([
        [w, -z, y, x],
        [z, w, -x, y],
        [-y, x, w, z],
        [-x, -y, -z, w],
    ])


def quaternion_multiply(p, q) -> np.ndarray:
    return quaternion_left_matrix(p) @ np.asarray(q, float)


def quaternion_to_rotation(q) -> np.ndarray:
    x, y, z, w = _unit_quaternion(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotation_to_quaternion(rotation) -> np.ndarray:
    """Proper rotation -> unit quaternion (x, y, z, w), canonical sign w >= 0 (Shepperd's method)."""
    r = validate_rotation(rotation)
    if r.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    trace = float(np.trace(r))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = np.array([(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s])
    elif r[0, 0] >= r[1, 1] and r[0, 0] >= r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = np.array([0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s])
    elif r[1, 1] >= r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = np.array([(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = np.array([(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s])
    q = q / np.linalg.norm(q)
    return -q if q[3] < 0 else q


def pose_to_transform(position, quaternion) -> np.ndarray:
    """^F T_body from a pose (position, orientation) expressed in frame F."""
    out = np.eye(4)
    out[:3, :3] = quaternion_to_rotation(quaternion)
    out[:3, 3] = _vec3(position, "position")
    return out


def pose_transform_jacobian(transform) -> np.ndarray:
    """J = blockdiag(R, L(q_R), R, R): Jacobian of the 13-D pose map under a static rigid transform.

    Position: p' = R p + t -> R.  Orientation: q' = q_R ⊗ q -> L(q_R).  Linear/angular velocity
    expressed in the pose frame: v' = R v, w' = R w -> R (frames of the registry are static relative
    to each other, so no transport term appears).
    """
    t = _se3(transform)
    r = t[:3, :3]
    j = np.zeros((13, 13))
    j[0:3, 0:3] = r
    j[3:7, 3:7] = quaternion_left_matrix(rotation_to_quaternion(r))
    j[7:10, 7:10] = r
    j[10:13, 10:13] = r
    return j


def transform_pose_state(transform, pose_vector, covariance) -> FormulaResult:
    """Transform the full 13-D pose p_e (UMRM 3.1) and its 13x13 covariance into another frame.

    UMRM 3.2 gives Sigma_W = R Sigma_S R^T for a point; for the full pose of 3.1 the same
    first-order law is Sigma_W = J Sigma_S J^T with the block Jacobian of :func:`pose_transform_jacobian`.
    The map is linear in p (quaternion product is linear in q), so the propagation is exact.
    """
    t = _se3(transform)
    x = np.asarray(pose_vector, float)
    if x.shape != (13,) or not np.all(np.isfinite(x)):
        raise ValueError("pose_vector must be a finite 13-vector (x,y,z,q_x,q_y,q_z,q_w,v_x,v_y,v_z,w_x,w_y,w_z)")
    _unit_quaternion(x[3:7], "pose orientation")
    sigma = validate_covariance(covariance, name="pose covariance", shape=(13, 13))
    j = pose_transform_jacobian(t)
    r = t[:3, :3]
    out = np.empty(13)
    out[0:3] = r @ x[0:3] + t[:3, 3]
    out[3:7] = j[3:7, 3:7] @ x[3:7]
    out[3:7] /= np.linalg.norm(out[3:7])  # remove rounding drift; the product of unit quaternions is unit
    if out[6] < 0:  # canonical sign; q and -q are the same orientation, flip the covariance rows with it
        out[3:7] = -out[3:7]
        j[3:7, 3:7] = -j[3:7, 3:7]
    out[7:10] = r @ x[7:10]
    out[10:13] = r @ x[10:13]
    cov = j @ sigma @ j.T
    cov = 0.5 * (cov + cov.T)
    return FormulaResult({"vector": out, "covariance": cov, "jacobian": j}, F_POSE_TRANSFORM,
                         {"rotation": r, "translation": t[:3, 3]})


# ---------------------------------------------------------------------------------------------
# Frame registry and computed frame resolution (UMRM 3.2, pages 2-3).
# ---------------------------------------------------------------------------------------------

@dataclass
class FrameRegistry:
    """earth:wgs84 → city → district → site → building → floor → room → asset → sensor (Master A3)."""
    root: str = "earth:wgs84"
    _frames: dict = field(default_factory=dict)  # frame -> (parent, ^parent T_frame)
    _version: int = 0

    def register(self, frame_id: str, parent_id: str, parent_from_frame) -> None:
        if not isinstance(frame_id, str) or not isinstance(parent_id, str):
            raise ValueError("frame ids must be strings")
        if not frame_id.strip() or not parent_id.strip() or frame_id in (parent_id, self.root):
            raise ValueError("invalid frame/parent pair")
        self._frames[frame_id] = (parent_id, _se3(parent_from_frame).copy())
        self._version += 1

    @property
    def version(self) -> int:
        return self._version

    def parent_of(self, frame_id: str) -> str | None:
        entry = self._frames.get(frame_id)
        return None if entry is None else entry[0]

    def path_to_root(self, frame_id: str) -> tuple[str, ...]:
        """(frame, parent, ..., root); raises frame_unresolved when the chain is broken or cyclic."""
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise FrameUnresolvedError("frame_unresolved: empty frame id")
        path, cur = [frame_id], frame_id
        while cur != self.root:
            if cur not in self._frames:
                raise FrameUnresolvedError(f"frame_unresolved: no transform from '{cur}' to '{self.root}'")
            parent = self._frames[cur][0]
            if parent in path:
                raise FrameUnresolvedError(f"frame_unresolved: frame cycle at '{parent}'")
            path.append(parent)
            cur = parent
        return tuple(path)

    def edge_transform(self, a: str, b: str) -> np.ndarray:
        """^a T_b for two frames adjacent in the registry tree (either direction)."""
        if self.parent_of(b) == a:
            return self._frames[b][1].copy()
        if self.parent_of(a) == b:
            return rigid_inverse(self._frames[a][1])
        raise FrameUnresolvedError(f"frame_unresolved: '{a}' and '{b}' are not adjacent in the frame registry")

    def chain_between(self, source: str, target: str) -> tuple[str, ...]:
        """Frames from ``target`` to ``source`` through their lowest common ancestor.

        For target = world root and source = sensor this is (W, site, building, floor, sensor), i.e. the
        order of the product ^W T_site · ^site T_building · ^building T_floor · ^floor T_sensor.
        """
        up_s, up_t = self.path_to_root(source), self.path_to_root(target)
        i_s = next(i for i, frame in enumerate(up_s) if frame in up_t)
        i_t = up_t.index(up_s[i_s])
        return tuple(up_t[: i_t + 1]) + tuple(reversed(up_s[:i_s]))

    def transform_along_chain(self, chain: Sequence[str]) -> np.ndarray:
        frames = tuple(chain)
        if len(frames) == 1:
            return np.eye(4)
        return compose_transform_chain([self.edge_transform(a, b) for a, b in zip(frames, frames[1:])]).value

    def transform_between(self, source: str, target: str) -> FormulaResult:
        chain = self.chain_between(source, target)
        return FormulaResult(self.transform_along_chain(chain), F_CHAIN, {"chain": chain})

    def resolve(self, source: str, target: str | None = None) -> "FrameResolution":
        return FrameResolution(self, source, self.root if target is None else target)

    def verify_chain(self, chain: Sequence[str]) -> "FrameResolution":
        """Verify a declared chain (target, ..., source) edge-by-edge against this registry."""
        if isinstance(chain, (str, bytes)) or not isinstance(chain, (list, tuple)) or len(chain) < 2:
            raise FrameUnresolvedError("frame_unresolved: transform_chain must list at least two frame ids")
        if any(not isinstance(f, str) or not f.strip() for f in chain):
            raise FrameUnresolvedError("frame_unresolved: transform_chain contains an empty frame id")
        resolution = self.resolve(chain[-1], chain[0])
        if resolution.chain != tuple(chain):
            raise FrameUnresolvedError(
                f"frame_unresolved: declared chain {tuple(chain)} does not match the registry chain {resolution.chain}"
            )
        return resolution

    def transform_pose(self, source, target, point, covariance) -> FormulaResult:
        t = self.transform_between(source, target).value
        return FormulaResult({"position": transform_point(t, point).value,
                              "covariance": transform_covariance(t[:3, :3], covariance).value,
                              "frame_id": target}, F_TRANSFORM)

    def transform_pose_state(self, source, target, pose_vector, covariance) -> FormulaResult:
        """Full 13-D pose + 13x13 covariance from ``source`` into ``target`` via the registry chain."""
        resolution = self.resolve(source, target)
        out = transform_pose_state(resolution.transform, pose_vector, covariance)
        return FormulaResult({**out.value, "frame_id": target}, F_POSE_TRANSFORM,
                             {**out.intermediates, "chain": resolution.chain})


@dataclass(frozen=True)
class FrameResolution:
    """A *computed* transform chain ^target T_source.  Only the registry fields are inputs; the chain,
    the transform and the registry version are computed from the registry, never accepted."""

    registry: FrameRegistry = field(compare=False, repr=False)
    source_frame: str
    target_frame: str
    chain: tuple[str, ...] = field(init=False)
    transform: np.ndarray = field(init=False, compare=False)
    registry_version: int = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.registry, FrameRegistry):
            raise TypeError("registry must be a FrameRegistry")
        chain = self.registry.chain_between(self.source_frame, self.target_frame)
        transform = self.registry.transform_along_chain(chain)
        transform.setflags(write=False)
        object.__setattr__(self, "chain", chain)
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "registry_version", self.registry.version)

    def is_current(self) -> bool:
        return self.registry.version == self.registry_version

    def refreshed(self) -> "FrameResolution":
        """Re-resolve against the registry now; raises frame_unresolved if the chain no longer exists."""
        return FrameResolution(self.registry, self.source_frame, self.target_frame)

    def as_record(self) -> dict:
        return {"source_frame": self.source_frame, "target_frame": self.target_frame, "chain": self.chain,
                "registry_root": self.registry.root, "registry_version": self.registry_version}


def observation_model(h: Callable[[Any, Any, Any, Any], Sequence[float]], *, state, geometry, body,
                      calibration, noise_covariance, rng: np.random.Generator | None = None) -> FormulaResult:
    """Одна форма для BMS, камеры, LiDAR, V2X, микрофона, GNSS и отчёта оператора.
    expected = h(·), noise_covariance = R; при переданном rng ещё и z = h + ν, ν ~ N(0, R).

    Сама формула z = h(X, G, b, θ) + ν — единственная реализация ``reality_formulas.generative_observation``;
    здесь только векторная форма h и гауссова модель шума R."""
    from .reality_formulas import generative_observation

    z = np.asarray(h(state, geometry, body, calibration), float).reshape(-1)
    r = np.asarray(noise_covariance, float)
    if z.size == 0 or not np.all(np.isfinite(z)):
        raise ValueError("h must return a non-empty finite vector")
    if r.shape != (z.size, z.size) or not np.allclose(r, r.T) or np.min(np.linalg.eigvalsh(r)) < -1e-12:
        raise ValueError("noise covariance must be symmetric PSD and match z")
    sample = None
    if rng is not None:
        sample = generative_observation(latent_state=z, geometry_topology={}, body_perspective={},
                                        sensor_calibration={}, noise=rng.multivariate_normal(np.zeros(z.size), r),
                                        measurement_model=lambda x, *_: x).value
    return FormulaResult({"expected": z, "noise_covariance": r, "z": sample}, F_OBS)


def project_point(*, intrinsics, rotation, translation, world_point) -> FormulaResult:
    k = np.asarray(intrinsics, float)
    if k.shape != (3, 3) or not np.all(np.isfinite(k)):
        raise ValueError("K must be a finite 3x3 matrix")
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], (0, 0, 1)) or k[1, 0] != 0:
        raise ValueError("K must be upper-triangular, positive focal lengths, K[2,2]=1")
    r = validate_rotation(rotation)  # orthonormal AND det(R)=+1: a reflection is not a camera pose
    if r.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    cam = r @ _vec3(world_point, "world_point") + _vec3(translation, "translation")
    if cam[2] <= 0:  # за камерой: не наблюдаемо, а не ошибка
        return FormulaResult({"pixel": None, "depth": float(cam[2]), "in_front_of_camera": False}, F_CAMERA)
    uvw = k @ cam
    return FormulaResult({"pixel": (float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])),
                          "depth": float(cam[2]), "in_front_of_camera": True}, F_CAMERA)


class VisibilityState(str, Enum):
    """UMRM 4.3 (page 4) -- the single visibility enum of the code base; values are the TZ strings."""
    NOT_OBSERVED = "not_observed"
    OCCLUDED = "not_visible_due_to_occlusion"
    OUT_OF_RANGE = "out_of_range"
    SENSOR_UNAVAILABLE = "sensor_unavailable"
    LOW_CONFIDENCE = "observed_with_low_confidence"
    CALIBRATED_CONFIDENCE = "observed_with_calibrated_confidence"
    CALIBRATED = "observed_with_calibrated_confidence"  # alias of CALIBRATED_CONFIDENCE (same member)


def visibility(*, sensor_available: bool, in_field_of_view: bool, line_of_sight: bool, in_range: bool,
               occluded: bool, detection_confidence: float | None, confidence_threshold: float,
               sensor_calibrated: bool) -> FormulaResult:
    """Отсутствие детекции ≠ отсутствие объекта. Негативное свидетельство — только если объект
    геометрически виден, датчик исправен и откалиброван."""
    if not 0 <= confidence_threshold <= 1 or (detection_confidence is not None and not 0 <= detection_confidence <= 1):
        raise ValueError("confidences must be in [0, 1]")
    visible = in_field_of_view and line_of_sight and in_range and not occluded
    if not sensor_available:
        state = VisibilityState.SENSOR_UNAVAILABLE
    elif not in_range:
        state = VisibilityState.OUT_OF_RANGE
    elif occluded or not line_of_sight:
        state = VisibilityState.OCCLUDED
    elif not in_field_of_view or detection_confidence is None:
        state = VisibilityState.NOT_OBSERVED
    elif detection_confidence >= confidence_threshold and sensor_calibrated:
        state = VisibilityState.CALIBRATED_CONFIDENCE
    else:
        state = VisibilityState.LOW_CONFIDENCE
    return FormulaResult({"visible": bool(visible and sensor_available), "state": state,
                          "negative_evidence_eligible": bool(sensor_available and visible and sensor_calibrated
                                                             and detection_confidence is None)}, F_VISIBILITY)
