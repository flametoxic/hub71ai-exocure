from __future__ import annotations

import math
from collections.abc import Mapping as _MappingABC
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional
from uuid import uuid4

import numpy as np

from ..common.epistemic import EpistemicStatus
from .spatial_formulas import FrameRegistry, FrameResolution
from .spatial_formulas import FrameUnresolvedError as _SpatialFrameUnresolvedError


class FrameUnresolvedError(_SpatialFrameUnresolvedError, ValueError):
    """frame_unresolved (UMRM 3.2, pages 2-3): without a transform chain to the world frame no
    world-coordinate assertion is created.  It is an honest knowledge state, not error recovery.

    Subclasses both ``spatial_formulas.FrameUnresolvedError`` and ``ValueError``.
    """


# A spatial assertion is one whose value is a SpatialPose, whose metadata names a ``coordinate_frame``
# (or an alias frame key), whose attribute ends in ``_world``, or whose attribute is a position /
# location / pose carrying a coordinate vector.  It is a *world-coordinate* assertion whenever its frame
# differs from the frame of the sensor that produced it (``metadata["sensor_frame"]``) and ALWAYS when
# its frame is a world frame ("world", the default FrameRegistry root "earth:wgs84", or the root of the
# registry in use) -- declaring ``sensor_frame`` equal to a world frame does not make it sensor-local.
# Without a declared sensor frame every spatial assertion is treated as world-coordinate (fail-closed).
# A world-coordinate assertion is created only if the transform chain is COMPUTED from a FrameRegistry
# (UMRM 3.2, pages 2-3); a declared list of frame names is checked edge-by-edge, never trusted.
WORLD_COORDINATE_SUFFIX = "_world"
WORLD_COORDINATE_FRAME = "world"  # alias of the registry root frame
GLOBAL_FRAMES = frozenset({WORLD_COORDINATE_FRAME, FrameRegistry.root})
SPATIAL_ATTRIBUTE_TOKENS = frozenset({"position", "location", "pose"})
_FRAME_ALIAS_KEYS = ("frame", "frame_id", "reference_frame")
_COMPUTED_ONLY_METADATA = ("frame_resolved", "frame_resolution")


def finite_real(name: str, value: Any) -> float:
    """Return ``value`` as a finite float; booleans, non-numbers, NaN and +/-inf are rejected."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def reject_non_finite(name: str, value: Any) -> None:
    """Walk scalars, arrays, mappings and sequences and reject NaN/inf anywhere numeric."""
    seen: set[int] = set()

    def walk(item: Any, path: str) -> None:
        if item is None or isinstance(item, (bool, np.bool_, str, bytes, datetime, Enum)):
            return
        if isinstance(item, (int, float, np.integer, np.floating)):
            if not math.isfinite(float(item)):
                raise ValueError(f"{path} must be finite")
            return
        if isinstance(item, (complex, np.complexfloating)):
            if not (math.isfinite(item.real) and math.isfinite(item.imag)):
                raise ValueError(f"{path} must be finite")
            return
        if id(item) in seen:
            return
        if isinstance(item, np.ndarray):
            if item.dtype.kind in "fc":
                if not np.all(np.isfinite(item)):
                    raise ValueError(f"{path} must contain only finite values")
                return
            if item.dtype.kind != "O":
                return
            seen.add(id(item))
            for index, element in enumerate(item.ravel()):
                walk(element, f"{path}[{index}]")
            return
        if isinstance(item, _MappingABC):
            seen.add(id(item))
            for key, element in item.items():
                walk(element, f"{path}[{key!r}]")
            return
        if isinstance(item, (list, tuple, set, frozenset)):
            seen.add(id(item))
            for index, element in enumerate(item):
                walk(element, f"{path}[{index}]")

    walk(value, name)


def _pose_like(value: Any) -> bool:
    return hasattr(value, "frame_id") and hasattr(value, "frame_resolution")


def _is_coordinate_vector(value: Any) -> bool:
    if isinstance(value, (str, bytes, _MappingABC)):
        return False
    try:
        array = np.asarray(value)
    except Exception:  # pragma: no cover - exotic containers are not coordinates
        return False
    return array.ndim == 1 and array.size in (2, 3) and array.dtype.kind in "iuf"


def _is_spatial_attribute(attribute_or_relation: str, value: Any) -> bool:
    tokens = "".join(ch if ch.isalnum() else " " for ch in attribute_or_relation.lower()).split()
    return any(token in SPATIAL_ATTRIBUTE_TOKENS for token in tokens) and _is_coordinate_vector(value)


def assertion_frame(attribute_or_relation: str, value: Any, metadata: Mapping[str, Any]) -> str | None:
    """Coordinate frame of a spatial assertion; ``None`` for a non-spatial one, ``""`` if unknown."""
    frame = metadata.get("coordinate_frame") if isinstance(metadata, _MappingABC) else None
    if frame is None and isinstance(metadata, _MappingABC):
        alias = [key for key in _FRAME_ALIAS_KEYS if isinstance(metadata.get(key), str)]
        if alias:
            raise FrameUnresolvedError(
                f"frame_unresolved: metadata {alias} is not the coordinate frame key; declare 'coordinate_frame'"
            )
    if _pose_like(value):
        if frame is not None and frame != value.frame_id:
            raise FrameUnresolvedError(
                f"frame_unresolved: metadata coordinate_frame '{frame}' contradicts the pose frame '{value.frame_id}'"
            )
        return value.frame_id
    if frame is not None:
        return frame.strip() if isinstance(frame, str) else ""
    if attribute_or_relation.strip().endswith(WORLD_COORDINATE_SUFFIX):
        return ""
    if _is_spatial_attribute(attribute_or_relation, value):
        return ""  # a position/location/pose vector without a frame: unknown frame -> fail-closed
    return None


def _world_frames(registry: Any) -> frozenset[str]:
    root = getattr(registry, "root", None)
    return GLOBAL_FRAMES | ({root} if isinstance(root, str) and root else frozenset())


_NO_METADATA = object()


def is_world_coordinate_assertion(attribute_or_relation: str, value: Any, metadata: Any = _NO_METADATA,
                                  frame_registry: FrameRegistry | None = None) -> bool:
    if metadata is _NO_METADATA:
        # совместимость со старым вызовом (attribute_or_relation, metadata): значения нет
        if not isinstance(value, _MappingABC):
            raise TypeError("is_world_coordinate_assertion(attribute, value, metadata) requires metadata")
        value, metadata = None, value
    frame = assertion_frame(attribute_or_relation, value, metadata)
    if frame is None:
        return False
    if attribute_or_relation.strip().endswith(WORLD_COORDINATE_SUFFIX):
        return True
    if not frame or frame in _world_frames(frame_registry):
        return True  # a world frame is never "sensor-local", whatever sensor_frame says
    sensor = metadata.get("sensor_frame") if isinstance(metadata, _MappingABC) else None
    return not (isinstance(sensor, str) and sensor.strip() and sensor.strip() == frame)


def require_world_frame_chain(
    attribute_or_relation: str,
    value: Any,
    metadata: Mapping[str, Any],
    frame_registry: FrameRegistry | None = None,
) -> FrameResolution | None:
    """Fail-closed frame check for world-coordinate assertions (UMRM 3.2, page 3).

    Returns ``None`` for a non-spatial or sensor-local assertion, otherwise the FrameResolution
    computed from the registry: the assertion frame must be grounded in the registry root, the sensor
    frame (if declared) must be connected to it, and a declared ``metadata["transform_chain"]`` must be
    exactly the registry chain.  Anything else raises ``FrameUnresolvedError`` (frame_unresolved).
    """
    if isinstance(metadata, _MappingABC):
        supplied = [key for key in _COMPUTED_ONLY_METADATA if key in metadata]
        if supplied:
            raise ValueError(f"{supplied} are computed from the FrameRegistry and cannot be supplied in metadata")
    registry = frame_registry
    if registry is None and _pose_like(value) and value.frame_resolution is not None:
        registry = value.frame_resolution.registry
    if not is_world_coordinate_assertion(attribute_or_relation, value, metadata, registry):
        return None
    frame = assertion_frame(attribute_or_relation, value, metadata)
    if not frame:
        raise FrameUnresolvedError(
            f"frame_unresolved: world-coordinate assertion '{attribute_or_relation}' names no coordinate frame"
        )
    if registry is None:
        raise FrameUnresolvedError(
            f"frame_unresolved: world-coordinate assertion '{attribute_or_relation}' in frame '{frame}' needs a "
            "FrameRegistry to compute the transform chain (a list of frame names is not a transform)"
        )
    if not isinstance(registry, FrameRegistry):
        raise TypeError("frame_registry must be a FrameRegistry")
    if frame == WORLD_COORDINATE_FRAME:
        frame = registry.root
    registry.resolve(frame, registry.root)  # the frame itself must be grounded in the world root
    sensor = metadata.get("sensor_frame") if isinstance(metadata, _MappingABC) else None
    if sensor is not None and (not isinstance(sensor, str) or not sensor.strip()):
        raise FrameUnresolvedError("frame_unresolved: sensor_frame must be a non-empty frame id")
    if sensor == WORLD_COORDINATE_FRAME:
        sensor = registry.root
    resolution = registry.resolve(sensor, frame) if sensor else registry.resolve(frame, registry.root)
    declared = metadata.get("transform_chain") if isinstance(metadata, _MappingABC) else None
    if declared is not None:
        verified = registry.verify_chain(declared)
        if verified.chain != resolution.chain:
            raise FrameUnresolvedError(
                f"frame_unresolved: declared transform_chain {verified.chain} is not the chain {resolution.chain} "
                f"between sensor frame and assertion frame"
            )
    return resolution


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def require_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


class QualityStatus(str, Enum):
    VALID = "valid"
    DEGRADED = "degraded"
    STALE = "stale"
    INVALID = "invalid"


@dataclass(frozen=True)
class WorldModelAssertion:
    entity_id: str
    attribute_or_relation: str
    value: Any
    event_time: datetime
    knowledge_time: datetime
    valid_time_start: datetime
    source_id: str
    assertion_id: str = field(default_factory=lambda: f"assertion:{uuid4().hex}")
    unit: Optional[str] = None
    valid_time_end: Optional[datetime] = None
    provenance_refs: tuple[str, ...] = ()
    epistemic_status: EpistemicStatus = EpistemicStatus.OBSERVED
    quality_status: QualityStatus = QualityStatus.VALID
    freshness_ms: int = 0
    confidence: float = 1.0
    uncertainty: float = 0.0
    scope: str = "default"
    world_model_version: str = "v4"
    schema_version: str = "1"
    trace_id: str = ""
    action_authority: str = "eligible"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    frame_registry: Any = field(default=None, compare=False, repr=False)
    frame_resolution: Any = field(init=False, default=None, compare=False)

    def __post_init__(self) -> None:
        if not self.entity_id.strip() or not self.attribute_or_relation.strip() or not self.source_id.strip():
            raise ValueError("entity_id, attribute_or_relation, and source_id are required")
        for name in ("event_time", "knowledge_time", "valid_time_start"):
            object.__setattr__(self, name, require_aware(getattr(self, name), name))
        if self.valid_time_end is not None:
            object.__setattr__(self, "valid_time_end", require_aware(self.valid_time_end, "valid_time_end"))
            if self.valid_time_end <= self.valid_time_start:
                raise ValueError("valid_time_end must be after valid_time_start")
        if not 0.0 <= finite_real("confidence", self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if finite_real("uncertainty", self.uncertainty) < 0.0:
            raise ValueError("uncertainty must be non-negative")
        if finite_real("freshness_ms", self.freshness_ms) < 0:
            raise ValueError("freshness_ms must be non-negative")
        reject_non_finite("value", self.value)
        object.__setattr__(
            self,
            "frame_resolution",
            require_world_frame_chain(self.attribute_or_relation, self.value, self.metadata, self.frame_registry),
        )

    def valid_at(self, at: datetime) -> bool:
        instant = require_aware(at, "valid_at")
        return self.valid_time_start <= instant and (self.valid_time_end is None or instant < self.valid_time_end)

    def known_at(self, at: datetime) -> bool:
        return self.knowledge_time <= require_aware(at, "known_at")

    def with_valid_time_end(self, end: datetime) -> "WorldModelAssertion":
        return replace(self, valid_time_end=require_aware(end, "valid_time_end"))


__all__ = [
    "EpistemicStatus",
    "FrameUnresolvedError",
    "GLOBAL_FRAMES",
    "QualityStatus",
    "SPATIAL_ATTRIBUTE_TOKENS",
    "WORLD_COORDINATE_FRAME",
    "WORLD_COORDINATE_SUFFIX",
    "WorldModelAssertion",
    "assertion_frame",
    "finite_real",
    "is_world_coordinate_assertion",
    "reject_non_finite",
    "require_aware",
    "require_world_frame_chain",
    "utc_now",
]

