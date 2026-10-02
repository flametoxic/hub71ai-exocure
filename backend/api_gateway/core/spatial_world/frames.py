"""SpatialReferenceService (мастер-ТЗ A3, Gap Closure v1 §4, v2 §4): дерево систем координат с историей.

    earth:wgs84 → city → district → site → building → floor → room/zone → asset → sensor
                                         ↘ road/lane → vehicle sensor frame; ↘ robot/phone/wearable body frame
    ^W p = ^W T_S · ^S p,   ^W T_sensor = ^W T_site · ^site T_building · ^building T_floor · ^floor T_sensor
    Σ_W = R Σ_S R^T;  для позы (положение+ориентация) — якобиан преобразования (UMRM spatial_formulas)

Каждое ребро дерева (кадр → родитель, ^parent T_frame) имеет valid_from/valid_to и время знания: калибровку
камеры поменяли — старое ребро закрыто, новое открыто, прошлые наблюдения пересчитываются по старому.
Нет цепочки → frame_unresolved (наблюдение отклоняется, а не угадывается). Математика — та же, что в
world_model.spatial_formulas (FrameRegistry): здесь она собирается на момент времени.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..world_model.spatial_formulas import FrameRegistry, FrameUnresolvedError, transform_covariance
from .journal import WorldJournal
from .policy import F_COV, F_FRAME, SpatialWorldError, aware, text

ROOT = FrameRegistry.root


@dataclass(frozen=True)
class FrameEdge:
    frame_id: str
    parent_id: str
    matrix: tuple            # 16 чисел, построчно, ^parent T_frame
    valid_from: datetime
    valid_to: Optional[datetime]
    knowledge_time: datetime
    level: str               # city/district/site/building/floor/zone/asset/sensor/body/…
    provenance: str
    edge_seq: int


@dataclass(frozen=True)
class SpatialPose:
    entity_id: str
    frame_id: str
    position: tuple[float, float, float]
    covariance: tuple                      # 3×3 построчно (обязательна — WM Spec §2.4)
    valid_time: datetime
    source_ref: str
    confidence: float
    orientation: Optional[tuple[float, float, float, float]] = None

    def __post_init__(self) -> None:
        text("entity_id", self.entity_id)
        text("frame_id", self.frame_id)
        p = np.asarray(self.position, float)
        c = np.asarray(self.covariance, float).reshape(3, 3)
        if p.shape != (3,) or not np.all(np.isfinite(p)) or not np.all(np.isfinite(c)):
            raise SpatialWorldError("position must be 3 finite numbers and covariance a finite 3×3")
        if not np.allclose(c, c.T) or np.min(np.linalg.eigvalsh(c)) < -1e-12:
            raise SpatialWorldError("covariance must be symmetric positive semi-definite")
        object.__setattr__(self, "position", tuple(float(x) for x in p))
        object.__setattr__(self, "covariance", tuple(float(x) for x in c.ravel()))
        object.__setattr__(self, "valid_time", aware("valid_time", self.valid_time))

    @property
    def cov(self) -> np.ndarray:
        return np.asarray(self.covariance, float).reshape(3, 3)

    def to_dict(self) -> dict:
        return {"entity_id": self.entity_id, "frame_id": self.frame_id, "position": list(self.position),
                "covariance": list(self.covariance), "valid_time": self.valid_time.isoformat(),
                "source_ref": self.source_ref, "confidence": self.confidence,
                "orientation": None if self.orientation is None else list(self.orientation)}


class SpatialReferenceService:
    def __init__(self, journal: WorldJournal) -> None:
        self.journal = journal
        self._edges: list[FrameEdge] = []
        self._closed: dict[int, tuple[datetime, datetime]] = {}      # edge_seq → (valid_to, knowledge_time)
        for r in journal.rows(("frame_open", "frame_close")):
            self._apply(r)

    def _apply(self, r) -> None:
        p = r.payload
        if r.kind == "frame_open":
            self._edges.append(FrameEdge(p["frame_id"], p["parent_id"], tuple(p["matrix"]),
                                         datetime.fromisoformat(p["valid_from"]), None, r.knowledge_time,
                                         p["level"], p["provenance"], r.seq))
        else:
            self._closed[p["edge_seq"]] = (datetime.fromisoformat(p["valid_to"]), r.knowledge_time)

    # ---------------------------------------------------------------- запись (журналом)
    def register_frame(self, frame_id: str, parent_id: str, parent_from_frame: Any, *, valid_from: datetime,
                       level: str, provenance: str, knowledge_time: datetime) -> FrameEdge:
        m = np.asarray(parent_from_frame, float)
        if m.shape != (4, 4) or not np.allclose(m[3], [0, 0, 0, 1]) or \
                not np.allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=1e-9) or np.linalg.det(m[:3, :3]) < 0:
            raise SpatialWorldError("parent_from_frame must be a rigid SE(3) 4×4 transform")
        text("frame_id", frame_id)
        text("parent_id", parent_id)
        if frame_id == ROOT or frame_id == parent_id:
            raise SpatialWorldError("invalid frame/parent pair")
        vf = aware("valid_from", valid_from)
        cur = self._active_edge(frame_id, vf, aware("knowledge_time", knowledge_time))
        if cur is not None:
            self.close_frame(frame_id, valid_to=vf, knowledge_time=knowledge_time, reason="re-registered")
        # цикл запрещён: родитель не должен быть потомком кадра на этот момент
        reg = self.registry_at(vf, known_at=knowledge_time)
        if parent_id != ROOT and parent_id in self._frames_of(reg):
            try:
                if frame_id in reg.path_to_root(parent_id):
                    raise SpatialWorldError(f"frame cycle: {parent_id} is below {frame_id}")
            except FrameUnresolvedError:
                pass
        r = self.journal.append("frame_open", {"frame_id": frame_id, "parent_id": parent_id,
                                               "matrix": [float(x) for x in m.ravel()], "valid_from": vf.isoformat(),
                                               "level": text("level", level), "provenance": text("provenance", provenance)},
                                knowledge_time=knowledge_time)
        self._apply(r)
        return self._edges[-1]

    def close_frame(self, frame_id: str, *, valid_to: datetime, knowledge_time: datetime, reason: str) -> None:
        vt, kt = aware("valid_to", valid_to), aware("knowledge_time", knowledge_time)
        e = self._active_edge(frame_id, vt, kt)
        if e is None:
            raise SpatialWorldError(f"no open edge for frame {frame_id} at {vt}")
        r = self.journal.append("frame_close", {"frame_id": frame_id, "edge_seq": e.edge_seq,
                                                "valid_to": vt.isoformat(), "reason": text("reason", reason)},
                                knowledge_time=kt)
        self._apply(r)

    # ---------------------------------------------------------------- чтение на момент времени
    def _valid(self, e: FrameEdge, t: datetime, k: datetime) -> bool:
        if e.knowledge_time > k or e.valid_from > t:
            return False
        c = self._closed.get(e.edge_seq)
        return c is None or c[1] > k or c[0] > t

    def _active_edge(self, frame_id: str, t: datetime, k: datetime) -> Optional[FrameEdge]:
        es = [e for e in self._edges if e.frame_id == frame_id and self._valid(e, t, k)]
        return max(es, key=lambda e: (e.valid_from, e.edge_seq)) if es else None

    @staticmethod
    def _frames_of(reg: FrameRegistry) -> set:
        return set(reg._frames)

    def registry_at(self, t: datetime, *, known_at: Optional[datetime] = None) -> FrameRegistry:
        """FrameRegistry, каким он был в мире в момент t по знаниям на момент known_at (по умолчанию — t)."""
        t = aware("t", t)
        k = aware("known_at", known_at) if known_at is not None else t
        reg = FrameRegistry()
        for fid in sorted({e.frame_id for e in self._edges}):
            e = self._active_edge(fid, t, k)
            if e is not None:
                reg.register(e.frame_id, e.parent_id, np.asarray(e.matrix, float).reshape(4, 4))
        return reg

    def chain(self, source: str, target: str = ROOT, *, at: datetime, known_at: Optional[datetime] = None) -> tuple:
        return self.registry_at(at, known_at=known_at).chain_between(source, target)

    def transform(self, pose: SpatialPose, to_frame: str = ROOT, *, known_at: Optional[datetime] = None) -> dict:
        """Полная цепочка ^target T_source и перенос ковариации Σ' = R Σ R^T; нет цепочки → frame_unresolved."""
        reg = self.registry_at(pose.valid_time, known_at=known_at)
        res = reg.transform_between(pose.frame_id, to_frame)
        t = res.value
        p = (t @ np.append(np.asarray(pose.position), 1.0))[:3]
        cov = transform_covariance(t[:3, :3], pose.cov).value
        return {"pose": SpatialPose(pose.entity_id, to_frame, tuple(p), tuple(np.asarray(cov).ravel()), pose.valid_time,
                                    pose.source_ref, pose.confidence),
                "chain": res.intermediates["chain"], "formula": (F_FRAME, F_COV), "registry_version": reg.version}

    def history(self, frame_id: str) -> list[dict]:
        out = []
        for e in self._edges:
            if e.frame_id == frame_id:
                c = self._closed.get(e.edge_seq)
                out.append({"parent": e.parent_id, "valid_from": e.valid_from.isoformat(),
                            "valid_to": None if c is None else c[0].isoformat(), "provenance": e.provenance,
                            "knowledge_time": e.knowledge_time.isoformat()})
        return out


def translation(x: float, y: float, z: float, yaw_rad: float = 0.0) -> np.ndarray:
    """Удобство для тестов и импорта: поворот вокруг Z + перенос."""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    m = np.eye(4)
    m[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    m[:3, 3] = [x, y, z]
    return m


def as_positions(poses: Sequence[SpatialPose]) -> Mapping[str, np.ndarray]:
    return {p.entity_id: np.asarray(p.position) for p in poses}


__all__ = ["FrameEdge", "ROOT", "SpatialPose", "SpatialReferenceService", "as_positions", "translation"]
