"""GeometryService, SpatialRelationEngine, SensorCoverageService (мастер-ТЗ A3; WM Spec §2.4, §2.9).

Уровни геометрии: L0 грубая (этажи, помещения), L1 геометрия активов, L2 сетка/облако точек (только
ссылка), L3 динамическая занятость. OpenUSD/3D Tiles — только обмен и визуализация; истина остаётся
в семантическом/доказательном слое.
    near(a,b,t,scale) ⟺ ‖p_a(t) − p_b(t)‖ < r_scale        — вычисляется, не хранится как факт
    V(agent,obj,t) = FOV ∩ line_of_sight ∩ range ∩ ¬occlusion
Ответ на «какие датчики могут подтвердить инцидент в коридоре B» — по каждому датчику: дальность, поле
зрения, прямая видимость, доля перекрытия, уверенность покрытия (с учётом здоровья датчика, если оно дано).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Optional, Sequence

import numpy as np

from .frames import SpatialReferenceService
from .journal import WorldJournal
from .policy import F_NEAR, F_VISIBLE, SpatialWorldError, SpatialWorldPolicy, aware, text

TIERS = ("L0", "L1", "L2", "L3")


@dataclass(frozen=True)
class GeometryRef:
    entity_id: str
    tier: str
    geometry_type: str                   # bounding_box | polygon | mesh_ref | pointcloud_ref | occupancy
    frame_id: str
    bbox: Optional[tuple]                # (x, y, z, dx, dy, dz) в кадре frame_id; x,y,z — минимальный угол
    footprint: Optional[tuple]           # ((x,y), …) в кадре
    ref: Optional[str]                   # usd://…, ifc://… — только ссылка
    occluder: bool                       # перекрывает ли обзор (стены, колонны, оборудование)
    valid_from: datetime
    knowledge_time: datetime


@dataclass(frozen=True)
class SensorView:
    sensor_id: str
    frame_id: str                        # кадр датчика: начало — точка обзора, +X — ось взгляда
    fov_h_deg: float
    fov_v_deg: Optional[float]
    range_m: float


def _seg_aabb(p0: np.ndarray, p1: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
    """Пересекает ли отрезок p0→p1 закрытый параллелепипед (метод слоёв)."""
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for i in range(3):
        if abs(d[i]) < 1e-15:
            if p0[i] < lo[i] or p0[i] > hi[i]:
                return False
        else:
            a, b = (lo[i] - p0[i]) / d[i], (hi[i] - p0[i]) / d[i]
            if a > b:
                a, b = b, a
            t0, t1 = max(t0, a), min(t1, b)
            if t0 > t1:
                return False
    return True


class GeometryService:
    def __init__(self, journal: WorldJournal, frames: SpatialReferenceService, policy: SpatialWorldPolicy) -> None:
        self.journal, self.frames, self.policy = journal, frames, policy
        self.geoms: list[GeometryRef] = []
        for r in journal.rows(("geometry",)):
            self._apply(r)

    def _apply(self, r) -> None:
        p = r.payload
        self.geoms.append(GeometryRef(p["entity_id"], p["tier"], p["geometry_type"], p["frame_id"],
                                      None if p["bbox"] is None else tuple(p["bbox"]),
                                      None if p["footprint"] is None else tuple(tuple(x) for x in p["footprint"]),
                                      p.get("ref"), p["occluder"], datetime.fromisoformat(p["valid_from"]),
                                      r.knowledge_time))

    def register(self, *, entity_id: str, tier: str, geometry_type: str, frame_id: str, valid_from: datetime,
                 knowledge_time: datetime, bbox: Optional[Sequence[float]] = None,
                 footprint: Optional[Sequence[Sequence[float]]] = None, ref: Optional[str] = None,
                 occluder: bool = False) -> GeometryRef:
        if tier not in TIERS:
            raise SpatialWorldError(f"tier must be one of {TIERS}")
        if tier in ("L0", "L1") and bbox is None and footprint is None:
            raise SpatialWorldError("L0/L1 geometry needs a bounding box or footprint")
        if tier == "L2" and not ref:
            raise SpatialWorldError("L2 geometry is referenced (mesh/point cloud), not inlined")
        if bbox is not None:
            b = [float(x) for x in bbox]
            if len(b) != 6 or any(not math.isfinite(x) for x in b) or min(b[3:]) <= 0:
                raise SpatialWorldError("bbox is (x, y, z, dx, dy, dz) with positive extents")
        self._apply(self.journal.append("geometry", {
            "entity_id": text("entity_id", entity_id), "tier": tier, "geometry_type": text("geometry_type", geometry_type),
            "frame_id": text("frame_id", frame_id), "bbox": None if bbox is None else [float(x) for x in bbox],
            "footprint": None if footprint is None else [[float(v) for v in pt] for pt in footprint], "ref": ref,
            "occluder": bool(occluder), "valid_from": aware("valid_from", valid_from).isoformat()},
            knowledge_time=knowledge_time))
        return self.geoms[-1]

    def _current(self, t: datetime, k: datetime) -> dict[str, GeometryRef]:
        out: dict[str, GeometryRef] = {}
        for g in self.geoms:
            if g.valid_from <= t and g.knowledge_time <= k:
                prev = out.get(g.entity_id)
                if prev is None or (g.valid_from, g.knowledge_time) >= (prev.valid_from, prev.knowledge_time):
                    out[g.entity_id] = g
        return out

    def world_aabb(self, g: GeometryRef, t: datetime, k: datetime) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if g.bbox is None:
            return None
        reg = self.frames.registry_at(t, known_at=k)
        T = reg.transform_between(g.frame_id, reg.root).value
        x, y, z, dx, dy, dz = g.bbox
        corners = np.array([[x + i * dx, y + j * dy, z + q * dz, 1.0] for i in (0, 1) for j in (0, 1) for q in (0, 1)])
        w = (T @ corners.T).T[:, :3]
        return w.min(axis=0), w.max(axis=0)

    def aabbs(self, t: datetime, *, known_at: Optional[datetime] = None) -> dict[str, tuple]:
        t = aware("t", t)
        k = aware("known_at", known_at) if known_at is not None else t
        out = {}
        for eid, g in self._current(t, k).items():
            box = self.world_aabb(g, t, k)
            if box is not None:
                out[eid] = (box[0], box[1], g)
        return out

    def center(self, entity_id: str, t: datetime, *, known_at: Optional[datetime] = None) -> Optional[np.ndarray]:
        b = self.aabbs(t, known_at=known_at).get(entity_id)
        return None if b is None else (b[0] + b[1]) / 2.0

    # ---------------------------------------------------------------- пространственные запросы
    def contains(self, container: str, point_world: Sequence[float], t: datetime) -> bool:
        b = self.aabbs(t).get(container)
        if b is None:
            raise SpatialWorldError(f"{container} has no L0/L1 box geometry at {t}")
        p = np.asarray(point_world, float)
        return bool(np.all(p >= b[0] - 1e-9) and np.all(p <= b[1] + 1e-9))

    def within_distance(self, entity_id: str, radius_m: float, t: datetime) -> list[str]:
        boxes = self.aabbs(t)
        if entity_id not in boxes:
            raise SpatialWorldError(f"{entity_id} has no geometry at {t}")
        c = (boxes[entity_id][0] + boxes[entity_id][1]) / 2
        out = []
        for eid, (lo, hi, _) in boxes.items():
            if eid == entity_id:
                continue
            nearest = np.clip(c, lo, hi)
            if float(np.linalg.norm(nearest - c)) < radius_m:
                out.append(eid)
        return sorted(out)

    def near(self, a_pos: Sequence[float], b_pos: Sequence[float], scale: str) -> dict:
        r = self.policy.near_radii_m.get(scale)
        if r is None:
            raise SpatialWorldError(f"no near radius for scale {scale!r}")
        d = float(np.linalg.norm(np.asarray(a_pos, float) - np.asarray(b_pos, float)))
        return {"near": d < float(r), "distance_m": d, "radius_m": float(r), "durable_fact": False, "formula": F_NEAR}

    def line_of_sight(self, p0: Sequence[float], p1: Sequence[float], t: datetime, *,
                      exclude: Sequence[str] = ()) -> list[str]:
        """Какие перекрывающие объекты пересекает отрезок (пусто — прямая видимость)."""
        a, b = np.asarray(p0, float), np.asarray(p1, float)
        blockers = []
        for eid, (lo, hi, g) in self.aabbs(t).items():
            if g.occluder and eid not in exclude and _seg_aabb(a, b, lo, hi):
                blockers.append(eid)
        return sorted(blockers)

    def _sensor_pose(self, view: SensorView, t: datetime) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reg = self.frames.registry_at(t)
        T = reg.transform_between(view.frame_id, reg.root).value
        return T[:3, 3], T[:3, :3] @ np.array([1.0, 0, 0]), T[:3, :3] @ np.array([0, 0, 1.0])

    def visibility(self, view: SensorView, target: str, t: datetime, *, sensor_entity: Optional[str] = None,
                   health: Optional[float] = None) -> dict:
        """V = FOV ∩ LoS ∩ range ∩ ¬occlusion по центру и 8 углам цели."""
        t = aware("t", t)
        boxes = self.aabbs(t)
        if target not in boxes:
            raise SpatialWorldError(f"target {target} has no geometry at {t}")
        lo, hi, _ = boxes[target]
        pts = [(lo + hi) / 2] + [np.array([lo[0] if i == 0 else hi[0], lo[1] if j == 0 else hi[1],
                                           lo[2] if k == 0 else hi[2]]) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        # точки чуть внутрь бокса, чтобы сама цель не считалась преградой собственному углу
        c = (lo + hi) / 2
        pts = [c + 0.98 * (p - c) for p in pts]
        origin, fwd, up = self._sensor_pose(view, t)
        in_range = in_fov = clear = 0
        blockers: set = set()
        for p in pts:
            v = p - origin
            d = float(np.linalg.norm(v))
            if d > view.range_m or d == 0:
                continue
            in_range += 1
            u = v / d
            # горизонтальный угол — в плоскости, перпендикулярной «вверх» датчика
            h = u - np.dot(u, up) * up
            fh = fwd - np.dot(fwd, up) * up
            ang_h = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(h, fh) / (np.linalg.norm(h) * np.linalg.norm(fh)
                                                                                         + 1e-15))))))
            ang_v = math.degrees(math.asin(max(-1.0, min(1.0, float(np.dot(u, up))))))
            if ang_h > view.fov_h_deg / 2 or (view.fov_v_deg is not None and abs(ang_v) > view.fov_v_deg / 2):
                continue
            in_fov += 1
            b = self.line_of_sight(origin, p, t, exclude=[target] + ([sensor_entity] if sensor_entity else []))
            if b:
                blockers.update(b)
            else:
                clear += 1
        n = len(pts)
        occl = 0.0 if in_fov == 0 else 1 - clear / in_fov
        los = "clear" if in_fov and clear == in_fov else ("partial" if clear else "blocked")
        conf = clear / n
        if health is not None:
            conf *= float(health)
        return {"sensor_id": view.sensor_id, "target": target, "visible": clear > 0, "in_range_fraction": in_range / n,
                "in_fov_fraction": in_fov / n, "line_of_sight": los, "occlusion_ratio": occl,
                "occluded_by": sorted(blockers), "distance_m": float(np.linalg.norm((lo + hi) / 2 - origin)),
                "coverage_confidence": conf, "health_applied": health is not None, "formula": F_VISIBLE}


class SensorCoverageService:
    def __init__(self, geometry: GeometryService) -> None:
        self.geometry = geometry
        self.views: dict[str, tuple[SensorView, Optional[str]]] = {}

    def register_sensor(self, view: SensorView, *, sensor_entity: Optional[str] = None) -> None:
        if view.range_m <= 0 or not 0 < view.fov_h_deg <= 360:
            raise SpatialWorldError("sensor needs a positive range and 0 < fov ≤ 360")
        self.views[view.sensor_id] = (view, sensor_entity)

    def who_can_confirm(self, target: str, t: datetime, *, health: Optional[Mapping[str, float]] = None) -> list[dict]:
        """«Какие датчики могут подтвердить инцидент в зоне X»: покрытие, LoS, перекрытие, уверенность."""
        out = []
        for sid, (view, ent) in sorted(self.views.items()):
            try:
                v = self.geometry.visibility(view, target, t, sensor_entity=ent,
                                             health=None if health is None else health.get(sid))
            except Exception as exc:                 # нет цепочки кадров и т. п. — датчик честно «не может»
                v = {"sensor_id": sid, "target": target, "visible": False, "error": str(exc),
                     "coverage_confidence": 0.0}
            out.append(v)
        return sorted(out, key=lambda r: -r["coverage_confidence"])


__all__ = ["GeometryRef", "GeometryService", "SensorCoverageService", "SensorView", "TIERS"]
