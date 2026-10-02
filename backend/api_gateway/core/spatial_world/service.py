"""SpatialWorldModel — фасад пространственной модели мира (WM Spec §2.16; Gap Closure v2 §4).

Одна точка входа: журнал мира → кадры, утверждения, сущности, топология, геометрия, покрытие датчиков,
слияние состояния, динамические треки, конвейер проекции. Читать можно всё; писать — только через
``pipeline``. Любой ответ несёт data_mode (durable / volatile / degraded), версию модели мира,
версию топологии и голову журнала — по ним ответ воспроизводится.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np

from ..operational_integrity.context import weakest_mode
from .dynamic_state import DynamicStateService
from .entities import EntityRegistry
from .frames import ROOT, SpatialReferenceService
from .fusion import StateFusionService
from .geometry import GeometryService, SensorCoverageService
from .journal import WorldJournal, digest
from .ledger import BitemporalLedger
from .pipeline import ProjectionPipeline
from .policy import SpatialWorldError, SpatialWorldPolicy, aware, text
from .topology import HIERARCHY_TYPES, TopologyService


@dataclass(frozen=True)
class WorldSnapshot:
    snapshot_id: str
    valid_at: str
    known_at: str
    world_model_version: str
    topology_version: int
    journal_head: str
    data_mode: str                      # хранилище: durable | volatile | degraded
    data_origin: Optional[str]          # WM-05 §7: самый слабый режим данных среди утверждений снимка
    data_modes_present: tuple
    state: dict


class SpatialWorldModel:
    def __init__(self, policy: SpatialWorldPolicy, *, journal_path: Optional[str] = None,
                 world_model_version: str, accel_noise: Optional[float] = None) -> None:
        self.policy = policy
        self.world_model_version = text("world_model_version", world_model_version)
        self.journal = WorldJournal(journal_path)
        self.frames = SpatialReferenceService(self.journal)
        self.entities = EntityRegistry(self.journal, policy)
        self.ledger = BitemporalLedger(self.journal, policy, frames=self.frames)
        self.topology = TopologyService(self.journal, entity_exists=lambda e: e in self.entities.entities)
        self.geometry = GeometryService(self.journal, self.frames, policy)
        self.coverage = SensorCoverageService(self.geometry)
        self.fusion = StateFusionService(policy)
        self.dynamic = None if accel_noise is None else DynamicStateService(policy, accel_noise=accel_noise)
        self.pipeline = ProjectionPipeline(ledger=self.ledger, entities=self.entities, frames=self.frames,
                                           world_model_version=self.world_model_version, topology=self.topology)

    @property
    def data_mode(self) -> str:
        if not self.journal.available:
            return "degraded"
        return "durable" if self.journal.path else "volatile"

    def _meta(self) -> dict:
        return {"data_mode": self.data_mode, "world_model_version": self.world_model_version,
                "topology_version": self.topology.version, "journal_head": self.journal.head(),
                "action_authority_restricted": self.data_mode != "durable"}

    # ---------------------------------------------------------------- состояние
    def state_at(self, entity_id: str, *, valid_at: datetime, known_at: Optional[datetime] = None) -> dict:
        eid = self.entities.canonical(entity_id) if entity_id in self.entities.entities else entity_id
        return {"entity_id": eid, "state": self.ledger.entity_state_at(eid, valid_at=valid_at, known_at=known_at),
                **self._meta()}

    def current(self, entity_id: str, attribute: str, *, now: datetime) -> dict:
        return {**self.ledger.get_current(entity_id, attribute, now=now), **self._meta()}

    def snapshot(self, *, valid_at: datetime, known_at: Optional[datetime] = None) -> WorldSnapshot:
        t = aware("valid_at", valid_at)
        k = aware("known_at", known_at) if known_at is not None else t
        state, modes = {}, set()
        for e in sorted(self.ledger.entities()):
            attrs = {}
            for attr, r in self.ledger.entity_state_at(e, valid_at=t, known_at=k).items():
                a = r["assertion"]
                dm = None if a is None else dict(a.metadata).get("data_mode")
                if dm:
                    modes.add(dm)
                attrs[attr] = {"status": r["status"], "value": None if a is None else a.value,
                               "assertion_id": None if a is None else a.assertion_id, "data_mode": dm}
            state[e] = attrs
        body = {"valid_at": t.isoformat(), "known_at": k.isoformat(), "state": state,
                "topology": sorted(e.edge_id for e in self.topology.edges_at(t, known_at=k))}
        m = self._meta()
        wm = weakest_mode(modes)
        return WorldSnapshot(f"snapshot:{digest(body)[:16]}", t.isoformat(), k.isoformat(), self.world_model_version,
                             m["topology_version"], m["journal_head"], m["data_mode"],
                             None if wm is None else wm.value, tuple(sorted(modes)), state)

    # ---------------------------------------------------------------- пространство
    def position_of(self, entity_id: str, t: datetime, *, known_at: Optional[datetime] = None) -> Optional[dict]:
        """Положение в корневом кадре: сначала утверждение position_world, иначе центр геометрии L0/L1."""
        r = self.ledger.get_as_of(entity_id, "position_world", valid_at=t, known_at=known_at) \
            if "position_world" in self.policy.attribute_class else {"assertion": None}
        a = r["assertion"]
        if a is not None:
            return {"position": list(a.value), "source": "assertion", "assertion_id": a.assertion_id,
                    "covariance": a.metadata.get("position_covariance")}
        c = self.geometry.center(entity_id, t, known_at=known_at)
        return None if c is None else {"position": c.tolist(), "source": "geometry_center"}

    def locate(self, entity_id: str, t: datetime) -> dict:
        """Где объект: цепочка «содержится в» (сущности и топология), кадр привязки, соседи по физике."""
        if entity_id not in self.entities.entities:
            raise SpatialWorldError(f"unknown entity {entity_id}")
        chain, cur, seen = [], entity_id, set()
        while cur and cur not in seen:
            seen.add(cur)
            parent = self.entities.entities[cur].parent_id
            if parent is None:
                ups = self.topology.neighbors(cur, t, types=("located_in",))
                parent = ups[0] if ups else None
            if parent:
                chain.append(parent)
            cur = parent
        e = self.entities.entities[entity_id]
        return {"entity_id": entity_id, "contained_in": chain, "frame": e.spatial_anchor,
                "frame_chain": None if not e.spatial_anchor else list(self.frames.chain(e.spatial_anchor, ROOT, at=t)),
                "position": self.position_of(entity_id, t),
                "physically_upstream": self.topology.upstream(entity_id, t),
                "contains": self.topology.neighbors(entity_id, t, types=HIERARCHY_TYPES[:1]) or self.entities.children(entity_id),
                **self._meta()}

    def near(self, a: str, b: str, t: datetime, *, scale: str) -> dict:
        pa, pb = self.position_of(a, t), self.position_of(b, t)
        if pa is None or pb is None:
            return {"near": None, "reason": "position_unknown", **self._meta()}
        return {**self.geometry.near(pa["position"], pb["position"], scale), **self._meta()}

    def who_can_confirm(self, target: str, t: datetime, *, health: Optional[dict] = None) -> dict:
        return {"target": target, "sensors": self.coverage.who_can_confirm(target, t, health=health), **self._meta()}

    def status(self) -> dict[str, Any]:
        return {**self._meta(), "entities": len(self.entities.entities), "frames": len(self.frames._edges),
                "assertions": len(self.ledger._entries), "open_conflicts": len(self.ledger.open_conflicts()),
                "quarantined": len(self.entities.quarantine), "journal_rows": len(self.journal),
                "journal_chain_ok": self.journal.verify()}


def positions_array(model: SpatialWorldModel, ids, t) -> np.ndarray:
    return np.array([model.position_of(i, t)["position"] for i in ids], float)


__all__ = ["SpatialWorldModel", "WorldSnapshot", "positions_array"]
