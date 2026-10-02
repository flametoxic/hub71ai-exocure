"""TopologyService (мастер-ТЗ A3; Gap Closure v1 §4; City WM §2.2; WM Spec §2.4 TopologyChangeDetector).

Версионированный граф физических и операционных связей. Каждое ребро имеет valid_from/valid_to и
событие-доказательство. Переключение (задвижка, байпас, резерв) не перезаписывает граф:
    edge_old.valid_to = t_switch;  edge_new.valid_from = t_switch;  topology_version ← topology_version + 1
Поэтому историческая топология восстанавливается на любой момент события. Топология не устаревает
по TTL — она версионируется. Путь по физическим рёбрам — основание для причинных гипотез.
"""
from __future__ import annotations

import secrets
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

from .journal import WorldJournal
from .policy import F_SWITCH, SpatialWorldError, aware, text

EDGE_TYPES = ("contains", "located_in", "connected_to", "supplies", "powered_by", "serves", "upstream_of",
              "downstream_of", "controls", "observes", "has_redundancy_with", "depends_on")
PHYSICAL_TYPES = ("connected_to", "supplies", "powered_by", "serves", "upstream_of", "downstream_of",
                  "has_redundancy_with")
OPERATIONAL_TYPES = ("controls", "observes", "depends_on")
HIERARCHY_TYPES = ("contains", "located_in")


@dataclass(frozen=True)
class TopologyEdge:
    edge_id: str
    source: str
    target: str
    edge_type: str
    directed: bool
    capacity: Optional[float]
    weight: Optional[float]
    valid_from: datetime
    evidence_event_id: str
    opened_in_version: int
    knowledge_time: datetime


class TopologyService:
    def __init__(self, journal: WorldJournal, *, entity_exists=None) -> None:
        self.journal = journal
        self._entity_exists = entity_exists
        self.edges: dict[str, TopologyEdge] = {}
        self._closed: dict[str, tuple] = {}          # edge_id → (valid_to, knowledge_time, version)
        self.version = 0
        self.versions: list[dict] = []
        for r in journal.rows(("topology_open", "topology_close", "topology_version")):
            self._apply(r)

    def _apply(self, r) -> None:
        p = r.payload
        if r.kind == "topology_open":
            self.edges[p["edge_id"]] = TopologyEdge(p["edge_id"], p["source"], p["target"], p["edge_type"], p["directed"],
                                                    p.get("capacity"), p.get("weight"),
                                                    datetime.fromisoformat(p["valid_from"]), p["evidence_event_id"],
                                                    p["version"], r.knowledge_time)
        elif r.kind == "topology_close":
            self._closed[p["edge_id"]] = (datetime.fromisoformat(p["valid_to"]), r.knowledge_time, p["version"])
        else:
            self.version = p["version"]
            self.versions.append(dict(p, knowledge_time=r.knowledge_time.isoformat()))

    def _bump(self, reason: str, evidence: str, t: datetime, kt: datetime) -> int:
        self._apply(self.journal.append("topology_version", {"version": self.version + 1, "reason": reason,
                                                             "evidence_event_id": evidence, "valid_from": t.isoformat()},
                                        knowledge_time=kt))
        return self.version

    def _open(self, source, target, edge_type, *, directed, capacity, weight, t, evidence, kt, version) -> TopologyEdge:
        if edge_type not in EDGE_TYPES:
            raise SpatialWorldError(f"edge type {edge_type!r} is not one of {EDGE_TYPES}")
        for x in (source, target):
            if self._entity_exists is not None and not self._entity_exists(x):
                raise SpatialWorldError(f"topology edge endpoint {x} is not a canonical entity")
        eid = f"edge:{secrets.token_hex(6)}"
        self._apply(self.journal.append("topology_open", {
            "edge_id": eid, "source": text("source", source), "target": text("target", target), "edge_type": edge_type,
            "directed": bool(directed), "capacity": capacity, "weight": weight, "valid_from": t.isoformat(),
            "evidence_event_id": text("evidence_event_id", evidence), "version": version}, knowledge_time=kt))
        return self.edges[eid]

    def add_edge(self, source: str, target: str, edge_type: str, *, valid_from: datetime, evidence_event_id: str,
                 knowledge_time: datetime, directed: bool = True, capacity: Optional[float] = None,
                 weight: Optional[float] = None) -> TopologyEdge:
        t, kt = aware("valid_from", valid_from), aware("knowledge_time", knowledge_time)
        v = self._bump(f"add:{edge_type}", evidence_event_id, t, kt)
        return self._open(source, target, edge_type, directed=directed, capacity=capacity, weight=weight, t=t,
                          evidence=evidence_event_id, kt=kt, version=v)

    def on_switching_event(self, *, close_edge_ids: Iterable[str], open_edges: Iterable[dict], t_switch: datetime,
                           evidence_event_id: str, knowledge_time: datetime) -> int:
        """F_SWITCH: закрыть затронутые рёбра, открыть новые, одна новая версия топологии."""
        t, kt = aware("t_switch", t_switch), aware("knowledge_time", knowledge_time)
        close_ids = list(close_edge_ids)
        for eid in close_ids:
            if eid not in self.edges or eid in self._closed:
                raise SpatialWorldError(f"edge {eid} is not open")
        v = self._bump("switch", evidence_event_id, t, kt)
        for eid in close_ids:
            self._apply(self.journal.append("topology_close", {"edge_id": eid, "valid_to": t.isoformat(), "version": v,
                                                               "evidence_event_id": evidence_event_id}, knowledge_time=kt))
        for spec in open_edges:
            self._open(spec["source"], spec["target"], spec["edge_type"], directed=spec.get("directed", True),
                       capacity=spec.get("capacity"), weight=spec.get("weight"), t=t, evidence=evidence_event_id,
                       kt=kt, version=v)
        return v

    # ---------------------------------------------------------------- запросы на момент времени
    def edges_at(self, t: datetime, *, known_at: Optional[datetime] = None,
                 types: Optional[Iterable[str]] = None) -> list[TopologyEdge]:
        t = aware("t", t)
        k = aware("known_at", known_at) if known_at is not None else t
        ts = None if types is None else set(types)
        out = []
        for e in self.edges.values():
            if e.knowledge_time > k or e.valid_from > t or (ts is not None and e.edge_type not in ts):
                continue
            c = self._closed.get(e.edge_id)
            if c is not None and c[1] <= k and c[0] <= t:
                continue
            out.append(e)
        return out

    def neighbors(self, entity_id: str, t: datetime, *, types: Optional[Iterable[str]] = None,
                  direction: str = "out", known_at: Optional[datetime] = None) -> list[str]:
        out = []
        for e in self.edges_at(t, known_at=known_at, types=types):
            if direction in ("out", "both") and e.source == entity_id:
                out.append(e.target)
            if direction in ("in", "both") and e.target == entity_id:
                out.append(e.source)
            if not e.directed and direction == "out" and e.target == entity_id:
                out.append(e.source)
        return sorted(set(out))

    def path(self, a: str, b: str, t: datetime, *, types: Iterable[str] = PHYSICAL_TYPES,
             max_depth: Optional[int] = None, known_at: Optional[datetime] = None) -> Optional[list[str]]:
        """Кратчайший путь по рёбрам заданных типов (BFS) — основание для структурной причинной гипотезы."""
        types = tuple(types)
        prev, q = {a: None}, deque([(a, 0)])
        while q:
            n, d = q.popleft()
            if n == b:
                path = [n]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            if max_depth is not None and d >= max_depth:
                continue
            for m in self.neighbors(n, t, types=types, known_at=known_at):
                if m not in prev:
                    prev[m] = n
                    q.append((m, d + 1))
        return None

    def upstream(self, entity_id: str, t: datetime, *, known_at: Optional[datetime] = None) -> list[str]:
        """Всё, что физически влияет на объект (обход против направления supplies/powered_by/serves/upstream_of)."""
        seen, q = set(), deque([entity_id])
        while q:
            n = q.popleft()
            for e in self.edges_at(t, known_at=known_at, types=("supplies", "powered_by", "serves", "upstream_of")):
                src = e.source if e.edge_type != "powered_by" else e.target
                dst = e.target if e.edge_type != "powered_by" else e.source
                if dst == n and src not in seen and src != entity_id:
                    seen.add(src)
                    q.append(src)
        return sorted(seen)

    def detect_changes(self, observed: Iterable[tuple], t: datetime) -> list[dict]:
        """Детектор изменений: наблюдаемые связи против графа → кандидаты на переключение (не запись!)."""
        obs = {(s, d, ty) for s, d, ty in observed}
        cur = {(e.source, e.target, e.edge_type): e.edge_id for e in self.edges_at(t)}
        out = []
        for key in sorted(obs - set(cur)):
            out.append({"kind": "edge_observed_not_in_topology", "edge": key, "status": "candidate"})
        types = {ty for _, _, ty in obs}
        for key, eid in sorted(cur.items()):
            if key[2] in types and key not in obs:
                out.append({"kind": "edge_in_topology_not_observed", "edge": key, "edge_id": eid, "status": "candidate"})
        return out


__all__ = ["EDGE_TYPES", "F_SWITCH", "HIERARCHY_TYPES", "OPERATIONAL_TYPES", "PHYSICAL_TYPES", "TopologyEdge",
           "TopologyService"]
