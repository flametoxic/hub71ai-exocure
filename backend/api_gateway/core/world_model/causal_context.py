from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

from .assertions import require_aware
from .reality_contracts import RealityState


@dataclass(frozen=True)
class CausalContext:
    """C_t(q) = {X_local, G_spatial, G_topology, E_1:t, U_exo, P_policy, Sigma_t} (CEOS p.3).

    All mappings are deep copies of the caller's data: the context never aliases world state.
    """

    query_id: str
    entity_ids: tuple[str, ...]
    physical_state: Mapping[str, Any]
    spatial_neighbors: Mapping[str, tuple[str, ...]]
    topology_edges: tuple[Mapping[str, Any], ...]
    recent_events: tuple[Mapping[str, Any], ...]
    active_policies: tuple[str, ...]
    maintenance_modes: Mapping[str, str]
    sensor_coverage: Mapping[str, float]
    uncertainty: Mapping[str, float]
    exogenous_context: Mapping[str, Any]
    as_of: datetime
    excluded_events: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.query_id.strip() or not self.entity_ids:
            raise ValueError("causal context requires query_id and at least one entity")
        object.__setattr__(self, "as_of", require_aware(self.as_of, "as_of"))


def _parse_time(value: Any, field_name: str) -> datetime:
    if isinstance(value, datetime):
        return require_aware(value, field_name)
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return require_aware(datetime.fromisoformat(text), field_name)
    raise ValueError(f"{field_name} must be a timezone-aware datetime or ISO-8601 string")


class CausalContextBuilder:
    """Build C_t(q) over only the physically and operationally relevant subgraph, cut at time t."""

    def __init__(self, *, max_topology_depth: int = 2, event_window: timedelta | None = None) -> None:
        if max_topology_depth < 0:
            raise ValueError("max_topology_depth must be non-negative")
        if event_window is not None and event_window <= timedelta(0):
            raise ValueError("event_window must be positive")
        self.max_topology_depth = int(max_topology_depth)
        self.event_window = event_window

    def build(
        self,
        *,
        query: Mapping[str, Any],
        physical_state: Mapping[str, Any],
        topology_edges: Sequence[Mapping[str, Any]],
        spatial_neighbors: Mapping[str, Sequence[str]],
        recent_events: Sequence[Mapping[str, Any]],
        active_policies: Sequence[str],
        maintenance_modes: Mapping[str, str],
        sensor_coverage: Mapping[str, float],
        uncertainty: Mapping[str, float],
        exogenous_context: Mapping[str, Any],
        as_of: datetime | str | None = None,
    ) -> CausalContext:
        query_id = str(query.get("query_id") or query.get("incident_id") or "").strip()
        roots = {
            str(value).strip()
            for value in (
                *(query.get("entity_ids") or ()),
                query.get("entity_id"),
                query.get("target_entity_id"),
            )
            if str(value or "").strip()
        }
        if not query_id or not roots:
            raise ValueError("query must identify itself and its affected entities")
        moment_value = as_of if as_of is not None else query.get("as_of")
        if moment_value is None:
            raise ValueError("causal context requires the query time t (as_of): E_1:t is cut at t")
        moment = _parse_time(moment_value, "as_of")

        normalized_edges = tuple(deepcopy(dict(edge)) for edge in topology_edges)
        adjacency: dict[str, set[str]] = {}
        for edge in normalized_edges:
            source = str(edge.get("source") or edge.get("cause") or "").strip()
            target = str(edge.get("target") or edge.get("effect") or "").strip()
            if not source or not target:
                continue
            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)

        local = set(roots)
        frontier: deque[tuple[str, int]] = deque((root, 0) for root in sorted(roots))
        while frontier:
            entity_id, depth = frontier.popleft()
            if depth >= self.max_topology_depth:
                continue
            for neighbor in sorted(adjacency.get(entity_id, set())):
                if neighbor in local:
                    continue
                local.add(neighbor)
                frontier.append((neighbor, depth + 1))

        local_spatial: dict[str, tuple[str, ...]] = {}
        for entity_id in tuple(local):
            neighbors = tuple(sorted({str(value) for value in spatial_neighbors.get(entity_id, ()) if str(value)}))
            if neighbors:
                local_spatial[entity_id] = neighbors
                local.update(neighbors)

        local_edges = tuple(
            edge
            for edge in normalized_edges
            if str(edge.get("source") or edge.get("cause") or "") in local
            and str(edge.get("target") or edge.get("effect") or "") in local
        )
        local_events, excluded = self._events_up_to(recent_events, local=local, moment=moment)
        return CausalContext(
            query_id=query_id,
            entity_ids=tuple(sorted(local)),
            physical_state={key: deepcopy(value) for key, value in physical_state.items() if str(key) in local},
            spatial_neighbors=local_spatial,
            topology_edges=local_edges,
            recent_events=local_events,
            active_policies=tuple(dict.fromkeys(str(value) for value in active_policies if str(value))),
            maintenance_modes={key: value for key, value in maintenance_modes.items() if str(key) in local},
            sensor_coverage={key: float(value) for key, value in sensor_coverage.items() if str(key) in local},
            uncertainty={key: float(value) for key, value in uncertainty.items() if str(key) in local},
            exogenous_context=deepcopy(dict(exogenous_context)),
            as_of=moment,
            excluded_events=excluded,
        )

    def build_from_reality_state(
        self,
        *,
        query: Mapping[str, Any],
        reality_state: RealityState,
        recent_events: Sequence[Mapping[str, Any]] = (),
        active_policies: Sequence[str] = (),
        maintenance_modes: Mapping[str, str] | None = None,
        sensor_coverage: Mapping[str, float] | None = None,
        uncertainty: Mapping[str, float] | None = None,
        exogenous_context: Mapping[str, Any] | None = None,
    ) -> CausalContext:
        topology = dict(reality_state.spatial_topology)
        return self.build(
            query=query,
            physical_state=reality_state.physical_state,
            topology_edges=tuple(topology.get("edges") or ()),
            spatial_neighbors=dict(topology.get("spatial_neighbors") or {}),
            recent_events=recent_events,
            active_policies=active_policies,
            maintenance_modes=maintenance_modes or {},
            sensor_coverage=sensor_coverage or {},
            uncertainty=uncertainty or {},
            exogenous_context=exogenous_context or {},
            as_of=query.get("as_of") or reality_state.observed_at,
        )

    def _events_up_to(
        self,
        events: Sequence[Mapping[str, Any]],
        *,
        local: set[str],
        moment: datetime,
    ) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
        """E_1:t: in-scope events with event_time <= t (and inside the optional look-back window).

        Out-of-time or undatable events are not silently dropped: they are reported with a reason.
        """

        kept: list[Mapping[str, Any]] = []
        excluded: list[Mapping[str, Any]] = []
        start = moment - self.event_window if self.event_window is not None else None
        for event in events:
            if not self._event_entities(event).intersection(local):
                continue
            reference = str(event.get("event_id") or event.get("id") or "")
            try:
                event_time = _parse_time(event.get("event_time"), "event_time")
            except ValueError as exc:
                excluded.append({"event_id": reference, "reason": f"invalid_event_time: {exc}"})
                continue
            if event_time > moment:
                excluded.append({"event_id": reference, "reason": "after_query_time"})
                continue
            if start is not None and event_time < start:
                excluded.append({"event_id": reference, "reason": "outside_event_window"})
                continue
            kept.append(deepcopy(dict(event)))
        return tuple(kept), tuple(excluded)

    @staticmethod
    def _event_entities(event: Mapping[str, Any]) -> set[str]:
        values = {
            event.get("entity_id"),
            event.get("actor_entity_id"),
            event.get("target_entity_id"),
        }
        values.update(event.get("entity_ids") or ())
        return {str(value) for value in values if str(value or "").strip()}


__all__ = ["CausalContext", "CausalContextBuilder"]

