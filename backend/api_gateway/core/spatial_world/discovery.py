"""Ограниченный поиск причинных связей (WM Engineering Spec §2.23, формула F26; мастер-ТЗ B3; Gap Closure v2 §6).

F26: кандидатные пары (X, Y) = пути графа знаний по типам [causes, depends_on, uses, powered_by, supplies, serves]
глубиной ≤ D ∩ пространственная досягаемость (радиус масштаба) ∩ временное окно ∩ ontology.permits.
При CAUSAL_DISCOVERY_PHYSICS_BOUNDED поиск (ваш reasoning/causal_discovery.discover) перебирает ТОЛЬКО эти пары,
а не все N² сочетаний: высокая корреляция без физического пути отвергается, не доходя до статистики.
Найденная связь — только observational_candidate (через register_discovery_candidate реестра CEOS).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable, Iterable, Mapping, Optional

import numpy as np

from ..world_model.reality_formulas import FormulaReference
from .policy import SPEC, SpatialWorldError, aware

F_BOUNDED_PAIRS = FormulaReference("SPEC-F26-BOUNDED-PAIRS", SPEC, 14,
                                   "pairs = KG paths[causes, depends_on, uses, powered_by, supplies, serves] ≤ depth "
                                   "∩ spatial reachability(r per scale) ∩ temporal window ∩ ontology.permits")
KG_TYPES = ("causes", "depends_on", "uses", "powered_by", "supplies", "serves", "upstream_of", "connected_to")


def enumerate_candidate_pairs(model, *, at: datetime, variables: Mapping[str, Iterable[str]], max_depth: int,
                              scale: str, window_s: float,
                              ontology_permits: Callable[[str, str, str, str], bool],
                              known_at: Optional[datetime] = None) -> dict:
    """variables: entity_id → атрибуты-кандидаты. Возвращает пары ((e1, a1), (e2, a2)) и отчёт об отсечении."""
    if max_depth < 1 or window_s <= 0:
        raise SpatialWorldError("max_depth ≥ 1 and a positive temporal window come from policy")
    t = aware("at", at)
    r = model.policy.near_radii_m.get(scale)
    if r is None:
        raise SpatialWorldError(f"no reachability radius for scale {scale!r}")
    ents = sorted(variables)
    pos = {}
    for e in ents:
        p = model.position_of(e, t, known_at=known_at)
        pos[e] = None if p is None else np.asarray(p["position"], float)
    lo = t - timedelta(seconds=float(window_s))

    def recent(e: str, a: str) -> bool:
        return any(lo <= x.event_time <= t for x in model.ledger.history(e, a))

    pairs, dropped = [], {"no_kg_path": 0, "out_of_reach": 0, "no_recent_data": 0, "ontology": 0}
    total = 0
    for i, e1 in enumerate(ents):
        for e2 in ents[i + 1:]:
            for a1 in variables[e1]:
                for a2 in variables[e2]:
                    total += 1
            path = model.topology.path(e1, e2, t, types=KG_TYPES, max_depth=max_depth, known_at=known_at) or \
                model.topology.path(e2, e1, t, types=KG_TYPES, max_depth=max_depth, known_at=known_at)
            n = sum(1 for _ in variables[e1]) * sum(1 for _ in variables[e2])
            if path is None:
                dropped["no_kg_path"] += n
                continue
            if pos[e1] is None or pos[e2] is None or float(np.linalg.norm(pos[e1] - pos[e2])) > float(r):
                dropped["out_of_reach"] += n
                continue
            for a1 in variables[e1]:
                for a2 in variables[e2]:
                    if not (recent(e1, a1) and recent(e2, a2)):
                        dropped["no_recent_data"] += 1
                        continue
                    if not ontology_permits(e1, a1, e2, a2):
                        dropped["ontology"] += 1
                        continue
                    pairs.append({"x": (e1, a1), "y": (e2, a2), "kg_path": path})
    return {"pairs": pairs, "candidates_total": total, "dropped": dropped, "formula": F_BOUNDED_PAIRS}


def admit_discovered_edge(edge: Mapping[str, str], bounded: Mapping) -> dict:
    """Статистически найденная связь допускается к регистрации кандидатом только если её пара в ограниченном
    множестве. Иначе — отказ (например, корреляция 0.99 без физического пути)."""
    allowed = {(tuple(p["x"]), tuple(p["y"])) for p in bounded["pairs"]}
    x, y = (edge["cause_entity"], edge["cause_attribute"]), (edge["effect_entity"], edge["effect_attribute"])
    ok = (x, y) in allowed or (y, x) in allowed
    return {"admitted": ok, "lifecycle": "observational_candidate" if ok else None,
            "reason": None if ok else "no_physical_path_or_reach_or_window_or_ontology: rejected before statistics"}


__all__ = ["F_BOUNDED_PAIRS", "KG_TYPES", "admit_discovered_edge", "enumerate_candidate_pairs"]
