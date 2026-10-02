from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from datetime import datetime
from typing import Callable, Mapping
import numpy as np
from .reality_formulas import FormulaReference, FormulaResult

UMRM = "CURE-Unified-Mathematical-Reality-Model"
CEOS = "EXO-causal-engine-operational-specification"
F_CONTEXT = FormulaReference("CEOS-4-CAUSAL-CONTEXT", CEOS, 3, "C_t(q) = {X_local, G_spatial, G_topology, E_1:t, U_exo, P_policy, Sigma_t}")
F_H4 = FormulaReference("UMRM-7-ADMISSIBLE-CAUSAL-SET", UMRM, 6, "H_causal = G_topology ∩ G_spatial ∩ G_temporal ∩ G_ontology")
F_H5 = FormulaReference("CEOS-6-DISCOVERY-BOUNDARY", CEOS, 5, "H_causal = G_topology ∩ G_spatial ∩ G_temporal ∩ G_ontology ∩ G_constraints")
F_REGISTRY = FormulaReference("CEOS-5-MECHANISM-REGISTRY", CEOS, 3, "X_i := f_i(PA_i, U_i; theta_i) — versioned mechanism with trust ladder")
F_BELIEF = FormulaReference("CEOS-8-BELIEF", CEOS, 6, "Belief(q) = f(E_independent, Q_source, F_freshness, C_consistency, A_alternatives)")
F_CONSTRAINTS = FormulaReference("CEOS-8-CONSTRAINT-SET", CEOS, 6, "X_valid = {x : g_i(x) <= 0, h_j(x) = 0}")


@dataclass(frozen=True)
class CausalContext:
    query_id: str
    local_entities: frozenset
    spatial_neighbors: frozenset
    topology_edges: tuple
    events: tuple
    exogenous: Mapping[str, float]
    active_policies: tuple
    uncertainty: Mapping[str, float]


def build_causal_context(*, query_id, affected, topology_edges, spatial_neighbors, events, exogenous,
                         active_policies, uncertainty, window: tuple[datetime, datetime], hops: int = 2) -> FormulaResult:
    """Рассуждение никогда не идёт по всему городу: только локальный подграф вокруг инцидента."""
    if not affected or hops < 0:
        raise ValueError("affected entities required, hops >= 0")
    adj = {}
    for a, b, _kind in topology_edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    local, frontier = set(affected), deque((e, 0) for e in affected)
    while frontier:
        node, depth = frontier.popleft()
        if depth < hops:
            for nxt in adj.get(node, ()):
                if nxt not in local:
                    local.add(nxt)
                    frontier.append((nxt, depth + 1))
    spatial = {n for e in affected for n in spatial_neighbors.get(e, ())}
    scope, (start, end) = local | spatial, window
    ctx = CausalContext(query_id, frozenset(local), frozenset(spatial),
                        tuple(e for e in topology_edges if e[0] in scope and e[1] in scope),
                        tuple(ev for ev in events if ev["entity_id"] in scope and start <= ev["event_time"] <= end),
                        dict(exogenous), tuple(active_policies),
                        {k: float(v) for k, v in uncertainty.items() if k in scope})
    return FormulaResult(ctx, F_CONTEXT, {"hops": hops, "entities_in_scope": len(scope)})


def admissible_causal_pairs(*, topology, spatial, temporal, ontology, constraints=None) -> FormulaResult:
    """Пара (причина, следствие) вне пересечения не доходит до корреляционных тестов."""
    sets = [frozenset(topology), frozenset(spatial), frozenset(temporal), frozenset(ontology)]
    if constraints is not None:
        sets.append(frozenset(constraints))
    return FormulaResult(frozenset.intersection(*sets), F_H5 if constraints is not None else F_H4,
                         {"set_sizes": tuple(len(s) for s in sets)})


# CEOS p.4-5: the one mechanism registry (trust ladder, "May affect action?" table, origin caps)
# is causal_registry.CausalMechanismRegistry. This module keeps only the formulas it builds on.
NON_OBSERVATIONAL_PREFIXES = ("llm:", "moe:", "sim:", "simulation:", "synthetic:", "counterfactual:")


def belief_support(*, independent_evidence, source_quality, freshness, consistency, alternatives,
                   weights: Mapping[str, float]) -> FormulaResult:
    """Форма f: нормированная взвешенная сумма; сильные альтернативы снижают belief.
    Статусы observed/inferred/predicted/simulated здесь не смешиваются."""
    names = ("independent_evidence", "source_quality", "freshness", "consistency", "alternatives")
    vals = dict(zip(names, (independent_evidence, source_quality, freshness, consistency, alternatives)))
    if set(weights) != set(names):
        raise ValueError(f"weights for {names} required")
    if any(not 0 <= float(v) <= 1 for v in (*vals.values(), *weights.values())):
        raise ValueError("inputs and weights must be in [0, 1]")
    pos = sum(weights[k] for k in names[:4])
    if pos <= 0:
        raise ValueError("positive weights must not all be zero")
    raw = sum(weights[k] * vals[k] for k in names[:4]) / pos - weights["alternatives"] * vals["alternatives"]
    return FormulaResult(min(max(raw, 0.0), 1.0), F_BELIEF, {"raw": raw})


def constraint_check(x, *, inequalities: Mapping[str, Callable], equalities: Mapping[str, Callable] | None = None,
                     tolerance: float = 1e-9) -> FormulaResult:
    """Физика, кинематика, топология, политика, порядок во времени, законы сохранения."""
    margins, violations = {}, []
    for name, g in inequalities.items():
        v = float(g(x))
        margins[name] = -v
        if not np.isfinite(v) or v > tolerance:
            violations.append(name)
    for name, h in (equalities or {}).items():
        v = float(h(x))
        margins[name] = -abs(v)
        if not np.isfinite(v) or abs(v) > tolerance:
            violations.append(name)
    return FormulaResult(not violations, F_CONSTRAINTS, {"margins": margins, "violations": tuple(violations)})


# ---------------------------------------------------------------- совместимость со старым API
# Прежние типы лестницы доверия оставлены как данные (ничего не решают). Реестр механизмов в системе один —
# causal_registry.CausalMechanismRegistry (CEOS): старые имена указывают на него, второго реестра нет.
class TrustLevel(IntEnum):
    STRUCTURAL_POSSIBLE = 0
    OBSERVATIONAL_CANDIDATE = 1
    PHYSICS_GROUNDED = 2
    INTERVENTION_VALIDATED = 3


CEOS_LEVEL_NAMES = {0: "structural_possible", 1: "observational_candidate", 2: "mechanism_grounded",
                    3: "intervention_validated"}  # CEOS стр. 4 называет ступень 2 mechanism_grounded; Master — physics_grounded


@dataclass(frozen=True)
class Mechanism:
    edge_id: str
    source: str
    target: str
    mechanism_type: str
    functional_form: str
    physical_basis: str
    scope: str
    lag_p50_s: float
    lag_p95_s: float
    trust: TrustLevel
    evidence_refs: tuple = ()
    counterexamples: tuple = ()
    outcome_refs: tuple = ()  # id проверенных исходов из OutcomeLedger (не число от вызывающего)
    scm_version: str = "scm:0"
    version: int = 1
    valid_to: datetime | None = None  # механизм с истёкшим сроком не даёт права на планирование


_REGISTRY_NAMES = ("CausalMechanismRegistry", "ALLOWED_USE", "ORIGIN_CAP", "REGISTRATION_CEILING")


def __getattr__(name: str):
    if name in _REGISTRY_NAMES:
        from . import causal_registry

        return getattr(causal_registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
