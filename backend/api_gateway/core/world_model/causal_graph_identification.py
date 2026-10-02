from __future__ import annotations
from typing import Iterable, Mapping
from .reality_formulas import FormulaReference, FormulaResult
from .causal_identification import (
    NEXT_ACTION_ADJUST,
    NEXT_ACTION_REQUEST_OBSERVATION,
    NON_IDENTIFIED_PLANNING_USE,
    IdentificationMethod,
    IdentificationResult,
    IdentificationStatus,
)

CCIA = "EXO-causal-reasoning-memory-planning-integration-addendum"
F_CRITERION = FormulaReference("CCIA-2.1-BACKDOOR-CRITERION", CCIA, 3, "Z blocks every backdoor path and contains no descendant of X")
F_EXPECTATION = FormulaReference("CCIA-2.1-BACKDOOR-EXPECTATION", CCIA, 2, "E[Y|do(X=x)] = sum_z E[Y|X=x,Z=z] P(Z=z)")


def _closure(edges, start, forward: bool) -> set:
    nbr = {}
    for a, b in edges:
        nbr.setdefault(a if forward else b, set()).add(b if forward else a)
    out, stack = set(), list(start)
    while stack:
        for n in nbr.get(stack.pop(), ()):
            if n not in out:
                out.add(n)
                stack.append(n)
    return out


def _nodes(edges) -> set:
    return {n for e in edges for n in e}


def d_separated(edges: Iterable[tuple[str, str]], xs, ys, zs, *, nodes: Iterable[str] | None = None) -> bool:
    """X ⊥ Y | Z через моральный граф предков (Lauritzen). edges = (причина, следствие)."""
    edges, xs, ys, zs = list(edges), set(xs), set(ys), set(zs)
    if xs & ys or (xs | ys) & zs:
        raise ValueError("X, Y, Z must be disjoint")
    unknown = (xs | ys | zs) - (set(nodes) if nodes is not None else _nodes(edges))
    if unknown:
        raise ValueError(f"unknown nodes in the causal graph: {sorted(unknown)}")
    keep = xs | ys | zs | _closure(edges, xs | ys | zs, forward=False)
    adj, parents = {n: set() for n in keep}, {}
    for a, b in edges:
        if a in keep and b in keep:
            adj[a].add(b)
            adj[b].add(a)
            parents.setdefault(b, []).append(a)
    for ps in parents.values():  # соединяем родителей общего ребёнка
        for i, p in enumerate(ps):
            for q in ps[i + 1:]:
                adj[p].add(q)
                adj[q].add(p)
    seen, stack = set(xs), list(xs)
    while stack:
        for n in adj.get(stack.pop(), ()):
            if n in zs or n in seen:
                continue
            if n in ys:
                return False
            seen.add(n)
            stack.append(n)
    return True


def backdoor_criterion(edges, treatment: str, outcome: str, adjustment, *, latent: Iterable[str] = ()) -> FormulaResult:
    """Латентные конфаундеры — узлы графа из `latent`: по ним корректировать нельзя,
    поэтому неперекрываемый латентный путь честно проваливает критерий."""
    edges, z, latent = list(edges), set(adjustment), set(latent)
    if treatment in z or outcome in z:
        raise ValueError("adjustment set cannot contain X or Y")
    graph_nodes = _nodes(edges)
    if {treatment, outcome} - graph_nodes:
        raise ValueError("treatment and outcome must be nodes of the graph")
    reasons = [f"latent_in_adjustment_set:{v}" for v in sorted(z & latent)]
    reasons += [f"descendant_of_treatment:{d}" for d in sorted(z & _closure(edges, {treatment}, forward=True))]
    if not d_separated([(a, b) for a, b in edges if a != treatment], {treatment}, {outcome}, z,
                       nodes=graph_nodes | z):
        reasons.append("open_backdoor_path")
    return FormulaResult(not reasons, F_CRITERION, {"reasons": tuple(reasons)})


def backdoor_adjustment_expectation(*, conditional_means: Mapping[str, float],
                                    stratum_probabilities: Mapping[str, float]) -> FormulaResult:
    """Для непрерывных метрик пилота: температура, кВт·ч, давление."""
    if not conditional_means or set(conditional_means) != set(stratum_probabilities):
        raise ValueError("strata must match and be non-empty")
    probs = {k: float(v) for k, v in stratum_probabilities.items()}
    if any(p < 0 for p in probs.values()) or abs(sum(probs.values()) - 1) > 1e-9:
        raise ValueError("P(Z) must be a distribution")
    terms = {k: float(conditional_means[k]) * probs[k] for k in probs}
    return FormulaResult(sum(terms.values()), F_EXPECTATION, {"stratum_terms": terms})


EXHAUSTIVE_SEARCH_LIMIT = 12  # algorithmic budget (2^12 d-separation tests), not a model constant


def _subsets_by_size(candidates: list[str], limit: int):
    from itertools import combinations

    for size in range(len(candidates) + 1):
        for subset in combinations(candidates, size):
            yield subset


def _inclusion_minimal(valid, start: tuple[str, ...]) -> tuple[str, ...]:
    current = list(start)
    changed = True
    while changed:
        changed = False
        for node in sorted(current):
            trial = [n for n in current if n != node]
            if valid(tuple(trial)):
                current, changed = trial, True
                break
    return tuple(sorted(current))


def minimal_backdoor_set(edges, treatment: str, outcome: str, *, latent: Iterable[str] = (),
                         measurable_latent: bool = False) -> tuple[str, ...] | None:
    """Smallest Z satisfying the backdoor criterion (None when no such Z exists).

    Candidates are An({X, Y}) minus De(X) minus {X, Y}: if any backdoor set exists, this one does
    (Tian, Paz & Pearl 1998), so the search is complete. With ``measurable_latent`` latent nodes are
    treated as if they could be measured -- the answer then names what must be observed.
    """
    edges, latent = list(edges), set(latent)
    descendants = _closure(edges, {treatment}, forward=True)
    ancestors = _closure(edges, {treatment, outcome}, forward=False)
    candidates = sorted((ancestors - descendants - {treatment, outcome}) - (set() if measurable_latent else latent))
    effective_latent = set() if measurable_latent else latent

    def valid(z: tuple[str, ...]) -> bool:
        return bool(backdoor_criterion(edges, treatment, outcome, z, latent=effective_latent).value)

    if not valid(tuple(candidates)):
        return None
    if len(candidates) <= EXHAUSTIVE_SEARCH_LIMIT:
        for subset in _subsets_by_size(candidates, EXHAUSTIVE_SEARCH_LIMIT):
            if valid(subset):
                return tuple(sorted(subset))
    return _inclusion_minimal(valid, tuple(candidates))


def identify_backdoor_from_graph(*, edges, treatment, outcome, adjustment, conditional_means,
                                 stratum_probabilities, unobserved_confounder_risk: float,
                                 latent: Iterable[str] = ()) -> IdentificationResult:
    """Даже при успехе — только review: ActionAuthority ≠ CausalConfidence (CEOS стр. 9).
    Риск незаявленных конфаундеров задаётся явно, без умолчания в 0.

    CCIA 2.1 (p.3): при отказе — причины отдельно (``reasons``), в ``required_confounders``
    минимальный достаточный набор (наблюдаемый backdoor-набор, если он есть, иначе минимальный набор
    переменных, которые нужно измерить), ``next_action`` — скорректировать или запросить наблюдение.
    """
    risk = float(unobserved_confounder_risk)
    if not 0 <= risk <= 1:
        raise ValueError("unobserved_confounder_risk must be in [0, 1]")
    edges, latent = list(edges), set(latent)
    check = backdoor_criterion(edges, treatment, outcome, adjustment, latent=latent)
    minimal = minimal_backdoor_set(edges, treatment, outcome, latent=latent)
    if not check.value:
        if minimal is not None:
            required, next_action = minimal, NEXT_ACTION_ADJUST
        else:
            required = minimal_backdoor_set(edges, treatment, outcome, latent=latent, measurable_latent=True) or ()
            next_action = NEXT_ACTION_REQUEST_OBSERVATION
        return IdentificationResult(
            treatment, outcome, IdentificationStatus.NON_IDENTIFIED, IdentificationMethod.ADJUSTMENT,
            tuple(required), 1.0, ("graph is correct",), NON_IDENTIFIED_PLANNING_USE, None,
            reasons=tuple(check.intermediates["reasons"]), next_action=next_action, formula=F_CRITERION,
        )
    effect = backdoor_adjustment_expectation(conditional_means=conditional_means,
                                             stratum_probabilities=stratum_probabilities).value
    return IdentificationResult(treatment, outcome, IdentificationStatus.IDENTIFIED, IdentificationMethod.ADJUSTMENT,
                                tuple(minimal or adjustment), risk,
                                ("graph is correct", "consistency", "positivity", "no undeclared confounders"),
                                "review", effect, expression="Σ_z E[Y|X=x,Z=z] P(Z=z)", formula=F_EXPECTATION)
