"""Полный аппарат Перла для полумарковских моделей: ADMG, c-компоненты, do-исчисление, алгоритм ID.

* ``ADMG`` — ацикличный смешанный граф: направленные рёбра (причина, следствие) и двунаправленные
  рёбра ``a <-> b`` (общая латентная причина). Строится явно или латентной проекцией DAG с латентными
  узлами (Verma): ``a -> b``, если есть направленный путь a -> ... -> b только через латентные узлы;
  ``a <-> b``, если есть латентный узел L с направленными путями к a и к b только через латентные узлы.
* Три правила do-исчисления (Pearl 1995, Theorem 3) — проверяемые функции: каждая строит нужный
  модифицированный граф (G_{\\bar X}, G_{\\bar X \\underline Z}, G_{\\bar X \\overline{Z(W)}}) и проверяет
  d-разделение.
* Алгоритм ID (Shpitser & Pearl 2006, Fig. 3): либо символьное выражение P_x(y) через наблюдаемое
  распределение (дерево сумм / произведений / частных условных вероятностей), либо hedge — пара
  c-лесов F, F', доказывающая неидентифицируемость (Theorem 4 статьи).
* Оценщик выражения на дискретных данных (эмпирические частоты) или на точной таблице распределения.
  Пустая ячейка условной вероятности — нарушение positivity: ошибка, а не молчаливый ноль.

Наблюдательная идентификация не даёт права на действие: максимум ``review``
(CEOS стр. 9: ActionAuthority != CausalConfidence).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product as cartesian
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .causal_graph_identification import d_separated as _dag_d_separated
from .reality_formulas import FormulaReference, FormulaResult

SHPITSER_PEARL_2006 = (
    "Shpitser & Pearl (2006), Identification of Joint Interventional Distributions in Recursive "
    "Semi-Markovian Causal Models, AAAI-06"
)
PEARL_1995 = "Pearl (1995), Causal diagrams for empirical research, Biometrika 82(4), Theorem 3"

F_ID = FormulaReference(
    "PEARL-ID-ALGORITHM", SHPITSER_PEARL_2006, 0,
    "ID(y, x, P, G), Fig. 3: lines 1-7; FAIL(F, F') = hedge for P_x(y) (Theorem 4)",
)
F_RULE1 = FormulaReference(
    "PEARL-DO-RULE-1", PEARL_1995, 0,
    "P(y|do(x),z,w) = P(y|do(x),w) if (Y ⊥ Z | X, W) in G_{bar X}",
)
F_RULE2 = FormulaReference(
    "PEARL-DO-RULE-2", PEARL_1995, 0,
    "P(y|do(x),do(z),w) = P(y|do(x),z,w) if (Y ⊥ Z | X, W) in G_{bar X, underline Z}",
)
F_RULE3 = FormulaReference(
    "PEARL-DO-RULE-3", PEARL_1995, 0,
    "P(y|do(x),do(z),w) = P(y|do(x),w) if (Y ⊥ Z | X, W) in G_{bar X, overline Z(W)}, Z(W) = Z \\ An(W)_{G_{bar X}}",
)
F_ID_ESTIMATE = FormulaReference(
    "PEARL-ID-PLUGIN-ESTIMATE", SHPITSER_PEARL_2006, 0,
    "P_x(y) evaluated by plugging empirical conditional frequencies into the ID expression",
)

MAX_PLANNING_USE = "review"


# ============================================================================ graph


def _pair(a: str, b: str) -> frozenset[str]:
    return frozenset((a, b))


@dataclass(frozen=True)
class ADMG:
    """Acyclic directed mixed graph. Bidirected edges are unordered pairs of distinct nodes."""

    nodes: frozenset[str]
    directed: frozenset[tuple[str, str]] = frozenset()
    bidirected: frozenset[frozenset[str]] = frozenset()

    def __post_init__(self) -> None:
        nodes = frozenset(str(node) for node in self.nodes)
        directed = frozenset((str(a), str(b)) for a, b in self.directed)
        bidirected = frozenset(frozenset(str(v) for v in pair) for pair in self.bidirected)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "directed", directed)
        object.__setattr__(self, "bidirected", bidirected)
        for a, b in directed:
            if a == b:
                raise ValueError(f"self-loop {a}->{a} is not allowed")
            if a not in nodes or b not in nodes:
                raise ValueError(f"directed edge {a}->{b} uses a node outside the graph")
        for pair in bidirected:
            if len(pair) != 2 or not pair <= nodes:
                raise ValueError(f"bidirected edge {sorted(pair)} must join two distinct graph nodes")
        self.topological_order()  # raises on a directed cycle

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_edges(
        cls,
        directed: Iterable[tuple[str, str]],
        bidirected: Iterable[tuple[str, str]] = (),
        *,
        nodes: Iterable[str] = (),
    ) -> "ADMG":
        directed = [(str(a), str(b)) for a, b in directed]
        bidirected = [(str(a), str(b)) for a, b in bidirected]
        all_nodes = {str(n) for n in nodes} | {n for e in directed for n in e} | {n for e in bidirected for n in e}
        return cls(frozenset(all_nodes), frozenset(directed), frozenset(_pair(a, b) for a, b in bidirected))

    @classmethod
    def from_latent_dag(
        cls,
        edges: Iterable[tuple[str, str]],
        latent: Iterable[str] = (),
        *,
        nodes: Iterable[str] = (),
    ) -> "ADMG":
        """Latent projection of a DAG with explicit latent nodes onto its observed nodes (Verma 1993)."""

        edges = [(str(a), str(b)) for a, b in edges]
        latent = {str(v) for v in latent}
        all_nodes = {str(n) for n in nodes} | {n for e in edges for n in e}
        unknown = latent - all_nodes
        if unknown:
            raise ValueError(f"latent variables are not nodes of the graph: {sorted(unknown)}")
        children: dict[str, set[str]] = {}
        for a, b in edges:
            children.setdefault(a, set()).add(b)
        observed = all_nodes - latent

        def observed_reach(start: str) -> set[str]:
            """Observed nodes reachable from start by a directed path whose intermediates are latent."""
            out, stack, seen = set(), list(children.get(start, ())), set()
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                if node in latent:
                    stack.extend(children.get(node, ()))
                else:
                    out.add(node)
            return out

        directed = {(a, b) for a in observed for b in observed_reach(a) if a != b}
        bidirected: set[frozenset[str]] = set()
        for hidden in latent:
            reach = sorted(observed_reach(hidden))
            for i, a in enumerate(reach):
                for b in reach[i + 1:]:
                    bidirected.add(_pair(a, b))
        return cls(frozenset(observed), frozenset(directed), frozenset(bidirected))

    # ---------------------------------------------------------------- queries
    def parents(self, node: str) -> frozenset[str]:
        return frozenset(a for a, b in self.directed if b == node)

    def children(self, node: str) -> frozenset[str]:
        return frozenset(b for a, b in self.directed if a == node)

    def ancestors(self, targets: Iterable[str]) -> frozenset[str]:
        """An(S)_G, S included (Shpitser & Pearl convention)."""
        out = set(self._require(targets))
        stack = list(out)
        while stack:
            node = stack.pop()
            for parent in self.parents(node):
                if parent not in out:
                    out.add(parent)
                    stack.append(parent)
        return frozenset(out)

    def descendants(self, sources: Iterable[str]) -> frozenset[str]:
        """De(S)_G, S included."""
        out = set(self._require(sources))
        stack = list(out)
        while stack:
            node = stack.pop()
            for child in self.children(node):
                if child not in out:
                    out.add(child)
                    stack.append(child)
        return frozenset(out)

    def topological_order(self) -> tuple[str, ...]:
        indegree = {node: 0 for node in self.nodes}
        for _, b in self.directed:
            indegree[b] += 1
        ready = sorted(node for node, degree in indegree.items() if degree == 0)
        order: list[str] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for child in sorted(self.children(node)):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
            ready.sort()
        if len(order) != len(self.nodes):
            raise ValueError("directed cycle: the causal graph must be acyclic (recursive model)")
        return tuple(order)

    def subgraph(self, keep: Iterable[str]) -> "ADMG":
        keep = frozenset(self._require(keep))
        return ADMG(
            keep,
            frozenset((a, b) for a, b in self.directed if a in keep and b in keep),
            frozenset(pair for pair in self.bidirected if pair <= keep),
        )

    def without_incoming(self, targets: Iterable[str]) -> "ADMG":
        """G_{bar X}: delete directed edges into X and bidirected edges touching X."""
        targets = frozenset(self._require(targets))
        return ADMG(
            self.nodes,
            frozenset((a, b) for a, b in self.directed if b not in targets),
            frozenset(pair for pair in self.bidirected if not pair & targets),
        )

    def without_outgoing(self, sources: Iterable[str]) -> "ADMG":
        """G_{underline Z}: delete directed edges out of Z (bidirected edges stay)."""
        sources = frozenset(self._require(sources))
        return ADMG(self.nodes, frozenset((a, b) for a, b in self.directed if a not in sources), self.bidirected)

    def c_components(self) -> tuple[frozenset[str], ...]:
        """Districts: connected components over bidirected edges, deterministic order."""
        neighbours: dict[str, set[str]] = {node: set() for node in self.nodes}
        for pair in self.bidirected:
            a, b = sorted(pair)
            neighbours[a].add(b)
            neighbours[b].add(a)
        seen: set[str] = set()
        components: list[frozenset[str]] = []
        for node in self.topological_order():
            if node in seen:
                continue
            component, stack = set(), [node]
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(neighbours[current] - component)
            seen |= component
            components.append(frozenset(component))
        return tuple(components)

    def as_latent_dag(self) -> tuple[tuple[tuple[str, str], ...], frozenset[str], frozenset[str]]:
        """Explicit-latent DAG: each a <-> b becomes a fresh latent parent of a and b."""
        edges = list(self.directed)
        latent: set[str] = set()
        for pair in sorted(self.bidirected, key=lambda p: tuple(sorted(p))):
            a, b = sorted(pair)
            hidden = f"__U[{a}<->{b}]"
            latent.add(hidden)
            edges += [(hidden, a), (hidden, b)]
        return tuple(edges), frozenset(latent), frozenset(self.nodes | latent)

    def _require(self, names: Iterable[str]) -> set[str]:
        values = {str(name) for name in names}
        unknown = values - self.nodes
        if unknown:
            raise ValueError(f"unknown nodes in the causal graph: {sorted(unknown)}")
        return values


def d_separated(graph: ADMG, xs: Iterable[str], ys: Iterable[str], zs: Iterable[str] = ()) -> bool:
    """(X ⊥ Y | Z) in an ADMG (bidirected edges read as independent latent common causes)."""
    xs, ys, zs = set(xs), set(ys), set(zs)
    if not xs or not ys:
        return True
    edges, _latent, nodes = graph.as_latent_dag()
    unknown = (xs | ys | zs) - graph.nodes
    if unknown:
        raise ValueError(f"unknown nodes in the causal graph: {sorted(unknown)}")
    if xs & ys:
        return False
    return _dag_d_separated(edges, xs, ys, zs, nodes=nodes)


# ============================================================================ do-calculus


def _disjoint(**sets: Iterable[str]) -> dict[str, frozenset[str]]:
    parsed = {name: frozenset(str(v) for v in values) for name, values in sets.items()}
    names = list(parsed)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if parsed[a] & parsed[b]:
                raise ValueError(f"{a} and {b} must be disjoint")
    if not parsed["y"]:
        raise ValueError("y must be non-empty")
    return parsed


def do_rule1(graph: ADMG, *, y: Iterable[str], x: Iterable[str], z: Iterable[str], w: Iterable[str] = ()) -> FormulaResult:
    """Insertion/deletion of observations: valid iff (Y ⊥ Z | X, W) in G_{bar X}."""
    s = _disjoint(y=y, x=x, z=z, w=w)
    mutilated = graph.without_incoming(s["x"])
    holds = d_separated(mutilated, s["y"], s["z"], s["x"] | s["w"])
    return FormulaResult(holds, F_RULE1, {"graph": "G_bar_X", "test": "(Y ⊥ Z | X, W)"})


def do_rule2(graph: ADMG, *, y: Iterable[str], x: Iterable[str], z: Iterable[str], w: Iterable[str] = ()) -> FormulaResult:
    """Action/observation exchange: valid iff (Y ⊥ Z | X, W) in G_{bar X, underline Z}."""
    s = _disjoint(y=y, x=x, z=z, w=w)
    mutilated = graph.without_incoming(s["x"]).without_outgoing(s["z"])
    holds = d_separated(mutilated, s["y"], s["z"], s["x"] | s["w"])
    return FormulaResult(holds, F_RULE2, {"graph": "G_bar_X_underline_Z", "test": "(Y ⊥ Z | X, W)"})


def do_rule3(graph: ADMG, *, y: Iterable[str], x: Iterable[str], z: Iterable[str], w: Iterable[str] = ()) -> FormulaResult:
    """Insertion/deletion of actions: valid iff (Y ⊥ Z | X, W) in G_{bar X, overline Z(W)}."""
    s = _disjoint(y=y, x=x, z=z, w=w)
    g_bar_x = graph.without_incoming(s["x"])
    z_w = s["z"] - (g_bar_x.ancestors(s["w"]) if s["w"] else frozenset())
    mutilated = g_bar_x.without_incoming(z_w)
    holds = d_separated(mutilated, s["y"], s["z"], s["x"] | s["w"])
    return FormulaResult(
        holds, F_RULE3, {"graph": "G_bar_X_overline_Z(W)", "z_of_w": tuple(sorted(z_w)), "test": "(Y ⊥ Z | X, W)"}
    )


# ============================================================================ expressions


class Expr:
    """Symbolic expression over the observational distribution P(v)."""

    free: frozenset[str]

    def __str__(self) -> str:
        return render_expression(self, self.free)


def _names(values: Iterable[str], rename: Mapping[str, str]) -> str:
    return ",".join(rename.get(v, v) for v in values)


@dataclass(frozen=True)
class Prob(Expr):
    """P(vars | given) of the observational distribution."""

    vars: tuple[str, ...]
    given: tuple[str, ...] = ()
    free: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "vars", tuple(sorted(self.vars)))
        object.__setattr__(self, "given", tuple(sorted(self.given)))
        if not self.vars or set(self.vars) & set(self.given):
            raise ValueError("P(vars | given) needs non-empty, disjoint vars and given")
        object.__setattr__(self, "free", frozenset(self.vars) | frozenset(self.given))


@dataclass(frozen=True)
class Product(Expr):
    factors: tuple[Expr, ...]
    free: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "free", frozenset().union(*(f.free for f in self.factors)) if self.factors else frozenset())


@dataclass(frozen=True)
class Sum(Expr):
    over: tuple[str, ...]
    body: Expr
    free: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "over", tuple(sorted(self.over)))
        object.__setattr__(self, "free", self.body.free - frozenset(self.over))


@dataclass(frozen=True)
class Quotient(Expr):
    numerator: Expr
    denominator: Expr
    free: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "free", self.numerator.free | self.denominator.free)


def _render(expr: Expr, rename: Mapping[str, str]) -> str:
    """Readable rendering; a summation index that shadows an outer free variable gets a prime."""
    if isinstance(expr, Prob):
        head = _names(expr.vars, rename)
        return f"P({head}|{_names(expr.given, rename)})" if expr.given else f"P({head})"
    if isinstance(expr, Product):
        return "".join(
            f"[{_render(f, rename)}]" if isinstance(f, (Sum, Quotient)) and len(expr.factors) > 1 else _render(f, rename)
            for f in expr.factors
        ) or "1"
    if isinstance(expr, Sum):
        inner = dict(rename)
        visible = set(rename.values())
        for var in expr.over:
            name = var
            while name in visible:
                name += "'"
            if var in rename or name != var:
                inner[var] = name
            visible.add(name)
        index = ",".join(inner.get(v, v) for v in expr.over)
        return f"Σ_{{{index}}} {_render(expr.body, inner)}"
    if isinstance(expr, Quotient):
        return f"({_render(expr.numerator, rename)}) / ({_render(expr.denominator, rename)})"
    raise TypeError(f"unknown expression node {type(expr).__name__}")


def render_expression(expr: Expr, free: Iterable[str] = ()) -> str:
    """Render with the query's free variables reserved, so shadowing summations are primed."""
    return _render(expr, {v: v for v in free})


def _sum(over: Iterable[str], body: Expr) -> Expr:
    over = tuple(sorted(set(over)))
    return Sum(over, body) if over else body


def _product(factors: Sequence[Expr]) -> Expr:
    flat: list[Expr] = []
    for factor in factors:
        flat.extend(factor.factors if isinstance(factor, Product) else (factor,))
    return flat[0] if len(flat) == 1 else Product(tuple(flat))


# ============================================================================ current distribution


@dataclass(frozen=True)
class _Dist:
    """Current distribution in the recursion: an expression that is a distribution over ``vars``
    (other free variables are fixed context, e.g. values of x)."""

    expr: Expr
    vars: frozenset[str]

    @property
    def is_joint(self) -> bool:
        return isinstance(self.expr, Prob) and not self.expr.given

    def marginal(self, keep: Iterable[str]) -> Expr:
        keep = frozenset(keep)
        if not keep <= self.vars:
            raise ValueError("cannot marginalise onto variables outside the distribution")
        if keep == self.vars:
            return self.expr
        if self.is_joint:
            if not keep:
                raise ValueError("empty marginal of a joint distribution")
            return Prob(tuple(keep))
        return _sum(self.vars - keep, self.expr)

    def conditional(self, target: str, given: Iterable[str]) -> Expr:
        given = frozenset(given)
        if self.is_joint:
            return Prob((target,), tuple(given))
        numerator = self.marginal(given | {target})
        denominator = self.marginal(given) if given else _sum(self.vars, self.expr)
        return Quotient(numerator, denominator)


# ============================================================================ ID algorithm


@dataclass(frozen=True)
class Hedge:
    """FAIL(F, F') of Shpitser & Pearl (2006): F' ⊂ F are c-forests sharing a root set R ⊆ An(Y),
    F ∩ X ≠ ∅, F' ∩ X = ∅. Its existence proves P_x(y) not identifiable."""

    f_nodes: frozenset[str]
    f_prime_nodes: frozenset[str]
    y: frozenset[str]
    x: frozenset[str]

    def describe(self) -> str:
        return (
            f"hedge F={{{','.join(sorted(self.f_nodes))}}} F'={{{','.join(sorted(self.f_prime_nodes))}}} "
            f"for P_{{{','.join(sorted(self.x))}}}({','.join(sorted(self.y))})"
        )


class NotIdentifiable(Exception):
    def __init__(self, hedge: Hedge) -> None:
        super().__init__(hedge.describe())
        self.hedge = hedge


@dataclass(frozen=True)
class IDResult:
    identified: bool
    y: tuple[str, ...]
    x: tuple[str, ...]
    expression: Expr | None
    rendered: str | None
    hedge: Hedge | None
    trace: tuple[str, ...]
    formula: FormulaReference = F_ID
    allowed_planning_use: str = MAX_PLANNING_USE
    #: variables added to the intervention by line 3: the expression is invariant to their values in
    #: the population (Shpitser & Pearl, Lemma for line 3); the estimator averages over P(w).
    invariant_variables: tuple[str, ...] = ()


def identify(graph: ADMG, *, y: Iterable[str], x: Iterable[str]) -> IDResult:
    """P_x(y) for disjoint non-empty Y and (possibly empty) X, via ID (Shpitser & Pearl 2006, Fig. 3)."""

    ys = frozenset(graph._require(y))
    xs = frozenset(graph._require(x))
    if not ys:
        raise ValueError("y must be non-empty")
    if ys & xs:
        raise ValueError("y and x must be disjoint")
    order = graph.topological_order()
    trace: list[str] = []
    added: set[str] = set()
    base = _Dist(Prob(tuple(graph.nodes)), graph.nodes)
    try:
        expression = _id(ys, xs, base, graph, order, trace, added, depth=0)
    except NotIdentifiable as failure:
        hedge = Hedge(failure.hedge.f_nodes, failure.hedge.f_prime_nodes, ys, xs)
        return IDResult(False, tuple(sorted(ys)), tuple(sorted(xs)), None, None, hedge, tuple(trace),
                        allowed_planning_use="informational_or_human_review")
    invariant = expression.free - ys - xs
    if invariant - added:  # pragma: no cover - guards the implementation, never a user error
        raise AssertionError(f"ID expression has unbound variables {sorted(invariant - added)}")
    return IDResult(True, tuple(sorted(ys)), tuple(sorted(xs)), expression,
                    render_expression(expression, ys | xs | invariant), None, tuple(trace),
                    invariant_variables=tuple(sorted(invariant)))


def _restrict(order: Sequence[str], nodes: frozenset[str]) -> tuple[str, ...]:
    return tuple(v for v in order if v in nodes)


def _id(y: frozenset[str], x: frozenset[str], dist: _Dist, graph: ADMG, order: Sequence[str],
        trace: list[str], added: set[str], *, depth: int) -> Expr:
    pad = "  " * depth
    v = graph.nodes
    # line 1
    if not x:
        trace.append(f"{pad}line1: x=∅ -> Σ_(v\\y) P")
        return dist.marginal(y)
    # line 2
    an_y = graph.ancestors(y)
    if v != an_y:
        trace.append(f"{pad}line2: restrict to An(Y)={sorted(an_y)}")
        return _id(y, x & an_y, _Dist(dist.marginal(an_y), an_y), graph.subgraph(an_y), order, trace, added,
                   depth=depth + 1)
    # line 3
    w = (v - x) - graph.without_incoming(x).ancestors(y)
    if w:
        trace.append(f"{pad}line3: add W={sorted(w)} to the intervention")
        added.update(w)
        return _id(y, x | w, dist, graph, order, trace, added, depth=depth + 1)
    components = graph.subgraph(v - x).c_components()
    # line 4
    if len(components) > 1:
        trace.append(f"{pad}line4: C(G\\X)={[sorted(s) for s in components]}")
        factors = [_id(s, v - s, dist, graph, order, trace, added, depth=depth + 1) for s in components]
        return _sum(v - (y | x), _product(factors))
    (s,) = components
    graph_components = graph.c_components()
    # line 5
    if graph_components == (v,) or (len(graph_components) == 1 and graph_components[0] == v):
        trace.append(f"{pad}line5: FAIL hedge F={sorted(v)} F'={sorted(s)}")
        raise NotIdentifiable(Hedge(v, s, y, x))
    local_order = _restrict(order, v)
    # line 6
    if s in graph_components:
        trace.append(f"{pad}line6: S={sorted(s)} is a c-component of G")
        factors = []
        for node in local_order:
            if node in s:
                predecessors = frozenset(local_order[: local_order.index(node)])
                factors.append(dist.conditional(node, predecessors))
        return _sum(s - y, _product(factors))
    # line 7
    s_prime = next(component for component in graph_components if s < component)
    trace.append(f"{pad}line7: S={sorted(s)} ⊂ S'={sorted(s_prime)}")
    factors = []
    for node in local_order:
        if node in s_prime:
            predecessors = frozenset(local_order[: local_order.index(node)])
            factors.append(dist.conditional(node, predecessors))
    new_dist = _Dist(_product(factors), s_prime)
    return _id(y, x & s_prime, new_dist, graph.subgraph(s_prime), order, trace, added, depth=depth + 1)


# ============================================================================ evaluation


class PositivityError(ValueError):
    """A conditional probability of the ID expression conditions on an empty stratum."""


class DiscreteDistribution:
    """Protocol: domains and (conditional) probabilities of a discrete observational distribution."""

    variables: frozenset[str]

    def domain(self, variable: str) -> tuple[Any, ...]:  # pragma: no cover - protocol
        raise NotImplementedError

    def probability(self, event: Mapping[str, Any], given: Mapping[str, Any]) -> float:  # pragma: no cover
        raise NotImplementedError


class EmpiricalDistribution(DiscreteDistribution):
    """Relative frequencies of i.i.d. discrete observations (one array per variable)."""

    def __init__(self, data: Mapping[str, Sequence[Any]], *, min_cell: int = 1) -> None:
        if not data:
            raise ValueError("data must contain at least one variable")
        arrays = {str(k): np.asarray(v) for k, v in data.items()}
        sizes = {a.shape for a in arrays.values()}
        if len(sizes) != 1 or len(next(iter(sizes))) != 1 or next(iter(sizes))[0] == 0:
            raise ValueError("every variable needs a non-empty 1-D column of equal length")
        if int(min_cell) < 1:
            raise ValueError("min_cell must be >= 1")
        self._data = arrays
        self._n = next(iter(sizes))[0]
        self._min_cell = int(min_cell)
        self.variables = frozenset(arrays)
        self._domains = {k: tuple(sorted(np.unique(a).tolist())) for k, a in arrays.items()}
        self._cache: dict[tuple, int] = {}

    @property
    def sample_size(self) -> int:
        return int(self._n)

    def domain(self, variable: str) -> tuple[Any, ...]:
        return self._domains[variable]

    def _count(self, assignment: Mapping[str, Any]) -> int:
        key = tuple(sorted((k, assignment[k]) for k in assignment))
        if key not in self._cache:
            mask = np.ones(self._n, dtype=bool)
            for name, value in key:
                mask &= self._data[name] == value
            self._cache[key] = int(mask.sum())
        return self._cache[key]

    def probability(self, event: Mapping[str, Any], given: Mapping[str, Any]) -> float:
        denominator = self._count(given) if given else self._n
        if denominator < self._min_cell:
            raise PositivityError(f"no observations with {dict(given)} (positivity violated)")
        return self._count({**given, **event}) / denominator


class TabularDistribution(DiscreteDistribution):
    """Exact joint distribution given as {(v1, v2, ...): p} over an ordered variable tuple."""

    def __init__(self, variables: Sequence[str], table: Mapping[tuple, float], *, atol: float = 1e-9) -> None:
        self._vars = tuple(str(v) for v in variables)
        self.variables = frozenset(self._vars)
        rows = {tuple(k): float(p) for k, p in table.items()}
        if any(len(k) != len(self._vars) for k in rows) or any(p < 0 or not isfinite(p) for p in rows.values()):
            raise ValueError("table rows must match the variables and hold finite non-negative mass")
        if abs(sum(rows.values()) - 1.0) > atol:
            raise ValueError("joint distribution must sum to 1")
        self._rows = rows
        self._domains = {v: tuple(sorted({k[i] for k in rows})) for i, v in enumerate(self._vars)}
        self._cache: dict[tuple, float] = {}

    def domain(self, variable: str) -> tuple[Any, ...]:
        return self._domains[variable]

    def _mass(self, assignment: Mapping[str, Any]) -> float:
        key = tuple(sorted(assignment.items()))
        if key not in self._cache:
            index = {v: i for i, v in enumerate(self._vars)}
            self._cache[key] = sum(
                p for k, p in self._rows.items() if all(k[index[name]] == value for name, value in key)
            )
        return self._cache[key]

    def probability(self, event: Mapping[str, Any], given: Mapping[str, Any]) -> float:
        denominator = self._mass(given) if given else 1.0
        if denominator <= 0.0:
            raise PositivityError(f"P({dict(given)}) = 0 (positivity violated)")
        return self._mass({**given, **event}) / denominator


def evaluate(expr: Expr, assignment: Mapping[str, Any], distribution: DiscreteDistribution) -> float:
    """Value of the expression for the given values of its free variables."""

    missing = expr.free - set(assignment)
    if missing:
        raise ValueError(f"values required for free variables {sorted(missing)}")
    unknown = _variables(expr) - distribution.variables
    if unknown:
        raise ValueError(f"distribution lacks variables {sorted(unknown)}")
    memo: dict[tuple, float] = {}
    return _eval(expr, dict(assignment), distribution, memo)


def _variables(expr: Expr) -> frozenset[str]:
    if isinstance(expr, Prob):
        return expr.free
    if isinstance(expr, Product):
        return frozenset().union(*(_variables(f) for f in expr.factors))
    if isinstance(expr, Sum):
        return _variables(expr.body) | frozenset(expr.over)
    if isinstance(expr, Quotient):
        return _variables(expr.numerator) | _variables(expr.denominator)
    raise TypeError(type(expr).__name__)


def _eval(expr: Expr, env: dict[str, Any], dist: DiscreteDistribution, memo: dict[tuple, float]) -> float:
    key = (id(expr), tuple(sorted((v, env[v]) for v in expr.free)))
    if key in memo:
        return memo[key]
    if isinstance(expr, Prob):
        value = dist.probability({v: env[v] for v in expr.vars}, {v: env[v] for v in expr.given})
    elif isinstance(expr, Product):
        value = 1.0
        # cheap factors first; a zero factor short-circuits (never evaluates terms multiplied by 0)
        for factor in sorted(expr.factors, key=lambda f: 0 if isinstance(f, Prob) else 1):
            value *= _eval(factor, env, dist, memo)
            if value == 0.0:
                break
    elif isinstance(expr, Sum):
        value = 0.0
        domains = [dist.domain(v) for v in expr.over]
        inner = dict(env)
        for values in cartesian(*domains):
            inner.update(zip(expr.over, values))
            value += _eval(expr.body, inner, dist, memo)
    elif isinstance(expr, Quotient):
        numerator = _eval(expr.numerator, env, dist, memo)
        denominator = _eval(expr.denominator, env, dist, memo)
        if denominator <= 0.0:
            raise PositivityError("zero denominator in the ID expression (positivity violated)")
        value = numerator / denominator
    else:
        raise TypeError(type(expr).__name__)
    memo[key] = value
    return value


def interventional_distribution(
    result: IDResult, *, x_values: Mapping[str, Any], distribution: DiscreteDistribution
) -> FormulaResult:
    """P_x(y) for every y in the joint domain of Y, at the given values of X."""

    if not result.identified or result.expression is None:
        raise ValueError("P_x(y) is not identifiable: no numerical value may be produced")
    if set(x_values) != set(result.x):
        raise ValueError(f"values required exactly for X={list(result.x)}")
    expression = result.expression
    if result.invariant_variables:
        # The expression does not depend on these values in the population; averaging over P(w)
        # is population-equivalent and uses every observed stratum instead of an arbitrary one.
        expression = Sum(result.invariant_variables, _product([Prob(result.invariant_variables), expression]))
    table: dict[tuple, float] = {}
    for values in cartesian(*(distribution.domain(v) for v in result.y)):
        table[tuple(values)] = evaluate(expression, {**dict(zip(result.y, values)), **x_values}, distribution)
    total = sum(table.values())
    return FormulaResult(table, F_ID_ESTIMATE, {"y": result.y, "x": dict(x_values), "total_mass": total,
                                               "expression": result.rendered,
                                               "allowed_planning_use": MAX_PLANNING_USE})


def interventional_mean(
    result: IDResult, *, outcome: str, x_values: Mapping[str, Any], distribution: DiscreteDistribution
) -> FormulaResult:
    """E[Y | do(X=x)] = Σ_y y · P_x(y) for a single numerically coded outcome."""

    if result.y != (outcome,):
        raise ValueError("interventional mean needs the ID result for exactly P_x(outcome)")
    table = interventional_distribution(result, x_values=x_values, distribution=distribution)
    mean = sum(float(values[0]) * p for values, p in table.value.items())
    return FormulaResult(mean, F_ID_ESTIMATE, {**dict(table.intermediates), "distribution": dict(table.value)})


__all__ = [
    "ADMG",
    "DiscreteDistribution",
    "EmpiricalDistribution",
    "Expr",
    "F_ID",
    "F_RULE1",
    "F_RULE2",
    "F_RULE3",
    "Hedge",
    "IDResult",
    "NotIdentifiable",
    "PositivityError",
    "Prob",
    "Product",
    "Quotient",
    "Sum",
    "TabularDistribution",
    "d_separated",
    "do_rule1",
    "do_rule2",
    "do_rule3",
    "evaluate",
    "identify",
    "interventional_distribution",
    "interventional_mean",
    "render_expression",
]
