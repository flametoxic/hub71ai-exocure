"""CCIA §2.3 (p.3): управление изменениями причинного графа через версионированные diff.

edge proposed -> evidence review -> candidate SCM version -> replay/simulation -> approved scoped version
-> active -> deprecated/retracted.

Активный граф нельзя изменить напрямую: любое изменение ребра (в том числе только его
discovery score) — это новый ``CausalGraphChange``, проходящий весь цикл. Изменение, построенное на
устаревшей версии графа, не активируется (оптимистическая блокировка). История изменений и версий
графа не стирается.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .assertions import require_aware
from .causal_id_algorithm import ADMG
from .causal_mechanisms import NON_OBSERVATIONAL_PREFIXES
from .reality_formulas import FormulaReference

F_GRAPH_CHANGE = FormulaReference(
    "CCIA-2.3-GRAPH-CHANGE",
    "EXO-causal-reasoning-memory-planning-integration-addendum",
    3,
    "edge proposed -> evidence review -> candidate SCM -> replay/simulation -> approved scoped -> active -> deprecated/retracted",
)


class GraphChangeState(str, Enum):
    PROPOSED = "proposed"
    EVIDENCE_REVIEW = "evidence_review"
    CANDIDATE_SCM = "candidate_scm"
    REPLAY = "replay"
    APPROVED_SCOPED = "approved_scoped"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETRACTED = "retracted"


_FORWARD = {
    GraphChangeState.PROPOSED: GraphChangeState.EVIDENCE_REVIEW,
    GraphChangeState.EVIDENCE_REVIEW: GraphChangeState.CANDIDATE_SCM,
    GraphChangeState.CANDIDATE_SCM: GraphChangeState.REPLAY,
    GraphChangeState.REPLAY: GraphChangeState.APPROVED_SCOPED,
    GraphChangeState.APPROVED_SCOPED: GraphChangeState.ACTIVE,
}
_FINAL = {GraphChangeState.DEPRECATED, GraphChangeState.RETRACTED}


def _freeze(mapping: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping or {}))


@dataclass(frozen=True)
class EdgeSpec:
    source: str
    target: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.source).strip() or not str(self.target).strip() or self.source == self.target:
            raise ValueError("edge needs distinct non-empty source and target")
        object.__setattr__(self, "attributes", _freeze(self.attributes))

    @property
    def key(self) -> tuple[str, str]:
        return (self.source, self.target)

    def same_as(self, other: "EdgeSpec") -> bool:
        return self.key == other.key and dict(self.attributes) == dict(other.attributes)


@dataclass(frozen=True)
class EdgeDiff:
    added: tuple[EdgeSpec, ...] = ()
    removed: tuple[EdgeSpec, ...] = ()
    modified: tuple[tuple[EdgeSpec, EdgeSpec], ...] = ()

    def __post_init__(self) -> None:
        keys = [e.key for e in self.added] + [e.key for e in self.removed] + [b.key for b, _ in self.modified]
        if not keys:
            raise ValueError("an empty diff is not a change")
        if len(keys) != len(set(keys)):
            raise ValueError("an edge may appear only once in a diff")
        for before, after in self.modified:
            if before.key != after.key:
                raise ValueError("a modification keeps the edge endpoints")
            if before.same_as(after):
                raise ValueError("a modification must change the edge")

    def touched(self) -> frozenset[tuple[str, str]]:
        return frozenset([e.key for e in self.added] + [e.key for e in self.removed] + [b.key for b, _ in self.modified])

    def inverse(self) -> "EdgeDiff":
        return EdgeDiff(added=self.removed, removed=self.added, modified=tuple((a, b) for b, a in self.modified))


@dataclass(frozen=True)
class Transition:
    state: GraphChangeState
    at: datetime
    actor: str
    note: str = ""


@dataclass(frozen=True)
class CausalGraphChange:
    change_id: str
    version: int
    state: GraphChangeState
    base_graph_version: int
    diff: EdgeDiff
    scope: tuple[str, ...]
    proposer: str
    evidence_refs: tuple[str, ...] = ()
    candidate_scm_version: str | None = None
    replay_ref: str | None = None
    replay_passed: bool | None = None
    approver: str | None = None
    approved_scope: tuple[str, ...] = ()
    activated_graph_version: int | None = None
    transitions: tuple[Transition, ...] = ()


@dataclass(frozen=True)
class GraphVersion:
    version: int
    edges: Mapping[tuple[str, str], EdgeSpec]
    change_id: str | None
    recorded_at: datetime | None


class CausalGraphChangeManager:
    """Owner of the active causal graph; the only way to change it is a fully reviewed change."""

    def __init__(self, *, initial_edges: Iterable[EdgeSpec] = (), latent: Iterable[str] = ()) -> None:
        edges = {e.key: e for e in initial_edges}
        self._latent = frozenset(str(v) for v in latent)
        self._check_acyclic(edges)
        self._versions: list[GraphVersion] = [GraphVersion(0, MappingProxyType(dict(edges)), None, None)]
        self._changes: dict[str, list[CausalGraphChange]] = {}
        self._edge_owner: dict[tuple[str, str], str] = {}

    # ------------------------------------------------------------------ read side
    @property
    def graph_version(self) -> int:
        return self._versions[-1].version

    def active_edges(self) -> Mapping[tuple[str, str], EdgeSpec]:
        return MappingProxyType(dict(self._versions[-1].edges))

    def latent_variables(self) -> frozenset[str]:
        """Latent (unmeasured) nodes of the active graph."""
        nodes = {node for key in self._versions[-1].edges for node in key}
        return frozenset(self._latent & nodes)

    def structure(self) -> tuple[tuple[tuple[str, str], ...], frozenset[str]]:
        """(directed edges, latent nodes) of the active graph -- what identification runs on."""
        return tuple(sorted(self._versions[-1].edges)), self.latent_variables()

    def graph_at(self, version: int) -> Mapping[tuple[str, str], EdgeSpec]:
        for item in self._versions:
            if item.version == version:
                return MappingProxyType(dict(item.edges))
        raise KeyError(f"unknown graph version {version}")

    def change(self, change_id: str) -> CausalGraphChange:
        return self._changes[change_id][-1]

    def change_history(self, change_id: str) -> tuple[CausalGraphChange, ...]:
        return tuple(self._changes.get(change_id, ()))

    def graph_versions(self) -> tuple[GraphVersion, ...]:
        return tuple(self._versions)

    # ------------------------------------------------------------------ lifecycle
    def propose(self, *, change_id: str, diff: EdgeDiff, base_graph_version: int, scope: Iterable[str],
                proposer: str, at: datetime) -> CausalGraphChange:
        token = str(change_id).strip()
        if not token or token in self._changes:
            raise ValueError("change_id must be new and non-empty")
        scope = tuple(str(s).strip() for s in scope)
        if not scope or any(not s for s in scope) or not str(proposer).strip():
            raise ValueError("a change needs a scope and a proposer")
        base = self.graph_at(base_graph_version)
        self._check_diff_against(base, diff)
        change = CausalGraphChange(
            change_id=token, version=1, state=GraphChangeState.PROPOSED, base_graph_version=int(base_graph_version),
            diff=diff, scope=scope, proposer=str(proposer),
            transitions=(Transition(GraphChangeState.PROPOSED, require_aware(at, "at"), str(proposer)),),
        )
        self._changes[token] = [change]
        return change

    def review_evidence(self, change_id: str, *, evidence_refs: Iterable[str], reviewer: str,
                        at: datetime) -> CausalGraphChange:
        refs = tuple(str(r).strip() for r in evidence_refs)
        if not refs or any(not r or r.lower().startswith(NON_OBSERVATIONAL_PREFIXES) for r in refs):
            raise ValueError("evidence review needs observational evidence refs (no LLM/simulation refs)")
        return self._advance(change_id, GraphChangeState.EVIDENCE_REVIEW, reviewer, at, evidence_refs=refs)

    def build_candidate_scm(self, change_id: str, *, actor: str, at: datetime) -> CausalGraphChange:
        current = self.change(change_id)
        candidate = self._apply(self.graph_at(current.base_graph_version), current.diff)
        self._check_acyclic(candidate)
        return self._advance(change_id, GraphChangeState.CANDIDATE_SCM, actor, at,
                             candidate_scm_version=f"scm:graph:{current.base_graph_version}+{change_id}")

    def record_replay(self, change_id: str, *, replay_ref: str, passed: bool, actor: str,
                      at: datetime) -> CausalGraphChange:
        if not str(replay_ref).strip() or not isinstance(passed, bool):
            raise ValueError("replay needs a report reference and an explicit pass/fail")
        return self._advance(change_id, GraphChangeState.REPLAY, actor, at, replay_ref=str(replay_ref),
                             replay_passed=passed)

    def approve(self, change_id: str, *, approver: str, approved_scope: Iterable[str], at: datetime) -> CausalGraphChange:
        current = self.change(change_id)
        approver = str(approver).strip()
        scope = tuple(str(s).strip() for s in approved_scope)
        if not approver or approver == current.proposer:
            raise ValueError("approval needs an approver distinct from the proposer (no self-promotion)")
        if current.replay_passed is not True:
            raise ValueError("approval requires a passed replay/simulation")
        if not scope or not set(scope) <= set(current.scope):
            raise ValueError("approved scope must be a non-empty subset of the proposed scope")
        return self._advance(change_id, GraphChangeState.APPROVED_SCOPED, approver, at, approver=approver,
                             approved_scope=scope)

    def activate(self, change_id: str, *, actor: str, at: datetime) -> CausalGraphChange:
        current = self.change(change_id)
        if current.state is not GraphChangeState.APPROVED_SCOPED:
            raise ValueError(f"only an approved scoped change can become active (state: {current.state.value})")
        if current.base_graph_version != self.graph_version:
            raise ValueError("change was built on a stale graph version: propose it again on the current graph")
        new_edges = self._apply(self.active_edges(), current.diff)
        self._check_acyclic(new_edges)
        version = self.graph_version + 1
        moment = require_aware(at, "at")
        self._versions.append(GraphVersion(version, MappingProxyType(new_edges), change_id, moment))
        superseded = {self._edge_owner[key] for key in current.diff.touched() if key in self._edge_owner}
        for key in current.diff.touched():
            if key in new_edges:
                self._edge_owner[key] = change_id
            else:
                self._edge_owner.pop(key, None)
        activated = self._advance(change_id, GraphChangeState.ACTIVE, actor, at, activated_graph_version=version)
        for other in sorted(superseded - {change_id}):
            if self.change(other).state is GraphChangeState.ACTIVE:
                self._set_state(other, GraphChangeState.DEPRECATED, actor, moment, note=f"superseded_by:{change_id}")
        return activated

    def retract(self, change_id: str, *, actor: str, reason: str, at: datetime) -> CausalGraphChange:
        current = self.change(change_id)
        if current.state in _FINAL:
            raise ValueError(f"change is already {current.state.value}")
        if not str(reason).strip():
            raise ValueError("retraction needs a reason")
        moment = require_aware(at, "at")
        if current.state is GraphChangeState.ACTIVE:
            foreign = [key for key in current.diff.touched() if self._edge_owner.get(key) not in (None, change_id)]
            if foreign:
                raise ValueError(f"a later change owns edges {sorted(foreign)}: retract or supersede it first")
            reverted = self._apply(self.active_edges(), current.diff.inverse())
            self._check_acyclic(reverted)
            self._versions.append(GraphVersion(self.graph_version + 1, MappingProxyType(reverted),
                                               f"retract:{change_id}", moment))
            for key in current.diff.touched():
                self._edge_owner.pop(key, None)
        return self._set_state(change_id, GraphChangeState.RETRACTED, actor, moment, note=str(reason))

    # ------------------------------------------------------------------ internals
    def _advance(self, change_id: str, target: GraphChangeState, actor: str, at: datetime, **updates) -> CausalGraphChange:
        current = self.change(change_id)
        if _FORWARD.get(current.state) is not target:
            raise ValueError(f"invalid graph change transition: {current.state.value} -> {target.value}")
        if not str(actor).strip():
            raise ValueError("every transition needs an actor")
        moment = require_aware(at, "at")
        if current.transitions and moment < current.transitions[-1].at:
            raise ValueError("transition time precedes the previous transition")
        updated = replace(current, state=target, version=current.version + 1,
                          transitions=(*current.transitions, Transition(target, moment, str(actor))), **updates)
        self._changes[change_id].append(updated)
        return updated

    def _set_state(self, change_id: str, target: GraphChangeState, actor: str, at: datetime, *, note: str) -> CausalGraphChange:
        current = self.change(change_id)
        updated = replace(current, state=target, version=current.version + 1,
                          transitions=(*current.transitions, Transition(target, at, str(actor), note)))
        self._changes[change_id].append(updated)
        return updated

    @staticmethod
    def _check_diff_against(base: Mapping[tuple[str, str], EdgeSpec], diff: EdgeDiff) -> None:
        for edge in diff.added:
            if edge.key in base:
                raise ValueError(f"edge {edge.key} already exists: propose a modification")
        for edge in diff.removed:
            if edge.key not in base or not base[edge.key].same_as(edge):
                raise ValueError(f"edge {edge.key} to remove does not match the base graph")
        for before, _after in diff.modified:
            if before.key not in base or not base[before.key].same_as(before):
                raise ValueError(f"edge {before.key} to modify does not match the base graph")

    def _apply(self, base: Mapping[tuple[str, str], EdgeSpec], diff: EdgeDiff) -> dict[tuple[str, str], EdgeSpec]:
        self._check_diff_against(base, diff)
        edges = dict(base)
        for edge in diff.removed:
            edges.pop(edge.key)
        for _before, after in diff.modified:
            edges[after.key] = after
        for edge in diff.added:
            edges[edge.key] = edge
        return edges

    def _check_acyclic(self, edges: Mapping[tuple[str, str], EdgeSpec]) -> None:
        ADMG.from_latent_dag(list(edges), self._latent & {n for key in edges for n in key})


__all__ = [
    "CausalGraphChange",
    "CausalGraphChangeManager",
    "EdgeDiff",
    "EdgeSpec",
    "F_GRAPH_CHANGE",
    "GraphChangeState",
    "GraphVersion",
    "Transition",
]
