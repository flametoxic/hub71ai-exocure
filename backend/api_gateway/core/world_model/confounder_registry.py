"""CCIA §2.2 (p.3): ConfounderRegistry — confounders are registry entries, not free-form event JSON.

Each entry: confounder_id, variable, scope, mechanisms affected, measurement quality,
observed/latent status, proxy variables, validity period (+ the treatments/outcomes it
confounds and the evidence that it exists). Identification queries the registry; a causal
result is reported as Effect_estimated +/- Sensitivity_unobserved_confounding.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

from .causal_identification import (
    NEXT_ACTION_ADJUST,
    NEXT_ACTION_REQUEST_OBSERVATION,
    NON_IDENTIFIED_PLANNING_USE,
    IdentificationMethod,
    IdentificationResult,
    IdentificationStatus,
    apply_abstention,
    identify_backdoor_effect,
)
from .causal_identification_pipeline import identify_causal_effect
from .causal_sensitivity import SENSITIVITY_INTERVAL_FORMULA, sensitivity_interval
from .reality_formulas import FormulaReference


CONFOUNDER_REGISTRY_REF = FormulaReference(
    "CCIA-2.2-CONFOUNDER-REGISTRY",
    "EXO-causal-reasoning-memory-planning-integration-addendum",
    3,
    "Effect_reported = Effect_estimated +/- Sensitivity_unobserved_confounding",
)

NON_IDENTIFIED_NEXT_ACTION = NEXT_ACTION_REQUEST_OBSERVATION


def _aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _tokens(values: Iterable[Any], name: str, *, required: bool = False) -> tuple[str, ...]:
    items = tuple(str(item).strip() for item in values)
    if any(not item for item in items):
        raise ValueError(f"{name} contains an empty value")
    if required and not items:
        raise ValueError(f"{name} is required")
    return items


class ConfounderObservability(str, Enum):
    OBSERVED = "observed"
    LATENT = "latent"


class ConfounderKind(str, Enum):
    """Examples across bodies listed in CCIA §2.2."""

    WEATHER = "weather"
    PUBLIC_EVENT = "public_event"
    POLICY_CHANGE = "policy_change"
    MAINTENANCE_MODE = "maintenance_mode"
    OPERATOR_INTERVENTION = "operator_intervention"
    SENSOR_OUTAGE = "sensor_outage"
    TRAFFIC_DEMAND = "traffic_demand"
    OCCUPANCY_SHIFT = "occupancy_shift"
    COMMUNICATION_DEGRADATION = "communication_degradation"
    OTHER = "other"


@dataclass(frozen=True)
class ConfounderEntry:
    confounder_id: str
    variable: str
    kind: ConfounderKind
    scope: tuple[str, ...]
    treatments: tuple[str, ...]
    outcomes: tuple[str, ...]
    mechanisms_affected: tuple[str, ...]
    measurement_quality: float
    observability: ConfounderObservability
    proxy_variables: tuple[str, ...]
    valid_from: datetime
    valid_until: Optional[datetime]
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not str(self.confounder_id).strip() or not str(self.variable).strip():
            raise ValueError("confounder_id and variable are required")
        object.__setattr__(self, "kind", ConfounderKind(self.kind))
        object.__setattr__(self, "observability", ConfounderObservability(self.observability))
        object.__setattr__(self, "scope", _tokens(self.scope, "scope", required=True))
        object.__setattr__(self, "treatments", _tokens(self.treatments, "treatments", required=True))
        object.__setattr__(self, "outcomes", _tokens(self.outcomes, "outcomes", required=True))
        object.__setattr__(self, "mechanisms_affected", _tokens(self.mechanisms_affected, "mechanisms_affected"))
        object.__setattr__(self, "proxy_variables", _tokens(self.proxy_variables, "proxy_variables"))
        object.__setattr__(self, "evidence_refs", _tokens(self.evidence_refs, "evidence_refs", required=True))
        quality = float(self.measurement_quality)
        if not isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("measurement_quality must be finite and in [0, 1]")
        object.__setattr__(self, "measurement_quality", quality)
        start = _aware(self.valid_from, "valid_from")
        end = _aware(self.valid_until, "valid_until") if self.valid_until is not None else None
        if end is not None and end <= start:
            raise ValueError("valid_until must be after valid_from")
        object.__setattr__(self, "valid_from", start)
        object.__setattr__(self, "valid_until", end)
        if self.variable in self.treatments or self.variable in self.outcomes:
            raise ValueError("a confounder cannot be its own treatment or outcome")

    def active_during(self, start: datetime, end: datetime) -> bool:
        start, end = _aware(start, "start"), _aware(end, "end")
        return self.valid_from <= end and (self.valid_until is None or self.valid_until >= start)

    def matches(self, *, treatment: Optional[str], outcome: str, scope: Iterable[str]) -> bool:
        wanted = set(str(item) for item in scope)
        return (
            outcome in self.outcomes
            and (treatment is None or treatment in self.treatments)
            and bool(set(self.scope) & wanted)
        )


@dataclass(frozen=True)
class ConfounderQuery:
    """What identification must account for, answered by the registry."""

    treatment: str
    outcome: str
    scope: tuple[str, ...]
    entries: tuple[ConfounderEntry, ...]

    @property
    def observed_variables(self) -> tuple[str, ...]:
        return tuple(sorted({e.variable for e in self.entries if e.observability is ConfounderObservability.OBSERVED}))

    @property
    def latent_entries(self) -> tuple[ConfounderEntry, ...]:
        return tuple(e for e in self.entries if e.observability is ConfounderObservability.LATENT)

    @property
    def latent_without_proxy(self) -> tuple[str, ...]:
        return tuple(sorted(e.confounder_id for e in self.latent_entries if not e.proxy_variables))

    @property
    def has_unobserved_confounding(self) -> bool:
        return bool(self.latent_entries)

    def missing_from(self, adjustment_set: Iterable[str]) -> tuple[str, ...]:
        """Registered confounders that an adjustment set fails to block (observed var or latent proxies)."""
        adjusted = set(adjustment_set)
        missing = [variable for variable in self.observed_variables if variable not in adjusted]
        for entry in self.latent_entries:
            if not entry.proxy_variables or not set(entry.proxy_variables) <= adjusted:
                missing.append(entry.confounder_id)
        return tuple(sorted(set(missing)))


@dataclass(frozen=True)
class ConfounderVersion:
    """One append-only version of a registry entry (bitemporal: validity + recorded_at)."""

    entry: ConfounderEntry
    version: int
    recorded_at: datetime
    reason: str


class ConfounderRepository(Protocol):
    """Durable storage port of the ConfounderRegistry (append-only versions)."""

    def append(self, version: ConfounderVersion) -> None: ...

    def load(self) -> Sequence[ConfounderVersion]: ...


class InMemoryConfounderRepository:
    """Process-local implementation of the repository port (tests, single-process runs)."""

    def __init__(self) -> None:
        self._rows: list[ConfounderVersion] = []

    def append(self, version: ConfounderVersion) -> None:
        self._rows.append(version)

    def load(self) -> Sequence[ConfounderVersion]:
        return tuple(self._rows)


def _entry_to_payload(entry: ConfounderEntry) -> dict[str, Any]:
    return {
        "confounder_id": entry.confounder_id, "variable": entry.variable, "kind": entry.kind.value,
        "scope": list(entry.scope), "treatments": list(entry.treatments), "outcomes": list(entry.outcomes),
        "mechanisms_affected": list(entry.mechanisms_affected), "measurement_quality": entry.measurement_quality,
        "observability": entry.observability.value, "proxy_variables": list(entry.proxy_variables),
        "valid_from": entry.valid_from.isoformat(),
        "valid_until": entry.valid_until.isoformat() if entry.valid_until else None,
        "evidence_refs": list(entry.evidence_refs),
    }


def _entry_from_payload(payload: Mapping[str, Any]) -> ConfounderEntry:
    values = dict(payload)
    values["valid_from"] = datetime.fromisoformat(values["valid_from"])
    values["valid_until"] = datetime.fromisoformat(values["valid_until"]) if values.get("valid_until") else None
    for key in ("scope", "treatments", "outcomes", "mechanisms_affected", "proxy_variables", "evidence_refs"):
        values[key] = tuple(values.get(key) or ())
    return ConfounderEntry(**values)


class JsonlConfounderRepository:
    """File-backed durable implementation: one JSON line per version, fsync-free append."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def append(self, version: ConfounderVersion) -> None:
        row = {"entry": _entry_to_payload(version.entry), "version": version.version,
               "recorded_at": version.recorded_at.isoformat(), "reason": version.reason}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def load(self) -> Sequence[ConfounderVersion]:
        if not self._path.exists():
            return ()
        rows = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            rows.append(ConfounderVersion(_entry_from_payload(data["entry"]), int(data["version"]),
                                          _aware(datetime.fromisoformat(data["recorded_at"]), "recorded_at"),
                                          str(data["reason"])))
        return tuple(rows)


class ConfounderRegistry:
    """Append-only registry of confounders per treatment / outcome / scope, with evidence.

    Durable through a repository port: a registry built on the same repository sees every entry and
    version ever registered (CCIA §2.2: a registry, not a field in event JSON). Nothing is erased;
    closing validity appends a new version.
    """

    def __init__(self, repository: ConfounderRepository | None = None, *, clock=None) -> None:
        self._repository = repository if repository is not None else InMemoryConfounderRepository()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._versions: dict[str, list[ConfounderVersion]] = {}
        for version in sorted(self._repository.load(), key=lambda v: (v.entry.confounder_id, v.version)):
            history = self._versions.setdefault(version.entry.confounder_id, [])
            if version.version != len(history) + 1:
                raise ValueError(f"corrupt confounder history for {version.entry.confounder_id}")
            history.append(version)

    @property
    def _entries(self) -> dict[str, ConfounderEntry]:
        return {cid: history[-1].entry for cid, history in self._versions.items()}

    def _append(self, entry: ConfounderEntry, reason: str) -> ConfounderEntry:
        history = self._versions.setdefault(entry.confounder_id, [])
        version = ConfounderVersion(entry, len(history) + 1, _aware(self._clock(), "recorded_at"), reason)
        self._repository.append(version)
        history.append(version)
        return entry

    def register(self, entry: ConfounderEntry) -> ConfounderEntry:
        if not isinstance(entry, ConfounderEntry):
            raise TypeError("entry must be a ConfounderEntry")
        if entry.confounder_id in self._versions:
            raise ValueError(f"confounder already registered: {entry.confounder_id}")
        return self._append(entry, "registered")

    def close_validity(self, confounder_id: str, *, valid_until: datetime, evidence_refs: tuple[str, ...]) -> ConfounderEntry:
        """End a confounder's validity period (with evidence); history is not erased."""
        entry = self.get(confounder_id)
        if not evidence_refs:
            raise ValueError("closing a confounder requires evidence")
        closed = replace(
            entry,
            valid_until=_aware(valid_until, "valid_until"),
            evidence_refs=tuple(dict.fromkeys((*entry.evidence_refs, *evidence_refs))),
        )
        return self._append(closed, "validity_closed")

    def get(self, confounder_id: str) -> ConfounderEntry:
        return self._versions[confounder_id][-1].entry

    def history(self, confounder_id: str) -> tuple[ConfounderVersion, ...]:
        return tuple(self._versions.get(confounder_id, ()))

    def query(
        self,
        *,
        treatment: str,
        outcome: str,
        scope: Iterable[str],
        window: tuple[datetime, datetime],
    ) -> ConfounderQuery:
        scope = tuple(str(item) for item in scope)
        if not scope:
            raise ValueError("scope is required")
        start, end = window
        entries = tuple(
            entry for entry in sorted(self._entries.values(), key=lambda e: e.confounder_id)
            if entry.matches(treatment=treatment, outcome=outcome, scope=scope) and entry.active_during(start, end)
        )
        return ConfounderQuery(treatment=treatment, outcome=outcome, scope=scope, entries=entries)

    def active_refs(
        self,
        *,
        outcome: str,
        scope: Iterable[str],
        window: tuple[datetime, datetime],
        treatment: Optional[str] = None,
    ) -> tuple[str, ...]:
        """Confounder ids active during an outcome verification window (CCIA §6 'confounders during window')."""
        scope = tuple(str(item) for item in scope)
        start, end = window
        return tuple(
            entry.confounder_id for entry in sorted(self._entries.values(), key=lambda e: e.confounder_id)
            if entry.matches(treatment=treatment, outcome=outcome, scope=scope) and entry.active_during(start, end)
        )


def identify_with_registry(
    registry: ConfounderRegistry,
    *,
    treatment: str,
    outcome: str,
    scope: Iterable[str],
    window: tuple[datetime, datetime],
    adjustment_set: tuple[str, ...],
    unobserved_confounding_sensitivity: float | None = None,
    **identification_inputs: Any,
) -> IdentificationResult:
    """Backdoor identification gated by the registry (CCIA §2.1-2.2).

    * a registered confounder missing from the adjustment set -> non_identified; ``reasons`` name the
      missing ones, ``required_confounders`` is the minimal sufficient set (registered observed
      variables + proxies of latent ones), ``next_action`` adjusts or requests observation;
    * a latent confounder without proxies -> non_identified (next action: observe / safe experiment);
    * latent confounders adjusted only through proxies -> at most ``partial``;
    otherwise the causal engine's backdoor identification decides. An identified effect is reported
    as Effect_estimated +/- Sensitivity (``reported_effect``) when the sensitivity is supplied.
    """
    query = registry.query(treatment=treatment, outcome=outcome, scope=scope, window=window)
    missing = query.missing_from(adjustment_set)
    if missing:
        required = set(query.observed_variables)
        latent_unmeasurable = []
        for entry in query.latent_entries:
            if entry.proxy_variables:
                required.update(entry.proxy_variables)
            else:
                required.add(entry.variable)
                latent_unmeasurable.append(entry.confounder_id)
        next_action = NON_IDENTIFIED_NEXT_ACTION if latent_unmeasurable else NEXT_ACTION_ADJUST
        return IdentificationResult(
            treatment=treatment,
            outcome=outcome,
            status=IdentificationStatus.NON_IDENTIFIED,
            method=IdentificationMethod.ADJUSTMENT,
            required_confounders=tuple(sorted(required)),
            unobserved_confounder_risk=1.0,
            assumptions=("registry lists every relevant confounder",),
            allowed_planning_use=NON_IDENTIFIED_PLANNING_USE,
            effect=None,
            reasons=tuple(f"registered_confounder_not_blocked:{item}" for item in missing),
            next_action=next_action,
            formula=CONFOUNDER_REGISTRY_REF,
        )
    result = identify_backdoor_effect(
        treatment=treatment,
        outcome=outcome,
        adjustment_set=adjustment_set,
        **identification_inputs,
    )
    if result.status is IdentificationStatus.IDENTIFIED and query.has_unobserved_confounding:
        proxies = sorted({p for entry in query.latent_entries for p in entry.proxy_variables})
        result = replace(
            result,
            status=IdentificationStatus.PARTIAL,
            assumptions=(*result.assumptions, f"latent confounders adjusted only via proxies: {', '.join(proxies)}"),
            reasons=(*result.reasons, "latent_confounders_adjusted_via_proxies_only"),
            allowed_planning_use=NON_IDENTIFIED_PLANNING_USE,
            effect=None,
        )
    if result.status is IdentificationStatus.IDENTIFIED and result.effect is not None:
        if unobserved_confounding_sensitivity is None:
            result = replace(result, reasons=(*result.reasons, "sensitivity_not_reported"))
        else:
            result = replace(result, reported_effect=report_effect(
                estimated_effect=result.effect,
                unobserved_confounding_sensitivity=unobserved_confounding_sensitivity,
                query=query,
            ))
    return result


REGISTRY_ASSUMPTION = "registry lists every relevant confounder"


@dataclass(frozen=True)
class RegistryAugmentedGraph:
    """The causal graph with the registry's confounders for one query made explicit (CCIA §2.1-2.2).

    Every registered confounder C of (treatment, outcome) becomes a node with edges C -> treatment and
    C -> outcome; latent entries are latent nodes (their proxies hang below them: C -> proxy), so no
    identification strategy can treat a latent confounder as adjusted for.
    """

    edges: tuple[tuple[str, str], ...]
    latent: frozenset[str]
    added_edges: tuple[tuple[str, str], ...]
    query: ConfounderQuery


def graph_with_registered_confounders(
    edges: Iterable[tuple[str, str]], latent: Iterable[str], query: ConfounderQuery
) -> RegistryAugmentedGraph:
    base = [(str(a), str(b)) for a, b in edges]
    present = set(base)
    hidden = set(str(v) for v in latent)
    added: list[tuple[str, str]] = []
    for entry in query.entries:
        pairs = [(entry.variable, query.treatment), (entry.variable, query.outcome)]
        if entry.observability is ConfounderObservability.LATENT:
            hidden.add(entry.variable)
            pairs += [(entry.variable, proxy) for proxy in entry.proxy_variables]
        for pair in pairs:
            if pair not in present:
                present.add(pair)
                added.append(pair)
    all_edges = tuple(base + added)
    nodes = {node for pair in all_edges for node in pair}
    return RegistryAugmentedGraph(all_edges, frozenset(hidden & nodes), tuple(added), query)


def identify_on_graph_with_registry(
    registry: ConfounderRegistry | None,
    *,
    edges: Iterable[tuple[str, str]],
    latent: Iterable[str],
    treatment: str,
    outcome: str,
    scope: Iterable[str],
    window: tuple[datetime, datetime] | None,
    unobserved_confounder_risk: float | None,
    uncertainty_threshold: float | None,
    assumptions_valid: bool = True,
    evidence_sufficient: bool = True,
    iv_assumption: Any = None,
) -> IdentificationResult:
    """Runtime identification (backdoor -> frontdoor -> ID -> IV-partial) on the causal graph augmented
    with the ConfounderRegistry, followed by the CCIA 2.4 abstention gate.

    Fail-closed: no registry, no scope or no time window -> CAUSE_NOT_IDENTIFIABLE (the confounders the
    answer would have to account for are unknown). A registry entry that contradicts the graph (cycle)
    -> non_identified. ``uncertainty_threshold`` None -> abstain (policy not configured).
    """
    treatment, outcome = str(treatment), str(outcome)
    scope = tuple(str(item) for item in scope)

    def refuse(reason: str) -> IdentificationResult:
        return IdentificationResult(
            treatment, outcome, IdentificationStatus.NON_IDENTIFIED, None, (), 1.0, (REGISTRY_ASSUMPTION,),
            NON_IDENTIFIED_PLANNING_USE, None, reasons=(reason,), next_action=NEXT_ACTION_REQUEST_OBSERVATION,
            formula=CONFOUNDER_REGISTRY_REF,
        )

    if registry is None:
        return refuse("confounder_registry_not_configured")
    if not scope or window is None:
        return refuse("confounder_registry_query_needs_scope_and_window")
    query = registry.query(treatment=treatment, outcome=outcome, scope=scope, window=window)
    graph = graph_with_registered_confounders(edges, latent, query)
    try:
        result = identify_causal_effect(edges=graph.edges, latent=graph.latent, treatment=treatment, outcome=outcome,
                                        unobserved_confounder_risk=unobserved_confounder_risk,
                                        iv_assumption=iv_assumption)
    except ValueError as exc:
        return refuse(f"registry_graph_conflict:{exc}")
    registered = tuple(entry.confounder_id for entry in query.entries)
    reasons = (*result.reasons, *(f"registry:confounder:{cid}" for cid in registered))
    if result.method is IdentificationMethod.ADJUSTMENT and result.status is IdentificationStatus.IDENTIFIED:
        unblocked = [v for v in query.observed_variables if v not in set(result.required_confounders)]
        if unblocked:  # defensive: the augmented graph forces them into every valid backdoor set
            return refuse("registered_confounder_not_blocked:" + ",".join(sorted(unblocked)))
    result = replace(result, assumptions=tuple(dict.fromkeys((*result.assumptions, REGISTRY_ASSUMPTION))),
                     reasons=tuple(dict.fromkeys(reasons)))
    return apply_abstention(result, uncertainty_threshold=uncertainty_threshold, assumptions_valid=assumptions_valid,
                            evidence_sufficient=evidence_sufficient)


@dataclass(frozen=True)
class ReportedEffect:
    estimated: float
    lower: float
    upper: float
    sensitivity: float
    latent_confounders: tuple[str, ...]
    formula: FormulaReference = SENSITIVITY_INTERVAL_FORMULA


def report_effect(
    *,
    estimated_effect: float,
    unobserved_confounding_sensitivity: float,
    query: ConfounderQuery,
) -> ReportedEffect:
    """Effect_reported = Effect_estimated +/- Sensitivity_unobserved_confounding.

    The sensitivity is a required input (no default). Claiming zero sensitivity while the
    registry holds a latent confounder for this treatment/outcome/scope is rejected.
    """
    sensitivity = float(unobserved_confounding_sensitivity)
    if query.has_unobserved_confounding and sensitivity == 0.0:
        raise ValueError("latent confounders are registered: sensitivity to unobserved confounding cannot be 0")
    lower, upper = sensitivity_interval(
        estimated_effect=estimated_effect,
        unobserved_confounding_sensitivity=sensitivity,
    )
    return ReportedEffect(
        estimated=float(estimated_effect),
        lower=lower,
        upper=upper,
        sensitivity=sensitivity,
        latent_confounders=tuple(entry.confounder_id for entry in query.latent_entries),
    )


__all__ = [
    "CONFOUNDER_REGISTRY_REF",
    "ConfounderRepository",
    "ConfounderVersion",
    "InMemoryConfounderRepository",
    "JsonlConfounderRepository",
    "ConfounderEntry",
    "ConfounderKind",
    "ConfounderObservability",
    "ConfounderQuery",
    "ConfounderRegistry",
    "ReportedEffect",
    "REGISTRY_ASSUMPTION",
    "RegistryAugmentedGraph",
    "graph_with_registered_confounders",
    "identify_on_graph_with_registry",
    "identify_with_registry",
    "report_effect",
]

