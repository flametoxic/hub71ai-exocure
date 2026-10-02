from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Tuple


StructuralFunction = Callable[[Mapping[str, float], float, Mapping[str, float]], float]


@dataclass(frozen=True)
class StructuralEquation:
    variable: str
    parents: Tuple[str, ...]
    function: StructuralFunction
    parameters: Mapping[str, float]
    version: str


@dataclass(frozen=True)
class AdjustmentSetValidation:
    treatment: str
    outcome: str
    adjustment_set: Tuple[str, ...]
    blocks_backdoor_paths: bool
    contains_treatment_descendant: bool
    contains_latent: bool = False

    @property
    def valid(self) -> bool:
        return self.blocks_backdoor_paths and not self.contains_treatment_descendant and not self.contains_latent


LATENT_VARIABLE_TYPE = "latent"


def _copy_equation(equation: StructuralEquation) -> StructuralEquation:
    """Structural equations carry versioned theta; copies never share the parameter mapping."""

    return StructuralEquation(
        variable=equation.variable,
        parents=tuple(equation.parents),
        function=equation.function,
        parameters={str(key): float(value) for key, value in equation.parameters.items()},
        version=equation.version,
    )


@dataclass
class StructuralVariable:
    name: str
    variable_type: str = "state"
    parents: Set[str] = field(default_factory=set)
    children: Set[str] = field(default_factory=set)


@dataclass
class StructuralCausalModel:
    variables: Dict[str, StructuralVariable] = field(default_factory=dict)
    edge_data: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict)
    assumptions: Dict[str, Any] = field(default_factory=dict)
    equations: Dict[str, StructuralEquation] = field(default_factory=dict)
    model_version: str = "scm:graph:1"

    @classmethod
    def from_causal_graph(cls, graph_payload: Dict[str, Any]) -> "StructuralCausalModel":
        variables: Dict[str, StructuralVariable] = {}
        edge_data: Dict[Tuple[str, str], Dict[str, Any]] = {}
        latent = {str(item).strip() for item in list((graph_payload or {}).get("latent_variables") or []) if str(item).strip()}
        for row in list((graph_payload or {}).get("edges") or []):
            if not isinstance(row, dict):
                continue
            src = str(row.get("cause") or "").strip()
            dst = str(row.get("effect") or "").strip()
            rel = str(row.get("type") or row.get("relation_type") or "").strip().upper()
            if not src or not dst:
                continue
            variables.setdefault(src, StructuralVariable(name=src, variable_type=_infer_variable_type(src)))
            variables.setdefault(dst, StructuralVariable(name=dst, variable_type=_infer_variable_type(dst)))
            variables[src].children.add(dst)
            variables[dst].parents.add(src)
            edge_data[(src, dst)] = {
                "relation_type": rel,
                "strength": float(row.get("strength", 0.5) or 0.5),
                "probability": float(row.get("probability", row.get("strength", 0.5)) or 0.5),
                "lag": float(row.get("lag", row.get("lag_minutes", 0.0)) or 0.0),
            }
        unknown_latent = latent - set(variables)
        if unknown_latent:
            raise ValueError(f"latent variables are not nodes of the causal graph: {sorted(unknown_latent)}")
        for name in latent:
            variables[name].variable_type = LATENT_VARIABLE_TYPE
        return cls(
            variables=variables,
            edge_data=edge_data,
            assumptions={
                "graph_type": "directed_symbolic_causal_graph",
                "edge_semantics": "symbolic_causal_edges",
                "latent_confounders_possible": True,
                "effect_estimation_mode": "heuristic_graph_weighted",
            },
        )

    def has_node(self, node: str) -> bool:
        return str(node) in self.variables

    def has_edge(self, src: str, dst: str) -> bool:
        return (str(src), str(dst)) in self.edge_data

    def edge(self, src: str, dst: str) -> Dict[str, Any]:
        return dict(self.edge_data.get((str(src), str(dst)), {}))

    def parents(self, node: str) -> Set[str]:
        item = self.variables.get(str(node))
        return set(item.parents) if item is not None else set()

    def children(self, node: str) -> Set[str]:
        item = self.variables.get(str(node))
        return set(item.children) if item is not None else set()

    def ancestors(self, node: str) -> Set[str]:
        return self._walk(direction="parents", start={str(node)})

    def descendants(self, node: str) -> Set[str]:
        return self._walk(direction="children", start={str(node)})

    def directed_edges(self) -> Tuple[Tuple[str, str], ...]:
        """All (cause, effect) pairs, whether declared as graph edges or via register_equation."""

        pairs = {(parent, name) for name, variable in self.variables.items() for parent in variable.parents}
        pairs.update(self.edge_data.keys())
        return tuple(sorted(pairs))

    def latent_variables(self) -> Set[str]:
        return {name for name, variable in self.variables.items() if variable.variable_type == LATENT_VARIABLE_TYPE}

    def _require_nodes(self, *nodes: str) -> None:
        unknown = sorted({str(node) for node in nodes if not self.has_node(str(node))})
        if unknown:
            raise ValueError(f"unknown nodes in the causal graph: {unknown}")

    def register_equation(
        self,
        *,
        variable: str,
        parents: Iterable[str],
        function: StructuralFunction,
        parameters: Mapping[str, float],
        version: str,
    ) -> str:
        token = str(variable).strip()
        if not token or not version.strip() or not callable(function):
            raise ValueError("variable, callable function, and version are required")
        parent_tokens = tuple(str(parent).strip() for parent in parents)
        if any(not parent for parent in parent_tokens):
            raise ValueError("equation parents must be non-empty names")
        self.equations[token] = StructuralEquation(
            variable=token,
            parents=parent_tokens,
            function=function,
            parameters={str(key): float(value) for key, value in parameters.items()},
            version=version,
        )
        self.variables.setdefault(token, StructuralVariable(name=token))
        self.variables[token].parents.update(parent_tokens)
        for parent in parent_tokens:
            self.variables.setdefault(parent, StructuralVariable(name=parent))
            self.variables[parent].children.add(token)
        self.model_version = version
        return version

    def evaluate_equation(
        self,
        variable: str,
        *,
        parent_values: Mapping[str, float],
        disturbance: float = 0.0,
    ) -> float:
        token = str(variable)
        if token not in self.equations:
            raise KeyError(f"no structural equation registered for {token}")
        equation = self.equations[token]
        missing = set(equation.parents) - set(parent_values)
        if missing:
            raise ValueError(f"missing parent values: {sorted(missing)}")
        return float(equation.function(parent_values, float(disturbance), equation.parameters))

    def do_intervention(self, treatment: str, value: Optional[float] = None) -> "StructuralCausalModel":
        """SCM_do(X=x) = SCM \\ Parents(X) ∪ {X := x} (UMRM p.6, CEOS p.7).

        Both halves are mandatory: the parents are cut AND X gets the constant equation X := x. A
        call without a value would leave X's old equation f(PA_X) in place, so it is an error.
        """
        treatment_token = str(treatment)
        self._require_nodes(treatment_token)
        if value is None:
            raise ValueError("do(X) requires a value: SCM_do(X=x) replaces X's equation by X := x")
        intervention_value = float(value)
        if intervention_value != intervention_value or intervention_value in (float("inf"), float("-inf")):
            raise ValueError("intervention value must be finite")
        variables: Dict[str, StructuralVariable] = {}
        edge_data: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for name, variable in self.variables.items():
            variables[name] = StructuralVariable(
                name=name,
                variable_type=variable.variable_type,
                parents=set(variable.parents),
                children=set(variable.children),
            )
        for (src, dst), payload in self.edge_data.items():
            if dst == treatment_token:
                continue
            edge_data[(src, dst)] = dict(payload)
        if treatment_token in variables:
            variables[treatment_token].parents.clear()
        for name, variable in variables.items():
            if treatment_token in variable.children and (name, treatment_token) not in edge_data:
                variable.children.discard(treatment_token)
        clone = StructuralCausalModel(
            variables=variables,
            edge_data=edge_data,
            assumptions=deepcopy(dict(self.assumptions or {})),
            equations={name: _copy_equation(equation) for name, equation in self.equations.items()},
            model_version=f"{self.model_version}|do:{treatment_token}",
        )
        clone.assumptions["intervention"] = {"do": treatment_token, "value": intervention_value}
        clone.equations[treatment_token] = StructuralEquation(
            variable=treatment_token,
            parents=(),
            function=lambda _parents, _disturbance, _parameters, fixed=intervention_value: fixed,
            parameters={},
            version=clone.model_version,
        )
        return clone

    def d_separated(self, x: str, y: str, conditioned_on: Optional[Iterable[str]] = None) -> bool:
        return DSeparationAnalyzer(self).is_d_separated(x, y, conditioned_on=conditioned_on)

    def blocks_backdoor_paths(
        self,
        treatment: str,
        outcome: str,
        adjustment_set: Iterable[str],
    ) -> bool:
        """Apply Pearl's backdoor graph criterion instead of trusting a caller flag."""

        treatment_token = str(treatment)
        outcome_token = str(outcome)
        conditioned = {str(value) for value in adjustment_set}
        self._require_nodes(treatment_token, outcome_token, *conditioned)
        if {treatment_token, outcome_token} & conditioned:
            raise ValueError("adjustment set cannot contain the treatment or the outcome")
        if conditioned.intersection(self.descendants(treatment_token)):
            return False
        backdoor_graph = self._without_outgoing_edges(treatment_token)
        return backdoor_graph.d_separated(treatment_token, outcome_token, conditioned_on=conditioned)

    def validate_adjustment_set(
        self,
        treatment: str,
        outcome: str,
        adjustment_set: Iterable[str],
    ) -> AdjustmentSetValidation:
        values = tuple(sorted({str(value) for value in adjustment_set}))
        blocks = self.blocks_backdoor_paths(treatment, outcome, values)
        descendants = self.descendants(str(treatment))
        contains_descendant = bool(set(values).intersection(descendants))
        return AdjustmentSetValidation(
            treatment=str(treatment),
            outcome=str(outcome),
            adjustment_set=values,
            blocks_backdoor_paths=blocks,
            contains_treatment_descendant=contains_descendant,
            contains_latent=bool(set(values) & self.latent_variables()),
        )

    def simple_paths(self, src: str, dst: str, cutoff: int = 4) -> List[List[str]]:
        src_token = str(src)
        dst_token = str(dst)
        if src_token == dst_token:
            return [[src_token]]
        out: List[List[str]] = []

        def dfs(node: str, path: List[str]) -> None:
            if len(path) > cutoff + 1:
                return
            for child in sorted(self.children(node)):
                if child in path:
                    continue
                next_path = path + [child]
                if child == dst_token:
                    out.append(next_path)
                    continue
                dfs(child, next_path)

        if self.has_node(src_token) and self.has_node(dst_token):
            dfs(src_token, [src_token])
        return out

    def _walk(self, *, direction: str, start: Set[str]) -> Set[str]:
        visited: Set[str] = set()
        frontier = {token for token in set(start) if self.has_node(token)}
        while frontier:
            node = frontier.pop()
            neighbors = self.parents(node) if direction == "parents" else self.children(node)
            for neighbor in neighbors:
                if neighbor in visited or neighbor in start:
                    continue
                visited.add(neighbor)
                frontier.add(neighbor)
        return visited

    def _without_outgoing_edges(self, source: str) -> "StructuralCausalModel":
        source_token = str(source)
        variables = {
            name: StructuralVariable(
                name=variable.name,
                variable_type=variable.variable_type,
                parents=set(variable.parents),
                children=set(variable.children),
            )
            for name, variable in self.variables.items()
        }
        # Surgery over the variable adjacency itself: SCMs built via register_equation have no edge_data.
        if source_token in variables:
            for child in tuple(variables[source_token].children):
                variables[child].parents.discard(source_token)
            variables[source_token].children.clear()
        edge_data: Dict[Tuple[str, str], Dict[str, Any]] = {
            (left, right): dict(payload)
            for (left, right), payload in self.edge_data.items()
            if left != source_token
        }
        return StructuralCausalModel(
            variables=variables,
            edge_data=edge_data,
            assumptions=deepcopy(dict(self.assumptions)),
            equations={name: _copy_equation(equation) for name, equation in self.equations.items()},
            model_version=self.model_version,
        )


class DSeparationAnalyzer:
    def __init__(self, scm: StructuralCausalModel) -> None:
        self.scm = scm

    def is_d_separated(self, x: str, y: str, conditioned_on: Optional[Iterable[str]] = None) -> bool:
        x_token = str(x)
        y_token = str(y)
        conditioned = {str(item) for item in list(conditioned_on or [])}
        # Unknown nodes are an error, never a silent "independent" (fail-closed).
        self.scm._require_nodes(x_token, y_token, *conditioned)
        if x_token == y_token or {x_token, y_token} & conditioned:
            raise ValueError("X, Y and the conditioning set must be disjoint")
        ancestor_set = {x_token, y_token} | conditioned
        for token in list({x_token, y_token} | conditioned):
            ancestor_set |= self.scm.ancestors(token)
        undirected = self._moralized_adjacency(ancestor_set)
        for node in conditioned:
            undirected.pop(node, None)
        for neighbors in undirected.values():
            neighbors.difference_update(conditioned)
        return not self._has_undirected_path(undirected, x_token, y_token)

    def _moralized_adjacency(self, nodes: Set[str]) -> Dict[str, Set[str]]:
        graph: Dict[str, Set[str]] = {node: set() for node in nodes if self.scm.has_node(node)}
        for node in list(graph.keys()):
            for child in self.scm.children(node):
                if child in graph:
                    graph[node].add(child)
                    graph[child].add(node)
        for node in list(graph.keys()):
            parents = [parent for parent in self.scm.parents(node) if parent in graph]
            for i, left in enumerate(parents):
                for right in parents[i + 1 :]:
                    graph[left].add(right)
                    graph[right].add(left)
        return graph

    @staticmethod
    def _has_undirected_path(graph: Dict[str, Set[str]], src: str, dst: str) -> bool:
        if src not in graph or dst not in graph:
            return False
        frontier = [src]
        visited = {src}
        while frontier:
            node = frontier.pop()
            for neighbor in graph.get(node, set()):
                if neighbor == dst:
                    return True
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                frontier.append(neighbor)
        return False


class ConfounderDetector:
    def __init__(self, scm: StructuralCausalModel) -> None:
        self.scm = scm

    def detect(self, treatment: str, outcome: str) -> List[str]:
        treatment_token = str(treatment)
        outcome_token = str(outcome)
        shared = self.scm.ancestors(treatment_token) & self.scm.ancestors(outcome_token)
        descendants_of_treatment = self.scm.descendants(treatment_token)
        out = sorted(node for node in shared if node not in descendants_of_treatment and node not in {treatment_token, outcome_token})
        return out


class MediatorAnalysis:
    def __init__(self, scm: StructuralCausalModel) -> None:
        self.scm = scm

    def identify(self, treatment: str, outcome: str) -> List[str]:
        mediators: Set[str] = set()
        for path in self.scm.simple_paths(str(treatment), str(outcome), cutoff=4):
            for node in path[1:-1]:
                mediators.add(node)
        return sorted(mediators)


class BackdoorAdjustment:
    def __init__(self, scm: StructuralCausalModel) -> None:
        self.scm = scm

    def identify_set(self, treatment: str, outcome: str) -> List[str]:
        latent = self.scm.latent_variables()
        return [node for node in ConfounderDetector(self.scm).detect(treatment, outcome) if node not in latent]

    def estimate_effect(self, treatment: str, outcome: str) -> Dict[str, Any]:
        """Совместимость со старым API. Число эффекта только по графу запрещено (CCIA 2.1: «нельзя выдавать
        numerical causal effect только потому, что есть causal graph»): здесь — лишь проверка критерия
        backdoor; оценка — CausalMetaEngine.estimate_effect(..., data=...)."""
        treatment_token = str(treatment)
        outcome_token = str(outcome)
        adjustment_set = self.identify_set(treatment_token, outcome_token)
        validation = self.scm.validate_adjustment_set(treatment_token, outcome_token, adjustment_set)
        return _graph_only_refusal("backdoor_adjustment", graph_identifiable=bool(validation.valid),
                                   adjustment_set=adjustment_set,
                                   blocks_backdoor_paths=validation.blocks_backdoor_paths,
                                   contains_treatment_descendant=validation.contains_treatment_descendant)


def identify_effect(scm: StructuralCausalModel, treatment: str, outcome: str) -> Dict[str, Any]:
    """Graph identification of P(Y|do(X)): identified only when an observed adjustment set satisfies
    the backdoor criterion (CCIA 2.1). Statistical dependence of X and Y is NOT identification."""

    treatment_token = str(treatment)
    outcome_token = str(outcome)
    scm._require_nodes(treatment_token, outcome_token)
    if treatment_token == outcome_token:
        raise ValueError("treatment and outcome must differ")
    # Front-door and instrument criteria come from the single identification pipeline (exact graph
    # criteria); the former edge-strength heuristics (InstrumentalVariableEstimator/FrontdoorAdjustment)
    # were wrong on textbook graphs and were removed.
    from .causal_identification_pipeline import find_frontdoor_set, find_instrument

    backdoor = BackdoorAdjustment(scm)
    edges_all, latent_all = tuple(scm.directed_edges()), set(scm.latent_variables())
    frontdoor_set = find_frontdoor_set(edges_all, treatment_token, outcome_token, latent=latent_all)
    instrument = find_instrument(edges_all, treatment_token, outcome_token, latent=latent_all)
    confounders = ConfounderDetector(scm).detect(treatment_token, outcome_token)
    mediators = MediatorAnalysis(scm).identify(treatment_token, outcome_token)
    instruments = [instrument] if instrument is not None else []
    dsep_without_adjustment = scm.d_separated(treatment_token, outcome_token, conditioned_on=set())
    latent = scm.latent_variables()
    candidates = [
        tuple(backdoor.identify_set(treatment_token, outcome_token)),
        tuple(sorted(parent for parent in scm.parents(treatment_token) if parent not in latent and parent != outcome_token)),
    ]
    valid_set: Optional[Tuple[str, ...]] = None
    for candidate in candidates:
        if outcome_token in candidate:
            continue
        if scm.validate_adjustment_set(treatment_token, outcome_token, candidate).valid:
            valid_set = candidate
            break
    identified = valid_set is not None
    directed_path_exists = outcome_token in scm.descendants(treatment_token)
    return {
        "treatment": treatment_token,
        "outcome": outcome_token,
        "identified": identified,
        "identification_method": "backdoor_adjustment" if identified else None,
        "adjustment_set": list(valid_set) if identified else None,
        "directed_path_exists": directed_path_exists,
        "latent_confounders": sorted(set(confounders) & latent),
        "backdoor_set": backdoor.identify_set(treatment_token, outcome_token),
        "frontdoor_set": list(frontdoor_set) if frontdoor_set is not None else [],
        "confounders": confounders,
        "mediators": mediators,
        "instruments": instruments,
        "d_separated_without_adjustment": bool(dsep_without_adjustment),
        "assumptions": dict(scm.assumptions or {}),
        # Graph identification never grants action authority (CEOS p.9: ActionAuthority != CausalConfidence).
        "allowed_planning_use": "review" if identified and directed_path_exists else "informational_or_human_review",
        "refutation_supported": False,
        "sensitivity_supported": False,
        "unsupported_features": ["refutation", "sensitivity", "collider_explicit_api"],
    }


def _infer_variable_type(token: str) -> str:
    text = str(token or "").strip().lower()
    if not text:
        return "state"
    if any(marker in text for marker in ("toggle", "start", "stop", "enable", "disable", "set_", "mitigate_", "prepare_", "turn_")):
        return "action"
    return "state"

# ---------------------------------------------------------------- совместимость со старым API команды
# Прежние эвристики давали число эффекта из «силы» рёбер графа — это запрещено ТЗ (CCIA 2.1) и было неверно
# на учебных графах. Имена оставлены, чтобы существующий код импортировался и вызывался; критерии теперь точные
# (causal_identification_pipeline), а числа из одного графа не выдаются: identified=False, estimate=0.0 и
# причина. Оценка по данным — только CausalMetaEngine.estimate_effect(..., data=...).
GRAPH_ONLY_REASON = ("graph_only_estimate_refused: a causal graph alone does not give an effect size (CCIA 2.1); "
                     "use CausalMetaEngine.estimate_effect with data")


def _graph_only_refusal(method: str, *, graph_identifiable: bool, **extra: Any) -> Dict[str, Any]:
    return {"method": method, "identified": False, "estimate": 0.0, "graph_identifiable": bool(graph_identifiable),
            "data_backed_estimation": False, "reason": GRAPH_ONLY_REASON, **extra}


class InstrumentalVariableEstimator:
    """Совместимость: инструмент ищется по точному критерию IV; числа из графа нет."""

    def __init__(self, scm: StructuralCausalModel) -> None:
        self.scm = scm

    def identify_instruments(self, treatment: str, outcome: str) -> List[str]:
        from .causal_identification_pipeline import find_instrument

        found = find_instrument(tuple(self.scm.directed_edges()), str(treatment), str(outcome),
                                latent=set(self.scm.latent_variables()))
        return [found] if found is not None else []

    def estimate_effect(self, treatment: str, outcome: str, *,
                        instruments: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        chosen = [str(x) for x in (instruments if instruments is not None
                                   else self.identify_instruments(treatment, outcome))]
        return _graph_only_refusal("instrumental_variable", graph_identifiable=bool(chosen), instruments=chosen)


class FrontdoorAdjustment:
    """Совместимость: front-door множество — по точному критерию; числа из графа нет."""

    def __init__(self, scm: StructuralCausalModel) -> None:
        self.scm = scm

    def identify_set(self, treatment: str, outcome: str) -> List[str]:
        from .causal_identification_pipeline import find_frontdoor_set

        found = find_frontdoor_set(tuple(self.scm.directed_edges()), str(treatment), str(outcome),
                                   latent=set(self.scm.latent_variables()))
        return list(found) if found is not None else []

    def estimate_effect(self, treatment: str, outcome: str) -> Dict[str, Any]:
        found = self.identify_set(treatment, outcome)
        return _graph_only_refusal("frontdoor_adjustment", graph_identifiable=bool(found), frontdoor_set=found)


def estimate_effect(scm: StructuralCausalModel, treatment: str, outcome: str) -> Dict[str, Any]:
    """Совместимость со старым API: прежняя форма ответа, но без числа из графа (best_estimate = 0.0,
    selected_method = "none"). Критерии идентификации — точные."""
    treatment_token, outcome_token = str(treatment), str(outcome)
    methods = {"backdoor": BackdoorAdjustment(scm).estimate_effect(treatment_token, outcome_token),
               "frontdoor": FrontdoorAdjustment(scm).estimate_effect(treatment_token, outcome_token),
               "instrumental_variable": InstrumentalVariableEstimator(scm).estimate_effect(treatment_token,
                                                                                           outcome_token)}
    return {"treatment": treatment_token, "outcome": outcome_token, "best_estimate": 0.0, "selected_method": "none",
            "estimation_mode": "refused_graph_only", "data_backed_estimation": False, "methods": methods,
            "graph_identifiable": any(m["graph_identifiable"] for m in methods.values()),
            "assumptions": dict(scm.assumptions or {}), "refutation_supported": False,
            "sensitivity_supported": False, "reason": GRAPH_ONLY_REASON}
