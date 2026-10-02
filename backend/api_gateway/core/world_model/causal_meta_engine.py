"""CausalMetaEngine — единственный фасад причинного движка (CEOS стр. 8, §13).

Публичная поверхность: ``explain_why``, ``predict_next_state``, ``simulate`` (+ ``simulate_result``),
``counterfactual``, ``get_causal_graph``, ``propose_observation``; плюс идентификация/оценка эффекта
(CCIA §2) и управляемые входы TMS/реестра. Все внутренние движки (legacy CausalEngine,
CounterfactualEngine, WorldSimulator, CausalDiscovery, SCM, TMS, совет гипотез) — приватные атрибуты.

Каждый ответ — структурированный результат с версией модели, областью, допущениями и
неопределённостью; ни один ответ не является действием (real_action=False).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from math import isfinite, log2
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from uuid import uuid4

import numpy as np

from api_gateway.core.reasoning.active_observation import (
    ObservationCandidate,
    hypothesis_information_gain,
    propose_observations,
)
from api_gateway.core.reasoning.hypothesis_council import (
    STATUS_ACCEPT,
    Hypothesis,
    HypothesisCouncil,
)

from .belief_dynamics import factual_rollout, predictive_distribution
from .causal_context import CausalContext, CausalContextBuilder, _parse_time
from .causal_contracts import (
    CausalMechanism,
    CausalQueryLevel,
    CausalResult,
    MechanismLifecycle,
    MechanismOrigin,
    PlanningUse,
    has_quantitative_uncertainty,
    seal_bounded_result,
)
from .causal_graph_lifecycle import CausalGraphChange, CausalGraphChangeManager, EdgeDiff, EdgeSpec
from .causal_identification import (
    IdentificationResult,
    IdentificationStatus,
    IVAssumption,
    identify_backdoor_effect,
)
from .causal_identification_pipeline import estimate_identified_effect, identify_causal_effect
from .causal_mechanisms import constraint_check
from .causal_registry import CausalMechanismRegistry
from .causal_sensitivity import sensitivity_interval
from .confounder_registry import ConfounderRegistry, identify_on_graph_with_registry
from .causal_simulation import InterventionSimulator, counterfactual_distribution
from .causal_tms import CausalBeliefStatus, CausalTruthMaintenance
from .structural_causal_model import StructuralCausalModel
from .structural_equations import StructuralEquationModel

# CEOS p.8 §12: the only allowed observation actions.
OBSERVATION_KINDS = frozenset({
    "request_existing_sensor_reading",
    "raise_sample_rate_within_policy",
    "query_another_body_or_device",
    "request_operator_inspection",
    "request_maintenance_record",
    "propose_safe_micro_experiment",
})
ENSEMBLE_COMPONENTS = ("symbolic", "world_simulator", "counterfactual", "causal_reasoner", "mental_simulation")
# UMRM p.6 fixes the counterfactual summary levels {mean, p05, p95}, i.e. a central 90% interval.
UMRM_COUNTERFACTUAL_ALPHA = 0.10


@dataclass(frozen=True)
class SimulationPolicy:
    """Monte Carlo policy for P(Y | do(X=x), C_t) (CEOS p.7). CEOS fixes neither N nor α: both are
    required configuration, as are the variables whose distribution is reported as risk.

    ``min_tail_samples`` k: CI = [Q_α/2, Q_1−α/2] is estimable only when each tail holds at least k
    draws, N·α/2 ≥ k (k ≥ 1 is the mathematical floor; the value of k is policy, CEOS fixes none)."""

    samples: int
    seed: int
    alpha: float
    risk_variables: tuple[str, ...]
    source: str
    min_tail_samples: int

    def __post_init__(self) -> None:
        if int(self.samples) <= 0 or not 0.0 < float(self.alpha) < 1.0:
            raise ValueError("simulation policy needs samples > 0 and alpha in (0, 1)")
        if isinstance(self.min_tail_samples, bool) or not isinstance(self.min_tail_samples, int) \
                or self.min_tail_samples < 1:
            raise ValueError("simulation policy needs an integer min_tail_samples >= 1")
        if int(self.samples) * float(self.alpha) / 2.0 < self.min_tail_samples:
            raise ValueError("simulation policy: samples * alpha / 2 must reach min_tail_samples per CI tail")
        if not self.risk_variables or any(not str(v).strip() for v in self.risk_variables):
            raise ValueError("simulation policy must name its risk variables")
        if not str(self.source).strip():
            raise ValueError("simulation policy must name its source")


class CausalMetaEngine:
    """The single CEOS facade. Internal engines are private implementation details."""

    def __init__(
        self,
        *,
        causal_engine=None,
        counterfactual_engine=None,
        world_simulator=None,
        causal_reasoner=None,
        mental_simulation=None,
        causal_discovery=None,
        context_builder: CausalContextBuilder | None = None,
        constraint_engine=None,
        mechanism_registry: CausalMechanismRegistry | None = None,
        truth_maintenance: CausalTruthMaintenance | None = None,
        scm: StructuralCausalModel | None = None,
        structural_equations: StructuralEquationModel | None = None,
        hypothesis_council: HypothesisCouncil | None = None,
        dynamics_model: Any = None,
        clock: Callable[[], datetime] | None = None,
        ensemble_weights: Mapping[str, float] | None = None,
        ensemble_weights_source: str = "",
        simulation_policy: SimulationPolicy | None = None,
        action_constraints: Mapping[str, Mapping[str, Callable[[Any], float]]] | None = None,
        confounder_registry: ConfounderRegistry | None = None,
        graph_change_manager: CausalGraphChangeManager | None = None,
    ) -> None:
        self._causal_engine = causal_engine
        self._counterfactual_engine = counterfactual_engine
        self._world_simulator = world_simulator
        self._causal_reasoner = causal_reasoner
        self._mental_simulation = mental_simulation
        self._causal_discovery = causal_discovery
        self._context_builder = context_builder or CausalContextBuilder()
        self._constraint_engine = constraint_engine
        self._mechanism_registry = mechanism_registry or CausalMechanismRegistry(clock=clock)
        self._truth_maintenance = truth_maintenance or CausalTruthMaintenance()
        self._scm = scm
        self._structural_equations = structural_equations
        self._hypothesis_council = hypothesis_council
        self._dynamics_model = dynamics_model
        self._clock = clock
        self._simulation_policy = simulation_policy
        self._confounder_registry = confounder_registry
        # CCIA §2.3: the causal graph identification runs on is owned by the change manager; the
        # configured SCM (or legacy engine graph) only seeds version 0.
        self._graph_changes = graph_change_manager
        self._scm_graph_seed: frozenset[tuple[str, str]] | None = None
        if graph_change_manager is None and scm is not None:
            self._graph_changes = self._seed_graph_manager(scm)
        self._action_constraints = self._parse_constraints(action_constraints)
        if ensemble_weights is not None:
            weights = {str(k): float(v) for k, v in ensemble_weights.items()}
            if not set(weights) <= set(ENSEMBLE_COMPONENTS) or any(not isfinite(v) or v < 0 for v in weights.values()):
                raise ValueError(f"ensemble weights must be non-negative and name components of {ENSEMBLE_COMPONENTS}")
            if not str(ensemble_weights_source).strip():
                # No TZ fixes these numbers: they are configuration and must name their source.
                raise ValueError("ensemble weights must name their source (policy / configuration id)")
            self._weights: Dict[str, float] | None = weights
        else:
            self._weights = None
        self._weights_source = str(ensemble_weights_source or "")

    # =============================================================== §10 simulate
    async def simulate(
        self,
        *,
        state: Dict[str, Any] | Any | None = None,
        snapshot: Dict[str, Any] | Any | None = None,
        intervention: Optional[Dict[str, Any]] = None,
        scenarios: Optional[List[Dict[str, Any]]] = None,
        actions: Optional[List[Dict[str, Any]]] = None,
        action_seq: Optional[List[Dict[str, Any]]] = None,
        max_steps: int = 20,
        constraints: Mapping[str, Mapping[str, Callable[[Any], float]]] | None = None,
        context_query: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """CEOS p.7: intervention, expected effects, side effects, risk distribution, blast radius,
        constraint margins, uncertainty, counterfactual baseline, model/policy versions -- plus the p.7
        requirements: scenario assumptions, model version, horizon, confidence and validity boundaries."""

        state = snapshot if snapshot is not None else state
        if int(max_steps) < 1:
            raise ValueError("max_steps (horizon) must be positive")
        chosen_actions = [dict(x) for x in list(actions or action_seq or ([intervention] if intervention else []))
                          if isinstance(x, dict)]
        symbolic = await self._run_symbolic_simulation(state=state, actions=chosen_actions, max_steps=max_steps)
        sim_world = await self._run_world_simulator(state=state, actions=chosen_actions, max_steps=max_steps)
        sim_reasoner = await self._run_reasoner_projection(state=state, actions=chosen_actions)
        sim_mental = await self._run_mental_rehearsal(state=state, actions=chosen_actions)
        baseline_cf = await self._run_counterfactual_baseline(state=state, actions=chosen_actions)
        state_payload = self._to_state_dict(state)
        context = self._context_for(context_query, state_payload)

        base = dict(symbolic or {})
        components = {"symbolic": symbolic, "world_simulator": sim_world, "counterfactual": baseline_cf,
                      "causal_reasoner": sim_reasoner, "mental_simulation": sim_mental}
        engines_used = sorted(name for name, payload in components.items() if payload)
        legacy_simulated = bool(symbolic or sim_world or baseline_cf)
        # C_t for P(Y | do(X=x), C_t): observed exogenous roots supplied with the snapshot
        structural_context = {str(k): float(v) for k, v in dict(state_payload.get("structural_context") or {}).items()}
        structural = self._structural_monte_carlo(chosen_actions, context=structural_context or None)
        simulated = legacy_simulated or structural is not None
        unknowns: List[str] = []
        assumptions: List[str] = ["no_unmodelled_intervention_during_horizon"]

        # Ensemble aggregates only with configured, sourced weights -- never invented numbers.
        if self._weights is not None and engines_used:
            base["total_risk"] = self._aggregate_risk(symbolic=symbolic, sim_world=sim_world, baseline_cf=baseline_cf)
            base["confidence"] = self._aggregate_confidence(symbolic=symbolic, sim_world=sim_world,
                                                            sim_reasoner=sim_reasoner, sim_mental=sim_mental)
        else:
            base["total_risk"] = None
            base["confidence"] = None
            if engines_used:
                unknowns.append("ensemble_weights_not_configured")
        base["engines_used"] = engines_used + (["structural_monte_carlo"] if structural is not None else [])
        base["simulation_status"] = "simulated" if simulated else "no_simulation_engine"
        warnings = list(base.get("warnings") or [])
        warnings.extend(list((sim_reasoner or {}).get("warnings") or []))
        warnings.extend(list((sim_mental or {}).get("warnings") or []))
        base["warnings"] = list(dict.fromkeys(str(x) for x in warnings if str(x).strip()))
        base["ensemble"] = {
            "engine": "CausalMetaEngine",
            "components": {
                "symbolic": dict(symbolic or {}),
                "world_simulator": dict(sim_world or {}),
                "counterfactual_baseline": dict(baseline_cf or {}),
                "causal_reasoner": dict(sim_reasoner or {}),
                "mental_simulation": dict(sim_mental or {}),
            },
            "weights_used": self._weights_used(components),
            "weights_source": self._weights_source or None,
        }

        # --- effects, side effects, risk distribution, uncertainty
        if structural is not None:
            expected_effects = dict(structural["expected_effects"])
            side_effects = list(structural["side_effects"])
            risk_distribution = dict(structural["risk_distribution"])
            uncertainty = dict(structural["uncertainty"])
            counterfactual_baseline = dict(structural["baseline"])
            assumptions += list(structural["assumptions"])
        else:
            expected_effects = dict(base.get("aggregated_effects") or base.get("expected_effects") or {})
            # side effects are reported effects, never the warnings list (CEOS p.7)
            side_effects = list(base.get("side_effects") or [])
            component_risks = {name: payload.get("total_risk", payload.get("average_risk"))
                               for name, payload in (("symbolic", symbolic), ("world_simulator", sim_world))
                               if payload}
            risk_distribution = {"status": "not_quantified", "component_risks": component_risks}
            uncertainty = {"quantified": False,
                           "component_confidence": {k: v.get("confidence") for k, v in components.items()
                                                    if v and "confidence" in v}}
            counterfactual_baseline = dict(base.get("counterfactual_baseline") or baseline_cf or {})
            if simulated:
                unknowns.append("uncertainty_not_quantified")
                assumptions.append("legacy_ensemble_components_uncalibrated")
        if not simulated:
            unknowns.append("simulation_unavailable")

        # --- blast radius (restricted to the local causal context when one is built, CEOS §4)
        affected = base.get("affected_entities") or {}
        if isinstance(affected, dict):
            blast_radius = sorted(str(entity) for entity, probability in affected.items() if float(probability) > 0.0)
        else:
            blast_radius = sorted(str(entity) for entity in list(affected or []))
        if not blast_radius:
            blast_radius = sorted({
                str(action.get("target_entity_id") or action.get("entity_id") or "")
                for action in chosen_actions
                if str(action.get("target_entity_id") or action.get("entity_id") or "").strip()
            })
        outside_context: List[str] = []
        if context is not None:
            outside_context = [e for e in blast_radius if e not in set(context.entity_ids)]
            if outside_context:
                unknowns.append("effects_outside_local_context")

        # --- constraint margins through X_valid (CEOS p.6)
        predicted = {var: row["mean"] for var, row in (uncertainty.get("intervals") or {}).items()} or None
        margins, violations, checked = self._check_action_constraints(
            constraints, intervention=chosen_actions, predicted=predicted, state=state_payload)
        if not checked:
            unknowns.append("constraints_not_checked")

        # --- scenarios are actually simulated, never echoed
        scenario_results = [await self._run_scenario(dict(sc), chosen_actions, state, max_steps)
                            for sc in list(scenarios or [])]
        for row in scenario_results:
            assumptions += [f"scenario:{row['scenario_id']}:{a}" for a in row.get("assumptions", [])]

        scm_version = self._structural_version()
        if scm_version is None:
            unknowns.append("scm_unavailable")
        validity = {
            "horizon_steps": int(max_steps),
            "scm_version": scm_version,
            "state_timestamp": state_payload.get("timestamp") or state_payload.get("as_of"),
            "scope_entities": list(context.entity_ids) if context is not None else None,
            "context_as_of": context.as_of.isoformat() if context is not None else None,
            "scenario_ids": [row["scenario_id"] for row in scenario_results],
            "monte_carlo": ({"samples": uncertainty.get("samples"), "alpha": uncertainty.get("alpha"),
                             "seed": uncertainty.get("seed")} if structural is not None else None),
        }
        base.update({
            "intervention": chosen_actions,
            "expected_effects": expected_effects,
            "side_effects": side_effects,
            "risk_distribution": risk_distribution,
            "blast_radius": blast_radius,
            "blast_radius_outside_context": outside_context,
            "constraint_margins": margins,
            "constraint_violations": violations,
            "constraints_checked": checked,
            "uncertainty": uncertainty,
            "counterfactual_baseline": counterfactual_baseline,
            "model_versions": list(state_payload.get("model_versions") or []),
            "policy_versions": list(state_payload.get("policy_versions") or []),
            "scm_version": scm_version,
            "horizon": int(max_steps),
            "assumptions": list(dict.fromkeys(assumptions)),
            "validity_boundaries": validity,
            "scenarios": scenario_results,
            "unknowns": list(dict.fromkeys(unknowns)),
            "epistemic_status": "simulated",
            "execution_status": "not_executed_simulation",
            "real_action": False,
        })
        return base

    async def simulate_result(
        self,
        *,
        snapshot: Dict[str, Any] | Any,
        intervention: Dict[str, Any],
        trace_id: str,
        scope: tuple[str, ...],
        world_snapshot_id: str,
        scm_version: str,
        evidence_refs: tuple[str, ...],
        accepted_mechanism_ids: tuple[str, ...],
        rejected_mechanism_ids: tuple[str, ...] = (),
        hypotheses: tuple[Any, ...] = (),
        assumptions: tuple[str, ...] = (),
        unknowns: tuple[str, ...] = (),
        recommended_next_observations: tuple[str, ...] = (),
        baseline_state: Mapping[str, float] | None = None,
        target_variables: tuple[str, ...] = (),
        blast_threshold: float | None = None,
        at: datetime | None = None,
        body_id: str = "",
        constraints: Mapping[str, Mapping[str, Callable[[Any], float]]] | None = None,
        context_query: Mapping[str, Any] | None = None,
        max_steps: int = 20,
    ) -> CausalResult:
        """Intervention-level CausalResult (CEOS p.7-9). Fail-closed: BOUNDED only with a real,
        quantified (Monte Carlo) simulation, complete provenance incl. body_id, a local causal context,
        passed constraints and registry-verified intervention_validated mechanisms."""

        structural_requested = "variable" in intervention and "value" in intervention
        for engine_version in (self._scm.model_version if self._scm is not None else None,
                               self._structural_equations.version if self._structural_equations is not None else None):
            if structural_requested and engine_version is not None and scm_version != engine_version:
                raise ValueError(f"scm_version {scm_version!r} does not match the engine SCM {engine_version!r}")
        simulation = await self.simulate(state=snapshot, actions=[intervention], max_steps=max_steps,
                                         constraints=constraints, context_query=context_query)
        state_payload = self._to_state_dict(snapshot)
        if structural_requested and self._scm is not None:
            if baseline_state is None or not target_variables or blast_threshold is None:
                raise ValueError("structural intervention requires baseline_state, target_variables and blast_threshold")
            affected = simulation.get("affected_entities")
            point = InterventionSimulator(self._scm).simulate(
                intervention=intervention,
                baseline_state=baseline_state,
                target_variables=tuple(target_variables),
                scope=tuple(scope),
                affected_entities=dict(affected) if isinstance(affected, dict) else {},
                blast_threshold=float(blast_threshold),
                constraint_margins=dict(simulation.get("constraint_margins") or {}),
                side_effects=(),
                model_versions=tuple(str(item) for item in list(state_payload.get("model_versions") or [])),
                policy_versions=tuple(str(item) for item in list(state_payload.get("policy_versions") or [])),
            )
            simulation["structural_intervention"] = asdict(point)
            simulation["abducted_expected_effects"] = dict(point.expected_effects)
            if "structural_monte_carlo" not in simulation["engines_used"]:
                simulation["expected_effects"] = dict(point.expected_effects)
                simulation["counterfactual_baseline"] = dict(point.counterfactual_baseline)
            simulation["engines_used"] = sorted(set(simulation["engines_used"]) | {"structural_scm"})
            simulation["simulation_status"] = "simulated"
        simulated = simulation.get("simulation_status") == "simulated"
        uncertainty = dict(simulation.get("uncertainty") or {})
        result_unknowns = list(unknowns) + list(simulation.get("unknowns") or [])
        result_assumptions = [str(item) for item in assumptions if str(item).strip()]
        result_assumptions += [str(a) for a in simulation.get("assumptions") or []]
        if structural_requested and self._scm is not None:
            result_assumptions += [f"scm:{self._scm.model_version}", "additive_noise_abduction_from_current_state"]
        result_assumptions = list(dict.fromkeys(result_assumptions))
        context = self._context_for(context_query, state_payload)
        if context is not None and not set(scope) <= set(context.entity_ids):
            raise ValueError("result scope must lie inside the local causal context (CEOS §4)")
        accepted, rejected, checks = self._check_mechanisms(
            accepted_mechanism_ids=tuple(accepted_mechanism_ids),
            intervention=intervention,
            scope=tuple(scope),
            scm_version=str(scm_version),
            at=at,
        )
        blockers: List[str] = []
        if not (world_snapshot_id and scm_version and evidence_refs and scope):
            blockers.append("provenance_incomplete")
        if not str(body_id).strip():
            blockers.append("body_id_missing")
        if context is None:
            blockers.append("local_context_missing")
        if not accepted_mechanism_ids or len(accepted) != len(tuple(accepted_mechanism_ids)):
            blockers.append("mechanisms_not_validated")
        if not has_quantitative_uncertainty(uncertainty):
            blockers.append("uncertainty_not_quantified")
        if not simulation.get("constraints_checked"):
            blockers.append("constraints_not_checked")
        if simulation.get("constraint_violations"):
            blockers.append("constraint_violated")
        if not result_assumptions:
            blockers.append("assumptions_missing")
        grant_bounded = False
        if not simulated or simulation.get("constraint_violations"):
            allowed_use = PlanningUse.INFORMATIONAL
        elif not blockers:
            allowed_use = PlanningUse.REVIEW  # BOUNDED only through the sealed registry re-verification below
            grant_bounded = True
        else:
            allowed_use = PlanningUse.REVIEW
        result = CausalResult(
            causal_result_id=f"cr:{uuid4().hex}",
            trace_id=str(trace_id),
            query_level=CausalQueryLevel.INTERVENTION,
            scope=tuple(scope),
            world_snapshot_id=str(world_snapshot_id),
            scm_version=str(scm_version),
            hypotheses=tuple(hypotheses),
            accepted_mechanism_ids=accepted,
            rejected_mechanism_ids=tuple(dict.fromkeys((*rejected_mechanism_ids, *rejected))),
            assumptions=tuple(result_assumptions),
            unknowns=tuple(dict.fromkeys(result_unknowns)),
            simulation=simulation,
            uncertainty=uncertainty,
            recommended_next_observations=tuple(recommended_next_observations),
            allowed_planning_use=allowed_use,
            evidence_refs=tuple(evidence_refs),
            epistemic_status="simulated",
            metadata={"mechanism_checks": checks, "bounded_blockers": blockers},
            body_id=str(body_id),
        )
        if grant_bounded:
            moment = at if at is not None else (self._clock() if self._clock is not None else None)
            result = seal_bounded_result(result, registry=self._mechanism_registry, at=moment)
        return result

    def _check_mechanisms(
        self,
        *,
        accepted_mechanism_ids: tuple[str, ...],
        intervention: Mapping[str, Any],
        scope: tuple[str, ...],
        scm_version: str,
        at: datetime | None,
    ) -> tuple[tuple[str, ...], tuple[str, ...], Dict[str, Dict[str, Any]]]:
        """A mechanism supports a bounded result only if it is planning-eligible now (registry status
        intervention_validated, verified via OutcomeLedger) AND relevant: both endpoints inside the query
        scope, anchored at the intervention, same SCM version."""

        moment = at if at is not None else (self._clock() if self._clock is not None else None)
        variable = str(intervention.get("variable") or "").strip()
        target_entity = str(intervention.get("target_entity_id") or intervention.get("entity_id") or "").strip()
        scope_set = {str(item) for item in scope}
        accepted: List[str] = []
        rejected: List[str] = []
        checks: Dict[str, Dict[str, Any]] = {}
        for edge_id in accepted_mechanism_ids:
            failures: List[str] = []
            lifecycle = None
            eligible = False
            if not self._mechanism_registry.contains(edge_id):
                failures.append("unknown_mechanism")
            else:
                mechanism: CausalMechanism = self._mechanism_registry.current(edge_id)
                lifecycle = mechanism.lifecycle.value
                eligible = self._mechanism_registry.planning_eligible(edge_id, at=moment)
                if not eligible:
                    failures.append("not_planning_eligible")
                if mechanism.source_variable not in scope_set or mechanism.target_variable not in scope_set:
                    failures.append("mechanism_outside_query_scope")
                anchored = (variable and variable == mechanism.source_variable) or (
                    target_entity and target_entity in set(mechanism.scope)
                )
                if not anchored:
                    failures.append("mechanism_not_anchored_at_intervention")
                if mechanism.scm_version != scm_version:
                    failures.append("scm_version_mismatch")
            checks[str(edge_id)] = {"failures": failures, "lifecycle": lifecycle, "planning_eligible": eligible,
                                    "validation": dict(self._mechanism_registry.validation_record(edge_id) or {})}
            (rejected if failures else accepted).append(str(edge_id))
        return tuple(accepted), tuple(rejected), checks

    def _structural_monte_carlo(self, actions: Sequence[Mapping[str, Any]], *, context: Mapping[str, float] | None):
        """P(Y | do(X=x), C_t) by Monte Carlo over structural noise (CEOS p.7), with the no-action
        baseline from the same seed. None when no structural intervention or no SEM/policy."""

        do = {str(a["variable"]): float(a["value"]) for a in actions
              if str(a.get("variable") or "").strip() and "value" in a}
        if not do or self._structural_equations is None or self._simulation_policy is None:
            return None
        policy = self._simulation_policy
        sem = self._structural_equations
        intervened = sem.sample_interventional(do, policy.samples, policy.seed, context=context)
        baseline = sem.sample_interventional({}, policy.samples, policy.seed, context=context)

        def summary(values) -> Dict[str, float]:
            values = np.asarray(values, float)
            return {"mean": float(np.mean(values)),
                    "lower": float(np.quantile(values, policy.alpha / 2.0)),
                    "upper": float(np.quantile(values, 1.0 - policy.alpha / 2.0))}

        risk_vars = [v for v in policy.risk_variables if v in intervened.value]
        if len(risk_vars) != len(policy.risk_variables):
            raise ValueError(f"risk variables {sorted(set(policy.risk_variables) - set(risk_vars))} are not in the SCM")
        intervals = {v: summary(values) for v, values in intervened.value.items()}
        base_means = {v: float(np.mean(values)) for v, values in baseline.value.items()}
        downstream = self._sem_descendants(set(do))
        side_effects = [
            {"variable": v, "mean_shift": intervals[v]["mean"] - base_means[v], "lower": intervals[v]["lower"],
             "upper": intervals[v]["upper"]}
            for v in sorted(downstream - set(policy.risk_variables) - set(do))
        ]
        return {
            "expected_effects": {v: intervals[v]["mean"] - base_means[v] for v in risk_vars},
            "side_effects": side_effects,
            "risk_distribution": {v: {**intervals[v], "alpha": policy.alpha, "samples": policy.samples,
                                      "seed": policy.seed, "source": "structural_monte_carlo"} for v in risk_vars},
            "uncertainty": {"method": "monte_carlo", "alpha": policy.alpha, "samples": int(policy.samples),
                            "min_tail_samples": int(policy.min_tail_samples),
                            "seed": int(policy.seed), "intervals": {v: intervals[v] for v in risk_vars},
                            "formula": intervened.formula.formula_id, "policy_source": policy.source},
            "baseline": {v: base_means[v] for v in risk_vars},
            "assumptions": [f"scm:{sem.version}", "structural_noise_as_calibrated",
                            f"monte_carlo:N={policy.samples},alpha={policy.alpha}"],
        }

    def _sem_descendants(self, roots: set[str]) -> set[str]:
        sem = self._structural_equations
        children: Dict[str, set[str]] = {}
        for variable, equation in getattr(sem, "_eq", {}).items():
            for parent in equation.parents:
                children.setdefault(parent, set()).add(variable)
        out, stack = set(), list(roots)
        while stack:
            for child in children.get(stack.pop(), ()):
                if child not in out:
                    out.add(child)
                    stack.append(child)
        return out

    async def _run_scenario(self, scenario: Dict[str, Any], actions, state, max_steps) -> Dict[str, Any]:
        scenario_id = str(scenario.get("scenario_id") or scenario.get("id") or "").strip()
        if not scenario_id:
            raise ValueError("every scenario needs a scenario_id")
        assumptions = [str(a) for a in list(scenario.get("assumptions") or []) if str(a).strip()]
        row: Dict[str, Any] = {"scenario_id": scenario_id, "assumptions": assumptions}
        context = {str(k): float(v) for k, v in dict(scenario.get("context") or {}).items()}
        try:
            structural = self._structural_monte_carlo(actions, context=context or None)
        except (KeyError, ValueError) as exc:
            row.update({"status": "rejected", "reason": str(exc)})
            return row
        if structural is not None:
            row.update({"status": "simulated", "method": "structural_monte_carlo",
                        "risk_distribution": structural["risk_distribution"],
                        "expected_effects": structural["expected_effects"]})
            return row
        if self._causal_engine is not None and hasattr(self._causal_engine, "simulate"):
            scenario_state = self._to_state_dict(state)
            scenario_state.update(dict(scenario.get("state_overrides") or {}))
            out = await self._run_symbolic_simulation(state=scenario_state, actions=list(actions), max_steps=max_steps)
            row.update({"status": "simulated", "method": "legacy_symbolic", "total_risk": out.get("total_risk"),
                        "expected_effects": dict(out.get("aggregated_effects") or {}),
                        "uncertainty": {"quantified": False}})
            return row
        row.update({"status": "not_simulated", "reason": "no_engine_for_scenario"})
        return row

    # =============================================================== §11 counterfactual
    async def counterfactual(
        self,
        *,
        factual_trace: Mapping[str, Any],
        alternate_action: Mapping[str, Any],
        actual_action: Mapping[str, Any],
        target_variables: tuple[str, ...],
        scope: tuple[str, ...],
        trace_id: str | None = None,
        body_id: str = "",
        world_snapshot_id: str = "",
        evidence_refs: tuple[str, ...] = (),
        noise_std: Mapping[str, float] | None = None,
        exogenous_prior: Mapping[str, tuple[float, float]] | None = None,
        samples: int | None = None,
        seed: int | None = None,
    ) -> CausalResult:
        """CEOS p.7-8: abduction -> action -> prediction on the versioned SCM, U drawn from P(U|evidence)
        (UMRM p.6), summarised as {mean, p05, p95}. Labelled simulated (CEOS test 5) and estimated
        (UMRM p.6), versioned and scoped; post-incident analysis only, never a measured fact."""

        if self._scm is None:
            raise ValueError("counterfactual requires a versioned SCM; rollout comparisons are not counterfactuals")
        if not self._is_structural_action(actual_action) or not self._is_structural_action(alternate_action):
            raise ValueError("counterfactual actions must be structural: {'variable', 'value'}")
        if samples is None or seed is None:
            if self._simulation_policy is None:
                raise ValueError("counterfactual needs samples and seed (or a simulation policy)")
            samples = self._simulation_policy.samples if samples is None else samples
            seed = self._simulation_policy.seed if seed is None else seed
        trace = {str(k): float(v) for k, v in dict(factual_trace).items()}
        result = counterfactual_distribution(
            self._scm, factual_trace=trace, factual_intervention=actual_action,
            alternate_intervention=alternate_action, target_variables=tuple(target_variables), scope=tuple(scope),
            noise_std=dict(noise_std or {}), exogenous_prior=dict(exogenous_prior or {}), samples=int(samples),
            seed=int(seed),
        )
        intervals = {t: {"mean": row["mean"], "lower": row["p05"], "upper": row["p95"]}
                     for t, row in result.distribution.items()}
        simulation = {
            "method": "scm_abduction_action_prediction",
            "counterfactual_process": {
                "abduction": {"factual_trace": trace, "posterior_sampled": list(result.posterior_sampled),
                              "complete": result.abduction_complete},
                "action": dict(result.action),
                "prediction": {t: dict(row) for t, row in result.distribution.items()},
            },
            "distribution": {t: dict(row) for t, row in result.distribution.items()},
            "diff": {t: dict(row) for t, row in result.difference.items()},
            "factual_outcome": dict(result.factual_outcome),
            "intervened_scm_version": result.scm_version,
            "effective_sample_size": result.effective_sample_size,
            "execution_status": "not_executed_simulation",
            "real_action": False,
        }
        return CausalResult(
            causal_result_id=f"cr:{uuid4().hex}",
            trace_id=str(trace_id or f"tr:{uuid4().hex}"),
            query_level=CausalQueryLevel.COUNTERFACTUAL,
            scope=tuple(result.scope),
            world_snapshot_id=str(world_snapshot_id),
            scm_version=self._scm.model_version,
            hypotheses=(),
            accepted_mechanism_ids=(),
            rejected_mechanism_ids=(),
            assumptions=(f"scm:{self._scm.model_version}", "additive_noise", "posterior_abduction_P(U|evidence)",
                         "same_exogenous_world_under_alternate_action"),
            unknowns=tuple(f"posterior_sampled:{v}" for v in result.posterior_sampled),
            simulation=simulation,
            uncertainty={"method": "posterior_monte_carlo", "alpha": UMRM_COUNTERFACTUAL_ALPHA,
                         "samples": int(result.samples), "seed": int(result.seed), "intervals": intervals},
            recommended_next_observations=(),
            # Counterfactuals are post-incident analysis, plan comparison and calibration (CEOS p.8).
            allowed_planning_use=PlanningUse.REVIEW,
            evidence_refs=tuple(evidence_refs),
            epistemic_status="simulated",
            metadata={"estimate_label": "estimated", "measured_fact": False,
                      "label_sources": {"simulated": "CEOS p.10 test 5", "estimated": "UMRM p.6"}},
            body_id=str(body_id),
        )

    @staticmethod
    def _is_structural_action(action: Mapping[str, Any]) -> bool:
        return bool(str(action.get("variable") or "").strip()) and "value" in action

    # =============================================================== §7-8 explain_why
    async def explain_why(
        self,
        *,
        query: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        evidence_cutoff: str | datetime | None = None,
        hypotheses: Iterable[Hypothesis] | None = None,
        causal_uncertainty: float | None = None,
        assumptions_valid: bool = True,
        evidence_sufficient: bool = True,
        hypothesis_states: Mapping[str, Mapping[str, Any]] | None = None,
        constraints: Mapping[str, Mapping[str, Callable[[Any], float]]] | None = None,
        observation_context: Mapping[str, Any] | None = None,
        trace_id: str | None = None,
        world_snapshot_id: str = "",
        body_id: str = "",
    ) -> CausalResult:
        """CEOS p.3-6: local context C_t(q) -> competing hypotheses (all ten fields) -> constraints /
        falsification -> council decision -> best next observation by VOI when ambiguous.

        ``evidence_cutoff`` (default: query time t) removes every evidence ref recorded after it or with
        an unknown time (``snapshot["evidence_times"]``); a hypothesis that lost evidence has its E_k
        zeroed -- its evidence support was computed on refs that are now out of bounds."""

        cutoff_raw = evidence_cutoff if evidence_cutoff is not None else query.get("as_of")
        context = self._build_context(query=dict(query), snapshot=dict(snapshot), as_of=cutoff_raw)
        cutoff = context.as_of
        times = dict(snapshot.get("evidence_times") or {})
        excluded: Dict[str, str] = {}
        prepared: List[Hypothesis] = []
        outside_context: List[str] = []
        local = set(context.entity_ids)
        for hypothesis in list(hypotheses or []):
            if not set(hypothesis.scope) & local:
                outside_context.append(hypothesis.hypothesis_id)
                continue
            kept: List[str] = []
            for ref in hypothesis.evidence_refs:
                raw = times.get(ref)
                try:
                    moment = _parse_time(raw, "evidence_time") if raw is not None else None
                except ValueError:
                    moment = None
                if moment is None:
                    excluded[ref] = "unknown_evidence_time"
                elif moment > cutoff:
                    excluded[ref] = "after_evidence_cutoff"
                else:
                    kept.append(ref)
            if len(kept) != len(hypothesis.evidence_refs):
                hypothesis = replace(hypothesis, evidence_refs=tuple(kept), evidence_support=0.0)
            belief = self._tms_belief(hypothesis.hypothesis_id)
            if belief is not None:
                hypothesis = replace(hypothesis, confidence=belief.confidence)
            prepared.append(hypothesis)

        scope = tuple(context.entity_ids)
        trace = str(trace_id or f"tr:{uuid4().hex}")
        metadata: Dict[str, Any] = {
            "context": {"query_id": context.query_id, "entity_ids": list(context.entity_ids),
                        "as_of": context.as_of.isoformat(), "excluded_events": [dict(e) for e in context.excluded_events]},
            "evidence_cutoff": {"cutoff": cutoff.isoformat(), "excluded_refs": excluded},
            "hypotheses_outside_local_context": outside_context,
        }
        if hypotheses is None:
            metadata["council_status"] = "ABSTAIN"
            return self._why_result(trace, scope, world_snapshot_id, body_id, (), PlanningUse.INFORMATIONAL,
                                    ("competing_hypotheses",), (), {"causal_uncertainty": None}, (), metadata)
        if self._hypothesis_council is None:
            raise ValueError("hypothesis council is not configured")
        if causal_uncertainty is None:
            raise ValueError("causal_uncertainty is required to evaluate hypotheses")

        verdicts: Dict[str, bool] | None = None
        checks: Dict[str, Any] = {}
        if hypothesis_states is not None:
            verdicts = {}
            for hypothesis in prepared:
                values = hypothesis_states.get(hypothesis.hypothesis_id)
                if values is None:
                    continue
                check = self.validate_hypothesis(hypothesis_id=hypothesis.hypothesis_id, variable_values=dict(values),
                                                 inequalities=(constraints or {}).get("inequalities"),
                                                 equalities=(constraints or {}).get("equalities"))
                checks[hypothesis.hypothesis_id] = check
                if check["status"] != "constraint_engine_unavailable":
                    verdicts[hypothesis.hypothesis_id] = bool(check["accepted"])
        # CCIA 2.4: with no evidence left inside the cutoff the evidence is insufficient by definition
        has_evidence = any(h.evidence_refs for h in prepared)
        metadata["evidence_within_cutoff"] = has_evidence
        decision = self._hypothesis_council.evaluate(
            tuple(prepared),
            causal_uncertainty=causal_uncertainty,
            assumptions_valid=assumptions_valid,
            evidence_sufficient=bool(evidence_sufficient) and has_evidence,
            constraint_verdicts=verdicts,
        )
        unresolved = [item.hypothesis.hypothesis_id for item in decision.ranked_hypotheses]
        recommended: tuple[str, ...] = ()
        unknowns = list(decision.missing_information)
        if decision.status != STATUS_ACCEPT:
            if observation_context is not None and len(unresolved) >= 2:
                proposal = await self.propose_observation(
                    uncertainty_context={**dict(observation_context), "unresolved_hypothesis_ids": unresolved})
                metadata["observation_proposal"] = proposal
                recommended = tuple(proposal.get("ranked_observation_ids") or ())
            else:
                unknowns.append("next_observation_not_ranked:voi_context_missing")
                metadata["unranked_required_observations"] = list(decision.recommended_next_observations)
        metadata.update({
            "council_status": decision.status,
            "accepted_hypothesis_id": decision.accepted_hypothesis_id,
            "decision_reasons": list(decision.decision_reasons),
            "elimination_trace": [dict(item) for item in decision.elimination_trace],
            "weights_source": decision.weights_source,
            "constraint_checks": checks,
            "belief_by_status": {hid: dict(b.belief_by_status) for hid in unresolved
                                 if (b := self._tms_belief(hid)) is not None},
        })
        rows = tuple(item.as_contract() for item in (*decision.ranked_hypotheses, *decision.rejected_hypotheses))
        use = PlanningUse.REVIEW if decision.status == STATUS_ACCEPT else PlanningUse.INFORMATIONAL
        evidence = tuple(dict.fromkeys(ref for h in prepared for ref in h.evidence_refs))
        assumptions = tuple(dict.fromkeys(a for h in prepared for a in h.assumptions)) + ("local_causal_context_only",)
        return self._why_result(trace, scope, world_snapshot_id, body_id, rows, use, tuple(unknowns), recommended,
                                {"causal_uncertainty": decision.causal_uncertainty}, evidence, metadata,
                                assumptions=assumptions)

    def _why_result(self, trace, scope, world_snapshot_id, body_id, rows, use, unknowns, recommended, uncertainty,
                    evidence, metadata, *, assumptions=()) -> CausalResult:
        return CausalResult(
            causal_result_id=f"cr:{uuid4().hex}",
            trace_id=trace,
            query_level=CausalQueryLevel.OBSERVATION,
            scope=scope,
            world_snapshot_id=str(world_snapshot_id),
            scm_version=self._structural_version() or "",
            hypotheses=tuple(rows),
            accepted_mechanism_ids=(),
            rejected_mechanism_ids=(),
            assumptions=tuple(assumptions),
            unknowns=tuple(dict.fromkeys(unknowns)),
            simulation={},
            uncertainty=dict(uncertainty),
            recommended_next_observations=tuple(recommended),
            allowed_planning_use=use,
            evidence_refs=tuple(evidence),
            epistemic_status="hypothesis",
            metadata=metadata,
            body_id=str(body_id),
        )

    def _tms_belief(self, belief_id: str):
        try:
            return self._truth_maintenance.get(belief_id)
        except KeyError:
            return None

    # =============================================================== §9 predict_next_state
    async def predict_next_state(
        self,
        current_state: Dict[str, Any] | Any | None = None,
        action: Optional[Dict[str, Any]] = None,
        *,
        snapshot: Dict[str, Any] | Any | None = None,
        actions: Optional[List[Any]] = None,
        horizon: int | None = None,
        exogenous: Optional[Sequence[Mapping[str, float]]] = None,
        samples: int = 0,
        seed: int | None = None,
        alpha: float | None = None,
        scope: tuple[str, ...] = (),
        body_id: str = "",
        trace_id: str | None = None,
        world_snapshot_id: str = "",
    ) -> CausalResult:
        """CEOS p.6-7: factual rollout x_t+k+1 = f(x_t+k, u_t+k, d_t+k) over H steps and, with a
        stochastic dynamics model, the predictive distribution p(x_t:t+H | b_t, u_t:t+H-1).
        Returned as a PREDICTION-level CausalResult (query "predict"), never action authority."""

        current_state = snapshot if snapshot is not None else current_state
        steps = 1 if horizon is None else int(horizon)
        if steps < 1:
            raise ValueError("horizon must be a positive number of steps")
        planned = list(actions or ([action] if action is not None else []))
        if len(planned) > steps:
            raise ValueError("more actions than horizon steps")
        if samples > 0 and alpha is None:
            raise ValueError("alpha (CI level) is required for a predictive distribution: CEOS fixes no value")
        if self._dynamics_model is not None:
            payload = self._dynamics_rollout(snapshot=dict(current_state or {}), actions=planned, horizon=steps,
                                             exogenous=list(exogenous or []), samples=int(samples), seed=seed,
                                             alpha=float(alpha) if alpha is not None else 0.0)
        elif self._causal_engine is None or not hasattr(self._causal_engine, "predict_next_state"):
            payload = {"status": "no_dynamics_model", "trajectory": [], "final_state": None, "horizon": steps,
                       "distribution": None, "model_version": None,
                       "warnings": ["no dynamics model or causal engine configured"]}
        else:
            payload = await self._legacy_rollout(current_state, planned, steps)
        payload["validity_boundaries"] = {"horizon_steps": steps, "model_version": payload.get("model_version"),
                                          "exogenous_path": "as_provided" if exogenous else "not_modelled",
                                          "state_as_of": self._to_state_dict(current_state).get("as_of")}
        payload.update({"execution_status": "not_executed_simulation", "real_action": False})
        distribution = payload.get("distribution")
        uncertainty: Dict[str, Any] = {}
        if distribution:
            final = {key: distribution[key][-1] for key in ("mean", "lower", "upper")}
            uncertainty = {"method": "monte_carlo", "alpha": distribution["alpha"], "samples": distribution["samples"],
                           "seed": distribution["seed"],
                           "intervals": {v: {"mean": final["mean"][v], "lower": final["lower"][v],
                                             "upper": final["upper"][v]} for v in final["mean"]}}
        result_scope = tuple(scope) or tuple(sorted(dict(payload.get("final_state") or {}).keys()))
        if not result_scope:
            result_scope = ("unscoped",)
        return CausalResult(
            causal_result_id=f"cr:{uuid4().hex}",
            trace_id=str(trace_id or f"tr:{uuid4().hex}"),
            query_level=CausalQueryLevel.PREDICTION,
            scope=result_scope,
            world_snapshot_id=str(world_snapshot_id),
            scm_version=str(payload.get("model_version") or ""),
            hypotheses=(),
            accepted_mechanism_ids=(),
            rejected_mechanism_ids=(),
            assumptions=tuple(payload.get("assumptions") or ()),
            unknowns=tuple(payload.get("warnings") or ()),
            simulation=payload,
            uncertainty=uncertainty,
            recommended_next_observations=(),
            allowed_planning_use=PlanningUse.INFORMATIONAL,
            evidence_refs=(),
            epistemic_status="predicted",
            body_id=str(body_id),
        )

    async def _legacy_rollout(self, current_state, planned, steps) -> Dict[str, Any]:
        state = current_state
        trajectory = [self._to_state_dict(current_state)]
        warnings: List[str] = []
        for step in range(steps):
            step_action = planned[step] if step < len(planned) else None
            out = self._causal_engine.predict_next_state(current_state=state, action=step_action)
            if hasattr(out, "__await__"):
                out = await out
            if not (isinstance(out, tuple) and len(out) == 2):
                raise ValueError("causal engine returned an invalid next-state prediction")
            state = dict(out[0] or {})
            warnings.extend(str(item) for item in list(out[1] or []))
            trajectory.append(dict(state))
        warnings.append("uncertainty_not_quantified: legacy engine returns point estimates only")
        return {"status": "predicted", "trajectory": trajectory, "final_state": dict(state), "horizon": steps,
                "model_version": None, "distribution": None,
                "assumptions": ["no new intervention beyond the provided actions"],
                "warnings": list(dict.fromkeys(warnings))}

    def _dynamics_rollout(
        self,
        *,
        snapshot: Dict[str, Any],
        actions: List[Any],
        horizon: int,
        exogenous: List[Mapping[str, float]],
        samples: int,
        seed: int | None,
        alpha: float,
    ) -> Dict[str, Any]:
        model = self._dynamics_model
        state_vars = tuple(getattr(model, "state_variables", ()) or ())
        control_vars = tuple(getattr(model, "control_variables", ()) or ())
        exo_vars = tuple(getattr(model, "exogenous_variables", ()) or ())
        version = str(getattr(model, "model_version", "") or "")
        if not state_vars or not version or not callable(getattr(model, "transition", None)):
            raise ValueError("dynamics model must declare state_variables, model_version and transition()")
        state = dict(snapshot.get("state") or {})
        missing = [name for name in state_vars if name not in state]
        if missing:
            raise ValueError(f"snapshot state lacks {missing}")
        x0 = np.asarray([float(state[name]) for name in state_vars], float)

        def vector(rows: List[Any], names: tuple[str, ...], label: str) -> List[np.ndarray]:
            if not names:
                return [np.zeros(0) for _ in range(horizon)]
            if len(rows) != horizon:
                raise ValueError(f"{label} must be given for every one of the {horizon} steps")
            out = []
            for row in rows:
                row = dict(row or {})
                absent = [name for name in names if name not in row]
                if absent:
                    raise ValueError(f"{label} step lacks {absent}")
                out.append(np.asarray([float(row[name]) for name in names], float))
            return out

        controls = vector(actions, control_vars, "controls")
        disturbances = vector(exogenous, exo_vars, "exogenous")
        rollout = factual_rollout(model.transition, initial_state=x0, controls=controls, exogenous=disturbances)
        trajectory = [dict(zip(state_vars, (float(v) for v in row))) for row in rollout.value]
        distribution = None
        if samples > 0:
            if seed is None:
                raise ValueError("predictive distribution requires an explicit seed (one seed -> one result)")
            sample_transition = getattr(model, "sample_transition", None)
            if not callable(sample_transition):
                raise ValueError("dynamics model has no stochastic sample_transition() for a predictive distribution")
            std_map = dict(snapshot.get("state_std") or {})
            std = np.asarray([float(std_map.get(name, 0.0)) for name in state_vars], float)
            if np.any(std < 0) or not np.all(np.isfinite(std)):
                raise ValueError("state_std must be finite and non-negative")
            predictive = predictive_distribution(
                sample_transition,
                sample_initial_state=lambda rng: x0 + rng.normal(0.0, 1.0, x0.shape) * std,
                controls=controls,
                sample_exogenous=lambda _rng, k: disturbances[k],
                samples=samples,
                seed=int(seed),
                alpha=alpha,
            )
            distribution = {
                key: [dict(zip(state_vars, (float(v) for v in row))) for row in predictive.value[key]]
                for key in ("mean", "lower", "upper")
            }
            distribution.update({"alpha": alpha, "samples": samples, "seed": int(seed),
                                 "formula": predictive.formula.formula_id})
        return {
            "status": "predicted",
            "trajectory": trajectory,
            "final_state": trajectory[-1],
            "horizon": horizon,
            "model_version": version,
            "distribution": distribution,
            "assumptions": ["no new intervention beyond the provided controls", "exogenous path as provided"],
            "formula": rollout.formula.formula_id,
            "warnings": [] if distribution is not None else ["point rollout only: request samples for uncertainty"],
        }

    # =============================================================== §12 propose_observation
    async def propose_observation(self, *, uncertainty_context: Dict[str, Any]) -> Dict[str, Any]:
        """CEOS p.8: VOI(a) = IG(a) - C(a) - lambda_r R(a) - lambda_p PrivacyCost(a).

        Only the six CEOS observation kinds are allowed; "raise sample rate" only within the policy
        maximum; nothing that would activate a sensor. Consent is default-deny: only ids listed in
        allowed_observation_ids are consented. IG is computed from P(o|H) when given and is always
        bounded by log2(#hypotheses); only VOI > 0 is proposed. Never executes anything."""

        context = dict(uncertainty_context or {})
        unresolved = [str(item) for item in list(context.get("unresolved_hypothesis_ids") or [])]
        unresolved = list(dict.fromkeys(unresolved))
        allowed = {str(item) for item in list(context.get("allowed_observation_ids") or [])}
        if len(unresolved) < 2:
            return self._defer("information_gain_requires_competing_hypotheses")
        if "risk_weight" not in context or "privacy_weight" not in context:
            return self._defer("lambda_r_and_lambda_p_must_be_configured")
        risk_weight = float(context["risk_weight"])
        privacy_weight = float(context["privacy_weight"])
        prior = context.get("hypothesis_prior")
        candidates: List[ObservationCandidate] = []
        payloads: Dict[str, Dict[str, Any]] = {}
        rejected: Dict[str, str] = {}
        gains: Dict[str, Dict[str, Any]] = {}
        for raw in list(context.get("candidates") or []):
            if not isinstance(raw, dict):
                continue
            candidate = dict(raw)
            observation_id = str(candidate.get("observation_id") or "").strip()
            if not observation_id:
                continue
            kind_rejection = self._observation_kind_rejection(candidate)
            if kind_rejection:
                rejected[observation_id] = kind_rejection
                continue
            distinguishes = {str(item) for item in list(candidate.get("distinguishes_hypotheses") or [])}
            if not distinguishes.intersection(unresolved):
                rejected[observation_id] = "does_not_discriminate_unresolved_hypotheses"
                continue
            likelihood = candidate.get("observation_likelihood")
            if likelihood is not None:
                if not isinstance(prior, Mapping) or set(prior) != set(unresolved):
                    rejected[observation_id] = "hypothesis_prior_required_over_unresolved_hypotheses"
                    continue
                if not isinstance(likelihood, Mapping) or {str(key) for key in likelihood} != set(unresolved):
                    rejected[observation_id] = "observation_likelihood_required_for_every_unresolved_hypothesis"
                    continue
                computed = hypothesis_information_gain(
                    hypothesis_prior={str(k): float(v) for k, v in prior.items()},
                    observation_likelihood={str(h): dict(rows) for h, rows in dict(likelihood).items()},
                )
                information_gain = float(computed.value)
                gains[observation_id] = {"bits": information_gain, "source": "computed", "formula": computed.formula.formula_id}
            else:
                information_gain = float(candidate.get("information_gain", 0.0) or 0.0)
                gains[observation_id] = {"bits": information_gain, "source": "declared"}
            candidates.append(
                ObservationCandidate(
                    observation_id=observation_id,
                    information_gain_bits=information_gain,
                    cost=float(candidate.get("cost", 0.0) or 0.0),
                    risk=float(candidate.get("risk", 0.0) or 0.0),
                    privacy_cost=float(candidate.get("privacy_cost", 0.0) or 0.0),
                    consent_granted=observation_id in allowed,
                )
            )
            payloads[observation_id] = candidate
        selection = propose_observations(
            candidates,
            n_hypotheses=len(unresolved),
            risk_weight=risk_weight,
            privacy_weight=privacy_weight,
        )
        rejected.update(dict(selection.intermediates.get("rejected") or {}))
        ranked = list(selection.value)
        if not ranked:
            out = self._defer("no_policy_allowed_observation")
            out["rejected"] = rejected
            return out
        observation_id = ranked[0]
        candidate = payloads[observation_id]
        return {
            "status": "proposed",
            "observation_id": observation_id,
            "kind": candidate["kind"],
            "requires_safety_review": candidate["kind"] == "propose_safe_micro_experiment",
            "value_of_information": float(selection.intermediates["voi"][observation_id]),
            "information_gain": gains[observation_id],
            "information_gain_bound_bits": log2(len(unresolved)),
            "candidate": candidate,
            "ranked_observation_ids": ranked,
            "voi": {k: float(v) for k, v in dict(selection.intermediates["voi"]).items()},
            "rejected": rejected,
            "distinguishes_hypotheses": sorted(
                {str(item) for item in list(candidate.get("distinguishes_hypotheses") or [])}.intersection(unresolved)
            ),
            "formula": selection.formula.formula_id,
            "execution_status": "not_executed_simulation",
            "real_action": False,
        }

    @staticmethod
    def _observation_kind_rejection(candidate: Mapping[str, Any]) -> str | None:
        kind = str(candidate.get("kind") or "").strip()
        if kind not in OBSERVATION_KINDS:
            return "observation_kind_not_allowed"
        if candidate.get("activates_sensor"):
            return "would_activate_unauthorized_sensor"
        if kind == "raise_sample_rate_within_policy":
            requested, limit = candidate.get("requested_sample_rate"), candidate.get("policy_max_sample_rate")
            try:
                requested, limit = float(requested), float(limit)
            except (TypeError, ValueError):
                return "sample_rate_policy_missing"
            if not (isfinite(requested) and isfinite(limit)) or requested <= 0 or requested > limit:
                return "sample_rate_outside_policy"
        return None

    @staticmethod
    def _defer(reason: str) -> Dict[str, Any]:
        return {
            "status": "defer",
            "reason": reason,
            "execution_status": "not_executed_simulation",
            "real_action": False,
        }

    # =============================================================== §13 get_causal_graph
    def get_causal_graph(
        self,
        scope: tuple[str, ...] | None = None,
        scm_version: str | None = None,
    ) -> Dict[str, Any]:
        """Graph of the requested SCM version from the SCM and the mechanism registry. An unknown
        version is an error, never echoed back."""

        known = set()
        if self._scm is not None:
            known.add(self._scm.model_version)
        known |= {m.scm_version for m in self._mechanism_registry.mechanisms()}
        version = scm_version if scm_version is not None else (self._scm.model_version if self._scm else None)
        if version is None or version not in known:
            raise ValueError(f"unknown scm_version {scm_version!r}; known: {sorted(known)}")
        edges: Dict[tuple[str, str], Dict[str, Any]] = {}
        latent: set[str] = set()
        graph_version = None
        if self._scm is not None and self._scm.model_version == version:
            active_edges, active_latent = self.causal_graph_structure()
            graph_version = self.graph_version
            latent = set(active_latent)
            for source, target in active_edges:
                edges[(source, target)] = {"source": source, "target": target, "in_scm": True, "mechanisms": []}
        for mechanism in self._mechanism_registry.mechanisms():
            if mechanism.scm_version != version:
                continue
            key = (mechanism.source_variable, mechanism.target_variable)
            row = edges.setdefault(key, {"source": key[0], "target": key[1], "in_scm": False, "mechanisms": []})
            row["mechanisms"].append({
                "edge_id": mechanism.edge_id, "lifecycle": mechanism.lifecycle.value,
                "allowed_use": self._mechanism_registry.allowed_use(mechanism.edge_id),
                "version": mechanism.version, "trust_level": mechanism.trust_level,
            })
        rows = list(edges.values())
        if scope is not None:
            allowed = {str(item) for item in scope}
            rows = [row for row in rows if row["source"] in allowed and row["target"] in allowed]
        return {
            "scm_version": version,
            "edges": sorted(rows, key=lambda r: (r["source"], r["target"])),
            "latent_variables": sorted(latent),
            "total_edges": len(rows),
            "graph_version": graph_version,
            "source": "scm+mechanism_registry",
        }

    def causal_graph_structure(self) -> tuple[tuple[tuple[str, str], ...], frozenset[str]]:
        """(directed edges, latent nodes) of the ACTIVE graph of the CCIA §2.3 change manager -- the graph
        identification runs on. A structural graph mutated outside the change lifecycle (e.g. a direct
        ``register_equation`` on the configured SCM) is refused, never silently used."""
        return self._graph_manager().structure()

    def _seed_graph_manager(self, graph: StructuralCausalModel) -> CausalGraphChangeManager:
        pairs = tuple(graph.directed_edges())
        self._scm_graph_seed = frozenset(pairs)
        return CausalGraphChangeManager(initial_edges=[EdgeSpec(a, b, graph.edge(a, b)) for a, b in pairs],
                                        latent=graph.latent_variables())

    def _graph_manager(self) -> CausalGraphChangeManager:
        if self._graph_changes is None:
            self._graph_changes = self._seed_graph_manager(self._structural_graph())
        if self._scm is not None and self._scm_graph_seed is not None \
                and frozenset(self._scm.directed_edges()) != self._scm_graph_seed:
            raise ValueError("causal graph mutated outside the CCIA 2.3 change lifecycle: "
                             "propose the edge change through the graph change manager")
        return self._graph_changes

    # ---------------------------------------------------------- CCIA §2.3 graph change lifecycle
    @property
    def graph_version(self) -> int:
        return self._graph_manager().graph_version

    def propose_graph_change(self, *, change_id: str, diff: EdgeDiff, base_graph_version: int,
                             scope: Iterable[str], proposer: str, at: datetime) -> CausalGraphChange:
        return self._graph_manager().propose(change_id=change_id, diff=diff, base_graph_version=base_graph_version,
                                             scope=scope, proposer=proposer, at=at)

    def review_graph_change(self, change_id: str, *, evidence_refs: Iterable[str], reviewer: str,
                            at: datetime) -> CausalGraphChange:
        return self._graph_manager().review_evidence(change_id, evidence_refs=evidence_refs, reviewer=reviewer, at=at)

    def build_candidate_graph_scm(self, change_id: str, *, actor: str, at: datetime) -> CausalGraphChange:
        return self._graph_manager().build_candidate_scm(change_id, actor=actor, at=at)

    def record_graph_change_replay(self, change_id: str, *, replay_ref: str, passed: bool, actor: str,
                                   at: datetime) -> CausalGraphChange:
        return self._graph_manager().record_replay(change_id, replay_ref=replay_ref, passed=passed, actor=actor, at=at)

    def approve_graph_change(self, change_id: str, *, approver: str, approved_scope: Iterable[str],
                             at: datetime) -> CausalGraphChange:
        return self._graph_manager().approve(change_id, approver=approver, approved_scope=approved_scope, at=at)

    def activate_graph_change(self, change_id: str, *, actor: str, at: datetime) -> CausalGraphChange:
        return self._graph_manager().activate(change_id, actor=actor, at=at)

    def retract_graph_change(self, change_id: str, *, actor: str, reason: str, at: datetime) -> CausalGraphChange:
        return self._graph_manager().retract(change_id, actor=actor, reason=reason, at=at)

    def graph_change_history(self, change_id: str) -> tuple[CausalGraphChange, ...]:
        return self._graph_manager().change_history(change_id)

    def export_edge_pairs(self) -> set[tuple[str, str]]:
        edges, _latent = self.causal_graph_structure()
        return set(edges)

    def _structural_graph(self) -> StructuralCausalModel:
        if self._scm is not None:
            return self._scm
        payload = {}
        if self._causal_engine is not None and hasattr(self._causal_engine, "get_causal_graph"):
            payload = dict(self._causal_engine.get_causal_graph() or {})
        return StructuralCausalModel.from_causal_graph(payload)

    def _structural_version(self) -> str | None:
        if self._scm is not None:
            return self._scm.model_version
        if self._structural_equations is not None:
            return self._structural_equations.version
        return None

    # =============================================================== §4 context
    def build_context(self, *, query: Dict[str, Any], snapshot: Dict[str, Any]) -> CausalContext:
        return self._build_context(query=query, snapshot=snapshot, as_of=None)

    def _build_context(self, *, query: Mapping[str, Any], snapshot: Mapping[str, Any], as_of) -> CausalContext:
        return self._context_builder.build(
            as_of=as_of if as_of is not None else (query.get("as_of") or snapshot.get("as_of")),
            query=query,
            physical_state=dict(snapshot.get("physical_state") or {}),
            topology_edges=tuple(snapshot.get("topology_edges") or ()),
            spatial_neighbors=dict(snapshot.get("spatial_neighbors") or {}),
            recent_events=tuple(snapshot.get("recent_events") or ()),
            active_policies=tuple(snapshot.get("active_policies") or ()),
            maintenance_modes=dict(snapshot.get("maintenance_modes") or {}),
            sensor_coverage=dict(snapshot.get("sensor_coverage") or {}),
            uncertainty=dict(snapshot.get("uncertainty") or {}),
            exogenous_context=dict(snapshot.get("exogenous_context") or {}),
        )

    def _context_for(self, context_query: Mapping[str, Any] | None, state_payload: Mapping[str, Any]):
        if context_query is None:
            return None
        return self._build_context(query=dict(context_query), snapshot=dict(state_payload), as_of=None)

    # =============================================================== §8 constraints / TMS
    def validate_hypothesis(
        self,
        *,
        hypothesis_id: str,
        variable_values: Dict[str, Any],
        domain: str | None = None,
        inequalities: Mapping[str, Callable[[Any], float]] | None = None,
        equalities: Mapping[str, Callable[[Any], float]] | None = None,
    ) -> Dict[str, Any]:
        """CEOS p.6: X_valid = {x : g_i(x) <= 0, h_j(x) = 0}. A hypothesis is accepted only when every
        configured constraint source passes; with no constraint source it is never accepted."""

        has_engine = self._constraint_engine is not None and hasattr(self._constraint_engine, "check_all")
        has_set = bool(inequalities) or bool(equalities)
        if not has_engine and not has_set:
            return {
                "hypothesis_id": str(hypothesis_id),
                "accepted": False,
                "status": "constraint_engine_unavailable",
                "violations": [],
            }
        violations: List[Dict[str, Any]] = []
        margins: Dict[str, float] = {}
        if has_engine:
            for row in list(self._constraint_engine.check_all(dict(variable_values), domain=domain) or []):
                violations.append(
                    {
                        "constraint_id": str(getattr(row, "constraint_id", "")),
                        "severity": float(getattr(row, "severity", 0.0) or 0.0),
                        "suggestion": str(getattr(row, "suggestion", "")),
                    }
                )
        if has_set:
            checked = constraint_check(dict(variable_values), inequalities=dict(inequalities or {}), equalities=dict(equalities or {}))
            margins = dict(checked.intermediates["margins"])
            violations.extend(
                {"constraint_id": name, "severity": 1.0, "suggestion": "outside X_valid"}
                for name in checked.intermediates["violations"]
            )
        return {
            "hypothesis_id": str(hypothesis_id),
            "accepted": not violations,
            "status": "accepted" if not violations else "rejected_physically_or_operationally_impossible",
            "violations": violations,
            "constraint_margins": margins,
        }

    @staticmethod
    def _parse_constraints(constraints) -> Dict[str, Dict[str, Callable[[Any], float]]] | None:
        if constraints is None:
            return None
        inequalities = dict((constraints or {}).get("inequalities") or {})
        equalities = dict((constraints or {}).get("equalities") or {})
        if not inequalities and not equalities:
            return None
        return {"inequalities": inequalities, "equalities": equalities}

    def _check_action_constraints(self, constraints, *, intervention, predicted, state):
        """X_valid over {intervention, predicted, state}: margins -g_i / -|h_j|; a constraint that cannot
        be evaluated (e.g. needs a prediction that does not exist) is a violation, not a pass."""

        parsed = self._parse_constraints(constraints) or self._action_constraints
        if parsed is None:
            return {}, [], False
        subject = {"intervention": list(intervention), "predicted": predicted, "state": dict(state)}
        margins: Dict[str, float] = {}
        violations: List[str] = []
        safe_ineq: Dict[str, Callable[[Any], float]] = {}
        safe_eq: Dict[str, Callable[[Any], float]] = {}
        for bucket, target in (("inequalities", safe_ineq), ("equalities", safe_eq)):
            for name, fn in parsed[bucket].items():
                try:
                    float(fn(subject))
                except (KeyError, TypeError, ValueError, AttributeError, IndexError):
                    violations.append(f"unevaluable:{name}")
                    continue
                target[name] = fn
        checked = constraint_check(subject, inequalities=safe_ineq, equalities=safe_eq)
        margins.update({k: float(v) for k, v in checked.intermediates["margins"].items()})
        violations.extend(checked.intermediates["violations"])
        return margins, violations, True

    def assert_causal_evidence(self, evidence_id: str, *, confidence: float,
                               status: CausalBeliefStatus = CausalBeliefStatus.OBSERVED, **kwargs):
        return self._truth_maintenance.assert_evidence(evidence_id, confidence=confidence, status=status, **kwargs)

    def assert_causal_hypothesis(self, hypothesis_id: str, **belief_input):
        return self._truth_maintenance.assert_hypothesis(hypothesis_id, **belief_input)

    def retract_causal_evidence(self, evidence_id: str, **kwargs) -> tuple[str, ...]:
        return self._truth_maintenance.retract(evidence_id, **kwargs)

    def mark_causal_evidence_stale(self, evidence_id: str, **kwargs) -> tuple[str, ...]:
        return self._truth_maintenance.mark_stale(evidence_id, **kwargs)

    def fail_causal_evidence_integrity(self, evidence_id: str, *, reason: str, **kwargs) -> tuple[str, ...]:
        return self._truth_maintenance.fail_integrity(evidence_id, reason=reason, **kwargs)

    def replay_beliefs(self, as_of: datetime, **kwargs):
        return self._truth_maintenance.replay(as_of, **kwargs)

    # =============================================================== §6 discovery
    def apply_discovered_links(
        self,
        links: List[Dict[str, Any]],
        *,
        topology: Iterable[tuple[str, str]],
        spatial: Iterable[tuple[str, str]],
        temporal: Iterable[tuple[str, str]],
        ontology: Iterable[tuple[str, str]],
        constraints: Iterable[tuple[str, str]],
        scm_version: str,
        valid_from: datetime,
    ) -> Dict[str, Any]:
        """Discovered links become observational_candidate mechanisms only inside
        H_causal = G_topology ∩ G_spatial ∩ G_temporal ∩ G_ontology ∩ G_constraints (CEOS p.5).
        Nothing is written to any internal engine; promotion only through the registry ladder."""

        layers = {name: tuple(tuple(pair) for pair in values) for name, values in
                  (("topology", topology), ("spatial", spatial), ("temporal", temporal), ("ontology", ontology),
                   ("constraints", constraints))}
        registered: List[str] = []
        rejected: Dict[str, str] = {}
        for index, raw in enumerate(list(links or [])):
            link = dict(raw or {})
            source = str(link.get("cause") or link.get("source") or "").strip()
            target = str(link.get("effect") or link.get("target") or "").strip()
            edge_id = str(link.get("edge_id") or f"discovered:{source}->{target}").strip()
            try:
                mechanism = CausalMechanism(
                    edge_id=edge_id, source_variable=source, target_variable=target,
                    mechanism_type=str(link.get("mechanism_type") or "discovered"),
                    functional_form=str(link.get("functional_form") or "unknown"),
                    conditions=dict(link.get("conditions") or {}),
                    lag_distribution=dict(link.get("lag_distribution") or {}),
                    scope=tuple(link.get("scope") or ()),
                    physical_topological_basis=tuple(link.get("physical_topological_basis") or ()),
                    evidence_refs=tuple(link.get("evidence_refs") or ()),
                    counterexamples=(), causal_confidence=float(link.get("causal_confidence", 0.0) or 0.0),
                    trust_level=0.0, scm_version=str(scm_version), valid_from=valid_from, valid_to=None,
                    lifecycle=MechanismLifecycle.OBSERVATIONAL_CANDIDATE, origin=MechanismOrigin.DISCOVERY,
                )
                self._mechanism_registry.register_discovery_candidate(mechanism, **layers)
            except (TypeError, ValueError) as exc:
                rejected[edge_id or f"link:{index}"] = str(exc)
                continue
            registered.append(edge_id)
        return {"registered": registered, "rejected": rejected, "lifecycle": "observational_candidate"}

    # =============================================================== CCIA §2 identification / estimation
    def identify(
        self,
        *,
        treatment: str,
        outcome: str,
        scope: Iterable[str],
        window: tuple[datetime, datetime] | None,
        unobserved_confounder_risk: float | None,
        uncertainty_threshold: float | None,
        assumptions_valid: bool = True,
        iv_assumption: IVAssumption | str | None = None,
    ) -> IdentificationResult:
        """Backdoor -> frontdoor -> ID -> (IV as ``partial`` under a declared parametric assumption) on the
        active causal graph augmented with the ConfounderRegistry (CCIA 2.1-2.2), then the CCIA 2.4
        abstention gate. No registry / scope / window / threshold -> CAUSE_NOT_IDENTIFIABLE (fail-closed).
        Observational identification never exceeds review."""

        edges, latent = self.causal_graph_structure()
        result = identify_on_graph_with_registry(
            self._confounder_registry, edges=edges, latent=latent, treatment=treatment, outcome=outcome,
            scope=scope, window=window, unobserved_confounder_risk=unobserved_confounder_risk,
            uncertainty_threshold=uncertainty_threshold, assumptions_valid=assumptions_valid,
            iv_assumption=iv_assumption)
        return replace(result, reasons=(*result.reasons, f"graph_version:{self.graph_version}"))

    def estimate_effect(
        self,
        *,
        treatment: str,
        outcome: str,
        data: Mapping[str, Sequence[Any]],
        treatment_value: Any,
        reference_value: Any,
        scope: Iterable[str],
        window: tuple[datetime, datetime] | None,
        unobserved_confounder_risk: float,
        unobserved_confounding_sensitivity: float,
        uncertainty_threshold: float,
        assumptions_valid: bool = True,
        iv_assumption: IVAssumption | str | None = None,
    ) -> IdentificationResult:
        """Data-backed E[Y|do(x)] - E[Y|do(x_ref)] via causal_estimators (backdoor strata, front-door,
        2SLS) or the ID expression -- the first identified method, never the maximum of heuristics.
        A ``partial`` IV answer carries its declared estimand (LATE / linear-SCM slope) and at most
        informational use. Reported as Effect +/- Sensitivity (CCIA 2.2) with the registry's latent
        confounders; abstains per CCIA 2.4."""

        identified = self.identify(treatment=treatment, outcome=outcome, scope=scope, window=window,
                                   unobserved_confounder_risk=unobserved_confounder_risk,
                                   uncertainty_threshold=uncertainty_threshold, assumptions_valid=assumptions_valid,
                                   iv_assumption=iv_assumption)
        partial_iv = identified.status is IdentificationStatus.PARTIAL and bool(identified.estimand)
        if identified.abstain or (identified.status is not IdentificationStatus.IDENTIFIED and not partial_iv):
            return identified
        edges, latent = self.causal_graph_structure()
        estimate = estimate_identified_effect(identified, edges=edges, latent=latent, data=data,
                                              treatment_value=treatment_value, reference_value=reference_value)
        effect = float(estimate.value)
        confounder_query = self._confounder_registry.query(treatment=treatment, outcome=outcome, scope=scope,
                                                           window=window)  # identify() refused if None
        if confounder_query.entries:
            from .confounder_registry import report_effect

            reported = report_effect(estimated_effect=effect,
                                     unobserved_confounding_sensitivity=unobserved_confounding_sensitivity,
                                     query=confounder_query)
        else:
            lower, upper = sensitivity_interval(estimated_effect=effect,
                                                unobserved_confounding_sensitivity=unobserved_confounding_sensitivity)
            reported = {"estimated": effect, "lower": lower, "upper": upper,
                        "sensitivity": float(unobserved_confounding_sensitivity)}
        use = str(estimate.intermediates.get("allowed_planning_use") or "review")
        reasons = identified.reasons
        if estimate.intermediates.get("weak_instrument"):
            reasons = (*reasons, "weak_instrument")
        if partial_iv:
            return replace(identified, effect=effect, reported_effect=reported, allowed_planning_use="informational",
                           reasons=reasons)
        return replace(identified, effect=effect, reported_effect=reported,
                       allowed_planning_use="review" if use == "review" else "informational_or_human_review",
                       reasons=reasons)

    def identify_backdoor_effect(
        self,
        *,
        treatment: str,
        outcome: str,
        adjustment_set: tuple[str, ...],
        conditional_outcomes: Mapping[str, float],
        confounder_probabilities: Mapping[str, float],
        stratum_assignments: Mapping[str, Mapping[str, Any]],
        unobserved_confounder_risk: float,
        latent: Iterable[str] = (),
    ) -> IdentificationResult:
        """CCIA 2.1 identification over the engine's own causal graph; never above "review"."""

        return identify_backdoor_effect(
            treatment=treatment,
            outcome=outcome,
            adjustment_set=adjustment_set,
            conditional_outcomes=conditional_outcomes,
            confounder_probabilities=confounder_probabilities,
            stratum_assignments=stratum_assignments,
            unobserved_confounder_risk=unobserved_confounder_risk,
            scm=self._structural_graph(),
            latent=latent,
        )

    def interventional_distribution(
        self,
        *,
        do: Mapping[str, float],
        context: Mapping[str, float] | None = None,
        samples: int,
        seed: int,
        alpha: float,
    ) -> Dict[str, Any]:
        """CEOS p.7: P(Y | do(X=x), C_t) as a Monte Carlo distribution over structural noise.
        CI = [Q_α/2, Q_1−α/2]; α is required (CEOS fixes no value)."""

        if self._structural_equations is None:
            raise ValueError("interventional distribution requires structural equations; none are configured")
        if samples <= 0 or not 0.0 < float(alpha) < 1.0:
            raise ValueError("samples > 0 and alpha in (0, 1) are required")
        result = self._structural_equations.sample_interventional(do, samples, seed, context=context)
        summary = {
            variable: {
                "mean": float(np.mean(values)),
                "lower": float(np.quantile(values, float(alpha) / 2.0)),
                "upper": float(np.quantile(values, 1.0 - float(alpha) / 2.0)),
            }
            for variable, values in result.value.items()
        }
        return {
            "distribution": summary,
            "alpha": float(alpha),
            "samples": int(samples),
            "seed": int(seed),
            "do": dict(result.intermediates["do"]),
            "context": dict(result.intermediates["context"]),
            "scm_version": result.intermediates["scm_version"],
            "formula": result.formula.formula_id,
            "epistemic_status": "simulated",
            "execution_status": "not_executed_simulation",
            "real_action": False,
        }

    # =============================================================== plan comparison
    async def compare_plans(
        self, *, interventions: List[Dict[str, Any]], world_state: Dict[str, Any] | Any
    ) -> Dict[str, Any]:
        """Публичный фасад CEOS для сравнения планов-вмешательств (планировщик не зовёт движки напрямую).

        Результат — симуляция: execution_status=not_executed_simulation, real_action=False.
        """
        if self._counterfactual_engine is None or not hasattr(self._counterfactual_engine, "compare_interventions"):
            return {"ranked_scenarios": [], "simulation_status": "no_counterfactual_engine", "real_action": False}
        out = self._counterfactual_engine.compare_interventions(
            interventions=[dict(x) for x in list(interventions or [])],
            world_state=self._to_state_dict(world_state),
        )
        if hasattr(out, "__await__"):
            out = await out
        payload = dict(out.to_dict() or {}) if hasattr(out, "to_dict") else dict(out or {})
        if hasattr(out, "ranked_scenarios") and "ranked_scenarios" not in payload:
            payload["ranked_scenarios"] = list(out.ranked_scenarios or [])
        payload.setdefault("ranked_scenarios", [])
        payload["execution_status"] = "not_executed_simulation"
        payload["real_action"] = False
        payload["engine"] = "CausalMetaEngine"
        return payload

    # =============================================================== private engine adapters
    async def _run_symbolic_simulation(self, *, state: Dict[str, Any] | Any, actions: List[Dict[str, Any]], max_steps: int) -> Dict[str, Any]:
        if self._causal_engine is None or not hasattr(self._causal_engine, "simulate"):
            return {}
        out = self._causal_engine.simulate(state=state, actions=actions, max_steps=max_steps)
        if hasattr(out, "__await__"):
            out = await out
        return dict(out or {})

    async def _run_world_simulator(self, *, state: Dict[str, Any] | Any, actions: List[Dict[str, Any]], max_steps: int) -> Dict[str, Any]:
        if self._world_simulator is None or not hasattr(self._world_simulator, "simulate"):
            return {}
        world_state = self._to_state_dict(state)
        plan = {"steps": [self._action_to_plan_step(x) for x in list(actions or [])]}
        out = self._world_simulator.simulate(
            plan=plan,
            world_state=world_state,
            n_scenarios=min(64, max(12, len(actions or []) * 8 or 12)),
            steps=max_steps,
        )
        if hasattr(out, "__await__"):
            out = await out
        if hasattr(out, "to_dict"):
            return dict(out.to_dict() or {})
        return dict(out or {})

    async def _run_counterfactual_baseline(self, *, state: Dict[str, Any] | Any, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not actions:
            return {}
        if self._counterfactual_engine is None or not hasattr(self._counterfactual_engine, "compare_interventions"):
            return {}
        interventions = [
            {"id": "baseline_noop", "actions": []},
            {"id": "proposed_plan", "actions": [dict(x) for x in list(actions or [])]},
        ]
        world_state = self._to_state_dict(state)
        out = self._counterfactual_engine.compare_interventions(
            interventions=interventions,
            world_state=world_state,
            steps=max(1, min(6, len(actions or []))),
        )
        if hasattr(out, "__await__"):
            out = await out
        if hasattr(out, "to_dict"):
            return dict(out.to_dict() or {})
        return dict(out or {})

    async def _run_reasoner_projection(self, *, state: Dict[str, Any] | Any, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self._causal_reasoner is None or not hasattr(self._causal_reasoner, "what_if"):
            return {}
        scenario = {"actions": [dict(x) for x in list(actions or [])], "id": "meta_projection"}
        out = self._causal_reasoner.what_if(
            scenario,
            {"world_state": self._to_state_dict(state)},
        )
        if hasattr(out, "__await__"):
            out = await out
        result = dict(out or {})
        consequences = [c for c in list(result.get("consequences") or [])
                        if isinstance((c or {}).get("confidence"), (int, float))]
        # a mean only of confidences actually reported; none reported -> not quantified (never 0)
        result["confidence"] = round(mean(float(c["confidence"]) for c in consequences), 4) if consequences else None
        return result

    async def _run_mental_rehearsal(self, *, state: Dict[str, Any] | Any, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self._mental_simulation is None or not hasattr(self._mental_simulation, "rehearse"):
            return {}
        plan = [self._action_label(x) for x in list(actions or [])]
        out = self._mental_simulation.rehearse(plan=plan, initial_state=self._to_state_dict(state))
        if hasattr(out, "__await__"):
            out = await out
        return self._simulation_to_dict(out)

    def _weights_used(self, components: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
        if self._weights is None:
            return {}
        out = {name: self._weights[name] for name, payload in components.items() if payload and name in self._weights}
        total = sum(out.values())
        return {k: round(v / total, 4) for k, v in out.items()} if total > 0 else {}

    def _weighted(self, pairs: List[tuple[Any, str]]) -> float | None:
        assert self._weights is not None
        weighted = [(float(v), self._weights[k]) for v, k in pairs
                    if k in self._weights and isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(float(v))]
        denominator = sum(w for _, w in weighted)
        if not weighted or denominator <= 0:
            return None
        return round(sum(v * w for v, w in weighted) / denominator, 4)

    def _aggregate_risk(self, *, symbolic, sim_world, baseline_cf) -> float | None:
        pairs: List[tuple[Any, str]] = []
        if symbolic:
            pairs.append((symbolic.get("total_risk"), "symbolic"))
        if sim_world:
            pairs.append((sim_world.get("average_risk"), "world_simulator"))
        if baseline_cf:
            risks = [row.get("total_risk") for row in list(baseline_cf.get("ranked_scenarios") or [])[:2]
                     if isinstance(row, dict) and isinstance(row.get("total_risk"), (int, float))]
            if risks:
                pairs.append((mean(float(r) for r in risks), "counterfactual"))
        return self._weighted(pairs)

    def _aggregate_confidence(self, *, symbolic, sim_world, sim_reasoner, sim_mental) -> float | None:
        pairs: List[tuple[Any, str]] = []
        if symbolic:
            pairs.append((symbolic.get("confidence"), "symbolic"))
        if sim_world:
            pairs.append((sim_world.get("success_probability"), "world_simulator"))
        if sim_reasoner:
            pairs.append((sim_reasoner.get("confidence"), "causal_reasoner"))
        if sim_mental:
            pairs.append((sim_mental.get("confidence"), "mental_simulation"))
        return self._weighted(pairs)

    @staticmethod
    def _action_to_plan_step(action: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "skill_or_tool": str(action.get("skill") or action.get("tool") or action.get("id") or action.get("action_id") or "noop"),
            "params": dict(action.get("params") or {}),
        }

    @staticmethod
    def _action_label(action: Dict[str, Any]) -> str:
        return str(action.get("skill") or action.get("tool") or action.get("id") or action.get("action_id") or "noop")

    @staticmethod
    def _to_state_dict(state: Dict[str, Any] | Any) -> Dict[str, Any]:
        if isinstance(state, dict):
            return deepcopy(state)
        if state is None:
            return {}
        return {
            "session_id": str(getattr(state, "session_id", "") or ""),
            "timestamp": str(getattr(state, "timestamp", "") or ""),
            "user": dict(getattr(state, "user", {}) or {}),
            "env": dict(getattr(state, "env", {}) or {}),
            "devices": dict(getattr(state, "devices", {}) or {}),
            "tasks": list(getattr(state, "tasks", []) or []),
            "active_states": [],
        }

    @staticmethod
    def _simulation_to_dict(result: Any) -> Dict[str, Any]:
        if result is None:
            return {}
        if hasattr(result, "to_dict"):
            return dict(result.to_dict() or {})
        if isinstance(result, dict):
            return dict(result)
        return {}


# совместимость: имена, которые прежний модуль реэкспортировал
from .causal_simulation import CounterfactualEngine  # noqa: E402,F401
from .structural_causal_model import estimate_effect, identify_effect  # noqa: E402,F401

__all__ = ["CausalMetaEngine", "CounterfactualEngine", "OBSERVATION_KINDS", "SimulationPolicy", "estimate_effect",
           "identify_effect"]
