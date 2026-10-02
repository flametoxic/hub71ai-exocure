from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .reality_formulas import FormulaReference, FormulaResult


DOCUMENT = "EXO-causal-reasoning-memory-planning-integration-addendum"


class IdentificationStatus(str, Enum):
    IDENTIFIED = "identified"
    PARTIAL = "partial"
    NON_IDENTIFIED = "non_identified"


class IdentificationMethod(str, Enum):
    """CCIA 2.1 (p.2) lists randomized | adjustment | frontdoor | transport | simulation-only.

    INSTRUMENTAL_VARIABLE (parametric: linear SCM, 2SLS) and ID_ALGORITHM (Shpitser & Pearl 2006,
    complete for non-parametric identification in semi-Markovian models) extend that list: they are
    the remaining identification strategies of Pearl's do-calculus apparatus.
    """

    RANDOMIZED = "randomized"
    ADJUSTMENT = "adjustment"
    FRONTDOOR = "frontdoor"
    TRANSPORT = "transport"
    SIMULATION_ONLY = "simulation_only"
    INSTRUMENTAL_VARIABLE = "instrumental_variable"
    ID_ALGORITHM = "id_algorithm"


class IVAssumption(str, Enum):
    """Parametric assumption under which an instrument yields a number (CCIA 2.1 ``partially``).

    Without one of these, an instrument does not identify the effect non-parametrically (Pearl,
    Causality §8.2: the IV graph only bounds the ATE). The caller's policy declares which one holds.
    """

    #: linear SCM with a homogeneous X->Y effect: the 2SLS slope equals the ATE
    LINEAR_CONSTANT_EFFECT = "linear_constant_effect"
    #: monotone compliance (no defiers): the 2SLS / Wald ratio is the LATE of compliers, not the ATE
    MONOTONICITY = "monotonicity"


# CCIA 2.4 (p.4): a first-class valid answer, not a failure.
CAUSE_NOT_IDENTIFIABLE = "CAUSE_NOT_IDENTIFIABLE"
EFFECT_IDENTIFIED = "EFFECT_IDENTIFIED"
#: CCIA 2.1 "identifiable: partially" -- a number exists only under declared parametric assumptions.
EFFECT_PARTIALLY_IDENTIFIED = "EFFECT_PARTIALLY_IDENTIFIED"
# CCIA 2.1 (p.3): non_identified -> allowed use informational/human review, next action below.
NON_IDENTIFIED_PLANNING_USE = "informational_or_human_review"
NEXT_ACTION_REQUEST_OBSERVATION = "request_observation_or_propose_safe_experiment"
NEXT_ACTION_ADJUST = "adjust_for_required_confounders"
F_ABSTAIN = FormulaReference(
    "CCIA-2.4-ABSTAIN",
    "EXO-causal-reasoning-memory-planning-integration-addendum",
    4,
    "abstain = 1[U_causal > tau_u OR AssumptionsInvalid OR EvidenceInsufficient]",
)
_IDENTIFIABLE = {"identified": "yes", "partial": "partially", "non_identified": "no"}


@dataclass(frozen=True)
class IdentificationResult:
    """CCIA 2.1-2.4 identification answer.

    ``reasons`` explain a refusal (never mixed into ``assumptions``); ``required_confounders`` is a
    minimal sufficient set (smallest valid backdoor set, or the smallest set of variables that would
    have to be measured); ``next_action`` is set whenever the effect is not identified;
    ``reported_effect`` is Effect_estimated +/- Sensitivity (CCIA 2.2, p.3) once a numeric effect exists;
    ``answer`` is CAUSE_NOT_IDENTIFIABLE whenever abstention applies (CCIA 2.4, p.4).

    ``partial`` results (e.g. an instrument under a declared parametric assumption) never exceed
    ``informational`` use and carry a number only together with the ``estimand`` it estimates
    (e.g. LATE of compliers, not the ATE).
    """

    treatment: str
    outcome: str
    status: IdentificationStatus
    method: IdentificationMethod | None
    required_confounders: tuple[str, ...]
    unobserved_confounder_risk: float | None
    assumptions: tuple[str, ...]
    allowed_planning_use: str
    effect: float | None = None
    reasons: tuple[str, ...] = ()
    next_action: str | None = None
    reported_effect: Any = None
    answer: str = ""
    abstain: bool = False
    expression: str | None = None
    hedge: str | None = None
    mediators: tuple[str, ...] = ()
    instrument: str | None = None
    formula: FormulaReference | None = None
    estimand: str | None = None

    def __post_init__(self) -> None:
        if self.status is IdentificationStatus.PARTIAL:
            if self.allowed_planning_use not in {NON_IDENTIFIED_PLANNING_USE, "informational"}:
                # a partially identified effect rests on untestable parametric assumptions
                raise ValueError("a partially identified effect allows at most informational use")
            if self.effect is not None and not str(self.estimand or "").strip():
                raise ValueError("a partially identified number must name its estimand (e.g. LATE)")
        elif self.status is not IdentificationStatus.IDENTIFIED and self.effect is not None:
            # CCIA 2.1: no numerical causal effect only because a causal graph exists.
            raise ValueError("a non-identified effect cannot carry a numerical value")
        if self.status is IdentificationStatus.NON_IDENTIFIED:
            if not self.next_action:
                object.__setattr__(self, "next_action", NEXT_ACTION_REQUEST_OBSERVATION)
            if not self.answer:
                object.__setattr__(self, "answer", CAUSE_NOT_IDENTIFIABLE)
            object.__setattr__(self, "abstain", True)
        elif not self.answer:
            if self.abstain:
                answer = CAUSE_NOT_IDENTIFIABLE
            elif self.status is IdentificationStatus.PARTIAL:
                answer = EFFECT_PARTIALLY_IDENTIFIED
            else:
                answer = EFFECT_IDENTIFIED
            object.__setattr__(self, "answer", answer)
        if self.answer == CAUSE_NOT_IDENTIFIABLE and self.allowed_planning_use not in {
            NON_IDENTIFIED_PLANNING_USE, "informational"
        }:
            raise ValueError("CAUSE_NOT_IDENTIFIABLE allows only informational/human review use")
        if self.allowed_planning_use not in {NON_IDENTIFIED_PLANNING_USE, "informational", "review"}:
            # CEOS p.9: observational identification never exceeds review (ActionAuthority != CausalConfidence).
            raise ValueError("observational identification cannot exceed review")

    @property
    def identifiable(self) -> str:
        """CCIA 2.1 vocabulary: yes | no | partially."""
        return _IDENTIFIABLE[self.status.value]


def apply_abstention(
    result: IdentificationResult,
    *,
    uncertainty_threshold: float | None,
    assumptions_valid: bool,
    evidence_sufficient: bool = True,
) -> IdentificationResult:
    """CCIA 2.4: abstain = 1[U_causal > tau_u OR AssumptionsInvalid OR EvidenceInsufficient].

    U_causal is the unobserved-confounder risk of the identification (None = not assessed = 1.0);
    tau_u comes from policy (no default): ``None`` means the policy is not configured and the answer
    abstains (fail-closed). A non-identified effect always abstains (evidence cannot identify it).
    Abstention withholds the numeric effect and answers CAUSE_NOT_IDENTIFIABLE.
    """

    from api_gateway.core.reasoning.proof_obligations import should_abstain

    uncertainty = 1.0 if result.unobserved_confounder_risk is None else result.unobserved_confounder_risk
    threshold_missing = uncertainty_threshold is None
    abstain = threshold_missing or should_abstain(
        causal_uncertainty=uncertainty,
        uncertainty_threshold=float(uncertainty_threshold),
        assumptions_valid=assumptions_valid,
        evidence_sufficient=bool(evidence_sufficient) and result.status is not IdentificationStatus.NON_IDENTIFIED,
    )
    if not abstain:
        return replace(result, formula=result.formula or F_ABSTAIN)
    reasons = list(result.reasons)
    if threshold_missing:
        reasons.append("abstain:uncertainty_threshold_not_configured")
    elif uncertainty > float(uncertainty_threshold):
        reasons.append("abstain:causal_uncertainty_above_threshold")
    if assumptions_valid is not True:
        reasons.append("abstain:assumptions_invalid")
    if result.status is IdentificationStatus.NON_IDENTIFIED or evidence_sufficient is not True:
        reasons.append("abstain:evidence_insufficient")
    return replace(
        result,
        effect=None,
        reported_effect=None,
        answer=CAUSE_NOT_IDENTIFIABLE,
        abstain=True,
        allowed_planning_use=NON_IDENTIFIED_PLANNING_USE,
        next_action=result.next_action or NEXT_ACTION_REQUEST_OBSERVATION,
        reasons=tuple(dict.fromkeys(reasons)),
        formula=F_ABSTAIN,
    )


@dataclass(frozen=True)
class MonteCarloSummary:
    mean: float
    lower: float
    upper: float
    sample_count: int
    alpha: float
    formula: FormulaReference


def backdoor_adjustment(
    *,
    conditional_outcomes: Mapping[str, float],
    confounder_probabilities: Mapping[str, float],
) -> FormulaResult:
    if not conditional_outcomes or set(conditional_outcomes) != set(confounder_probabilities):
        raise ValueError("conditional outcomes and confounder probabilities must share a non-empty support")
    probability_sum = 0.0
    terms: dict[str, float] = {}
    for stratum in conditional_outcomes:
        outcome_probability = float(conditional_outcomes[stratum])
        stratum_probability = float(confounder_probabilities[stratum])
        if not 0.0 <= outcome_probability <= 1.0 or not 0.0 <= stratum_probability <= 1.0:
            raise ValueError("all probabilities must be in [0, 1]")
        probability_sum += stratum_probability
        terms[stratum] = outcome_probability * stratum_probability
    if not np.isclose(probability_sum, 1.0):
        raise ValueError("confounder probabilities must sum to 1")
    return FormulaResult(
        value=sum(terms.values()),
        formula=FormulaReference(
            "CCIA-2.1-BACKDOOR",
            DOCUMENT,
            2,
            "P(Y|do(X=x)) = sum_z P(Y|X=x,Z=z)P(Z=z)",
        ),
        intermediates={"stratum_terms": terms},
    )


def _validated_strata(
    *,
    adjustment_set: tuple[str, ...],
    conditional_outcomes: Mapping[str, float],
    confounder_probabilities: Mapping[str, float],
    stratum_assignments: Mapping[str, Mapping[str, Any]],
) -> None:
    """Every stratum must be an explicit assignment Z=z over exactly the adjustment variables."""

    keys = set(conditional_outcomes)
    if not keys or keys != set(confounder_probabilities) or keys != set(stratum_assignments):
        raise ValueError("conditional outcomes, P(Z=z) and stratum assignments must share one non-empty support")
    expected = set(adjustment_set)
    seen: set[tuple[tuple[str, str], ...]] = set()
    for stratum, assignment in stratum_assignments.items():
        variables = {str(name) for name in assignment}
        if variables != expected:
            raise ValueError(
                f"stratum {stratum!r} assigns {sorted(variables)}, expected exactly the adjustment set {sorted(expected)}"
            )
        signature = tuple(sorted((str(name), repr(value)) for name, value in assignment.items()))
        if signature in seen:
            raise ValueError(f"duplicate stratum assignment for Z: {dict(assignment)!r}")
        seen.add(signature)
        if not np.isfinite(float(conditional_outcomes[stratum])):
            raise ValueError("conditional outcomes must be finite")


def identify_backdoor_effect(
    *,
    treatment: str,
    outcome: str,
    adjustment_set: tuple[str, ...],
    conditional_outcomes: Mapping[str, float],
    confounder_probabilities: Mapping[str, float],
    stratum_assignments: Mapping[str, Mapping[str, Any]],
    unobserved_confounder_risk: float,
    scm=None,
    edges: Iterable[tuple[str, str]] | None = None,
    latent: Iterable[str] = (),
) -> IdentificationResult:
    """CCIA 2.1: E[Y|do(X=x)] = sum_z E[Y|X=x,Z=z] P(Z=z), only when Z satisfies the backdoor criterion
    computed from a causal graph (never from caller-asserted flags). Works for probabilities and for
    continuous outcomes. Observational identification never grants action authority: at most "review".
    """

    if (scm is None) == (edges is None):
        raise ValueError("exactly one causal graph source is required: scm or edges")
    latent_nodes = {str(node) for node in latent}
    if scm is not None:
        graph_edges = tuple(scm.directed_edges())
        latent_nodes |= set(scm.latent_variables())
        graph_nodes = set(scm.variables)
    else:
        graph_edges = tuple((str(cause), str(effect)) for cause, effect in edges)
        graph_nodes = {node for edge in graph_edges for node in edge}
    adjustment = tuple(str(node) for node in adjustment_set)
    unknown = sorted(set(adjustment) - graph_nodes)
    if unknown:
        raise ValueError(f"adjustment variables are not nodes of the causal graph: {unknown}")
    if str(treatment) in adjustment or str(outcome) in adjustment:
        raise ValueError("adjustment set cannot contain X or Y")
    _validated_strata(
        adjustment_set=adjustment,
        conditional_outcomes=conditional_outcomes,
        confounder_probabilities=confounder_probabilities,
        stratum_assignments=stratum_assignments,
    )
    # Local import: causal_graph_identification imports the result types from this module.
    from .causal_graph_identification import identify_backdoor_from_graph

    return identify_backdoor_from_graph(
        edges=graph_edges,
        treatment=str(treatment),
        outcome=str(outcome),
        adjustment=adjustment,
        conditional_means={str(key): float(value) for key, value in conditional_outcomes.items()},
        stratum_probabilities={str(key): float(value) for key, value in confounder_probabilities.items()},
        unobserved_confounder_risk=float(unobserved_confounder_risk),
        latent=latent_nodes,
    )


def monte_carlo_summary(samples: Sequence[float], *, alpha: float) -> MonteCarloSummary:
    """CEOS p.7: Ŷ = mean, CI = [Q_α/2, Q_1−α/2]; α is policy (CEOS fixes no value, no default)."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("samples must be a non-empty finite vector")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return MonteCarloSummary(
        mean=float(np.mean(values)),
        lower=float(np.quantile(values, alpha / 2.0)),
        upper=float(np.quantile(values, 1.0 - alpha / 2.0)),
        sample_count=int(values.size),
        alpha=float(alpha),
        formula=FormulaReference(
            "CEOS-9-MONTE-CARLO",
            "EXO-causal-engine-operational-specification",
            7,
            "Y_hat=(1/N)sum_n Y^(n); CI=[Q_(alpha/2)(Y),Q_(1-alpha/2)(Y)]",
        ),
    )


__all__ = [
    "CAUSE_NOT_IDENTIFIABLE",
    "EFFECT_IDENTIFIED",
    "EFFECT_PARTIALLY_IDENTIFIED",
    "F_ABSTAIN",
    "IVAssumption",
    "NEXT_ACTION_ADJUST",
    "NEXT_ACTION_REQUEST_OBSERVATION",
    "NON_IDENTIFIED_PLANNING_USE",
    "apply_abstention",
    "IdentificationMethod",
    "IdentificationResult",
    "IdentificationStatus",
    "MonteCarloSummary",
    "backdoor_adjustment",
    "identify_backdoor_effect",
    "monte_carlo_summary",
]

