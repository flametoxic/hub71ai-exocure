from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from ..world_model.reality_formulas import FormulaReference


HYPOTHESIS_SCORE_FORMULA = FormulaReference(
    formula_id="CEOS-HYPOTHESIS-SCORE",
    document="EXO-causal-engine-operational-specification",
    page=5,
    expression="Score(H)=w_E E+w_P P+w_T T+w_S S+w_I I-w_X X",
)
# CURE-REALITY-ENGINE-MASTER-TZ B4 (p.11) and TZ-WM-ENGINEERING-SPEC 6.5 (F39/C8-C10, p.17-18).
COUNCIL_SCORE_WEIGHTS_FORMULA = FormulaReference(
    formula_id="MASTER-B4-COUNCIL-SCORE",
    document="CURE-REALITY-ENGINE-MASTER-TZ",
    page=11,
    expression="Score(H_k)=0.30E_k+0.25P_k+0.15T_k+0.15S_k-0.40X_k",
)
COUNCIL_DECISION_FORMULA = FormulaReference(
    formula_id="MASTER-B4-COUNCIL-DECISION",
    document="CURE-REALITY-ENGINE-MASTER-TZ",
    page=11,
    expression=(
        "Accept <=> Score(H1)-Score(H2)>0.15 AND X_H1<0.2 AND P_H1=1; "
        "else CONFLICTED; all scores <0.4: ABSTAIN with missing_information"
    ),
)
ACCEPT_MARGIN = 0.15
ACCEPT_MAX_CONTRADICTION = 0.2
ACCEPT_REQUIRED_PHYSICS = 1.0
ABSTAIN_SCORE = 0.4

STATUS_ACCEPT = "ACCEPT"
STATUS_CONFLICTED = "CONFLICTED"
STATUS_ABSTAIN = "ABSTAIN"
STATUS_NOT_IDENTIFIABLE = "CAUSE_NOT_IDENTIFIABLE"  # CCIA 2.4 abstention gate
_DECIMALS = 12  # compare thresholds on rounded scores so float noise never crosses a boundary


def _q(value: float) -> float:
    return round(float(value), _DECIMALS)


class HypothesisFamily(str, Enum):
    PHYSICAL = "physical_or_kinematic"
    TOPOLOGY_CONFIGURATION = "topology_or_configuration"
    EXOGENOUS = "exogenous"
    SENSOR_TIME_INTEGRITY = "sensor_calibration_or_time_integrity"
    UNKNOWN_CONFOUNDER = "unknown_mechanism_or_confounder"


#: CEOS §7: alternatives every root-cause decision must have considered (H4, H5)
REQUIRED_ALTERNATIVE_FAMILIES = (HypothesisFamily.SENSOR_TIME_INTEGRITY, HypothesisFamily.UNKNOWN_CONFOUNDER)


class HypothesisStatus(str, Enum):
    """CEOS p.5 hypothesis field ``status``."""

    CANDIDATE = "candidate"
    COMPETING = "competing"
    ACCEPTED = "accepted"
    ELIMINATED = "eliminated"
    CONFLICTED = "conflicted"
    REJECTED_BY_CONSTRAINTS = "rejected_by_constraints"
    ABSTAINED = "abstained"


@dataclass(frozen=True)
class HypothesisScoreWeights:
    """w_E, w_P, w_T, w_S, w_I, w_X of CEOS p.5. CEOS fixes no numbers: weights are configuration and
    must name their ``source`` (a policy id or a TZ reference)."""

    evidence: float
    physics: float
    topology: float
    simulation: float
    intervention: float
    contradiction: float
    source: str = ""

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in (
            self.evidence,
            self.physics,
            self.topology,
            self.simulation,
            self.intervention,
            self.contradiction,
        )):
            raise ValueError("hypothesis score weights must be non-negative")

    @classmethod
    def uniform(cls, *, source: str = "policy:uniform") -> "HypothesisScoreWeights":
        return cls(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, source=source)

    @classmethod
    def master_b4(cls) -> "HypothesisScoreWeights":
        """Weights exactly as written in Master TZ B4 / engineering spec F39.

        Those documents define no I (intervention/outcome support) term, so its weight is 0 here;
        a deployment that wants CEOS's w_I term must configure it explicitly.
        """

        return cls(evidence=0.30, physics=0.25, topology=0.15, simulation=0.15, intervention=0.0, contradiction=0.40,
                   source="CURE-REALITY-ENGINE-MASTER-TZ B4 p.11 / TZ-WM-ENGINEERING-SPEC F39")


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    family: HypothesisFamily
    claim: str
    causal_path: tuple[str, ...]
    expected_observations: tuple[str, ...]
    contradictory_observations: tuple[str, ...]
    required_next_observations: tuple[str, ...]
    scope: tuple[str, ...]
    assumptions: tuple[str, ...]
    evidence_support: float
    physics_consistency: float
    topology_compatibility: float
    simulation_fit: float
    intervention_support: float
    contradictory_evidence: float
    evidence_refs: tuple[str, ...]
    llm_generated: bool = False
    #: CEOS p.5 fields; confidence is None until quantified (e.g. by the TMS), never an invented number
    confidence: float | None = None
    status: HypothesisStatus = HypothesisStatus.CANDIDATE

    def __post_init__(self) -> None:
        if not self.hypothesis_id.strip() or not self.claim.strip() or not self.scope:
            raise ValueError("hypothesis identity, claim, and scope are required")
        object.__setattr__(self, "status", HypothesisStatus(self.status))
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        for name in (
            "evidence_support",
            "physics_consistency",
            "topology_compatibility",
            "simulation_fit",
            "intervention_support",
            "contradictory_evidence",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.llm_generated and not any(str(item).strip() for item in self.assumptions):
            # CEOS p.6: an LLM may only propose a candidate and must state its assumptions.
            raise ValueError("LLM-generated hypothesis must state its assumptions")

    def score(self, weights: HypothesisScoreWeights) -> float:
        evidence = 0.0 if self.llm_generated else self.evidence_support
        return (
            weights.evidence * evidence
            + weights.physics * self.physics_consistency
            + weights.topology * self.topology_compatibility
            + weights.simulation * self.simulation_fit
            + weights.intervention * self.intervention_support
            - weights.contradiction * self.contradictory_evidence
        )


@dataclass(frozen=True)
class RankedHypothesis:
    hypothesis: Hypothesis
    score: float
    status: HypothesisStatus = HypothesisStatus.COMPETING

    def as_contract(self) -> dict[str, Any]:
        """The ten CEOS p.5 fields (+ family, score, evidence refs)."""
        h = self.hypothesis
        return {
            "hypothesis_id": h.hypothesis_id,
            "claim": h.claim,
            "causal_path": list(h.causal_path),
            "expected_observations": list(h.expected_observations),
            "contradictory_observations": list(h.contradictory_observations),
            "required_next_observations": list(h.required_next_observations),
            "scope": list(h.scope),
            "confidence": h.confidence,
            "assumptions": list(h.assumptions),
            "status": self.status.value,
            "family": h.family.value,
            "score": self.score,
            "evidence_refs": list(h.evidence_refs),
            "llm_generated": h.llm_generated,
        }


@dataclass(frozen=True)
class HypothesisCouncilResult:
    status: str
    ranked_hypotheses: tuple[RankedHypothesis, ...]
    recommended_next_observations: tuple[str, ...]
    allowed_planning_use: str
    causal_uncertainty: float
    accepted_hypothesis_id: str | None = None
    decision_reasons: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()
    elimination_trace: tuple[Mapping[str, Any], ...] = ()
    rejected_hypotheses: tuple[RankedHypothesis, ...] = ()
    weights: HypothesisScoreWeights | None = None
    weights_source: str = ""


class HypothesisCouncil:
    """Ranks competing hypotheses and applies the Master B4 / F39 decision rule.

    Order: CCIA 2.4 abstention gate -> top score < 0.4: ABSTAIN -> accept iff
    Score(H1)-Score(H2) > 0.15, X_H1 < 0.2, P_H1 = 1 -> otherwise CONFLICTED. CEOS p.5: a root cause
    is never emitted without alternatives, so a single hypothesis cannot be accepted. Even ACCEPT
    only reaches "review": ActionAuthority != CausalConfidence (CEOS p.9).
    """

    def __init__(
        self,
        *,
        uncertainty_threshold: float,
        weights: HypothesisScoreWeights,
    ) -> None:
        if not 0.0 <= uncertainty_threshold <= 1.0:
            raise ValueError("uncertainty_threshold must be in [0, 1]")
        if not isinstance(weights, HypothesisScoreWeights) or not str(weights.source).strip():
            # CEOS p.5 gives the form of Score(H_k) but no numbers: the weights come from configuration
            # or policy and must name their source (e.g. HypothesisScoreWeights.master_b4()).
            raise ValueError("hypothesis score weights must be configured explicitly with their source")
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.weights = weights

    def evaluate(
        self,
        hypotheses: Iterable[Hypothesis],
        *,
        causal_uncertainty: float,
        assumptions_valid: bool = True,
        evidence_sufficient: bool = True,
        constraint_verdicts: Mapping[str, bool] | None = None,
    ) -> HypothesisCouncilResult:
        """``constraint_verdicts``: hypothesis_id -> passed X_valid (CEOS p.6). Rejected hypotheses are
        removed before ranking; without a verdict for every hypothesis nothing can be ACCEPTed."""

        uncertainty = float(causal_uncertainty)
        if not 0.0 <= uncertainty <= 1.0:
            raise ValueError("causal_uncertainty must be in [0, 1]")
        items = tuple(hypotheses)
        if len({item.hypothesis_id for item in items}) != len(items):
            raise ValueError("hypothesis ids must be unique")
        verdicts = {str(k): v for k, v in dict(constraint_verdicts or {}).items()}
        rejected = tuple(
            RankedHypothesis(item, item.score(self.weights), HypothesisStatus.REJECTED_BY_CONSTRAINTS)
            for item in items if verdicts.get(item.hypothesis_id) is False
        )
        constraints_checked = constraint_verdicts is not None and all(
            verdicts.get(item.hypothesis_id) in (True, False) for item in items
        )
        items = tuple(item for item in items if verdicts.get(item.hypothesis_id) is not False)
        ranked = tuple(sorted(
            (RankedHypothesis(hypothesis=item, score=item.score(self.weights)) for item in items),
            key=lambda item: (-item.score, item.hypothesis.hypothesis_id),
        ))
        observations = tuple(dict.fromkeys(
            observation
            for item in ranked
            for observation in item.hypothesis.required_next_observations
        ))

        gate = [
            reason
            for reason, failed in (
                ("causal_uncertainty_above_threshold", uncertainty > self.uncertainty_threshold),
                ("assumptions_invalid", not assumptions_valid),
                ("evidence_insufficient", not evidence_sufficient),
            )
            if failed
        ]
        common = dict(rejected=rejected, weights=self.weights)
        if gate:
            return self._result(STATUS_NOT_IDENTIFIABLE, ranked, observations, uncertainty, reasons=tuple(gate),
                                **common)
        if not ranked:
            return self._result(
                STATUS_ABSTAIN,
                ranked,
                observations,
                uncertainty,
                reasons=("no_hypotheses",) if not rejected else ("all_hypotheses_rejected_by_constraints",),
                missing=("competing_hypotheses",),
                **common,
            )
        top = ranked[0]
        if _q(top.score) < ABSTAIN_SCORE:
            return self._result(
                STATUS_ABSTAIN,
                ranked,
                observations,
                uncertainty,
                reasons=(f"all_scores_below_{ABSTAIN_SCORE}",),
                missing=observations or ("discriminating_observation",),
                **common,
            )

        reasons: list[str] = []
        if len(ranked) < 2:
            reasons.append("no_competing_alternative")
        # CEOS §7: "never emits one root cause without alternatives" -- H4 (sensor/calibration/time
        # integrity) and H5 (unobserved confounder / unknown mechanism) must have been considered (ranked
        # or ruled out by constraints) before any root cause is accepted.
        considered = {item.hypothesis.family for item in (*ranked, *rejected)}
        missing_families = [family.value for family in REQUIRED_ALTERNATIVE_FAMILIES if family not in considered]
        if missing_families:
            reasons.append("missing_alternative_families:" + ",".join(missing_families))
        elif not _q(top.score - ranked[1].score) > ACCEPT_MARGIN:
            reasons.append(f"score_margin_not_above_{ACCEPT_MARGIN}")
        if not top.hypothesis.contradictory_evidence < ACCEPT_MAX_CONTRADICTION:
            reasons.append(f"top_contradiction_not_below_{ACCEPT_MAX_CONTRADICTION}")
        if top.hypothesis.physics_consistency != ACCEPT_REQUIRED_PHYSICS:
            reasons.append("top_physics_consistency_not_1")
        if top.hypothesis.llm_generated:
            # CEOS p.6: an LLM hypothesis is a candidate, never an accepted root cause by itself.
            reasons.append("top_hypothesis_llm_generated_candidate_only")
        if not constraints_checked:
            # CEOS p.6: constraints reject impossible hypotheses -- unchecked ones cannot be accepted.
            reasons.append("hypotheses_not_constraint_checked")
        if reasons:
            return self._result(STATUS_CONFLICTED, ranked, observations, uncertainty, reasons=tuple(reasons),
                                **common)

        trace = tuple(
            {
                "hypothesis_id": item.hypothesis.hypothesis_id,
                "score": item.score,
                "eliminated_by": f"score {item.score:.4f} trails accepted {top.score:.4f} by more than {ACCEPT_MARGIN}",
            }
            for item in ranked[1:]
        )
        statuses = [RankedHypothesis(top.hypothesis, top.score, HypothesisStatus.ACCEPTED)] + [
            RankedHypothesis(item.hypothesis, item.score, HypothesisStatus.ELIMINATED) for item in ranked[1:]
        ]
        return HypothesisCouncilResult(
            status=STATUS_ACCEPT,
            ranked_hypotheses=tuple(statuses),
            recommended_next_observations=(),
            allowed_planning_use="review",
            causal_uncertainty=uncertainty,
            accepted_hypothesis_id=top.hypothesis.hypothesis_id,
            decision_reasons=("accept_rule_satisfied",),
            elimination_trace=trace,
            rejected_hypotheses=rejected,
            weights=self.weights,
            weights_source=self.weights.source,
        )

    @staticmethod
    def _result(
        status: str,
        ranked: tuple[RankedHypothesis, ...],
        observations: tuple[str, ...],
        uncertainty: float,
        *,
        reasons: tuple[str, ...],
        missing: tuple[str, ...] = (),
        rejected: tuple[RankedHypothesis, ...] = (),
        weights: HypothesisScoreWeights | None = None,
    ) -> HypothesisCouncilResult:
        label = HypothesisStatus.CONFLICTED if status == STATUS_CONFLICTED else HypothesisStatus.ABSTAINED
        relabelled = tuple(
            RankedHypothesis(item.hypothesis, item.score,
                             HypothesisStatus.CANDIDATE if item.hypothesis.llm_generated else label)
            for item in ranked
        )
        return HypothesisCouncilResult(
            status=status,
            ranked_hypotheses=relabelled,
            recommended_next_observations=observations,
            allowed_planning_use="informational_or_human_review",
            causal_uncertainty=uncertainty,
            decision_reasons=reasons,
            missing_information=missing,
            rejected_hypotheses=rejected,
            weights=weights,
            weights_source=weights.source if weights is not None else "",
        )


__all__ = [
    "ABSTAIN_SCORE",
    "ACCEPT_MARGIN",
    "ACCEPT_MAX_CONTRADICTION",
    "ACCEPT_REQUIRED_PHYSICS",
    "COUNCIL_DECISION_FORMULA",
    "COUNCIL_SCORE_WEIGHTS_FORMULA",
    "HYPOTHESIS_SCORE_FORMULA",
    "Hypothesis",
    "HypothesisCouncil",
    "HypothesisCouncilResult",
    "HypothesisFamily",
    "HypothesisScoreWeights",
    "HypothesisStatus",
    "RankedHypothesis",
    "STATUS_ABSTAIN",
    "REQUIRED_ALTERNATIVE_FAMILIES",
    "STATUS_ACCEPT",
    "STATUS_CONFLICTED",
    "STATUS_NOT_IDENTIFIABLE",
]

