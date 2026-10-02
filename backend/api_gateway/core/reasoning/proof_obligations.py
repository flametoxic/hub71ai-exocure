from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from math import isfinite
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

from ..world_model.reality_formulas import FormulaReference, FormulaResult


ABSTENTION_FORMULA = FormulaReference(
    "CCIA-ABSTENTION",
    "EXO-causal-reasoning-memory-planning-integration-addendum",
    4,
    "Abstain=1[U_causal>tau_u OR assumptions_invalid OR evidence_insufficient]",
)

PROOF_LEVEL_FORMULA = FormulaReference(
    "CCIA-3.1-PROOF-LEVEL",
    "EXO-causal-reasoning-memory-planning-integration-addendum",
    4,
    "ProofLevel(q,use) = f(Risk, Reversibility, BlastRadius, Uncertainty, Authority)",
)


class ClaimUse(str, Enum):
    UI_EXPLANATION = "ui_explanation"
    DIAGNOSTIC_RECOMMENDATION = "diagnostic_recommendation"
    HIGH_IMPACT_PLAN_SUPPORT = "high_impact_plan_support"
    PHYSICAL_ACTION = "physical_action"
    RULE_PROMOTION = "rule_promotion"


class ProofLevel(IntEnum):
    EVIDENCE_AND_UNCERTAINTY = 1
    COMPETING_HYPOTHESES_AND_CONSTRAINTS = 2
    IDENTIFIED_MECHANISM_AND_SIMULATION = 3
    SAFETY_CASE_POLICY_VERIFICATION = 4
    FORMAL_VALIDATION_REPLAY_REVIEW = 5


class ProofElement(str, Enum):
    """Minimum requirements of the CCIA §3.1 table (p.4)."""

    EVIDENCE_LINKS = "evidence_links"
    UNCERTAINTY = "uncertainty"
    COMPETING_HYPOTHESES = "competing_hypotheses"
    CONSTRAINTS = "constraints"
    IDENTIFIED_MECHANISM = "identified_causal_mechanism"
    SIMULATION_EVIDENCE = "simulation_evidence"
    SAFETY_CASE = "safety_case"
    POLICY = "policy"
    VERIFICATION_PLAN = "verification_plan"
    FORMAL_VALIDATION = "formal_validation"
    REPLAY = "replay"
    REVIEW = "review"


_LEVEL_ELEMENTS = {
    ProofLevel.EVIDENCE_AND_UNCERTAINTY: frozenset({ProofElement.EVIDENCE_LINKS, ProofElement.UNCERTAINTY}),
    ProofLevel.COMPETING_HYPOTHESES_AND_CONSTRAINTS: frozenset(
        {ProofElement.COMPETING_HYPOTHESES, ProofElement.CONSTRAINTS}
    ),
    ProofLevel.IDENTIFIED_MECHANISM_AND_SIMULATION: frozenset(
        {ProofElement.IDENTIFIED_MECHANISM, ProofElement.SIMULATION_EVIDENCE}
    ),
    ProofLevel.SAFETY_CASE_POLICY_VERIFICATION: frozenset(
        {ProofElement.SAFETY_CASE, ProofElement.POLICY, ProofElement.VERIFICATION_PLAN}
    ),
    ProofLevel.FORMAL_VALIDATION_REPLAY_REVIEW: frozenset(
        {ProofElement.FORMAL_VALIDATION, ProofElement.REPLAY, ProofElement.REVIEW}
    ),
}


def level_elements(level: ProofLevel) -> frozenset[ProofElement]:
    """The row of the CCIA §3.1 table for one level (nothing inherited from other rows)."""
    return _LEVEL_ELEMENTS[ProofLevel(level)]


def required_proof_elements(level: ProofLevel, *, claim_use: "ClaimUse | None" = None) -> frozenset[ProofElement]:
    """Minimum requirements of a proof obligation (CCIA §3.1, p.4).

    The addendum lists one «minimum requirement» per claim/use and does not say that a row inherits
    the rows above it (a physical action needs «safety case + policy + verification plan», not a
    theorem-proved mechanism). So requirements do **not** accumulate across levels: the obligation is
    the row of the required level plus the row of the use's own minimum (when the policy escalated
    the level above the use's minimum, the use's own minimum still applies).
    """
    elements = set(level_elements(level))
    if claim_use is not None:
        elements |= level_elements(_MINIMUM[ClaimUse(claim_use)])
    return frozenset(elements)


def satisfied_levels(elements: Iterable[ProofElement]) -> frozenset[ProofLevel]:
    """Levels whose table row is fully covered by ``elements``."""
    present = {ProofElement(item) for item in elements}
    return frozenset(level for level in ProofLevel if _LEVEL_ELEMENTS[level] <= present)


def achieved_proof_level(elements: Iterable[ProofElement]) -> ProofLevel | None:
    """Highest level whose own row is satisfied; None if no row is satisfied (reporting only —
    whether an obligation is met is decided by :func:`required_proof_elements` ⊆ elements)."""
    levels = satisfied_levels(elements)
    return max(levels) if levels else None


_MINIMUM = {
    ClaimUse.UI_EXPLANATION: ProofLevel.EVIDENCE_AND_UNCERTAINTY,
    ClaimUse.DIAGNOSTIC_RECOMMENDATION: ProofLevel.COMPETING_HYPOTHESES_AND_CONSTRAINTS,
    ClaimUse.HIGH_IMPACT_PLAN_SUPPORT: ProofLevel.IDENTIFIED_MECHANISM_AND_SIMULATION,
    ClaimUse.PHYSICAL_ACTION: ProofLevel.SAFETY_CASE_POLICY_VERIFICATION,
    ClaimUse.RULE_PROMOTION: ProofLevel.FORMAL_VALIDATION_REPLAY_REVIEW,
}


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite (fail-closed)")
    return number


def should_abstain(
    *,
    causal_uncertainty: float,
    uncertainty_threshold: float,
    assumptions_valid: bool,
    evidence_sufficient: bool,
) -> bool:
    """CCIA p.4: 1[U_causal > tau_u OR invalid assumptions OR insufficient evidence].

    Fail-closed: non-finite / non-numeric uncertainty or threshold, or flags that are not
    literally ``True``, mean the preconditions of an answer are unknown -> abstain.
    """
    try:
        uncertainty = float(causal_uncertainty)
        threshold = float(uncertainty_threshold)
    except (TypeError, ValueError):
        return True
    if not isfinite(uncertainty) or not isfinite(threshold):
        return True
    return uncertainty > threshold or assumptions_valid is not True or evidence_sufficient is not True


@dataclass(frozen=True)
class ProofEscalationRule:
    """One escalation row of the deployment policy: if ANY condition holds, the proof obligation is
    raised to at least ``level``. Every threshold is mandatory (no defaults, CCIA §3.1 gives only f)."""

    level: ProofLevel
    risk_at_least: float
    reversibility_below: float
    blast_radius_above: float
    uncertainty_at_least: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", ProofLevel(self.level))
        for name in ("risk_at_least", "reversibility_below", "uncertainty_at_least"):
            value = _finite(name, getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)
        radius = _finite("blast_radius_above", self.blast_radius_above)
        if radius < 0.0:
            raise ValueError("blast_radius_above must be non-negative")
        object.__setattr__(self, "blast_radius_above", radius)

    def triggered(self, *, risk: float, reversibility: float, blast_radius: float, uncertainty: float) -> bool:
        return (
            risk >= self.risk_at_least
            or reversibility < self.reversibility_below
            or blast_radius > self.blast_radius_above
            or uncertainty >= self.uncertainty_at_least
        )


@dataclass(frozen=True)
class ProofObligationConfig:
    """Deployment configuration of ProofLevel(q, use) = f(Risk, Reversibility, BlastRadius,
    Uncertainty, Authority) (CCIA §3.1, p.4).

    The addendum fixes only the arguments of f, not its form. The form used here is:

        ProofLevel = max( Minimum(use),
                          max{ rule.level : rule triggered by (Risk, Rev, BlastRadius, U) },
                          insufficient_authority_level  if rank(Authority) < rank(required_authority[use]) )

    Minimum(use) is the §3.1 table. Thresholds, the ordered authority scale, the authority each use
    requires and the level forced by insufficient authority are policy — there are no defaults.
    Authority not on the scale is treated as the lowest possible (fail-closed).
    """

    escalation_rules: tuple[ProofEscalationRule, ...]
    #: ordered lowest → highest, e.g. ("observer", "operator", "supervisor", "safety_officer")
    authority_scale: tuple[str, ...]
    #: authority required for each claim/use; must cover every ClaimUse
    required_authority: Mapping[ClaimUse, str]
    insufficient_authority_level: ProofLevel

    def __post_init__(self) -> None:
        rules = tuple(self.escalation_rules)
        if any(not isinstance(rule, ProofEscalationRule) for rule in rules):
            raise TypeError("escalation_rules must be ProofEscalationRule instances")
        object.__setattr__(self, "escalation_rules", rules)
        scale = tuple(str(item).strip() for item in self.authority_scale)
        if not scale or any(not item for item in scale) or len(set(scale)) != len(scale):
            raise ValueError("authority_scale must be a non-empty ordered list of distinct authority names")
        object.__setattr__(self, "authority_scale", scale)
        required = {ClaimUse(use): str(level).strip() for use, level in dict(self.required_authority).items()}
        missing = [use.value for use in ClaimUse if use not in required]
        if missing:
            raise ValueError(f"required_authority must cover every claim use; missing {missing}")
        unknown = sorted({level for level in required.values() if level not in scale})
        if unknown:
            raise ValueError(f"required_authority names authorities not on the scale: {unknown}")
        object.__setattr__(self, "required_authority", MappingProxyType(required))
        object.__setattr__(self, "insufficient_authority_level", ProofLevel(self.insufficient_authority_level))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ProofObligationConfig":
        """Build from settings, e.g. ``settings.PROOF_OBLIGATION_POLICY`` (all keys required)::

            {"escalation_rules": [{"level": 3, "risk_at_least": ..., "reversibility_below": ...,
                                   "blast_radius_above": ..., "uncertainty_at_least": ...}, ...],
             "authority_scale": ["observer", "operator", "supervisor"],
             "required_authority": {"ui_explanation": "observer", ...},
             "insufficient_authority_level": 4}
        """
        data = dict(payload)
        return cls(
            escalation_rules=tuple(ProofEscalationRule(**dict(rule)) for rule in data["escalation_rules"]),
            authority_scale=tuple(data["authority_scale"]),
            required_authority={ClaimUse(key): value for key, value in dict(data["required_authority"]).items()},
            insufficient_authority_level=ProofLevel(int(data["insufficient_authority_level"])),
        )


@dataclass(frozen=True)
class ProofInputs:
    """Arguments of f in ProofLevel(q, use) = f(Risk, Reversibility, BlastRadius, Uncertainty, Authority)."""

    risk: float
    reversibility: float
    blast_radius: float
    uncertainty: float
    authority: str


class ProofObligationPolicy:
    def __init__(self, config: ProofObligationConfig) -> None:
        if not isinstance(config, ProofObligationConfig):
            raise TypeError("ProofObligationPolicy requires an explicit ProofObligationConfig (no defaults)")
        self.config = config

    def authority_rank(self, authority: Any) -> Optional[int]:
        name = authority.strip() if isinstance(authority, str) else ""
        return self.config.authority_scale.index(name) if name in self.config.authority_scale else None

    def evaluate(
        self,
        *,
        claim_use: ClaimUse,
        risk: float,
        reversibility: float,
        blast_radius: float,
        uncertainty: float,
        authority: str,
    ) -> FormulaResult:
        """Fail-closed: non-finite or out-of-range inputs raise; unknown authority ranks lowest."""
        use = ClaimUse(claim_use)
        minimum = _MINIMUM[use]
        risk = _finite("risk", risk)
        reversibility = _finite("reversibility", reversibility)
        uncertainty = _finite("uncertainty", uncertainty)
        radius = _finite("blast_radius", blast_radius)
        for name, value in (("risk", risk), ("reversibility", reversibility), ("uncertainty", uncertainty)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if radius < 0:
            raise ValueError("blast_radius must be non-negative")
        triggered = [
            rule.level for rule in self.config.escalation_rules
            if rule.triggered(risk=risk, reversibility=reversibility, blast_radius=radius, uncertainty=uncertainty)
        ]
        escalation = max(triggered) if triggered else minimum
        rank = self.authority_rank(authority)
        required_name = self.config.required_authority[use]
        required_rank = self.config.authority_scale.index(required_name)
        authority_sufficient = rank is not None and rank >= required_rank
        authority_level = minimum if authority_sufficient else self.config.insufficient_authority_level
        level = max(minimum, escalation, authority_level)
        return FormulaResult(
            value=level,
            formula=PROOF_LEVEL_FORMULA,
            intermediates={
                "minimum_for_use": minimum,
                "risk_escalation": escalation if triggered else None,
                "authority": authority if isinstance(authority, str) else None,
                "authority_rank": rank,
                "required_authority": required_name,
                "authority_sufficient": authority_sufficient,
            },
        )

    def required_level(self, **kwargs: Any) -> ProofLevel:
        return ProofLevel(self.evaluate(**kwargs).value)

    def required_level_for(self, claim_use: ClaimUse, inputs: ProofInputs) -> ProofLevel:
        return self.required_level(
            claim_use=claim_use, risk=inputs.risk, reversibility=inputs.reversibility,
            blast_radius=inputs.blast_radius, uncertainty=inputs.uncertainty, authority=inputs.authority,
        )



# ---------------------------------------------------------------- совместимость со старым API
# Прежний класс порогов команды оставлен для импорта и чтения. ProofObligationPolicy его НЕ использует:
# пороги эскалации приходят только из конфигурации политики (без значений по умолчанию).
@dataclass(frozen=True)
class ProofEscalationThresholds:
    """Team-configured escalation thresholds (the addendum gives only f(...))."""

    mechanism_risk: float = 0.5
    mechanism_uncertainty: float = 0.5
    mechanism_blast_radius: int = 1
    mechanism_reversibility: float = 0.5
    safety_risk: float = 0.8
    safety_uncertainty: float = 0.8
    safety_blast_radius: int = 10
    safety_reversibility: float = 0.2

    def __post_init__(self) -> None:
        for name in (
            "mechanism_risk",
            "mechanism_uncertainty",
            "mechanism_reversibility",
            "safety_risk",
            "safety_uncertainty",
            "safety_reversibility",
        ):
            value = _finite(name, getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.mechanism_blast_radius < 0 or self.safety_blast_radius < 0:
            raise ValueError("blast radius thresholds must be non-negative")

__all__ = [
    "ABSTENTION_FORMULA",
    "PROOF_LEVEL_FORMULA",
    "ClaimUse",
    "ProofElement",
    "ProofEscalationRule",
    "ProofInputs",
    "ProofLevel",
    "ProofObligationConfig",
    "ProofObligationPolicy",
    "achieved_proof_level",
    "level_elements",
    "required_proof_elements",
    "satisfied_levels",
    "should_abstain",
]

