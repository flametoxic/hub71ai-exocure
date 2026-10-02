from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np
from ..world_model.reality_formulas import (
    FormulaReference,
    FormulaResult,
    body_capability,
    information_action_value,
    is_strict_true,
)

UMRM = "CURE-Unified-Mathematical-Reality-Model"
CEOS = "EXO-causal-engine-operational-specification"
F_EIG = FormulaReference("UMRM-8-EIG-EXPECTED", UMRM, 7, "EIG(a) = H(B_t) - E_o~P(o|a)[H(B_t+1|o,a)]", output_unit="bits")
F_IG = FormulaReference("CEOS-12-INFORMATION-GAIN", CEOS, 8, "IG(a) = H(H|D) - E_o~P(o|a,D)[H(H|D,o,a)]", output_unit="bits")
F_VOI = FormulaReference("CEOS-12-VOI", CEOS, 8, "VOI(a) = IG(a) - C(a) - lambda_r R(a) - lambda_p PrivacyCost(a)")
F_ACTION = FormulaReference("UMRM-8-PHYSICAL-ACTION", UMRM, 7, "a* = argmax_a U(a | B_t, C_t, M_t) subject to constraints")
F_GATE = FormulaReference("UMRM-9-ACTION-GATE", UMRM, 7, "U_epistemic > theta_action_class => DEFER")
F_INFO_SELECT = FormulaReference("UMRM-8-INFORMATION-ACTION-SELECT", UMRM, 7,
                                 "a* = argmax_a [EIG(a) - lambda_c Cost(a) - lambda_r Risk(a) - lambda_t Latency(a)]")
F_PLAN_CHOICE = FormulaReference("UMRM-8-PHYSICAL-OR-INFORMATION", UMRM, 7,
                                 "A plan chooses either a physical action or an information action; "
                                 "U_epistemic > theta_action_class => DEFER (p.7, section 9)")


def _h_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def expected_information_gain_discrete(*, prior, likelihood) -> FormulaResult:
    """likelihood[s][o] = P(o | s, a). Считает само ожидание, а не принимает его готовым."""
    b, lik = np.asarray(prior, float), np.asarray(likelihood, float)
    if b.ndim != 1 or np.any(b < 0) or not np.isclose(b.sum(), 1):
        raise ValueError("prior must be a distribution")
    if lik.ndim != 2 or lik.shape[0] != b.size or np.any(lik < 0) or not np.allclose(lik.sum(1), 1):
        raise ValueError("likelihood rows must be distributions over observations")
    p_obs = b @ lik
    expected = sum(po * _h_bits(b * lik[:, o] / po) for o, po in enumerate(p_obs) if po > 0)
    h0 = _h_bits(b)
    return FormulaResult(max(h0 - expected, 0.0), F_EIG, {"prior_entropy": h0,
                                                         "expected_posterior_entropy": expected,
                                                         "p_observation": p_obs})


def hypothesis_information_gain(*, hypothesis_prior: Mapping[str, float],
                                observation_likelihood: Mapping[str, Mapping[str, float]]) -> FormulaResult:
    """Какое разрешённое измерение сильнее всего разделит H1 и H2."""
    hs = sorted(hypothesis_prior)
    obs = sorted({o for h in hs for o in observation_likelihood[h]})
    r = expected_information_gain_discrete(prior=[hypothesis_prior[h] for h in hs],
                                           likelihood=[[observation_likelihood[h].get(o, 0.0) for o in obs] for h in hs])
    return FormulaResult(r.value, F_IG, {**r.intermediates, "hypotheses": tuple(hs), "observations": tuple(obs)})


def causal_observation_value(*, information_gain, cost, risk, privacy_cost, risk_weight, privacy_weight) -> FormulaResult:
    """C(a) — в тех же нормированных единицах, что и IG."""
    for name, v in (("cost", cost), ("risk", risk), ("privacy_cost", privacy_cost),
                    ("risk_weight", risk_weight), ("privacy_weight", privacy_weight)):
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"{name} must be non-negative")
    terms = {"cost": cost, "risk": risk_weight * risk, "privacy": privacy_weight * privacy_cost}
    return FormulaResult(float(information_gain) - sum(terms.values()), F_VOI, terms)


ACTION_CONSTRAINTS = ("physical", "spatial_topological", "policy_scope", "consent", "capability",
                      "uncertainty", "safety_assurance")


@dataclass(frozen=True)
class ActionCandidate:
    action_id: str
    utility: float  # U(a | B_t, C_t, M_t)
    constraints: Mapping[str, bool]


def select_physical_action(candidates: Sequence[ActionCandidate]) -> FormulaResult:
    """Нет допустимых → DEFER, с причиной по каждому отклонённому варианту."""
    rejected, feasible = {}, []
    for c in candidates:
        failed = tuple(k for k in ACTION_CONSTRAINTS if not is_strict_true(c.constraints.get(k)))  # строго bool True
        if failed or not np.isfinite(c.utility):
            rejected[c.action_id] = failed or ("non_finite_utility",)
        else:
            feasible.append(c)
    best = sorted(feasible, key=lambda c: (-c.utility, c.action_id))[0] if feasible else None
    return FormulaResult(best.action_id if best else None, F_ACTION,
                         {"decision": "select" if best else "defer", "rejected": rejected})


def epistemic_action_gate(*, epistemic_uncertainty: float, action_class_threshold: float,
                          missing_observations: Sequence[str]) -> FormulaResult:
    """UMRM 9 (p.7): U_epistemic > theta_action_class => DEFER.

    The single implementation of the gate (``reality_formulas.epistemic_action_gate`` and
    ``choose_physical_or_information_action`` both go through it).  Fail-closed: U and theta must be
    finite non-negative real numbers (NaN/inf/bool -> ValueError, never PROCEED); the missing
    observations must be a sequence of non-empty names.  The system must name which observation or
    capability would remove the uncertainty.
    """
    for name, value in (("epistemic_uncertainty", epistemic_uncertainty),
                        ("action_class_threshold", action_class_threshold)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a finite real number")
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if isinstance(missing_observations, (str, bytes)) or not isinstance(missing_observations, Sequence):
        raise TypeError("missing_observations must be a sequence of names, not a single string")
    names = tuple(missing_observations)
    if any(not isinstance(item, str) or not item.strip() for item in names):
        raise ValueError("missing_observations must contain non-empty names")
    defer = float(epistemic_uncertainty) > float(action_class_threshold)
    if defer and not names:
        raise ValueError("DEFER must name the missing observation or capability")
    return FormulaResult("DEFER" if defer else "PROCEED", F_GATE,
                         {"missing_observations": names if defer else ()})


MOE_CHECKS = ("schema", "ontology", "constraints", "physics", "safety", "policy")


def accept_moe_candidate(checks: Mapping[str, bool]) -> FormulaResult:
    """Mapping-form alias of the single Accept(c) (UMRM 11) -- ``reality_formulas.accept_moe_candidate``.

    No second implementation: a missing check is ``None`` and fails the strict-bool gate there.
    Принятый кандидат остаётся кандидатом: фактом он становится только через ProjectionPipeline."""
    from ..world_model.reality_formulas import accept_moe_candidate as _accept

    return _accept(**{f"{k}_valid": checks.get(k) for k in MOE_CHECKS})


def action_capability(**factors) -> FormulaResult:
    """UMRM 4.4: Capability(action,t) -- the single form ``reality_formulas.body_capability``.

    Kept as an alias so that reasoning code has no second, divergent capability formula."""
    return body_capability(**factors)


@dataclass(frozen=True)
class ObservationCandidate:
    observation_id: str
    information_gain_bits: float
    cost: float
    risk: float
    privacy_cost: float
    consent_granted: bool  # нет явного согласия → наблюдение не предлагается (CEOS стр. 8)


def propose_observations(candidates: Sequence[ObservationCandidate], *, n_hypotheses: int, risk_weight: float,
                         privacy_weight: float) -> FormulaResult:
    """Только с согласием, только VOI > 0, IG не больше log2(#гипотез)."""
    if n_hypotheses < 2:
        raise ValueError("information gain needs at least two competing hypotheses")
    max_ig = float(np.log2(n_hypotheses))
    ranked, rejected = [], {}
    for c in candidates:
        if c.consent_granted is not True:
            rejected[c.observation_id] = "no_consent"
            continue
        if not 0 <= c.information_gain_bits <= max_ig + 1e-9:
            rejected[c.observation_id] = "information_gain_out_of_bounds"
            continue
        voi = causal_observation_value(information_gain=c.information_gain_bits, cost=c.cost, risk=c.risk,
                                       privacy_cost=c.privacy_cost, risk_weight=risk_weight,
                                       privacy_weight=privacy_weight).value
        if voi <= 0:
            rejected[c.observation_id] = "non_positive_voi"
            continue
        ranked.append((voi, c.observation_id))
    ranked.sort(key=lambda t: (-t[0], t[1]))
    return FormulaResult(tuple(i for _, i in ranked), F_VOI, {"voi": dict((i, v) for v, i in ranked),
                                                              "rejected": rejected})



# ---------------------------------------------------------------------------------------------
# UMRM 8 (page 7): information actions and the choice "physical action vs information action".
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class InformationActionPolicy:
    """lambda_c, lambda_r, lambda_t of UMRM 8 -- from policy, required (the TZ gives no values)."""
    policy_version: str
    cost_weight: float
    risk_weight: float
    latency_weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise ValueError("policy_version is required")
        for name in ("cost_weight", "risk_weight", "latency_weight"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) \
                    or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number from policy")


@dataclass(frozen=True)
class InformationActionCandidate:
    """An observation / sensor / operator / micro-experiment choice.  ``eig`` must be the result of
    an EIG computation (UMRM-8-EIG*), not a free number."""
    action_id: str
    eig: FormulaResult
    cost: float
    risk: float
    latency: float
    constraints: Mapping[str, bool]


_EIG_FORMULAS = ("UMRM-8-EIG", "UMRM-8-EIG-EXPECTED")


def select_information_action(candidates: Sequence[InformationActionCandidate], *,
                              policy: InformationActionPolicy) -> FormulaResult:
    """a* = argmax_a [EIG(a) - lambda_c Cost(a) - lambda_r Risk(a) - lambda_t Latency(a)] (UMRM 8, p.7).

    Information actions are actions: they pass the same constraint classes as physical ones
    (ACTION_CONSTRAINTS, strict bool True each).  No feasible candidate -> ``DEFER``.  Ties -> action_id.
    """
    if not isinstance(policy, InformationActionPolicy):
        raise TypeError("policy (lambda_c, lambda_r, lambda_t) is required")
    rejected, scored = {}, []
    for c in candidates:
        if not isinstance(c, InformationActionCandidate):
            raise TypeError("candidates must be InformationActionCandidate")
        if not isinstance(c.eig, FormulaResult) or c.eig.formula.formula_id not in _EIG_FORMULAS:
            rejected[c.action_id] = ("eig_not_computed",)
            continue
        failed = tuple(k for k in ACTION_CONSTRAINTS if not is_strict_true(c.constraints.get(k)))
        if failed:
            rejected[c.action_id] = failed
            continue
        value = information_action_value(information_gain=c.eig.value, cost=c.cost, risk=c.risk, latency=c.latency,
                                         cost_weight=policy.cost_weight, risk_weight=policy.risk_weight,
                                         latency_weight=policy.latency_weight)
        scored.append((float(value.value), c.action_id, value.intermediates))
    scored.sort(key=lambda t: (-t[0], t[1]))
    best = scored[0] if scored else None
    return FormulaResult(best[1] if best else None, F_INFO_SELECT, {
        "decision": "select" if best else "defer",
        "values": {i: v for v, i, _ in scored},
        "terms": {i: t for _, i, t in scored},
        "rejected": rejected,
        "policy_version": policy.policy_version,
    })


def choose_physical_or_information_action(*, physical_candidates: Sequence[ActionCandidate],
                                          information_candidates: Sequence[InformationActionCandidate],
                                          information_policy: InformationActionPolicy,
                                          epistemic_uncertainty: float, action_class_threshold: float,
                                          missing_observations: Sequence[str]) -> FormulaResult:
    """UMRM 8: "A plan chooses either a physical action or an information action."

    The TZ does not define an exchange rate between U(a|B,C,M) and the information value, so the two
    are not compared numerically.  The choice is made by the UMRM 9 action gate:
    U_epistemic <= theta -> the best feasible physical action (``select_physical_action``);
    U_epistemic > theta (DEFER) or no feasible physical action -> the best information action
    (``select_information_action``), which is how the missing observation is acquired.  If neither
    exists the result is ``DEFER`` naming the missing observations.  Never a real action.
    """
    gate = epistemic_action_gate(epistemic_uncertainty=epistemic_uncertainty,
                                 action_class_threshold=action_class_threshold,
                                 missing_observations=missing_observations)
    physical = select_physical_action(list(physical_candidates))
    information = select_information_action(list(information_candidates), policy=information_policy)
    if gate.value == "PROCEED" and physical.value is not None:
        kind, chosen, reason = "physical", physical.value, "epistemic_gate_proceed"
    elif information.value is not None:
        kind, chosen = "information", information.value
        reason = "epistemic_gate_defer" if gate.value == "DEFER" else "no_feasible_physical_action"
    else:
        kind, chosen, reason = "defer", None, "no_feasible_action"
    return FormulaResult(chosen, F_PLAN_CHOICE, {
        "kind": kind,
        "reason": reason,
        "gate": gate.value,
        "missing_observations": gate.intermediates["missing_observations"] or tuple(missing_observations),
        "physical": physical.intermediates,
        "information": information.intermediates,
        "real_action": False,
    })


# совместимость: ссылки на формулы, которые прежний модуль экспортировал (сами проверки — в reality_formulas)
F_ACCEPT = FormulaReference("UMRM-11-MOE-ACCEPT", UMRM, 8, "Accept(c) = Schema ∧ Ontology ∧ Constraints ∧ Physics ∧ Safety ∧ Policy")
F_CAPABILITY = FormulaReference("UMRM-4.4-CAPABILITY", UMRM, 4, "Capability(action,t) = f(body availability, actuator health, network, policy, scope)")
