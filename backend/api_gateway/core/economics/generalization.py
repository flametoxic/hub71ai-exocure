"""Обобщение без самообмана (§4–§8): аналогии, формальные доказательства, цели, улучшение возможностей, Genesis.

§4 Структурная аналогия: отображение φ графа-источника в граф-цель обязано сохранять выбранные связи
    (u,v) ∈ E_S ⇒ (φ(u),φ(v)) ∈ E_T и совместимость механизмов M_T(φ(u),φ(v)) ≈ Ψ(M_S(u,v)) в пределах допуска
    политики. Результат — шаблон гипотезы со ступенью structural_possible, не перенесённая истина.
§5 Genesis-цикл для нефизических доменов: эксперимент в коде — только в песочнице; права на физическое
    исполнение результат не даёт.
§6 Формальное открытие: A ⊢ φ проверяется символически (SymPy: A ∧ ¬φ невыполнима). «Следует из аксиом» ≠
    «верно в мире»: операционным правилом становится только после связи с эмпирическим механизмом, повтора
    и проверки человеком, с областью применимости.
§7 Цели: Utility(g) = V_human + V_knowledge + V_service − λ_r·Risk − λ_p·PolicyViolation − … (последний член в PDF
    обрезан — не выдумываем). Любое нарушение политики — отказ, любопытство его не перевешивает; физическая
    цель без одобрения — отказ.
§8 Улучшение возможностей: ΔCapability > τ_gain ∧ ΔSafety ≥ 0 ∧ ΔReliability ≥ −ε на одних и тех же задачах
    бенчмарка; итог — только право на канарейку, развёртывание — через конвейер обучения (никогда само).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from ._base import F_CAPABILITY, F_MAPPING, F_PROOF, F_UTILITY, EconomicsError, number, result, text, texts

# ------------------------------------------------------------------------------------------ §4 аналогии
MOTIFS: Mapping[str, tuple[tuple[str, str, str], ...]] = {
    "source_bottleneck_degradation": (("source", "flows_to", "bottleneck"), ("bottleneck", "degrades", "service")),
    "feedback_delay_saturation": (("controller", "actuates", "plant"), ("plant", "measured_by", "sensor"),
                                  ("sensor", "feeds_back", "controller")),
    "flow_capacity_spillback": (("resource", "flows_to", "capacity"), ("capacity", "queues", "queue"),
                                ("queue", "spills_back", "resource")),
    "latent_fault_cascade": (("fault", "emits", "weak_signal"), ("fault", "propagates_to", "downstream"),
                             ("downstream", "cascades_to", "failure")),
}


@dataclass(frozen=True)
class HypothesisTemplate:
    motif: str
    mapping: Mapping[str, str]
    preserved: tuple[tuple[str, str, str], ...]
    missing: tuple[tuple[str, str, str], ...]
    mechanism_errors: Mapping[str, float]
    admissible: bool
    lifecycle: str = "structural_possible"
    requires: str = "target evidence and calibration"


def map_structure(motif: str, source_edges: Sequence[tuple[str, str, str]], target_edges: Sequence[tuple[str, str, str]],
                  mapping: Mapping[str, str], *, relations: Sequence[str],
                  source_mechanisms: Optional[Mapping[tuple[str, str], float]] = None,
                  target_mechanisms: Optional[Mapping[tuple[str, str], float]] = None,
                  psi: Optional[Callable[[float], float]] = None, rel_tolerance: Optional[float] = None):
    """φ: G_S → G_T. Каждое ребро выбранных типов обязано отобразиться в ребро цели того же типа."""
    rels = set(texts("relations", relations))
    T = {(a, r, b) for a, r, b in target_edges}
    preserved, missing = [], []
    for a, r, b in source_edges:
        if r not in rels:
            continue
        if a not in mapping or b not in mapping:
            missing.append((a, r, b))
            continue
        (preserved if (mapping[a], r, mapping[b]) in T else missing).append((a, r, b))
    errors = {}
    if source_mechanisms is not None:
        if target_mechanisms is None or psi is None or rel_tolerance is None:
            raise EconomicsError("mechanism compatibility needs target mechanisms, Ψ and a tolerance")
        tol = number("rel_tolerance", rel_tolerance, low=0.0)
        for (a, b), ms in source_mechanisms.items():
            key = (mapping.get(a), mapping.get(b))
            if key not in target_mechanisms:
                errors[f"{a}->{b}"] = float("inf")
                continue
            pred = float(psi(float(ms)))
            errors[f"{a}->{b}"] = abs(float(target_mechanisms[key]) - pred) / max(abs(pred), np.finfo(float).tiny)
        bad = {k: v for k, v in errors.items() if v > tol}
    else:
        bad = {}
    ok = not missing and not bad and bool(preserved)
    tpl = HypothesisTemplate(text("motif", motif), dict(mapping), tuple(preserved), tuple(missing), errors, ok)
    return result(tpl, F_MAPPING, admissible=ok)


# ------------------------------------------------------------------------------------------ §5 Genesis
GENESIS_STEPS = ("observe", "represent", "hypothesize", "derive_prediction", "run_experiment", "verify_outcome",
                 "update_knowledge")
NONPHYSICAL_DOMAINS = ("code", "economics", "law_policy")


def genesis_experiment_gate(domain: str, *, sandboxed: bool) -> str:
    """Физический домен идёт в планирование (CPR) с его гейтами; нефизический — только в песочнице."""
    d = text("domain", domain)
    if d == "physical_asset":
        return "route_to_constrained_planning"
    if d not in NONPHYSICAL_DOMAINS:
        raise EconomicsError(f"unknown Genesis domain {d!r}")
    if sandboxed is not True:
        raise EconomicsError("non-physical experiments run only in a sandbox (§5)")
    return "sandbox_only_no_physical_authority"


# ------------------------------------------------------------------------------------------ §6 формальное
FORMAL_STAGES = ("candidate", "formalized", "proved_in_axioms", "empirically_linked", "replayed", "human_reviewed",
                 "versioned_rule")


@dataclass
class Conjecture:
    conjecture_id: str
    statement: str
    scope: tuple[str, ...]
    stage: str = "candidate"
    proof: Optional[dict] = None
    evidence: tuple[str, ...] = ()
    history: tuple[tuple[str, str], ...] = ()

    def _advance(self, to: str, note: str) -> None:
        if FORMAL_STAGES.index(to) != FORMAL_STAGES.index(self.stage) + 1:
            raise EconomicsError(f"stage {to} out of order after {self.stage}")
        self.stage = to
        self.history += ((to, note),)

    def formalize(self, axioms: Sequence[str], phi: str) -> None:
        import sympy

        try:
            self._axioms = [sympy.sympify(a) for a in axioms]
            self._phi = sympy.sympify(phi)
        except (sympy.SympifyError, TypeError) as exc:
            raise EconomicsError(f"formalization failed: {exc}") from exc
        self._advance("formalized", f"{len(self._axioms)} axioms")

    def prove(self):
        """A ⊢ φ ⇔ A ∧ ¬φ невыполнима. Иначе — контрпример (модель)."""
        from sympy import And, Not
        from sympy.logic.inference import satisfiable

        if self.stage != "formalized":
            raise EconomicsError("formalize before proving")
        model = satisfiable(And(*self._axioms, Not(self._phi)))
        if model is False:
            self.proof = {"follows_from_axioms": True, "meaning": "follows from the stated axioms, not empirical truth"}
            self._advance("proved_in_axioms", "A ∧ ¬φ unsatisfiable")
        else:
            self.proof = {"follows_from_axioms": False, "counterexample": {str(k): bool(v) for k, v in model.items()}}
        return result(self.proof, F_PROOF)

    def link_empirical(self, outcome_refs: Sequence[str]) -> None:
        refs = texts("outcome_refs", outcome_refs)
        if not refs:
            raise EconomicsError("an operational rule needs empirical validation (verified outcomes)")
        self.evidence = refs
        self._advance("empirically_linked", ",".join(refs))

    def replay(self, replay_ref: str) -> None:
        self._advance("replayed", text("replay_ref", replay_ref))

    def review(self, reviewer: str, approved: bool) -> None:
        if approved is not True:
            raise EconomicsError("rejected by human/domain review")
        self._advance("human_reviewed", text("reviewer", reviewer))

    def version(self, version: str) -> dict:
        if not self.scope:
            raise EconomicsError("a versioned rule needs an explicit scope")
        self._advance("versioned_rule", text("version", version))
        return {"rule_id": self.conjecture_id, "version": version, "scope": list(self.scope),
                "statement": self.statement, "evidence": list(self.evidence)}


# ------------------------------------------------------------------------------------------ §7 цели
GOAL_CLASSES = ("research", "observation", "diagnostic", "physical")


@dataclass(frozen=True)
class GoalCandidate:
    goal_id: str
    goal_class: str
    purpose: str
    beneficiary: str
    v_human: float
    v_knowledge: float                 # сюда входит любопытство — и только сюда
    v_service: float
    risk: float
    policy_violation: float
    resource_budget: float
    allowed_tools: tuple[str, ...]
    expiry: datetime
    approval_ref: Optional[str] = None


def goal_utility(g: GoalCandidate, *, lambda_risk: float, lambda_policy: float):
    lr, lp = number("lambda_risk", lambda_risk, low=0.0), number("lambda_policy", lambda_policy, low=0.0)
    u = (number("v_human", g.v_human) + number("v_knowledge", g.v_knowledge) + number("v_service", g.v_service)
         - lr * number("risk", g.risk, low=0.0) - lp * number("policy_violation", g.policy_violation, low=0.0))
    return result(u, F_UTILITY, truncated_term_not_applied="lambda_c C…", terms={"lambda_risk": lr,
                                                                                    "lambda_policy": lp})


def admit_goal(g: GoalCandidate, *, now: datetime, lambda_risk: float, lambda_policy: float) -> tuple[bool, tuple[str, ...]]:
    reasons = []
    if g.goal_class not in GOAL_CLASSES:
        reasons.append("unknown_goal_class")
    for name in ("purpose", "beneficiary"):
        if not str(getattr(g, name) or "").strip():
            reasons.append(f"{name}_missing")
    if not g.allowed_tools:
        reasons.append("allowed_tools_missing")
    if g.resource_budget <= 0:
        reasons.append("resource_budget_missing")
    if g.expiry <= now:
        reasons.append("expired")
    if g.policy_violation > 0:
        reasons.append("policy_violation: never outweighed by curiosity or value")
    if g.goal_class == "physical" and not g.approval_ref:
        reasons.append("physical_goal_requires_human_or_policy_approval")
    if not reasons and goal_utility(g, lambda_risk=lambda_risk, lambda_policy=lambda_policy).value <= 0:
        reasons.append("non_positive_utility")
    return not reasons, tuple(reasons)


# ------------------------------------------------------------------------------------------ §8 возможности
def capability_gate(base: Mapping[str, Mapping[str, float]], new: Mapping[str, Mapping[str, float]], *,
                    tau_gain: float, epsilon: float):
    """base/new: задача бенчмарка → {"capability", "safety", "reliability"} на ОДНИХ и тех же задачах."""
    if set(base) != set(new) or not base:
        raise EconomicsError("the candidate must be evaluated on exactly the baseline benchmark tasks (replay)")
    d = {k: float(np.mean([new[t][k] - base[t][k] for t in base])) for k in ("capability", "safety", "reliability")}
    ok = d["capability"] > number("tau_gain", tau_gain) and d["safety"] >= 0.0 and \
        d["reliability"] >= -number("epsilon", epsilon, low=0.0)
    return result({"eligible_for_canary": ok, "self_deploy": False, **d}, F_CAPABILITY,
                  next_step="learning pipeline: shadow → approval (not the author) → canary → versioned deploy"
                  if ok else "rejected")


__all__ = ["Conjecture", "FORMAL_STAGES", "GENESIS_STEPS", "GOAL_CLASSES", "GoalCandidate", "HypothesisTemplate",
           "MOTIFS", "NONPHYSICAL_DOMAINS", "admit_goal", "capability_gate", "genesis_experiment_gate", "goal_utility",
           "map_structure"]
