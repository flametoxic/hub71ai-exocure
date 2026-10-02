"""Общество гипотез (§3): специализированные агенты исследуют параллельно, но истину не пишут.

Каждый агент выдаёт подписанное утверждение (claim_id, тип, область, свидетельства, допущения, контрпримеры,
уверенность, версия модели). Подпись — HMAC-SHA256 ключом агента из реестра: подделанное или изменённое
утверждение отвергается. Агрегация — не подсчёт голосов:
    Score(q) = Σ_i w_i·Support_i(q) − Σ_j v_j·Contradiction_j(q),
вес зависит от независимости свидетельств, калибровки источника, физической согласованности и применимости
области: w = confidence · calibration · physics · scope / (число утверждений на тех же свидетельствах).
Десять агентов, пересказавших одно наблюдение, весят как одно. Граф обоснований сохраняется (для TMS).
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

from ._base import F_SCORE, EconomicsError, number, result, text, texts

AGENT_KINDS = ("physics", "causal_discovery", "maintenance", "spatial_topology", "policy",
               "counterfactual_simulation", "operator_explanation")


@dataclass(frozen=True)
class AgentClaim:
    claim_id: str
    agent_id: str
    claim_type: str
    question: str
    stance: str                         # "support" | "contradict"
    scope: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    assumptions: tuple[str, ...]
    counterexamples: tuple[str, ...]
    confidence: float
    model_version: str
    signature: str = ""

    def payload(self) -> bytes:
        body = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "signature"}
        return json.dumps(body, sort_keys=True, default=list).encode("utf-8")

    def __post_init__(self) -> None:
        for n in ("claim_id", "agent_id", "claim_type", "question", "model_version"):
            text(n, getattr(self, n))
        if self.stance not in ("support", "contradict"):
            raise EconomicsError("stance must be 'support' or 'contradict'")
        if not texts("evidence_refs", self.evidence_refs):
            raise EconomicsError("an agent claim must cite evidence")
        number("confidence", self.confidence, low=0.0, high=1.0)


class AgentKeyRegistry:
    def __init__(self) -> None:
        self._keys: dict[str, tuple[str, bytes]] = {}

    def register(self, agent_id: str, kind: str, key: bytes) -> None:
        if kind not in AGENT_KINDS:
            raise EconomicsError(f"agent kind must be one of {AGENT_KINDS}")
        if not isinstance(key, bytes) or len(key) < 16:
            raise EconomicsError("agent key must be at least 16 bytes")
        self._keys[text("agent_id", agent_id)] = (kind, key)

    def sign(self, claim: AgentClaim, key: bytes) -> AgentClaim:
        sig = hmac.new(key, claim.payload(), hashlib.sha256).hexdigest()
        return AgentClaim(**{**{k: getattr(claim, k) for k in claim.__dataclass_fields__}, "signature": sig})

    def verify(self, claim: AgentClaim) -> bool:
        entry = self._keys.get(claim.agent_id)
        if entry is None or not claim.signature:
            return False
        expected = hmac.new(entry[1], claim.payload(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, claim.signature)


def aggregate(question: str, claims: Sequence[AgentClaim], *, registry: AgentKeyRegistry,
              source_calibration: Mapping[str, float], physics_consistency: Mapping[str, float],
              scope_validity: Mapping[str, float]):
    """Score(q) с весами по независимости, калибровке, физике и области. Неподписанные — отвергаются."""
    q = text("question", question)
    rejected, used = [], []
    for c in claims:
        if c.question != q:
            continue
        if not registry.verify(c):
            rejected.append((c.claim_id, "signature_invalid"))
            continue
        missing = [n for n, m in (("source_calibration", source_calibration), ("physics_consistency",
                   physics_consistency), ("scope_validity", scope_validity)) if c.claim_id not in m]
        if missing:
            rejected.append((c.claim_id, f"missing_weight_inputs:{missing}"))
            continue
        used.append(c)
    shared: dict[frozenset, int] = {}
    for c in used:
        shared[frozenset(c.evidence_refs)] = shared.get(frozenset(c.evidence_refs), 0) + 1
    support = contra = 0.0
    weights, graph = {}, {}
    for c in used:
        w = (c.confidence * number("calibration", source_calibration[c.claim_id], low=0.0, high=1.0)
             * number("physics", physics_consistency[c.claim_id], low=0.0, high=1.0)
             * number("scope", scope_validity[c.claim_id], low=0.0, high=1.0)) / shared[frozenset(c.evidence_refs)]
        weights[c.claim_id] = w
        graph[c.claim_id] = {"stance": c.stance, "evidence": list(c.evidence_refs), "assumptions": list(c.assumptions),
                             "counterexamples": list(c.counterexamples), "agent": c.agent_id}
        if c.stance == "support":
            support += w
        else:
            contra += w
    return result(support - contra, F_SCORE, support=support, contradiction=contra, weights=weights,
                  justification_graph=graph, rejected=tuple(rejected), aggregation="weighted_not_vote",
                  epistemic_status="hypothesis")


__all__ = ["AGENT_KINDS", "AgentClaim", "AgentKeyRegistry", "aggregate"]
