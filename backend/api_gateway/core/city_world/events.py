"""City WM §8–§11: подписанный журнал событий, набор данных причинного движка, правила безопасности, дата-продукты.

§8  events — append-only и подписанные; causal_parent_event_id — не истина, а связь с типом доказательства:
    observed_temporal_predecessor | mechanistically_confirmed | intervention_confirmed | model_inferred | human_validated.
§9  интервенция — формальное do(X = x): P(Y | do(X = x), Z); конфаундеры обязательны;
    контрфактическая запись Y_{u'}(x) — прогноз результата при альтернативном действии u' (не наблюдение).
§10 safety_rules: жёсткие/мягкие, формальное выражение, кто может переопределить, действие при нарушении…
§11 world_state_snapshot, causal_timeseries — материализованные представления.
"""
from __future__ import annotations

import ast
import hashlib
import hmac
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

from ..spatial_world.journal import WorldJournal, canonical
from ..world_model.reality_formulas import FormulaReference
from .schema import DOC, CitySchemaError, is_uuid7, uuid7

F_DO = FormulaReference("CITY-9.1-DO", DOC, 12, "intervention = do(X = x): P(Y | do(X = x), Z)")
F_CF = FormulaReference("CITY-9.3-CF", DOC, 13, "Y_{u'}(x) — forecast of the outcome under the alternative action u'")
EVENT_CLASSES = ("state_change", "physical_movement", "transaction", "environmental_shift", "system_intervention")
CAUSAL_EVIDENCE = ("observed_temporal_predecessor", "mechanistically_confirmed", "intervention_confirmed",
                   "model_inferred", "human_validated")
CONFOUNDERS = ("holiday_effect_score", "weather_impact_score", "vip_movement_flag", "public_event_active",
               "construction_zone_flag", "sensor_coverage_score", "baseline_demand_forecast")
TRIGGER_SOURCES = ("predictive_model", "emergency_override", "human_operator")
RULE_DOMAINS = ("traffic", "energy", "privacy", "care", "airspace")


@dataclass(frozen=True)
class CityEvent:
    event_id: str
    event_timestamp_ns: int
    recorded_at: datetime
    event_type_class: str
    actor_entity_id: Optional[str]
    target_entity_id: str
    payload: Mapping[str, Any]
    schema_version: str
    causal_parent_event_id: Optional[str] = None
    causal_evidence_type: Optional[str] = None
    correlation_id: Optional[str] = None
    causal_confidence: Optional[float] = None
    anomaly_score: Optional[float] = None
    evidence_refs: tuple = ()
    policy_decision_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not is_uuid7(self.event_id):
            raise CitySchemaError("event_id must be UUIDv7")
        if self.event_type_class not in EVENT_CLASSES:
            raise CitySchemaError(f"event_type_class must be one of {EVENT_CLASSES}")
        if self.causal_parent_event_id is not None:
            if self.causal_evidence_type not in CAUSAL_EVIDENCE:
                raise CitySchemaError("a causal parent needs an evidence type — it is not established truth")
            if self.causal_confidence is None or not 0 <= self.causal_confidence <= 1:
                raise CitySchemaError("a causal parent needs a confidence in [0, 1] (no causal hallucination)")
        if self.anomaly_score is not None and not 0 <= self.anomaly_score <= 1:
            raise CitySchemaError("anomaly_score must be in [0, 1]")
        if not str(self.schema_version).strip():
            raise CitySchemaError("payload needs a schema version")


class CityEventLedger:
    """Append-only и подписанный: подпись HMAC ключом площадки поверх канонического тела события."""

    def __init__(self, journal: WorldJournal, *, key: bytes, key_id: str) -> None:
        if len(key) < 16:
            raise CitySchemaError("event signing key must be ≥ 16 bytes")
        self.journal, self._key, self.key_id = journal, key, key_id

    def append(self, e: CityEvent) -> dict:
        body = {k: v for k, v in e.__dict__.items()}
        sig = hmac.new(self._key, canonical(body).encode(), hashlib.sha256).hexdigest()
        row = self.journal.append("city_event", {"event": body, "signature": sig, "key_id": self.key_id},
                                  knowledge_time=e.recorded_at)
        return {"seq": row.seq, "signature": sig}

    def verify(self) -> bool:
        for r in self.journal.rows(("city_event",)):
            body = r.payload["event"]
            if not hmac.compare_digest(hmac.new(self._key, canonical(body).encode(), hashlib.sha256).hexdigest(),
                                       r.payload["signature"]):
                return False
        return self.journal.verify()

    def events(self) -> list[dict]:
        return [r.payload["event"] for r in self.journal.rows(("city_event",))]


@dataclass(frozen=True)
class Intervention:
    intervention_id: str
    decision_id: str
    target_system: str
    target_entity_id: str
    action_type: str
    action_magnitude: float
    trigger_source: str
    executed_at: datetime
    execution_status: str
    policy_rule_ids: tuple
    approval_id: Optional[str]
    pre_state_hash: str
    post_state_window_s: float
    confounders: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.trigger_source not in TRIGGER_SOURCES:
            raise CitySchemaError(f"trigger_source must be one of {TRIGGER_SOURCES}")
        missing = [c for c in CONFOUNDERS if c not in self.confounders]
        if missing:
            raise CitySchemaError(f"intervention without confounder context {missing}: selection bias would pass as effect")
        if not self.policy_rule_ids:
            raise CitySchemaError("an intervention cites the policy rules that allowed it")

    def as_do(self) -> dict:
        return {"do": {self.target_entity_id: {self.action_type: self.action_magnitude}},
                "Z": dict(self.confounders), "formula": F_DO}


@dataclass(frozen=True)
class CounterfactualRollout:
    decision_id: str
    simulated_alternate_action: str
    simulated_alternate_outcome_score: float
    risk_of_failure_pct: float
    model_version: str
    simulation_seed: int
    uncertainty_interval: tuple
    epistemic_status: str = "predicted"            # никогда не observed
    formula: Any = F_CF

    def __post_init__(self) -> None:
        if self.epistemic_status != "predicted":
            raise CitySchemaError("a counterfactual Y_{u'}(x) is a forecast, never an observation")
        lo, hi = self.uncertainty_interval
        if not lo <= self.simulated_alternate_outcome_score <= hi:
            raise CitySchemaError("the counterfactual point must lie inside its uncertainty interval")


# -------------------------------------------------------------------- правила безопасности
_ALLOWED_NODES = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.Compare, ast.Name,
                  ast.Constant, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.BinOp, ast.Add, ast.Sub,
                  ast.Load)


def _check_expr(expr: str) -> ast.Expression:
    tree = ast.parse(expr, mode="eval")
    for n in ast.walk(tree):
        if not isinstance(n, _ALLOWED_NODES):
            raise CitySchemaError(f"construct {type(n).__name__} not allowed in a formal safety expression")
    return tree


def _eval(node: ast.AST, env: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise KeyError(node.id)
        return env[node.id]
    if isinstance(node, ast.BoolOp):
        vals = [_eval(v, env) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.UnaryOp):
        return not _eval(node.operand, env)
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, env), _eval(node.right, env)
        return a + b if isinstance(node.op, ast.Add) else a - b
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, env)
            ok = {ast.Eq: left == right, ast.NotEq: left != right, ast.Lt: left < right, ast.LtE: left <= right,
                  ast.Gt: left > right, ast.GtE: left >= right}[type(op)]
            if not ok:
                return False
            left = right
        return True
    raise CitySchemaError("unsupported expression")


@dataclass(frozen=True)
class SafetyRule:
    rule_id: str
    rule_domain: str
    rule_name: str
    target_entity_type: str
    formal_expression: str               # «if A then B» пишется как  not (A) or (B)
    is_hard_constraint: bool
    who_can_override: tuple
    action_on_violation: str
    notification_recipient_role: str
    max_execution_delay_ms: float
    requires_human_approval: bool
    data_masking_required: bool
    rule_version: str
    valid_from: datetime
    valid_to: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.rule_domain not in RULE_DOMAINS:
            raise CitySchemaError(f"rule_domain must be one of {RULE_DOMAINS}")
        _check_expr(self.formal_expression)

    def evaluate(self, state: Mapping[str, Any]) -> dict:
        try:
            ok = bool(_eval(_check_expr(self.formal_expression), state))
            return {"rule_id": self.rule_id, "holds": ok, "hard": self.is_hard_constraint,
                    "action_on_violation": None if ok else self.action_on_violation}
        except KeyError as exc:                      # не хватает данных → жёсткое правило считается нарушенным
            return {"rule_id": self.rule_id, "holds": not self.is_hard_constraint, "hard": self.is_hard_constraint,
                    "missing": str(exc), "action_on_violation": self.action_on_violation if self.is_hard_constraint else None}


def check_rules(rules: Sequence[SafetyRule], state: Mapping[str, Any], *, at: datetime) -> dict:
    active = [r for r in rules if r.valid_from <= at and (r.valid_to is None or at < r.valid_to)]
    res = [r.evaluate(state) for r in active]
    hard_fail = [x for x in res if x["hard"] and not x["holds"]]
    return {"allowed": not hard_fail, "hard_violations": hard_fail, "soft_violations":
            [x for x in res if not x["hard"] and not x["holds"]], "evaluated": len(res)}


# -------------------------------------------------------------------- дата-продукты
def world_state_snapshot(model, *, valid_at: datetime, known_at: Optional[datetime] = None) -> list[dict]:
    """world_state_snapshot(entity_id, entity_type, state_json, geometry, confidence, observed_at, valid_at, provenance)"""
    snap = model.snapshot(valid_at=valid_at, known_at=known_at)
    rows = []
    for eid, attrs in snap.state.items():
        ent = model.entities.entities.get(eid)
        known = {k: v for k, v in attrs.items() if v["status"] == "known"}
        confs, obs, prov = [], [], []
        for k in known:
            a = model.ledger.get_as_of(eid, k, valid_at=valid_at, known_at=known_at)["assertion"]
            confs.append(a.confidence)
            obs.append(a.event_time.isoformat())
            prov.append(a.assertion_id)
        g = model.geometry.aabbs(valid_at, known_at=known_at).get(eid)
        rows.append({"entity_id": eid, "entity_type": None if ent is None else ent.entity_type,
                     "state_json": {k: v["value"] for k, v in known.items()},
                     "geometry": None if g is None else {"min": g[0].tolist(), "max": g[1].tolist()},
                     "confidence": min(confs) if confs else None, "observed_at": max(obs) if obs else None,
                     "valid_at": snap.valid_at, "provenance": prov, "snapshot_id": snap.snapshot_id,
                     "data_origin": snap.data_origin})
    return rows


def causal_timeseries(model, entity_id: str, metric_id: str, *, interventions: Sequence[Intervention] = (),
                      confounder_context_id: Optional[str] = None) -> list[dict]:
    """causal_timeseries(timestamp, entity_id, metric_id, value, intervention_id, confounder_context_id, quality_score)"""
    rows = []
    for a in model.ledger.history(entity_id, metric_id):
        iv = next((i.intervention_id for i in interventions if i.target_entity_id == entity_id
                   and i.executed_at <= a.event_time and
                   (a.event_time - i.executed_at).total_seconds() <= i.post_state_window_s), None)
        rows.append({"timestamp": a.event_time.isoformat(), "entity_id": entity_id, "metric_id": metric_id,
                     "value": a.value, "intervention_id": iv, "confounder_context_id": confounder_context_id,
                     "quality_score": max(0.0, 1.0 - float(a.uncertainty)), "epistemic_status": a.epistemic_status.value})
    return rows


def new_event_id() -> str:
    return uuid7()


__all__ = ["CAUSAL_EVIDENCE", "CONFOUNDERS", "CityEvent", "CityEventLedger", "CounterfactualRollout", "EVENT_CLASSES",
           "F_CF", "F_DO", "Intervention", "RULE_DOMAINS", "SafetyRule", "TRIGGER_SOURCES", "causal_timeseries",
           "check_rules", "new_event_id", "world_state_snapshot"]
