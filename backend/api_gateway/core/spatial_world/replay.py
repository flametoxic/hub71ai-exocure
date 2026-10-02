"""DecisionReplayEngine и ReplayPack (WM Engineering Spec §2.8, формула F35; мастер-ТЗ DoD 12; City WM §13 п.8).

F35: решение повторяется на замороженном снимке мира — тех же утверждениях (по хэшам), тех же версиях моделей,
политики и порогов. Любое расхождение попадает в marked_diff с объяснением, deterministic = False; повтор
никогда не падает молча. Решение — зарегистрированная чистая функция (id + версия), а не произвольный код.
ReplayPack экспортируется как JSON с подписью HMAC (аудит, due diligence).
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Mapping, Optional

from ..world_model.reality_formulas import FormulaReference
from .journal import canonical, digest
from .ledger import to_payload
from .policy import SPEC, SpatialWorldError, aware, text

F_REPLAY = FormulaReference("SPEC-F35-REPLAY", SPEC, 8,
                            "replay(decision) on the frozen snapshot; any divergence → marked_diff, deterministic = False")


@dataclass
class ReplayPack:
    decision_id: str
    world_snapshot_version: str
    evidence_set: list                     # [{assertion_id, hash}]
    model_versions: dict
    policy_version: str
    original_decision: dict
    replayed_decision: Optional[dict]
    marked_diff: list = field(default_factory=list)
    deterministic: bool = False
    data_mode: Optional[str] = None


class DecisionReplayEngine:
    def __init__(self, model, *, signing_key: bytes) -> None:
        if len(signing_key) < 16:
            raise SpatialWorldError("replay pack signing key must be ≥ 16 bytes")
        self.model, self._key = model, signing_key
        self._fns: dict[tuple, Callable[[Mapping[str, Any]], Mapping[str, Any]]] = {}
        self._captured: dict[str, dict] = {}

    def register_decision_function(self, fn_id: str, version: str, fn: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> None:
        self._fns[(text("fn_id", fn_id), text("version", version))] = fn

    def _inputs(self, evidence_ids, valid_at: datetime, known_at: datetime) -> tuple[dict, list]:
        by_id = {a.assertion_id: a for e in self.model.ledger._entries.values() for a in (e.a,)}
        inputs, ev = {}, []
        for aid in evidence_ids:
            a = by_id.get(aid)
            if a is None or a.knowledge_time > known_at:
                raise SpatialWorldError(f"evidence {aid} was not known at {known_at.isoformat()}")
            p = to_payload(a)
            ev.append({"assertion_id": aid, "hash": digest(p)})
            inputs[aid] = {"entity_id": a.entity_id, "attribute": a.attribute_or_relation, "value": a.value,
                           "event_time": a.event_time.isoformat(), "epistemic_status": a.epistemic_status.value}
        return inputs, ev

    def capture(self, decision_id: str, *, fn_id: str, fn_version: str, valid_at: datetime, known_at: datetime,
                evidence_ids: tuple, model_versions: Mapping[str, str], policy_version: str,
                thresholds: Mapping[str, Any]) -> dict:
        """Вызывается при каждом решении: замораживает снимок, версии и хэши свидетельств, и считает решение."""
        if decision_id in self._captured:
            raise SpatialWorldError("decision already captured (immutable)")
        fn = self._fns.get((fn_id, fn_version))
        if fn is None:
            raise SpatialWorldError(f"decision function {fn_id}@{fn_version} is not registered")
        t, k = aware("valid_at", valid_at), aware("known_at", known_at)
        snap = self.model.snapshot(valid_at=t, known_at=k)
        inputs, ev = self._inputs(evidence_ids, t, k)
        original = dict(fn({"evidence": inputs, "thresholds": dict(thresholds), "snapshot_id": snap.snapshot_id}))
        self._captured[decision_id] = {"fn": (fn_id, fn_version), "valid_at": t, "known_at": k, "snapshot_id": snap.snapshot_id,
                                       "evidence": ev, "evidence_ids": tuple(evidence_ids), "model_versions": dict(model_versions),
                                       "policy_version": text("policy_version", policy_version),
                                       "thresholds": dict(thresholds), "original": original,
                                       "data_mode": snap.data_origin}
        return original

    def replay(self, decision_id: str, *, model_versions_now: Optional[Mapping[str, str]] = None,
               policy_version_now: Optional[str] = None) -> ReplayPack:
        c = self._captured.get(decision_id)
        if c is None:
            raise SpatialWorldError(f"decision {decision_id} was never captured")
        diff: list[dict] = []
        snap = self.model.snapshot(valid_at=c["valid_at"], known_at=c["known_at"])
        if snap.snapshot_id != c["snapshot_id"]:
            diff.append({"field": "world_snapshot", "original": c["snapshot_id"], "replayed": snap.snapshot_id,
                         "explanation": "the frozen snapshot does not reconstruct — history was altered or lost"})
        replayed = None
        try:
            inputs, ev = self._inputs(c["evidence_ids"], c["valid_at"], c["known_at"])
            for a, b in zip(c["evidence"], ev):
                if a["hash"] != b["hash"]:
                    diff.append({"field": f"evidence:{a['assertion_id']}", "original": a["hash"], "replayed": b["hash"],
                                 "explanation": "evidence content changed"})
            fn = self._fns.get(c["fn"])
            if fn is None:
                raise SpatialWorldError(f"decision function {c['fn']} no longer registered")
            replayed = dict(fn({"evidence": inputs, "thresholds": c["thresholds"], "snapshot_id": snap.snapshot_id}))
        except Exception as exc:                      # повтор никогда не падает молча
            diff.append({"field": "replay", "original": None, "replayed": None,
                         "explanation": f"{type(exc).__name__}: {exc}"})
        if replayed is not None:
            for key in sorted(set(c["original"]) | set(replayed)):
                if canonical(c["original"].get(key)) != canonical(replayed.get(key)):
                    diff.append({"field": key, "original": c["original"].get(key), "replayed": replayed.get(key),
                                 "explanation": "decision output differs on the same frozen inputs"})
        for name, v in dict(model_versions_now or {}).items():
            if c["model_versions"].get(name) != v:
                diff.append({"field": f"model_version:{name}", "original": c["model_versions"].get(name), "replayed": v,
                             "explanation": "model version changed since the decision (replay uses frozen inputs)"})
        if policy_version_now is not None and policy_version_now != c["policy_version"]:
            diff.append({"field": "policy_version", "original": c["policy_version"], "replayed": policy_version_now,
                         "explanation": "policy changed since the decision"})
        return ReplayPack(decision_id, c["snapshot_id"], c["evidence"], c["model_versions"], c["policy_version"],
                          c["original"], replayed, diff, not diff, c["data_mode"])

    def export_pack(self, pack: ReplayPack) -> bytes:
        body = json.loads(canonical({k: getattr(pack, k) for k in pack.__dataclass_fields__}))
        sig = hmac.new(self._key, canonical(body).encode(), hashlib.sha256).hexdigest()
        return json.dumps({"replay_pack": body, "signature": sig, "formula": F_REPLAY.formula_id},
                          ensure_ascii=False, sort_keys=True).encode("utf-8")

    def verify_export(self, blob: bytes) -> bool:
        d = json.loads(blob)
        return hmac.compare_digest(hmac.new(self._key, canonical(d["replay_pack"]).encode(), hashlib.sha256).hexdigest(),
                                   d["signature"])


__all__ = ["DecisionReplayEngine", "F_REPLAY", "ReplayPack"]
