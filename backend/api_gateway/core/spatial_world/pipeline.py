"""ProjectionPipeline — единственный путь записи в модель мира (Gap Closure v1 §1, v2 §1; WM Spec §2.15, §6.1).

    событие → целостность → разрешение сущности → пространственная привязка (цепочка кадров)
    → битемпоральное утверждение → журнал мира
Ни LLM, ни UI, ни адаптер, ни симулятор, ни нейросеть не пишут состояние напрямую: токен записи журнала
утверждений есть только у этого конвейера. Неразрешённая сущность → карантин (в операционное состояние не
попадает); вероятностное разрешение → запись без права на действие; нет цепочки кадров → frame_unresolved.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Optional

import numpy as np

from ..common.epistemic import EpistemicStatus
from ..world_model.assertions import WorldModelAssertion
from ..world_model.spatial_formulas import FrameUnresolvedError
from .entities import EntityRegistry, ResolutionStatus
from .frames import ROOT, SpatialPose, SpatialReferenceService
from .fusion import FusedEstimate
from ..operational_integrity.context import DataMode, ProjectionContext
from .ledger import BitemporalLedger
from .policy import SpatialWorldError, aware, text

_ALLOWED_INPUT = (EpistemicStatus.OBSERVED, EpistemicStatus.MANUAL_VERIFIED)


class ProjectionPipeline:
    def __init__(self, *, ledger: BitemporalLedger, entities: EntityRegistry, frames: SpatialReferenceService,
                 world_model_version: str, topology: Any = None) -> None:
        self.ledger, self.entities, self.frames, self.topology = ledger, entities, frames, topology
        self.world_model_version = text("world_model_version", world_model_version)
        self._token = ledger.writer_token(self)
        self.pipeline_id = f"projection:{self.world_model_version}"
        self.rejected: list[dict] = []

    def _reject(self, event: Mapping[str, Any], reason: str) -> dict:
        self.rejected.append({"event": dict(event), "reason": reason})
        return {"status": "rejected", "reason": reason}

    def project(self, event: Mapping[str, Any]) -> dict:
        """event: {source_namespace, source_id, attribute, value, unit, event_time, knowledge_time, source_ref,
        integrity, epistemic_status, confidence, uncertainty, scope, trace_id,
        context: {class_hint, keys, names}, pose: {frame_id, position, covariance} (необязательно),
        observation_id, schema_version, data_mode}  — последние три обязательны (WM-05 §1.2, §7)"""
        ev = dict(event)
        if ev.get("integrity") != "verified":
            return self._reject(ev, f"integrity:{ev.get('integrity')}")
        try:
            t = aware("event_time", ev["event_time"])
            kt = aware("knowledge_time", ev["knowledge_time"])
            status = EpistemicStatus(ev.get("epistemic_status", "observed"))
            pctx = ProjectionContext(self.pipeline_id, str(ev.get("trace_id") or ""), (str(ev.get("observation_id") or ""),),
                                     str(ev.get("schema_version") or ""), DataMode(ev.get("data_mode")))
        except (KeyError, ValueError, SpatialWorldError, RuntimeError) as exc:
            return self._reject(ev, f"schema:{exc}")
        if status not in _ALLOWED_INPUT:
            return self._reject(ev, f"epistemic_status_{status.value}_cannot_enter_as_observation")
        if kt < t:
            return self._reject(ev, "knowledge_time_before_event_time")
        ctx = dict(ev.get("context") or {})
        ctx["event_time"] = t
        # пространственная привязка: позу датчика/объекта переводим в корневой кадр
        world_pose = None
        chain: tuple = ()
        if ev.get("pose") is not None:
            p = ev["pose"]
            try:
                sp = SpatialPose("pending", p["frame_id"], tuple(p["position"]), tuple(np.asarray(p["covariance"],
                                                                                                    float).ravel()),
                                 t, str(ev.get("source_ref", "")), float(ev.get("confidence", 1.0)))
                out = self.frames.transform(sp, ROOT, known_at=kt)
            except FrameUnresolvedError as exc:
                return self._reject(ev, f"frame_unresolved:{exc}")
            world_pose, chain = out["pose"], out["chain"]
            ctx.setdefault("position_world", world_pose.position)
        # разрешение сущности
        res = self.entities.resolve(text("source_namespace", ev.get("source_namespace")),
                                    text("source_id", ev.get("source_id")), ctx)
        if res.status in (ResolutionStatus.NEW_ENTITY, ResolutionStatus.CONFLICT):
            qid = self.entities.quarantine_event(ev | {"event_time": t.isoformat(), "knowledge_time": kt.isoformat()},
                                                 res, knowledge_time=kt)
            return {"status": "quarantined", "quarantine_id": qid, "resolution": res}
        source_eid = res.canonical_entity_id
        authority = "eligible" if res.status is ResolutionStatus.RESOLVED else "none"
        # наблюдаемый объект: явно указанный, иначе единственная связь «observes» источника, иначе сам источник
        target = ev.get("observed")
        if target is not None:
            tres = self.entities.resolve(text("observed.namespace", target.get("namespace")),
                                         text("observed.id", target.get("id")), ctx)
            if tres.status in (ResolutionStatus.NEW_ENTITY, ResolutionStatus.CONFLICT):
                qid = self.entities.quarantine_event(ev | {"event_time": t.isoformat(), "knowledge_time": kt.isoformat()},
                                                     tres, knowledge_time=kt)
                return {"status": "quarantined", "quarantine_id": qid, "resolution": tres}
            eid = tres.canonical_entity_id
            if tres.status is not ResolutionStatus.RESOLVED:
                authority = "none"
        else:
            obs = [] if self.topology is None else self.topology.neighbors(source_eid, t, types=("observes",), known_at=kt)
            if len(obs) > 1:
                return self._reject(ev, f"ambiguous_observed_entity:{obs}")
            eid = obs[0] if obs else source_eid
        attribute = text("attribute", ev.get("attribute"))
        value = ev.get("value")
        metadata: dict = {"resolution_status": res.status.value, "resolution_method": res.method,
                          "resolution_score": res.score, "source_namespace": ev["source_namespace"],
                          "source_native_id": ev["source_id"], "source_entity_id": source_eid,
                          "data_mode": pctx.data_mode.value, "observation_id": ev["observation_id"]}
        if world_pose is not None and attribute.endswith("_world"):
            value = list(world_pose.position)
            metadata.update({"coordinate_frame": ROOT, "position_covariance": list(world_pose.covariance),
                             "transform_chain_computed": list(chain)})
        reg = self.frames.registry_at(t, known_at=kt)
        try:
            a = WorldModelAssertion(
                entity_id=eid, attribute_or_relation=attribute, value=value, event_time=t, knowledge_time=kt,
                valid_time_start=t, source_id=source_eid, unit=ev.get("unit"),
                provenance_refs=(text("source_ref", ev.get("source_ref")), *tuple(ev.get("evidence_refs") or ())),
                epistemic_status=status,
                freshness_ms=int((kt - t).total_seconds() * 1000), confidence=float(ev.get("confidence", 1.0)),
                uncertainty=float(ev.get("uncertainty", 0.0)), scope=str(ev.get("scope", "default")),
                world_model_version=self.world_model_version, trace_id=str(ev.get("trace_id", "")),
                action_authority=authority, metadata=metadata, frame_registry=reg)
        except (ValueError, FrameUnresolvedError) as exc:
            return self._reject(ev, f"assertion_invalid:{exc}")
        self.ledger.assert_fact(a, token=self._token, context=pctx)
        return {"status": "accepted", "assertion_id": a.assertion_id, "entity_id": eid,
                "epistemic_status": status.value, "action_authority": authority, "resolution": res,
                "transform_chain": chain, "data_mode": pctx.data_mode.value}

    def reproject_quarantined(self, quarantine_id: str, *, entity_id: str, reviewer: str,
                              knowledge_time: datetime) -> dict:
        """WM-05 §2.1: карантин снимает человек — алиас источника привязывается к сущности, событие проецируется."""
        q = self.entities.resolve_quarantine(quarantine_id, entity_id=entity_id, reviewer=reviewer,
                                             knowledge_time=knowledge_time)
        ev = dict(q["event"])
        ev["event_time"] = datetime.fromisoformat(ev["event_time"])
        ev["knowledge_time"] = aware("knowledge_time", knowledge_time)
        ev["evidence_refs"] = tuple(ev.get("evidence_refs") or ()) + (f"quarantine_review:{quarantine_id}:{reviewer}",)
        return self.project(ev)

    # ---------------------------------------------------------------- производные утверждения
    def project_fused(self, est: FusedEstimate, *, source_ref: str, knowledge_time: datetime,
                      source_assertion_ids: tuple, trace_id: str, data_mode: Any, schema_version: str,
                      unit: Optional[str] = None, resolves_conflicts: tuple = ()) -> dict:
        """Слитое значение — INFERRED с владельцем state_fusion (единственный писатель слитого состояния,
        WM-05 §3.2); события дрейфа — утверждения sensor_health (источник не удаляется)."""
        kt = aware("knowledge_time", knowledge_time)
        pctx = ProjectionContext(self.pipeline_id, trace_id, tuple(source_assertion_ids), schema_version,
                                 DataMode(data_mode))
        a = WorldModelAssertion(
            entity_id=est.entity_id, attribute_or_relation=est.attribute, value=est.value, event_time=est.at,
            knowledge_time=kt, valid_time_start=est.at, source_id=text("source_ref", source_ref), unit=unit,
            provenance_refs=tuple(f"source:{s}" for s in est.contributing_sources),
            epistemic_status=EpistemicStatus.INFERRED, confidence=1.0, uncertainty=float(np.sqrt(est.variance)),
            world_model_version=self.world_model_version, action_authority="eligible",
            trace_id=trace_id,
            metadata={"fusion_method": est.method, "effective_variance": dict(est.effective_variance),
                      "sensor_health": dict(est.sensor_health), "owner": "state_fusion",
                      "data_mode": pctx.data_mode.value})
        if resolves_conflicts:
            self.ledger.resolve_conflicts(resolves_conflicts, resolution=a, resolved_by=f"fusion:{est.method}",
                                          token=self._token, context=pctx)
        else:
            self.ledger.assert_fact(a, token=self._token, context=pctx)
        written = [a.assertion_id]
        for ev in est.events:
            h = WorldModelAssertion(
                entity_id=ev["sensor_id"], attribute_or_relation="sensor_health", value=ev["sensor_health"],
                event_time=est.at, knowledge_time=kt, valid_time_start=est.at, source_id=f"fusion:{est.entity_id}",
                epistemic_status=EpistemicStatus.INFERRED, world_model_version=self.world_model_version,
                trace_id=trace_id, metadata={**{k: v for k, v in ev.items() if k not in ("sensor_health",)},
                                             "owner": "state_fusion", "data_mode": pctx.data_mode.value})
            self.ledger.assert_fact(h, token=self._token, context=pctx)
            written.append(h.assertion_id)
        return {"status": "accepted", "assertion_ids": written}

    def operator_verify(self, *, entity_id: str, attribute: str, value: Any, operator_id: str, at: datetime,
                        knowledge_time: datetime, statement_id: str, trace_id: str, data_mode: Any, schema_version: str,
                        resolves_conflicts: tuple = (), unit: Optional[str] = None) -> str:
        pctx = ProjectionContext(self.pipeline_id, trace_id, (statement_id,), schema_version, DataMode(data_mode))
        a = WorldModelAssertion(entity_id=entity_id, attribute_or_relation=attribute, value=value,
                                event_time=aware("at", at), knowledge_time=aware("knowledge_time", knowledge_time),
                                valid_time_start=at, source_id=f"operator:{text('operator_id', operator_id)}", unit=unit,
                                epistemic_status=EpistemicStatus.MANUAL_VERIFIED, trace_id=trace_id,
                                world_model_version=self.world_model_version,
                                metadata={"data_mode": pctx.data_mode.value, "statement_id": statement_id})
        if resolves_conflicts:
            self.ledger.resolve_conflicts(resolves_conflicts, resolution=a, resolved_by=f"operator:{operator_id}",
                                          token=self._token, context=pctx)
        else:
            self.ledger.assert_fact(a, token=self._token, context=pctx)
        return a.assertion_id

    def merge_entities(self, survivor: str, duplicate: str, *, evidence: tuple, approved_by: str,
                       knowledge_time: datetime, trace_id: str, data_mode: Any, schema_version: str) -> dict:
        """WM-05 §2.2: сущность не удаляется; lifecycle = merged, merged_into = выживший; история сохраняется,
        перенесённые утверждения несут merge provenance."""
        out = self.entities.merge(survivor, duplicate, evidence=evidence, approved_by=approved_by,
                                  knowledge_time=knowledge_time)
        pctx = ProjectionContext(self.pipeline_id, trace_id, tuple(evidence), schema_version, DataMode(data_mode))
        out["moved_assertions"] = self.ledger.reassign_entity(duplicate, survivor, token=self._token,
                                                              knowledge_time=knowledge_time, context=pctx)
        return out

__all__ = ["ProjectionPipeline"]
