"""Битемпоральный журнал утверждений (мастер-ТЗ A1; Gap Closure v1 §5; WM Spec §2.1–2.2).

    State(T | K) = {a | a.knowledge_time ≤ K ∧ a.valid_from ≤ T ∧ (a.valid_to = ∅ ∨ a.valid_to > T)}
    (valid_to берётся таким, каким он был известен на момент K)
    «что происходило в T» = State(T | now);  «что CURE знала в T» = State(T | T)
    freshness(a, t) = t − a.knowledge_time;  stale ⟺ freshness > τ_attribute (topology — без TTL)
    conflict(a1,a2) ⟺ same(entity, attribute) ∧ observed(a1,a2) ∧ |v1 − v2| > δ_attribute

Правила:
  • запись только из ProjectionPipeline (у него токен записи) — прямой записи в состояние нет;
  • новый отсчёт того же источника закрывает прежний (valid_to = valid_from нового) — история остаётся;
  • разные источники с расхождением > δ в окне политики → оба CONFLICTED, текущего значения нет, пока
    конфликт не разрешат слиянием или оператором; ничто не перезаписывается молча;
  • устаревшее утверждение возвращается с quality = stale, пишется событие устаревания;
  • всё — строки единого журнала (WorldJournal), поэтому ответ на любой момент K воспроизводим.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Mapping, Optional

import numpy as np

from ..common.epistemic import EpistemicStatus
from ..world_model.assertions import QualityStatus, WorldModelAssertion
from ..operational_integrity.context import CanonicalWriteViolation as _OIWriteViolation
from ..operational_integrity.context import ProjectionContext
from .frames import SpatialReferenceService
from .journal import WorldJournal
from .policy import F_CONFLICT, F_FRESHNESS, F_STATE_AT, SpatialWorldError, SpatialWorldPolicy, aware, text

_FIELDS = ("assertion_id", "entity_id", "attribute_or_relation", "value", "event_time", "knowledge_time",
           "valid_time_start", "source_id", "unit", "valid_time_end", "provenance_refs", "epistemic_status",
           "quality_status", "freshness_ms", "confidence", "uncertainty", "scope", "world_model_version",
           "schema_version", "trace_id", "action_authority", "metadata")


class CanonicalWriteViolation(SpatialWorldError, _OIWriteViolation):
    """Запись в модель мира не через ProjectionPipeline (WM Spec §2.16 write_guard; WM-05 §1.2)."""


def to_payload(a: WorldModelAssertion) -> dict:
    out = {}
    for n in _FIELDS:
        v = getattr(a, n)
        if isinstance(v, datetime):
            v = v.isoformat()
        elif hasattr(v, "value") and hasattr(v, "name"):
            v = v.value
        elif hasattr(v, "tolist"):
            v = v.tolist()
        elif isinstance(v, tuple):
            v = list(v)
        elif isinstance(v, Mapping):
            v = dict(v)
        out[n] = v
    return out


def from_payload(p: Mapping[str, Any], frames: Optional[SpatialReferenceService]) -> WorldModelAssertion:
    kw = dict(p)
    for n in ("event_time", "knowledge_time", "valid_time_start", "valid_time_end"):
        if kw.get(n) is not None:
            kw[n] = datetime.fromisoformat(kw[n])
    kw["epistemic_status"] = EpistemicStatus(kw["epistemic_status"])
    kw["quality_status"] = QualityStatus(kw["quality_status"])
    kw["provenance_refs"] = tuple(kw.get("provenance_refs") or ())
    reg = None if frames is None else frames.registry_at(kw["valid_time_start"], known_at=kw["knowledge_time"])
    return WorldModelAssertion(**kw, frame_registry=reg)


@dataclass
class ConflictRecord:
    conflict_id: str
    entity_id: str
    attribute: str
    assertion_ids: tuple
    detected_at: datetime
    reason: str
    resolved_by: Optional[str] = None
    resolution_assertion_id: Optional[str] = None


@dataclass
class _Entry:
    a: WorldModelAssertion
    closes: list = field(default_factory=list)     # (valid_to, knowledge_time)
    statuses: list = field(default_factory=list)   # (status, knowledge_time, ref)


class BitemporalLedger:
    def __init__(self, journal: WorldJournal, policy: SpatialWorldPolicy, *,
                 frames: Optional[SpatialReferenceService] = None) -> None:
        self.journal, self.policy, self.frames = journal, policy, frames
        self._write_token = secrets.token_hex(16)
        self._entries: dict[str, _Entry] = {}
        self._by_key: dict[tuple, list[str]] = {}
        self.conflicts: dict[str, ConflictRecord] = {}
        self.stale_events: list[dict] = []
        self.audit: list[dict] = []
        for r in journal.rows(("assertion", "assertion_close", "assertion_status", "conflict", "conflict_resolved")):
            self._apply(r)

    # ---------------------------------------------------------------- журнал → индекс
    def _apply(self, r) -> None:
        p = r.payload
        if r.kind == "assertion":
            a = from_payload(p, self.frames)
            self._entries[a.assertion_id] = _Entry(a)
            self._by_key.setdefault((a.entity_id, a.attribute_or_relation), []).append(a.assertion_id)
        elif r.kind == "assertion_close":
            self._entries[p["assertion_id"]].closes.append((datetime.fromisoformat(p["valid_to"]), r.knowledge_time))
        elif r.kind == "assertion_status":
            self._entries[p["assertion_id"]].statuses.append((p["status"], r.knowledge_time, p.get("ref")))
        elif r.kind == "conflict":
            self.conflicts[p["conflict_id"]] = ConflictRecord(p["conflict_id"], p["entity_id"], p["attribute"],
                                                              tuple(p["assertion_ids"]), r.knowledge_time, p["reason"])
        elif r.kind == "conflict_resolved":
            c = self.conflicts[p["conflict_id"]]
            c.resolved_by, c.resolution_assertion_id = p["resolved_by"], p["resolution_assertion_id"]

    def writer_token(self, requester: Any) -> str:
        """Токен записи выдаётся только ProjectionPipeline (проверка по типу вызывающего)."""
        from .pipeline import ProjectionPipeline
        if not isinstance(requester, ProjectionPipeline):
            raise CanonicalWriteViolation("only ProjectionPipeline may write to the world model")
        return self._write_token

    # ---------------------------------------------------------------- запись
    def _blocked(self, reason: str, a: Any) -> None:
        """WM-05 §1.2: попытка записи мимо конвейера — событие аудита wm.write_blocked (в журнал, если он жив)."""
        self.audit.append({"event": "wm.write_blocked", "reason": reason,
                           "entity_id": getattr(a, "entity_id", None)})
        try:
            self.journal.append("audit", {"event": "wm.write_blocked", "reason": reason,
                                          "entity_id": getattr(a, "entity_id", None)},
                                knowledge_time=getattr(a, "knowledge_time", None) or self._now())
        except Exception:
            pass

    @staticmethod
    def _now() -> datetime:
        from datetime import timezone
        return datetime.now(timezone.utc)

    def assert_fact(self, a: WorldModelAssertion, *, token: str,
                    context: Optional[ProjectionContext] = None) -> WorldModelAssertion:
        if token != self._write_token:
            self._blocked("direct write without ProjectionPipeline token", a)
            raise CanonicalWriteViolation("direct write without ProjectionPipeline context")
        if not isinstance(context, ProjectionContext):
            self._blocked("missing ProjectionContext", a)
            raise CanonicalWriteViolation("world model mutation requires a ProjectionContext (WM-05 §1.2)")
        if not isinstance(a, WorldModelAssertion):
            raise SpatialWorldError("only WorldModelAssertion can be written")
        if dict(a.metadata).get("projection_context") != context.to_dict():
            a = replace(a, metadata={**dict(a.metadata), "projection_context": context.to_dict()},
                        frame_registry=None if self.frames is None else
                        self.frames.registry_at(a.valid_time_start, known_at=a.knowledge_time))
        if a.assertion_id in self._entries:
            raise SpatialWorldError("assertion id already exists (append-only)")
        self.policy.ttl_for(a.attribute_or_relation)          # класс атрибута обязан быть известен
        kt = a.knowledge_time
        r = self.journal.append("assertion", to_payload(a), knowledge_time=kt)
        self._apply(r)
        if a.epistemic_status not in (EpistemicStatus.OBSERVED, EpistemicStatus.MANUAL_VERIFIED):
            return a
        for other_id in list(self._by_key[(a.entity_id, a.attribute_or_relation)]):
            if other_id == a.assertion_id:
                continue
            o = self._entries[other_id]
            # сравниваем со всеми ещё действующими наблюдениями, в том числе уже конфликтными:
            # новый отсчёт одного источника не «снимает» конфликт молча
            if o.a.knowledge_time > kt or self._valid_to(o, kt) is not None \
                    or o.a.epistemic_status is not EpistemicStatus.OBSERVED:
                continue
            if o.a.source_id == a.source_id:
                if o.a.valid_time_start < a.valid_time_start:
                    self._close(other_id, a.valid_time_start, kt, f"superseded_by:{a.assertion_id}")
                elif o.a.valid_time_start > a.valid_time_start:        # опоздавший старый отсчёт
                    self._close(a.assertion_id, o.a.valid_time_start, kt, f"late_arrival_before:{other_id}")
                continue
            if abs((o.a.event_time - a.event_time).total_seconds()) > self.policy.conflict_window_s:
                continue
            if self._differs(o.a, a):
                self._conflict(o.a, a, kt)
        return a

    def _differs(self, a1: WorldModelAssertion, a2: WorldModelAssertion) -> bool:
        attr = a1.attribute_or_relation
        if attr not in self.policy.conflict_delta:
            raise SpatialWorldError(f"no conflict δ for {attr!r} in policy (fail-closed)")
        d = float(self.policy.conflict_delta[attr])
        try:
            v1, v2 = np.asarray(a1.value, float), np.asarray(a2.value, float)
            return bool(np.linalg.norm(v1 - v2) > d)
        except (TypeError, ValueError):
            return a1.value != a2.value                        # категориальные: несовпадение = конфликт

    def _close(self, aid: str, valid_to: datetime, kt: datetime, reason: str) -> None:
        r = self.journal.append("assertion_close", {"assertion_id": aid, "valid_to": valid_to.isoformat(),
                                                    "reason": reason}, knowledge_time=kt)
        self._apply(r)

    def _status(self, aid: str, status: str, kt: datetime, ref: str) -> None:
        r = self.journal.append("assertion_status", {"assertion_id": aid, "status": status, "ref": ref}, knowledge_time=kt)
        self._apply(r)

    def _conflict(self, a1: WorldModelAssertion, a2: WorldModelAssertion, kt: datetime) -> None:
        cid = f"conflict:{secrets.token_hex(6)}"
        r = self.journal.append("conflict", {"conflict_id": cid, "entity_id": a1.entity_id,
                                             "attribute": a1.attribute_or_relation,
                                             "assertion_ids": [a1.assertion_id, a2.assertion_id],
                                             "reason": "observed values differ by more than δ within the window"},
                                knowledge_time=kt)
        self._apply(r)
        for x in (a1, a2):
            self._status(x.assertion_id, "conflicted", kt, cid)

    def resolve_conflicts(self, conflict_ids, *, resolution: WorldModelAssertion, resolved_by: str,
                          token: str, context: ProjectionContext) -> None:
        """Разрешение — только новым утверждением (слияние или оператор); исходные остаются в истории."""
        ids = [conflict_ids] if isinstance(conflict_ids, str) else list(conflict_ids)
        cs = [self.conflicts.get(i) for i in ids]
        if not ids or any(c is None or c.resolved_by for c in cs):
            raise SpatialWorldError("unknown or already resolved conflict")
        if resolution.epistemic_status not in (EpistemicStatus.INFERRED, EpistemicStatus.MANUAL_VERIFIED):
            raise SpatialWorldError("a conflict is resolved by fusion (inferred) or an operator (manual_verified)")
        if any((c.entity_id, c.attribute) != (resolution.entity_id, resolution.attribute_or_relation) for c in cs):
            raise SpatialWorldError("resolution must be for the conflicted entity and attribute")
        self.assert_fact(resolution, token=token, context=context)
        kt = resolution.knowledge_time
        for c in cs:
            for aid in c.assertion_ids:
                if self._valid_to(self._entries[aid], kt) is None:
                    self._close(aid, resolution.valid_time_start, kt, f"conflict_resolved:{c.conflict_id}")
            r = self.journal.append("conflict_resolved", {"conflict_id": c.conflict_id,
                                                          "resolved_by": text("resolved_by", resolved_by),
                                                          "resolution_assertion_id": resolution.assertion_id},
                                    knowledge_time=kt)
            self._apply(r)

    # ---------------------------------------------------------------- чтение
    def _valid_to(self, e: _Entry, k: datetime) -> Optional[datetime]:
        ends = [vt for vt, kt in e.closes if kt <= k]
        if e.a.valid_time_end is not None:
            ends.append(e.a.valid_time_end)
        return min(ends) if ends else None

    def _conflicted(self, e: _Entry, k: datetime) -> bool:
        return any(s == "conflicted" and kt <= k for s, kt, _ in e.statuses)

    def _is_current(self, e: _Entry, k: datetime) -> bool:
        return e.a.knowledge_time <= k and self._valid_to(e, k) is None and not self._conflicted(e, k)

    def get_as_of(self, entity_id: str, attribute: str, *, valid_at: datetime,
                  known_at: Optional[datetime] = None) -> dict:
        """F_STATE_AT. Возвращает {assertion|None, status, conflicted: [...], formula}."""
        t = aware("valid_at", valid_at)
        k = aware("known_at", known_at) if known_at is not None else t
        cands, conflicted = [], []
        for aid in self._by_key.get((entity_id, attribute), ()):
            e = self._entries[aid]
            if e.a.knowledge_time > k or e.a.valid_time_start > t:
                continue
            vt = self._valid_to(e, k)
            if vt is not None and vt <= t:
                continue
            (conflicted if self._conflicted(e, k) else cands).append(e.a)
        if attribute in set(self.policy.extra.get("fused_attributes", ())):
            # WM-05 §3: у слитого состояния один владелец — StateFusionService; сырые измерения — отдельно
            owned = [a for a in cands if dict(a.metadata).get("owner") == "state_fusion"
                     or a.epistemic_status is EpistemicStatus.MANUAL_VERIFIED]
            measurements = [a for a in cands + conflicted if a not in owned]
            if not owned:
                return {"assertion": None, "status": "not_fused", "conflicted": conflicted,
                        "measurements": measurements, "formula": F_STATE_AT}
            best = max(owned, key=lambda a: (a.valid_time_start, a.knowledge_time))
            return {"assertion": best, "status": "known", "conflicted": conflicted, "measurements": measurements,
                    "formula": F_STATE_AT}
        if conflicted and not cands:
            return {"assertion": None, "status": "conflicted", "conflicted": conflicted, "formula": F_STATE_AT}
        if not cands:
            return {"assertion": None, "status": "unknown", "conflicted": conflicted, "formula": F_STATE_AT}
        best = max(cands, key=lambda a: (a.valid_time_start, a.knowledge_time))
        return {"assertion": best, "status": "known", "conflicted": conflicted, "formula": F_STATE_AT}

    def get_current(self, entity_id: str, attribute: str, *, now: datetime) -> dict:
        now = aware("now", now)
        out = self.get_as_of(entity_id, attribute, valid_at=now, known_at=now)
        a = out["assertion"]
        if a is None:
            return out
        ttl = self.policy.ttl_for(attribute)
        age = (now - a.knowledge_time).total_seconds()
        if ttl is not None and age > ttl:
            out["assertion"] = replace(a, quality_status=QualityStatus.STALE, action_authority="none",
                                       frame_registry=None if self.frames is None else
                                       self.frames.registry_at(a.valid_time_start, known_at=a.knowledge_time))
            out["status"] = "stale"
            ev = {"entity_id": entity_id, "attribute": attribute, "age_s": age, "ttl_s": ttl, "at": now.isoformat(),
                  "formula": F_FRESHNESS.formula_id}
            self.stale_events.append(ev)
        out["freshness_s"] = age
        return out

    def entity_state_at(self, entity_id: str, *, valid_at: datetime, known_at: Optional[datetime] = None) -> dict:
        attrs = {attr for (e, attr) in self._by_key if e == entity_id}
        return {attr: self.get_as_of(entity_id, attr, valid_at=valid_at, known_at=known_at) for attr in sorted(attrs)}

    def history(self, entity_id: str, attribute: str) -> list[WorldModelAssertion]:
        return [self._entries[a].a for a in self._by_key.get((entity_id, attribute), ())]

    def entities(self) -> set:
        return {e for e, _ in self._by_key}

    def open_conflicts(self, entity_id: Optional[str] = None) -> list[ConflictRecord]:
        return [c for c in self.conflicts.values() if not c.resolved_by and (entity_id is None or c.entity_id == entity_id)]

    def reassign_entity(self, old_entity: str, new_entity: str, *, token: str, knowledge_time: datetime,
                        context: Optional[ProjectionContext] = None) -> int:
        """Слияние сущностей: утверждения дубликата копируются на выжившую (история дубликата остаётся)."""
        if token != self._write_token or not isinstance(context, ProjectionContext):
            self._blocked("entity reassignment outside projection context", None)
            raise CanonicalWriteViolation("entity reassignment only through the projection context")
        n = 0
        for (e, attr), ids in list(self._by_key.items()):
            if e != old_entity:
                continue
            for aid in list(ids):
                a = self._entries[aid].a
                vt = self._valid_to(self._entries[aid], knowledge_time)
                moved = replace(a, assertion_id=f"{a.assertion_id}@{new_entity}", entity_id=new_entity,
                                knowledge_time=aware("knowledge_time", knowledge_time), valid_time_end=vt,
                                provenance_refs=tuple(a.provenance_refs) + (f"merged_from:{aid}",),
                                metadata={**dict(a.metadata), "merge_provenance": {"from_entity": old_entity,
                                                                                   "source_assertion_id": aid,
                                                                                   "merge_context": context.to_dict()}},
                                frame_registry=None if self.frames is None else
                                self.frames.registry_at(a.valid_time_start, known_at=knowledge_time))
                r = self.journal.append("assertion", to_payload(moved), knowledge_time=knowledge_time)
                self._apply(r)
                n += 1
        return n


__all__ = ["BitemporalLedger", "CanonicalWriteViolation", "ConflictRecord", "F_CONFLICT", "from_payload", "to_payload"]
