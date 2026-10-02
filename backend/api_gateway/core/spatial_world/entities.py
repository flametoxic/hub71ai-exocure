"""Реестр сущностей и разрешение идентичности (мастер-ТЗ A2; Gap Closure v1 §3, v2 §2; WM Spec §2.3; City WM §2.1).

    AHU-4 = ahu_04 = HVAC_AHU_004 = MECH-000441  →  asset:pilot-01:ahu:004
    Score(c|o) = 0.30·S_alias + 0.35·S_det + 0.15·S_spatial + 0.10·S_type + 0.10·S_temporal,  S_spatial = exp(−d²/2σ²)
    resolved ≥ 0.90;  0.70–0.90 — вероятностно, на проверку человеком;  top-2 ближе 0.05 — конфликт;
    < 0.70 — новый кандидат, не операционная сущность, пока политика не разрешит создание.
    Уникальное совпадение детерминированного ключа (serial / BIM GUID / BMS point / CMMS id / MAC) — Score = 1.0.

Порядок (WM Spec): детерминированный ключ → индекс алиасов → перебор кандидатов → оценка → решение.
Неоднозначное разрешение не даёт права на действие. Слияние — с доказательствами и автором, обратимо.
Всё хранится строками журнала мира.
"""
from __future__ import annotations

import math
import re
import secrets
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Optional

from .journal import WorldJournal
from .policy import F_IDENTITY, SpatialWorldError, SpatialWorldPolicy, aware, text

DETERMINISTIC_NAMESPACES = ("serial", "bim_guid", "bms_point", "cmms_id", "mac", "vin_pseudonym", "hardware_id",
                            "v2x_id")


class ResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    RESOLVED_PROBABILISTIC = "resolved_probabilistic"
    CONFLICT = "conflict"
    NEW_ENTITY = "new_entity"


@dataclass
class EntityIdentity:
    entity_id: str
    entity_type: str
    ontology_class: str
    canonical_name: str
    site_scope: str
    parent_id: Optional[str]
    district_id: Optional[str]
    spatial_anchor: Optional[str]                # кадр, к которому привязана сущность
    position_world: Optional[tuple]               # ориентир для S_spatial (в корневом кадре)
    valid_from: datetime
    valid_to: Optional[datetime] = None
    aliases: dict = field(default_factory=dict)   # namespace → [alias]
    keys: dict = field(default_factory=dict)      # детерминированные ключи namespace → key
    lifecycle_state: str = "active"               # active | candidate | merged | decommissioned
    identity_confidence: float = 1.0
    merged_into: Optional[str] = None

    def active_at(self, t: datetime) -> bool:
        return self.valid_from <= t and (self.valid_to is None or t < self.valid_to)


@dataclass
class ResolutionResult:
    canonical_entity_id: Optional[str]
    status: ResolutionStatus
    score: float
    method: str
    candidates: list                              # [{entity_id, score, components}]
    evidence_refs: tuple
    requires_human_review: bool
    action_authority: str                         # eligible | none
    formula: Any = F_IDENTITY


def normalize_alias(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    toks = re.findall(r"[a-z]+|\d+", s)
    return " ".join(str(int(t)) if t.isdigit() else t for t in toks)


def alias_similarity(a: str, b: str) -> float:
    """Совпадение токенов (буквенных и числовых без ведущих нулей): |A∩B| / min(|A|,|B|)."""
    ta, tb = set(normalize_alias(a).split()), set(normalize_alias(b).split())
    if not ta or not tb:
        return 0.0
    if not ({t for t in ta if t.isdigit()} & {t for t in tb if t.isdigit()}) and \
            ({t for t in ta if t.isdigit()} or {t for t in tb if t.isdigit()}):
        return 0.0                                 # номера разные — это другие объекты (ahu 4 ≠ ahu 5)
    return len(ta & tb) / min(len(ta), len(tb))


class EntityRegistry:
    def __init__(self, journal: WorldJournal, policy: SpatialWorldPolicy) -> None:
        self.journal, self.policy = journal, policy
        self.entities: dict[str, EntityIdentity] = {}
        self._alias_index: dict[tuple, set] = {}      # (namespace, normalized) → {entity_id}
        self._key_index: dict[tuple, set] = {}        # (namespace, key) → {entity_id}
        self._confirmed: dict[tuple, int] = {}        # (namespace, entity) → подтверждённые совпадения
        self.quarantine: list[dict] = []
        self.merges: list[dict] = []
        for r in journal.rows(("entity_create", "entity_alias", "entity_key", "entity_merge", "entity_unmerge",
                               "entity_quarantine", "entity_match_confirmed", "entity_close",
                               "entity_quarantine_status")):
            self._apply(r)

    # ---------------------------------------------------------------- журнал → индекс
    def _apply(self, r) -> None:
        p = r.payload
        k = r.kind
        if k == "entity_create":
            e = EntityIdentity(p["entity_id"], p["entity_type"], p["ontology_class"], p["canonical_name"],
                               p["site_scope"], p.get("parent_id"), p.get("district_id"), p.get("spatial_anchor"),
                               None if p.get("position_world") is None else tuple(p["position_world"]),
                               datetime.fromisoformat(p["valid_from"]), lifecycle_state=p["lifecycle_state"],
                               identity_confidence=p["identity_confidence"])
            self.entities[e.entity_id] = e
        elif k == "entity_alias":
            e = self.entities[p["entity_id"]]
            e.aliases.setdefault(p["namespace"], []).append(p["alias"])
            self._alias_index.setdefault((p["namespace"], normalize_alias(p["alias"])), set()).add(e.entity_id)
        elif k == "entity_key":
            e = self.entities[p["entity_id"]]
            e.keys[p["namespace"]] = p["key"]
            self._key_index.setdefault((p["namespace"], str(p["key"]).strip().lower()), set()).add(e.entity_id)
        elif k == "entity_merge":
            d = self.entities[p["duplicate"]]
            d.merged_into, d.lifecycle_state = p["survivor"], "merged"
            self.merges.append(dict(p, at=r.knowledge_time.isoformat()))
        elif k == "entity_unmerge":
            d = self.entities[p["duplicate"]]
            d.merged_into, d.lifecycle_state = None, "active"
            self.merges.append(dict(p, at=r.knowledge_time.isoformat(), undo=True))
        elif k == "entity_quarantine":
            self.quarantine.append(dict(p, created_at=r.knowledge_time.isoformat()))
        elif k == "entity_quarantine_status":
            q = next(x for x in self.quarantine if x["quarantine_id"] == p["quarantine_id"])
            q["status"] = p["status"]
            q["closed_by"] = p["reviewer"]
        elif k == "entity_match_confirmed":
            key = (p["namespace"], p["entity_id"])
            self._confirmed[key] = self._confirmed.get(key, 0) + 1
        elif k == "entity_close":
            self.entities[p["entity_id"]].valid_to = datetime.fromisoformat(p["valid_to"])
            self.entities[p["entity_id"]].lifecycle_state = p["lifecycle_state"]

    # ---------------------------------------------------------------- регистрация (импорт BIM/BMS/CMMS, оператор)
    def register(self, *, entity_id: str, entity_type: str, ontology_class: str, canonical_name: str,
                 site_scope: str, valid_from: datetime, knowledge_time: datetime, parent_id: Optional[str] = None,
                 district_id: Optional[str] = None, spatial_anchor: Optional[str] = None,
                 position_world: Optional[tuple] = None, aliases: Optional[Mapping[str, list]] = None,
                 keys: Optional[Mapping[str, str]] = None, lifecycle_state: str = "active") -> EntityIdentity:
        if entity_id in self.entities:
            raise SpatialWorldError(f"entity {entity_id} already registered (no silent duplicate)")
        if parent_id is not None and parent_id not in self.entities:
            raise SpatialWorldError(f"parent {parent_id} is not registered")
        r = self.journal.append("entity_create", {
            "entity_id": text("entity_id", entity_id), "entity_type": text("entity_type", entity_type),
            "ontology_class": text("ontology_class", ontology_class), "canonical_name": text("canonical_name", canonical_name),
            "site_scope": text("site_scope", site_scope), "parent_id": parent_id, "district_id": district_id,
            "spatial_anchor": spatial_anchor, "position_world": None if position_world is None else list(position_world),
            "valid_from": aware("valid_from", valid_from).isoformat(), "lifecycle_state": lifecycle_state,
            "identity_confidence": 1.0}, knowledge_time=knowledge_time)
        self._apply(r)
        for ns, als in dict(aliases or {}).items():
            for al in als:
                self.register_alias(entity_id, ns, al, knowledge_time=knowledge_time)
        for ns, key in dict(keys or {}).items():
            self.register_key(entity_id, ns, key, knowledge_time=knowledge_time)
        return self.entities[entity_id]

    def register_alias(self, entity_id: str, namespace: str, alias: str, *, knowledge_time: datetime) -> None:
        if entity_id not in self.entities:
            raise SpatialWorldError(f"unknown entity {entity_id}")
        holders = self._alias_index.get((namespace, normalize_alias(alias)), set()) - {entity_id}
        if holders:
            raise SpatialWorldError(f"alias {namespace}:{alias} already points to {sorted(holders)} — resolve by merge")
        self._apply(self.journal.append("entity_alias", {"entity_id": entity_id, "namespace": text("namespace", namespace),
                                                         "alias": text("alias", alias)}, knowledge_time=knowledge_time))

    def register_key(self, entity_id: str, namespace: str, key: str, *, knowledge_time: datetime) -> None:
        if namespace not in DETERMINISTIC_NAMESPACES:
            raise SpatialWorldError(f"{namespace!r} is not a deterministic key namespace")
        self._apply(self.journal.append("entity_key", {"entity_id": entity_id, "namespace": namespace,
                                                       "key": text("key", str(key))}, knowledge_time=knowledge_time))

    def decommission(self, entity_id: str, *, valid_to: datetime, knowledge_time: datetime) -> None:
        self._apply(self.journal.append("entity_close", {"entity_id": entity_id, "valid_to": aware("valid_to", valid_to).isoformat(),
                                                         "lifecycle_state": "decommissioned"}, knowledge_time=knowledge_time))

    def canonical(self, entity_id: str) -> str:
        seen = set()
        while self.entities[entity_id].merged_into and entity_id not in seen:
            seen.add(entity_id)
            entity_id = self.entities[entity_id].merged_into
        return entity_id

    # ---------------------------------------------------------------- разрешение
    def resolve(self, namespace: str, source_id: str, context: Optional[Mapping[str, Any]] = None) -> ResolutionResult:
        """context: {event_time, class_hint, position_world, keys: {ns: key}, names: [..]}"""
        ctx = dict(context or {})
        t = aware("event_time", ctx.get("event_time"))
        keys = dict(ctx.get("keys") or {})
        if namespace in DETERMINISTIC_NAMESPACES:
            keys.setdefault(namespace, source_id)
        th = self.policy.identity_thresholds
        # 1 — детерминированный ключ
        for ns, key in keys.items():
            hit = {self.canonical(e) for e in self._key_index.get((ns, str(key).strip().lower()), set())
                   if self.entities[e].active_at(t)}
            if len(hit) == 1:
                eid = hit.pop()
                return ResolutionResult(eid, ResolutionStatus.RESOLVED, 1.0, f"deterministic_key:{ns}",
                                        [{"entity_id": eid, "score": 1.0}], (f"{ns}:{key}",), False, "eligible")
            if len(hit) > 1:                   # один «уникальный» ключ у нескольких записей — ошибка данных
                return ResolutionResult(None, ResolutionStatus.CONFLICT, 1.0, f"deterministic_key_ambiguous:{ns}",
                                        [{"entity_id": e, "score": 1.0} for e in sorted(hit)], (f"{ns}:{key}",), True,
                                        "none")
        # 2 — индекс алиасов (зарегистрированный алиас авторитетен)
        hit = {self.canonical(e) for e in self._alias_index.get((namespace, normalize_alias(source_id)), set())
               if self.entities[e].active_at(t)}
        if len(hit) == 1:
            eid = hit.pop()
            return ResolutionResult(eid, ResolutionStatus.RESOLVED, 1.0, f"alias_index:{namespace}",
                                    [{"entity_id": eid, "score": 1.0}], (f"{namespace}:{source_id}",), False, "eligible")
        # 3 — перебор кандидатов и оценка
        w = self.policy.identity_weights
        names = [source_id, *[str(n) for n in ctx.get("names") or ()]]
        scored = []
        for e in self.entities.values():
            if e.merged_into or e.lifecycle_state not in ("active",) or not e.active_at(t):
                continue
            all_aliases = [a for als in e.aliases.values() for a in als] + [e.canonical_name]
            s_alias = max((alias_similarity(n, a) for n in names for a in all_aliases), default=0.0)
            s_det = 1.0 if any(str(e.keys.get(ns, "")).strip().lower() == str(k).strip().lower() for ns, k in keys.items()) \
                else 0.0
            s_sp, note = 0.0, None
            pos = ctx.get("position_world")
            if pos is not None and e.position_world is not None:
                sigma = self.policy.spatial_sigma_m.get(e.entity_type)
                if sigma is None:
                    note = f"no_spatial_sigma_for:{e.entity_type}"
                else:
                    d = math.dist(tuple(pos), tuple(e.position_world))
                    s_sp = math.exp(-d * d / (2 * float(sigma) ** 2))
            hint = ctx.get("class_hint")
            s_type = 1.0 if hint is not None and hint in (e.entity_type, e.ontology_class) else 0.0
            s_temp = 1.0 if self._confirmed.get((namespace, e.entity_id), 0) > 0 else 0.0
            comp = {"alias": s_alias, "det": s_det, "spatial": s_sp, "type": s_type, "temporal": s_temp}
            score = sum(float(w[k]) * v for k, v in comp.items())
            if score > 0:
                scored.append({"entity_id": e.entity_id, "score": score, "components": comp,
                               **({"note": note} if note else {})})
        scored.sort(key=lambda c: -c["score"])
        top = scored[:5]
        if not top or top[0]["score"] < th["candidate"]:
            return ResolutionResult(None, ResolutionStatus.NEW_ENTITY, top[0]["score"] if top else 0.0, "candidate_scan",
                                    top, (), True, "none")
        if len(top) > 1 and top[0]["score"] - top[1]["score"] < th["conflict_delta"]:
            return ResolutionResult(None, ResolutionStatus.CONFLICT, top[0]["score"], "candidate_scan", top, (), True, "none")
        best = top[0]
        if best["score"] >= th["resolved"]:
            return ResolutionResult(best["entity_id"], ResolutionStatus.RESOLVED, best["score"], "candidate_scan", top,
                                    (), False, "eligible")
        return ResolutionResult(best["entity_id"], ResolutionStatus.RESOLVED_PROBABILISTIC, best["score"],
                                "candidate_scan", top, (), True, "none")

    def confirm_match(self, namespace: str, entity_id: str, *, confirmed_by: str, knowledge_time: datetime) -> None:
        self._apply(self.journal.append("entity_match_confirmed", {"namespace": namespace, "entity_id": entity_id,
                                                                   "confirmed_by": text("confirmed_by", confirmed_by)},
                                        knowledge_time=knowledge_time))

    def quarantine_event(self, event: Mapping[str, Any], result: ResolutionResult, *, knowledge_time: datetime) -> str:
        """WM-05 §2.1 EntityResolutionQuarantine: ни утверждения, ни причинной связи, ни права на действие —
        только запись карантина и задача на проверку человеком."""
        qid = f"quarantine:{secrets.token_hex(6)}"
        reason = {ResolutionStatus.CONFLICT: "top_candidates_too_close_or_duplicate_key",
                  ResolutionStatus.NEW_ENTITY: "confidence_below_threshold"}.get(result.status, result.status.value)
        self._apply(self.journal.append("entity_quarantine", {
            "quarantine_id": qid, "raw_observation_ref": str(event.get("observation_id") or ""),
            "source_namespace": str(event.get("source_namespace") or ""),
            "source_entity_key": str(event.get("source_id") or ""), "candidate_entities": result.candidates,
            "ambiguity_reason": reason, "resolution_status": result.status.value, "score": result.score,
            "event": dict(event), "status": "pending", "review_task": f"review:{qid}"}, knowledge_time=knowledge_time))
        return qid

    def quarantine_record(self, quarantine_id: str) -> dict:
        for q in self.quarantine:
            if q["quarantine_id"] == quarantine_id:
                return q
        raise SpatialWorldError(f"unknown quarantine {quarantine_id}")

    def pending_quarantine(self) -> list[dict]:
        return [q for q in self.quarantine if q["status"] == "pending"]

    def resolve_quarantine(self, quarantine_id: str, *, entity_id: str, reviewer: str, knowledge_time: datetime) -> dict:
        q = self.quarantine_record(quarantine_id)
        if q["status"] != "pending":
            raise SpatialWorldError("quarantine already closed")
        if entity_id not in self.entities:
            raise SpatialWorldError(f"unknown entity {entity_id}")
        ns, key = q["source_namespace"], q["source_entity_key"]
        if ns in DETERMINISTIC_NAMESPACES:
            self.register_key(entity_id, ns, key, knowledge_time=knowledge_time)
        else:
            self.register_alias(entity_id, ns, key, knowledge_time=knowledge_time)
        self.confirm_match(ns, entity_id, confirmed_by=reviewer, knowledge_time=knowledge_time)
        self._apply(self.journal.append("entity_quarantine_status", {"quarantine_id": quarantine_id, "status": "resolved",
                                                                     "entity_id": entity_id,
                                                                     "reviewer": text("reviewer", reviewer)},
                                        knowledge_time=knowledge_time))
        return q

    def reject_quarantine(self, quarantine_id: str, *, reviewer: str, reason: str, knowledge_time: datetime) -> None:
        q = self.quarantine_record(quarantine_id)
        if q["status"] != "pending":
            raise SpatialWorldError("quarantine already closed")
        self._apply(self.journal.append("entity_quarantine_status", {"quarantine_id": quarantine_id, "status": "rejected",
                                                                     "reviewer": text("reviewer", reviewer),
                                                                     "reason": text("reason", reason)},
                                        knowledge_time=knowledge_time))

    # ---------------------------------------------------------------- слияние (обратимо)
    def merge(self, survivor: str, duplicate: str, *, evidence: tuple, approved_by: str, knowledge_time: datetime) -> dict:
        if not evidence or not all(str(x).strip() for x in evidence):
            raise SpatialWorldError("a merge needs evidence")
        text("approved_by", approved_by)
        for x in (survivor, duplicate):
            if x not in self.entities:
                raise SpatialWorldError(f"unknown entity {x}")
        if survivor == duplicate or self.canonical(survivor) == duplicate:
            raise SpatialWorldError("cannot merge an entity into itself")
        p = {"survivor": survivor, "duplicate": duplicate, "evidence": list(evidence), "approved_by": approved_by}
        self._apply(self.journal.append("entity_merge", p, knowledge_time=knowledge_time))
        return p

    def unmerge(self, duplicate: str, *, reason: str, approved_by: str, knowledge_time: datetime) -> None:
        if not self.entities[duplicate].merged_into:
            raise SpatialWorldError("entity is not merged")
        self._apply(self.journal.append("entity_unmerge", {"duplicate": duplicate, "reason": text("reason", reason),
                                                           "approved_by": text("approved_by", approved_by)},
                                        knowledge_time=knowledge_time))

    def children(self, entity_id: str) -> list[str]:
        return sorted(e.entity_id for e in self.entities.values() if e.parent_id == entity_id and not e.merged_into)


__all__ = ["DETERMINISTIC_NAMESPACES", "EntityIdentity", "EntityRegistry", "ResolutionResult", "ResolutionStatus",
           "alias_similarity", "normalize_alias"]
