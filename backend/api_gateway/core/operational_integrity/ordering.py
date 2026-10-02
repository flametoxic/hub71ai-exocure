"""WM-05 §10: распределённый порядок событий и служба причинности.

Событие несёт event_time, knowledge_time, ingested_time, processed_time, acted_time, номер последовательности
источника, логические часы, неопределённость часов и причинных родителей.
    new → accepted → duplicate | late | out_of_order | replayed | causally_waiting → projected | quarantined | rejected
    Accept(e) = SignatureValid ∧ SequenceValid ∧ TimeWithinPolicy ∧ NotReplay ∧ SchemaV… (обрезано в PDF)
Частичный порядок — по причинным родителям и логическим часам (Лэмпорт), когда порядок по стенным часам
неоднозначен. Опоздавшее событие может уточнить историческое убеждение, но не переписывает молча контекст
уже исполненного решения.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from ..world_model.reality_formulas import FormulaReference
from .context import DOC, IntegrityError

F_ACCEPT = FormulaReference("WM05-10-ACCEPT", DOC, 11,
                            "Accept(e) = SignatureValid ∧ SequenceValid ∧ TimeWithinPolicy ∧ NotReplay ∧ SchemaV… (cut off)")
STATES = ("new", "accepted", "duplicate", "late", "out_of_order", "replayed", "causally_waiting", "projected",
          "quarantined", "rejected")


@dataclass
class OrderedEvent:
    event_id: str
    source_id: str
    sequence: int
    event_time: datetime
    knowledge_time: datetime
    ingested_time: datetime
    logical_clock: int
    clock_uncertainty_s: float
    causal_parents: tuple = ()
    schema_version: str = ""
    signature_valid: bool = False
    nonce: str = ""
    processed_time: Optional[datetime] = None
    acted_time: Optional[datetime] = None
    state: str = "new"
    reasons: list = field(default_factory=list)


class EventOrderingService:
    def __init__(self, *, max_clock_skew_s: float, late_after_s: float, schema_versions: tuple,
                 project: Callable[[OrderedEvent], str]) -> None:
        if max_clock_skew_s <= 0 or late_after_s <= 0 or not schema_versions:
            raise IntegrityError("clock skew, lateness window and accepted schema versions come from policy")
        self.max_skew, self.late_after, self.schemas, self._project = max_clock_skew_s, late_after_s, set(schema_versions), project
        self.lamport = 0
        self.seen: dict[str, OrderedEvent] = {}
        self._last_seq: dict[str, int] = {}
        self._nonces: set = set()
        self.waiting: dict[str, OrderedEvent] = {}
        self.decision_contexts: dict[str, dict] = {}         # decision_id → замороженный контекст
        self.history_updates: list[dict] = []

    def accept(self, e: OrderedEvent, *, now: datetime) -> OrderedEvent:
        if e.event_id in self.seen:
            e.state = "duplicate"
            return e
        checks = {"SignatureValid": e.signature_valid is True,
                  "SequenceValid": e.sequence > 0,
                  "TimeWithinPolicy": abs((e.knowledge_time - now).total_seconds()) <= self.max_skew + e.clock_uncertainty_s
                  and e.event_time <= now,
                  "NotReplay": (e.source_id, e.nonce) not in self._nonces,
                  "SchemaValid": e.schema_version in self.schemas}
        failed = [k for k, v in checks.items() if not v]
        if failed:
            e.state = "replayed" if failed == ["NotReplay"] else "rejected"
            e.reasons = failed
            return e
        self._nonces.add((e.source_id, e.nonce))
        self.lamport = max(self.lamport, e.logical_clock) + 1
        e.logical_clock = self.lamport
        e.state = "accepted"
        last = self._last_seq.get(e.source_id, 0)
        if e.sequence <= last:
            e.state, e.reasons = "out_of_order", [f"sequence {e.sequence} ≤ last {last}"]
        elif (now - e.event_time).total_seconds() > self.late_after:
            e.state = "late"
        self._last_seq[e.source_id] = max(last, e.sequence)
        missing = [p for p in e.causal_parents if p not in self.seen or self.seen[p].state not in ("projected",)]
        self.seen[e.event_id] = e
        if missing:
            e.state, e.reasons = "causally_waiting", [f"waiting_for:{p}" for p in missing]
            self.waiting[e.event_id] = e
            return e
        self._do_project(e, now)
        self._release(now)
        return e

    def _do_project(self, e: OrderedEvent, now: datetime) -> None:
        was_late = e.state in ("late", "out_of_order")
        result = self._project(e)
        if result not in ("projected", "quarantined", "rejected"):
            raise IntegrityError(f"projection returned {result!r}")
        e.state, e.processed_time = result, now
        if was_late and result == "projected":
            touched = [d for d, ctx in self.decision_contexts.items() if ctx["as_of"] >= e.event_time]
            self.history_updates.append({"event_id": e.event_id, "updates_history_before": e.event_time.isoformat(),
                                         "executed_decisions_not_rewritten": touched})

    def _release(self, now: datetime) -> None:
        progress = True
        while progress:
            progress = False
            for eid, w in list(self.waiting.items()):
                if all(p in self.seen and self.seen[p].state == "projected" for p in w.causal_parents):
                    del self.waiting[eid]
                    w.reasons = []
                    self._do_project(w, now)
                    progress = True

    def freeze_decision_context(self, decision_id: str, *, as_of: datetime, snapshot_id: str) -> None:
        """Контекст исполненного решения неизменяем: поздние события его не переписывают."""
        if decision_id in self.decision_contexts:
            raise IntegrityError("decision context is immutable once frozen")
        self.decision_contexts[decision_id] = {"as_of": as_of, "snapshot_id": snapshot_id}


__all__ = ["EventOrderingService", "F_ACCEPT", "OrderedEvent", "STATES"]
