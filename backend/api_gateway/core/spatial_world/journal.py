"""Единый долговечный журнал модели мира: только INSERT (мастер-ТЗ A1 «append-only evidence ledger in SQL»).

Каждая запись — (seq, kind, knowledge_time, payload, previous, digest); цепочка дайджестов делает подмену
видимой. В SQLite UPDATE/DELETE запрещены триггерами. Всё состояние модели (утверждения, системы
координат, топология, сущности, геометрия) восстанавливается повтором журнала — поэтому любой ответ
«что CURE знала в момент K» воспроизводим. Отказ долговечной записи — не тихий: журнал переходит в
состояние degraded, запись не принимается, а сервис сообщает data_mode = degraded (Gap Closure §11,
WM Spec §2.17/§2.20).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Optional

from .policy import SpatialWorldError, aware


def canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_default)


def _default(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat()
    if hasattr(v, "value") and hasattr(v, "name"):          # Enum
        return v.value
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (set, frozenset, tuple)):
        return list(v)
    if hasattr(v, "items"):
        return dict(v)
    raise TypeError(f"not JSON-serialisable: {type(v).__name__}")


def digest(payload: Any) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class JournalRow:
    seq: int
    kind: str
    knowledge_time: datetime
    payload: dict
    previous: str
    digest: str


class WorldJournal:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.available = True
        self.degraded_reason: Optional[str] = None
        self._rows: list[JournalRow] = []
        self._lock = threading.Lock()
        self._db: Optional[sqlite3.Connection] = None
        if path:
            self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
            self._db.executescript("""
                PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
                CREATE TABLE IF NOT EXISTS world_journal (
                    seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, knowledge_time TEXT NOT NULL,
                    payload TEXT NOT NULL, previous TEXT NOT NULL, digest TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS world_journal_no_update BEFORE UPDATE ON world_journal
                    BEGIN SELECT RAISE(ABORT, 'world_journal is append-only'); END;
                CREATE TRIGGER IF NOT EXISTS world_journal_no_delete BEFORE DELETE ON world_journal
                    BEGIN SELECT RAISE(ABORT, 'world_journal is append-only'); END;
            """)
            for seq, kind, kt, payload, prev, dg in self._db.execute(
                    "SELECT seq, kind, knowledge_time, payload, previous, digest FROM world_journal ORDER BY seq"):
                self._rows.append(JournalRow(seq, kind, datetime.fromisoformat(kt), json.loads(payload), prev, dg))
            if not self.verify():
                raise SpatialWorldError("world journal chain is broken: refusing to start (tampered history)")

    def append(self, kind: str, payload: dict, *, knowledge_time: datetime) -> JournalRow:
        kt = aware("knowledge_time", knowledge_time)
        with self._lock:
            if not self.available:
                raise SpatialWorldError(f"world journal degraded ({self.degraded_reason}): write refused")
            seq = len(self._rows)
            prev = self._rows[-1].digest if self._rows else ""
            body = {"seq": seq, "kind": kind, "knowledge_time": kt.isoformat(), "payload": payload, "previous": prev}
            d = digest(body)
            clean = json.loads(canonical(payload))
            if self._db is not None:
                try:
                    self._db.execute("INSERT INTO world_journal VALUES (?,?,?,?,?,?)",
                                     (seq, kind, kt.isoformat(), canonical(clean), prev, d))
                except sqlite3.Error as exc:
                    self.available, self.degraded_reason = False, f"durable write failed: {exc}"
                    raise SpatialWorldError(self.degraded_reason) from exc
            row = JournalRow(seq, kind, kt, clean, prev, d)
            self._rows.append(row)
            return row

    def verify(self) -> bool:
        prev = ""
        for r in self._rows:
            body = {"seq": r.seq, "kind": r.kind, "knowledge_time": r.knowledge_time.isoformat(),
                    "payload": r.payload, "previous": prev}
            if r.previous != prev or digest(body) != r.digest:
                return False
            prev = r.digest
        return True

    def rows(self, kinds: Optional[Iterable[str]] = None) -> tuple[JournalRow, ...]:
        ks = None if kinds is None else set(kinds)
        return tuple(r for r in self._rows if ks is None or r.kind in ks)

    def head(self) -> str:
        return self._rows[-1].digest if self._rows else ""

    def __len__(self) -> int:
        return len(self._rows)


__all__ = ["JournalRow", "WorldJournal", "canonical", "digest"]
