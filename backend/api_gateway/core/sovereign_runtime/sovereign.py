"""Суверенный режим (§1, §2), деградация (§9) и подписанная отложенная синхронизация (§1, тест 8).

Инвариант §1: без внешнего канала разрешены только классы действий, чьи модели, политики, свидетельства
и полномочия доступны ЛОКАЛЬНО и свежи. При разрыве: удалённое обучение и внешний поиск выключены, локальная
жёсткая безопасность остаётся, полномочия пересчитываются из локальной уверенности и политики.

§9: журнал недоступен → runtime degraded, физическая запись заблокирована; мир устарел → уровень автономии
ниже; артефакт политики невалиден → остаётся только локальное аппаратно-безопасное поведение; модель
недоступна → никакой LLM-«заглушки» под видом причинной рекомендации.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Mapping, Optional

from .autonomy import AutonomyTier
from .contracts import SovereignError, aware, digest, text
from .trust import ExecutionJournal, merkle_root, sign_artifact, verify_artifact


class Connectivity(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"


class RuntimeStatus(str, Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class LocalArtifacts:
    """Что должно быть локально и свежо, чтобы класс действий жил без облака."""

    action_class: str
    model_version: str
    model_fresh_until: datetime
    policy_hash: str
    evidence_fresh_until: datetime
    authority_ref: str
    max_tier_disconnected: AutonomyTier

    def __post_init__(self) -> None:
        for n in ("action_class", "model_version", "policy_hash", "authority_ref"):
            text(n, getattr(self, n))
        object.__setattr__(self, "model_fresh_until", aware("model_fresh_until", self.model_fresh_until))
        object.__setattr__(self, "evidence_fresh_until", aware("evidence_fresh_until", self.evidence_fresh_until))
        object.__setattr__(self, "max_tier_disconnected", AutonomyTier(self.max_tier_disconnected))


#: источники рекомендаций, которые считаются причинными (не LLM). Остальное — информационно.
CAUSAL_SOURCES = ("causal_engine", "scm", "domain_dynamics", "operator")


@dataclass
class SovereignState:
    connectivity: Connectivity = Connectivity.CONNECTED
    model_available: bool = True
    world_state_fresh: bool = True
    local: dict = field(default_factory=dict)      # action_class → LocalArtifacts
    events: list = field(default_factory=list)

    # ---------------------------------------------------------------- связь
    def set_connectivity(self, c: Connectivity, *, at: datetime) -> None:
        c = Connectivity(c)
        if c is not self.connectivity:
            self.events.append({"at": aware("at", at).isoformat(), "connectivity": c.value})
        self.connectivity = c

    @property
    def remote_learning_allowed(self) -> bool:
        return self.connectivity is Connectivity.CONNECTED

    @property
    def external_retrieval_allowed(self) -> bool:
        return self.connectivity is Connectivity.CONNECTED

    def register_local(self, a: LocalArtifacts) -> None:
        self.local[a.action_class] = a

    # ---------------------------------------------------------------- инвариант §1
    def authority(self, action_class: str, *, active_policy_hash: Optional[str], now: datetime
                  ) -> tuple[bool, Optional[AutonomyTier], tuple[str, ...]]:
        """(разрешён ли класс, потолок уровня или None = без потолка, причины)."""
        now = aware("now", now)
        reasons = []
        # недоступность модели не блокирует класс целиком (оператор может действовать, жёсткий контур жив) —
        # она запрещает только выдавать не-причинный источник за причинную рекомендацию (recommendation_admissible)
        if self.connectivity is Connectivity.CONNECTED:
            return True, None, ()
        a = self.local.get(action_class)
        if a is None:
            return False, AutonomyTier.SHADOW, ("disconnected:action_class_not_locally_available",)
        if a.model_fresh_until <= now:
            reasons.append("disconnected:local_model_stale")
        if a.evidence_fresh_until <= now:
            reasons.append("disconnected:local_evidence_stale")
        if active_policy_hash is None or active_policy_hash != a.policy_hash:
            reasons.append("disconnected:local_policy_missing_or_different")
        return (not reasons), a.max_tier_disconnected, tuple(reasons)

    def recommendation_admissible(self, source: str) -> tuple[bool, tuple[str, ...]]:
        """§9: при недоступной модели никакая LLM-заглушка не выдаётся за причинную рекомендацию."""
        if source not in CAUSAL_SOURCES:
            return False, (f"non_causal_recommendation_source:{source}",)
        if source != "operator" and not self.model_available:
            return False, ("model_unavailable:no_fallback_may_masquerade_as_causal",)
        return True, ()


def runtime_status(journal: ExecutionJournal) -> RuntimeStatus:
    return RuntimeStatus.NORMAL if journal.available else RuntimeStatus.DEGRADED


# -------------------------------------------------------------------- синхронизация
@dataclass(frozen=True)
class SignedSyncBatch:
    site: str
    from_seq: int
    to_seq: int
    previous_digest: str          # дайджест последней уже синхронизированной записи ("" для первой пачки)
    entries: tuple
    merkle_root: str
    signature: str


def export_sync_batch(journal: ExecutionJournal, *, site: str, since_seq: int, key: bytes) -> SignedSyncBatch:
    all_e = journal.entries()
    if since_seq < 0 or since_seq > len(all_e):
        raise SovereignError("since_seq outside journal")
    entries = tuple(all_e[since_seq:])
    prev = all_e[since_seq - 1]["entry_digest"] if since_seq > 0 else ""
    root = merkle_root([e["entry_digest"] for e in entries])
    body = {"site": site, "from": since_seq, "to": since_seq + len(entries), "previous": prev, "root": root}
    return SignedSyncBatch(site, since_seq, since_seq + len(entries), prev, entries, root, sign_artifact(body, key))


class ReconciliationLedger:
    """Приёмная сторона (облако/центр): принимает пачки строго по порядку, проверяет подпись площадки,
    непрерывность цепочки, целостность каждой записи и корень Меркла. Развилка истории — отказ."""

    def __init__(self, *, site_keys: Mapping[str, bytes]) -> None:
        self._keys = dict(site_keys)
        self.last_seq: dict[str, int] = {}
        self.last_digest: dict[str, str] = {}
        self.accepted: dict[str, list] = {}

    def reconcile(self, batch: SignedSyncBatch) -> tuple[bool, tuple[str, ...]]:
        reasons = []
        key = self._keys.get(batch.site)
        body = {"site": batch.site, "from": batch.from_seq, "to": batch.to_seq, "previous": batch.previous_digest,
                "root": batch.merkle_root}
        if key is None or not verify_artifact(body, batch.signature, key):
            reasons.append("sync_signature_invalid")
        if batch.from_seq != self.last_seq.get(batch.site, 0):
            reasons.append("sync_gap_or_overlap")
        if batch.previous_digest != self.last_digest.get(batch.site, ""):
            reasons.append("sync_history_fork")
        prev = batch.previous_digest
        for i, e in enumerate(batch.entries):
            b = {k: v for k, v in e.items() if k != "entry_digest"}
            if e.get("seq") != batch.from_seq + i or b.get("previous") != prev or digest(b) != e.get("entry_digest"):
                reasons.append(f"sync_entry_tampered:{batch.from_seq + i}")
                break
            prev = e["entry_digest"]
        if merkle_root([e["entry_digest"] for e in batch.entries]) != batch.merkle_root:
            reasons.append("sync_merkle_root_mismatch")
        if len(batch.entries) != batch.to_seq - batch.from_seq:
            reasons.append("sync_count_mismatch")
        if reasons:
            return False, tuple(reasons)
        self.accepted.setdefault(batch.site, []).extend(batch.entries)
        self.last_seq[batch.site] = batch.to_seq
        if batch.entries:
            self.last_digest[batch.site] = batch.entries[-1]["entry_digest"]
        return True, ()


__all__ = ["CAUSAL_SOURCES", "Connectivity", "LocalArtifacts", "ReconciliationLedger", "RuntimeStatus",
           "SignedSyncBatch", "SovereignState", "export_sync_batch", "runtime_status"]
