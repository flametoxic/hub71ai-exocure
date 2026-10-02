"""Порты пространственной модели мира для Sovereign Runtime (IdentityPort, FactPort).

Так исполнение опирается на ту же модель мира, что и рассуждение:
  • resolved(entity_id) — сущность каноническая, активна, не слита, не кандидат;
  • fact("entity_id#attribute#expected") — предусловие: текущее значение равно ожидаемому; время наблюдения
    = время знания; healthy = не устарело, не в конфликте и получено с правом на действие.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

from ..world_model.assertions import QualityStatus
from .service import SpatialWorldModel


class WorldModelIdentityPort:
    def __init__(self, model: SpatialWorldModel) -> None:
        self.model = model

    def resolved(self, entity_id: str) -> bool:
        e = self.model.entities.entities.get(entity_id)
        return e is not None and e.lifecycle_state == "active" and not e.merged_into


class WorldModelFactPort:
    def __init__(self, model: SpatialWorldModel, *, clock: Callable[[], datetime]) -> None:
        self.model, self.clock = model, clock

    def fact(self, name: str):
        from ..sovereign_runtime.pipeline import FactStatus

        parts = str(name).split("#")
        if len(parts) != 3:
            return None
        entity_id, attribute, expected = parts
        try:
            cur = self.model.ledger.get_current(entity_id, attribute, now=self.clock())
        except Exception:
            return None
        a = cur.get("assertion")
        if a is None:
            return None
        healthy = cur["status"] == "known" and a.quality_status is not QualityStatus.STALE \
            and a.action_authority == "eligible"
        return FactStatus(holds=str(a.value) == expected, observed_at=a.knowledge_time, healthy=healthy)


def ports(model: SpatialWorldModel, *, clock: Optional[Callable[[], datetime]] = None):
    from datetime import timezone
    return WorldModelIdentityPort(model), WorldModelFactPort(model, clock=clock or (lambda: datetime.now(timezone.utc)))


__all__ = ["WorldModelFactPort", "WorldModelIdentityPort", "ports"]
