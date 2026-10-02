"""WM-05 §1 и §7: контекст проекции, честный режим данных, отделение эфемерного контекста.

    Production Reality = Canonical Writes + Fused State + Causal Integrity + Version Compatibility + …
    (строка «Core rule» в PDF обрезана по краю страницы — перечислено то, что видно)

§1.2 Любое изменение модели мира требует ProjectionContext(pipeline_id, trace_id, source_assertion_ids,
schema_version, data_mode); без источников — CanonicalWriteViolation и событие аудита wm.write_blocked.
§1.4 Понимание LLM, стиль, композиция ответа, брифинг UI, временные подсказки и кэш предпочтений сессии
живут в эфемерных контекстах и не могут записать ничего в модель мира.
§7 data_mode ∈ {synthetic, replay, shadow, production} обязателен во всех контрактах; при
data_mode ≠ production выход не может называться «live», «verified in field», «production performance».
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

from ..world_model.reality_formulas import FormulaReference

DOC = "EXO_WM05_operational_integrity_reality_deployment"
F_CORE_RULE = FormulaReference("WM05-CORE-RULE", DOC, 1,
                               "Production Reality = Canonical Writes + Fused State + Causal Integrity + Version "
                               "Compatib… (cut off at the page edge in the PDF)")


class IntegrityError(ValueError):
    pass


class CanonicalWriteViolation(RuntimeError):
    """WM-05 §1.2."""


class DataMode(str, Enum):
    SYNTHETIC = "synthetic"
    REPLAY = "replay"
    SHADOW = "shadow"
    PRODUCTION = "production"

    @property
    def rank(self) -> int:
        return ("synthetic", "replay", "shadow", "production").index(self.value)


def weakest_mode(modes: Iterable[Any]) -> DataMode | None:
    ms = [DataMode(m) for m in modes]
    return min(ms, key=lambda m: m.rank) if ms else None


@dataclass(frozen=True)
class ProjectionContext:
    pipeline_id: str
    trace_id: str
    source_assertion_ids: tuple[str, ...]
    schema_version: str
    data_mode: DataMode

    def __post_init__(self) -> None:
        for n in ("pipeline_id", "trace_id", "schema_version"):
            if not str(getattr(self, n) or "").strip():
                raise CanonicalWriteViolation(f"projection context needs {n}")
        ids = tuple(str(x) for x in self.source_assertion_ids if str(x).strip())
        if not ids:
            raise CanonicalWriteViolation("projection context needs source_assertion_ids (WM-05 §1.2)")
        object.__setattr__(self, "source_assertion_ids", ids)
        object.__setattr__(self, "data_mode", DataMode(self.data_mode))

    def to_dict(self) -> dict:
        return {"pipeline_id": self.pipeline_id, "trace_id": self.trace_id,
                "source_assertion_ids": list(self.source_assertion_ids), "schema_version": self.schema_version,
                "data_mode": self.data_mode.value}


_FORBIDDEN_LABELS = (r"\blive\b", r"verified in field", r"production performance", r"в реальном времени",
                     r"проверено на объекте", r"боевые показатели")


def truthful_label(text: str, data_mode: Any) -> str:
    """UI-правило §7: не-production выход не может выдавать себя за живой/полевой."""
    mode = DataMode(data_mode)
    if mode is not DataMode.PRODUCTION:
        for p in _FORBIDDEN_LABELS:
            if re.search(p, str(text), flags=re.IGNORECASE):
                raise IntegrityError(f"label {text!r} is not allowed for data_mode={mode.value} (WM-05 §7)")
    return f"{text} [{mode.value}]"


def require_data_mode(obj: Mapping[str, Any], name: str = "object") -> DataMode:
    try:
        return DataMode(obj["data_mode"])
    except (KeyError, ValueError) as exc:
        raise IntegrityError(f"{name} must declare data_mode ∈ synthetic/replay/shadow/production (WM-05 §7)") from exc


@dataclass
class EphemeralSessionContext:
    """§1.4: понимание LLM, стиль, композиция ответа, брифинг UI, подсказки, кэш предпочтений сессии.
    Ничего отсюда не становится фактом о внешнем мире; записи в модель мира у этого объекта нет."""

    session_id: str
    llm_understanding: dict = field(default_factory=dict)
    personality_style: dict = field(default_factory=dict)
    response_composition: dict = field(default_factory=dict)
    ui_briefing: dict = field(default_factory=dict)
    cognitive_hints: list = field(default_factory=list)
    preference_cache: dict = field(default_factory=dict)

    def write_to_world_model(self, *_: Any, **__: Any) -> None:
        raise CanonicalWriteViolation("ephemeral session context cannot write canonical world state (WM-05 §1.4)")


WorkingMemoryContext = EphemeralSessionContext
UIProjectionContext = EphemeralSessionContext

__all__ = ["CanonicalWriteViolation", "DataMode", "EphemeralSessionContext", "F_CORE_RULE", "IntegrityError",
           "ProjectionContext", "UIProjectionContext", "WorkingMemoryContext", "require_data_mode", "truthful_label",
           "weakest_mode"]
