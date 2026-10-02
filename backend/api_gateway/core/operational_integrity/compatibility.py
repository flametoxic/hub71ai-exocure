"""WM-05 §9: матрица совместимости версий.

    Compatible(a, b) = 1[Schema_a ~ Schema_b ∧ Ontology_a ~ Ontology_b ∧ Policy_a ~ Policy_b ∧ S… ]
    (формула в PDF обрезана по краю — проверяются все 10 компонент вектора версий из §9.1)
Исходы: compatible → использовать; compatible_with_adapter → преобразовать записанным адаптером;
incompatible → блок/отложить; unknown → на проверку человеку. Молчаливого приведения схем нет.
Проверку обязаны проходить повтор, импорт федерации, выборка памяти, симуляция и план.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Callable, Mapping, Optional

from ..world_model.reality_formulas import FormulaReference
from .context import DOC, IntegrityError

F_COMPATIBLE = FormulaReference("WM05-9.1-COMPATIBLE", DOC, 10,
                                "Compatible(a,b) = 1[Schema_a ~ Schema_b ∧ Ontology_a ~ Ontology_b ∧ Policy_a ~ Policy_b "
                                "∧ S… (cut off)]")


@dataclass(frozen=True)
class VersionVector:
    schema_version: Optional[str]
    ontology_version: Optional[str]
    topology_version: Optional[str]
    scm_version: Optional[str]
    dynamics_version: Optional[str]
    policy_version: Optional[str]
    skill_version: Optional[str]
    adapter_version: Optional[str]
    model_version: Optional[str]
    site_pack_version: Optional[str]


class Compatibility(str, Enum):
    COMPATIBLE = "compatible"
    COMPATIBLE_WITH_ADAPTER = "compatible_with_adapter"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


ACTION = {Compatibility.COMPATIBLE: "use", Compatibility.COMPATIBLE_WITH_ADAPTER: "transform_with_recorded_adapter",
          Compatibility.INCOMPATIBLE: "block_or_defer", Compatibility.UNKNOWN: "human_review"}
OPERATIONS = ("replay", "federation_import", "memory_retrieval", "simulation", "plan")


class CompatibilityMatrix:
    """Правила по компонентам: равенство версий, объявленная совместимость или зарегистрированный адаптер."""

    def __init__(self) -> None:
        self._compatible: set[tuple] = set()                  # (component, a, b)
        self._incompatible: set[tuple] = set()
        self._adapters: dict[tuple, Callable] = {}             # (component, a, b) → функция преобразования
        self.adapter_log: list[dict] = []

    def declare(self, component: str, a: str, b: str, *, compatible: bool) -> None:
        self._check(component)
        (self._compatible if compatible else self._incompatible).add((component, a, b))

    def register_adapter(self, component: str, a: str, b: str, fn: Callable, *, adapter_id: str) -> None:
        self._check(component)
        if not adapter_id.strip():
            raise IntegrityError("adapter needs an id (it is recorded on every use)")
        fn.adapter_id = adapter_id                              # type: ignore[attr-defined]
        self._adapters[(component, a, b)] = fn

    @staticmethod
    def _check(component: str) -> None:
        if component not in {f.name for f in fields(VersionVector)}:
            raise IntegrityError(f"unknown version component {component!r}")

    def component(self, name: str, a: Optional[str], b: Optional[str]) -> Compatibility:
        if a is None or b is None:
            return Compatibility.UNKNOWN
        if a == b or (name, a, b) in self._compatible or (name, b, a) in self._compatible:
            return Compatibility.COMPATIBLE
        if (name, a, b) in self._incompatible or (name, b, a) in self._incompatible:
            return Compatibility.INCOMPATIBLE
        if (name, a, b) in self._adapters:
            return Compatibility.COMPATIBLE_WITH_ADAPTER
        return Compatibility.UNKNOWN

    def check(self, source: VersionVector, target: VersionVector, *, operation: str,
              relevant: Optional[tuple] = None) -> dict:
        if operation not in OPERATIONS:
            raise IntegrityError(f"operation must be one of {OPERATIONS}")
        names = relevant or tuple(f.name for f in fields(VersionVector))
        per = {n: self.component(n, getattr(source, n), getattr(target, n)) for n in names}
        order = (Compatibility.INCOMPATIBLE, Compatibility.UNKNOWN, Compatibility.COMPATIBLE_WITH_ADAPTER,
                 Compatibility.COMPATIBLE)
        overall = next(o for o in order if o in per.values()) if per else Compatibility.COMPATIBLE
        return {"operation": operation, "result": overall.value, "action": ACTION[overall],
                "components": {k: v.value for k, v in per.items()}, "formula": F_COMPATIBLE}

    def transform(self, component: str, a: str, b: str, obj: Mapping) -> dict:
        fn = self._adapters.get((component, a, b))
        if fn is None:
            raise IntegrityError(f"no recorded adapter for {component} {a}→{b}: no silent coercion")
        out = fn(dict(obj))
        self.adapter_log.append({"component": component, "from": a, "to": b, "adapter_id": fn.adapter_id})
        return out


__all__ = ["ACTION", "Compatibility", "CompatibilityMatrix", "F_COMPATIBLE", "OPERATIONS", "VersionVector"]
