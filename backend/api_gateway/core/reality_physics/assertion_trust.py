"""Мастер-ТЗ B2 (стр. 10): доверие к утверждению / связи.

    Trust(a) = w_s S + w_q Q + w_f F + w_c C + w_v V

S — доверие источнику, Q — качество, F — свежесть, C — согласованность, V — проверка (все в [0, 1]).
Веса — из политики (неотрицательные, сумма 1; значений по умолчанию нет).
«Отсутствие исхода не делает наблюдение недействительным, но блокирует операционную проверку»:
V = None → вклад проверки 0, наблюдение остаётся действительным, operational_validation = blocked.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional

from ._base import PhysicsError, need, num, prob, ref

F_TRUST = ref("MASTER-B2-TRUST", 10, "Trust(a) = w_s S + w_q Q + w_f F + w_c C + w_v V")
KEYS = ("source", "quality", "freshness", "consistency", "verification")


def assertion_trust(*, source: float, quality: float, freshness: float, consistency: float,
                    verification: Optional[float], weights: Mapping[str, float]) -> dict:
    w = {k: num(f"w_{k}", need(weights, k, "trust weights"), low=0.0) for k in KEYS}
    if not math.isclose(sum(w.values()), 1.0, abs_tol=1e-9):
        raise PhysicsError("trust weights must sum to 1 (policy)")
    vals = {"source": prob("S", source), "quality": prob("Q", quality), "freshness": prob("F", freshness),
            "consistency": prob("C", consistency), "verification": 0.0 if verification is None else prob("V", verification)}
    return {"trust": sum(w[k] * vals[k] for k in KEYS), "observation_valid": True,
            "operational_validation": "blocked_missing_outcome" if verification is None else "available",
            "components": vals, "formula": F_TRUST.formula_id}


__all__ = ["F_TRUST", "KEYS", "assertion_trust"]
