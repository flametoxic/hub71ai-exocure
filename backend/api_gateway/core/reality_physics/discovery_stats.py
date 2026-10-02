"""Мастер-ТЗ B3 (стр. 10): статистика поиска причинных связей внутри физических границ и ансамбль.

    r(X,Y) = cov(X,Y) / √(var X · var Y)
    r(X,Y | Z) = (r_XY − r_XZ r_YZ) / √((1 − r²_XZ)(1 − r²_YZ))
    lag*(X→Y) = argmax_τ CCF(X, Y, τ)

Статистика считается ТОЛЬКО для пар из ограниченного множества F26 (spatial_world.discovery): пара вне
пересечения «путь графа знаний ∩ досягаемость ∩ окно ∩ онтология» до корреляции не доходит — корреляция 0.99
без топологического пути отвергается. Допуск в ансамбль: ≥ 2 независимых метода + физические границы +
проверка здравого смысла. Максимальный статус результата — observational_candidate.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional

import numpy as np

from ._base import PhysicsError, num, ref

F_PEARSON = ref("MASTER-B3-PEARSON", 10, "r(X,Y) = cov(X,Y)/√(var(X)var(Y))")
F_PARTIAL = ref("MASTER-B3-PARTIAL", 10, "r(X,Y|Z) = (r_XY − r_XZ r_YZ)/√((1−r²_XZ)(1−r²_YZ))")
F_LAG = ref("MASTER-B3-LAG", 10, "lag*(X→Y) = argmax_τ CCF(X,Y,τ)")
MAX_STATUS = "observational_candidate"


def _series(name, v) -> np.ndarray:
    a = np.asarray(v, float).reshape(-1)
    if a.size < 3 or not np.all(np.isfinite(a)):
        raise PhysicsError(f"{name}: at least 3 finite samples required")
    return a


def pearson(x, y) -> float:
    x, y = _series("X", x), _series("Y", y)
    if x.size != y.size:
        raise PhysicsError("X and Y must be aligned (same length)")
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        raise PhysicsError("constant series: correlation undefined")
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def partial_correlation(x, y, z) -> float:
    rxy, rxz, ryz = pearson(x, y), pearson(x, z), pearson(y, z)
    den = (1 - rxz ** 2) * (1 - ryz ** 2)
    if den <= 0:
        raise PhysicsError("Z is collinear with X or Y: partial correlation undefined")
    return float((rxy - rxz * ryz) / np.sqrt(den))


def ccf(x, y, lag: int) -> float:
    """CCF(X,Y,τ) = corr(X_t, Y_{t+τ}) по перекрывающейся части (τ ≥ 0 — Y отстаёт от X)."""
    x, y = _series("X", x), _series("Y", y)
    if lag >= 0:
        a, b = x[: x.size - lag], y[lag:]
    else:
        a, b = x[-lag:], y[: y.size + lag]
    return pearson(a, b)


def best_lag(x, y, *, max_lag: int, min_lag: int = 0, sample_period_s: Optional[float] = None) -> dict:
    """lag* = argmax_τ CCF в диапазоне [min_lag, max_lag] (диапазон — из политики/физики, не угадывается)."""
    if int(max_lag) < int(min_lag) or len(np.asarray(x)) - int(max_lag) < 3:
        raise PhysicsError("invalid lag range for the series length")
    vals = {t: ccf(x, y, t) for t in range(int(min_lag), int(max_lag) + 1)}
    t = max(vals, key=lambda k: vals[k])
    out = {"lag_steps": t, "ccf": vals[t], "formula": F_LAG.formula_id}
    if sample_period_s is not None:
        out["lag_s"] = t * num("sample_period_s", sample_period_s, low=0.0, strict_low=True)
    return out


def pair_statistics(x_key: tuple, y_key: tuple, x, y, *, bounded: Mapping, z=None, max_lag: int = 0) -> dict:
    """Статистика для пары только если пара в ограниченном множестве F26; иначе отказ до статистики."""
    allowed = {(tuple(p["x"]), tuple(p["y"])) for p in bounded["pairs"]}
    if (tuple(x_key), tuple(y_key)) not in allowed and (tuple(y_key), tuple(x_key)) not in allowed:
        return {"tested": False, "reason": "outside_physics_bounds_F26: no statistics computed", "status": None}
    out = {"tested": True, "r": pearson(x, y), "status": MAX_STATUS}
    if z is not None:
        out["partial_r"] = partial_correlation(x, y, z)
    if max_lag > 0:
        out["lag"] = best_lag(x, y, max_lag=max_lag)
    return out


def ensemble_admission(*, methods_agreeing: Iterable[str], independent_groups: Mapping[str, str], physics_bounded: bool,
                       sanity_passed: bool) -> dict:
    """≥ 2 независимых метода (разные группы независимости) + физические границы + проверка здравого смысла."""
    groups = {independent_groups.get(m, m) for m in methods_agreeing}
    reasons = []
    if len(groups) < 2:
        reasons.append("fewer_than_two_independent_methods")
    if physics_bounded is not True:
        reasons.append("outside_physics_bounds")
    if sanity_passed is not True:
        reasons.append("sanity_check_failed")
    return {"admitted": not reasons, "status": MAX_STATUS if not reasons else None, "reasons": reasons,
            "graph_authority": False, "note": "discovery never exceeds observational_candidate"}


__all__ = ["F_LAG", "F_PARTIAL", "F_PEARSON", "MAX_STATUS", "best_lag", "ccf", "ensemble_admission",
           "partial_correlation", "pearson", "pair_statistics"]
