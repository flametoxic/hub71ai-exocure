"""Оценки причинного эффекта по данным: frontdoor (Pearl) и инструментальные переменные (2SLS).

Граф отвечает на вопрос «идентифицируем ли эффект и каким способом», данные — «какой он».
Оба метода — наблюдательная идентификация: максимум право `review` (CEOS стр. 9:
ActionAuthority ≠ CausalConfidence).
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .causal_graph_identification import _closure, _nodes, d_separated
from .reality_formulas import FormulaReference, FormulaResult

PEARL = "Pearl, Causality (2009), §3.3.2 front-door criterion"
F_FRONTDOOR_CRITERION = FormulaReference(
    "CAUSAL-FRONTDOOR-CRITERION", PEARL, 0,
    "M intercepts all directed X->Y paths; no unblocked backdoor X->M; all backdoor M->Y blocked by X")
F_FRONTDOOR = FormulaReference(
    "CAUSAL-FRONTDOOR-ADJUSTMENT", PEARL, 0,
    "E[Y|do(X=x)] = sum_m P(m|x) sum_x' E[Y|m,x'] P(x')")
F_IV = FormulaReference(
    "CAUSAL-IV-2SLS", "Angrist & Pischke, Mostly Harmless Econometrics (2009), §4.1", 0,
    "X_hat = Z(Z'Z)^-1 Z'X; beta = (X_hat'X)^-1 X_hat'Y; first-stage F for instrument strength")
F_IV_CRITERION = FormulaReference(
    "CAUSAL-IV-CRITERION", PEARL, 0,
    "Z not d-separated from X (relevance); Z d-separated from Y given W in G with X's outgoing edges removed (exclusion)")

MAX_PLANNING_USE = "review"


def _directed_paths(edges, src: str, dst: str) -> list[list[str]]:
    children: dict[str, list[str]] = {}
    for a, b in edges:
        children.setdefault(a, []).append(b)
    out, stack = [], [(src, [src])]
    while stack:
        node, path = stack.pop()
        for c in children.get(node, ()):
            if c in path:
                continue
            if c == dst:
                out.append(path + [c])
            else:
                stack.append((c, path + [c]))
    return out


def frontdoor_criterion(edges: Iterable[tuple[str, str]], treatment: str, outcome: str,
                        mediators: Iterable[str], *, latent: Iterable[str] = ()) -> FormulaResult:
    """Критерий front-door по графу. Латентные узлы в M недопустимы (M должен быть измерен)."""
    edges, m, latent = list(edges), set(mediators), set(latent)
    nodes = _nodes(edges)
    if {treatment, outcome} - nodes or not m or m - nodes:
        raise ValueError("treatment, outcome and every mediator must be nodes of the graph")
    if {treatment, outcome} & m:
        raise ValueError("mediator set cannot contain X or Y")
    reasons = [f"latent_mediator:{v}" for v in sorted(m & latent)]
    # 1) M перехватывает все направленные пути X -> Y
    if any(not (set(p[1:-1]) & m) for p in _directed_paths(edges, treatment, outcome)):
        reasons.append("directed_path_not_intercepted")
    if not _directed_paths(edges, treatment, outcome):
        reasons.append("no_directed_path")
    # 2) нет открытого backdoor-пути X -> M (граф без исходящих из X рёбер)
    no_out_x = [(a, b) for a, b in edges if a != treatment]
    if not d_separated(no_out_x, {treatment}, m, set(), nodes=nodes):
        reasons.append("open_backdoor_x_to_m")
    # 3) все backdoor-пути M -> Y блокируются X (граф без исходящих из M рёбер)
    no_out_m = [(a, b) for a, b in edges if a not in m]
    if not d_separated(no_out_m, m, {outcome}, {treatment}, nodes=nodes):
        reasons.append("backdoor_m_to_y_not_blocked_by_x")
    return FormulaResult(not reasons, F_FRONTDOOR_CRITERION, {"reasons": tuple(reasons)})


def _discrete(name: str, values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    return arr


def frontdoor_effect(*, x, m, y, x_value, min_cell: int = 1) -> FormulaResult:
    """E[Y|do(X=x)] = Σ_m P(m|x) Σ_x' E[Y|m,x'] P(x') по наблюдениям (X и M дискретные, Y числовой).

    Пустая ячейка (m, x') делает оценку неидентифицируемой на этих данных (нарушение positivity) —
    ошибка, а не молчаливый ноль.
    """
    x, m = _discrete("x", x), _discrete("m", m)
    y = np.asarray(y, float)
    if not (x.size == m.size == y.size) or not np.all(np.isfinite(y)):
        raise ValueError("x, m, y must align and y must be finite")
    if not np.any(x == x_value):
        raise ValueError("no observations with X = x_value (positivity violated)")
    n = x.size
    xs, ms = np.unique(x), np.unique(m)
    p_x = {xv: float(np.mean(x == xv)) for xv in xs}
    sel = x == x_value
    total, terms = 0.0, {}
    for mv in ms:
        p_m_given_x = float(np.mean(m[sel] == mv))
        if p_m_given_x == 0.0:
            continue
        inner = 0.0
        for xv in xs:
            cell = (m == mv) & (x == xv)
            if cell.sum() < min_cell:
                raise ValueError(f"empty cell m={mv!r}, x'={xv!r}: E[Y|m,x'] not estimable (positivity)")
            inner += float(y[cell].mean()) * p_x[xv]
        terms[str(mv)] = p_m_given_x * inner
        total += terms[str(mv)]
    return FormulaResult(total, F_FRONTDOOR, {"terms": terms, "n": n,
                                              "allowed_planning_use": MAX_PLANNING_USE})


F_BACKDOOR_DATA = FormulaReference(
    "CCIA-2.1-BACKDOOR-EXPECTATION-DATA",
    "EXO-causal-reasoning-memory-planning-integration-addendum",
    2,
    "E[Y|do(X=x)] = sum_z E[Y|X=x,Z=z] P(Z=z), empirical strata",
)


def backdoor_effect_from_data(*, x, y, z, x_value, min_cell: int = 1) -> FormulaResult:
    """E[Y|do(X=x)] = Σ_z E[Y|X=x,Z=z] P(Z=z) по наблюдениям (Z дискретные, Y числовой).

    ``z`` — {имя: столбец} (пустой набор -> E[Y|X=x]). Пустая страта (x, z) — нарушение positivity:
    ошибка, а не молчаливый ноль.
    """
    x = _discrete("x", x)
    y = np.asarray(y, float)
    columns = {str(name): np.asarray(values) for name, values in dict(z).items()}
    if x.size != y.size or any(col.shape != x.shape for col in columns.values()) or not np.all(np.isfinite(y)):
        raise ValueError("x, y and every z column must align and y must be finite")
    names = sorted(columns)
    keys = [tuple(columns[name][i] for name in names) for i in range(x.size)] if names else [()] * x.size
    strata: dict[tuple, np.ndarray] = {}
    for index, key in enumerate(keys):
        strata.setdefault(key, []).append(index)  # type: ignore[arg-type]
    total, terms = 0.0, {}
    for key, rows in strata.items():
        rows = np.asarray(rows)
        cell = rows[x[rows] == x_value]
        if cell.size < min_cell:
            raise ValueError(f"empty stratum X={x_value!r}, Z={dict(zip(names, key))!r} (positivity violated)")
        term = float(y[cell].mean()) * rows.size / x.size
        terms[repr(key)] = term
        total += term
    return FormulaResult(total, F_BACKDOOR_DATA, {"terms": terms, "n": int(x.size), "adjustment_set": tuple(names),
                                                  "allowed_planning_use": MAX_PLANNING_USE})


def iv_criterion(edges: Iterable[tuple[str, str]], instrument: str, treatment: str, outcome: str,
                 covariates: Iterable[str] = ()) -> FormulaResult:
    edges, w = list(edges), set(covariates)
    nodes = _nodes(edges)
    if {instrument, treatment, outcome} - nodes or w - nodes:
        raise ValueError("instrument, treatment, outcome and covariates must be nodes of the graph")
    reasons = []
    if instrument in _closure(edges, {treatment}, forward=True):
        reasons.append("instrument_is_descendant_of_treatment")
    if d_separated(edges, {instrument}, {treatment}, w, nodes=nodes):
        reasons.append("instrument_not_relevant")
    no_out_x = [(a, b) for a, b in edges if a != treatment]
    if not d_separated(no_out_x, {instrument}, {outcome}, w, nodes=nodes):
        reasons.append("exclusion_violated")
    return FormulaResult(not reasons, F_IV_CRITERION, {"reasons": tuple(reasons)})


def two_stage_least_squares(*, y, x, z, covariates=None, min_first_stage_f: float = 10.0) -> FormulaResult:
    """2SLS для одной эндогенной переменной X с инструментами Z (и экзогенными W).

    Слабый инструмент (F первой стадии < min_first_stage_f; 10 — правило Staiger–Stock) → оценка
    помечается weak_instrument и не даёт даже review.
    """
    y = np.asarray(y, float).reshape(-1)
    x = np.asarray(x, float).reshape(-1)
    z = np.asarray(z, float)
    z = z.reshape(-1, 1) if z.ndim == 1 else z
    n = y.size
    w = np.ones((n, 1)) if covariates is None else np.column_stack([np.ones(n), np.asarray(covariates, float)])
    if x.size != n or z.shape[0] != n or w.shape[0] != n:
        raise ValueError("y, x, z, covariates must have the same number of rows")
    if not all(np.all(np.isfinite(a)) for a in (y, x, z, w)):
        raise ValueError("inputs must be finite")
    k_z = z.shape[1]
    zf = np.column_stack([w, z])                 # все экзогенные
    xf = np.column_stack([w, x])                 # регрессоры 2-й стадии
    if n <= zf.shape[1] + 1:
        raise ValueError("not enough observations")
    # первая стадия и F-тест значимости инструментов
    coef_full, *_ = np.linalg.lstsq(zf, x, rcond=None)
    rss_full = float(np.sum((x - zf @ coef_full) ** 2))
    coef_r, *_ = np.linalg.lstsq(w, x, rcond=None)
    rss_r = float(np.sum((x - w @ coef_r) ** 2))
    df = n - zf.shape[1]
    f_stat = ((rss_r - rss_full) / k_z) / (rss_full / df) if rss_full > 0 else float("inf")
    # вторая стадия
    x_hat = zf @ np.linalg.lstsq(zf, xf, rcond=None)[0]   # проекция на Z без матрицы n×n
    beta = np.linalg.solve(x_hat.T @ xf, x_hat.T @ y)
    resid = y - xf @ beta
    sigma2 = float(resid @ resid) / (n - xf.shape[1])
    cov = sigma2 * np.linalg.inv(x_hat.T @ x_hat)
    effect, se = float(beta[-1]), float(np.sqrt(cov[-1, -1]))
    weak = f_stat < float(min_first_stage_f)
    return FormulaResult(effect, F_IV, {
        "std_error": se, "ci95": (effect - 1.96 * se, effect + 1.96 * se),
        "first_stage_f": f_stat, "weak_instrument": weak, "n": n,
        "allowed_planning_use": "informational_or_human_review" if weak else MAX_PLANNING_USE,
    })


__all__ = ["backdoor_effect_from_data", "frontdoor_criterion", "frontdoor_effect", "iv_criterion", "two_stage_least_squares"]
