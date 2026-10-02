"""Мастер-ТЗ B1b (стр. 9): исследовательский контур PINN / FNO / HNN.

    L_PINN = λ_d L_data + λ_p L_PDE + λ_b L_boundary
    K(a)(x) = F⁻¹(R_φ · (F a))(x)                  (спектральный слой нейрооператора Фурье)
    dq/dt = ∂H/∂p,  dp/dt = −∂H/∂q                 (гамильтоновы ограничения для сохраняющих систем)

«Не продакшен, пока не прошёл бенчмарк»: модель B1b идёт только за исследовательским флагом и только после
того, как B1a прошёл свою приёмку; маршрутизировать продакшен-вывод может лишь модель, которая побила
откалиброванную RC-базу на отложенных данных И прошла тест сохранения/ограничений (research_gate).
Дальше — общий гейт допуска нейромодели WM-05 (Promote). Обучение сетей здесь не делается: модуль даёт
функции потерь, спектральный слой, гамильтонов поток с симплектическим шагом и гейт.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from ._base import PhysicsError, num, ref

F_PINN = ref("MASTER-B1-PINN-LOSS", 9, "L_PINN = λ_d L_data + λ_p L_PDE + λ_b L_boundary")
F_FNO = ref("MASTER-B1-FNO", 9, "K(a)(x) = F⁻¹(R_φ·(F a))(x)")
F_HNN = ref("MASTER-B1-HNN", 9, "dq/dt = ∂H/∂p, dp/dt = −∂H/∂q")


def _mse(name, r) -> float:
    a = np.asarray(r, float).reshape(-1)
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise PhysicsError(f"{name} residuals must be a non-empty finite array")
    return float(np.mean(a * a))


def pinn_loss(*, data_residuals, pde_residuals, boundary_residuals, lambda_data: float, lambda_pde: float,
              lambda_boundary: float) -> dict:
    """Каждый член — средний квадрат соответствующих невязок; λ — из конфигурации эксперимента."""
    lam = [num(n, v, low=0.0) for n, v in (("λ_d", lambda_data), ("λ_p", lambda_pde), ("λ_b", lambda_boundary))]
    if sum(lam) <= 0:
        raise PhysicsError("at least one λ must be positive")
    parts = {"data": _mse("data", data_residuals), "pde": _mse("pde", pde_residuals),
             "boundary": _mse("boundary", boundary_residuals)}
    return {"loss": lam[0] * parts["data"] + lam[1] * parts["pde"] + lam[2] * parts["boundary"], "parts": parts,
            "lambdas": dict(zip(("data", "pde", "boundary"), lam)), "formula": F_PINN.formula_id}


def fno_spectral_layer(a, weights) -> np.ndarray:
    """a: (n,) или (n, c_in) на равномерной сетке; weights R_φ: (modes,) или (modes, c_in, c_out) комплексные.
    Спектр F a обрезается до первых `modes` частот (остальные зануляются), умножается на R_φ, F⁻¹ возвращает поле."""
    x = np.asarray(a, float)
    if x.ndim == 1:
        x = x[:, None]
    w = np.asarray(weights, complex)
    if w.ndim == 1:
        w = w[:, None, None] * np.eye(x.shape[1])[None]
    n, c_in = x.shape
    modes = w.shape[0]
    if w.shape[1] != c_in or modes > n // 2 + 1:
        raise PhysicsError("weights must be (modes, c_in, c_out) with modes ≤ n/2 + 1")
    fa = np.fft.rfft(x, axis=0)
    out = np.zeros((n // 2 + 1, w.shape[2]), complex)
    out[:modes] = np.einsum("mi,mio->mo", fa[:modes], w)
    y = np.fft.irfft(out, n=n, axis=0)
    return y[:, 0] if y.shape[1] == 1 and np.asarray(a).ndim == 1 else y


def hamiltonian_field(dH_dq: Callable, dH_dp: Callable, q, p) -> tuple[np.ndarray, np.ndarray]:
    q, p = np.asarray(q, float), np.asarray(p, float)
    return np.asarray(dH_dp(q, p), float), -np.asarray(dH_dq(q, p), float)


def leapfrog(dH_dq: Callable, dH_dp: Callable, q, p, *, dt: float, steps: int, separable: bool = True):
    """Симплектический шаг (Штёрмер–Верле) для сепарабельного H(q,p) = T(p) + V(q): энергия не дрейфует систематически."""
    if not separable:
        raise PhysicsError("leapfrog is exact-symplectic only for separable H; supply an implicit integrator otherwise")
    dt = num("dt", dt, low=0.0, strict_low=True)
    q, p = np.asarray(q, float).copy(), np.asarray(p, float).copy()
    for _ in range(int(steps)):
        p = p - 0.5 * dt * np.asarray(dH_dq(q, p), float)
        q = q + dt * np.asarray(dH_dp(q, p), float)
        p = p - 0.5 * dt * np.asarray(dH_dq(q, p), float)
    return q, p


def conservation_test(H: Callable, trajectory: Sequence[tuple], *, rel_tolerance: float) -> dict:
    e = np.array([float(H(np.asarray(q), np.asarray(pp))) for q, pp in trajectory])
    scale = max(abs(e[0]), 1e-12)
    drift = float(np.max(np.abs(e - e[0])) / scale)
    return {"passed": drift <= num("rel_tolerance", rel_tolerance, low=0.0), "max_relative_energy_drift": drift}


def research_gate(*, research_flag: bool, b1a_passed: bool, candidate_holdout_rmse: float, rc_baseline_holdout_rmse: float,
                  conservation_passed: bool, constraints_passed: bool) -> dict:
    """ТЗ B1b: только за флагом, только после B1a; продакшен-маршрут — лишь если бьёт RC-базу на holdout и
    проходит тест сохранения/ограничений. Иначе — только исследование."""
    reasons = []
    if research_flag is not True:
        reasons.append("research_flag_off")
    if b1a_passed is not True:
        reasons.append("b1a_baseline_not_accepted")
    c = num("candidate_holdout_rmse", candidate_holdout_rmse, low=0.0)
    b = num("rc_baseline_holdout_rmse", rc_baseline_holdout_rmse, low=0.0)
    if not c < b:
        reasons.append("does_not_beat_calibrated_rc_baseline")
    if conservation_passed is not True:
        reasons.append("conservation_test_failed")
    if constraints_passed is not True:
        reasons.append("constraint_test_failed")
    return {"may_route_production_inference": not reasons, "route": "wm05_promotion_gate" if not reasons else "research_only",
            "reasons": reasons, "note": "passing this gate is necessary, not sufficient: WM-05 Promote(m) still applies"}


__all__ = ["F_FNO", "F_HNN", "F_PINN", "conservation_test", "fno_spectral_layer", "hamiltonian_field", "leapfrog",
           "pinn_loss", "research_gate"]
