"""Мастер-ТЗ B5 (стр. 11–12): вероятность ребра для симулятора и размер выборки микроэксперимента.

    p_e ~ Beta(1 + n_e·p, 1 + n_e·(1 − p))          — меньше наблюдений → шире интервал
    n = (z_{1−α/2} + z_{1−β})² · 2σ² / δ²_min        — на группу (двухвыборочное сравнение средних)

Микроэксперимент не запускается без границ безопасности, плана отката, одобрения политики и выполнимости
по мощности (n ≤ доступного числа наблюдений в окне).
"""
from __future__ import annotations

import math

import numpy as np

from ._base import num, prob, ref

F_BETA = ref("MASTER-B5-EDGE-BETA", 11, "p_e ~ Beta(1 + n_e p, 1 + n_e (1 − p))")
F_SAMPLE = ref("MASTER-B5-SAMPLE-SIZE", 12, "n = (z_{1−α/2} + z_{1−β})²·2σ²/δ²_min")


def edge_beta(*, p: float, n_evidence: float) -> dict:
    a = 1.0 + num("n_e", n_evidence, low=0.0) * prob("p", p)
    b = 1.0 + n_evidence * (1.0 - p)
    return {"alpha": a, "beta": b, "mean": a / (a + b), "variance": a * b / ((a + b) ** 2 * (a + b + 1)),
            "formula": F_BETA.formula_id}


def sample_edge_probability(*, p: float, n_evidence: float, samples: int, seed: int, interval: float) -> dict:
    """Сэмплы p_e для Монте-Карло симулятора и центральный интервал заданного уровня."""
    d = edge_beta(p=p, n_evidence=n_evidence)
    lvl = num("interval", interval, low=0.0, strict_low=True, high=1.0)
    x = np.random.default_rng(int(seed)).beta(d["alpha"], d["beta"], int(samples))
    lo, hi = np.quantile(x, [(1 - lvl) / 2, 1 - (1 - lvl) / 2])
    return {**d, "samples": x, "interval": (float(lo), float(hi))}


def microexperiment_sample_size(*, alpha: float, power: float, sigma: float, delta_min: float) -> dict:
    from scipy.stats import norm

    a = num("alpha", alpha, low=0.0, strict_low=True, high=1.0)
    pw = num("power", power, low=0.0, strict_low=True, high=1.0)
    s = num("sigma", sigma, low=0.0, strict_low=True)
    d = num("delta_min", delta_min, low=0.0, strict_low=True)
    za, zb = float(norm.ppf(1 - a / 2)), float(norm.ppf(pw))          # z_{1−β}, β = 1 − мощность
    n_exact = (za + zb) ** 2 * 2 * s * s / (d * d)
    return {"n_per_group": int(math.ceil(n_exact)), "n_exact": n_exact, "z_alpha": za, "z_beta": zb,
            "formula": F_SAMPLE.formula_id}


def experiment_gate(*, n_required: int, n_available: int, safety_envelope: bool, rollback_plan: bool,
                    policy_approved: bool, shadow_or_gate_approved: bool) -> dict:
    reasons = []
    for flag, name in ((safety_envelope, "no_safety_envelope"), (rollback_plan, "no_rollback_plan"),
                       (policy_approved, "no_policy_approval"), (shadow_or_gate_approved, "no_shadow_or_gate_approval")):
        if flag is not True:
            reasons.append(name)
    if int(n_required) > int(n_available):
        reasons.append(f"power_infeasible: need {int(n_required)} per group, window gives {int(n_available)}")
    return {"may_run": not reasons, "reasons": reasons, "dispatch": "via_sovereign_pipeline_only" if not reasons else None}


__all__ = ["F_BETA", "F_SAMPLE", "edge_beta", "experiment_gate", "microexperiment_sample_size",
           "sample_edge_probability"]
