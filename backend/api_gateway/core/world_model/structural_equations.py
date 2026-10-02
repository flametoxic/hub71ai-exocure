from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping
import numpy as np
from .reality_formulas import FormulaReference, FormulaResult

UMRM = "CURE-Unified-Mathematical-Reality-Model"
CEOS = "EXO-causal-engine-operational-specification"
F_SCM = FormulaReference("UMRM-7-SCM", UMRM, 6, "X_i := f_i(PA_i, U_i, theta_i)")
F_DO = FormulaReference("UMRM-7-DO-SURGERY", UMRM, 6, "SCM_do(X=x) = SCM \\ Parents(X) ∪ {X := x}; CEOS p7: do(X_j=x*) => X_j := x*")
F_INTERVENTIONAL = FormulaReference("CEOS-10-INTERVENTIONAL-QUERY", CEOS, 7, "P(Y | do(X=x), C_t)")
F_COUNTERFACTUAL = FormulaReference("UMRM-7-COUNTERFACTUAL", UMRM, 6, "abduction P(U|e) -> do(A=a) -> Y_do(A=a) -> {mean,p05,p95}; CEOS p7: Y_a'(e) = Predict(M_do(A=a'), U_e)")

Equation = Callable[[Mapping[str, np.ndarray], Mapping[str, float]], np.ndarray]  # numpy-векторизованная f_i без шума


@dataclass(frozen=True)
class StructuralEquation:
    variable: str
    parents: tuple[str, ...]
    fn: Equation
    theta: Mapping[str, float]
    noise_std: float  # U_i ~ N(0, σ²), аддитивный


def _bump(v: str) -> str:
    head, _, tail = v.rpartition(":")
    return f"{head}:{int(tail) + 1}" if head and tail.isdigit() else f"{v}:1"


class StructuralEquationModel:
    def __init__(self, version: str = "scm:0"):
        self._eq: dict[str, StructuralEquation] = {}
        self.version = version

    def register_equation(self, variable: str, fn: Equation, parents, theta, noise_std: float) -> None:
        if not variable or noise_std < 0:
            raise ValueError("variable required, noise_std >= 0")
        prev = self._eq.get(variable)
        self._eq[variable] = StructuralEquation(variable, tuple(parents), fn, dict(theta), float(noise_std))
        try:
            self.order(strict=False)
        except ValueError:
            if prev is None:
                del self._eq[variable]
            else:
                self._eq[variable] = prev
            raise
        self.version = _bump(self.version)

    def order(self, strict: bool = True) -> list[str]:
        indeg, children = {}, {}
        for v, eq in self._eq.items():
            known = [p for p in eq.parents if p in self._eq]
            if strict and len(known) != len(eq.parents):
                raise ValueError(f"{v}: parents without equations {sorted(set(eq.parents) - set(known))}")
            indeg[v] = len(known)
            for p in known:
                children.setdefault(p, []).append(v)
        queue, out = sorted(v for v, d in indeg.items() if d == 0), []
        while queue:
            v = queue.pop(0)
            out.append(v)
            for c in children.get(v, []):
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        if len(out) != len(self._eq):
            raise ValueError("causal cycle: SCM must be acyclic")
        return out

    def _propagate(self, n: int, rng: np.random.Generator, noise: Mapping[str, np.ndarray] | None = None):
        values = {}
        for v in self.order():
            eq = self._eq[v]
            if noise and v in noise:
                u = noise[v]
            else:
                u = rng.normal(0.0, eq.noise_std, n) if eq.noise_std > 0 else np.zeros(n)
            base = np.broadcast_to(np.asarray(eq.fn({p: values[p] for p in eq.parents}, eq.theta), float), (n,))
            values[v] = base + u
        return values

    def sample_observational(self, n: int, seed: int) -> FormulaResult:
        return FormulaResult(self._propagate(n, np.random.default_rng(seed)), F_SCM,
                             {"scm_version": self.version, "method": "observational"})

    def intervene(self, variable: str, value: float) -> "StructuralEquationModel":
        """Возвращает копию; исходная модель не мутирует."""
        if variable not in self._eq:
            raise KeyError(variable)
        copy = StructuralEquationModel(f"{self.version}|do({variable}={value})")
        copy._eq = dict(self._eq)
        copy._eq[variable] = StructuralEquation(variable, (), lambda _p, _t, x=float(value): x, {}, 0.0)
        return copy

    def sample_interventional(self, do: Mapping[str, float], n: int, seed: int,
                              context: Mapping[str, float] | None = None) -> FormulaResult:
        """Контекст C_t — только корневые (экзогенные) переменные: у них фиксация = обусловливание."""
        context = dict(context or {})
        for c in context:
            if self._eq[c].parents:
                raise ValueError(f"context '{c}' has parents: conditioning needs inference, not fixing")
        model = self
        for var, val in {**context, **do}.items():
            model = model.intervene(var, val)
        return FormulaResult(model._propagate(n, np.random.default_rng(seed)), F_INTERVENTIONAL,
                             {"do": dict(do), "context": context, "scm_version": model.version,
                              "method": "interventional"})

    def counterfactual(self, evidence: Mapping[str, float], do: Mapping[str, float],
                       n: int = 2000, seed: int = 0) -> FormulaResult:
        """Abduction → action → prediction (UMRM стр. 6). Всегда estimated (Master, правило 4).

        Abduction samples U from the posterior P(U | evidence), not from the prior: every variable
        without evidence draws U_i ~ N(0, σ_i²); every observed variable gets U_i = x_i − f_i(PA_i) and
        contributes the likelihood N(U_i; 0, σ_i²) to the sample weight (likelihood weighting; additive
        noise, unit Jacobian). With complete evidence all U are fixed and the result is a point.
        Output: {mean, p05, p95} per variable (quantile levels fixed by UMRM p.6).
        """
        if n <= 0:
            raise ValueError("n must be positive")
        unknown = (set(evidence) | set(do)) - set(self._eq)
        if unknown:
            raise KeyError(f"unknown variables {sorted(unknown)}")
        rng = np.random.default_rng(seed)
        order = self.order()
        values: dict[str, np.ndarray] = {}
        noise: dict[str, np.ndarray] = {}
        log_weight = np.zeros(n)
        deterministic: set[str] = set()
        for v in order:
            eq = self._eq[v]
            base = np.broadcast_to(np.asarray(eq.fn({p: values[p] for p in eq.parents}, eq.theta), float), (n,))
            if v in evidence:
                observed = float(evidence[v])
                u = observed - base
                if eq.noise_std > 0:
                    log_weight += -0.5 * (u / eq.noise_std) ** 2
                else:
                    log_weight += np.where(np.abs(u) <= 1e-9 * max(1.0, abs(observed)), 0.0, -np.inf)
                values[v] = np.full(n, observed)
                if all(p in deterministic for p in eq.parents):
                    deterministic.add(v)
            else:
                u = rng.normal(0.0, eq.noise_std, n) if eq.noise_std > 0 else np.zeros(n)
                values[v] = base + u
            noise[v] = np.asarray(u, float)
        if not np.any(np.isfinite(log_weight)):
            raise ValueError("evidence has zero likelihood under the SCM: abduction impossible")
        weights = np.exp(log_weight - np.max(log_weight))
        weights /= weights.sum()
        model = self
        for var, val in do.items():
            model = model.intervene(var, val)
        # the intervened equations X := x carry no disturbance: their abducted U is dropped
        prediction_noise = {k: u for k, u in noise.items() if k not in do}
        samples = model._propagate(n, np.random.default_rng(seed), noise=prediction_noise)
        summary = {v: _weighted_summary(s, weights) for v, s in samples.items()}
        incomplete = tuple(v for v in order if v not in deterministic and v not in do)
        return FormulaResult(summary, F_COUNTERFACTUAL, {
            "epistemic_status": "estimated",
            "abduction": "posterior_likelihood_weighting",
            "abduction_incomplete": incomplete,
            "effective_sample_size": float(1.0 / np.sum(weights ** 2)),
            "scm_version": model.version,
        })


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values, sorted_weights = values[order], weights[order]
    cumulative = np.cumsum(sorted_weights)
    index = int(np.searchsorted(cumulative, q * cumulative[-1], side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def _weighted_summary(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """{mean, p05, p95} (UMRM p.6 output levels) of a weighted sample."""
    values = np.asarray(values, float)
    return {"mean": float(np.sum(values * weights)),
            "p05": _weighted_quantile(values, weights, 0.05),
            "p95": _weighted_quantile(values, weights, 0.95)}
