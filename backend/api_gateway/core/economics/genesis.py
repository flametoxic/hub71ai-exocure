"""Economic Optimizer §5 (Genesis за пределами физики) и §6 (формальное открытие через Z3).

§5 Цикл: observe → represent → hypothesize → derive testable prediction → run permitted experiment/tool call
→ verify outcome → update versioned knowledge. Для нефизических доменов (код, экономика, право/политика)
эксперимент идёт только в песочнице; физический домен — только через планирование (CPR) и его гейты.
Цикл хранит каждую итерацию; «знание» обновляется версией кандидата, а не правкой продакшена.
§6 A ⊢ φ ⇔ A ∧ ¬φ невыполнимо (Z3, линейная/полиномиальная арифметика над действительными);
доказано в аксиомах ≠ эмпирически верно. Выражения разбираются безопасным разбором AST (без eval).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from ._base import EconomicsError
from .generalization import NONPHYSICAL_DOMAINS, genesis_experiment_gate

GENESIS_STEPS = ("observe", "represent", "hypothesize", "derive_prediction", "experiment", "verify", "update_knowledge")


# ------------------------------------------------------------------------------------------ §6 Z3
def _z3_expr(node: ast.AST, env: dict):
    import z3

    if isinstance(node, ast.Expression):
        return _z3_expr(node.body, env)
    if isinstance(node, ast.Name):
        if node.id not in env:
            env[node.id] = z3.Real(node.id)
        return env[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return z3.RealVal(str(node.value))
    if isinstance(node, ast.UnaryOp):
        v = _z3_expr(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.Not):
            return z3.Not(v)
        return v
    if isinstance(node, ast.BinOp):
        a, b = _z3_expr(node.left, env), _z3_expr(node.right, env)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            return a / b
        if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
            return a ** node.right.value
        raise EconomicsError(f"operator {type(node.op).__name__} not supported")
    if isinstance(node, ast.BoolOp):
        vals = [_z3_expr(v, env) for v in node.values]
        return z3.And(*vals) if isinstance(node.op, ast.And) else z3.Or(*vals)
    if isinstance(node, ast.Compare):
        left, parts = _z3_expr(node.left, env), []
        for op, comp in zip(node.ops, node.comparators):
            right = _z3_expr(comp, env)
            parts.append({ast.Lt: left < right, ast.LtE: left <= right, ast.Gt: left > right, ast.GtE: left >= right,
                          ast.Eq: left == right, ast.NotEq: left != right}[type(op)])
            left = right
        return z3.And(*parts) if len(parts) > 1 else parts[0]
    raise EconomicsError(f"construct {type(node).__name__} not allowed in a formal statement")


def z3_entails(axioms: Sequence[str], phi: str, *, timeout_ms: int) -> dict:
    """A ⊢ φ? proved | refuted (с контрпримером) | unknown. Без z3-solver — unknown (fail-closed)."""
    try:
        import z3
    except ImportError:
        return {"result": "unknown", "reason": "z3-solver not installed", "follows_from_axioms": None}
    env: dict = {}
    try:
        A = [_z3_expr(ast.parse(a, mode="eval"), env) for a in axioms]
        P = _z3_expr(ast.parse(phi, mode="eval"), env)
    except SyntaxError as exc:
        raise EconomicsError(f"formalization failed: {exc.msg}") from exc
    s = z3.Solver()
    s.set("timeout", int(timeout_ms))
    s.add(*A)
    s.add(z3.Not(P))
    r = s.check()
    if r == z3.unsat:
        return {"result": "proved", "follows_from_axioms": True,
                "meaning": "follows from the stated axioms, not empirical truth"}
    if r == z3.sat:
        m = s.model()
        return {"result": "refuted", "follows_from_axioms": False,
                "counterexample": {str(d): str(m[d]) for d in m.decls()}}
    return {"result": "unknown", "follows_from_axioms": None, "reason": str(s.reason_unknown())}


# ------------------------------------------------------------------------------------------ §5 цикл Genesis
@dataclass
class GenesisIteration:
    index: int
    hypothesis: Mapping[str, Any]
    prediction: Mapping[str, Any]
    experiment_result: Mapping[str, Any]
    verified: bool
    knowledge_version: Optional[str]
    trace: list = field(default_factory=list)


class GenesisSandboxLoop:
    """Нефизический цикл открытия. Каждая стадия — функция домена; эксперимент выполняется песочницей домена
    (например: патч + тесты в изолированной копии репозитория; распределение бюджета в симуляторе экономики)."""

    def __init__(self, domain: str, *, observe: Callable[[], Mapping], represent: Callable[[Mapping], Mapping],
                 hypothesize: Callable[[Mapping], Sequence[Mapping]], derive_prediction: Callable[[Mapping], Mapping],
                 sandbox_experiment: Callable[[Mapping], Mapping], verify: Callable[[Mapping, Mapping], bool],
                 max_iterations: int) -> None:
        if genesis_experiment_gate(domain, sandboxed=True) != "sandbox_only_no_physical_authority":
            raise EconomicsError("physical domains go through constrained planning, not the Genesis sandbox loop")
        if domain not in NONPHYSICAL_DOMAINS or int(max_iterations) < 1:
            raise EconomicsError("non-physical domain and a positive iteration budget (policy) are required")
        self.domain = domain
        self._f = dict(observe=observe, represent=represent, hypothesize=hypothesize,
                       derive_prediction=derive_prediction, experiment=sandbox_experiment, verify=verify)
        self.max_iterations = int(max_iterations)
        self.iterations: list[GenesisIteration] = []
        self.knowledge: list[dict] = []                # версии знания (кандидаты, не продакшен)

    def run(self) -> dict:
        obs = dict(self._f["observe"]())
        rep = dict(self._f["represent"](obs))
        hyps = list(self._f["hypothesize"](rep))
        for i, h in enumerate(hyps[: self.max_iterations]):
            trace = [("observe", sorted(obs)), ("represent", sorted(rep)), ("hypothesize", h.get("id"))]
            pred = dict(self._f["derive_prediction"](h))
            trace.append(("derive_prediction", pred))
            res = dict(self._f["experiment"](h))
            if res.get("sandboxed") is not True:
                raise EconomicsError("experiment runner did not confirm sandbox execution: aborted")
            trace.append(("experiment", {k: v for k, v in res.items() if k != "artifact"}))
            ok = bool(self._f["verify"](pred, res))
            trace.append(("verify", ok))
            ver = None
            if ok:
                ver = f"{self.domain}:knowledge:v{len(self.knowledge) + 1}"
                self.knowledge.append({"version": ver, "hypothesis": h, "prediction": pred, "evidence": res,
                                       "status": "candidate", "deployment": "requires_human_review_and_pipeline"})
                trace.append(("update_knowledge", ver))
            self.iterations.append(GenesisIteration(i, h, pred, res, ok, ver, trace))
        return {"domain": self.domain, "iterations": len(self.iterations),
                "verified": [it.hypothesis.get("id") for it in self.iterations if it.verified],
                "knowledge_versions": [k["version"] for k in self.knowledge], "physical_authority": "none",
                "steps": GENESIS_STEPS}


__all__ = ["GENESIS_STEPS", "GenesisIteration", "GenesisSandboxLoop", "z3_entails"]
