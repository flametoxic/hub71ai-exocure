from __future__ import annotations
import numpy as np
from .reality_formulas import FormulaReference, FormulaResult

UMRM = "CURE-Unified-Mathematical-Reality-Model"
CEOS = "EXO-causal-engine-operational-specification"
F_POSTERIOR = FormulaReference("UMRM-5-BELIEF-POSTERIOR", UMRM, 4, "p(X_t|z_1:t) ∝ p(z_t|X_t) ∫ p(X_t|X_t-1,u_t) p(X_t-1|z_1:t-1) dX_t-1")
F_CROWD = FormulaReference("UMRM-6-CROWD-CONSERVATION", UMRM, 6, "d rho/dt + div(rho v) = s(x,t)")
F_HYBRID = FormulaReference("CEOS-5-HYBRID-DYNAMICS", CEOS, 3, "x_t+1 = f_physics(x,u,d;theta) + f_residual(x,u,d) + eps")
F_ROLLOUT = FormulaReference("CEOS-9-FACTUAL-ROLLOUT", CEOS, 6, "x_hat_t+k+1 = f(x_hat_t+k, u_t+k, d_t+k)")
F_PREDICTIVE = FormulaReference("CEOS-9-PREDICTIVE-DISTRIBUTION", CEOS, 7, "p(x_t:t+H | b_t, u_t:t+H-1)")


def bayes_filter_step(*, prior, transition, likelihood) -> FormulaResult:
    """transition[i][j] = p(X_t=j | X_t-1=i, u_t); likelihood[j] = p(z_t | X_t=j)."""
    b, t, lik = (np.asarray(v, float) for v in (prior, transition, likelihood))
    if b.ndim != 1 or np.any(b < 0) or not np.isclose(b.sum(), 1):
        raise ValueError("prior must be a distribution")
    if t.shape != (b.size, b.size) or np.any(t < 0) or not np.allclose(t.sum(1), 1):
        raise ValueError("transition rows must be distributions")
    if lik.shape != b.shape or np.any(lik < 0):
        raise ValueError("likelihood must match the state space")
    predicted = b @ t
    unnorm = lik * predicted
    evidence = float(unnorm.sum())
    if evidence <= 0:
        raise ValueError("observation impossible under the model: no silent update")
    return FormulaResult(unnorm / evidence, F_POSTERIOR, {"predicted": predicted, "evidence": evidence})


# UMRM 6: X_{t+1} = F(...) has ONE implementation -- ``reality_formulas.universal_transition`` routed
# through ``dynamics_registry.DomainDynamicsRegistry`` (the former ``domain_transition`` duplicate was removed).


def crowd_conservation_step(*, density, face_velocity, source, dx: float, dt: float,
                            boundary_density: tuple[float, float] = (0.0, 0.0)) -> FormulaResult:
    """1D конечные объёмы, upwind: n ячеек, n+1 скоростей на гранях. Коридоры, входы, сегменты дорог."""
    rho, v, s = (np.asarray(a, float) for a in (density, face_velocity, source))
    n = rho.size
    if n == 0 or v.shape != (n + 1,) or s.shape != (n,) or np.any(rho < 0) or dx <= 0 or dt <= 0:
        raise ValueError("need n>0 cells, n+1 face velocities, n sources, rho>=0, dx,dt>0")
    cfl = float(np.max(np.abs(v)) * dt / dx)
    if cfl > 1:
        raise ValueError(f"CFL violated ({cfl:.3f} > 1): reduce dt")
    padded = np.concatenate(([boundary_density[0]], rho, [boundary_density[1]]))
    flux = np.where(v >= 0, padded[:-1], padded[1:]) * v
    new = rho - dt / dx * (flux[1:] - flux[:-1]) + dt * s
    if np.any(new < -1e-12):
        raise ValueError("negative density: sinks exceed occupancy")
    new = np.maximum(new, 0.0)
    return FormulaResult(new, F_CROWD, {"flux": flux, "cfl": cfl, "mass_before": float(rho.sum() * dx),
                                        "mass_after": float(new.sum() * dx)})


def hybrid_dynamics_step(*, f_physics, state, control, exogenous, theta, f_residual=None,
                         residual_validated: bool = False, noise=None) -> FormulaResult:
    """Нейро-остаток без promotion gate в прогнозе не участвует, и это видно в intermediates."""
    x, u, d = (np.asarray(a, float) for a in (state, control, exogenous))
    phys = np.asarray(f_physics(x, u, d, theta), float)
    if phys.shape != x.shape or not np.all(np.isfinite(phys)):
        raise ValueError("f_physics must return a finite vector of the state's shape")
    res = np.zeros_like(x)
    if f_residual is not None and residual_validated:
        res = np.asarray(f_residual(x, u, d), float)
        if res.shape != x.shape or not np.all(np.isfinite(res)):
            raise ValueError("invalid residual")
    eps = np.zeros_like(x) if noise is None else np.asarray(noise, float)
    return FormulaResult(phys + res + eps, F_HYBRID, {
        "physics": phys, "residual": res,
        "residual_ignored_not_validated": f_residual is not None and not residual_validated})


def factual_rollout(f, *, initial_state, controls, exogenous) -> FormulaResult:
    if len(controls) != len(exogenous):
        raise ValueError("controls and exogenous must share the horizon")
    x = np.asarray(initial_state, float)
    traj = [x]
    for u, d in zip(controls, exogenous):
        x = np.asarray(f(x, np.asarray(u, float), np.asarray(d, float)), float)
        if x.shape != traj[0].shape or not np.all(np.isfinite(x)):
            raise ValueError("transition returned an invalid state")
        traj.append(x)
    return FormulaResult(np.stack(traj), F_ROLLOUT, {"horizon": len(controls)})


def predictive_distribution(f, *, sample_initial_state, controls, sample_exogenous, samples: int,
                            seed: int, alpha: float = 0.1) -> FormulaResult:
    """Один seed → один результат. f(x, u, d, rng) может сэмплировать шум модели."""
    if samples <= 0 or not 0 < alpha < 1:
        raise ValueError("samples > 0, alpha in (0, 1)")
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(samples):
        x = np.asarray(sample_initial_state(rng), float)
        traj = [x]
        for k, u in enumerate(controls):
            x = np.asarray(f(x, np.asarray(u, float), np.asarray(sample_exogenous(rng, k), float), rng), float)
            traj.append(x)
        runs.append(np.stack(traj))
    paths = np.stack(runs)
    if not np.all(np.isfinite(paths)):
        raise ValueError("simulation produced non-finite states")
    return FormulaResult({"mean": paths.mean(0), "lower": np.quantile(paths, alpha / 2, 0),
                          "upper": np.quantile(paths, 1 - alpha / 2, 0), "trajectories": paths,
                          "epistemic_status": "predicted"}, F_PREDICTIVE, {"seed": seed, "samples": samples})


# ---------------------------------------------------------------- совместимость со старым API
# Прежний параллельный вход F_domain. Теперь это тонкая обёртка над ЕДИНСТВЕННЫМ контрактом —
# DomainDynamicsRegistry/universal_transition (глубокая заморозка входов, обязательная неопределённость,
# результат PREDICTED). Своей реализации перехода здесь нет.
F_TRANSITION = FormulaReference("UMRM-6-DOMAIN-TRANSITION", UMRM, 5,
                                "X_t+1 = F(X_t,u_t,G_t,theta_t,eps_t); F_domain -> (next state, uncertainty)")


class DomainDynamicsPack:
    """Прежний протокол пакета: атрибуты ``pack_id``, ``model_version`` и метод
    ``transition(state, controls, topology, boundary_conditions, parameters) -> (next_state, variance)``."""

    pack_id: str
    model_version: str

    def transition(self, state, controls, topology, boundary_conditions, parameters):  # pragma: no cover
        raise NotImplementedError


class PredictedState:
    __slots__ = ("next_state", "variance", "pack_id", "model_version", "epistemic_status")

    def __init__(self, next_state, variance, pack_id, model_version, epistemic_status="predicted"):
        self.next_state, self.variance = next_state, variance
        self.pack_id, self.model_version, self.epistemic_status = pack_id, model_version, epistemic_status


def domain_transition(pack, *, state, controls, topology, boundary_conditions, parameters, noise=None) -> FormulaResult:
    """Совместимость: прежний вызов идёт через единый контракт реестра пакетов динамики."""
    from types import MappingProxyType

    from .dynamics_registry import DomainDynamicsPack as _RegistryPack, DomainDynamicsRegistry
    from .reality_formulas import universal_transition

    pack_id, version = getattr(pack, "pack_id", ""), getattr(pack, "model_version", "")
    if not pack_id or not version:
        raise ValueError("pack must declare pack_id and model_version")
    registry = DomainDynamicsRegistry()
    registry.register(_RegistryPack(str(pack_id), str(version), lambda *inputs: pack.transition(*inputs)))
    result = universal_transition(registry=registry, domain=str(pack_id), state=state, controls=controls,
                                  topology=topology, boundary_conditions=boundary_conditions, parameters=parameters).value
    nxt = {k: float(v) for k, v in result.next_state.items()}
    var = {k: float(v) for k, v in result.uncertainty.items()}
    if set(nxt) - set(var):
        raise ValueError(f"pack returned no uncertainty for {sorted(set(nxt) - set(var))}")
    if any(v < 0 or not np.isfinite(v) for v in var.values()):
        raise ValueError("pack returned negative or non-finite variance")
    for k, eps in (noise or {}).items():
        if k not in nxt:
            raise ValueError(f"noise for unknown variable {k}")
        nxt[k] += float(eps)
    return FormulaResult(PredictedState(MappingProxyType(nxt), MappingProxyType(var), str(pack_id), str(version)),
                         F_TRANSITION)

