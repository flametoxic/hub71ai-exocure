from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np


DOCUMENT = "CURE-Unified-Mathematical-Reality-Model"


@dataclass(frozen=True)
class FormulaReference:
    formula_id: str
    document: str
    page: int
    expression: str
    input_units: Mapping[str, str] = field(default_factory=dict)
    output_unit: str = "dimensionless"


@dataclass(frozen=True)
class FormulaResult:
    value: Any
    formula: FormulaReference
    intermediates: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScalarFusionResult:
    value: float
    variance: float
    formula: FormulaReference
    intermediates: Mapping[str, Any] = field(default_factory=dict)


# Numerical tolerances for floating-point *consistency checks* only (orthonormality, symmetry,
# positive semi-definiteness, unit quaternion).  They are not model constants of the UMRM document.
# ROTATION_ATOL matches the tolerance used by spatial_formulas._se3.
ROTATION_ATOL = 1e-6
MATRIX_RTOL = 1e-9


def _finite(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number, not a boolean")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _finite_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def is_strict_bool(value: Any) -> bool:
    """The single definition of "a boolean" for every UMRM gate: builtin ``bool`` or a numpy scalar
    ``np.bool_``.  Strings ("true"), numbers (1), None and arrays are not booleans."""
    return isinstance(value, (bool, np.bool_))


def is_strict_true(value: Any) -> bool:
    """A gate passes only on a strict boolean ``True`` (see :func:`is_strict_bool`)."""
    return is_strict_bool(value) and bool(value)


def strict_bool(name: str, value: Any) -> bool:
    """Validate an input flag: raise ``TypeError`` for anything that is not a strict boolean."""
    if not is_strict_bool(value):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return bool(value)


_strict_bool = strict_bool


def validate_rotation(rotation: Any, *, name: str = "rotation") -> np.ndarray:
    """R must be a finite square orthonormal matrix with det(R)=+1 (a proper rotation)."""
    r = np.asarray(rotation, dtype=float)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or r.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if not np.all(np.isfinite(r)):
        raise ValueError(f"{name} must contain finite values")
    if not np.allclose(r @ r.T, np.eye(r.shape[0]), rtol=0.0, atol=ROTATION_ATOL):
        raise ValueError(f"{name} must be orthonormal (R R^T = I)")
    if abs(float(np.linalg.det(r)) - 1.0) > ROTATION_ATOL:
        raise ValueError(f"{name} must be a proper rotation with det(R)=+1")
    return r


def validate_covariance(covariance: Any, *, name: str = "covariance", shape: tuple[int, int] | None = None) -> np.ndarray:
    """Sigma must be a finite, symmetric, positive semi-definite square matrix."""
    sigma = np.asarray(covariance, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1] or sigma.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if shape is not None and sigma.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {sigma.shape}")
    if not np.all(np.isfinite(sigma)):
        raise ValueError(f"{name} must contain finite values")
    scale = max(1.0, float(np.max(np.abs(sigma))))
    if float(np.max(np.abs(sigma - sigma.T))) > MATRIX_RTOL * scale:
        raise ValueError(f"{name} must be symmetric")
    if float(np.min(np.linalg.eigvalsh(sigma))) < -MATRIX_RTOL * scale:
        raise ValueError(f"{name} must be positive semi-definite")
    return sigma


def _positive(name: str, value: float) -> float:
    number = _finite(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _non_negative(name: str, value: float) -> float:
    number = _finite(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _probability(name: str, value: float, *, allow_zero: bool = True) -> float:
    number = _finite(name, value)
    lower_valid = number >= 0.0 if allow_zero else number > 0.0
    if not lower_valid or number > 1.0:
        boundary = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {boundary}")
    return number


def transform_covariance(rotation: Sequence[Sequence[float]], covariance: Sequence[Sequence[float]]) -> FormulaResult:
    """Document 1, page 2: Sigma_W = R Sigma_S R^T.

    Fail-closed: R must be a proper rotation (orthonormal, det=+1) and Sigma_S a finite
    symmetric positive semi-definite matrix of the same shape.
    """
    r = np.asarray(rotation, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    if r.ndim != 2 or sigma.ndim != 2 or r.shape[0] != r.shape[1] or r.shape != sigma.shape:
        raise ValueError("rotation and covariance must be square matrices of the same shape")
    r = validate_rotation(r)
    sigma = validate_covariance(sigma)
    value = r @ sigma @ r.T
    value = 0.5 * (value + value.T)  # remove floating-point asymmetry of the product
    return FormulaResult(
        value=value,
        formula=FormulaReference(
            "UMRM-3.2-COVARIANCE",
            DOCUMENT,
            2,
            "Sigma_W = R Sigma_S R^T",
            {"R": "dimensionless", "Sigma_S": "state-unit squared"},
            "state-unit squared",
        ),
        intermediates={"rotation_shape": r.shape, "covariance_shape": sigma.shape},
    )


def weighted_scalar_fusion(values: Sequence[float], variances: Sequence[float]) -> ScalarFusionResult:
    """Document 1, page 5: independent scalar inverse-variance fusion."""
    if len(values) == 0 or len(values) != len(variances):
        raise ValueError("values and variances must be non-empty and have equal length")
    z = np.asarray([_finite("value", value) for value in values], dtype=float)
    sigma2 = np.asarray([_positive("variance", value) for value in variances], dtype=float)
    precision = 1.0 / sigma2
    precision_sum = float(np.sum(precision))
    estimate = float(np.sum(z * precision) / precision_sum)
    variance = 1.0 / precision_sum
    return ScalarFusionResult(
        value=estimate,
        variance=variance,
        formula=FormulaReference(
            "UMRM-5-SCALAR-FUSION",
            DOCUMENT,
            5,
            "x_hat = sum(z_i/sigma_i^2)/sum(1/sigma_i^2); sigma_hat^2 = 1/sum(1/sigma_i^2)",
        ),
        intermediates={"precisions": tuple(float(v) for v in precision), "precision_sum": precision_sum},
    )


def sensor_effective_variance(
    *,
    measurement_variance: float,
    freshness: float,
    freshness_tau: float,
    healthy_probability: float,
    source_trust: float,
    trust_scale: float,
) -> FormulaResult:
    """Document 1, page 5.

    The document only gives ``sigma_trust^2 proportional to ...``; ``trust_scale`` is that
    proportionality constant.  The document gives no value, so it is a required argument and
    must be strictly positive (a zero scale would erase freshness/health/trust from the weight,
    contradicting "stale or degraded sensor has less weight").  The resulting variance must be
    strictly positive and finite because it is later used as 1/sigma^2 in fusion.
    """
    measurement = _non_negative("measurement_variance", measurement_variance)
    age = _non_negative("freshness", freshness)
    tau = _positive("freshness_tau", freshness_tau)
    healthy = _probability("healthy_probability", healthy_probability, allow_zero=False)
    trust = _probability("source_trust", source_trust, allow_zero=False)
    scale = _positive("trust_scale", trust_scale)
    trust_variance = scale * (1.0 + age / tau) / (healthy * trust)
    total = measurement + trust_variance
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("effective sensor variance must be finite and strictly positive")
    return FormulaResult(
        value=total,
        formula=FormulaReference(
            "UMRM-5-SENSOR-VARIANCE",
            DOCUMENT,
            5,
            "sigma_i^2 = sigma_measurement_i^2 + sigma_trust_i^2; sigma_trust_i^2 proportional to (1+freshness_i/tau)/(P(healthy_i)*sourceTrust_i)",
        ),
        intermediates={"trust_variance": trust_variance, "proportionality_scale": scale},
    )


def generative_observation(
    *,
    latent_state: Any,
    geometry_topology: Mapping[str, Any],
    body_perspective: Mapping[str, Any],
    sensor_calibration: Mapping[str, Any],
    noise: Any,
    measurement_model: Callable[[Any, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], Any],
) -> FormulaResult:
    """Execute z_t = h(X_t, G_t, b_t, theta_sensor) + nu_t."""

    if not callable(measurement_model):
        raise ValueError("measurement_model must be callable")
    expected = np.asarray(
        measurement_model(latent_state, geometry_topology, body_perspective, sensor_calibration),
        dtype=float,
    )
    disturbance = np.asarray(noise, dtype=float)
    if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(disturbance)):
        raise ValueError("measurement model output and noise must be finite")
    if disturbance.shape != expected.shape:
        # nu_t is the noise OF z_t: no silent broadcasting of a scalar h(.) into a vector (or vice versa)
        raise ValueError(f"noise shape {disturbance.shape} must equal the shape of h(.) {expected.shape}")
    value = expected + disturbance
    return FormulaResult(
        value=float(value) if value.ndim == 0 else value,
        formula=FormulaReference(
            "UMRM-4.1-OBSERVATION",
            DOCUMENT,
            3,
            "z_t = h(X_t, G_t, b_t, theta_sensor) + nu_t",
        ),
        intermediates={"expected_observation": float(expected) if expected.ndim == 0 else expected},
    )


def universal_transition(
    *,
    registry: Any,
    domain: str,
    state: Mapping[str, Any],
    controls: Mapping[str, Any],
    topology: Mapping[str, Any],
    boundary_conditions: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> FormulaResult:
    """UMRM section 6 (page 5): X_(t+1) = F(X_t, u_t, G_t, theta_t, eps_t).

    "F is implemented by Domain Dynamics Packs, not one global equation": the call is routed through
    a :class:`~api_gateway.core.world_model.dynamics_registry.DomainDynamicsRegistry` to the pack of
    ``domain`` (X_t = ``state``, u_t = ``controls``, G_t = ``topology`` + ``boundary_conditions``,
    theta_t = ``parameters``).  eps_t is represented by the uncertainty the pack must report for every
    predicted variable.  The result (``value``) is a PREDICTED
    :class:`~api_gateway.core.world_model.dynamics_registry.DynamicsTransitionResult` carrying
    model/version/uncertainty; it never mutates ``state``.  No registry / unknown domain -> error.
    """
    from .dynamics_registry import DomainDynamicsRegistry

    if not isinstance(registry, DomainDynamicsRegistry):
        raise TypeError("universal_transition requires a DomainDynamicsRegistry (F is implemented by domain packs)")
    result = registry.transition(
        domain=domain,
        state=state,
        controls=controls,
        topology=topology,
        boundary_conditions=boundary_conditions,
        parameters=parameters,
    )
    return FormulaResult(
        value=result,
        formula=UNIVERSAL_TRANSITION_FORMULA,
        intermediates={
            "routed_via": "DomainDynamicsRegistry",
            "domain": result.domain,
            "model": result.model,
            "model_version": result.pack_version,
            "epistemic_status": result.epistemic_status,
        },
    )


UNIVERSAL_TRANSITION_FORMULA = FormulaReference(
    "UMRM-6-UNIVERSAL-TRANSITION",
    DOCUMENT,
    5,
    "X_(t+1) = F(X_t, u_t, G_t, theta_t, eps_t); F is implemented by Domain Dynamics Packs "
    "F_domain: (state, controls, topology, boundary conditions, parameters) -> (next state, uncertainty)",
)


def accept_moe_candidate(
    *,
    schema_valid: bool,
    ontology_valid: bool,
    constraints_valid: bool,
    physics_valid: bool,
    safety_valid: bool,
    policy_valid: bool,
) -> FormulaResult:
    """Apply all deterministic gates to an MoE/LLM proposal.

    Fail-closed: a gate passes only when its check is a strict boolean ``True`` (``bool`` or
    ``np.bool_``, see :func:`is_strict_true`).  Strings such as ``"false"``/``"no"``/``"True"``, ``1``,
    ``None``, arrays or any other truthy object fail the gate.  The
    names of the failed gates are reported in ``intermediates["failed"]``.  An accepted candidate
    stays a candidate (it is not a fact).
    """

    raw = {
        "schema": schema_valid,
        "ontology": ontology_valid,
        "constraints": constraints_valid,
        "physics": physics_valid,
        "safety": safety_valid,
        "policy": policy_valid,
    }
    gates = {name: is_strict_true(value) for name, value in raw.items()}
    failed = tuple(name for name, passed in gates.items() if not passed)
    return FormulaResult(
        value=not failed,
        formula=FormulaReference(
            "UMRM-11-MOE-ACCEPTANCE",
            DOCUMENT,
            8,
            "Accept(c)=Schema(c) AND Ontology(c) AND Constraints(c) AND Physics(c) AND Safety(c) AND Policy(c)",
        ),
        intermediates={**gates, "failed": failed, "status_if_accepted": "candidate"},
    )


def thermal_rc_derivative(
    *,
    thermal_capacitance: float,
    indoor_temperature: float,
    outdoor_temperature: float,
    thermal_resistance: float,
    mass_flow_rate: float,
    specific_heat: float,
    supply_temperature: float,
    solar_gain: float,
    occupancy_gain: float,
    equipment_gain: float,
) -> FormulaResult:
    capacitance = _positive("thermal_capacitance", thermal_capacitance)
    resistance = _positive("thermal_resistance", thermal_resistance)
    indoor = _finite("indoor_temperature", indoor_temperature)
    outdoor = _finite("outdoor_temperature", outdoor_temperature)
    flow = _non_negative("mass_flow_rate", mass_flow_rate)
    cp = _positive("specific_heat", specific_heat)
    supply = _finite("supply_temperature", supply_temperature)
    gains = tuple(_finite(name, value) for name, value in (
        ("solar_gain", solar_gain),
        ("occupancy_gain", occupancy_gain),
        ("equipment_gain", equipment_gain),
    ))
    envelope = (outdoor - indoor) / resistance
    supply_exchange = flow * cp * (supply - indoor)
    total_power = envelope + supply_exchange + sum(gains)
    return FormulaResult(
        value=total_power / capacitance,
        formula=FormulaReference(
            "UMRM-6-THERMAL-RC",
            DOCUMENT,
            5,
            "C dT/dt = (T_out-T)/R + m_dot*c_p*(T_supply-T) + Q_solar + Q_occupancy + Q_equipment",
            output_unit="temperature-unit/time-unit",
        ),
        intermediates={"envelope_exchange": envelope, "supply_exchange": supply_exchange, "total_power": total_power},
    )


def time_to_collision(*, distance: float, closing_velocity: float, epsilon: float) -> FormulaResult:
    """UMRM section 6, page 6 (the "Vehicle TTC:" label is at the bottom of page 5, the formula on page 6)."""
    separation = _non_negative("distance", distance)
    closing = _finite("closing_velocity", closing_velocity)
    floor = _positive("epsilon", epsilon)
    denominator = max(closing, floor)
    return FormulaResult(
        value=separation / denominator,
        formula=FormulaReference(
            "UMRM-6-TTC",
            DOCUMENT,
            6,
            "TTC(a,b) = distance(a,b) / max(v_closing, epsilon)",
            {"distance": "length", "closing_velocity": "length/time", "epsilon": "length/time"},
            "time",
        ),
        intermediates={"denominator": denominator},
    )


def expected_information_gain(*, current_entropy: float, expected_posterior_entropy: float) -> FormulaResult:
    """UMRM section 8, page 7, from pre-computed entropies.

    EIG is the mutual information I(X; O | a) >= 0: in expectation an observation can never increase
    entropy.  A supplied ``expected_posterior_entropy`` larger than ``current_entropy`` is therefore an
    inconsistent input and is rejected (no silent clamping).  Prefer
    :func:`expected_information_gain_from_model`, which computes the expectation itself.
    """
    current = _non_negative("current_entropy", current_entropy)
    posterior = _non_negative("expected_posterior_entropy", expected_posterior_entropy)
    if posterior > current:
        raise ValueError(
            "expected_posterior_entropy exceeds current_entropy: EIG is mutual information and cannot be "
            "negative; the inputs are inconsistent (compute them with expected_information_gain_from_model)"
        )
    return FormulaResult(
        value=current - posterior,
        formula=FormulaReference(
            "UMRM-8-EIG",
            DOCUMENT,
            7,
            "EIG(a) = H(B_t) - E_o~P(o|a)[H(B_t+1|o,a)]",
        ),
        intermediates={"current_entropy": current, "expected_posterior_entropy": posterior},
    )


def expected_information_gain_from_model(*, prior: Sequence[float], likelihood: Sequence[Sequence[float]]) -> FormulaResult:
    """UMRM section 8, page 7: EIG computed from a discrete belief and observation model.

    ``prior[s] = B_t(s)``, ``likelihood[s][o] = P(o | s, a)``.  The expectation over o ~ P(o|a) is
    computed (delegated to ``reasoning.active_observation.expected_information_gain_discrete``).
    Mathematically the result is >= 0; a negative raw difference can only come from floating-point
    rounding, and is clamped to 0 with the explicit flag ``clamped_negative_to_zero``.
    """
    from ..reasoning.active_observation import expected_information_gain_discrete

    b = _finite_array("prior", prior)
    lik = _finite_array("likelihood", likelihood)
    result = expected_information_gain_discrete(prior=b, likelihood=lik)
    prior_entropy = float(result.intermediates["prior_entropy"])
    expected_posterior = float(result.intermediates["expected_posterior_entropy"])
    raw = prior_entropy - expected_posterior
    return FormulaResult(
        value=max(raw, 0.0),
        formula=result.formula,
        intermediates={
            **result.intermediates,
            "raw_difference": raw,
            "clamped_negative_to_zero": raw < 0.0,
        },
    )


def information_action_value(
    *,
    information_gain: float,
    cost: float,
    risk: float,
    latency: float,
    cost_weight: float,
    risk_weight: float,
    latency_weight: float,
) -> FormulaResult:
    eig = _non_negative("information_gain", information_gain)  # EIG is mutual information >= 0
    terms = {
        "weighted_cost": _non_negative("cost_weight", cost_weight) * _non_negative("cost", cost),
        "weighted_risk": _non_negative("risk_weight", risk_weight) * _non_negative("risk", risk),
        "weighted_latency": _non_negative("latency_weight", latency_weight) * _non_negative("latency", latency),
    }
    return FormulaResult(
        value=eig - sum(terms.values()),
        formula=FormulaReference(
            "UMRM-8-INFORMATION-ACTION",
            DOCUMENT,
            7,
            "VOI(a) = EIG(a) - lambda_c Cost(a) - lambda_r Risk(a) - lambda_t Latency(a)",
        ),
        intermediates=terms,
    )


def epistemic_uncertainty(
    *,
    coverage: float,
    evidence_count: int,
    freshness: float,
    freshness_tau: float,
    coverage_weight: float,
    evidence_weight: float,
    freshness_weight: float,
) -> FormulaResult:
    coverage_value = _probability("coverage", coverage)
    if (
        isinstance(evidence_count, (bool, np.bool_))
        or not isinstance(evidence_count, (int, float, np.integer, np.floating))
        or not np.isfinite(float(evidence_count))
        or float(evidence_count) != int(evidence_count)
        or evidence_count < 0
    ):
        raise ValueError("evidence_count must be a finite non-negative integer (not a bool, NaN or inf)")
    evidence_count = int(evidence_count)
    ratio = _non_negative("freshness", freshness) / _positive("freshness_tau", freshness_tau)
    terms = {
        "coverage": _non_negative("coverage_weight", coverage_weight) * (1.0 - coverage_value),
        "evidence": _non_negative("evidence_weight", evidence_weight) / (1.0 + evidence_count),
        "freshness": _non_negative("freshness_weight", freshness_weight) * min(1.0, ratio),
    }
    return FormulaResult(
        value=sum(terms.values()),
        formula=FormulaReference(
            "UMRM-9-EPISTEMIC-UNCERTAINTY",
            DOCUMENT,
            7,
            "U_epistemic = w_c(1-coverage) + w_e/(1+evidenceCount) + w_f min(1,freshness/tau)",
        ),
        intermediates=terms,
    )


def prediction_residual(*, observed: float, predicted: float) -> FormulaResult:
    actual = _finite("observed", observed)
    estimate = _finite("predicted", predicted)
    return FormulaResult(
        value=actual - estimate,
        formula=FormulaReference("UMRM-10-RESIDUAL", DOCUMENT, 8, "e_t = y_observed - y_predicted"),
    )


LEARNING_SIGN_NOTE = (
    "UMRM p.8 literally prints theta_candidate = theta + eta*grad_theta L(y_observed, y_predicted). "
    "L is not defined in the document; by notation it is a loss of the prediction residual e_t, so the "
    "literal '+' would be gradient ASCENT on the error. The printed sign is treated as a typo: we "
    "implement gradient descent theta - eta*grad_theta L (identical to the literal formula with L := -loss)."
)


LEARNING_CANDIDATE_FORMULA = FormulaReference(
    "UMRM-10-PARAMETER-CANDIDATE",
    DOCUMENT,
    8,
    "theta_candidate = theta - eta * grad_theta L(y_observed, y_predicted)  "
    "[document prints '+'; sign treated as typo, L is a loss]; result is a LearningCandidate, never production",
)


def learning_parameter_candidate(
    *,
    theta: Any,
    learning_rate: float,
    loss_gradient: Any,
    target: str,
    parameter_names: Sequence[str],
    proposed_by: str,
    evidence_refs: Sequence[str],
) -> FormulaResult:
    """UMRM section 10, page 8: parameter candidate by gradient DESCENT on the loss.

    ``theta`` and ``loss_gradient`` are vectors of the same shape (a scalar is a 1-vector);
    ``loss_gradient`` is grad_theta L(y_observed, y_predicted) where L is a loss (e.g. squared
    residual).  The value is a :class:`~api_gateway.core.world_model.learning_pipeline.LearningCandidate`
    with ``theta_candidate = theta - learning_rate * loss_gradient`` -- NOT a number that could be
    written into production.  "This never changes production directly": the candidate can only reach
    production through ``learning_pipeline`` (validation -> simulation -> golden replay -> shadow ->
    approval -> canary -> versioned deployment or rollback).

    Sign note: the document prints ``theta + eta * grad L``; with L a loss of the residual that would
    increase the error, so the printed sign is treated as a typo (equivalently the literal formula
    with L := -loss).  See :data:`LEARNING_SIGN_NOTE`.
    """
    from .learning_pipeline import LearningCandidate

    parameter = _finite_array("theta", theta).reshape(-1)
    gradient = _finite_array("loss_gradient", loss_gradient).reshape(-1)
    if parameter.size == 0 or parameter.shape != gradient.shape:
        raise ValueError("theta and loss_gradient must be non-empty vectors of the same shape")
    rate = _non_negative("learning_rate", learning_rate)
    update = -rate * gradient
    candidate = LearningCandidate(
        target=target,
        parameter_names=tuple(parameter_names),
        theta_current=parameter,
        theta_candidate=parameter + update,
        learning_rate=rate,
        loss_gradient=gradient,
        proposed_by=proposed_by,
        evidence_refs=tuple(evidence_refs),
    )
    return FormulaResult(
        value=candidate,
        formula=LEARNING_CANDIDATE_FORMULA,
        intermediates={"update": update, "sign_convention": "gradient_descent_on_loss", "note": LEARNING_SIGN_NOTE},
    )


class NotSpecifiedInTZ:
    """Marker for a part of a formula that the TZ PDF does not show (e.g. a truncated line)."""

    def __init__(self, where: str, visible_text: str) -> None:
        self.where = where
        self.visible_text = visible_text

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"NotSpecifiedInTZ({self.where!r}, visible={self.visible_text!r})"


PRODUCTION_OBJECTIVE_TAIL = NotSpecifiedInTZ(
    "UMRM 12, page 9: J",
    "J = lambda_1 Accuracy_state + lambda_2 Accuracy_causal + lambda_3 Accuracy_prediction + "
    "lambda_4 Accuracy_outcome - lambda_5 Unsa|<cut by the page edge>",
)


class MetricEnvironment(str, Enum):
    """UMRM 12 (page 9): every metric is separated by synthetic / replay / shadow / production."""

    SYNTHETIC = "synthetic"
    REPLAY = "replay"
    SHADOW = "shadow"
    PRODUCTION = "production"


@dataclass(frozen=True)
class MetricSlice:
    """UMRM 12 (page 9): mandatory metric slice labels.

    synthetic/replay/shadow/production x site x asset class x domain x action class x risk tier x data
    quality regime.  Every label is required and non-empty; metrics of different slices must not be
    aggregated into one number (:func:`require_same_slice`).
    """

    environment: MetricEnvironment
    site: str
    asset_class: str
    domain: str
    action_class: str
    risk_tier: str
    data_quality_regime: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "environment", MetricEnvironment(self.environment))
        for name in ("site", "asset_class", "domain", "action_class", "risk_tier", "data_quality_regime"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"metric slice label '{name}' is required (UMRM 12)")

    def key(self) -> tuple[str, ...]:
        return (self.environment.value, self.site, self.asset_class, self.domain, self.action_class,
                self.risk_tier, self.data_quality_regime)


def require_same_slice(slices: Sequence[MetricSlice]) -> MetricSlice:
    """Metrics may be combined only inside one slice (UMRM 12): mixing e.g. shadow and production raises."""
    items = tuple(slices)
    if not items or any(not isinstance(item, MetricSlice) for item in items):
        raise ValueError("at least one MetricSlice is required")
    if len({item.key() for item in items}) != 1:
        raise ValueError("metrics from different slices (environment/site/asset class/domain/action class/"
                         "risk tier/data quality regime) must not be aggregated (UMRM 12)")
    return items[0]


def production_objective(
    *,
    state_accuracy: float,
    causal_accuracy: float,
    prediction_accuracy: float,
    outcome_accuracy: float,
    unsafe: float,
    weights: Sequence[float],
    metric_slice: MetricSlice,
) -> FormulaResult:
    """UMRM section 12, page 9 -- ONLY the part of J visible in the document.

    The printed formula is cut by the page edge after ``- lambda_5 Unsa...``: nothing after that term
    is implemented or invented (``intermediates["tail"]`` is :data:`PRODUCTION_OBJECTIVE_TAIL`, a
    :class:`NotSpecifiedInTZ`).  The value is therefore the *visible part* of J.  Each Accuracy term is
    a fraction in [0, 1]; the Unsa[fe] term enters as a penalty (``- lambda_5 Unsafe``) so it must be
    non-negative; lambda_1..lambda_5 are non-negative weights.  ``metric_slice`` (UMRM 12: "every
    metric must be separated by ...") is mandatory and is returned with the value.
    """
    if not isinstance(metric_slice, MetricSlice):
        raise TypeError("metric_slice must be a MetricSlice (UMRM 12: every metric is separated by slice)")
    if len(weights) != 5:
        raise ValueError("weights must contain lambda_1 through lambda_5")
    accuracies = tuple(_probability(name, value) for name, value in (
        ("state_accuracy", state_accuracy),
        ("causal_accuracy", causal_accuracy),
        ("prediction_accuracy", prediction_accuracy),
        ("outcome_accuracy", outcome_accuracy),
    ))
    metrics = (*accuracies, _non_negative("unsafe", unsafe))
    lambdas = tuple(_non_negative("weight", value) for value in weights)
    value = sum(weight * metric for weight, metric in zip(lambdas[:4], metrics[:4])) - lambdas[4] * metrics[4]
    return FormulaResult(
        value=value,
        formula=FormulaReference(
            "UMRM-12-PRODUCTION-OBJECTIVE",
            DOCUMENT,
            9,
            "J = lambda_1 Accuracy_state + lambda_2 Accuracy_causal + lambda_3 Accuracy_prediction + "
            "lambda_4 Accuracy_outcome - lambda_5 Unsa[fe] ... (rest of the line not visible in the TZ)",
        ),
        intermediates={
            "weighted_accuracy": sum(weight * metric for weight, metric in zip(lambdas[:4], metrics[:4])),
            "weighted_unsafe": lambdas[4] * metrics[4],
            "metric_slice": metric_slice,
            "slice_key": metric_slice.key(),
            "completeness": "visible_terms_only",
            "tail": PRODUCTION_OBJECTIVE_TAIL,
        },
    )


# ---------------------------------------------------------------------------
# UMRM 4.4 -- capability gate (used by reality_contracts.BodyState).
# ---------------------------------------------------------------------------

CAPABILITY_FORMULA = FormulaReference(
    "UMRM-4.4-CAPABILITY",
    DOCUMENT,
    4,
    "Capability(action,t) = f(body availability, actuator health, network, policy, scope); "
    "f = declared AND body_available AND actuator_available AND actuator_health >= min_health "
    "AND network_available AND network_quality >= min_network AND policy AND scope",
)


def body_capability(
    *,
    capability_declared: bool,
    body_available: bool,
    actuator_available: bool,
    actuator_health: float,
    minimum_actuator_health: float,
    network_available: bool,
    network_quality: float,
    minimum_network_quality: float,
    policy_allows: bool,
    in_scope: bool,
) -> FormulaResult:
    """UMRM 4.4 (page 4): Capability(action, t) = f(body availability, actuator health, network, policy, scope).

    This is the ONLY implementation of the capability form in the code base (``BodyState.capability``,
    ``SelfModelState.capability`` and ``reasoning.active_observation.action_capability`` delegate
    here).  The form of f is a conjunction of hard gates; the document gives no thresholds, so both
    minimums are required and must lie in (0, 1] (a zero minimum would disable the check).
    ``actuator_available`` is the interoceptive actuator availability of I_t (UMRM 4.4); it is separate
    from ``actuator_health`` and from ``body_available``.  Every flag must be a strict bool; the result
    is ``True`` only when no blocker applies (fail-closed).
    """
    declared = _strict_bool("capability_declared", capability_declared)
    body = _strict_bool("body_available", body_available)
    actuators_up = _strict_bool("actuator_available", actuator_available)
    network_up = _strict_bool("network_available", network_available)
    policy = _strict_bool("policy_allows", policy_allows)
    scope = _strict_bool("in_scope", in_scope)
    health = _probability("actuator_health", actuator_health)
    min_health = _probability("minimum_actuator_health", minimum_actuator_health, allow_zero=False)
    network = _probability("network_quality", network_quality)
    min_network = _probability("minimum_network_quality", minimum_network_quality, allow_zero=False)
    blockers: list[str] = []
    if not declared:
        blockers.append("capability_not_declared")
    if not body:
        blockers.append("body_unavailable")
    if not actuators_up:
        blockers.append("actuator_unavailable")
    if health < min_health:
        blockers.append("actuator_health_below_minimum")
    if not network_up:
        blockers.append("network_unavailable")
    if network < min_network:
        blockers.append("network_quality_below_minimum")
    if not policy:
        blockers.append("policy_forbids")
    if not scope:
        blockers.append("out_of_scope")
    return FormulaResult(
        value=not blockers,
        formula=CAPABILITY_FORMULA,
        intermediates={
            "blockers": tuple(blockers),
            "actuator_health": health,
            "minimum_actuator_health": min_health,
            "network_quality": network,
            "minimum_network_quality": min_network,
        },
    )


# ---------------------------------------------------------------------------
# UMRM formulas implemented in helper modules (spatial_formulas, belief_dynamics,
# reasoning.active_observation), wired into this API.  Imports are lazy because
# those modules import FormulaReference/FormulaResult from this module.  The
# wrappers add the fail-closed input checks (finiteness, strict booleans, proper
# rotations, required boundary conditions) that the helpers leave open.
# ---------------------------------------------------------------------------

_LAZY_EXPORTS = {
    "FrameRegistry": (".spatial_formulas", "FrameRegistry"),
    "ActionCandidate": ("..reasoning.active_observation", "ActionCandidate"),
    "ACTION_CONSTRAINTS": ("..reasoning.active_observation", "ACTION_CONSTRAINTS"),
    "InformationActionCandidate": ("..reasoning.active_observation", "InformationActionCandidate"),
    "InformationActionPolicy": ("..reasoning.active_observation", "InformationActionPolicy"),
    "LearningCandidate": (".learning_pipeline", "LearningCandidate"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attribute = _LAZY_EXPORTS[name]
    return getattr(importlib.import_module(module_name, __package__), attribute)


def transform_point(transform: Any, point: Sequence[float]) -> FormulaResult:
    """UMRM 3.2 (page 2): ^W p = ^W T_S . ^S p  (T must be a proper rigid SE(3) transform)."""
    from .spatial_formulas import transform_point as _transform_point

    return _transform_point(transform, point)


def compose_transform_chain(transforms: Sequence[Any]) -> FormulaResult:
    """UMRM 3.2 (page 2): ^W T_sensor = ^W T_site . ^site T_building . ^building T_floor . ^floor T_sensor."""
    from .spatial_formulas import compose_transform_chain as _compose

    return _compose(list(transforms))


def camera_projection(*, intrinsics: Any, rotation: Any, translation: Sequence[float], world_point: Sequence[float]) -> FormulaResult:
    """UMRM 4.2 (page 3): u~ = K [R | t] X~_W.  A point behind the camera is reported as not
    projectable (``pixel=None``), never as a pixel."""
    from .spatial_formulas import project_point

    k = _finite_array("intrinsics", intrinsics)
    r = validate_rotation(rotation)
    if r.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    return project_point(intrinsics=k, rotation=r, translation=translation, world_point=world_point)


def visibility_state(
    *,
    sensor_available: bool,
    in_field_of_view: bool,
    line_of_sight: bool,
    in_range: bool,
    occluded: bool,
    detection_confidence: float | None,
    confidence_threshold: float,
    sensor_calibrated: bool,
) -> FormulaResult:
    """UMRM 4.3 (page 4): V(agent,obj,t) = FOV AND LoS AND Range AND NOT Occlusion.

    Missing detection is never absence: the result distinguishes the six visibility states.
    ``confidence_threshold`` has no value in the document and is therefore required.
    """
    from .spatial_formulas import visibility

    flags = {
        name: _strict_bool(name, value)
        for name, value in (
            ("sensor_available", sensor_available),
            ("in_field_of_view", in_field_of_view),
            ("line_of_sight", line_of_sight),
            ("in_range", in_range),
            ("occluded", occluded),
            ("sensor_calibrated", sensor_calibrated),
        )
    }
    confidence = None if detection_confidence is None else _probability("detection_confidence", detection_confidence)
    threshold = _probability("confidence_threshold", confidence_threshold)
    return visibility(detection_confidence=confidence, confidence_threshold=threshold, **flags)


def belief_posterior(*, prior: Sequence[float], transition: Sequence[Sequence[float]], likelihood: Sequence[float]) -> FormulaResult:
    """UMRM 5 (page 4): p(X_t|z_1:t) proportional to p(z_t|X_t) * sum p(X_t|X_t-1,u_t) p(X_t-1|z_1:t-1).

    Discrete Bayes filter step; an observation impossible under the model raises (no silent update).
    """
    from .belief_dynamics import bayes_filter_step

    return bayes_filter_step(
        prior=_finite_array("prior", prior),
        transition=_finite_array("transition", transition),
        likelihood=_finite_array("likelihood", likelihood),
    )


def crowd_conservation_step(
    *,
    density: Sequence[float],
    face_velocity: Sequence[float],
    source: Sequence[float],
    dx: float,
    dt: float,
    boundary_density: tuple[float, float],
) -> FormulaResult:
    """UMRM 6 (page 6): d rho/dt + div(rho v) = s(x, t)  -- 1D upwind finite-volume step.

    ``boundary_density`` (inflow densities at the left/right boundary) has no value in the
    document and is therefore a required argument.
    """
    from .belief_dynamics import crowd_conservation_step as _crowd_step

    boundary = _finite_array("boundary_density", boundary_density)
    if boundary.shape != (2,) or np.any(boundary < 0.0):
        raise ValueError("boundary_density must be two non-negative finite densities (left, right)")
    return _crowd_step(
        density=_finite_array("density", density),
        face_velocity=_finite_array("face_velocity", face_velocity),
        source=_finite_array("source", source),
        dx=_positive("dx", dx),
        dt=_positive("dt", dt),
        boundary_density=(float(boundary[0]), float(boundary[1])),
    )


def select_physical_action(candidates: Sequence[Any]) -> FormulaResult:
    """UMRM 8 (page 7): a* = argmax_a U(a | B_t, C_t, M_t) subject to the seven constraint classes.

    A constraint passes only when it is the boolean ``True``; no feasible candidate -> ``DEFER``.
    """
    from ..reasoning.active_observation import select_physical_action as _select

    return _select(list(candidates))


def select_information_action(candidates: Sequence[Any], *, policy: Any) -> FormulaResult:
    """UMRM 8 (page 7): a* = argmax_a [EIG(a) - lambda_c Cost(a) - lambda_r Risk(a) - lambda_t Latency(a)];
    lambda_c/r/t come from a required InformationActionPolicy."""
    from ..reasoning.active_observation import select_information_action as _select

    return _select(list(candidates), policy=policy)


def choose_physical_or_information_action(**kwargs: Any) -> FormulaResult:
    """UMRM 8 (page 7): a plan chooses either a physical action or an information action."""
    from ..reasoning.active_observation import choose_physical_or_information_action as _choose

    return _choose(**kwargs)


def epistemic_action_gate(
    *,
    epistemic_uncertainty: float,
    action_class_threshold: float,
    missing_observations: Sequence[str],
) -> FormulaResult:
    """UMRM 9 (page 7): U_epistemic > theta_action_class => DEFER.

    The threshold is per action class and has no value in the document (required).  A DEFER must
    name the missing observation/capability that would reduce the uncertainty.
    """
    from ..reasoning.active_observation import epistemic_action_gate as _gate

    # validation (finite, non-negative, named gaps) lives in the single gate implementation, so the
    # choose_physical_or_information_action path cannot bypass it (NaN/inf -> ValueError, not PROCEED)
    return _gate(epistemic_uncertainty=epistemic_uncertainty, action_class_threshold=action_class_threshold,
                 missing_observations=missing_observations)


__all__ = [
    "ACTION_CONSTRAINTS",
    "ActionCandidate",
    "CAPABILITY_FORMULA",
    "FormulaReference",
    "InformationActionCandidate",
    "InformationActionPolicy",
    "LEARNING_CANDIDATE_FORMULA",
    "LearningCandidate",
    "MetricEnvironment",
    "MetricSlice",
    "NotSpecifiedInTZ",
    "PRODUCTION_OBJECTIVE_TAIL",
    "FormulaResult",
    "FrameRegistry",
    "LEARNING_SIGN_NOTE",
    "ScalarFusionResult",
    "UNIVERSAL_TRANSITION_FORMULA",
    "accept_moe_candidate",
    "belief_posterior",
    "body_capability",
    "camera_projection",
    "choose_physical_or_information_action",
    "is_strict_bool",
    "is_strict_true",
    "require_same_slice",
    "select_information_action",
    "strict_bool",
    "compose_transform_chain",
    "crowd_conservation_step",
    "epistemic_action_gate",
    "epistemic_uncertainty",
    "expected_information_gain",
    "expected_information_gain_from_model",
    "generative_observation",
    "information_action_value",
    "learning_parameter_candidate",
    "prediction_residual",
    "production_objective",
    "select_physical_action",
    "sensor_effective_variance",
    "thermal_rc_derivative",
    "time_to_collision",
    "transform_covariance",
    "transform_point",
    "universal_transition",
    "validate_covariance",
    "validate_rotation",
    "visibility_state",
    "weighted_scalar_fusion",
]

