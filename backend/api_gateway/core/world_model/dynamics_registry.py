from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np

from ..common.epistemic import EpistemicStatus
from .assertions import reject_non_finite
from .reality_formulas import FormulaReference

if TYPE_CHECKING:  # pragma: no cover
    from .assertions import WorldModelAssertion


DynamicsPredictor = Callable[
    [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    tuple[Mapping[str, Any], Mapping[str, Any]],
]

# UMRM section 6 (pages 5-6): every Domain Dynamics Pack satisfies
#   F_domain: (state, controls, topology, boundary conditions, parameters) -> (next state, uncertainty)
# and "no domain pack is allowed to mutate universal state directly; it produces predicted
# assertions with model/version/uncertainty".
DOMAIN_DYNAMICS_FORMULA = FormulaReference(
    "UMRM-6-DOMAIN-DYNAMICS-CONTRACT",
    "CURE-Unified-Mathematical-Reality-Model",
    5,
    "F_domain: (state, controls, topology, boundary conditions, parameters) -> (next state, uncertainty); "
    "packs never mutate universal state (p.6)",
)

# The uncertainty of predicted variable ``v`` may be reported under ``v`` itself or under ``v``
# plus one of these suffixes (standard deviation / variance / covariance matrix for a vector
# variable).  Every predicted variable needs one.
UNCERTAINTY_KEY_SUFFIXES = ("", "_std", "_variance", "_covariance")
UNCERTAINTY_KIND = {"": "as_reported", "_std": "std", "_variance": "variance", "_covariance": "covariance"}


def _freeze(value: Any) -> Any:
    """Deep read-only copy handed to a pack, so it cannot mutate the caller's universal state."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        frozen = np.array(value, copy=True)
        frozen.setflags(write=False)
        return frozen
    return copy.deepcopy(value)


def _uncertainty_number(key: str, raw: Any) -> Any:
    """A finite non-negative scalar, or (for vector variables) an array of them; a ``*_covariance``
    entry must be a finite symmetric positive semi-definite matrix."""
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError(f"uncertainty '{key}' must be a finite non-negative number")
    if isinstance(raw, (int, float, np.integer, np.floating)):
        number = float(raw)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"uncertainty '{key}' must be a finite non-negative number")
        return number
    if isinstance(raw, (list, tuple, np.ndarray)):
        try:
            array = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"uncertainty '{key}' must be numeric") from exc
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError(f"uncertainty '{key}' must be finite and non-empty")
        if key.endswith("_covariance"):
            from .reality_formulas import validate_covariance

            return validate_covariance(array, name=f"uncertainty '{key}'")
        if np.any(array < 0.0):
            raise ValueError(f"uncertainty '{key}' must be non-negative")
        return array
    raise ValueError(f"uncertainty '{key}' must be a finite non-negative number")


def _uncertainty_value(variable: str, uncertainty: Mapping[str, Any]) -> tuple[str, Any]:
    for suffix in UNCERTAINTY_KEY_SUFFIXES:
        key = f"{variable}{suffix}"
        if key in uncertainty:
            return key, _uncertainty_number(key, uncertainty[key])
    raise ValueError(
        f"domain pack returned no uncertainty for predicted variable '{variable}' "
        f"(expected one of {[variable + suffix for suffix in UNCERTAINTY_KEY_SUFFIXES]})"
    )


@dataclass(frozen=True)
class DomainDynamicsPack:
    domain: str
    version: str
    predictor: DynamicsPredictor
    model: str = ""

    def __post_init__(self) -> None:
        if not self.domain.strip() or not self.version.strip() or not callable(self.predictor):
            raise ValueError("domain dynamics pack requires domain, version, and predictor")
        if not isinstance(self.model, str):
            raise ValueError("model must be a string")
        if not self.model.strip():
            object.__setattr__(self, "model", self.domain)


@dataclass(frozen=True)
class DynamicsTransitionResult:
    """A predicted assertion set (UMRM 6), never a mutation of the world state."""

    domain: str
    pack_version: str
    next_state: Mapping[str, Any]
    uncertainty: Mapping[str, Any]
    epistemic_status: EpistemicStatus = EpistemicStatus.PREDICTED
    formula: FormulaReference = field(default=DOMAIN_DYNAMICS_FORMULA)
    model: str = ""
    real_action: bool = False

    def to_assertions(
        self,
        *,
        entity_id: str,
        event_time: Any,
        knowledge_time: Any,
        valid_time_start: Any | None = None,
        scope: str = "default",
        trace_id: str = "",
        provenance_refs: tuple[str, ...] = (),
    ) -> tuple["WorldModelAssertion", ...]:
        """UMRM 6 (page 6): the pack output as *predicted assertions with model/version/uncertainty*.

        One :class:`WorldModelAssertion` per predicted variable, ``epistemic_status=PREDICTED``,
        ``source_id="dynamics:<domain>:<model>:<version>"``, ``action_authority="plan_support"``.
        ``uncertainty`` holds the scalar reported by the pack; for a vector variable it holds the
        largest element-wise value (a conservative scalar summary) and the full array/matrix is kept
        in ``metadata["uncertainty"]``.  These assertions are never OBSERVED facts.
        """
        from .assertions import WorldModelAssertion

        out = []
        for variable, value in self.next_state.items():
            key, raw = _uncertainty_value(str(variable), self.uncertainty)
            suffix = key[len(str(variable)):]
            if isinstance(raw, np.ndarray):
                summary = float(np.max(np.diag(raw) if suffix == "_covariance" else raw))
                detail: Any = raw.tolist()
            else:
                summary, detail = float(raw), float(raw)
            out.append(
                WorldModelAssertion(
                    entity_id=entity_id,
                    attribute_or_relation=str(variable),
                    value=np.asarray(value).tolist() if isinstance(value, (np.ndarray, tuple, list)) else value,
                    event_time=event_time,
                    knowledge_time=knowledge_time,
                    valid_time_start=event_time if valid_time_start is None else valid_time_start,
                    source_id=f"dynamics:{self.domain}:{self.model or self.domain}:{self.pack_version}",
                    provenance_refs=tuple(provenance_refs),
                    epistemic_status=EpistemicStatus.PREDICTED,
                    uncertainty=summary,
                    scope=scope,
                    trace_id=trace_id,
                    action_authority="plan_support",
                    metadata={
                        "model": self.model or self.domain,
                        "model_version": self.pack_version,
                        "domain": self.domain,
                        "uncertainty_key": key,
                        "uncertainty_kind": UNCERTAINTY_KIND[suffix],
                        "uncertainty": detail,
                        "formula_id": self.formula.formula_id,
                        "real_action": False,
                    },
                )
            )
        return tuple(out)


class DomainDynamicsRegistry:
    """Routes the universal F_domain contract to versioned domain implementations."""

    def __init__(self) -> None:
        self._packs: dict[str, DomainDynamicsPack] = {}

    def register(self, pack: DomainDynamicsPack) -> None:
        if pack.domain in self._packs:
            raise ValueError(f"domain dynamics pack already registered: {pack.domain}")
        self._packs[pack.domain] = pack

    def transition(
        self,
        *,
        domain: str,
        state: Mapping[str, Any],
        controls: Mapping[str, Any],
        topology: Mapping[str, Any],
        boundary_conditions: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> DynamicsTransitionResult:
        """Run ``F_domain`` on read-only deep copies of its inputs.

        Fail-closed: a pack that tries to write into its inputs raises (``TypeError`` from the
        read-only mapping); a non-empty next state is required and every predicted variable must
        carry a finite non-negative uncertainty; numeric predictions must be finite.  The result is
        marked ``EpistemicStatus.PREDICTED``.
        """
        try:
            pack = self._packs[str(domain)]
        except KeyError as exc:
            raise KeyError(f"no dynamics pack registered for domain: {domain}") from exc
        inputs = (state, controls, topology, boundary_conditions, parameters)
        if not all(isinstance(item, Mapping) for item in inputs):
            raise ValueError("state, controls, topology, boundary_conditions and parameters must be mappings")
        predicted = pack.predictor(*(_freeze(item) for item in inputs))
        if not isinstance(predicted, tuple) or len(predicted) != 2:
            raise ValueError("domain predictor must return (next_state, uncertainty)")
        next_state, uncertainty = predicted
        if not isinstance(next_state, Mapping) or not isinstance(uncertainty, Mapping):
            raise ValueError("domain predictor outputs must be mappings")
        if not next_state:
            raise ValueError("domain predictor returned an empty next state")
        reject_non_finite("next_state", next_state)
        for key, raw in uncertainty.items():
            _uncertainty_number(str(key), raw)
        for variable in next_state:
            _uncertainty_value(str(variable), uncertainty)
        return DynamicsTransitionResult(
            pack.domain,
            pack.version,
            _freeze(dict(next_state)),
            _freeze(dict(uncertainty)),
            model=pack.model,
        )

    def versions(self) -> Mapping[str, str]:
        return {domain: pack.version for domain, pack in self._packs.items()}

    def domains(self) -> tuple[str, ...]:
        return tuple(self._packs)

    def pack(self, domain: str) -> DomainDynamicsPack:
        return self._packs[domain]


__all__ = [
    "DOMAIN_DYNAMICS_FORMULA",
    "DomainDynamicsPack",
    "DomainDynamicsRegistry",
    "DynamicsTransitionResult",
    "UNCERTAINTY_KEY_SUFFIXES",
    "UNCERTAINTY_KIND",
]

