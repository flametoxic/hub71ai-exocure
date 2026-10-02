"""Shared basics: sourced data, refusal on missing data, the engine answer contract, Monte Carlo samples."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from api_gateway.core.operational_integrity.context import truthful_label, weakest_mode


class MissingData(Exception):
    """No value, or a value without a source → nothing is computed; the list of missing items is returned."""

    def __init__(self, missing: Iterable[str]):
        self.missing = sorted(set(missing))
        super().__init__(", ".join(self.missing))


class Store:
    """A YAML document where every number is {value, source | decided_by}. A number without a source does not exist."""

    def __init__(self, doc: Mapping[str, Any], name: str):
        self.doc, self.name = dict(doc), name
        self.meta = dict(doc.get("meta", {}))

    @classmethod
    def load(cls, path: Path) -> "Store":
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")) or {}, path.stem)

    @property
    def data_mode(self) -> str:
        return self.meta.get("data_mode", "synthetic")

    def raw(self, path: str) -> Any:
        node: Any = self.doc
        for key in path.split("."):
            if not isinstance(node, Mapping) or key not in node:
                raise MissingData([f"{self.name}:{path}"])
            node = node[key]
        return node

    def get(self, path: str) -> Any:
        node = self.raw(path)
        if isinstance(node, Mapping) and "value" in node:
            if node["value"] is None:
                raise MissingData([f"{self.name}:{path}"])
            if not (node.get("source") or node.get("decided_by")):
                raise MissingData([f"{self.name}:{path}.source"])
            return node["value"]
        return node

    def sources(self) -> list[dict]:
        return [dict(self.meta, dataset=self.name)]


def sample(spec: Mapping[str, Any], rng: np.random.Generator, n: int) -> np.ndarray:
    """The distribution is defined in the data: fixed / uniform / triangular."""
    d = spec.get("dist")
    if d == "fixed":
        return np.full(n, float(spec["value"]))
    if d == "uniform":
        return rng.uniform(float(spec["low"]), float(spec["high"]), n)
    if d == "triangular":
        return rng.triangular(float(spec["low"]), float(spec["mode"]), float(spec["high"]), n)
    raise MissingData([f"distribution:{d}"])


def interval(samples: Iterable[float], q: tuple[float, float]) -> dict:
    a = np.asarray(list(samples) if not isinstance(samples, np.ndarray) else samples, float)
    return {"low": float(np.quantile(a, q[0])), "mid": float(np.median(a)), "high": float(np.quantile(a, q[1])),
            "q": list(q), "method": "monte_carlo", "runs": int(a.size)}


ANSWERS: dict[str, dict] = {}  # trace_id → answer (for "why?" and number checks)


def answer(value: Any, *, unit: str, stores: Iterable[Store], formulas: Iterable[str], label: str = "",
           interval_: dict | None = None, extra: Mapping[str, Any] | None = None) -> dict:
    stores = list(stores)
    mode = weakest_mode([s.data_mode for s in stores]).value
    if label:
        truthful_label(label, mode)  # synthetic data can never be labelled "real time"
    out = {"trace_id": uuid.uuid4().hex, "calculation_status": "computed", "value": value, "unit": unit,
           "interval": interval_, "data_mode": mode, "sources": [m for s in stores for m in s.sources()],
           "formulas": list(formulas), "missing": [], "label": label, **dict(extra or {})}
    ANSWERS[out["trace_id"]] = out
    return out


def refused(err: MissingData, *, what: str) -> dict:
    out = {"trace_id": uuid.uuid4().hex, "calculation_status": "refused", "value": None, "what": what,
           "missing": err.missing, "message": "Cannot compute: data is missing. Needed: " + ", ".join(err.missing)}
    ANSWERS[out["trace_id"]] = out
    return out
