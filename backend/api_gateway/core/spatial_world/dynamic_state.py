"""DynamicStateService (мастер-ТЗ A4; WM Spec §2.10): треки, траектории, конфликт траекторий, слияние видов.

    d² = (z − H x̂⁻)^T S^−1 (z − H x̂⁻),  связывать при d² < 5.99 (95 %, 2 степени свободы)
    TTC(a,b) = d(a,b) / max(v_closing, ε)
    cross-view: score = LR / (1 + LR),  LR = N(ν; 0, S) / λ_clutter;  один канонический трек при score ≥ 0.85
Модель движения — постоянная скорость в плоскости мира (x, y, vx, vy). Трек — не сущность: канонический
entity_id появляется только после слияния видов / разрешения идентичности.
"""
from __future__ import annotations

import math
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

import numpy as np

from .policy import F_GATE, F_TTC, SpatialWorldError, SpatialWorldPolicy, aware, finite, text

H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])


@dataclass
class TrackState:
    track_id: str
    x: np.ndarray                 # [x, y, vx, vy] в корневом кадре
    p: np.ndarray
    last_seen: datetime
    sources: list = field(default_factory=list)
    entity_id: Optional[str] = None
    merged_into: Optional[str] = None


class DynamicStateService:
    def __init__(self, policy: SpatialWorldPolicy, *, accel_noise: float) -> None:
        self.policy = policy
        self.q = finite("accel_noise", accel_noise)          # спектральная плотность ускорения (пакет/площадка)
        if self.q < 0:
            raise SpatialWorldError("accel_noise must be ≥ 0")
        self.tracks: dict[str, TrackState] = {}

    def _predict(self, tr: TrackState, t: datetime) -> tuple[np.ndarray, np.ndarray]:
        dt = max(0.0, (t - tr.last_seen).total_seconds())
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        g = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        return F @ tr.x, F @ tr.p @ F.T + self.q * g @ g.T

    def gate(self, tr: TrackState, z: Sequence[float], r: np.ndarray, t: datetime) -> dict:
        x, p = self._predict(tr, t)
        nu = np.asarray(z, float) - H @ x
        s = H @ p @ H.T + r
        d2 = float(nu @ np.linalg.solve(s, nu))
        return {"d2": d2, "associate": d2 < self.policy.gate_chi2, "nu": nu, "S": s, "x": x, "P": p, "formula": F_GATE}

    def observe(self, source: str, z: Sequence[float], r: Sequence[Sequence[float]], t: datetime,
                velocity: Optional[Sequence[float]] = None) -> TrackState:
        """Связать наблюдение положения (в корневом кадре) с ближайшим треком по гейту или открыть новый."""
        t = aware("t", t)
        rr = np.asarray(r, float)
        best, best_g = None, None
        for tr in self.tracks.values():
            if tr.merged_into:
                continue
            g = self.gate(tr, z, rr, t)
            if g["associate"] and (best_g is None or g["d2"] < best_g["d2"]):
                best, best_g = tr, g
        if best is None:
            x0 = np.array([*z, *(velocity if velocity is not None else (0.0, 0.0))], float)
            p0 = np.zeros((4, 4))
            p0[:2, :2] = rr
            p0[2:, 2:] = np.eye(2) * (0.0 if velocity is not None else 1e6)   # скорость неизвестна до 2-го замера
            tr = TrackState(f"track:{secrets.token_hex(5)}", x0, p0, t, [text("source", source)])
            self.tracks[tr.track_id] = tr
            return tr
        k = best_g["P"] @ H.T @ np.linalg.inv(best_g["S"])
        best.x = best_g["x"] + k @ best_g["nu"]
        best.p = (np.eye(4) - k @ H) @ best_g["P"]
        best.last_seen = t
        if source not in best.sources:
            best.sources.append(source)
        return best

    def ttc(self, a: str, b: str, *, at: datetime) -> dict:
        ta, tb = self.tracks[a], self.tracks[b]
        xa, _ = self._predict(ta, aware("at", at))
        xb, _ = self._predict(tb, at)
        dp, dv = xb[:2] - xa[:2], xb[2:] - xa[2:]
        d = float(np.linalg.norm(dp))
        closing = -float(np.dot(dp, dv)) / d if d > 0 else 0.0
        eps = self.policy.ttc_epsilon_mps
        return {"ttc_s": d / max(closing, eps), "distance_m": d, "closing_mps": closing, "epsilon": eps,
                "formula": F_TTC}

    def cross_view_score(self, a: str, b: str, *, at: datetime) -> float:
        ta, tb = self.tracks[a], self.tracks[b]
        xa, pa = self._predict(ta, aware("at", at))
        xb, pb = self._predict(tb, at)
        nu = H @ (xa - xb)
        s = H @ (pa + pb) @ H.T
        like = math.exp(-0.5 * float(nu @ np.linalg.solve(s, nu))) / (2 * math.pi * math.sqrt(np.linalg.det(s)))
        lr = like / self.policy.clutter_density
        return lr / (1 + lr)

    def cross_view_merge(self, a: str, b: str, *, at: datetime) -> dict:
        score = self.cross_view_score(a, b, at=at)
        if score < self.policy.cross_view_min:
            return {"merged": False, "score": score, "threshold": self.policy.cross_view_min}
        ta, tb = self.tracks[a], self.tracks[b]
        xa, pa = self._predict(ta, at)
        xb, pb = self._predict(tb, at)
        pinv = np.linalg.inv(np.linalg.inv(pa + np.eye(4) * 1e-12) + np.linalg.inv(pb + np.eye(4) * 1e-12))
        ta.x = pinv @ (np.linalg.solve(pa + np.eye(4) * 1e-12, xa) + np.linalg.solve(pb + np.eye(4) * 1e-12, xb))
        ta.p, ta.last_seen = pinv, aware("at", at)
        ta.sources = sorted(set(ta.sources) | set(tb.sources))
        tb.merged_into = a
        return {"merged": True, "score": score, "canonical_track": a, "sources": ta.sources}


__all__ = ["DynamicStateService", "TrackState"]
