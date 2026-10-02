"""Уровни автономии (мастер-ТЗ E4, §14 «no global autonomy switch») и энтропийный сигнал (мастер-ТЗ D4).

  • Уровень выдаётся ТОЛЬКО на точный ключ (tenant, site, asset, action_class, risk_tier). Шаблонов, «*»
    и глобального переключателя нет по построению: ключ без выдачи → SHADOW.
  • Выше SHADOW — только при накопленных теневых свидетельствах: n_shadow ≥ N_min и доля подтверждённых
    ≥ a_min (пороги — из политики площадки, не константы кода) + подписанное одобрение человека.
  • Устаревшее состояние мира понижает уровень на одну ступень (ТЗ Sovereign §9).
  • Энтропия S = −Σ p·ln p над апостериорным распределением состояния (публикует WM), скорость dS/dt —
    наклон МНК по окну. dS/dt > S_crit → SAFE_MODE (автономия отозвана, локальные безопасные уставки,
    оператор); dS/dt > S_warn → HUMAN_REVIEW; иначе решение остаётся за политикой. SAFE_MODE снимает
    только оператор (подписанный сброс), сам сигнал его не снимает.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Mapping, Optional, Sequence

from ..world_model.reality_formulas import FormulaReference
from .contracts import SovereignError, aware, finite, text
from .trust import sign_artifact, verify_artifact

# Мастер-ТЗ D4 (стр. 14) пишет Ṡ_sys = d/dt Σ p_i ln p_i — без минуса это производная ОТРИЦАТЕЛЬНОЙ энтропии, и
# правило «dS/dt > S_crit → SAFE_MODE» срабатывало бы при РОСТЕ уверенности. Смысл правила — рост неопределённости,
# поэтому используется энтропия Шеннона S = −Σ p ln p (расхождение знака вынесено в сверку с ТЗ).
F_ENTROPY = FormulaReference("MASTER-D4-ENTROPY", "CURE-REALITY-ENGINE-MASTER-TZ", 14,
                             "S = −Σ p_i ln p_i; dS/dt > S_crit → SAFE_MODE; dS/dt > S_warn → HUMAN_REVIEW")


class AutonomyTier(str, Enum):
    SHADOW = "shadow"
    HUMAN_APPROVED = "human_approved"
    LIMITED_AUTONOMY = "limited_autonomy"

    @property
    def rank(self) -> int:
        return ("shadow", "human_approved", "limited_autonomy").index(self.value)

    @staticmethod
    def of_rank(r: int) -> "AutonomyTier":
        return (AutonomyTier.SHADOW, AutonomyTier.HUMAN_APPROVED, AutonomyTier.LIMITED_AUTONOMY)[max(0, min(2, r))]


class SafetySignal(str, Enum):
    NORMAL = "normal"
    HUMAN_REVIEW = "human_review"
    SAFE_MODE = "safe_mode"


@dataclass(frozen=True)
class AutonomyKey:
    tenant: str
    site: str
    asset: str
    action_class: str
    risk_tier: str

    def __post_init__(self) -> None:
        for n in ("tenant", "site", "asset", "action_class", "risk_tier"):
            v = text(n, getattr(self, n))
            if v in ("*", "any", "all", "global"):
                raise SovereignError("autonomy is granted per tenant/site/asset/action/risk tier — never globally")
            object.__setattr__(self, n, v)

    def as_list(self) -> list:
        return [self.tenant, self.site, self.asset, self.action_class, self.risk_tier]


@dataclass(frozen=True)
class AutonomyPolicy:
    policy_version: str
    min_shadow_runs: int
    min_shadow_agreement: float
    entropy_warn_rate: float        # S_warn, нат/с
    entropy_crit_rate: float        # S_crit, нат/с
    entropy_window_s: float
    max_world_state_age_s: float

    def __post_init__(self) -> None:
        text("policy_version", self.policy_version)
        if int(self.min_shadow_runs) < 1:
            raise SovereignError("min_shadow_runs must be ≥ 1: no autonomy without shadow evidence")
        if not 0.0 < finite("min_shadow_agreement", self.min_shadow_agreement) <= 1.0:
            raise SovereignError("min_shadow_agreement must be in (0, 1]")
        if not 0.0 < finite("entropy_warn_rate", self.entropy_warn_rate) < finite("entropy_crit_rate", self.entropy_crit_rate):
            raise SovereignError("need 0 < S_warn < S_crit")
        if finite("entropy_window_s", self.entropy_window_s) <= 0 or finite("max_world_state_age_s", self.max_world_state_age_s) <= 0:
            raise SovereignError("entropy window and world state age limit must be positive")


@dataclass(frozen=True)
class AutonomyGrant:
    key: AutonomyKey
    tier: AutonomyTier
    approved_by: str
    granted_at: datetime
    expires_at: datetime
    shadow_runs: int
    shadow_agreed: int
    policy_version: str
    signature: str

    def body(self) -> dict:
        return {"key": self.key.as_list(), "tier": self.tier.value, "approved_by": self.approved_by,
                "granted_at": self.granted_at, "expires_at": self.expires_at, "shadow_runs": self.shadow_runs,
                "shadow_agreed": self.shadow_agreed, "policy_version": self.policy_version}


def shannon_entropy(p: Sequence[float]) -> float:
    ps = [finite("p", x) for x in p]
    if not ps or any(x < 0 for x in ps) or abs(sum(ps) - 1.0) > 1e-6:
        raise SovereignError("entropy needs a probability distribution (non-negative, sums to 1)")
    return float(-sum(x * math.log(x) for x in ps if x > 0))


class EntropyMonitor:
    """dS/dt по окну; SAFE_MODE защёлкивается до подписанного сброса оператором."""

    def __init__(self, policy: AutonomyPolicy, *, operator_key: bytes) -> None:
        self.policy = policy
        self._key = operator_key
        self._samples: deque = deque()
        self.latched: Optional[dict] = None

    def observe(self, distribution: Sequence[float], *, at: datetime) -> SafetySignal:
        at = aware("at", at)
        self._samples.append((at, shannon_entropy(distribution)))
        horizon = at - timedelta(seconds=self.policy.entropy_window_s)
        while self._samples and self._samples[0][0] < horizon:
            self._samples.popleft()
        rate = self.rate()
        if rate is not None and rate > self.policy.entropy_crit_rate and self.latched is None:
            self.latched = {"at": at.isoformat(), "rate": rate, "s_crit": self.policy.entropy_crit_rate}
        return self.signal()

    def rate(self) -> Optional[float]:
        if len(self._samples) < 2:
            return None
        t0 = self._samples[0][0]
        ts = [(t - t0).total_seconds() for t, _ in self._samples]
        ss = [s for _, s in self._samples]
        mt, ms = sum(ts) / len(ts), sum(ss) / len(ss)
        den = sum((t - mt) ** 2 for t in ts)
        if den <= 0:
            return None
        return sum((t - mt) * (s - ms) for t, s in zip(ts, ss)) / den

    def signal(self) -> SafetySignal:
        if self.latched is not None:
            return SafetySignal.SAFE_MODE
        r = self.rate()
        if r is not None and r > self.policy.entropy_warn_rate:
            return SafetySignal.HUMAN_REVIEW
        return SafetySignal.NORMAL

    def operator_reset(self, *, operator_id: str, signature: str) -> None:
        if self.latched is None:
            return
        body = {"reset_safe_mode": self.latched, "operator": text("operator_id", operator_id)}
        if not verify_artifact(body, signature, self._key):
            raise SovereignError("SAFE_MODE reset refused: operator signature invalid")
        self.latched = None
        self._samples.clear()

    def reset_body(self, operator_id: str) -> dict:
        return {"reset_safe_mode": self.latched, "operator": operator_id}


class AutonomyGate:
    """Выдачи по точным ключам + теневой журнал. Эффективный уровень = min(запрошенный, выданный, понижения)."""

    def __init__(self, policy: AutonomyPolicy, *, approver_keys: Mapping[str, bytes]) -> None:
        if not approver_keys:
            raise SovereignError("autonomy grants need at least one registered approver key")
        self.policy = policy
        self._approvers = dict(approver_keys)
        self._grants: dict[tuple, AutonomyGrant] = {}
        self._shadow: dict[tuple, list[bool]] = {}

    @staticmethod
    def _k(key: AutonomyKey) -> tuple:
        return tuple(key.as_list())

    def record_shadow(self, key: AutonomyKey, *, expected_effect_held: bool) -> None:
        """Теневой прогон: действие НЕ исполнялось; сверка ожидаемого эффекта с тем, что наблюдалось
        (например, оператор сделал то же самое, или телеметрия в окне подтвердила прогноз)."""
        self._shadow.setdefault(self._k(key), []).append(bool(expected_effect_held))

    def shadow_stats(self, key: AutonomyKey) -> tuple[int, int]:
        runs = self._shadow.get(self._k(key), [])
        return len(runs), sum(runs)

    def grant(self, key: AutonomyKey, tier: AutonomyTier, *, approved_by: str, granted_at: datetime,
              expires_at: datetime, signature: str) -> AutonomyGrant:
        tier = AutonomyTier(tier)
        n, agreed = self.shadow_stats(key)
        g = AutonomyGrant(key, tier, text("approved_by", approved_by), aware("granted_at", granted_at),
                          aware("expires_at", expires_at), n, agreed, self.policy.policy_version, signature)
        k = self._approvers.get(g.approved_by)
        if k is None or not verify_artifact(g.body(), signature, k):
            raise SovereignError("autonomy grant refused: approver unknown or signature invalid")
        if g.expires_at <= g.granted_at:
            raise SovereignError("autonomy grant must expire after it is granted")
        if tier is not AutonomyTier.SHADOW:
            if n < self.policy.min_shadow_runs:
                raise SovereignError(f"autonomy grant refused: {n} shadow runs < required {self.policy.min_shadow_runs}")
            if agreed / n < self.policy.min_shadow_agreement:
                raise SovereignError(f"autonomy grant refused: shadow agreement {agreed / n:.3f} "
                                     f"< required {self.policy.min_shadow_agreement}")
        self._grants[self._k(key)] = g
        return g

    def grant_body(self, key: AutonomyKey, tier: AutonomyTier, *, approved_by: str, granted_at: datetime,
                   expires_at: datetime) -> dict:
        n, agreed = self.shadow_stats(key)
        return AutonomyGrant(key, AutonomyTier(tier), approved_by, aware("granted_at", granted_at),
                             aware("expires_at", expires_at), n, agreed, self.policy.policy_version, "").body()

    def revoke(self, key: AutonomyKey) -> None:
        self._grants.pop(self._k(key), None)

    def revoke_all_active(self) -> int:
        """SAFE_MODE: отзыв всех выдач выше SHADOW (выдачи не восстанавливаются сами)."""
        ks = [k for k, g in self._grants.items() if g.tier is not AutonomyTier.SHADOW]
        for k in ks:
            del self._grants[k]
        return len(ks)

    def effective_tier(self, key: AutonomyKey, *, requested: AutonomyTier, now: datetime,
                       world_state_age_s: Optional[float], signal: SafetySignal,
                       cap: Optional[AutonomyTier] = None) -> tuple[AutonomyTier, tuple[str, ...]]:
        now = aware("now", now)
        reasons = []
        g = self._grants.get(self._k(key))
        if g is None:
            granted = AutonomyTier.SHADOW
            reasons.append("no_grant_for_exact_key")
        elif g.expires_at <= now:
            granted = AutonomyTier.SHADOW
            reasons.append("grant_expired")
        else:
            granted = g.tier
        r = min(AutonomyTier(requested).rank, granted.rank)
        if world_state_age_s is None or world_state_age_s > self.policy.max_world_state_age_s:
            r -= 1
            reasons.append("world_state_stale:tier_decreased")
        if signal is SafetySignal.SAFE_MODE:
            r = 0
            reasons.append("safe_mode:autonomy_revoked")
        elif signal is SafetySignal.HUMAN_REVIEW and r > AutonomyTier.HUMAN_APPROVED.rank:
            r = AutonomyTier.HUMAN_APPROVED.rank
            reasons.append("entropy_rising:human_review")
        if cap is not None and r > cap.rank:
            r = cap.rank
            reasons.append(f"sovereign_cap:{cap.value}")
        return AutonomyTier.of_rank(r), tuple(reasons)


def sign_body(body: dict, key: bytes) -> str:
    return sign_artifact(body, key)


__all__ = ["AutonomyGate", "AutonomyGrant", "AutonomyKey", "AutonomyPolicy", "AutonomyTier", "EntropyMonitor",
           "F_ENTROPY", "SafetySignal", "shannon_entropy", "sign_body"]
