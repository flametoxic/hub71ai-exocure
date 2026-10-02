"""City WM §6: люди, профили и согласия; §13 guardrails.

Раздельные домены идентичности: resident_id (изолированная реальная личность), person_agent_token
(короткоживущий псевдоним для модели мира), consent_subject_id, access_subject_id, health_subject_id.
Связь между ними — только через TokenVault по политике. Медицинский контур — отдельно; сырые медицинские
данные, HRV, давление и экстренные данные не попадают в общую аналитику и в LLM.
    allow(data, action) = consent_valid ∧ purpose_allowed ∧ role_authorized ∧ data_minimized
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping, Optional, Sequence

from ..world_model.reality_formulas import FormulaReference
from .schema import DOC, CitySchemaError

F_ALLOW = FormulaReference("CITY-6.3-ALLOW", DOC, 10,
                           "allow(data, action) = consent_valid ∧ purpose_allowed ∧ role_authorized ∧ data_minimized")
DOMAINS = ("resident_id", "person_agent_token", "consent_subject_id", "access_subject_id", "health_subject_id")
SENSITIVE_CATEGORIES = ("health", "biometric", "emergency_medical")


class TokenVault:
    """Связи между доменами идентичности хранит только хранилище; раскрытие — по разрешённой паре доменов,
    цели и роли, с журналом. Псевдоним агента короткоживущий."""

    def __init__(self, *, key: bytes, allowed_links: Mapping[tuple, Sequence[str]], agent_token_ttl_s: float) -> None:
        if len(key) < 16 or agent_token_ttl_s <= 0:
            raise CitySchemaError("vault key ≥ 16 bytes and a positive agent token TTL (policy) are required")
        self._key, self._allowed = key, {k: set(v) for k, v in allowed_links.items()}
        self._ttl = timedelta(seconds=float(agent_token_ttl_s))
        self._links: dict[str, dict] = {}
        self.access_log: list[dict] = []

    def enroll(self, resident_id: str) -> dict:
        ids = {"resident_id": resident_id}
        for d in DOMAINS[2:]:
            ids[d] = hmac.new(self._key, f"{d}:{resident_id}".encode(), hashlib.sha256).hexdigest()[:24]
        self._links[resident_id] = ids
        return {k: v for k, v in ids.items() if k != "resident_id"}

    def agent_token(self, resident_id: str, *, now: datetime) -> dict:
        tok = f"pat:{secrets.token_hex(8)}"
        self._links[resident_id].setdefault("_agent_tokens", {})[tok] = now + self._ttl
        return {"person_agent_token": tok, "expires_at": (now + self._ttl).isoformat()}

    def resolve(self, from_domain: str, value: str, to_domain: str, *, purpose: str, role: str, now: datetime) -> Optional[str]:
        roles = self._allowed.get((from_domain, to_domain), set())
        entry = {"from": from_domain, "to": to_domain, "purpose": purpose, "role": role, "at": now.isoformat()}
        if role not in roles:
            self.access_log.append({**entry, "result": "denied"})
            return None
        for rid, ids in self._links.items():
            if from_domain == "person_agent_token":
                exp = ids.get("_agent_tokens", {}).get(value)
                hit = exp is not None and exp > now
            else:
                hit = ids.get(from_domain) == value
            if hit:
                self.access_log.append({**entry, "result": "granted"})
                return ids.get(to_domain)
        self.access_log.append({**entry, "result": "not_found_or_expired"})
        return None


@dataclass(frozen=True)
class Consent:
    consent_id: str
    subject_id: str                    # consent_subject_id
    purpose_id: str                    # indoor_fall_detection, emergency_dispatch, access_control, personalized_climate…
    data_category: str
    legal_basis: str
    scope: tuple
    granted_at: datetime
    expires_at: Optional[datetime]
    revoked_at: Optional[datetime]
    policy_version: str
    jurisdiction: str


class ConsentLedger:
    def __init__(self, *, purposes: Mapping[str, Mapping[str, object]]) -> None:
        """purposes: purpose_id → {allowed_categories, allowed_roles, allowed_fields (минимизация), actions}"""
        self.purposes = {k: dict(v) for k, v in purposes.items()}
        self.consents: dict[str, Consent] = {}

    def grant(self, c: Consent) -> None:
        if c.purpose_id not in self.purposes:
            raise CitySchemaError(f"purpose {c.purpose_id!r} is not defined by policy")
        self.consents[c.consent_id] = c

    def revoke(self, consent_id: str, *, at: datetime) -> None:
        c = self.consents[consent_id]
        self.consents[consent_id] = Consent(**{**c.__dict__, "revoked_at": at})

    def allow(self, *, subject_id: str, purpose_id: str, data_category: str, fields: Sequence[str], role: str,
              action: str, scope: str, at: datetime) -> dict:
        valid = [c for c in self.consents.values() if c.subject_id == subject_id and c.purpose_id == purpose_id
                 and c.data_category == data_category and scope in c.scope and c.granted_at <= at
                 and (c.expires_at is None or at < c.expires_at) and (c.revoked_at is None or at < c.revoked_at)]
        p = self.purposes.get(purpose_id, {})
        checks = {"consent_valid": bool(valid),
                  "purpose_allowed": data_category in p.get("allowed_categories", ()) and action in p.get("actions", ()),
                  "role_authorized": role in p.get("allowed_roles", ()),
                  "data_minimized": set(fields) <= set(p.get("allowed_fields", ()))}
        return {"allowed": all(checks.values()), "checks": checks, "consent_ids": [c.consent_id for c in valid],
                "formula": F_ALLOW}


def llm_view(record: Mapping[str, object], *, purpose_authorized: bool, allowed_fields: Sequence[str]) -> dict:
    """Приватность для LLM: только минимизированные, агрегированные, отфильтрованные факты; чувствительные категории
    и сырые идентификаторы не передаются никогда."""
    if not purpose_authorized:
        raise CitySchemaError("send_to_LLM requires an authorized purpose")
    if record.get("data_category") in SENSITIVE_CATEGORIES:
        raise CitySchemaError("sensitive health/biometric contour is never sent to an LLM")
    out = {k: v for k, v in record.items() if k in allowed_fields and not any(d in k for d in DOMAINS)}
    return {"masked": True, "fields": out}


__all__ = ["Consent", "ConsentLedger", "DOMAINS", "F_ALLOW", "SENSITIVE_CATEGORIES", "TokenVault", "llm_view"]
