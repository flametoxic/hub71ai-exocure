"""The door to CURE: voice → profile, pseudonym, consents, personal model.

CURE: city_world.consent.TokenVault (per-domain pseudonyms), ConsentLedger (purpose, role, field minimisation).
Health fragments are cut from the phrase BEFORE any language model call and are kept only in the sealed domain.

The personal model is a small Bayesian preference model (not a neural net):
  λ = (money, time, comfort) — posterior over a grid from the policy;
      choice of option j: P(j | λ) = softmax_j( −λ·x_j / T ),  x — normalised option costs;
  AC setpoint — a normal model with a conjugate update per observation.
"""
from __future__ import annotations

import os
import re
import uuid
from functools import lru_cache
from pathlib import Path
from datetime import datetime

import numpy as np

from api_gateway.core.city_world.consent import Consent, ConsentLedger, TokenVault

from .core import MissingData, Store, answer, refused

VAULT_ENV = "ARRIVAL_VAULT_KEY"


# ---------- voice: health never leaves ----------
def split_voice(text: str, policy: Store) -> dict:
    """Split the phrase; parts with health terms from the policy go to the sealed domain and are NEVER passed on."""
    terms = [re.compile(t, re.IGNORECASE) for t in policy.get("privacy.health_terms")]
    parts = [p.strip() for p in re.split(r"[,.;](?!\d)", text) if p.strip()]
    local = [p for p in parts if any(t.search(p) for t in terms)]
    shared = [p for p in parts if p not in local]
    return {"text_for_llm": ", ".join(shared), "device_only": local, "removed_count": len(local)}


PRIVATE_LLM_KEYS = {"sealed", "diagnosis", "health_record", "health_raw_text", "condition", "device_only",
                    "secrets", "api_key", "openai_api_key", "password", "token", "voice_example", "voice_example_ru",
                    "heard", "conversation", "outbox", "access_log"}
SECRET_TEXT = re.compile(r"sk-[A-Za-z0-9_-]+|(?i:bearer)\s+[A-Za-z0-9._-]+")


@lru_cache(maxsize=1)
def _privacy_policy() -> Store:
    return Store.load(Path(__file__).parent / "demo" / "policy.yaml")


def sanitize_for_llm(value, policy: Store | None = None):
    """Extend the existing voice boundary to every nested outbound payload.

    Raw conversation/health and secrets are never context. Derived numeric
    constraints are preserved. Health-bearing strings are removed conservatively.
    """
    policy = policy or _privacy_policy()
    if isinstance(value, dict):
        if str(value.get("source", "")).casefold() == "sealed" or value.get("kind") == "sealed":
            return {}
        return {str(k): sanitize_for_llm(v, policy) for k, v in value.items()
                if str(k).casefold() not in PRIVATE_LLM_KEYS
                and not any(part in str(k).casefold() for part in ("api_key", "secret", "password", "diagnos"))}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_llm(v, policy) for v in value]
    if isinstance(value, str):
        if split_voice(value, policy)["device_only"]:
            return "[private note omitted]"
        return SECRET_TEXT.sub("[secret omitted]", value)
    return value


# ---------- pseudonym and consents ----------
class Door:
    def __init__(self, *, profile: Store, policy: Store, now: datetime):
        key = os.environ.get(VAULT_ENV, "").encode()
        self.vault = TokenVault(key=key, agent_token_ttl_s=float(policy.get("vault.agent_token_ttl_s")),
                                allowed_links={("person_agent_token", "consent_subject_id"): ["arrival_twin"]})
        self.ids = self.vault.enroll(profile.get("resident_local_id"))
        self.agent_token = self.vault.agent_token(profile.get("resident_local_id"), now=now)
        self.subject = self.ids["consent_subject_id"]
        self.ledger = ConsentLedger(purposes=policy.raw("consent_purposes"))
        self._profile_consents = {c["purpose_id"]: dict(c) for c in profile.raw("consents")}
        self._policy_version = policy.meta.get("version", "unknown")
        self._legal_basis = policy.get("privacy.legal_basis")
        self._jurisdiction = policy.get("privacy.jurisdiction")
        pv = self._policy_version
        for c in profile.raw("consents"):
            self.ledger.grant(Consent(consent_id=f"c-{uuid.uuid4().hex[:8]}", subject_id=self.subject,
                                      purpose_id=c["purpose_id"], data_category=c["data_category"],
                                      legal_basis=policy.get("privacy.legal_basis"), scope=tuple(c["scope"]),
                                      granted_at=now, expires_at=None, revoked_at=None, policy_version=pv,
                                      jurisdiction=policy.get("privacy.jurisdiction")))

        self.log: list[dict] = []        # every request to the data: who, why, which fields, result
        self.outbox: list[dict] = []     # everything that left: where and what exactly

    def check(self, *, purpose: str, category: str, fields: list[str], role: str, action: str, now: datetime) -> dict:
        r = self.ledger.allow(subject_id=self.subject, purpose_id=purpose, data_category=category, fields=fields,
                              role=role, action=action, scope="arrival", at=now)
        self.log.append({"at": now.isoformat(), "purpose": purpose, "role": role, "fields": list(fields),
                         "action": action, "allowed": r["allowed"],
                         "failed": [k for k, v in r["checks"].items() if not v]})
        return r

    def sent(self, *, to: str, what, now: datetime) -> None:
        self.outbox.append({"at": now.isoformat(), "to": to, "what": what})

    def revoke(self, purpose: str, *, now: datetime) -> list[str]:
        ids = [c.consent_id for c in self.ledger.consents.values() if c.purpose_id == purpose and c.revoked_at is None]
        for cid in ids:
            self.ledger.revoke(cid, at=now)
        return ids

    def grant(self, purpose: str, *, now: datetime) -> str:
        template = self._profile_consents.get(purpose)
        if template is None:
            raise ValueError(f"unknown consent purpose: {purpose}")
        active = [c for c in self.ledger.consents.values() if c.purpose_id == purpose and c.revoked_at is None]
        if active:
            return active[0].consent_id
        consent_id = f"c-{uuid.uuid4().hex[:8]}"
        self.ledger.grant(Consent(consent_id=consent_id, subject_id=self.subject, purpose_id=purpose,
                                  data_category=template["data_category"], legal_basis=self._legal_basis,
                                  scope=tuple(template["scope"]), granted_at=now, expires_at=None, revoked_at=None,
                                  policy_version=self._policy_version, jurisdiction=self._jurisdiction))
        return consent_id

    def consents_view(self, now: datetime) -> list[dict]:
        return [{"purpose": c.purpose_id, "category": c.data_category, "scope": list(c.scope),
                 "status": "revoked" if c.revoked_at is not None and c.revoked_at <= now else "active"}
                for c in self.ledger.consents.values()]

    def require(self, **kw) -> None:
        r = self.check(**kw)
        if not r["allowed"]:
            raise PermissionError(f"consent check failed: {[k for k, v in r['checks'].items() if not v]}")


# ---------- personal model ----------
class PersonalModel:
    def __init__(self, policy: Store):
        self.grid = np.asarray(policy.get("personal_params.lambda_grid"), float)
        self.names = list(policy.get("personal_params.lambda_names"))
        self.T = float(policy.get("personal_params.choice_temperature"))
        self.log_post = np.zeros(len(self.grid))                     # uniform prior over the grid
        self.sp_mean = float(policy.get("personal_params.setpoint_prior.mean"))
        self.sp_sd = float(policy.get("personal_params.setpoint_prior.sd"))
        self.obs_sd = float(policy.get("personal_params.setpoint_obs_sd"))
        self.z = float(policy.get("sync.sigma_from_interval_z"))
        self.n_choices = self.n_setpoint = 0
        self.history: list[dict] = []     # what it learned and when — shown on the Memory screen

    def to_state(self) -> dict:
        return {"log_post": self.log_post.tolist(), "sp_mean": self.sp_mean, "sp_sd": self.sp_sd,
                "n_choices": self.n_choices, "n_setpoint": self.n_setpoint, "history": self.history}

    def load_state(self, st: dict) -> None:
        if len(st["log_post"]) != len(self.grid):
            raise ValueError("stored model does not match policy λ grid (policy changed): refusing to load")
        self.log_post = np.asarray(st["log_post"], float)
        self.sp_mean, self.sp_sd = float(st["sp_mean"]), float(st["sp_sd"])
        self.n_choices, self.n_setpoint, self.history = int(st["n_choices"]), int(st["n_setpoint"]), list(st["history"])

    def posterior(self) -> np.ndarray:
        p = np.exp(self.log_post - self.log_post.max())
        return p / p.sum()

    def observe_choice(self, options: dict[str, list[float]], chosen: str, at: str = "") -> None:
        """options: option → costs along self.names (lower is better)."""
        keys = list(options)
        X = np.asarray([options[k] for k in keys], float)
        lo, hi = X.min(0), X.max(0)
        X = (X - lo) / np.where(hi > lo, hi - lo, 1.0)
        u = -(self.grid @ X.T) / self.T                               # G × options
        logp = u - np.log(np.exp(u - u.max(1, keepdims=True)).sum(1, keepdims=True)) - u.max(1, keepdims=True)
        self.log_post += logp[:, keys.index(chosen)]
        self.n_choices += 1
        self.history.append({"at": at, "kind": "choice", "text": f"chose {chosen} of {', '.join(keys)}"})

    def observe_setpoint(self, x: float, at: str = "") -> None:
        prec = 1 / self.sp_sd ** 2 + 1 / self.obs_sd ** 2
        self.sp_mean = (self.sp_mean / self.sp_sd ** 2 + x / self.obs_sd ** 2) / prec
        self.sp_sd = prec ** -0.5
        self.n_setpoint += 1
        self.history.append({"at": at, "kind": "setpoint", "text": f"set the AC to {x:g} °C"})

    def lambda_mean(self) -> dict:
        return dict(zip(self.names, (self.posterior() @ self.grid).tolist()))

    def summary(self) -> dict:
        p = self.posterior()
        best = int(np.argmax(p))
        return {"lambda": {"mean": self.lambda_mean(), "most_likely": dict(zip(self.names, self.grid[best].tolist())),
                           "p_most_likely": float(p[best]), "observations": self.n_choices,
                           "status": "learned" if self.n_choices else "prior"},
                "setpoint_c": {"mid": self.sp_mean, "low": self.sp_mean - self.z * self.sp_sd,
                               "high": self.sp_mean + self.z * self.sp_sd, "observations": self.n_setpoint,
                               "status": "learned" if self.n_setpoint else "prior"}}


def open_door(*, profile: Store, policy: Store, now: datetime, voice: str | None = None, parsed: dict | None = None,
              model: PersonalModel) -> tuple[Door | None, dict]:
    """The whole door scene. parsed — a parsed profile (or None: the demo profile is used)."""
    try:
        door = Door(profile=profile, policy=policy, now=now)
        text = voice if voice is not None else profile.get("voice_example")
        split = split_voice(text, policy)
        door.require(purpose="arrival_planning", category="household", fields=["household", "flags"],
                     role="arrival_twin", action="compute", now=now)
        door.sent(to="openai" if parsed else "openai (not called: no key)", what=split["text_for_llm"], now=now)
        # what happens if someone asks for more — a ConsentLedger check, not our own if
        denied = door.check(purpose="arrival_vision", category="constraints", fields=["condition"],
                            role="arrival_twin", action="read", now=now)
        panel = {
            "pseudonyms": {k: v[:8] + "…" for k, v in door.ids.items()},
            "agent_token_expires_at": door.agent_token["expires_at"],
            "voice": {"sent_to_openai": split["text_for_llm"], "kept_on_device": len(split["device_only"]),
                      "note": "health fragments removed before the language model"},
            "profile": parsed or {"household": profile.raw("household"), "flags": profile.raw("flags"),
                                  "source": "demo profile (language model not called)"},
            "device_only_keys": sorted(f"{who}.{k}" for who, d in profile.raw("device_only").items() for k in d),
            "consents": door.consents_view(now),
            "denied_example": {"request": "arrival_vision: field condition", **denied},
            "model": model.summary()}
        return door, answer(panel, unit="panel", stores=[profile, policy],
                            formulas=["CITY TokenVault", "CITY ConsentLedger.allow", "Bayes λ grid", "Normal conjugate"],
                            label="Personal model")
    except MissingData as e:
        return None, refused(e, what="door")
