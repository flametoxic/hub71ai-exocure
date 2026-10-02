"""The resident's personal contour inside the CURE core. Not bound to a device: phone, laptop, home hub are only windows.

The contour holds: the personal model (weights, setpoint, history), memory (every fact with source, time, status),
executor decisions and effects, the trust ladder, logs of "what left the contour" and "who asked", settings, conversation.
The sealed health domain is kept apart: only derived constraints ever leave it, and nothing from it ever reaches
the language model (city_world.consent.llm_view). Per-domain pseudonyms come from TokenVault (person.Door).
Storage: one encrypted file per resident in the core store (Fernet; key ARRIVAL_CORE_KEY).
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock, RLock

from cryptography.fernet import Fernet, InvalidToken

from .core import Store
from .person import PersonalModel
from .trust import TrustLedger

KEY_ENV, DIR_ENV = "ARRIVAL_CORE_KEY", "ARRIVAL_CORE_DIR"
PRIVATE_PROFILE_KEYS = {"device_only", "voice_example", "voice_example_ru"}


def _sanitized_profile_doc(profile_doc: dict) -> dict:
    doc = copy.deepcopy(profile_doc)
    for key in PRIVATE_PROFILE_KEYS:
        doc.pop(key, None)
    return doc


@dataclass
class Contour:
    resident_id: str
    profile_doc: dict                       # profile without the sealed domain
    model: PersonalModel
    trust: TrustLedger
    sealed: dict = field(default_factory=dict)          # sealed health domain
    constraints: dict = field(default_factory=dict)     # derived physical constraints per person
    facts: list = field(default_factory=list)           # memory: {at, kind, text, source, status}
    outbox: list = field(default_factory=list)          # what left the contour: {at, to, what}
    access_log: list = field(default_factory=list)      # who asked: {at, purpose, role, fields, allowed}
    effects: list = field(default_factory=list)         # what the executor did
    pending: dict = field(default_factory=dict)         # executor proposals waiting for a yes
    devices: list = field(default_factory=list)
    revoked: list = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    conversation: list = field(default_factory=list)
    overrides: dict = field(default_factory=dict)       # confirmed fact cards → plan
    extra: dict = field(default_factory=dict)           # scenario state: delays, guest, budget rules

    def profile(self) -> Store:
        """Profile for the engines: physical constraints yes, diagnosis no."""
        doc = _sanitized_profile_doc(self.profile_doc)
        doc["constraints"] = {who: {k: {"value": v, "decided_by": "user"} for k, v in c.items()}
                              for who, c in self.constraints.items()}
        doc.pop("device_only", None)
        return Store(doc, "resident")

    def remember(self, *, at: datetime, kind: str, text: str, source: str, status: str = "active") -> dict:
        fact = {"at": at.isoformat(), "kind": kind, "text": text, "source": source, "status": status}
        self.facts.append(fact)
        return fact

    def say(self, role: str, text: str, *, at: datetime, trace_ids: list | None = None) -> None:
        self.conversation.append({"at": at.isoformat(), "role": role, "text": text, "trace_ids": trace_ids or []})
        self.conversation = self.conversation[-60:]


class ContourStore:
    """Personal contour store in the core. Open contours live in process memory; on disk there is only ciphertext."""

    def __init__(self, *, key: bytes, directory: Path, policy: Store):
        if len(key) < 16:
            raise ValueError("core key must be ≥ 16 bytes")
        self._f = Fernet(base64.urlsafe_b64encode(hashlib.sha256(key).digest()))
        self.dir, self.policy = directory, policy
        self.dir.mkdir(parents=True, exist_ok=True)
        self.open_contours: dict[str, Contour] = {}
        self._resident_locks: dict[str, RLock] = {}
        self._resident_locks_guard = Lock()

    def resident_lock(self, rid: str) -> RLock:
        """Return the process-local transaction lock for one resident contour."""
        with self._resident_locks_guard:
            return self._resident_locks.setdefault(rid, RLock())

    @classmethod
    def from_env(cls, policy: Store) -> "ContourStore":
        key = (os.environ.get(KEY_ENV) or os.environ.get("ARRIVAL_DEVICE_KEY") or "").encode()
        d = Path(os.environ.get(DIR_ENV, Path(__file__).parents[1] / "core_store"))
        return cls(key=key, directory=d, policy=policy)

    def _path(self, rid: str) -> Path:
        return self.dir / f"{hashlib.sha256(rid.encode()).hexdigest()[:24]}.bin"

    def create(self, rid: str, *, profile_doc: dict, sealed: dict, constraints: dict, now: datetime) -> Contour:
        with self.resident_lock(rid):
            doc = _sanitized_profile_doc(profile_doc)
            c = Contour(resident_id=rid, profile_doc=doc, model=PersonalModel(self.policy), trust=TrustLedger(self.policy),
                        sealed=copy.deepcopy(sealed), constraints=copy.deepcopy(constraints),
                        settings={"lang": "en", "quiet_hours": list(self.policy.get("assistant.quiet_hours")),
                                  "max_questions_per_day": int(self.policy.get("assistant.max_questions_per_day"))})
            c.remember(at=now, kind="contour", text="contour created", source="CURE")
            self.open_contours[rid] = c
            self.save(c)
            return c

    def exists(self, rid: str) -> bool:
        with self.resident_lock(rid):
            return rid in self.open_contours or self._path(rid).exists()

    def open(self, rid: str, *, device: str | None = None) -> Contour | None:
        with self.resident_lock(rid):
            c = self.open_contours.get(rid)
            if c is None and self._path(rid).exists():
                c = self._load(rid)
                self.open_contours[rid] = c
            if c is not None and device and device not in c.devices:
                c.devices.append(device)
            return c

    def forget_process_memory(self) -> None:
        """Like a service restart: nothing in memory, ciphertext on disk."""
        self.open_contours.clear()

    def save(self, c: Contour) -> None:
        with self.resident_lock(c.resident_id):
            c.profile_doc = _sanitized_profile_doc(c.profile_doc)
            st = {"resident_id": c.resident_id, "profile_doc": c.profile_doc, "model": c.model.to_state(),
                  "trust": c.trust.to_state(), "sealed": c.sealed, "constraints": c.constraints, "facts": c.facts,
                  "outbox": c.outbox, "access_log": c.access_log, "effects": c.effects, "pending": c.pending,
                  "devices": c.devices, "revoked": c.revoked, "settings": c.settings, "conversation": c.conversation,
                  "overrides": c.overrides, "extra": c.extra}
            destination = self._path(c.resident_id)
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_bytes(self._f.encrypt(json.dumps(st, ensure_ascii=False, default=str).encode()))
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)

    def _load(self, rid: str) -> Contour:
        try:
            st = json.loads(self._f.decrypt(self._path(rid).read_bytes()))
        except InvalidToken:
            raise ValueError("contour cannot be decrypted with this core key")
        c = Contour(resident_id=st["resident_id"], profile_doc=_sanitized_profile_doc(st["profile_doc"]), model=PersonalModel(self.policy),
                    trust=TrustLedger(self.policy))
        c.model.load_state(st["model"]); c.trust.load_state(st["trust"])
        for k in ("sealed", "constraints", "facts", "outbox", "access_log", "effects", "pending", "devices", "revoked",
                  "settings", "conversation", "overrides", "extra"):
            setattr(c, k, st.get(k, getattr(c, k)))
        return c

    def delete(self, rid: str) -> bool:
        with self.resident_lock(rid):
            self.open_contours.pop(rid, None)
            p = self._path(rid)
            if p.exists():
                p.unlink()
                return True
            return False

    def ciphertext_preview(self, rid: str) -> str | None:
        with self.resident_lock(rid):
            p = self._path(rid)
            return p.read_bytes()[:48].decode() + "…" if p.exists() else None
