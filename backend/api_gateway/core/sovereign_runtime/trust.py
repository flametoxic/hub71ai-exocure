"""Криптографическое доверие и журнал (§7, §8): подпись доказывает целостность и происхождение записи,
а не физическую истинность показания (мастер-ТЗ, правило 7).

  • идентичность устройства: у каждого устройства свой ключ; подписанные сообщения (HMAC-SHA256);
  • accept(m) = 1[sig valid ∧ nonce fresh ∧ seq > seq_last ∧ time within window] (§7.1);
  • журнал исполнения — только добавление, цепочка дайджестов и корень Меркла (§7.1, §8); если долговечная
    запись невозможна — среда переходит в деградацию и физическая запись блокируется (§9).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

from .contracts import SovereignError, aware, canonical, digest, text


@dataclass(frozen=True)
class SignedMessage:
    device_id: str
    seq: int
    nonce: str
    sent_at: datetime
    payload: dict
    signature: str = ""

    def body(self) -> bytes:
        return canonical({"device_id": self.device_id, "seq": self.seq, "nonce": self.nonce,
                          "sent_at": self.sent_at, "payload": self.payload}).encode("utf-8")


class DeviceTrust:
    """Реестр ключей устройств и проверка сообщений с защитой от повторов."""

    def __init__(self, *, max_clock_skew_s: float, nonce_ttl_s: float) -> None:
        if max_clock_skew_s <= 0 or nonce_ttl_s <= 0:
            raise SovereignError("clock skew window and nonce TTL must be positive (policy)")
        self.max_skew = timedelta(seconds=float(max_clock_skew_s))
        self.nonce_ttl = timedelta(seconds=float(nonce_ttl_s))
        self._keys: dict[str, bytes] = {}
        self._seq: dict[str, int] = {}
        self._nonces: dict[str, datetime] = {}

    def register(self, device_id: str, key: bytes) -> None:
        if not isinstance(key, bytes) or len(key) < 16:
            raise SovereignError("device key must be at least 16 bytes (secure element where available)")
        self._keys[text("device_id", device_id)] = key

    def sign(self, msg: SignedMessage, key: bytes) -> SignedMessage:
        return SignedMessage(msg.device_id, msg.seq, msg.nonce, msg.sent_at, dict(msg.payload),
                             hmac.new(key, msg.body(), hashlib.sha256).hexdigest())

    def accept(self, msg: SignedMessage, *, now: datetime) -> tuple[bool, tuple[str, ...]]:
        now = aware("now", now)
        reasons = []
        key = self._keys.get(msg.device_id)
        sig_ok = key is not None and bool(msg.signature) and hmac.compare_digest(
            hmac.new(key, msg.body(), hashlib.sha256).hexdigest(), msg.signature)
        if not sig_ok:
            reasons.append("signature_invalid_or_unknown_device")
        sent = aware("sent_at", msg.sent_at)
        if abs(now - sent) > self.max_skew:
            reasons.append("time_outside_window")
        for n, t in list(self._nonces.items()):
            if now - t > self.nonce_ttl:
                del self._nonces[n]
        nonce_key = f"{msg.device_id}:{msg.nonce}"
        if nonce_key in self._nonces:
            reasons.append("nonce_replayed")
        if msg.seq <= self._seq.get(msg.device_id, -1):
            reasons.append("sequence_not_increasing")
        if reasons:
            return False, tuple(reasons)
        self._nonces[nonce_key] = now
        self._seq[msg.device_id] = msg.seq
        return True, ()


def merkle_root(leaves: list[str]) -> str:
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = [bytes.fromhex(x) for x in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return level[0].hex()


class ExecutionJournal:
    """Неизменяемый журнал исполнения: цепочка дайджестов + корень Меркла; опционально долговечный JSONL.

    Любая ошибка долговечной записи делает журнал «недоступным»: runtime переходит в деградацию (§9)."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._entries: list[dict] = []
        self.available = True
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        self._entries.append(json.loads(line))
            if not self.verify():
                raise SovereignError("execution journal is tampered: refusing to start")

    def append(self, kind: str, payload: dict, *, at: datetime) -> dict:
        prev = self._entries[-1]["entry_digest"] if self._entries else ""
        entry = {"seq": len(self._entries), "kind": text("kind", kind), "at": aware("at", at).isoformat(),
                 "payload": json.loads(canonical(payload)), "previous": prev}
        entry["entry_digest"] = digest(entry)
        if self.path:
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            except OSError as exc:
                self.available = False
                raise SovereignError(f"durable journal unavailable: {exc}") from exc
        self._entries.append(entry)
        return dict(entry)

    def verify(self) -> bool:
        prev = ""
        for i, e in enumerate(self._entries):
            body = {k: v for k, v in e.items() if k != "entry_digest"}
            if e.get("seq") != i or body.get("previous") != prev or digest(body) != e.get("entry_digest"):
                return False
            prev = e["entry_digest"]
        return True

    def merkle_root(self) -> str:
        return merkle_root([e["entry_digest"] for e in self._entries])

    def entries(self, kind: Optional[str] = None) -> tuple[dict, ...]:
        return tuple(dict(e) for e in self._entries if kind is None or e["kind"] == kind)

    def trace(self, action_id: str) -> tuple[dict, ...]:
        """§8: по одному следу восстанавливается вся цепочка решения."""
        return tuple(dict(e) for e in self._entries if e["payload"].get("action_id") == action_id)


def sign_artifact(payload: Any, key: bytes) -> str:
    return hmac.new(key, canonical(payload).encode("utf-8"), hashlib.sha256).hexdigest()


def verify_artifact(payload: Any, signature: str, key: bytes) -> bool:
    return bool(signature) and hmac.compare_digest(sign_artifact(payload, key), signature)


__all__ = ["DeviceTrust", "ExecutionJournal", "SignedMessage", "merkle_root", "sign_artifact", "verify_artifact"]
