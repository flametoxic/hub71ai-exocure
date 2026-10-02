"""Trust ladder: 0 shadow → 1 advises → 2 proposes → 3 acts alone.

Rules (policy `trust`):
  • up slowly: after N approvals in a row the system OFFERS the person to allow the action without approval;
    only the person grants it. Actions outside autonomy_allowed_actions are never promoted;
  • down instantly: any refusal resets the streak and removes the permission; storm / degradation — everyone to shadow.
A grant changes the person's authority for THIS executor and the CURE contract is re-issued:
  DelegationContract requires authority ⊆ principal (incl. approval_required), so the person cannot be bypassed.
"""
from __future__ import annotations

from .core import Store

SHADOW, ADVISE, PROPOSE, AUTONOMOUS = 0, 1, 2, 3


class TrustLedger:
    def __init__(self, policy: Store):
        self.n = int(policy.get("trust.promote_after_approvals"))
        self.allowed = set(policy.get("trust.autonomy_allowed_actions"))
        self.names = list(policy.get("trust.level_names"))
        self.streak: dict[str, int] = {}          # "agent:action" → approvals in a row
        self.granted: dict[str, list[str]] = {}   # agent → actions allowed without approval
        self.history: list[dict] = []

    @staticmethod
    def _k(agent: str, action: str) -> str:
        return f"{agent}:{action}"

    def record(self, agent: str, action: str, *, approved: bool, at: str) -> dict:
        k = self._k(agent, action)
        if approved:
            self.streak[k] = self.streak.get(k, 0) + 1
        else:
            self.streak[k] = 0
            if action in self.granted.get(agent, []):
                self.granted[agent].remove(action)        # down instantly
        self.history.append({"at": at, "agent": agent, "action": action, "approved": approved})
        n = self.streak[k]
        if action in self.granted.get(agent, []):
            offer = None
        elif action not in self.allowed:
            offer = {"eligible": False, "reason": "policy: this action always requires your approval"}
        elif n >= self.n:
            offer = {"eligible": True, "text": f"Approvals in a row: {n}. Allow doing this alone?"}
        else:
            offer = {"eligible": False, "reason": f"approvals in a row: {n} of {self.n}"}
        return {"agent": agent, "action": action, "streak": n, "offer": offer}

    def grant(self, agent: str, action: str, *, at: str) -> None:
        if action not in self.allowed:
            raise PermissionError("policy never allows this action without approval")
        if self.streak.get(self._k(agent, action), 0) < self.n:
            raise PermissionError(f"needs {self.n} approvals in a row before autonomy can be granted")
        self.granted.setdefault(agent, [])
        if action not in self.granted[agent]:
            self.granted[agent].append(action)
        self.history.append({"at": at, "agent": agent, "action": action, "granted": True})

    def revoke(self, agent: str, action: str, *, at: str) -> None:
        if action in self.granted.get(agent, []):
            self.granted[agent].remove(action)
        self.streak[self._k(agent, action)] = 0
        self.history.append({"at": at, "agent": agent, "action": action, "granted": False})

    def autonomy(self) -> dict[str, set[str]]:
        return {a: set(x) for a, x in self.granted.items() if x}

    def levels(self, card: dict, *, degraded: bool = False) -> list[dict]:
        """Level for each action."""
        out = []
        for action in card["can"]:
            if degraded:
                lvl = SHADOW
            elif action in card["needs_approval"]:
                lvl = PROPOSE
            elif action in self.granted.get(card["id"], []):
                lvl = AUTONOMOUS
            else:
                lvl = ADVISE
            out.append({"action": action, "level": lvl, "name": self.names[lvl],
                        "streak": self.streak.get(self._k(card["id"], action), 0)})
        return out

    def to_state(self) -> dict:
        return {"streak": self.streak, "granted": self.granted, "history": self.history}

    def load_state(self, st: dict) -> None:
        self.streak, self.granted, self.history = dict(st["streak"]), {k: list(v) for k, v in st["granted"].items()}, list(st["history"])
