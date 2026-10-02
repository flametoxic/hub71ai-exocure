"""Resident agents: signed templates → CURE delegation contracts → reactions to events.

CURE: planning.delegation.AuthorityBoundary / DelegationContract (issue, authorize).
A template without a valid signature is never loaded. Agent authority = template ∩ what the person allowed.
Execution is selected separately: deterministic in local mode, OpenAI specialists in OpenAI mode.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timedelta, timezone

from api_gateway.core.planning.delegation import AuthorityBoundary, DelegationContract, DelegationDecision

from .core import Store, answer

SIGN_ENV = "ARRIVAL_TEMPLATE_KEY"


def _body(t: dict) -> bytes:
    return json.dumps({k: v for k, v in t.items() if k != "signature"}, sort_keys=True, ensure_ascii=False).encode()


def sign(t: dict, key: bytes) -> str:
    return hmac.new(key, _body(t), hashlib.sha256).hexdigest()


def load_templates(store: Store) -> list[dict]:
    key = os.environ.get(SIGN_ENV, "").encode()
    if not key:
        raise RuntimeError(f"{SIGN_ENV} is not set: unsigned templates are never loaded")
    ok = [t for t in store.raw("templates") if hmac.compare_digest(t.get("signature", ""), sign(t, key))]
    return ok


def build_agents(*, profile: Store, templates: list[dict], goal: dict, now: datetime,
                 autonomy: dict[str, set[str]] | None = None) -> list[dict]:
    """Agents get only the search_public and draft_text tools. Agents never get autonomy —
    only the CURE executor does (executor.py). The autonomy parameter is kept for compatibility and ignored."""
    g = profile.raw("grants")
    principal = AuthorityBoundary(allowed_actions=frozenset(g["agent_tools"]), allowed_scopes=frozenset(g["scopes"]))
    deadline = datetime.fromisoformat(goal["target"]).replace(tzinfo=timezone.utc)
    agents = []
    for t in templates:
        if goal["class"] not in t["goal_classes"]:
            continue
        if not all(profile.raw("flags").get(k) == v for k, v in t.get("applies_if", {}).items()):
            continue
        acts = frozenset(t["allowed_actions"]) & principal.allowed_actions
        scopes = frozenset(t["scopes"]) & principal.allowed_scopes
        if not acts or not scopes:
            continue
        authority = AuthorityBoundary(allowed_actions=acts, allowed_scopes=scopes)
        contract = DelegationContract.issue(
            principal_authority=principal, delegation_id=f"{profile.get('pseudonym')}:{t['id']}",
            principal=profile.get("pseudonym"), agent=t["id"], role="arrival_agent", subgoal=t["title"],
            scope=scopes, authority=authority, issued_at=now, deadline=deadline,
            required_evidence=tuple(t["required_evidence"]), escalation_path=tuple(t["escalation_path"]),
            completion_verification=t["completion_verification"], trace_id=goal["trace_id"])
        agents.append({"id": t["id"], "title": t["title"], "contract": contract, "template": t})
    return agents


def card(a: dict) -> dict:
    c = a["contract"]
    return {"id": a["id"], "title": a["title"], "can": sorted(c.authority.allowed_actions),
            "needs_approval": sorted(c.authority.approval_required_actions), "cannot": a["template"]["forbidden_display"],
            "deadline": c.deadline.date().isoformat(), "scopes": sorted(c.scope),
            "design_report": a["template"].get("design_report")}


def try_action(a: dict, *, action: str, scope: str, now: datetime) -> dict:
    r = a["contract"].authorize(action=action, scope=scope, at=now)
    return {"agent": a["id"], "action": action, "decision": r.decision.value, "reason": r.reason,
            "escalate_to": list(r.escalate_to)}


def on_event(agents: list[dict], *, event: str, rates: Store | None = None, now: datetime,
             plan_before: dict | None = None, plan_after: dict | None = None, payload: dict | None = None,
             mode: str = "active") -> list[dict]:
    """Agent reactions to an event are drafts only (draft_text). drafts.py builds the text from CURE facts.
    mode="shadow" (storm): the agent prepares nothing and is only marked as shadowed."""
    out = []
    for a in agents:
        for rule in a["template"].get("reactions", []):
            if rule["event"] != event:
                continue
            decision = try_action(a, action=rule["action"], scope=rule["scope"], now=now)
            if decision["decision"] != DelegationDecision.ALLOWED.value:
                continue
            out.append({"agent": a["id"], "tool": rule["action"], "draft": rule.get("draft"), "mode": mode,
                        "decision": decision["decision"], "payload": dict(payload or {}),
                        "confirm_required": False, "autonomous": False})
    return out


def agents_answer(agents: list[dict], stores: list[Store]) -> dict:
    return answer([card(a) for a in agents], unit="agents", stores=stores,
                  formulas=["CCIA delegation: authority ⊆ principal"], label="Resident agents")


def utcnow() -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=1)
