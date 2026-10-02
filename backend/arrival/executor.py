"""The CURE executor is the only one that acts. Agents only search and draft.

Every action = a CURE contract (planning.delegation: rights ⊆ the person's rights) + city_world.u_allowed
(physically safe · policy valid · consent · authority) + the person's yes
or a permission from the trust ladder. Money, signatures and sending messages for the person are never in the rights.
In the demo no external system is called: the effect is written to the contour log; a calendar event is a real VEVENT (.ics).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from api_gateway.core.city_world.formulas import u_allowed
from api_gateway.core.planning.delegation import AuthorityBoundary, DelegationContract, DelegationDecision

from .core import Store

EXECUTOR_ID = "cure"


def build_executor(*, profile: Store, now: datetime, deadline: datetime, autonomy: set[str], trace_id: str) -> dict:
    g = profile.raw("grants")
    acts, scopes = frozenset(g["cure_actions"]), frozenset(g["scopes"])
    required = frozenset(g["cure_approval_required"]) - frozenset(autonomy)      # only the person grants permissions
    auth = AuthorityBoundary(allowed_actions=acts, allowed_scopes=scopes, approval_required_actions=required & acts)
    contract = DelegationContract.issue(
        principal_authority=auth, delegation_id=f"{profile.get('pseudonym')}:{EXECUTOR_ID}:{uuid4().hex[:6]}",
        principal=profile.get("pseudonym"), agent=EXECUTOR_ID, role="reality_engine",
        subgoal="execution through the safety envelope", scope=scopes, authority=auth, issued_at=now,
        deadline=deadline, required_evidence=("engine_trace",), escalation_path=("resident",),
        completion_verification="resident", trace_id=trace_id)
    return {"id": EXECUTOR_ID, "title": "CURE Reality Engine", "contract": contract,
            "template": {"approval_required": list(g["cure_approval_required"]),
                         "forbidden_display": list(g["never_display"]), "design_report": "CURE core executor"}}


def card(ex: dict) -> dict:
    c = ex["contract"]
    return {"id": ex["id"], "title": ex["title"], "can": sorted(c.authority.allowed_actions),
            "needs_approval": sorted(c.authority.approval_required_actions),
            "cannot": ex["template"]["forbidden_display"], "scopes": sorted(c.scope)}


def propose(ex: dict, *, action: str, scope: str, facts: dict, title: str, when: str | None = None,
            now: datetime, physical_safe: bool = True, consent_valid: bool = True, trace_ids: list | None = None) -> dict:
    r = ex["contract"].authorize(action=action, scope=scope, at=now)
    gate = u_allowed(physical_safe=physical_safe, policy_valid=action in ex["contract"].authority.allowed_actions,
                     consent_valid=consent_valid, authorized=r.decision != DelegationDecision.ESCALATE)
    status = ("blocked" if not gate["allowed"] else
              "needs_you" if r.decision == DelegationDecision.REQUIRES_APPROVAL else "ready")
    return {"id": uuid4().hex[:10], "by": EXECUTOR_ID, "action": action, "scope": scope, "title": title,
            "facts": facts, "when": when, "decision": r.decision.value, "reason": r.reason,
            "gate": {"allowed": gate["allowed"], "failed": gate["failed"]}, "status": status,
            "autonomous": status == "ready" and action in ex["template"]["approval_required"],
            "trace_ids": list(trace_ids or [])}


def to_ics(facts: dict, *, uid: str) -> str:
    start = datetime.fromisoformat(facts["start"])
    end = start + timedelta(minutes=int(facts.get("duration_min", 60)))
    f = lambda d: d.strftime("%Y%m%dT%H%M%S")
    return "\r\n".join(["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//CURE//resident//EN", "BEGIN:VEVENT",
                        f"UID:{uid}@cure", f"DTSTAMP:{f(datetime.now(timezone.utc))}Z", f"DTSTART:{f(start)}",
                        f"DTEND:{f(end)}", f"SUMMARY:{facts.get('title', 'CURE')}", "END:VEVENT", "END:VCALENDAR"])


def execute(item: dict, *, approved: bool, effects: list, now: datetime) -> dict:
    if item["status"] == "blocked":
        return {**item, "result": "blocked"}
    if item["status"] == "needs_you" and not approved:
        return {**item, "result": "declined"}
    eff = {"at": now.isoformat(), "item": item["id"], "action": item["action"], "title": item["title"],
           "facts": item["facts"], "external_call": False,
           "note": "demo: written to the contour log, no external system called"}
    if item["action"] == "calendar_event" and "start" in item["facts"]:
        eff["ics"] = to_ics(item["facts"], uid=item["id"])
    effects.append(eff)
    return {**item, "result": "done", "effect": eff}
