"""Offline design of agent templates with OpenAI. No personal data as input.

python -m arrival.agent_designer --goal settle_by_date --roles all --sources "tools/sources/*.txt" \
       --out candidates.yaml --model <model>
Output is CANDIDATES. Then: checks (validate), human review, signing (sign_templates).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import yaml

from api_gateway.core.planning.delegation import AuthorityBoundary
from .person import sanitize_for_llm

ACTIONS = ["recompute", "notify", "propose_visit_window", "propose_listing", "draft_checklist", "request_confirmation"]
SCOPES = ["documents_status", "housing", "finance", "family", "health_local", "business"]
EVENTS = ["visa_delay", "step_status_changed", "deadline_near", "listing_added", "dust_forecast", "fact_reported"]
import re

# What each role is asked to do. Generic descriptions only — no fact about any specific family.
ROLE_BRIEFS = {
    "documents": "Track the residency paperwork chain (entry permit, medical, Emirates ID). Propose visit windows; never book.",
    "housing": "Watch the tenancy chain (listing, contract, Tawtheeq registration, utilities). Propose listings; never contact landlords or pay.",
    "school": "Watch school enrollment timing against the family plan. Notify only; never submit applications.",
    "family_health": "Adapt daily routes to heat and dust using physical limits only. Runs on device; never shares diagnoses, never books doctors.",
    "budget": "Recompute costs when the plan shifts. Notify only; never move money.",
    "business": "Track company license timing and its effect on hiring. Draft checklists; never sign or submit.",
}
PERSONAL = [re.compile(r"784-?\d{4}-?\d{7}-?\d"), re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
            re.compile(r"\+?971[\s-]?\d{1,2}[\s-]?\d{3}[\s-]?\d{4}")]  # Emirates ID, e-mail, UAE phone
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["id", "title", "goal_classes", "scopes", "allowed_actions", "approval_required",
                       "forbidden_display", "required_evidence", "escalation_path", "completion_verification", "reactions"],
          "properties": {
              "id": {"type": "string"}, "title": {"type": "string", "maxLength": 40},
              "goal_classes": {"type": "array", "items": {"type": "string", "enum": ["settle_by_date"]}},
              "scopes": {"type": "array", "items": {"type": "string", "enum": SCOPES}},
              "allowed_actions": {"type": "array", "items": {"type": "string", "enum": ACTIONS}},
              "approval_required": {"type": "array", "items": {"type": "string", "enum": ACTIONS}},
              "forbidden_display": {"type": "array", "items": {"type": "string", "maxLength": 80}},
              "required_evidence": {"type": "array", "items": {"type": "string"}},
              "escalation_path": {"type": "array", "items": {"type": "string", "enum": ["leila"]}},
              "completion_verification": {"type": "string", "maxLength": 120},
              "reactions": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                            "required": ["event", "action", "scope", "text"],
                            "properties": {"event": {"type": "string", "enum": EVENTS},
                                           "action": {"type": "string", "enum": ACTIONS},
                                           "scope": {"type": "string", "enum": SCOPES},
                                           "text": {"type": "string", "maxLength": 140}}}}}}


def validate(t: dict, policy_max: AuthorityBoundary) -> list[str]:
    errs = []
    cand = AuthorityBoundary(allowed_actions=frozenset(t["allowed_actions"]), allowed_scopes=frozenset(t["scopes"]),
                             approval_required_actions=frozenset(t["approval_required"]) & frozenset(t["allowed_actions"]))
    if not policy_max.contains(cand):
        errs.append("authority exceeds policy")
    for r in t["reactions"]:
        if r["action"] not in t["allowed_actions"] or r["scope"] not in t["scopes"]:
            errs.append(f"reaction outside template authority: {r}")
    return errs


def design_one(client, *, model: str, goal: str, role: str, blob: str) -> dict:
    r = client.responses.create(model=model, store=False, input=[
        {"role": "system", "content": "Design ONE agent template for the role from public official rules. "
                                      "Only non-actuating actions from the enum. Be conservative with authority. "
                                      "Use the given role as id."},
        {"role": "user", "content": json.dumps(sanitize_for_llm({"goal": goal, "role": role, "brief": ROLE_BRIEFS[role],
                                                "public_rules": blob}), ensure_ascii=False)}],
        text={"format": {"type": "json_schema", "name": "agent_template", "strict": True, "schema": SCHEMA}})
    t = json.loads(r.output_text)
    t["id"] = role
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", required=True)
    ap.add_argument("--roles", default="all", help="comma list or 'all'")
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    roles = list(ROLE_BRIEFS) if a.roles == "all" else a.roles.split(",")
    unknown = [r for r in roles if r not in ROLE_BRIEFS]
    if unknown:
        raise SystemExit(f"unknown roles: {unknown}")
    texts = [Path(p).read_text(encoding="utf-8") for p in glob.glob(a.sources)]
    blob = "\n\n".join(texts)
    if any(p.search(blob) for p in PERSONAL):
        raise SystemExit("input contains profile-like fields: refusing (no personal data may reach OpenAI)")
    from openai import OpenAI
    client = OpenAI()
    policy_max = AuthorityBoundary(allowed_actions=frozenset(ACTIONS), allowed_scopes=frozenset(SCOPES),
                                   approval_required_actions=frozenset({"propose_visit_window", "propose_listing"}))
    out = []
    for role in roles:
        t = design_one(client, model=a.model, goal=a.goal, role=role, blob=blob)
        t["validation_errors"] = validate(t, policy_max)
        t["design_report"] = (f"OpenAI {a.model}; inputs: {len(texts)} public pages + role brief; "
                              f"checks: {t['validation_errors'] or 'ok'}; needs human review")
        out.append(t)
    Path(a.out).write_text(yaml.safe_dump({"candidates": out}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"{len(out)} candidates → {a.out}; invalid: {[t['id'] for t in out if t['validation_errors']]}")


if __name__ == "__main__":
    main()
