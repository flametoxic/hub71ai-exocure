"""CURE HTTP API for the resident's windows (phone, home hub) and the stage console. Prefix /twin.

Windows store nothing: everything lives in the personal contour in the CURE core (contour.py). A device is only a window:
opening the contour from another device = the same resident, the same memory.

Environment:
  ARRIVAL_CORE_KEY (≥16 bytes), ARRIVAL_CORE_DIR — contour store in the core (ciphertext)
  ARRIVAL_VAULT_KEY (≥16 bytes) — pseudonyms (TokenVault)
  ARRIVAL_TEMPLATE_KEY — agent template signing (unsigned templates are never loaded)
  CURE_LANGUAGE_MODE=local — deterministic replies and resident agents, no API key required.
  CURE_LANGUAGE_MODE=openai + OPENAI_API_KEY + OPENAI_MODEL — OpenAI narration and bounded
      resident specialists; backend calculations, validation, approval, and execution stay in CURE.
"""
from __future__ import annotations

from datetime import date
import re
from threading import Lock
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import orchestrator as orc
from . import service as sv
from .core import ANSWERS
from .language import GatewayConfigurationError, GatewayError, NarrationGenerationError
from .modeling import ModelRegistry
from .profiles import ProfileService
from .mind import Mind
from . import dynamic_agents

router = APIRouter(prefix="/twin", tags=["cure"])
_RT: sv.Runtime | None = None
_CHAT_CREATE_LOCK = Lock()
_EXTERNAL_RID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def rt() -> sv.Runtime:
    global _RT
    if _RT is None:
        _RT = sv.Runtime()
    return _RT


def _registered_leila(runtime: sv.Runtime, *, device: str | None = None, lang: str = "en") -> sv.Session:
    existing = runtime.session(sv.LEILA, device=device)
    if existing is not None and existing.c.extra.get("registered_profile") == "leila-v1":
        return existing
    if existing is not None:
        runtime.store.delete(sv.LEILA)
        runtime.sessions.pop(sv.LEILA, None)
    voice = runtime.S["leila"].get("voice_example_ru" if lang == "ru" else "voice_example")
    return sv.leila_contour(runtime, voice, lang, device)


def _s(rid: str, device: str | None = None) -> sv.Session:
    runtime = rt()
    if rid == sv.LEILA:
        with _CHAT_CREATE_LOCK:
            return _registered_leila(runtime, device=device)
    s = runtime.session(rid, device=device)
    if s is None:
        raise HTTPException(404, "no contour for this resident — open the door first")
    return s


class DoorIn(BaseModel):
    voice: str | None = None
    lang: str | None = None
    newcomer: bool = False
    device: str = "phone"


class AskIn(BaseModel):
    rid: str = sv.LEILA
    text: str
    device: str = "phone"


class ActIn(BaseModel):
    rid: str = sv.LEILA
    action: str
    args: dict = {}
    device: str = "phone"


class RidIn(BaseModel):
    rid: str = sv.LEILA
    device: str = "phone"


class FindingDecisionIn(BaseModel):
    finding_id: str
    decision: Literal['approve', 'reject']


class AgentReviewIn(RidIn):
    decisions: list[FindingDecisionIn]


def _session_for_ask(x: AskIn) -> tuple[sv.Session, dict | None]:
    runtime = rt()
    session = runtime.session(x.rid, device=x.device)
    if session is not None and (
        x.rid != sv.LEILA or session.c.extra.get("registered_profile") == "leila-v1"
    ):
        return session, None
    if not _EXTERNAL_RID.fullmatch(x.rid):
        raise HTTPException(400, "rid must contain 1-64 ASCII letters, digits, underscores, or hyphens")
    with _CHAT_CREATE_LOCK:
        runtime = rt()
        session = runtime.session(x.rid, device=x.device)
        if x.rid == sv.LEILA:
            lang = sv.lang_of(x.text)
            return _registered_leila(runtime, device=x.device, lang=lang), None
        if session is not None:
            return session, None
        lang = sv.lang_of(x.text)
        session = sv.newcomer_contour(runtime, x.text, lang, x.device, resident_id=x.rid)
        session.c.say("you", x.text, at=session.now)
        card = orc.profile_intake_card(session)
        next_question = card.get("next_question")
        session.c.extra["profile_pending"] = (next_question or {}).get("variable_id")
        if next_question:
            suffix = " Можно написать «пропустить»." if lang == "ru" else " You can also say 'skip'."
            message = ({"ru": "Профиль создан. ", "en": "Profile created. "}[lang]
                       + next_question["text"] + suffix)
        else:
            message = {"ru": "Профиль создан.", "en": "Profile created."}[lang]
        reply = session.say(message, cards=[card], route="profile_intake")
        reply["meta"] = {"profile_complete": card["complete"]}
        return session, reply


# actions windows may call from buttons (allow-list of session methods)
ACTIONS = {"districts", "district_why", "district_switch", "whatif_month", "whatif_money", "choose", "apartment", "plan",
           "agent_search", "decide", "grant_trust", "revoke_trust", "visa_delay", "visa_why", "visa_counterfactual",
           "visa_levers", "report_fact", "car_taxi", "license", "what_i_know", "revoke_consent", "vision_request",
           "vision_chain", "vision_storm", "vision_neighbors", "vision_cohort", "vision_learning", "add_to_city",
           "newcomer_day", "delete", "setpoint", "guest_constraint", "set_arrival", "m_mon_home", "impact", "m_mon_morning",
           "grant_consent"}


@router.post("/reset")
def reset():
    rt().reset()
    return {"ok": True, "clock": rt().clock.isoformat()}


@router.post("/door")
def door(x: DoorIn):
    r = rt()
    voice = x.voice or r.S["leila"].get("voice_example_ru" if x.lang == "ru" else "voice_example")
    lang = x.lang or sv.lang_of(voice)
    if x.newcomer:
        s = sv.newcomer_contour(r, voice, lang, x.device)
    else:
        if r.store.exists(sv.LEILA):
            r.store.delete(sv.LEILA)
            r.sessions.pop(sv.LEILA, None)
        s = sv.leila_contour(r, voice, lang, x.device)
    out = s.open_door(voice)
    if x.newcomer:
        card = orc.profile_intake_card(s)
        s.c.extra["profile_pending"] = (card.get("next_question") or {}).get("variable_id")
        s.save()
        out["cards"].append(card)
        out["meta"] = {"language_source": "local", "profile_complete": card["complete"]}
    try:
        out = orc.narrate_action(s, "door", out)
    except NarrationGenerationError as error:
        raise HTTPException(502, str(error)) from error
    except (GatewayConfigurationError, GatewayError) as error:
        raise HTTPException(503, str(error)) from error
    return {**out, "rid": s.rid}


@router.post("/bootstrap")
def bootstrap(x: DoorIn):
    """For rehearsal and mid-show start: Leila's door, arrival date, district A — in one call."""
    r = rt()
    d = door(DoorIn(lang=x.lang or "en", device=x.device))
    s = _s(sv.LEILA)
    s.set_arrival(date.fromisoformat(r.P.get("scenario.move_in")))
    s.choose("A")
    return {"rid": sv.LEILA, "door": d, "state": state(sv.LEILA)}


@router.post("/ask")
def ask(x: AskIn):
    try:
        session, initial_reply = _session_for_ask(x)
        with rt().store.resident_lock(session.rid):
            reply = (orc.narrate_action(session, "profile_intake", initial_reply)
                     if initial_reply is not None else orc.ask(session, x.text))
        return {**reply, "rid": session.rid}
    except NarrationGenerationError as error:
        raise HTTPException(502, str(error)) from error
    except (GatewayConfigurationError, GatewayError) as error:
        raise HTTPException(503, str(error)) from error


@router.post("/act")
def act(x: ActIn):
    if x.action not in ACTIONS:
        raise HTTPException(400, f"unknown action: {x.action}")
    s = _s(x.rid, x.device)
    args = dict(x.args)
    if "lang" in args:
        s.c.settings["lang"] = args.pop("lang")
    if x.action == "set_arrival":
        args["d"] = date.fromisoformat(args.pop("date"))
    try:
        with rt().store.resident_lock(s.rid):
            reply = getattr(s, x.action)(**args)
            return orc.narrate_action(s, x.action, reply)
    except NarrationGenerationError as error:
        raise HTTPException(502, str(error)) from error
    except (GatewayConfigurationError, GatewayError) as error:
        raise HTTPException(503, str(error)) from error


@router.get("/moments")
def moments(lang: str = "en"):
    return {"moments": rt().moments(lang), "current": rt().moment_i, "clock": rt().clock.isoformat()}


@router.post("/moment/next")
def moment_next(x: RidIn):
    r = rt()
    if r.moment_i + 1 >= len(sv.MOMENTS):
        return {"done": True, "moments": r.moments()}
    s = _s(x.rid, x.device)
    action = sv.MOMENTS[r.moment_i + 1][0]
    try:
        return orc.narrate_action(s, action, s.run_moment(action))
    except NarrationGenerationError as error:
        raise HTTPException(502, str(error)) from error
    except (GatewayConfigurationError, GatewayError) as error:
        raise HTTPException(503, str(error)) from error


@router.post("/moment/{mid}")
def moment(mid: str, x: RidIn):
    if mid not in [m[0] for m in sv.MOMENTS]:
        raise HTTPException(404, "moment")
    s = _s(x.rid, x.device)
    try:
        return orc.narrate_action(s, mid, s.run_moment(mid))
    except NarrationGenerationError as error:
        raise HTTPException(502, str(error)) from error
    except (GatewayConfigurationError, GatewayError) as error:
        raise HTTPException(503, str(error)) from error


@router.post("/device/open")
def device_open(x: RidIn):
    """Open the contour from another device. The core service "restarts": memory is empty, the contour is read from disk."""
    r = rt()
    r.restart_process()
    s = _s(x.rid, x.device)
    name = s.name(next(p["id"] for p in s.profile().raw("household") if p["role"] == "adult"))
    out = s.say(s.r("welcome_back", name=name), cards=[{"type": "device", "devices": s.c.devices,
                                                          "on_disk_looks_like": r.store.ciphertext_preview(x.rid)}],
                route="welcome_back")
    try:
        return orc.narrate_action(s, "device_open", out)
    except NarrationGenerationError as error:
        raise HTTPException(502, str(error)) from error
    except (GatewayConfigurationError, GatewayError) as error:
        raise HTTPException(503, str(error)) from error


@router.get("/state")
def state(rid: str = sv.LEILA):
    r = rt()
    s = _s(rid) if rid == sv.LEILA else r.session(rid)
    base = {"clock": r.clock.isoformat(), "moment": r.moment_i, "moments": r.moments(), "residents": list(r.sessions),
            "city": {"extra_families": len(r.city["extra_families"]), "stress": r.city["stress"],
                     "ramp_cars": r.city["ramp_cars"]}}
    if s is None:
        return {**base, "resident": None}
    return {**base, "resident": rid, "lang": s.lang, "today": today(s), "pending": s.pending_view(), "drafts": s.c.extra.get("drafts", [])[-8:],
            "conversation": s.c.conversation[-30:], "model": s.c.model.summary(), "trust": s.trust_view(),
            "trust_policy": {"autonomy_allowed": list(s.P.get("trust.autonomy_allowed_actions")),
                             "promote_after": int(s.P.get("trust.promote_after_approvals"))},
            "effects": s.c.effects[-10:], "last": s.c.extra.get("last", {}), "rev": s.c.extra.get("rev", 0), "district": s.c.extra.get("district"),
            "arrival": s.c.extra.get("arrival"), "devices": s.c.devices,
            "household": [{"id": p["id"], "role": p["role"], "name": s.name(p["id"])} for p in s.profile().raw("household")],
            "constraints": {s.name(k) if k != "guest" else k: v for k, v in s.c.constraints.items()},
            "profile": ProfileService(ModelRegistry.load_default()).snapshot(s)}


def today(s: sv.Session) -> dict:
    """The "now" context for the home hub: today's PM10 by hour (synthetic week forecast), outdoor windows."""
    day = s.now.date().isoformat()
    f = next((d for d in s.S["week"].raw("forecast.days") if d["date"] == day), None)
    if f is None:
        return {"date": day, "forecast": None}
    w = s._windows()["value"]
    return {"date": day, "hour": s.now.hour, "pm10_by_hour": f["pm10_ugm3"], "temperature_by_hour": f["temperature_c"],
            "pm10_now": f["pm10_ugm3"][s.now.hour], "temperature_now": f["temperature_c"][s.now.hour],
            "general_limit": float(s.P.get("outdoor_limits.pm10_ugm3_max")), "limits_pm10": w["limits_pm10"],
            "windows": {s.name(pid): next((x["windows"] for x in v if x["date"] == day), []) for pid, v in w["people"].items()},
            "family": next((x["windows"] for x in w["family"] if x["date"] == day), []), "data_mode": "synthetic"}


@router.get("/memory")
def memory(rid: str = sv.LEILA):
    return _s(rid).memory()


class MindIn(BaseModel):
    rid: str = sv.LEILA
    query: str = "Current family schedule and previous decisions"
    kind: str = "routine"
    embeddings: bool = True


@router.get('/mind')
def mind(rid: str = sv.LEILA):
    with rt().store.resident_lock(rid):
        return Mind(_s(rid)).snapshot()


@router.post('/mind/recall')
def mind_recall(x: MindIn):
    with rt().store.resident_lock(x.rid):
        return Mind(_s(x.rid)).recall(x.query, embeddings=x.embeddings)


@router.post('/mind/infer')
def mind_infer(x: MindIn):
    if x.kind not in {'routine', 'availability', 'preference', 'current_situation'}:
        raise HTTPException(400, 'unknown inference kind')
    try:
        with rt().store.resident_lock(x.rid):
            return Mind(_s(x.rid)).infer(x.kind, x.query)
    except GatewayError as error:
        raise HTTPException(503, str(error)) from error


@router.post('/agents/dynamic')
def dynamic_start(x: AskIn):
    try:
        with rt().store.resident_lock(x.rid):
            run = dynamic_agents.start(_s(x.rid), x.text)
            return {'agent_id': run['agent_id'], 'status': run['status']}
    except GatewayError as error:
        raise HTTPException(503, str(error)) from error


@router.get('/agents/dynamic')
def dynamic_list(rid: str = sv.LEILA):
    # The worker releases resident locks while waiting for OpenAI.
    with rt().store.resident_lock(rid):
        return {'agents': list(_s(rid).c.extra.get('dynamic_agents', {}).values())}


@router.get('/agents/dynamic/{agent_id}')
def dynamic_get(agent_id: str, rid: str = sv.LEILA):
    with rt().store.resident_lock(rid):
        run = _s(rid).c.extra.get('dynamic_agents', {}).get(agent_id)
        if run is None:
            raise HTTPException(404, 'unknown agent')
        return run


@router.post('/agents/dynamic/{agent_id}/archive')
def dynamic_archive(agent_id: str, x: RidIn):
    with rt().store.resident_lock(x.rid):
        s = _s(x.rid)
        run = s.c.extra.get('dynamic_agents', {}).get(agent_id)
        if run is None or run['status'] not in {'COMPLETED', 'WAITING_USER', 'FAILED'}:
            raise HTTPException(409, 'only a terminal agent can be archived')
        dynamic_agents._transition(s, run, 'ARCHIVED')
        return run


@router.post('/agents/dynamic/{agent_id}/review')
def dynamic_review(agent_id: str, x: AgentReviewIn):
    with rt().store.resident_lock(x.rid):
        try:
            return dynamic_agents.review(_s(x.rid), agent_id, [decision.model_dump() for decision in x.decisions])
        except ValueError as error:
            raise HTTPException(409, str(error)) from error


@router.get("/why/{trace_id}")
def why(trace_id: str):
    a = ANSWERS.get(trace_id)
    if a is None:
        raise HTTPException(404, "trace")
    return {k: a.get(k) for k in ("sources", "formulas", "data_mode", "missing", "calculation_status", "label", "interval")}
