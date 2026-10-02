"""Context API integrated into the main CURE process."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from arrival import api as core_api
from arrival import service as sv

from . import city, life

router = APIRouter(prefix="/addon", tags=["cure-context"])


class RidIn(BaseModel):
    rid: str = sv.LEILA
    lang: Literal["en", "ru"] | None = None


def _session(rid: str, lang: str | None = None) -> sv.Session:
    session = core_api._s(rid)
    if rid == sv.LEILA:
        with core_api.rt().store.resident_lock(rid):
            life.seed(session)
    return session


def _leila_session(rid: str, lang: str | None = None) -> sv.Session:
    if rid != sv.LEILA:
        raise HTTPException(400, "This endpoint is available for the registered Leila profile only")
    session = _session(rid, lang)
    if session.c.extra.get("registered_profile") != "leila-v1":
        raise HTTPException(400, "This endpoint is available for the registered Leila profile only")
    return session


@contextmanager
def _display_language(session: sv.Session, lang: str | None):
    previous = session.c.settings.get("lang", "en")
    if lang:
        session.c.settings["lang"] = lang
    try:
        yield
    finally:
        session.c.settings["lang"] = previous


@router.post("/seed")
def seed(x: RidIn):
    session = _leila_session(x.rid, x.lang)
    with core_api.rt().store.resident_lock(x.rid), _display_language(session, x.lang):
        result = life.seed(session)
        return {**result, "notes": session.c.extra.get("notes", [])}


@router.post("/infer")
def infer(x: RidIn):
    session = _leila_session(x.rid, x.lang)
    with core_api.rt().store.resident_lock(x.rid), _display_language(session, x.lang):
        return {"found": life.infer(session), "pending": session.pending_view()}


@router.get("/life_map")
def life_map(rid: str = sv.LEILA, lang: Literal["en", "ru"] | None = None):
    session = _leila_session(rid, lang)
    with core_api.rt().store.resident_lock(rid), _display_language(session, lang):
        return life.life_map(session)


@router.get("/chronicle")
def chronicle(rid: str = sv.LEILA, lang: Literal["en", "ru"] | None = None):
    session = _leila_session(rid, lang)
    with core_api.rt().store.resident_lock(rid), _display_language(session, lang):
        return {"items": life.chronicle(session)}


@router.post("/cascade")
def cascade(x: RidIn):
    session = _leila_session(x.rid, x.lang)
    with core_api.rt().store.resident_lock(x.rid), _display_language(session, x.lang):
        return life.cascade(session)


@router.get("/city_gets")
def city_gets(rid: str = sv.LEILA, lang: Literal["en", "ru"] | None = None):
    session = _leila_session(rid, lang)
    with core_api.rt().store.resident_lock(rid), _display_language(session, lang):
        return life.city_gets(session)


@router.get("/city_profile")
def city_profile(lang: Literal["en", "ru"] = "en"):
    return city.profile(lang)
