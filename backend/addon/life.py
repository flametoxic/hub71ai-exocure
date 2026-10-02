"""Leila's fictional life model, provenance, inferences and city boundary views."""
from __future__ import annotations

import statistics
from pathlib import Path

import yaml

from arrival import service as sv
from arrival.vision import city_request

DATA = Path(__file__).parent / "data"
PROVENANCE = {
    "you": {"en": "you said", "ru": "ты сказала"},
    "core": {"en": "CURE inferred", "ru": "ядро вывело"},
    "official": {"en": "official source", "ru": "официальный источник"},
    "assumption": {"en": "assumption", "ru": "допущение"},
    "sealed": {"en": "sealed in your contour", "ru": "запечатано в контуре"},
    "approved": {"en": "CURE inferred, you confirmed", "ru": "ядро вывело, ты подтвердила"},
}
TRIP_NAMES = {
    "school_am": {"en": "School run", "ru": "Утром в школу"},
    "school_pm": {"en": "From school", "ru": "Из школы"},
    "office_am": {"en": "To work", "ru": "На работу"},
    "office_pm": {"en": "From work", "ru": "С работы"},
    "park": {"en": "Park", "ru": "Парк"},
}


def load_life() -> dict:
    return yaml.safe_load((DATA / "leila_life.yaml").read_text(encoding="utf-8"))


def _localized(value: dict, lang: str) -> str:
    return value.get(lang) or value.get("en") or ""


def _keep_note(session: sv.Session, note: dict, *, source: str = "you") -> None:
    notes = session.c.extra.setdefault("notes", [])
    if any(
        existing.get("fact") == note.get("fact")
        and existing.get("category") == note.get("category")
        and existing.get("about") == note.get("about")
        for existing in notes
    ):
        return
    stored = dict(note, source=source)
    notes.append(stored)
    session.c.remember(at=session.now, kind="note", text=stored["fact"], source=source)


def _translated_note(session: sv.Session, text: str) -> str:
    for note in session.c.extra.get("notes", []):
        if note.get("fact") == text and note.get("i18n"):
            return _localized(note["i18n"], session.lang)
    i18n = session.c.extra.get("notes_i18n", {}).get(text)
    return _localized(i18n, session.lang) if i18n else text


def seed(session: sv.Session) -> dict:
    """Seed the already-registered fictional resident without exposing sealed health data."""
    if session.c.extra.get("life_seeded"):
        return {"seeded": False, "people": session.c.extra.get("people", {})}
    life = load_life()
    people = {}
    for person_id, person in life["people"].items():
        people[person_id] = {"details": person.get("details", []), "title": person.get("title")}
        if person.get("name"):
            i18n = {
                "en": f"daughter's name is {person['name']['en']}",
                "ru": f"дочь зовут {person['name']['ru']}",
            }
            _keep_note(session, {
                "fact": i18n[session.lang], "category": "name", "about": person_id,
                "value": _localized(person["name"], session.lang), "i18n": i18n,
            })
            for household_member in session.c.profile_doc.get("household", []):
                if household_member["id"] == person_id:
                    household_member["name"] = dict(person["name"])
        for detail in person.get("details", []):
            if detail["source"] == "you" and detail.get("category") in {"work", "date", "preference", "style", "rhythm"}:
                _keep_note(session, {
                    "fact": _localized(detail, session.lang), "category": detail["category"],
                    "about": person_id, "value": None, "i18n": {"en": detail["en"], "ru": detail["ru"]},
                })
    session.c.extra["people"] = people
    session.c.extra["rhythms"] = life["rhythms"]
    session.c.extra["places"] = life["places"]
    session.c.extra["life_seeded"] = True
    session.save()
    return {"seeded": True, "people": people}


def _format_hour(hour: float) -> str:
    return f"{int(hour):02d}:{int(round((hour % 1) * 60)):02d}"


def infer(session: sv.Session) -> list[dict]:
    """Derive repeatable patterns, but keep every result pending until the resident approves it."""
    logs = load_life()["logs"]
    min_n = int(session.P.get("habits.min_n"))
    max_sd = float(session.P.get("habits.max_sd_h"))
    found = []
    beach = [item["hour"] for item in logs["beach"]]
    if len(beach) >= min_n and statistics.pstdev(beach) <= max_sd:
        at = _format_hour(statistics.median(beach))
        found.append({
            "id": "beach_saturday", "category": "rhythm", "n": len(beach),
            "i18n": {"en": f"on Saturdays you go to the beach around {at}", "ru": f"по субботам вы на пляже около {at}"},
        })
    approvals = logs["approvals"]
    morning = [hour for hour in approvals if 6 <= hour < 10]
    evening = [hour for hour in approvals if 20 <= hour < 23]
    if len(morning) >= min_n and len(evening) >= min_n:
        low = lambda values: _format_hour(int(min(values) * 2) / 2)
        high = lambda values: _format_hour(-(-max(values) * 2 // 1) / 2)
        m0, m1, e0, e1 = low(morning), high(morning), low(evening), high(evening)
        found.append({
            "id": "approval_hours", "category": "style", "n": len(approvals),
            "i18n": {
                "en": f"you answer between {m0}–{m1} and {e0}–{e1}: I'll ask then",
                "ru": f"ты отвечаешь на вопросы {m0}–{m1} и {e0}–{e1}: спрашиваю в это время",
            },
        })
    setpoints = logs["setpoint"]
    baseline = float(session.profile().get("params.setpoint_c"))
    median = statistics.median(setpoints)
    if len(setpoints) >= min_n and abs(median - baseline) >= 0.5:
        found.append({
            "id": "real_setpoint", "category": "preference", "n": len(setpoints),
            "i18n": {
                "en": f"you actually keep the AC at {median:g}°, not {baseline:g}°",
                "ru": f"на самом деле ты держишь {median:g}°, а не {baseline:g}°",
            },
        })
    meetings: dict[str, list[bool]] = {}
    for meeting in logs["meetings"]:
        meetings.setdefault(meeting["kind"], []).append(bool(meeting["moved"]))
    investor, client = meetings.get("investor", []), meetings.get("client", [])
    if len(investor) >= min_n and len(client) >= min_n and all(investor) and not any(client):
        found.append({
            "id": "movable_meetings", "category": "work", "n": len(investor) + len(client),
            "i18n": {
                "en": "you move investor meetings easily; your husband's client meetings never move",
                "ru": "встречи с инвесторами ты легко переносишь, встречи мужа с клиентами — никогда",
            },
        })

    i18n_store = session.c.extra.setdefault("notes_i18n", {})
    proposed = session.c.extra.setdefault("inferences", {})
    kept = {note["fact"] for note in session.c.extra.get("notes", [])}
    for item in found:
        item["fact"] = item["i18n"][session.lang]
        for translated in item["i18n"].values():
            i18n_store[translated] = item["i18n"]
            proposed[translated] = item["id"]
        pending_id = f"infer-{item['id']}"
        if kept & set(item["i18n"].values()) or pending_id in session.c.pending:
            continue
        session.c.pending[pending_id] = {
            "id": pending_id, "kind": "memory", "action": "remember", "title": item["fact"],
            "facts": {
                "fact": item["fact"], "category": item["category"], "about": None,
                "value": None, "inference": item["id"], "observations": item["n"],
            },
            "when": None, "status": "needs_you",
        }
    session.save()
    return found


def accept_memory(session: sv.Session, item: dict, approved: bool) -> dict:
    """Apply or decline a pending inferred memory through the core's normal approval path."""
    title = item["title"]
    session.c.remember(
        at=session.now, kind="decision", text=f"{'yes' if approved else 'no'}: {title}", source="you"
    )
    if approved:
        i18n = session.c.extra.get("notes_i18n", {}).get(title)
        _keep_note(session, dict(item["facts"], i18n=i18n), source="you (approved)")
    return session.say(
        session.r("approved" if approved else "declined", title=title),
        cards=[{"type": "memory", "status": "approved" if approved else "declined", **item["facts"]}],
        route="decide",
    )


def life_map(session: sv.Session) -> dict:
    lang = session.lang
    profile = session.profile()
    nodes, edges = [], []

    def node(node_id, label, kind, provenance, **extra):
        nodes.append({"id": node_id, "label": label, "kind": kind, "provenance": provenance, **extra})

    def edge(source, target, provenance, label=""):
        edges.append({"from": source, "to": target, "provenance": provenance, "label": label})

    household = profile.raw("household")
    me = next(person["id"] for person in household if person["role"] == "adult")
    node(me, session.name(me), "self", "you")
    people = session.c.extra.get("people", {})
    for person in household:
        if person["id"] != me:
            node(person["id"], session.name(person["id"]), "person", "you")
            edge(me, person["id"], "you")
    if "mother" in people or "guest" in session.c.constraints:
        mother = (people.get("mother") or {}).get("title") or {"en": "Mother", "ru": "Мама"}
        node("mother", _localized(mother, lang), "person", "you")
        edge(me, "mother", "you")
    for person_id, person in people.items():
        for index, detail in enumerate(person.get("details", [])):
            if detail.get("category") == "constraint":
                continue
            detail_id = f"{person_id}-d{index}"
            node(detail_id, _localized(detail, lang), "detail", detail["source"])
            edge(person_id if any(item["id"] == person_id for item in nodes) else me, detail_id, detail["source"])
    for trip in profile.raw("trips"):
        trip_id = f"trip-{trip['id']}"
        node(trip_id, _localized(TRIP_NAMES.get(trip["id"], {"en": trip["id"]}), lang), "rhythm", "core")
        for person_id in trip["who"]:
            edge(person_id, trip_id, "core")
    district = session.c.extra.get("district")
    places = {
        "home": {"en": f"Home · district {district}" if district else "Home", "ru": f"Дом · район {district}" if district else "Дом"},
        "school": {"en": "School", "ru": "Школа"}, "office": {"en": "Hub71 / office", "ru": "Hub71 / офис"},
        "park": {"en": "Park", "ru": "Парк"}, "beach": {"en": "Beach", "ru": "Пляж"},
        "centers": {"en": "Medical centre · ICP", "ru": "Медцентр · ICP"},
    }
    for place_id, label in places.items():
        node(f"place-{place_id}", _localized(label, lang), "place", "you" if place_id == "home" and district else "core")
    for trip in profile.raw("trips"):
        edge(f"trip-{trip['id']}", f"place-{trip['to'] if trip['to'] in places else 'home'}", "core")
    edge("place-home", "city", "core")
    node("city", {"en": "Abu Dhabi (city CURE)", "ru": "Абу-Даби (городской CURE)"}[lang], "city", "core")
    affected = {}
    for person_id, constraint in session.c.constraints.items():
        constraint_id = f"constraint-{person_id}"
        parts = []
        if "pm10_ugm3_max" in constraint:
            parts.append(f"PM10 ≤ {constraint['pm10_ugm3_max']:g}")
        if "outdoor_harsh_min_max" in constraint:
            parts.append({"en": f"{constraint['outdoor_harsh_min_max']:g} min outside in heat/dust", "ru": f"{constraint['outdoor_harsh_min_max']:g} мин на улице в жару/пыль"}[lang])
        if constraint.get("accessible_vehicle"):
            parts.append({"en": "car to the door", "ru": "машина до двери"}[lang])
        node(constraint_id, " · ".join(parts) or "—", "constraint", "sealed", sealed_reason=True)
        owner = person_id if person_id != "guest" else "mother"
        if any(item["id"] == owner for item in nodes):
            edge(owner, constraint_id, "sealed")
        hits = [f"trip-{trip['id']}" for trip in profile.raw("trips") if person_id in trip["who"]] + ["place-home", "city"]
        for target in hits:
            edge(constraint_id, target, "core", {"en": "shapes", "ru": "влияет"}[lang])
        affected[constraint_id] = hits
    inferred = session.c.extra.get("inferences", {})
    for note in session.c.extra.get("notes", []):
        if note["fact"] in inferred:
            learned_id = f"learned-{inferred[note['fact']]}"
            node(learned_id, _translated_note(session, note["fact"]), "learned", "approved")
            edge(me, learned_id, "approved")
    return {
        "nodes": nodes, "edges": edges, "affected_by_constraint": affected,
        "legend": {key: _localized(value, lang) for key, value in PROVENANCE.items()},
        "data_mode": "synthetic (fictional family)",
    }


def chronicle(session: sv.Session, limit: int = 40) -> list[dict]:
    result = []
    for fact in session.c.facts:
        source = (fact.get("source") or "").lower()
        if fact["kind"] == "sealed":
            provenance = "sealed"
        elif source.startswith("you (approved)"):
            provenance = "approved"
        elif source.startswith(("you", "voice")):
            provenance = "you"
        elif source.startswith("agent"):
            provenance = "official"
        else:
            provenance = "core"
        if fact["kind"] == "note" and fact["text"] in session.c.extra.get("inferences", {}):
            provenance = "approved"
        result.append({
            "at": fact["at"][:16].replace("T", " "), "kind": fact["kind"],
            "text": _translated_note(session, fact["text"]), "provenance": provenance,
            "label": _localized(PROVENANCE[provenance], session.lang),
        })
    return result[-limit:]


def cascade(session: sv.Session) -> dict:
    """Use core formula results to show how a dust forecast affects the household and building."""
    lang = session.lang
    items = []
    windows = session._windows()
    if windows["calculation_status"] == "computed":
        value = windows["value"]
        items.extend([
            {"who": session.name("son"), "icon": "child", "text": {"en": f"{session.name('son')}: school by car, door to door. Outdoor windows: {session._win_text(value['people'].get('son', []))}.", "ru": f"{session.name('son')}: в школу только на машине до двери. Окна для улицы: {session._win_text(value['people'].get('son', []))}."}[lang]},
            {"who": session.name("daughter"), "icon": "park", "text": {"en": f"{session.name('daughter')}: park when the air allows: {session._win_text(value['people'].get('daughter', []))}.", "ru": f"{session.name('daughter')}: парк — когда позволяет воздух: {session._win_text(value['people'].get('daughter', []))}."}[lang]},
        ])
    home = session._home(session._return_h("mon"))
    if home["calculation_status"] == "computed":
        value = home["value"]
        items.append({
            "who": {"en": "Home", "ru": "Дом"}[lang], "icon": "home",
            "text": {"en": f"Cooling from {value['start_at']}; at most {value['target_c']:g}° by {value['return_at']}.", "ru": f"Охлаждение с {value['start_at']}; к {value['return_at']} не выше {value['target_c']:g}°."}[lang],
        })
    conversation_size = len(session.c.conversation)
    neighbours = session.vision_neighbors()
    del session.c.conversation[conversation_size:]
    items.append({"who": {"en": "Lobby and neighbours", "ru": "Холл и соседи"}[lang], "icon": "building", "text": neighbours["text"]})
    conflict = session._conflict()
    if conflict["calculation_status"] == "computed" and conflict["value"].get("conflict"):
        move = conflict["value"]["move"]
        items.append({
            "who": {"en": "Car", "ru": "Машина"}[lang], "icon": "car",
            "text": {"en": f"Two people need the car: move the meeting from {move['from']} to {move['to']}.", "ru": f"Машина нужна двоим: перенести встречу с {move['from']} на {move['to']}."}[lang],
        })
    session.save()
    return {
        "headline": {"en": "Forecast: dust. Here's what changes:", "ru": "Прогноз: пыль. Вот что меняется:"}[lang],
        "items": items,
        "note": {"en": "Nothing runs without your yes — except what you trusted CURE with.", "ru": "Ничего не выполнено без твоего «да» — кроме того, что ты сама доверила CURE."}[lang],
    }


def city_gets(session: sv.Session) -> dict:
    consent_active = "arrival_vision" not in session.c.revoked
    city_card = None
    if consent_active:
        city_card = {"type": "city_request", **city_request(profile=session.profile(), policy=session.P)}
    text = {
        "en": "Preview only: this is the minimal derived context the city would receive.",
        "ru": "Только предпросмотр: это минимальный производный контекст, который получил бы город.",
    }[session.lang] if consent_active else {
        "en": "City sharing is disabled because arrival_vision consent was revoked.",
        "ru": "Передача городу отключена: согласие arrival_vision отозвано.",
    }[session.lang]
    return {
        "cure_knows": {
            "people": [session.name(person["id"]) for person in session.profile().raw("household")],
            "notes": len(session.c.extra.get("notes", [])), "memories": len(session.c.facts),
            "district": session.c.extra.get("district"), "arrival": session.c.extra.get("arrival"),
        },
        "city_gets": city_card,
        "text": text,
    }
