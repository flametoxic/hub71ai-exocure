"""CURE service layer: the resident session on top of the core. The phone, the home hub and the stage console use it.

Runtime — shared by the process: data (labelled synthetic), the city teacher, the personal contour store, the scene clock,
the "city" state (added families, stress, ramp cars).
Session — one resident: their contour (model, memory, sealed domain, trust, decisions), consents (Door), agents
(search and drafts only) and the CURE executor (the only one that acts — after a yes or within granted trust).

Every reply = an assistant.TEXTS template + engine numbers; language model styling is optional and passes number_guard.
Nothing is executed in external systems: effects are written to the contour log (external_call: False).
The Russian strings in this file are runtime replies/parsers for Russian-speaking residents.
"""
from __future__ import annotations

import copy
import re
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import os

from . import agents as ag
from . import assistant as A
from . import executor as ex
from . import gateway as gw
from . import life
from .agent_runtime import build_agent_runtime
from .apartment import cooling_bill, preload
from .cohort import cohort
from .config import Settings
from .contour import Contour, ContourStore
from .core import ANSWERS, MissingData, Store, refused, sample
from .day import simulate_day
from .explain import explain_delay
from .person import Door, PersonalModel, split_voice
from .plan import _mean, build_plan
from .sync import Teacher, learning_view, submit_fact
from .total_cost import cost_map
from .vision import arrival_counterfactual, city_request, neighbors, run_chain, storm

UTC = timezone.utc
LEILA = "leila"

# Week of Leila: moment → scene time (Abu Dhabi local time, shown as is)
MOMENTS = [
    ("sun_plan", "2026-11-22T20:00", {"ru": "Вс 20:00 · план недели", "en": "Sun 20:00 · week plan"}),
    ("mon_morning", "2026-11-23T06:50", {"ru": "Пн 06:50 · утро", "en": "Mon 06:50 · morning"}),
    ("mon_home", "2026-11-23T16:00", {"ru": "Пн 16:00 · дом к возвращению", "en": "Mon 16:00 · home"}),
    ("tue_errands", "2026-11-24T07:30", {"ru": "Вт 07:30 · дела одной поездкой", "en": "Tue 07:30 · errands"}),
    ("wed_windows", "2026-11-25T07:00", {"ru": "Ср 07:00 · окна для улицы", "en": "Wed 07:00 · outdoor windows"}),
    ("wed_reminder", "2026-11-25T20:00", {"ru": "Ср 20:00 · напоминание", "en": "Wed 20:00 · reminder"}),
    ("thu_conflict", "2026-11-26T06:45", {"ru": "Чт 06:45 · одна машина", "en": "Thu 06:45 · one car"}),
    ("fri_car", "2026-11-27T10:00", {"ru": "Пт 10:00 · вторая машина?", "en": "Fri 10:00 · second car?"}),
    ("sat_beach", "2026-11-28T09:00", {"ru": "Сб 09:00 · пляж", "en": "Sat 09:00 · beach"}),
    ("sat_guest", "2026-11-28T12:00", {"ru": "Сб 12:00 · приезд мамы", "en": "Sat 12:00 · mother's visit"}),
    ("month_end", "2026-11-30T19:00", {"ru": "30 ноя · бюджет", "en": "30 Nov · budget"}),
    ("month_later", "2026-12-22T19:00", {"ru": "Через месяц · привычки", "en": "A month later · habits"}),
]
WD = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
WD_SHORT = {"ru": ["пн", "вт", "ср", "чт", "пт", "сб", "вс"], "en": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]}
MONTHS_GEN = {"ru": ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
                     "ноября", "декабря"],
              "en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]}
MONTH_KEYS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_EN_M = ["jan(uary)?", "feb(ruary)?", "mar(ch)?", "apr(il)?", "may", "june?", "july?", "aug(ust)?", "sep(t(ember)?)?",
         "oct(ober)?", "nov(ember)?", "dec(ember)?"]
_RU_M = ["январ", "феврал", "март", "апрел", "ма[йяе]", "июн", "июл", "август", "сентябр", "октябр", "ноябр", "декабр"]
MONTH_RX = {k: rf"\b(?:{_RU_M[i]}|{_EN_M[i]})\w*" for i, k in enumerate(MONTH_KEYS)}
ACTION_LABELS = {
    "precool_home": {"ru": "охлаждение дома к приходу", "en": "pre-cooling the home"},
    "propose_visit_window": {"ru": "окно визита", "en": "visit window"},
    "calendar_event": {"ru": "событие в календаре", "en": "calendar event"},
    "reminder": {"ru": "напоминание", "en": "reminder"},
    "budget_rule": {"ru": "правило бюджета", "en": "budget rule"},
    "dispatch_ramp_car": {"ru": "машина с рампой", "en": "ramp car"},
    "propose_listing": {"ru": "квартира", "en": "listing"},
    "confirm_fact": {"ru": "факт → план", "en": "fact card → plan"},
    "grant_trust": {"ru": "разрешить делать самому", "en": "allow CURE to do it alone"},
    "model_update": {"ru": "учесть в модели", "en": "learn it in your model"}}
CATEGORY = {"rent": {"ru": "аренда", "en": "rent"}, "cooling": {"ru": "охлаждение", "en": "cooling"},
            "commute": {"ru": "дорога", "en": "commute"}, "school": {"ru": "школа", "en": "school"}}
BUDGET_WORDS = {"ru": ("меньше", "больше"), "en": ("under", "over")}


def _agent_runtime_from_settings(settings: Settings):
    """Late binding keeps local startup free of OpenAI requirements."""
    return build_agent_runtime(settings)


def lang_of(text: str | None, default: str = "en") -> str:
    if not text:
        return default
    return "ru" if re.search(r"[а-яё]", text, re.I) else "en"


def fmt_date(d, lang: str) -> str:
    d = date.fromisoformat(d) if isinstance(d, str) else d
    return f"{d.day} {MONTHS_GEN[lang][d.month - 1]}" if lang == "ru" else f"{MONTHS_GEN[lang][d.month - 1]} {d.day}"


def parse_date(text: str, *, today: date) -> date | None:
    t = text.lower()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", t)
    if m:
        return date(int(m[1]), int(m[2]), int(m[3]))
    m = re.search(r"\b(\d{1,2})[./](\d{1,2})\b", t)
    if m:
        d = date(today.year, int(m[2]), int(m[1]))
        return d if d >= today else date(today.year + 1, d.month, d.day)
    for i, k in enumerate(MONTH_KEYS):
        if re.search(MONTH_RX[k], t):
            day = re.search(r"\b(\d{1,2})\b", t)
            d = date(today.year, i + 1, int(day[1]) if day else 1)
            return d if d >= today else date(today.year + 1, d.month, d.day)
    return None


def month_in(text: str) -> str | None:
    t = text.lower()
    return next((k for k in MONTH_KEYS if re.search(MONTH_RX[k], t)), None)


def _num_words(t: str) -> dict:
    return {"вдвоем": 2, "вдвоём": 2, "втроем": 3, "втроём": 3, "вчетвером": 4, "впятером": 5, "вшестером": 6,
            "двое": 2, "трое": 3, "четверо": 4, "пятеро": 5, "один": 1, "одна": 1, "одного": 1, "два": 2, "две": 2,
            "три": 3, "четыре": 4, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def parse_household(text: str, *, infer_defaults: bool = True) -> dict:
    """Offline phrase parsing (no language model): people, children, budget, founder, arrival month."""
    t = text.lower().replace("\u00a0", " ")
    words = _num_words(t)
    people = None
    m = re.search(r"(вдвоем|вдвоём|втроем|втроём|вчетвером|впятером|вшестером)", t)
    if m:
        people = words[m[1]]
    m = m or re.search(r"(\d+|two|three|four|five|six|двое|трое|четверо|пятеро)\s*(человек|people|of us)", t)
    if people is None and m:
        people = int(m[1]) if m[1].isdigit() else words.get(m[1])
    m2 = re.search(r"(?:family of|we are|we're|нас)\s+(\d+|two|three|four|five|six|двое|трое|четверо|пятеро)", t)
    if people is None and m2:
        people = int(m2[1]) if m2[1].isdigit() else words[m2[1]]
    kids = None
    m = re.search(r"(\d+|один|одна|одного|двое|трое|два|две|три|one|two|three|a)\s*(ребен|ребён|дет|child|kid|son|daughter|сын|доч)", t)
    if m:
        kids = 1 if m[1] == "a" else (int(m[1]) if m[1].isdigit() else words.get(m[1], 1))
    elif re.search(r"\b(сын|дочь|дочк|son|daughter|kid|child|ребен|ребён)", t):
        kids = len(re.findall(r"\b(сын|дочь|дочк|son|daughter)", t)) or 1
    budget = None
    for mm in re.finditer(r"(\d[\d ,]*\d|\d+)\s*(k\b|к\b|тыс)?", t):
        val = int(re.sub(r"[ ,]", "", mm[1]))
        if mm[2]:
            val *= 1000
        if val >= 1000 and not (1900 <= val <= 2100 and not mm[2]):
            budget = val
    if people is None and infer_defaults:
        adults = 2 if re.search(r"\b(муж|жена|wife|husband|partner|мы|we)\b", t) else 1
        people = adults + (kids or 0)
    founder_match = bool(re.search(r"основател|стартап|founder|startup|hub71", t))
    return {"people": people, "children": kids if kids is not None else (0 if infer_defaults else None), "school": bool(kids),
            "budget": budget, "founder": founder_match if (founder_match or infer_defaults) else None,
            "month": month_in(t), "night": bool(re.search(r"ночн|night|late flight|поздн", t))}


class Runtime:
    """Process-wide state. Resident contours live in ContourStore (ciphertext on disk, core key)."""

    def __init__(self, *, demo_path: Path | None = None, seed: int | None = None, store: ContourStore | None = None):
        path = Path(demo_path or os.environ.get("ARRIVAL_DEMO_PATH", Path(__file__).parent / "demo"))
        self.S = {p.stem: Store.load(p) for p in path.glob("*.yaml")}
        self.P = self.S["policy"]
        self.seed = int(seed if seed is not None else os.environ.get("ARRIVAL_DEMO_SEED", "7"))
        self.listings = {x["id"]: x for x in self.S["listings"].raw("listings")}
        self.store = store or ContourStore.from_env(self.P)
        self._initialize_process()
        if os.environ.get("CURE_REAL_CITY", "1").strip().lower() not in {"0", "false", "no"}:
            from addon.city import apply_real_city
            apply_real_city(self)

    def reset(self) -> None:
        for rid in list(getattr(self, "sessions", {})):
            self.store.delete(rid)
        for p in self.store.dir.glob("*.bin"):
            p.unlink()
        self.store.forget_process_memory()
        self._initialize_process()

    def _initialize_process(self) -> None:
        """Initialize scene machinery without erasing persisted residents."""
        self.teacher = Teacher(policy=self.P, cohort=self.S["cohort"])
        self.clock = datetime.fromisoformat(self.P.get("scenario.today")).replace(hour=9, tzinfo=UTC)
        self.moment_i = -1
        self.city = {"extra_families": {}, "stress": False, "ramp_cars": 0, "pending_ramp": None}
        self.sessions: dict[str, Session] = {}
        ANSWERS.clear()

    # ---- scene clock
    def set_clock(self, iso: str) -> None:
        self.clock = datetime.fromisoformat(iso).replace(tzinfo=UTC)

    def today(self) -> date:
        return self.clock.date()

    def templates(self) -> list[dict]:
        try:
            return ag.load_templates(self.S["agents"])
        except RuntimeError:
            return []

    # ---- residents
    def session(self, rid: str, *, device: str | None = None) -> "Session | None":
        s = self.sessions.get(rid)
        if s is None:
            c = self.store.open(rid, device=device)
            if c is None:
                return None
            s = self.sessions[rid] = Session(self, c)
            from .dynamic_agents import recover
            recover(s)
            if rid == LEILA:
                from .mind import sync_week_events
                sync_week_events(s)
        elif device and device not in s.c.devices:
            s.c.devices.append(device)
        return s

    def new_session(self, rid: str, *, profile_doc: dict, sealed: dict, constraints: dict, lang: str,
                    device: str | None = None) -> "Session":
        c = self.store.create(rid, profile_doc=profile_doc, sealed=sealed, constraints=constraints, now=self.clock)
        c.settings["lang"] = lang
        if device:
            c.devices.append(device)
        s = self.sessions[rid] = Session(self, c)
        return s

    def restart_process(self) -> None:
        """Like a core service restart: memory is empty, contours are ciphertext on disk."""
        self.store.forget_process_memory()
        self.sessions.clear()

    def moments(self, lang: str = "en") -> list[dict]:
        return [{"id": m, "at": at, "title": t[lang], "done": i <= self.moment_i} for i, (m, at, t) in enumerate(MOMENTS)]


class Session:
    def __init__(self, rt: Runtime, c: Contour):
        self.rt, self.c, self.S, self.P = rt, c, rt.S, rt.P
        self.cache: dict = {}
        self._door: Door | None = None

    # ------------------------------------------------------------------ helpers
    @property
    def lang(self) -> str:
        return self.c.settings.get("lang", "en")

    @property
    def now(self) -> datetime:
        return self.rt.clock

    @property
    def rid(self) -> str:
        return self.c.resident_id

    def profile(self) -> Store:
        return self.c.profile()

    def save(self) -> None:
        self.rt.store.save(self.c)

    def door(self) -> Door:
        if self._door is None:
            self._door = Door(profile=self.profile(), policy=self.P, now=self.now)
            for pur in self.c.revoked:
                self._door.revoke(pur, now=self.now)
        return self._door

    def _consent(self, purpose: str) -> bool:
        cat, fields, role, action = {
            "arrival_planning": ("household", ["household", "flags"], "arrival_twin", "compute"),
            "arrival_cost": ("finance", ["budget", "value_of_time"], "arrival_twin", "compute"),
            "arrival_vision": ("constraints", ["pm10_ugm3_max", "outdoor_min_max", "indoor_c_max_on_arrival"],
                               "city_core", "propose")}[purpose]
        r = self.door().check(purpose=purpose, category=cat, fields=fields, role=role, action=action, now=self.now)
        self.c.access_log.append(self.door().log[-1])
        return bool(r["allowed"])

    def name(self, pid: str) -> str:
        p = next((p for p in self.profile().raw("household") if p["id"] == pid), None)
        n = (p or {}).get("name") or {}
        return n.get(self.lang) or n.get("en") or pid

    def kid(self) -> str:
        from .total_cost import focus_person
        return self.name(focus_person(self.profile()))

    def t(self, x: dict, key: str = "title") -> str:
        return x.get(f"{key}_ru", x.get(key)) if self.lang == "ru" else x.get(key)

    def say(self, parts, *, cards=(), traces=(), route: str = "", quick=()) -> dict:
        text = " ".join(p for p in ([parts] if isinstance(parts, str) else parts) if p)
        st = A.with_style(text, trace_ids=[t for t in traces if t], lang=self.lang, policy=self.P)
        self.c.say("cure", st["text"], at=self.now, trace_ids=[t for t in traces if t])
        self.c.conversation[-1].update(cards=list(cards), quick=list(quick), route=route)
        last = self.c.extra.setdefault("last", {})
        for c in cards:
            last[c.get("type", "card")] = c
        last["reply"] = {"route": route, "text": st["text"], "at": self.now.isoformat()}
        self.c.extra["rev"] = self.c.extra.get("rev", 0) + 1
        self.save()
        return {"route": route, "text": st["text"], "styled": st["styled"], "lang": self.lang, "cards": list(cards),
                "trace_ids": [t for t in traces if t], "quick": list(quick), "clock": self.now.isoformat(),
                "pending": self.pending_view()}

    def r(self, key: str, **kw) -> str:
        return A.render(key, self.lang, **kw)

    def refused_reply(self, ans: dict, route: str) -> dict:
        return self.say(self.r("dont_know", missing=", ".join(ans.get("missing", [])) or ans.get("what", "")),
                        cards=[{"type": "refused", "answer": ans}], traces=[ans.get("trace_id")], route=route)

    # ------------------------------------------------------------------ agents and executor
    def agents(self) -> list[dict]:
        tpls = self.rt.templates()
        if not tpls:
            return []
        target = max(self.now + timedelta(days=120), self.now + timedelta(days=1))
        return ag.build_agents(profile=self.profile(), templates=tpls, now=self.now,
                               goal={"class": "settle_by_date", "target": target.date().isoformat(), "trace_id": "goal"})

    def agent(self, aid: str) -> dict | None:
        return next((a for a in self.agents() if a["id"] == aid), None)

    def executor(self) -> dict:
        return ex.build_executor(profile=self.profile(), now=self.now, deadline=self.now + timedelta(days=365),
                                 autonomy=self.c.trust.autonomy().get(ex.EXECUTOR_ID, set()),
                                 trace_id=f"exec-{uuid.uuid4().hex[:6]}")

    def propose(self, action: str, *, scope: str, facts: dict, title: str, when: str | None = None,
                traces=(), physical_safe: bool = True, force_review: bool = False) -> tuple[dict, str]:
        """→ (proposal, text). Only what the person allowed on the trust ladder runs without them."""
        item = ex.propose(self.executor(), action=action, scope=scope, facts=facts, title=title, when=when,
                          now=self.now, physical_safe=physical_safe,
                          consent_valid=self._consent("arrival_planning"), trace_ids=list(traces))
        if item["status"] == "blocked":
            return item, self.r("blocked", title=title, failed=", ".join(item["gate"]["failed"]))
        if force_review and item["status"] == "ready":
            item.update(status="needs_you", autonomous=False, reason="dynamic specialist actions require user review")
        if item["status"] == "ready":
            res = ex.execute(item, approved=False, effects=self.c.effects, now=self.now)
            self.c.remember(at=self.now, kind="effect", text=title, source="CURE executor (autonomous)")
            return res, self.r("auto_done", title=title)
        self.c.pending[item["id"]] = item
        return item, ""

    def pending_view(self) -> list[dict]:
        return [{"id": k, "action": v["action"], "title": v["title"], "when": v.get("when"), "facts": v["facts"],
                 "label": ACTION_LABELS.get(v["action"], {}).get(self.lang, v["action"]), "kind": v.get("kind", "action")}
                for k, v in self.c.pending.items()]

    def draft(self, aid: str, draft_id: str, facts: dict, traces=()) -> dict | None:
        a = self.agent(aid)
        if a is None:
            return None
        runtime = _agent_runtime_from_settings(Settings.from_env())
        d = runtime.draft(a, draft_id, facts=facts, lang=self.lang, now=self.now, trace_ids=list(traces))
        if d.get("status") == "ready":
            d["id"] = uuid.uuid4().hex[:8]
            self.c.extra.setdefault("drafts", []).append(d)
            self.c.remember(at=self.now, kind="draft", text=f"{d['agent_title']}: {draft_id}", source=f"agent:{aid}")
        return d

    # ------------------------------------------------------------------ 1. door
    def open_door(self, voice: str) -> dict:
        split = split_voice(voice, self.P)
        partial_profile = "profile_fields" in self.c.extra
        parsed = parse_household(voice, infer_defaults=not partial_profile)  # parsed locally; only text_for_llm may leave
        self.c.extra["parsed"] = parsed
        self.door()
        self.c.outbox.append({"at": self.now.isoformat(), "to": "language model (only if connected)",
                              "what": split["text_for_llm"]})
        n_sealed = sum(len(v) for v in self.c.sealed.values())
        self.c.remember(at=self.now, kind="door", text=split["text_for_llm"], source="voice")
        card = {"type": "door", "heard": voice, "sent_to_language_model": split["text_for_llm"],
                "sealed_items": n_sealed, "sealed_where": "CURE core · your contour · encrypted",
                "constraints": {self.name(k): v for k, v in self.c.constraints.items()},
                "pseudonyms": {k: v[:8] + "…" for k, v in self.door().ids.items()},
                "consents": self.door().consents_view(self.now), "parsed": parsed}
        try:
            stored_budget = self.profile().get("params.budget_aed_month")
        except MissingData:
            stored_budget = None
        budget = parsed["budget"] or stored_budget
        school = parsed["children"] if parsed["school"] else (0 if parsed["children"] is not None else "not provided")
        self.c.extra["asked"] = self.c.extra.get("asked", 0) + 1
        self.c.extra["topic"] = "arrival_date"
        return self.say([self.r("door_ack", people=parsed["people"] or "not provided", school=school,
                                  budget=A.n0(budget) if budget is not None else "not provided"),
                         self.r("ask_arrival_date")], cards=[card], route="door")

    def set_arrival(self, d: date) -> dict:
        self.c.extra["arrival"] = d.isoformat()
        if "profile_fields" in self.c.extra:
            from .modeling import ModelRegistry
            from .profiles import ProfileService
            ProfileService(ModelRegistry.load_default()).record(self, "arrival_date", value=d.isoformat(), source="user")
        self.c.remember(at=self.now, kind="plan", text=f"arrival {d.isoformat()}", source="you")
        return self.plan(route="arrival")

    # ------------------------------------------------------------------ 2. districts
    def cost_map(self, force: bool = False) -> dict:
        if force or "cost_map" not in self.cache:
            if not self._consent("arrival_cost"):
                return refused(MissingData(["consent:arrival_cost"]), what="cost_map")
            self.cache["cost_map"] = cost_map(profile=self.profile(), city=self.S["city"], climate=self.S["climate"],
                                              tariffs=self.S["tariffs"], policy=self.P, listings=self.rt.listings,
                                              seed=self.rt.seed, lam=self.c.model.lambda_mean())
        return self.cache["cost_map"]

    def _axes(self, cm: dict, d: str) -> list[float]:
        return cm["value"]["axes"]["values"][d]

    def districts(self) -> dict:
        cm = self.cost_map(force=True)
        if cm["calculation_status"] != "computed":
            return self.refused_reply(cm, "districts")
        v = cm["value"]
        rec, other = v["recommended"], v.get("runner_up") or next(d for d in v["axes"]["values"] if d != v["recommended"])
        a, b = self._axes(cm, rec), self._axes(cm, other)
        more_money = a[0] > b[0]
        words = {"ru": (("дороже", "дешевле"), ("меньше", "больше")), "en": (("more", "less"), ("less", "more"))}[self.lang]
        switch = self.r("switch_line", w=A.n1(v["switch"]["money_weight"]), to=v["switch"]["to"]) if v.get("switch") else \
            self.r("no_switch", rec=rec)
        self.c.extra["topic"] = "district"
        text = self.r("district_reco", rec=rec, other=other, money_word=words[0][0 if more_money else 1],
                      money=A.n0(abs(a[0] - b[0])), hours=A.n1(abs(a[1] - b[1])),
                      hours_word=words[1][0 if a[1] < b[1] else 1], kid=self.kid(), adam_rec=A.n0(a[2]),
                      adam_other=A.n0(b[2]), why=A.AXIS[self.lang].get(v.get("why"), v.get("why")), switch=switch)
        card = {"type": "districts", "recommended": rec, "runner_up": other, "pareto": v["pareto"],
                "axes": v["axes"], "why": v.get("why"), "switch": v.get("switch"), "lambda": v["lambda_used"],
                "focus": self.kid(), "over_budget": v["over_housing_budget"],
                "rows": {d: {k: r[k]["mid"] for k in ("money", "rent", "cooling", "commute_money", "school", "time_h",
                                                       "son_harsh_min")} for d, r in v["districts"].items()}}
        return self.say(text, cards=[card], traces=[cm["trace_id"]], route="districts",
                        quick=[{"ru": "Почему?", "en": "Why?"}[self.lang],
                               {"ru": "А если деньги важнее?", "en": "What if money matters more?"}[self.lang],
                               {"ru": "А в январе?", "en": "What about January?"}[self.lang]])

    def district_why(self) -> dict:
        cm = self.cost_map()
        v = cm["value"]
        from .total_cost import why_and_switch
        ws = why_and_switch({d: v["axes"]["values"][d] for d in v["pareto"]}, v["lambda_used"], v["axes"]["names"]) \
            if len(v["pareto"]) > 1 else {"advantage_by_axis": {}, "runner_up": None}
        adv = ws["advantage_by_axis"]
        self.c.extra["topic"] = "district"
        text = self.r("district_why", rec=v["recommended"], other=ws.get("runner_up") or "—",
                      why=A.AXIS[self.lang].get(v.get("why"), v.get("why")),
                      a_money=f"{adv.get('money', 0):.2f}", a_time=f"{adv.get('time', 0):.2f}",
                      a_comfort=f"{adv.get('comfort', 0):.2f}")
        return self.say(text, cards=[{"type": "why_district", "advantage_by_axis": adv, "lambda": v["lambda_used"],
                                      "formula": "score = Σ λ·normalised(axis); advantage = λ·(runner_up − recommended)"}],
                        traces=[cm["trace_id"]], route="district_why")

    def district_switch(self) -> dict:
        v = self.cost_map()["value"]
        text = self.r("switch_line", w=A.n1(v["switch"]["money_weight"]), to=v["switch"]["to"]) if v.get("switch") \
            else self.r("no_switch", rec=v["recommended"])
        return self.say(text, traces=[self.cost_map()["trace_id"]], route="district_switch")

    def whatif_month(self, month: str) -> dict:
        cm = self.cost_map()
        d = self.c.extra.get("district") or cm["value"]["recommended"]
        from .apartment import calendar_months
        from .total_cost import focus_person
        f = focus_person(self.profile())
        kw = dict(profile=self.profile(), city=self.S["city"], climate=self.S["climate"], policy=self.P, districts=[d],
                  seed=self.rt.seed)
        then = simulate_day(**kw, month=month)
        if then["calculation_status"] != "computed":
            return self.refused_reply(then, "whatif_month")
        g = lambda a: (a["value"][d]["transit_min"]["mid"], a["value"][d]["harsh_outdoor_min"][f]["mid"])
        by_month = {m: simulate_day(**kw, month=m) for m in calendar_months(self.S["climate"]) if m != month}
        base = max(by_month, key=lambda m: g(by_month[m])[1])          # the hardest month in the data — for comparison
        now_ = by_month[base]
        (t0, h0), (t1, h1) = g(now_), g(then)
        text = self.r("whatif_month", month=A.MONTHS[self.lang][month], d=d, transit=A.n0(t1), transit_before=A.n0(t0),
                      kid=self.kid(), adam=A.n0(h1), adam_before=A.n0(h0), base=A.MONTHS[self.lang][base])
        return self.say(text, cards=[{"type": "day", "district": d, "month": month, "base_month": base,
                                      "after": then["value"][d], "before": now_["value"][d]}],
                        traces=[now_["trace_id"], then["trace_id"]], route="whatif_month")

    def whatif_money(self) -> dict:
        cm = self.cost_map()
        v = cm["value"]
        to = (v.get("switch") or {}).get("to")
        if not to:
            return self.say(self.r("no_switch", rec=v["recommended"]), traces=[cm["trace_id"]], route="whatif_money")
        a = self._axes(cm, to)
        self.c.extra["offer_switch"] = to
        self.c.extra["topic"] = "district"
        return self.say(self.r("whatif_money", to=to, money=A.n0(a[0]), hours=A.n1(a[1]), kid=self.kid(), adam=A.n0(a[2])),
                        traces=[cm["trace_id"]], route="whatif_money",
                        quick=[{"ru": "Применить", "en": "Apply"}[self.lang]])

    def choose(self, district: str | None = None) -> dict:
        cm = self.cost_map()
        vals = cm["value"]["axes"]["values"]
        d = district or self.c.extra.pop("offer_switch", None) or cm["value"]["recommended"]
        if d not in vals:
            return self.say(self.r("fallback"), route="choose")
        self.c.model.observe_choice(vals, d, at=self.now.isoformat())
        self.c.extra["district"] = d
        self.c.remember(at=self.now, kind="choice", text=f"district {d}", source="you")
        lam = self.c.model.lambda_mean()
        self.cache.pop("cost_map", None)
        return self.say(self.r("chosen", d=d, m=f"{lam['money']:.2f}", t=f"{lam['time']:.2f}", c=f"{lam['comfort']:.2f}"),
                        cards=[{"type": "model", "model": self.c.model.summary()}], route="choose")

    def apartment(self, listing_id: str | None = None, answer_key: str | None = None) -> dict:
        lid = listing_id or self.c.extra.get("listing") or self.S["city"].get(
            f"typical_listing.{self.c.extra.get('district') or 'A'}")
        answers = dict(self.c.extra.get("apt_answers", {}))
        first = "apt_bill" not in self.cache
        if answer_key:
            ask = self.cache.get("apt_ask")
            if ask:
                answers[ask] = answer_key
                self.c.extra["apt_answers"] = answers
        self.c.extra["listing"] = lid
        b = cooling_bill(listing=self.rt.listings[lid], profile=self.profile(), climate=self.S["climate"],
                         tariffs=self.S["tariffs"], policy=self.P, seed=self.rt.seed, answers=answers)
        if b["calculation_status"] != "computed":
            return self.refused_reply(b, "apartment")
        v = b["value"]
        self.cache["apt_bill"], self.cache["apt_ask"] = b, v["ask"]
        tot = v["total_months_in_data"]
        self.c.extra["topic"] = "apartment"
        opts = [{"key": o, "label": A.OPTION_LABELS.get(o, {}).get(self.lang, o)} for o in v["options"]]
        q = (A.QUESTIONS_RU.get(v["ask"]) if self.lang == "ru" else v["question"]) if v["ask"] else None
        card = {"type": "apartment", "listing": lid, "total": tot, "months": v["months"], "ask": v["ask"],
                "question": q, "options": opts, "answers": answers, "sensitivity": v["sensitivity"]}
        if answer_key or not q:
            return self.say(self.r("narrowed", low=A.n0(tot["low"]), high=A.n0(tot["high"])), cards=[card],
                            traces=[b["trace_id"]], route="apartment")
        return self.say(self.r("ask_param", question=q, low=A.n0(tot["low"]), high=A.n0(tot["high"])), cards=[card],
                        traces=[b["trace_id"]], route="apartment", quick=[o["label"] for o in opts])

    # ------------------------------------------------------------------ plan, agents, fact gateway
    def _plan(self, delays: dict | None = None) -> dict:
        start = date.fromisoformat(self.P.get("scenario.today"))
        target = date.fromisoformat(self.c.extra.get("arrival") or self.P.get("scenario.move_in"))
        ov = {k: {kk: vv for kk, vv in o.items()} for k, o in self.c.overrides.items()}
        return build_plan(profile=self.profile(), steps=self.S["steps"], queues=self.S["queues"], policy=self.P,
                          start=start, target=target, seed=self.rt.seed,
                          delays=delays if delays is not None else self.c.extra.get("delays", {}),
                          learned=self.rt.teacher.learned(), overrides=ov,
                          completed={sid: max(0, (date.fromisoformat(event['date']) - start).days)
                                     for sid, event in self.c.extra.get('completed_steps', {}).items()})

    def _step_title(self, sid: str, lower: bool = False) -> str:
        s = next((s for s in self.S["steps"].raw("steps") if s["id"] == sid), {"title": sid})
        t = self.t(s)
        proper = t.split()[0] in ("Emirates", "Tawtheeq")
        return t[0].lower() + t[1:] if lower and not proper and len(t) > 1 and t[1].islower() else t

    def plan_card(self, p: dict) -> dict:
        v = p["value"]
        return {"type": "plan", "ready_date": v["ready_date"], "p_by_target": v["p_by_target"],
                "critical_path": [self._step_title(s) for s in v["critical_path"]],
                "steps": [{"id": s["id"], "title": self._step_title(s["id"]), "finish_day": s["finish_day"]["mid"],
                           "critical": s["critical"], "source_card": s["source_card"],
                           "facts_from_teacher": s["facts_from_teacher"]} for s in v["steps"]],
                "target": self.c.extra.get("arrival") or self.P.get("scenario.move_in"), "trace_id": p["trace_id"]}

    def plan(self, route: str = "plan") -> dict:
        if not self._consent("arrival_planning"):
            return self.say(self.r("dont_know", missing="consent:arrival_planning"), route=route)
        p = self._plan()
        if p["calculation_status"] != "computed":
            return self.refused_reply(p, route)
        self.cache["plan"] = p
        v = p["value"]
        bott = self._step_title(v["critical_path"][1] if len(v["critical_path"]) > 1 else v["critical_path"][0])
        self.c.extra["topic"] = "plan"
        return self.say(self.r("plan_status", mid=fmt_date(v["ready_date"]["mid"], self.lang),
                               low=fmt_date(v["ready_date"]["low"], self.lang), high=fmt_date(v["ready_date"]["high"], self.lang),
                               bottleneck=bott), cards=[self.plan_card(p)], traces=[p["trace_id"]], route=route)

    def agent_search(self) -> dict:
        a = self.agent("documents")
        if a is None:
            return self.say(self.r("dont_know", missing="signed agent templates"), route="agent_search")
        allowed = ag.try_action(a, action="search_public", scope="documents_status", now=self.now)
        if allowed["decision"] != "allowed":
            return self.say(self.r("dont_know", missing=allowed["reason"]), route="agent_search")
        cards, parts = [], []
        pages = gw.search(self.S["web_corpus"], "step_durations")
        subjects = sorted(step["id"] for step in self.S["steps"].raw("steps"))
        runtime = _agent_runtime_from_settings(Settings.from_env())
        artifacts = runtime.extract_fact_cards(a, pages, allowed_subjects=subjects, now=self.now)
        page_text_by_url = {page["url"]: page["text"] for page in pages}
        for artifact in artifacts:
            card = artifact["card"]
            chk = gw.check_card(card, policy=self.P, steps=self.S["steps"])
            cid = uuid.uuid4().hex[:8]
            entry = {"type": "fact_card", "id": cid, "agent": a["id"], "status": chk["status"], "reasons": chk["reasons"],
                     "card": chk["card"], "page_text": page_text_by_url.get(card.get("source_url"), ""),
                     "agent_runtime": artifact["agent_runtime"], "model": artifact.get("model"),
                     "response_id": artifact.get("response_id")}
            if chk["accepted"]:
                self.c.extra.setdefault("fact_cards", {})[cid] = chk["card"]
                self.c.pending[cid] = {"id": cid, "action": "confirm_fact", "kind": "fact", "facts": chk["card"],
                                       "title": f"{self._step_title(chk['card']['subject'])}: {chk['card']['low']}–{chk['card']['high']}",
                                       "when": None, "status": "needs_you"}
                parts.append(self.r("fact_card", subject=self._step_title(chk["card"]["subject"]), low=chk["card"]["low"],
                                    high=chk["card"]["high"], source=chk["card"]["source_url"]))
            else:
                parts.append(self.r("fact_rejected", reasons=", ".join(chk["reasons"])) +
                             f" ({card.get('source_url', 'unknown source')})")
            cards.append(entry)
        self.c.extra["topic"] = "plan"
        return self.say(parts, cards=cards, route="agent_search")

    def confirm_fact(self, cid: str, approved: bool) -> dict:
        card = self.c.extra.get("fact_cards", {}).pop(cid, None)
        self.c.pending.pop(cid, None)
        if card is None:
            return self.say(self.r("fallback"), route="confirm_fact")
        if not approved:
            return self.say(self.r("declined", title=card["subject"]).split(".")[0] + ".", route="confirm_fact")
        before = self._plan()
        self.c.overrides[card["subject"]] = gw.to_override(card)
        self.c.remember(at=self.now, kind="fact", text=f"{card['subject']} {card['low']}–{card['high']} d",
                        source=card["source_url"])
        after = self._plan()
        self.cache["plan"] = after
        return self.say(self.r("plan_updated", mid=fmt_date(after["value"]["ready_date"]["mid"], self.lang),
                               before=fmt_date(before["value"]["ready_date"]["mid"], self.lang)),
                        cards=[self.plan_card(after)], traces=[before["trace_id"], after["trace_id"]], route="confirm_fact")

    # ------------------------------------------------------------------ visa +N: why, what if, what helps
    def visa_delay(self, days: float = 14) -> dict:
        before = self._plan()
        delays = {**self.c.extra.get("delays", {}), "entry_permit": float(days)}
        after = self._plan(delays)
        if after["calculation_status"] != "computed":
            return self.refused_reply(after, "visa_delay")
        self.c.extra["delays"] = delays
        self.cache["plan"] = after
        why = explain_delay(profile=self.profile(), steps=self.S["steps"], queues=self.S["queues"], policy=self.P,
                            seed=self.rt.seed, visa_delay_days=float(days))
        if why["calculation_status"] != "computed":
            return self.refused_reply(why, "visa_delay")
        self.cache["why"] = why
        w, va, vb = why["value"], after["value"], before["value"]
        shift = (date.fromisoformat(va["ready_date"]["mid"]) - date.fromisoformat(vb["ready_date"]["mid"])).days
        cost = max(shift, 0) * float(self.S["rates"].get("temporary_housing_per_day"))
        cp = set(va["critical_path"])
        chain = [self._step_title(s, lower=True) for s in w["why"]["chain"] if s != "entry_permit" and s in cp]
        lev = w["levers"]
        titles = {lv["id"]: lv for lv in self.P.get("plan_levers")}
        top = lev[0]
        bank = next((x for x in lev if x["id"] == "bank_early"), None)
        # medical slot: the day after the entry permit (plan engine) and the hour with the shortest queue (queue data)
        start = date.fromisoformat(self.P.get("scenario.today"))
        ep = next(s for s in va["steps"] if s["id"] == "entry_permit")
        mday = start + timedelta(days=int(np.ceil(ep["finish_day"]["mid"])))
        hours = self.S["queues"].raw("centers.medical.wait_days_by_hour")
        best_h = int(min(hours, key=lambda h: _mean(hours[h])))
        slot = f"{fmt_date(mday, self.lang)}, {best_h:02d}:00"
        item, auto = self.propose("propose_visit_window", scope="documents_status",
                                  facts={"center": "medical", "date": mday.isoformat(), "hour": best_h,
                                         "start": f"{mday.isoformat()}T{best_h:02d}:00", "duration_min": 60,
                                         "title": "Medical test"},
                                  title={"ru": f"медкомиссия {slot}", "en": f"medical test {slot}"}[self.lang],
                                  when=f"{mday.isoformat()}T{best_h:02d}:00", traces=[why["trace_id"], after["trace_id"]])
        rd = {k: fmt_date(va["ready_date"][k], self.lang) for k in ("low", "mid", "high")}
        letter = self.draft("documents", "employer_letter", {"ready_mid": rd["mid"], "ready_low": rd["low"],
                                                             "ready_high": rd["high"]}, traces=[after["trace_id"]])
        text = self.r("visa_delay", days=A.n0(days), mid=rd["mid"], low=rd["low"], high=rd["high"], cost=A.n0(cost),
                      chain=", ".join(chain), lever=self.t(titles[top["id"]]), lever_days=A.n1(top["days_saved"]),
                      bank_line=self.r("bank_line") if bank and bank["days_saved"] < 0.5 else "", slot=slot)
        if auto:
            text = text.split(("Предлагаю" if self.lang == "ru" else "I suggest"))[0] + auto
        self.c.extra["topic"] = "visa"
        cards = [{"type": "plan_change", "before": self.plan_card(before), "after": self.plan_card(after),
                  "temporary_housing_aed": cost},
                 {"type": "why", "chain": [self._step_title(s) for s in w["why"]["chain"]],
                  "untouched": [self._step_title(s) for s in w["why"]["untouched"]],
                  "counterfactual": w["counterfactual"], "levers": [{**x, "title": self.t(titles[x["id"]])} for x in lev],
                  "engine": w["engine"], "trace_id": why["trace_id"]},
                 {"type": "proposal", **item}]
        if letter:
            cards.append({"type": "draft", **letter})
        return self.say(text, cards=cards, traces=[before["trace_id"], after["trace_id"], why["trace_id"]], route="visa_delay",
                        quick=[{"ru": "Почему?", "en": "Why?"}[self.lang],
                               {"ru": "А если бы не задержка?", "en": "What if there was no delay?"}[self.lang],
                               {"ru": "Что поможет?", "en": "What helps?"}[self.lang]])

    def _why_ready(self) -> dict | None:
        return self.cache.get("why") or (self.visa_delay(self.c.extra.get("delays", {}).get("entry_permit", 14)) and
                                         self.cache.get("why"))

    def visa_why(self) -> dict:
        w = self._why_ready()
        v = w["value"]["why"]
        chain = " → ".join(self._step_title(s) for s in v["chain"])
        untouched = ", ".join(self._step_title(s) for s in v["untouched"]) or "—"
        text = {"ru": f"Причинная цепочка: {chain}. Не затронуто: {untouched}.",
                "en": f"Causal chain: {chain}. Not affected: {untouched}."}[self.lang]
        return self.say(text, cards=[{"type": "why", "chain": v["chain_titles"], "shift_days": v["shift_days"],
                                      "engine": w["value"]["engine"]}], traces=[w["trace_id"]], route="visa_why")

    def visa_counterfactual(self) -> dict:
        w = self._why_ready()
        cf = w["value"]["counterfactual"]
        return self.say(self.r("cf_line", days=A.n1(cf["days_lost"]), without=A.n1(cf["without_delay_day"]),
                               with_=A.n1(cf["with_delay_day"])), cards=[{"type": "counterfactual", **cf,
                                                                           "engine": w["value"]["engine"]}],
                        traces=[w["trace_id"]], route="visa_counterfactual")

    def visa_levers(self) -> dict:
        w = self._why_ready()
        titles = {lv["id"]: lv for lv in self.P.get("plan_levers")}
        items = "; ".join(f"{self.t(titles[x['id']])} — {A.n1(x['days_saved'])} " + {"ru": "дн", "en": "days"}[self.lang]
                          for x in w["value"]["levers"])
        return self.say(self.r("levers_line", items=items), cards=[{"type": "levers", "levers": w["value"]["levers"]}],
                        traces=[w["trace_id"]], route="visa_levers")

    def complete_step(self, step_id: str) -> dict:
        steps = {step['id']: step for step in self.S['steps'].raw('steps')}
        if step_id not in steps:
            raise ValueError('unknown step')
        completed = self.c.extra.setdefault('completed_steps', {})
        def record(sid, provenance):
            for blocker in steps[sid].get('blocks_on', []):
                if blocker not in completed:
                    record(blocker, 'core')
            completed.setdefault(sid, {'date': self.now.date().isoformat(), 'provenance': provenance,
                                       'evidence': 'user reports ' + step_id + ' completed'})
        record(step_id, 'user')
        self.c.remember(at=self.now, kind='step_completed', text=step_id + ' completed', source='user')
        self.cache.clear()
        reply = self.plan(route='complete_step')
        prefix = 'Emirates ID отмечен как полученный. Зависимые шаги открыты. ' if self.lang == 'ru' else 'Emirates ID is recorded as received. Dependent steps are available. '
        reply['text'] = prefix + reply['text']
        self.c.conversation[-1]['text'] = reply['text']
        self.save()
        return reply

    def report_fact(self, step_id: str = "emirates_id", days: float | None = None) -> dict:
        f = next((f for f in self.S["cohort"].raw("demo_facts") if f["step_id"] == step_id), None)
        observed = float(days if days is not None else f["observed_days"]["value"])
        r = submit_fact(self.rt.teacher, step_id=step_id, observed_days=observed,
                        source_integrity=(f or {}).get("source_integrity", "verified"),
                        pseudonym=self.profile().get("pseudonym"), steps=self.S["steps"], policy=self.P,
                        versions=self.S["versions"], seed=self.rt.seed)
        if r["calculation_status"] != "computed":
            return self.refused_reply(r, "report_fact")
        v = r["value"]
        if v.get("sent_to_teacher"):
            self.c.outbox.append({"at": self.now.isoformat(), "to": "city CURE (teacher)", "what": v["sent_to_teacher"]})
        text = self.r("teacher_learned", step=self._step_title(step_id), days=A.n1(observed)) if v["status"] == "stored_as_fact" \
            else {"ru": f"Факт не принят учителем: {v['verdict']}.", "en": f"The teacher did not accept the fact: {v['verdict']}."}[self.lang]
        return self.say(text, cards=[{"type": "teacher", **v}], traces=[r["trace_id"]], route="report_fact")

    # ------------------------------------------------------------------ decisions and trust
    def decide(self, item_id: str | None, approved: bool) -> dict:
        if item_id is None:
            if not self.c.pending:
                return self.say(self.r("fallback"), route="decide")
            item_id = list(self.c.pending)[-1]
        item = self.c.pending.get(item_id)
        if item is None:
            return self.say(self.r("fallback"), route="decide")
        kind = item.get("kind", "action")
        if kind == "fact":
            return self.confirm_fact(item_id, approved)
        if kind == "model":
            return self.apply_habit(item_id, approved)
        if kind == "memory":
            self.c.pending.pop(item_id)
            from addon.life import accept_memory
            return accept_memory(self, item, approved)
        if kind == "trust":
            self.c.pending.pop(item_id)
            return self.grant_trust(item["facts"]["action"]) if approved else self.say(
                self.r("declined", title=item["title"]).split(".")[0] + ".", route="trust")
        self.c.pending.pop(item_id)
        res = ex.execute(item, approved=approved, effects=self.c.effects, now=self.now)
        rec = self.c.trust.record(ex.EXECUTOR_ID, item["action"], approved=approved, at=self.now.isoformat())
        self.c.remember(at=self.now, kind="decision", text=f"{'yes' if approved else 'no'}: {item['title']}", source="you")
        parts = [self.r("approved" if approved else "declined", title=item["title"])]
        cards = [{"type": "effect", **res}]
        if approved and item["action"] == "dispatch_ramp_car":
            self.rt.city["ramp_cars"] += 1
            parts.append(self.vision_cohort(stress=True, speak=False))
        offer = rec.get("offer") or {}
        if approved and offer.get("eligible"):
            tid = f"trust-{item['action']}"
            label = ACTION_LABELS.get(item["action"], {}).get(self.lang, item["action"])
            self.c.pending[tid] = {"id": tid, "kind": "trust", "action": "grant_trust", "title": label,
                                   "facts": {"action": item["action"]}, "when": None, "status": "needs_you"}
            parts.append(self.r("trust_offer", n=rec["streak"], action=label))
        if item["action"] == "calendar_event" and res.get("effect", {}).get("ics"):
            cards.append({"type": "ics", "ics": res["effect"]["ics"]})
        return self.say(parts, cards=cards, traces=item.get("trace_ids", []), route="decide")

    def grant_trust(self, action: str) -> dict:
        try:
            self.c.trust.grant(ex.EXECUTOR_ID, action, at=self.now.isoformat())
        except PermissionError as e:
            return self.say(str(e), route="trust")
        self.c.remember(at=self.now, kind="trust", text=f"autonomy: {action}", source="you")
        return self.say(self.r("trust_granted"), cards=[{"type": "trust", "levels": self.trust_view()}], route="trust")

    def revoke_trust(self, action: str) -> dict:
        self.c.trust.revoke(ex.EXECUTOR_ID, action, at=self.now.isoformat())
        return self.say(self.r("declined", title=ACTION_LABELS.get(action, {}).get(self.lang, action)),
                        cards=[{"type": "trust", "levels": self.trust_view()}], route="trust")

    def trust_view(self) -> list[dict]:
        return self.c.trust.levels(ex.card(self.executor()))

    # ------------------------------------------------------------------ week of Leila
    def run_moment(self, mid: str) -> dict:
        at = next(m for m in MOMENTS if m[0] == mid)
        self.rt.set_clock(at[1])
        self.rt.moment_i = [m[0] for m in MOMENTS].index(mid)
        out = getattr(self, f"m_{mid}")()
        out["moment"] = {"id": mid, "title": at[2][self.lang], "at": at[1]}
        return out

    def _windows(self) -> dict:
        if "windows" not in self.cache:
            self.cache["windows"] = life.week_windows(profile=self.profile(), policy=self.P, week=self.S["week"])
        return self.cache["windows"]

    def _wd(self, iso: str) -> str:
        return WD_SHORT[self.lang][date.fromisoformat(iso).weekday()]

    def _win_text(self, days: list[dict]) -> str:
        groups: dict[str, list[str]] = {}
        for d in days:
            key = ", ".join(f"{w['from']}–{w['to']}" for w in d["windows"])
            if key:
                groups.setdefault(key, []).append(self._wd(d["date"]))
        if not groups:
            return {"ru": "нет окон", "en": "no windows"}[self.lang]
        return "; ".join(f"{', '.join(v)} {k}" for k, v in groups.items())

    def _return_h(self, weekday: str) -> float:
        learned = self.c.extra.get("return_by_weekday", {})
        if weekday in learned:
            return float(learned[weekday])
        return float(np.mean([r["hour"] for r in self.S["week"].raw("returns_log")]))

    def _home(self, return_h: float) -> dict:
        return life.home_schedule(listing=self.rt.listings[self.c.extra.get("listing", "apt_west")], profile=self.profile(),
                                  climate=self.S["climate"], policy=self.P, month=self.P.get("scenario.week_month"),
                                  return_h=return_h, seed=self.rt.seed)

    def m_sun_plan(self) -> dict:
        w = self._windows()
        er = life.errands(week=self.S["week"], queues=self.S["queues"], city=self.S["city"], policy=self.P)
        hm = self._home(self._return_h("mon"))
        if any(a["calculation_status"] != "computed" for a in (w, er, hm)):
            return self.refused_reply(next(a for a in (w, er, hm) if a["calculation_status"] != "computed"), "sun_plan")
        e, h = er["value"], hm["value"]
        son = w["value"]["people"].get("son", [])
        tasks = {t["id"]: t for t in self.S["week"].raw("errands")}
        order = " → ".join(self.t(tasks[i]) for i in e["order_ids"])
        cal, _ = self.propose("calendar_event", scope="family",
                              facts={"start": f"{e['date']}T{e['start_at']}", "duration_min": int(e["chain_min"]) + 60,
                                     "title": order}, title=f"{self._wd(e['date'])} {e['start_at']}: {order}",
                              when=f"{e['date']}T{e['start_at']}", traces=[er["trace_id"]])
        cool, auto = self.propose("precool_home", scope="housing",
                                  facts={"start_at": h["start_at"], "return_at": h["return_at"], "target_c": h["target_c"],
                                         "days": "mon-fri"},
                                  title={"ru": f"охлаждение по будням с {h['start_at']}",
                                         "en": f"weekday cooling from {h['start_at']}"}[self.lang],
                                  traces=[hm["trace_id"]])
        text = self.r("week_ready", adam_days=self._win_text(son), errand_day=self._wd(e["date"]), errand_at=e["start_at"],
                      n=len([p for p in self.c.pending.values() if p.get("kind", "action") == "action"]))
        card = {"type": "week", "days": [{"date": d["date"], "weekday": self._wd(d["date"]),
                                          "people": {self.name(pid): next(x["windows"] for x in v if x["date"] == d["date"])
                                                     for pid, v in w["value"]["people"].items()},
                                          "family": d["windows"]} for d in w["value"]["family"]],
                "errands": {**e, "order": order}, "home": h, "limits_pm10": w["value"]["limits_pm10"]}
        return self.say([text, auto], cards=[card, {"type": "proposal", **cal}, {"type": "proposal", **cool}],
                        traces=[w["trace_id"], er["trace_id"], hm["trace_id"]], route="sun_plan")

    def _week_day(self, weekday: int | None = None) -> tuple[str, bool]:
        """A forecast day of the week: today if it is in the forecast (and the weekday matches), otherwise the nearest match."""
        days = [d["date"] for d in self.S["week"].raw("forecast.days")]
        today = self.now.date().isoformat()
        if today in days and (weekday is None or self.now.weekday() == weekday):
            return today, True
        pick = [d for d in days if weekday is None or date.fromisoformat(d).weekday() == weekday]
        later = [d for d in pick if d >= today]
        return (later or pick or days)[0], False

    def m_mon_morning(self, day: str | None = None) -> dict:
        is_today = True
        if day is None:
            day, is_today = self._week_day()
            if not is_today and date.fromisoformat(day).weekday() >= 5:
                day, is_today = self._week_day(0)
        mp = life.morning_plan(profile=self.profile(), city=self.S["city"], policy=self.P, week=self.S["week"], day=day,
                               district=self.c.extra.get("district", "A"), seed=self.rt.seed)
        if mp["calculation_status"] != "computed":
            return self.refused_reply(mp, "morning")
        v = mp["value"]
        dep = {d["trip"]: d for d in v["departures"]}
        s, o = dep["school_am"], dep.get("office_am")
        general = float(self.P.get("outdoor_limits.pm10_ugm3_max"))
        dust = self.r("dust_line", at=v["dust_rises_at"]) if v["dust_rises_at"] else ""
        today = v["outdoor_today"]
        kids = ""
        if not today.get("son") and today.get("daughter"):
            ev = [x for x in today["daughter"] if x["from"] >= "15:00"] or today["daughter"]
            kids = self.r("kids_line", d_from=ev[0]["from"])
        text = self.r("morning", best=s["best"]["depart"], usual=s["usual"]["depart"], by=s["arrive_by"],
                      p_usual=A.n0(100 * s["usual"]["p_on_time"]),
                      l_best=o["best"]["depart"] if o else "—", l_best_min=A.n0(o["best"]["drive_min"]["mid"]) if o else "—",
                      l_usual_min=A.n0(o["usual"]["drive_min"]["mid"]) if o else "—", dust_line=dust, kids_line=kids)
        if not is_today:
            text = {"ru": f"На {fmt_date(day, 'ru')} ({self._wd(day)}). ", "en": f"For {fmt_date(day, 'en')} ({self._wd(day)}). "}[self.lang] + text
        card = {"type": "morning", "date": day, "departures": [{"trip": d["trip"], "arrive_by": d["arrive_by"],
                                                                 "best": d["best"], "usual": d["usual"],
                                                                 "options": [{"depart": x["depart"], "p": x["p_on_time"],
                                                                              "drive": x["drive_min"]["mid"]} for x in d["options"]]}
                                                                for d in v["departures"]],
                "pm10_by_hour": v["pm10_by_hour"], "limits_pm10": {self.name(k): x for k, x in v["limits_pm10"].items()},
                "general_limit": general, "outdoor_today": {self.name(k): x for k, x in today.items()}}
        self.c.extra["topic"] = "morning"
        return self.say(text, cards=[card], traces=[mp["trace_id"]], route="morning")

    def m_mon_home(self, return_h: float | None = None, late: bool = False) -> dict:
        wd = WD[self.now.weekday()] if self.now.weekday() < 5 else "mon"
        rh = return_h if return_h is not None else self._return_h(wd)
        hm = self._home(rh)
        if hm["calculation_status"] != "computed":
            return self.refused_reply(hm, "home")
        h = hm["value"]
        item, auto = self.propose("precool_home", scope="housing",
                                  facts={"start_at": h["start_at"], "return_at": h["return_at"], "target_c": h["target_c"],
                                         "date": self.now.date().isoformat()},
                                  title={"ru": f"охлаждение с {h['start_at']} к {h['return_at']}",
                                         "en": f"cooling from {h['start_at']} for {h['return_at']}"}[self.lang],
                                  traces=[hm["trace_id"]])
        key = "home_late" if late else "home_start"
        text = self.r(key, start=h["start_at"], ret=h["return_at"], target=A.n1(h["target_c"]))
        self.c.extra["topic"] = "home"
        return self.say([text, auto], cards=[{"type": "home", **h}, {"type": "proposal", **item}],
                        traces=[hm["trace_id"]], route="home")

    def m_tue_errands(self) -> dict:
        er = life.errands(week=self.S["week"], queues=self.S["queues"], city=self.S["city"], policy=self.P)
        if er["calculation_status"] != "computed":
            return self.refused_reply(er, "errands")
        e = er["value"]
        tasks = {t["id"]: t for t in self.S["week"].raw("errands")}
        order = " → ".join(self.t(tasks[i]) for i in e["order_ids"])
        return self.say(self.r("errands", order=order, start=e["start_at"], separate=A.n0(e["separate_min"]),
                               n_sep=e["trips"]["separate"], chain=A.n0(e["chain_min"])),
                        cards=[{"type": "errands", **e, "order": order}], traces=[er["trace_id"]], route="errands")

    def m_wed_windows(self) -> dict:
        w = self._windows()
        v = w["value"]
        text = self.r("windows", daughter=self._win_text(v["people"].get("daughter", [])),
                      adam=self._win_text(v["people"].get("son", [])), family=self._win_text(v["family"]))
        return self.say(text, cards=[{"type": "windows", "people": {self.name(k): x for k, x in v["people"].items()},
                                      "family": v["family"], "limits_pm10": v["limits_pm10"]}],
                        traces=[w["trace_id"]], route="windows")

    def _conflict(self) -> dict:
        return life.car_conflict(week=self.S["week"], city=self.S["city"], policy=self.P,
                                 district=self.c.extra.get("district", "A"), lam=self.c.model.lambda_mean(), seed=self.rt.seed)

    def m_wed_reminder(self) -> dict:
        cf = self._conflict()
        if cf["calculation_status"] != "computed" or not cf["value"].get("conflict"):
            return self.say(self.r("greeting"), route="reminder")
        self.c.extra["topic"] = "conflict"
        return self.say(self.r("reminder_conflict", at=cf["value"]["at"]), cards=[{"type": "reminder", "at": cf["value"]["at"],
                                                                                  "date": cf["value"]["date"]}],
                        traces=[cf["trace_id"]], route="reminder",
                        quick=[{"ru": "Покажи варианты", "en": "Show options"}[self.lang]])

    def m_thu_conflict(self) -> dict:
        cf = self._conflict()
        if cf["calculation_status"] != "computed":
            return self.refused_reply(cf, "conflict")
        v = cf["value"]
        cal = {e["id"]: e for e in self.S["week"].raw("calendar")}
        mv = cal[v["movable_id"]]
        rec = {"move": {"ru": "перенести встречу", "en": "moving the meeting"},
               "taxi": {"ru": "такси", "en": "the taxi"}}[v["recommended"]][self.lang]
        text = self.r("conflict", at=v["at"], taxi=A.n0(v["taxi"]["cost_aed"]["mid"]),
                      wait=A.n0(v["taxi"]["wait_outside_min"]["mid"]), meeting=self.t(mv), old=v["move"]["from"],
                      new=v["move"]["to"], new_drive=A.n0(v["move"]["drive_new_min"]["mid"]),
                      old_drive=A.n0(v["move"]["drive_now_min"]["mid"]), rec=rec)
        cards = [{"type": "conflict", **v, "meeting": self.t(mv)}]
        if v["recommended"] == "move":
            d = self.draft("business", "investor_reschedule", {"old_time": v["move"]["from"], "new_time": v["move"]["to"]},
                           traces=[cf["trace_id"]])
            if d:
                cards.append({"type": "draft", **d})
            item, auto = self.propose("calendar_event", scope="business",
                                      facts={"start": f"{v['date']}T{v['move']['to']}", "duration_min": 60,
                                             "title": self.t(mv)},
                                      title=f"{self.t(mv)} → {v['move']['to']}", when=f"{v['date']}T{v['move']['to']}",
                                      traces=[cf["trace_id"]])
            cards.append({"type": "proposal", **item})
        self.c.extra["topic"] = "conflict"
        return self.say(text, cards=cards, traces=[cf["trace_id"]], route="conflict")

    def car_taxi(self, money_first: bool = False) -> dict:
        r = life.car_vs_taxi(profile=self.profile(), city=self.S["city"], climate=self.S["climate"], policy=self.P,
                             district=self.c.extra.get("district", "A"), lam=self.c.model.lambda_mean(), seed=self.rt.seed,
                             money_first=money_first)
        if r["calculation_status"] != "computed":
            return self.refused_reply(r, "car_taxi")
        v = r["value"]
        self.c.extra["topic"] = "car"
        if money_first:
            text = self.r("car_taxi_money", adam_rules=A.n0(v["year"]["taxi_rules"]["adam_harsh_min"]["mid"]),
                          adam_taxi=A.n0(v["year"]["taxi"]["adam_harsh_min"]["mid"]))
        else:
            d = v["diff_car_minus_taxi"]
            word = {"ru": ("дороже", "дешевле"), "en": ("more", "less")}[self.lang][0 if d["money_aed"] > 0 else 1]
            rec = {"car": {"ru": "вторая машина", "en": "a second car"}, "taxi": {"ru": "такси", "en": "taxis"},
                   "taxi_rules": {"ru": "такси с правилами", "en": "taxis with rules"}}[v["recommended"]][self.lang]
            text = self.r("car_taxi", money_word=word, money=A.n0(abs(d["money_aed"])), wait=A.n0(abs(d["waiting_h"])),
                          adam=A.n0(abs(d["adam_harsh_min"])), rec=rec)
        return self.say(text, cards=[{"type": "car_taxi", **v}], traces=[r["trace_id"]], route="car_taxi",
                        quick=[] if money_first else [{"ru": "А если деньги важнее?", "en": "What if money matters more?"}[self.lang]])

    def license(self) -> dict:
        p = self._plan()
        if p["calculation_status"] != "computed":
            return self.refused_reply(p, "license")
        start = date.fromisoformat(self.P.get("scenario.today"))
        lic = next((s for s in p["value"]["steps"] if s["id"] == "company_license"), None)
        if lic is None:
            return self.say(self.r("dont_know", missing="company_license step (not a founder)"), route="license")
        lag = float(np.median(sample(self.P.raw("business.hire_lag_after_license_days"), np.random.default_rng(self.rt.seed),
                                     int(self.P.get("mc.runs")))))
        ld = start + timedelta(days=float(lic["finish_day"]["mid"]))
        hd = ld + timedelta(days=lag)
        d = self.draft("business", "vacancy", {"hire_mid": fmt_date(hd, self.lang)}, traces=[p["trace_id"]])
        return self.say(self.r("license", license=fmt_date(ld, self.lang), hire=fmt_date(hd, self.lang)),
                        cards=[{"type": "license", "license_date": ld.isoformat(), "hire_date": hd.isoformat(),
                                "hire_lag_days": lag}] + ([{"type": "draft", **d}] if d else []),
                        traces=[p["trace_id"]], route="license")

    def m_fri_car(self) -> dict:
        a = self.car_taxi()
        b = self.license()
        a["text"] = a["text"] + "\n\n" + b["text"]
        a["cards"] += b["cards"]
        a["trace_ids"] += b["trace_ids"]
        a["pending"] = b["pending"]
        self.c.conversation = self.c.conversation[:-2]
        self.c.say("cure", a["text"], at=self.now, trace_ids=a["trace_ids"])
        self.save()
        return a

    def m_sat_beach(self) -> dict:
        w = self._windows()
        day, is_today = self._week_day(5)
        fam = next((d for d in w["value"]["family"] if d["date"] == day), {"windows": []})
        now_hm = self.now.strftime("%H:%M") if is_today else "00:00"
        nxt = [x for x in fam["windows"] if x["to"] > now_hm and x["from"] >= now_hm] or \
            [x for x in fam["windows"] if x["to"] > now_hm]
        if not nxt:
            return self.say({"ru": "Сегодня общего окна для улицы нет.", "en": "No shared outdoor window today."}[self.lang],
                            traces=[w["trace_id"]], route="beach")
        win = nxt[0]
        rem_min = int(win["from"][:2]) * 60 + int(win["from"][3:]) - int(self.P.get("life.pack_reminder_min"))
        remind = life.hhmm(rem_min)
        item, auto = self.propose("reminder", scope="family", facts={"at": f"{day}T{remind}", "text": "beach"},
                                  title={"ru": f"напомнить в {remind}: собираться на пляж",
                                         "en": f"remind at {remind}: pack for the beach"}[self.lang],
                                  when=f"{day}T{remind}", traces=[w["trace_id"]])
        when = "" if is_today else {"ru": f"{fmt_date(day, 'ru')} ({self._wd(day)}): ", "en": f"{fmt_date(day, 'en')} ({self._wd(day)}): "}[self.lang]
        return self.say([when + self.r("beach", window=f"{win['from']}–{win['to']}", remind=remind), auto],
                        cards=[{"type": "beach", "window": win, "remind": remind}, {"type": "proposal", **item}],
                        traces=[w["trace_id"]], route="beach")

    def m_sat_guest(self) -> dict:
        g = life.guest_arrival(week=self.S["week"], profile=self.profile(), policy=self.P, climate=self.S["climate"],
                               scenario=self.S["vision_scenario"], listing=self.rt.listings[self.c.extra.get("listing", "apt_west")],
                               seed=self.rt.seed)
        if g["calculation_status"] != "computed":
            return self.refused_reply(g, "guest")
        v = g["value"]
        heat = max(r["heat_min_without_cure"]["mid"] for r in v["rows"])
        self.c.extra["topic"] = "guest"
        self.cache["guest"] = g
        return self.say(self.r("guest", best=v["best_hour"], heat=A.n0(heat)), cards=[{"type": "guest", **v}],
                        traces=[g["trace_id"]], route="guest")

    def guest_constraint(self, text: str) -> dict:
        split = split_voice(text, self.P)
        rules = self.P.get("privacy.derived_constraints")
        derived = {}
        for frag in split["device_only"]:
            for rule in rules:
                if any(re.search(t, frag, re.I) for t in rule["terms"]):
                    derived.update(rule["constraints"])
            self.c.sealed.setdefault("guest", []).append({"text": frag, "at": self.now.isoformat()})
        if derived:
            self.c.constraints.setdefault("guest", {}).update(derived)
        self.c.remember(at=self.now, kind="sealed", text="guest: 1 sealed note", source="you")
        best = (self.cache.get("guest") or {}).get("value", {}).get("best_hour", "—")
        return self.say(self.r("guest_saved", best=best), cards=[{"type": "constraints", "who": "guest", "derived": derived,
                                                                   "sealed_items": len(split["device_only"])}],
                        route="guest_constraint")

    def m_month_end(self) -> dict:
        b = life.budget_month(month_key="2026-11", profile=self.profile(), city=self.S["city"], climate=self.S["climate"],
                              tariffs=self.S["tariffs"], policy=self.P, listings=self.rt.listings,
                              district=self.c.extra.get("district", "A"), week=self.S["week"], seed=self.rt.seed)
        if b["calculation_status"] != "computed":
            return self.refused_reply(b, "budget")
        v = b["value"]
        under = v["total_diff"] < 0
        reserve = v["summer_reserve"]
        item, auto = self.propose("budget_rule", scope="finance",
                                  facts={"reserve_aed": round(reserve), "for": "summer cooling", "moves_money": False},
                                  title={"ru": f"правило: отложить {A.n0(reserve)} AED на летнее охлаждение",
                                         "en": f"rule: set aside {A.n0(reserve)} AED for summer cooling"}[self.lang],
                                  traces=[b["trace_id"]])
        text = self.r("budget", month={"ru": "Ноябрь", "en": "November"}[self.lang], diff=A.n0(abs(v["total_diff"])),
                      diff_word=BUDGET_WORDS[self.lang][0 if under else 1],
                      main=CATEGORY[v["main_saving"]][self.lang], summer=A.n0(reserve))
        return self.say([text, auto], cards=[{"type": "budget", **v}, {"type": "proposal", **item}],
                        traces=[b["trace_id"]], route="budget")

    def m_month_later(self) -> dict:
        hb = life.habits(week=self.S["week"], policy=self.P)
        if hb["calculation_status"] != "computed":
            return self.refused_reply(hb, "habits")
        parts, cards = [], [{"type": "habits", **hb["value"]}]
        for f in hb["value"]["found"]:
            hid = f"habit-{f['kind']}"
            if f["kind"] == "return_shift":
                wdn = A.WEEKDAY_NAMES[self.lang][f["weekday"]]
                parts.append(self.r("habit_return", weekday=wdn, at=f["at"], usual=f["usual"]))
                title = {"ru": f"учесть: по {wdn} дома в {f['at']}", "en": f"learn: home at {f['at']} on {wdn}s"}[self.lang]
            else:
                parts.append(self.r("habit_time", n=f["observations"]))
                title = {"ru": "учесть: время важнее", "en": "learn: time matters more"}[self.lang]
            self.c.pending[hid] = {"id": hid, "kind": "model", "action": "model_update", "title": title, "facts": f,
                                   "when": None, "status": "needs_you"}
        return self.say(parts, cards=cards, traces=[hb["trace_id"]], route="habits")

    def apply_habit(self, hid: str, approved: bool) -> dict:
        item = self.c.pending.pop(hid)
        f = item["facts"]
        if not approved:
            return self.say(self.r("declined", title=item["title"]).split(".")[0] + ".", route="habit")
        if f["kind"] == "return_shift":
            h, m = f["at"].split(":")
            self.c.extra.setdefault("return_by_weekday", {})[f["weekday"]] = int(h) + int(m) / 60
            self.c.remember(at=self.now, kind="habit", text=item["title"], source="you (approved)")
            return self.say(self.r("habit_return_applied", weekday=A.WEEKDAY_NAMES[self.lang][f["weekday"]], at=f["at"]),
                            route="habit")
        # time matters more: each logged choice is a choice in the personal model (axes: money, time, comfort)
        for ch in self.S["week"].raw("choices_log"):
            if ch.get("peak") and ch["chosen"] == "taxi":
                self.c.model.observe_choice(ch["options"], ch["chosen"], at=self.now.isoformat())
        self.cache.pop("cost_map", None)
        self.c.remember(at=self.now, kind="habit", text=item["title"], source="you (approved)")
        lam = self.c.model.lambda_mean()
        return self.say(self.r("habit_applied", m=f"{lam['money']:.2f}", t=f"{lam['time']:.2f}", c=f"{lam['comfort']:.2f}"),
                        cards=[{"type": "model", "model": self.c.model.summary()}], route="habit")

    # ------------------------------------------------------------------ ad-hoc questions
    def impact(self) -> dict:
        """Result: model comparison with the usual order (steps in list order, visits at a random hour)."""
        from .plan import impact
        r = impact(profile=self.profile(), steps=self.S["steps"], queues=self.S["queues"], policy=self.P,
                   seed=self.rt.seed, learned=self.rt.teacher.learned())
        if r["calculation_status"] != "computed":
            return self.refused_reply(r, "impact")
        d = r["value"]["days_saved"]
        return self.say({"ru": f"Модельное сравнение: с CURE семья готова раньше на {A.n1(d['mid'])} дн. (от {A.n1(d['low'])} до {A.n1(d['high'])}), чем по списку «как обычно». Данные синтетические.",
                         "en": f"Model comparison: with CURE the family is ready {A.n1(d['mid'])} days earlier (between {A.n1(d['low'])} and {A.n1(d['high'])}) than with the usual checklist. Synthetic data."}[self.lang],
                        cards=[{"type": "impact", **r["value"]}], traces=[r["trace_id"]], route="impact")

    def setpoint(self, value: float) -> dict:
        self.c.model.observe_setpoint(value, at=self.now.isoformat())
        s = self.c.model.summary()["setpoint_c"]
        return self.say({"ru": f"Запомнил: {A.n1(value)}°. Ваша уставка теперь около {A.n1(s['mid'])}°.",
                         "en": f"Noted: {A.n1(value)}°. Your setpoint is now about {A.n1(s['mid'])}°."}[self.lang],
                        cards=[{"type": "model", "model": self.c.model.summary()}], route="setpoint")

    def realtime(self) -> dict:
        h = self.now.hour
        day = next((d for d in self.S["week"].raw("forecast.days") if d["date"] == self.now.date().isoformat()), None)
        if day is None:
            return self.say(self.r("dont_know", missing=f"forecast {self.now.date().isoformat()}"), route="realtime")
        return self.say(self.r("realtime", h=f"{h:02d}", pm=A.n0(day["pm10_ugm3"][h])), route="realtime")

    def future_rent(self) -> dict:
        r = refused(MissingData(["price_dynamics:district_rent_history", "mortgage_rates:scenarios"]), what="rent_vs_buy_5y")
        return self.say(self.r("dont_know", missing=", ".join(r["missing"])), cards=[{"type": "refused", "answer": r}],
                        traces=[r["trace_id"]], route="future_rent")

    def diagnosis(self) -> dict:
        from .total_cost import focus_person
        c = self.c.constraints.get(focus_person(self.profile()), {})
        return self.say(self.r("diagnosis_refuse", pm=A.n0(c.get("pm10_ugm3_max", self.P.get("outdoor_limits.pm10_ugm3_max"))),
                               mins=A.n0(c.get("outdoor_harsh_min_max", 0))), route="diagnosis")

    # ------------------------------------------------------------------ memory
    def memory(self) -> dict:
        lam = self.c.model.summary()
        return {"resident": self.rid, "stored_where": "CURE core · personal contour · encrypted file (Fernet)",
                "on_disk_looks_like": self.rt.store.ciphertext_preview(self.rid),
                "devices": self.c.devices, "model": lam, "facts": self.c.facts[-40:],
                "sealed": [{"who": self.name(k) if k != "guest" else k, "items": len(v),
                            "note": "sealed · never sent to the language model or the city"} for k, v in self.c.sealed.items()],
                "constraints": {self.name(k) if k != "guest" else k: v for k, v in self.c.constraints.items()},
                "left_the_contour": self.c.outbox[-30:], "requests_to_my_data": self.c.access_log[-30:],
                "effects": self.c.effects[-30:], "pending": self.pending_view(), "drafts": self.c.extra.get("drafts", [])[-10:],
                "trust": self.trust_view(), "trust_history": self.c.trust.history[-20:],
                "consents": self.door().consents_view(self.now), "district": self.c.extra.get("district"),
                "arrival": self.c.extra.get("arrival"), "overrides": self.c.overrides}

    def what_i_know(self) -> dict:
        lam = self.c.model.lambda_mean()
        n = sum(len(v) for v in self.c.sealed.values())
        sealed = {"ru": f"{n} записи", "en": f"{n} item(s)"}[self.lang]
        text = self.r("memory_summary", people=len(self.profile().raw("household")),
                      district=self.c.extra.get("district") or "—", m=f"{lam['money']:.2f}", t=f"{lam['time']:.2f}",
                      c=f"{lam['comfort']:.2f}", sp=A.n1(self.c.model.summary()["setpoint_c"]["mid"]), sealed=sealed,
                      out=len(self.c.outbox))
        return self.say(text, cards=[{"type": "memory", **self.memory()}], route="memory")

    def revoke_consent(self, purpose: str) -> dict:
        self.door().revoke(purpose, now=self.now)
        if purpose not in self.c.revoked:
            self.c.revoked.append(purpose)
        self.cache.clear()
        return self.say({"ru": f"Согласие «{purpose}» отозвано. Считать и передавать по нему больше нельзя.",
                         "en": f"Consent '{purpose}' revoked. Nothing is computed or shared under it any more."}[self.lang],
                        cards=[{"type": "consents", "consents": self.door().consents_view(self.now)}], route="consent")

    def grant_consent(self, purpose: str) -> dict:
        self.door().grant(purpose, now=self.now)
        self.c.revoked = [item for item in self.c.revoked if item != purpose]
        self.cache.clear()
        return self.say({"ru": f"Согласие «{purpose}» выдано заново.",
                         "en": f"Consent '{purpose}' granted again."}[self.lang],
                        cards=[{"type": "consents", "consents": self.door().consents_view(self.now)}], route="consent")

    # ------------------------------------------------------------------ vision: the city (nothing is executed)
    def _preload(self, h: int) -> dict:
        return preload(listing=self.rt.listings[self.c.extra.get("listing", "apt_west")], profile=self.profile(),
                       climate=self.S["climate"], policy=self.P, month=self.S["vision_scenario"].get("arrival.month"),
                       arrival_h=h % 24, seed=self.rt.seed)

    def _cohort_preload(self, h: int) -> dict:
        """Cohort capacity uses the sourced population prior, never the requesting resident's private profile."""
        reference = Store({
            "meta": {"data_mode": self.P.data_mode, "source_kind": "policy", "source": "setpoint population prior"},
            "params": {"setpoint_c": {"value": self.P.get("personal_params.setpoint_prior.mean"),
                                         "source": "policy:personal_params.setpoint_prior.mean"}},
        }, "cohort_reference")
        return preload(listing=self.rt.listings["apt_west"], profile=reference, climate=self.S["climate"], policy=self.P,
                       month=self.S["vision_scenario"].get("arrival.month"), arrival_h=h % 24, seed=self.rt.seed)

    def vision_request(self) -> dict:
        if not self._consent("arrival_vision"):
            return self.say(self.r("dont_know", missing="consent:arrival_vision"), route="vision_request")
        req = city_request(profile=self.profile(), policy=self.P)
        self.c.outbox.append({"at": self.now.isoformat(), "to": "city CURE", "what": req})
        c = req["constraints"]
        return self.say(self.r("vision_request", p=req["pseudonym"], pm=A.n0(c["pm10_ugm3_max"]), mins=A.n0(c["outdoor_min_max"]),
                               t=A.n1(c["indoor_c_max_on_arrival"])), cards=[{"type": "city_request", **req}],
                        route="vision_request")

    def vision_chain(self, delay_h: int = 0) -> dict:
        h = int(self.S["vision_scenario"].get("arrival.arrival_h")) + delay_h
        pre = self._preload(h)
        cf = arrival_counterfactual(profile=self.profile(), policy=self.P, climate=self.S["climate"],
                                    scenario=self.S["vision_scenario"], seed=self.rt.seed, delay_h=delay_h)
        ch = run_chain(profile=self.profile(), policy=self.P, scenario=self.S["vision_scenario"], preload_answer=pre,
                       consent_valid=self._consent("arrival_vision"))
        if cf["calculation_status"] != "computed" or ch["calculation_status"] != "computed":
            return self.refused_reply(cf if cf["calculation_status"] != "computed" else ch, "vision_chain")
        links = ch["value"]["links"]
        ok = sum(1 for l in links if l["status"] == "ok")
        v = cf["value"]
        text = self.r("vision_chain", h=f"{h % 24:02d}", kid=self.kid(), without=A.n0(v["without"]["outdoor_min"]["mid"]),
                      dust=A.n0(v["without"]["dust_min"]["mid"]), with_=A.n0(v["with"]["outdoor_min"]["mid"]), ok=ok, n=len(links))
        return self.say(text, cards=[{"type": "chain", "delay_h": delay_h, "links": links, "arrival": v,
                                      "preload": pre.get("value"), "executed": False}],
                        traces=[pre.get("trace_id"), cf["trace_id"], ch["trace_id"]], route="vision_chain")

    def vision_storm(self) -> dict:
        pre = self._preload(int(self.S["vision_scenario"].get("arrival.arrival_h")))
        r = storm(profile=self.profile(), policy=self.P, climate=self.S["climate"], scenario=self.S["vision_scenario"],
                  rates=self.S["rates"], preload_answer=pre, agents=self.agents(), seed=self.rt.seed, now=self.now,
                  consent_valid=self._consent("arrival_vision"))
        if "chain" not in r:
            return self.refused_reply(r, "vision_storm")
        return self.say(self.r("vision_storm", pm=A.n0(r["constraint_kept"]["pm10_ugm3_max"])),
                        cards=[{"type": "storm", "links": r["chain"]["value"]["links"], "av": r["chain"]["value"]["av_allowed"],
                                "agents_mode": r["agents_mode"], "arbitration": r["arbitration"].get("value"),
                                "constraint_kept": r["constraint_kept"]}],
                        traces=[r["chain"]["trace_id"], r["arbitration"].get("trace_id")], route="vision_storm")

    def vision_neighbors(self, emergency: bool = False) -> dict:
        r = neighbors(policy=self.P, scenario=self.S["vision_scenario"], emergency=emergency)
        if r["calculation_status"] != "computed":
            return self.refused_reply(r, "vision_neighbors")
        return self.say(self.r("vision_neighbors", chosen=r["value"]["chosen_note_ru" if self.lang == "ru" else "chosen_note"]), cards=[{"type": "neighbors", **r["value"]}],
                        traces=[r["trace_id"]], route="vision_neighbors")

    def city_family(self) -> dict:
        """The family record for the city: constraints only, no diagnosis, no household."""
        cons = self.c.constraints
        pm = [v["pm10_ugm3_max"] for v in cons.values() if v.get("pm10_ugm3_max") is not None]
        harsh = [v["outdoor_harsh_min_max"] for v in cons.values() if v.get("outdoor_harsh_min_max") is not None]
        parsed = self.c.extra.get("parsed", {})
        return {"id": self.profile().get("pseudonym"), "who": self.c.extra.get("city_label", "new family"),
                "who_ru": self.c.extra.get("city_label_ru", self.c.extra.get("city_label", "new family")),
                "arrival_day": int(self.c.extra.get("city_day", 2)), "arrival_h": 23 if parsed.get("night") else 20,
                "limits": {"pm10_ugm3_max": min(pm) if pm else None, "outdoor_harsh_min_max": min(harsh) if harsh else None,
                           "accessible_vehicle": any(v.get("accessible_vehicle") for v in cons.values())},
                "slowest_member": {"mobility_aid": max([int(v.get("mobility_aid", 0)) for v in cons.values()] or [0]),
                                   "heavy_items": 0}}

    def add_to_city(self) -> dict:
        if not self._consent("arrival_vision"):
            return self.say(self.r("dont_know", missing="consent:arrival_vision"), route="add_to_city")
        fam = self.city_family()
        self.rt.city["extra_families"][self.rid] = fam
        self.c.outbox.append({"at": self.now.isoformat(), "to": "city CURE", "what": {"limits": fam["limits"]}})
        n = len(self.S["vision_scenario"].raw("cohort.families")) + len(self.rt.city["extra_families"])
        sent = ", ".join(f"{k}={v}" for k, v in fam["limits"].items() if v not in (None, False)) or "—"
        return self.say(self.r("city_added", n=n, sent=sent), cards=[{"type": "city_family", **fam}], route="add_to_city")

    def vision_cohort(self, stress: bool | None = None, speak: bool = True):
        if stress is not None:
            self.rt.city["stress"] = stress
        sc = self.S["vision_scenario"]
        hours = sc.get("cohort.arrival_hours")
        extra = list(self.rt.city["extra_families"].values())
        r = cohort(policy=self.P, scenario=sc, climate=self.S["climate"], queues=self.S["queues"],
                   cohort_store=self.S["cohort"], preload_by_hour={h: self._cohort_preload(h) for h in hours},
                   extra_families=extra, stress=self.rt.city["stress"], extra_ramp_cars=self.rt.city["ramp_cars"])
        if r["calculation_status"] != "computed":
            return self.refused_reply(r, "cohort") if speak else ""
        v = r["value"]
        text = self.r("cohort", n=v["families"], kept_with=v["limits"]["kept"]["with"], limited=v["limits"]["families_with_personal_limit"],
                      kept_without=v["limits"]["kept"]["without"], w_without=v["centers"]["wasted_trips"]["without"],
                      w_with=v["centers"]["wasted_trips"]["with"])
        self.cache["cohort"] = r
        if not speak:
            return text
        parts, cards = [text], [{"type": "cohort", **v}]
        if v.get("proposal"):
            item, auto = self.propose("dispatch_ramp_car", scope="family", facts={"waiting": v["proposal"]["families_waiting"]},
                                      title={"ru": "вторая машина с рампой из депо", "en": "second ramp car from the depot"}[self.lang],
                                      traces=[r["trace_id"]])
            parts.append(self.r("ramp_needed", who=", ".join(v["proposal"]["families_waiting" + ("_ru" if self.lang == "ru" else "")])))
            cards.append({"type": "proposal", **item})
        return self.say(parts, cards=cards, traces=[r["trace_id"]], route="cohort")

    def vision_learning(self) -> dict:
        r = learning_view(self.rt.teacher, steps=self.S["steps"], policy=self.P)
        row = max(r["value"]["steps"], key=lambda x: x["facts"])
        return self.say(self.r("learning", step=self._step_title(row["step_id"]), n=row["facts"],
                               share=A.n0(100 * row["share_from_facts"])),
                        cards=[{"type": "learning", **r["value"]}], traces=[r["trace_id"]], route="vision_learning")

    # ------------------------------------------------------------------ new resident (a judge instead of Leila)
    def newcomer_day(self) -> dict:
        month = self.c.extra.get("parsed", {}).get("month") or self.P.get("scenario.week_month")
        ds = list(self.S["city"].raw("rent_aed_month"))
        d = simulate_day(profile=self.profile(), city=self.S["city"], climate=self.S["climate"], policy=self.P,
                         districts=ds, month=month, seed=self.rt.seed)
        if d["calculation_status"] != "computed":
            return self.refused_reply(d, "newcomer_day")
        from .total_cost import focus_person
        f = focus_person(self.profile())
        lines = [self.r("newcomer_intro", month=A.MONTHS[self.lang][month])]
        for k in ds:
            lines.append(self.r("day_line", d=k, transit=A.n0(d["value"][k]["transit_min"]["mid"]), kid=self.kid(),
                                harsh=A.n0(d["value"][k]["harsh_outdoor_min"][f]["mid"])))
        return self.say(lines, cards=[{"type": "day_compare", "month": month, "focus": self.kid(),
                                       "districts": {k: {"transit_min": d["value"][k]["transit_min"]["mid"],
                                                         "harsh_min": d["value"][k]["harsh_outdoor_min"][f]["mid"]}
                                                     for k in ds}}], traces=[d["trace_id"]], route="newcomer_day")

    def delete(self) -> dict:
        self.rt.city["extra_families"].pop(self.rid, None)
        text = self.r("deleted")
        self.rt.store.delete(self.rid)
        self.rt.sessions.pop(self.rid, None)
        self.c.extra.clear()
        self.c.facts.clear()
        self.c.conversation.clear()
        return {"route": "deleted", "text": text, "lang": self.lang, "cards": [{"type": "deleted", "resident": self.rid,
                                                                                "file_exists": self.rt.store.exists(self.rid)}],
                "trace_ids": [], "quick": [], "clock": self.now.isoformat(), "pending": []}


# ---------------------------------------------------------------------- door: contour creation
WHO_RX = [("son", r"\bсын|\bson\b"), ("daughter", r"\bдоч|\bdaughter"), ("husband", r"\bмуж|\bhusband"),
          ("wife", r"\bжен[аы]\b|\bwife"), ("mother", r"\bмам|\bmother|\bmom\b"), ("self", r"\bу меня\b|\bi have\b|\bmy\b")]
NAMES = {"son": {"ru": "сын", "en": "your son"}, "daughter": {"ru": "дочь", "en": "your daughter"},
         "child": {"ru": "ребёнок", "en": "your child"}, "you": {"ru": "вы", "en": "you"},
         "partner": {"ru": "партнёр", "en": "your partner"}}


def derive(fragments: list[str], policy: Store, household_ids: list[str], text: str) -> tuple[dict, dict]:
    """Health fragments → (sealed domain, derived constraints). The diagnosis never enters the constraints."""
    sealed, cons = {}, {}
    rules = policy.get("privacy.derived_constraints")
    for frag in fragments:
        who = next((w for w, rx in WHO_RX if re.search(rx, frag, re.I)), None)
        pid = (who if who in household_ids else
               next((p for p in household_ids if p.startswith("child")), None) if who in ("son", "daughter") else
               household_ids[0])
        sealed.setdefault(pid, []).append({"text": frag})
        for rule in rules:
            if any(re.search(t, frag, re.I) for t in rule["terms"]):
                cons.setdefault(pid, {}).update(rule["constraints"])
    return sealed, cons


def leila_contour(rt: Runtime, voice: str, lang: str, device: str | None) -> Session:
    base = copy.deepcopy(rt.S["leila"].doc)
    split = split_voice(voice, rt.P)
    ids = [p["id"] for p in base["household"]]
    sealed, cons = derive(split["device_only"], rt.P, ids, voice)
    for k in ("device_only", "constraints", "voice_example", "voice_example_ru"):
        base.pop(k, None)
    s = rt.new_session(LEILA, profile_doc=base, sealed={k: [dict(x, at=rt.clock.isoformat()) for x in v]
                                                          for k, v in sealed.items()},
                       constraints={k: {kk: vv for kk, vv in v.items() if kk in ("pm10_ugm3_max", "outdoor_harsh_min_max",
                                                                                    "accessible_vehicle", "mobility_aid")}
                                    for k, v in cons.items()}, lang=lang, device=device)
    s.c.extra["city_label"], s.c.extra["city_label_ru"] = "Leila: four, a child who must avoid dust", "Лейла: четверо, ребёнку нельзя в пыль"
    from .modeling import ModelRegistry
    from .profiles import ProfileService
    ProfileService(ModelRegistry.load_default()).initialize(s, {
        "household_size": len(base["household"]),
        "children_count": sum(p["role"] == "child" for p in base["household"]),
        "is_founder": bool(base["flags"].get("founder")),
        "housing_budget_aed_month": base["params"]["budget_aed_month"]["value"],
        "preferred_setpoint_c": base["params"]["setpoint_c"]["value"],
    }, source="seeded resident profile")
    s.c.extra["registered_profile"] = "leila-v1"
    from addon.life import seed
    seed(s)
    for fact in s.c.facts:
        fact['provenance'] = 'synthetic'
    from .mind import sync_week_events
    sync_week_events(s)
    s.save()
    return s


def newcomer_contour(rt: Runtime, voice: str, lang: str, device: str | None,
                     resident_id: str | None = None) -> Session:
    """A family from a judge's phrase: the profile follows Leila's template (same city synthetic), the household comes from the phrase."""
    split = split_voice(voice, rt.P)
    p = parse_household(voice, infer_defaults=False)
    base = copy.deepcopy(rt.S["leila"].doc)
    base["meta"] = {"data_mode": "synthetic", "source_kind": "user_intake", "source": "new resident intake"}
    base["params"] = {}
    base.pop("occupancy_by_hour", None)
    t = voice.lower()
    kids = []
    if re.search(r"\bсын|\bson\b", t):
        kids.append("son")
    if re.search(r"\bдоч|\bdaughter", t):
        kids.append("daughter")
    children_count = p["children"] or 0
    people_count = p["people"] or max(1, children_count + 1)
    while len(kids) < children_count:
        kids.append(f"child{len(kids) + 1}")
    kids = kids[: max(children_count, len(kids))]
    adults = ["you"] + (["partner"] if people_count - len(kids) >= 2 else [])
    hh = [{"id": a, "role": "adult", "name": NAMES[a]} for a in adults] + \
         [{"id": k, "role": "child", "name": NAMES.get(k, NAMES["child"])} for k in kids]
    driver = adults[-1]
    trips = []
    for tr in base["trips"]:
        if tr["id"].startswith("office"):
            trips.append({**tr, "who": [adults[0]]})
        elif kids:
            who = ([driver] if tr["mode"] == "car" else []) + kids
            trips.append({**tr, "who": who})
    base.update({"household": hh, "trips": trips, "pseudonym": {"value": f"p-{uuid.uuid4().hex[:4]}", "decided_by": "TokenVault"},
                 "flags": {"has_school_children": bool(kids), "founder": bool(p["founder"])},
                 "resident_local_id": {"value": f"core:{uuid.uuid4().hex[:8]}", "decided_by": "CURE"}})
    if p["budget"]:
        base["params"]["budget_aed_month"] = {"value": p["budget"], "decided_by": "user"}
    for k in ("device_only", "constraints", "district_choices_demo", "setpoint_observations_demo",
              "voice_example", "voice_example_ru"):
        base.pop(k, None)
    sealed, cons = derive(split["device_only"], rt.P, [x["id"] for x in hh], voice)
    rid = resident_id or f"guest-{uuid.uuid4().hex[:6]}"
    s = rt.new_session(rid, profile_doc=base, sealed={k: [dict(x, at=rt.clock.isoformat()) for x in v] for k, v in sealed.items()},
                       constraints=cons, lang=lang, device=device)
    s.c.extra["city_label"], s.c.extra["city_label_ru"] = "Family from the audience", "Семья из зала"
    s.c.extra["city_day"] = 2
    from .modeling import ModelRegistry
    from .profiles import ProfileService
    ProfileService(ModelRegistry.load_default()).initialize(s, {
        "household_size": p["people"],
        "children_count": p["children"],
        "is_founder": p["founder"],
        "housing_budget_aed_month": p["budget"],
    }, source="door")
    return s
