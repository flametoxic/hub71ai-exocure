"""Conversation orchestrator: resident phrase → intent → session method (the core computes) → templated reply.

Safety first (emergency, medicine, other people's data, rule changes, discrimination, money, signatures),
then life topics. Parsing is offline (RU/EN regular expressions). If a language model is connected (OPENAI_API_KEY +
ARRIVAL_OPENAI_MODEL), it may pick an intent from the same closed list when the rules did not recognise the phrase;
only text without health fragments goes to it (person.split_voice). The model adds no numbers to replies.
Official-site search uses only domains from policy official_search.allowlist; without a model — an honest refusal.
Russian patterns below parse Russian-speaking residents' phrases.
"""
from __future__ import annotations

import copy
import json
import os
import re

from . import assistant as A
from .config import Settings
from .language import LanguageRequest, NarrationService, build_language_gateway
from .modeling import BackendResult, ModelRegistry, PlanValidator, QueryExecutor
from .modeling.adapters import reply_result
from .person import split_voice, sanitize_for_llm
from .profiles import ProfileService, ProfileValueError
from .service import Session, lang_of, month_in, parse_date

I = re.IGNORECASE

# (intent, pattern) — order matters: safety before topics
RULES: list[tuple[str, str]] = [
    ("emergency", r"не может дышать|трудно дышать|задыха|приступ|can'?t breathe|trouble breathing|ambulance|скор(ая|ую)"),
    ("prompt", r"ignore (all|previous|your)|забудь (все |свои )?правила|игнорируй|system prompt|системн\w+ промпт|you are now|ты теперь|developer mode|jailbreak"),
    ("medical_advice", r"лекарств|дозиров|ингалятор|таблетк|лечени|medicine|medication|dose|inhaler|treatment|pills?\b"),
    ("diagnosis", r"диагноз|чем (он |она )?болеет|какая (у него |у неё )?болезнь|diagnos|what (illness|condition)|health record|мед(ицинск\w+)? данн"),
    ("others", r"(данные|адрес|имя|имена|телефон)\w* (сосед|других|чуж)|кто (живёт|живет) (рядом|в доме|напротив)|other (famil|residents)\w*'? (data|names|address)|neighbou?rs'? (names|data|address)|someone else'?s"),
    ("discrimination", r"национальн|религи|nationalit|religio|ethnic|этнич|без (индус|араб|мусульман|христиан|евре)|(no|without) (indians|arabs|muslims|christians|jews)"),
    ("money", r"переведи|перевести|заплати|оплати|отправь деньги|\bpay\b|transfer|send money|wire\b"),
    ("sign", r"подпиши|подписать|sign (the|my|it|this|contract)"),
    ("send", r"отправь|отошли|send (it|the|this|email|letter|message)|email (him|her|them|my)"),
    ("delete", r"удали (меня|всё|все|мои|мой)|забудь меня|delete (me|my|everything)|forget me|erase"),
    ("memory", r"что ты (обо мне |про меня )?знаешь|what do you know|что ты помнишь|my data|мои данные"),
    ("who", r"кто ты|что ты такое|who are you|what are you|ты chatgpt|are you (chat)?gpt|ты gpt"),
    ("cant", r"что ты не (умеешь|можешь|делаешь)|what can'?t you|what won'?t you|чего ты не"),
    ("realtime", r"прямо сейчас|сейчас.*(пыль|воздух|pm)|right now|current (air|dust|pm)|live (data|sensor)"),
    ("future_rent", r"через (\d|пять|пару) лет|in (\d|five|a few) years|купить или снимать|buy or rent|цены .* будущ|rent .* future"),
    ("legal", r"закон|юрид|суд\b|legal|lawyer|court|по закону|штраф|fine\b"),
    ("offtopic", r"рецепт|recipe|футбол|football|анекдот|joke|биткоин|bitcoin|stock market|погода в (париж|лондон|москв)|weather in (paris|london|moscow)"),
    ("apply", r"^(применить|применяй|apply|да, применить|yes, apply)\b"),
    ("yes", r"^(да|yes|ок|ok|okay|подтверждаю|approve|давай|sure|конечно)\b"),
    ("no", r"^(нет|no|не надо|decline|отмена|cancel)\b"),
    ("search", r"найди|поищи|проверь сроки|search|look up|find out|official site|официальн"),
    ("visa_delay", r"виз\w*.*(задерж|сдвин|перенес|\+\s?\d+)|visa .*(delay|late|moved|\+\s?\d+)|задержк\w* виз"),
    ("counterfactual", r"(если бы|без) задержк|what if (there (was|were) )?no delay|without (the )?delay|если бы не"),
    ("levers", r"что поможет|как ускорить|what helps|speed (it|this|things)? ?up|faster|ускор"),
    ("switch", r"когда .*(смени|передума|другой район|поменя)|when would you (switch|change)|what would change your"),
    ("money_first", r"деньги важнее|money matters more|дешевле|cheaper|сэконом|save money"),
    ("why", r"^почему|^why|почему\b|\bwhy\b|объясни|explain"),
    ("guest", r"\bмам|гост|mother|\bmom\b|guest|приезжа"),
    ("beach", r"пляж|beach|\bморе|\bsea\b"),
    ("license", r"лиценз|найм|ваканс|licen[cs]e|hire|hiring|компани"),
    ("car", r"машин|такси|\bcar\b|taxi"),
    ("budget", r"бюджет|потрат|расход|budget|spend|spent|сколько (мы )?тратим"),
    ("late", r"буду (дома )?(в|к|после) \d|задерж\w* на работе|приду (в|к|после)|be home (at|by|after)|home at \d|i'?ll be late|опоздаю|late today"),
    ("setpoint", r"(постав\w+|set)\s.*(\d\d)\s*(°|град|degrees)|кондиционер на \d\d|ac (to|at) \d\d"),
    ("morning", r"утр|выезж|выезд|school run|morning|when (should|do) (we|i) leave|во сколько выезжать"),
    ("outdoor", r"гулять|на улиц|парк|outside|outdoor|\bpark\b|\bwalk\b|окна"),
    ("errands", r"поручен|дела|errand|pharmacy|аптек"),
    ("district", r"район|district|где (нам )?жить|where (should we |to )?live|neighbou?rhood|area"),
    ("apartment", r"квартир|сч[её]т|\bbill\b|\bflat\b|apartment|охлажд|cooling|кондиционер|\bac\b"),
    ("arrival", r"прилет|прилёт|прилетаем|\bland\b|arriv|переезжаем"),
    ("plan", r"\bплан\b|когда (мы )?(будем )?готов|when .*ready|\bplan\b|документ|documents"),
    ("greeting", r"^(привет|здравств|добрый|hi\b|hello|hey)"),
]
INTENTS = sorted({k for k, _ in RULES})


def _llm_intent(text: str) -> str | None:
    # Local routing stays offline. OpenAI routing uses the validated Responses planner.
    return None


def _official_search(text: str, s: Session) -> dict:
    """A rules question → only official sites from the policy. Without a model — an honest refusal."""
    settings = Settings.from_env()
    model = settings.openai_model
    if settings.language_mode != "openai" or not settings.openai_api_key:
        return s.say(s.r("search_offline"), route="official_search")
    domains = list(s.P.get("official_search.allowlist"))
    try:
        from openai import OpenAI
        r = OpenAI().responses.create(
            model=model, input=sanitize_for_llm(text, s.P), store=False, tool_choice="required",
            tools=[{"type": "web_search", "filters": {"allowed_domains": domains}}],
            instructions="Answer briefly using only the retrieved official pages. If they don't answer, say so.")
        urls = sorted({a.url for o in r.output if getattr(o, "type", "") == "message" for c in o.content
                       for a in (getattr(c, "annotations", None) or []) if getattr(a, "url", None)})
        from .gateway import allowed_source
        ok = [u for u in urls if allowed_source(u, domains)]
        if not ok:
            return s.say(s.r("search_none"), route="official_search")
        return s.say(s.r("search_found", answer=r.output_text.strip(), sources=", ".join(ok)),
                     cards=[{"type": "official_search", "sources": ok, "allowlist": domains}], route="official_search")
    except Exception as e:
        return s.say(s.r("search_none"), cards=[{"type": "error", "error": type(e).__name__}], route="official_search")


def route(text: str) -> str:
    t = text.strip().replace("’", "'").replace("‘", "'")
    for intent, rx in RULES:
        if re.search(rx, t, I):
            return intent
    return "fallback"


def _legacy_ask(s: Session, text: str) -> dict:
    lang = lang_of(text, s.lang)
    s.c.settings["lang"] = lang
    split = split_voice(text, s.P)
    s.c.say("you", text, at=s.now)
    topic = s.c.extra.get("topic")
    intent = route(text)
    if intent == "fallback":
        intent = _llm_intent(split["text_for_llm"]) or "fallback"

    # health in a free phrase: when answering about the guest it goes to the sealed domain
    if split["device_only"] and topic == "guest" and intent not in ("emergency", "medical_advice"):
        return s.guest_constraint(text)

    # ---- safety and honesty
    fixed = {"prompt": "prompt_refuse", "medical_advice": "medical_advice", "emergency": "medical_emergency",
             "others": "others_refuse", "discrimination": "discrimination_refuse", "money": "money_refuse",
             "sign": "sign_refuse", "send": "send_draft", "who": "who", "cant": "cant_do", "legal": "legal_safe",
             "offtopic": "off_topic", "greeting": "greeting"}
    if intent in fixed:
        return s.say(s.r(fixed[intent]), route=intent)
    if intent == "diagnosis":
        return s.diagnosis()
    if intent == "delete":
        return s.delete()
    if intent == "memory":
        return s.what_i_know()
    if intent == "realtime":
        return s.realtime()
    if intent == "future_rent":
        return s.future_rent()

    # ---- answers to proposals
    if intent == "apply" or (intent == "yes" and s.c.extra.get("offer_switch")):
        return s.choose()
    if intent in ("yes", "no"):
        if topic == "conflict" and intent == "yes" and not s.c.pending:
            return s.m_thu_conflict()
        return s.decide(None, intent == "yes")
    if topic == "apartment" and s.cache.get("apt_ask"):
        opts = s.cache["apt_bill"]["value"]["options"]
        for o in opts:
            labels = [A.OPTION_LABELS.get(o, {}).get(x, o).lower() for x in ("ru", "en")] + [o.replace("_", " ")]
            if any(l in text.lower() for l in labels):
                return s.apartment(answer_key=o)

    # ---- topics
    m = month_in(text)
    if intent == "arrival" or (topic == "arrival_date" and parse_date(text, today=s.rt.today())):
        d = parse_date(text, today=s.rt.today())
        if d:
            return s.set_arrival(d)
        return s.say(s.r("ask_arrival_date"), route="arrival")
    if intent == "visa_delay":
        n = re.search(r"(\d+)", text)
        return s.visa_delay(float(n[1]) if n else 14)
    if intent == "counterfactual":
        return s.visa_counterfactual()
    if intent == "levers":
        return s.visa_levers()
    if intent == "switch":
        return s.district_switch()
    if intent == "money_first":
        return s.car_taxi(money_first=True) if topic == "car" else s.whatif_money()
    if intent == "why":
        if topic == "visa":
            return s.visa_why()
        if topic in ("district", None):
            return s.district_why()
        return s.say(s.r("fallback"), route="why")
    if m and (intent in ("district", "fallback") or re.search(r"что если|а если|what if|what about|а в\b", text, I)):
        return s.whatif_month(m)
    if intent == "search":
        return s.agent_search() if re.search(r"срок|how long|duration|сколько дней|days", text, I) or topic == "plan" \
            else _official_search(split["text_for_llm"], s)
    handlers = {"guest": s.m_sat_guest, "beach": s.m_sat_beach, "license": s.license, "car": s.car_taxi,
                "budget": s.m_month_end, "morning": s.m_mon_morning, "outdoor": s.m_wed_windows,
                "errands": s.m_tue_errands, "district": s.districts, "apartment": s.apartment, "plan": s.plan}
    if intent == "late":
        n = re.search(r"(\d{1,2})(?:[:.](\d{2}))?", text)
        h = float(n[1]) + (float(n[2]) / 60 if n and n[2] else 0.0) if n else 20.0
        return s.m_mon_home(return_h=h if h > 12 else h + 12, late=True)
    if intent == "setpoint":
        n = re.search(r"(\d\d(?:[.,]\d)?)", text)
        return s.setpoint(float(n[1].replace(",", "."))) if n else s.say(s.r("fallback"), route="setpoint")
    if intent in handlers:
        return handlers[intent]()
    if re.search(r"правил|сколько (стоит|дней)|how (long|much)|requirement|нужн\w* документ|what do i need", text, I):
        return _official_search(split["text_for_llm"], s)
    return s.say(s.r("fallback"), route="fallback")


def _gateway_from_settings(settings: Settings):
    return build_language_gateway(settings)


def profile_intake_card(s: Session) -> dict:
    registry = ModelRegistry.load_default()
    profiles = ProfileService(registry)
    fields = profiles.snapshot(s)
    question = profiles.next_question(s, s.lang)
    return {
        "type": "profile_intake",
        "fields": fields,
        "next_question": question,
        "supported_domains": sorted(registry.domains),
        "complete": question is None,
    }


def _profile_intake_answer(s: Session, text: str) -> dict | None:
    variable_id = s.c.extra.get("profile_pending")
    if not variable_id:
        return None
    s.c.say("you", text, at=s.now)
    registry = ModelRegistry.load_default()
    profiles = ProfileService(registry)
    spec = registry.variable(variable_id)
    normalized = text.strip().lower()
    status, value = "known", None
    if normalized in {"skip", "пропустить", "не знаю", "don't know", "dont know"}:
        status = "skipped"
    elif normalized in {"decline", "отказываюсь", "не хочу отвечать", "prefer not to say"}:
        status = "declined"
    elif spec.value_type in {"number", "integer"}:
        match = re.search(r"[-+]?\d+(?:[.,]\d+)?", text)
        if match:
            raw = float(match.group().replace(",", "."))
            value = int(raw) if spec.value_type == "integer" and raw.is_integer() else raw
    elif spec.value_type == "boolean":
        if normalized in {"yes", "да", "true"}:
            value = True
        elif normalized in {"no", "нет", "false"}:
            value = False
    elif spec.value_type == "date":
        parsed = parse_date(text, today=s.rt.today())
        value = parsed.isoformat() if parsed else None
    else:
        value = text.strip() or None
    if status == "known" and value is None:
        prompt = next(field for field in registry.profile_schema() if field.variable_id == variable_id).prompts
        message = (prompt.get(s.lang) or prompt.get("en")) + " You can also say 'skip'."
        return s.say(message, cards=[profile_intake_card(s)], route="profile_intake")
    try:
        profiles.record(s, variable_id, value=value, status=status, source="chat")
    except ProfileValueError as error:
        return s.say(str(error), cards=[profile_intake_card(s)], route="profile_intake")
    card = profile_intake_card(s)
    s.c.extra["profile_pending"] = (card.get("next_question") or {}).get("variable_id")
    messages = {
        "en": {"known": "Saved.", "skipped": "Skipped."},
        "ru": {"known": "Сохранено.", "skipped": "Пропущено."},
    }
    message = messages.get(s.lang, messages["en"])["known" if status == "known" else "skipped"]
    if card["next_question"]:
        message += " " + card["next_question"]["text"]
    return s.say(message, cards=[card], route="profile_intake")


def _pipeline_response(s: Session, result: BackendResult, narration) -> dict:
    text = narration.text
    cards = list(result.cards)
    replaced = _replace_backend_message(s, result.local_text, text, cards=cards, route=result.operation, traces=result.evidence)
    if not replaced:
        s.c.say("cure", text, at=s.now, trace_ids=result.evidence)
        s.c.conversation[-1].update(cards=cards, quick=[], route=result.operation)
        s.c.extra["rev"] = s.c.extra.get("rev", 0) + 1
    s.c.extra.setdefault("last", {})["reply"] = {"route": result.operation, "text": text, "at": s.now.isoformat()}
    s.c.extra.setdefault('mind', {}).setdefault('last_trace', {}).update(
        operation=result.operation, status=result.status, provider=narration.source,
        response_id=narration.response_id, calculation_trace_ids=result.evidence)
    s.save()
    return {
        "route": result.operation,
        "text": text,
        "styled": narration.source == "openai",
        "lang": s.lang,
        "cards": cards,
        "trace_ids": result.evidence,
        "quick": [],
        "clock": s.now.isoformat(),
        "pending": s.pending_view(),
        "meta": {
            "language_source": narration.source,
            "model": narration.model,
            "response_id": narration.response_id,
            "operation": result.operation,
            "status": result.status,
        },
    }


def _replace_backend_message(
    s: Session,
    backend_text: str,
    model_text: str,
    *,
    cards: list[dict] | None = None,
    route: str,
    traces: list[str] | None = None,
) -> bool:
    if not s.c.conversation:
        return False
    latest = s.c.conversation[-1]
    if latest.get("role") != "cure" or latest.get("text") != backend_text:
        return False
    latest.update(
        text=model_text,
        cards=list(cards or latest.get("cards") or []),
        quick=[],
        route=route,
        trace_ids=list(traces or latest.get("trace_ids") or []),
    )
    return True


def _discard_backend_message(s: Session, backend_text: str) -> None:
    if s.c.conversation and s.c.conversation[-1].get("role") == "cure" and s.c.conversation[-1].get("text") == backend_text:
        s.c.conversation.pop()
        s.save()


def ask(s: Session, text: str) -> dict:
    settings = Settings.from_env()
    # Known safety and completion events remain backend authority in both modes.
    intent = route(text)
    if intent in {'delete', 'emergency', 'prompt', 'medical_advice', 'diagnosis', 'others', 'money', 'sign', 'send', 'discrimination'}:
        reply = _legacy_ask(s, text)
        return reply if intent == 'delete' else narrate_action(s, intent, reply)
    if re.search(r'(received|got|получил\w*).*emirates\s*id', text, I):
        s.c.say('you', text, at=s.now)
        reply = s.complete_step('emirates_id')
        return narrate_action(s, 'complete_step', reply)
    intake = _profile_intake_answer(s, text)
    if intake is not None:
        if settings.language_mode == "local":
            intake["meta"] = {"language_source": "local", "model": None, "operation": "record_event", "status": "computed"}
            return intake
        return narrate_action(s, "profile_intake", intake)
    if settings.language_mode == "local":
        reply = _legacy_ask(s, text)
        reply["meta"] = {"language_source": "local", "model": None, "operation": reply.get("route"), "status": "computed"}
        return reply

    registry = ModelRegistry.load_default()
    gateway = _gateway_from_settings(settings)
    split = split_voice(text, s.P)
    s.c.settings["lang"] = lang_of(text, s.lang)
    s.c.say("you", text, at=s.now)
    profile_statuses = {
        key: value["status"] for key, value in s.c.extra.get("profile_fields", {}).items()
    }
    request = LanguageRequest(
        text=split["text_for_llm"],
        lang=s.lang,
        resident_id=s.rid,
        context={"topic": s.c.extra.get("topic"), "profile_statuses": profile_statuses,
                 "supported_domains": sorted(registry.domains)},
    )
    from .mind import Mind
    from . import dynamic_agents
    request.context['world_model'] = Mind(s, settings).context(request.text)
    plan = gateway.plan(request, registry.tool_catalog() + [dynamic_agents.TOOL])
    specialist_calls = [call for call in plan.operations if call.name == 'create_specialist']
    if specialist_calls or not plan.operations:
        if specialist_calls and (len(plan.operations) != 1 or set(specialist_calls[0].arguments) != {'task'}
                                 or specialist_calls[0].arguments['task'] != request.text):
            return s.say('Specialist request failed backend task validation.', route='policy_blocked')
        run = dynamic_agents.start(s, request.text, settings=settings)
        message = 'Создаю специалиста для этой задачи. Его полномочия проверит CURE.' if s.lang == 'ru' else 'Building a specialist for this task. CURE will validate its permissions.'
        reply = s.say(message, cards=[{'type': 'dynamic_agent', 'agent_id': run['agent_id']}], route='dynamic_agent')
        reply['meta'] = {'language_source': 'openai', 'operation': 'dynamic_agent', 'agent_id': run['agent_id'], 'status': run['status']}
        return reply
    validated = PlanValidator(registry).validate(plan, s, user_text=request.text)
    result = QueryExecutor(registry).execute(validated, s)
    update = {"plan_response_id": plan.response_id}
    if len(plan.operations) > 1 and not result.tool_outputs:
        failure_output = result.model_dump(exclude={"tool_outputs", "plan_response_id", "tool_call_id"})
        update["tool_outputs"] = [
            {"call_id": call.call_id, "output": failure_output}
            for call in plan.operations
            if call.call_id
        ]
    result = result.model_copy(update=update)
    try:
        narration = NarrationService(gateway).render(request, result)
    except Exception:
        _discard_backend_message(s, result.local_text)
        raise
    return _pipeline_response(s, result, narration)


def narrate_action(s: Session, action: str, reply: dict) -> dict:
    if reply.get('route') == 'deleted':
        return reply
    settings = Settings.from_env()
    if settings.language_mode == "local":
        reply["meta"] = {**reply.get("meta", {}), "language_source": "local", "model": None,
                         "operation": action, "status": "computed"}
        return reply
    gateway = _gateway_from_settings(settings)
    request = LanguageRequest(text=f"Explicit UI action: {action}", lang=s.lang, resident_id=s.rid)
    language_reply = copy.deepcopy(reply)
    if action == "door":
        for card in language_reply.get("cards", []):
            if card.get("type") == "door":
                card["heard"] = card.get("sent_to_language_model", "")
    result = reply_result(action, language_reply, inputs={"action": action}, response_id=None, call_id=None)
    if action == "profile_intake":
        result = result.model_copy(update={"data_mode": "user"})
    try:
        narration = NarrationService(gateway).render(request, result)
    except Exception:
        _discard_backend_message(s, str(reply.get("text") or ""))
        raise
    _replace_backend_message(
        s,
        str(reply.get("text") or ""),
        narration.text,
        cards=list(reply.get("cards") or []),
        route=action,
        traces=list(reply.get("trace_ids") or []),
    )
    s.c.extra.setdefault("last", {})["reply"] = {"route": action, "text": narration.text, "at": s.now.isoformat()}
    s.save()
    output = dict(reply)
    output.update(text=narration.text, styled=narration.source == "openai")
    output["meta"] = {**reply.get("meta", {}), "language_source": narration.source, "model": narration.model,
                      "operation": action, "status": result.status}
    return output
