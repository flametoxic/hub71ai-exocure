"""Agent drafts (draft_text tool). The text is built ONLY from CURE facts; the person sends it.
In OpenAI mode a specialist selects an approved variant; the core renders and validates it.
The "ru" strings are runtime output for Russian-speaking residents."""
from __future__ import annotations

from datetime import datetime
from string import Formatter

DRAFTS = {
    "employer_letter": {
        "ru": "Здравствуйте. Из-за задержки визы моё заселение переносится ориентировочно на {ready_mid} "
              "(от {ready_low} до {ready_high}). Предлагаю перенести дату выхода на эти даты. С уважением, Лейла.",
        "en": "Hello. Because of a visa delay, my move-in is now expected around {ready_mid} "
              "(between {ready_low} and {ready_high}). Could we shift my start date accordingly? Best regards, Leila."},
    "landlord_note": {
        "ru": "Здравствуйте. Наш въезд сдвигается ориентировочно на {ready_mid}. Можем ли мы обсудить дату начала аренды?",
        "en": "Hello. Our move-in is shifting to around {ready_mid}. Could we discuss the lease start date?"},
    "school_note": {
        "ru": "Здравствуйте. Наш переезд сдвигается ориентировочно на {ready_mid}. Уточните, пожалуйста, крайний срок подачи документов.",
        "en": "Hello. Our relocation is shifting to around {ready_mid}. Could you confirm the document deadline?"},
    "dust_line": {
        "ru": "Сегодня пыльно: Адаму — только от двери до двери, прогулку лучше провести в помещении.",
        "en": "Dusty today: Adam goes door to door only; indoor play instead of a walk."},
    "investor_reschedule": {
        "ru": "Здравствуйте. Можем ли мы перенести сегодняшнюю встречу с {old_time} на {new_time}? "
              "Причина — логистика с детьми. С уважением, Лейла.",
        "en": "Hello. Could we move today's meeting from {old_time} to {new_time}? It's a school-run clash. Best, Leila."},
    "vacancy": {
        "ru": "Ищем первого сотрудника в стартап в Hub71. Старт — ориентировочно {hire_mid}. Абу-Даби, полный день.",
        "en": "We're hiring our first team member for a Hub71 startup. Expected start: around {hire_mid}. Abu Dhabi, full-time."},
}

CONCISE_DRAFTS = {
    "employer_letter": {
        "ru": "Здравствуйте. Из-за задержки визы заселение ожидается {ready_mid} (окно: {ready_low}–{ready_high}). "
              "Можем ли мы перенести дату выхода? С уважением, Лейла.",
        "en": "Hello. A visa delay puts my expected move-in at {ready_mid} ({ready_low}–{ready_high}). "
              "Could we move my start date? Best regards, Leila."},
    "landlord_note": {
        "ru": "Здравствуйте. Въезд ожидается {ready_mid}. Можем ли мы перенести начало аренды?",
        "en": "Hello. Our move-in is expected around {ready_mid}. Could we move the lease start date?"},
    "school_note": {
        "ru": "Здравствуйте. Переезд ожидается {ready_mid}. Какой крайний срок подачи документов?",
        "en": "Hello. Our relocation is expected around {ready_mid}. What is the document deadline?"},
    "dust_line": {
        "ru": "Сегодня пыльно: Адаму лучше остаться в помещении.",
        "en": "Dusty today: indoor play is the better option for Adam."},
    "investor_reschedule": {
        "ru": "Здравствуйте. Можем ли мы перенести встречу с {old_time} на {new_time}? С уважением, Лейла.",
        "en": "Hello. Could we move today's meeting from {old_time} to {new_time}? Best, Leila."},
    "vacancy": {
        "ru": "Ищем первого сотрудника для стартапа в Hub71. Старт около {hire_mid}, полный день, Абу-Даби.",
        "en": "Hiring our first Hub71 startup team member around {hire_mid}, full-time in Abu Dhabi."},
}


def fact_keys(draft_id: str) -> set[str]:
    """Return only fact slots declared by the core-owned draft variants."""
    templates = [*DRAFTS[draft_id].values(), *CONCISE_DRAFTS[draft_id].values()]
    return {
        field_name
        for template in templates
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name
    }


def draft(agent: dict, draft_id: str, *, facts: dict, lang: str, now: datetime, trace_ids: list | None = None,
          variant: str = "standard") -> dict:
    scope = sorted(agent["contract"].scope)[0]
    r = agent["contract"].authorize(action="draft_text", scope=scope, at=now)
    if r.decision.value != "allowed":
        return {"by": f"agent:{agent['id']}", "status": "refused", "reason": r.reason}
    templates = DRAFTS if variant == "standard" else CONCISE_DRAFTS
    lang = lang if lang in templates[draft_id] else "en"
    text = templates[draft_id][lang].format(**facts)
    return {"by": f"agent:{agent['id']}", "agent_title": agent["title"], "kind": "draft", "draft": draft_id,
            "text": text, "send": "you_send" , "trace_ids": list(trace_ids or []), "status": "ready",
            "variant": variant}
