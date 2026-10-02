"""Evidence-only narration with one bounded model retry."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Any

from arrival.guard import _numbers_in, _tokens
from arrival.modeling.results import BackendResult

from .contracts import LanguageGateway, LanguageRequest, NarrationResult


FORBIDDEN_KEYS = {"sealed", "diagnosis", "health_record", "condition"}
COMPLETION_WORDS = re.compile(
    r"\b(done|completed|booked|executed|paid|sent)\b|"
    r"\b(\u0441\u0434\u0435\u043b\u0430\u043d\u043e|\u0432\u044b\u043f\u043e\u043b\u043d\u0435\u043d\u043e|\u043e\u0442\u043f\u0440\u0430\u0432\u043b\u0435\u043d\u043e|\u043e\u043f\u043b\u0430\u0447\u0435\u043d\u043e)\b",
    re.I,
)
EXTERNAL_WORDS = re.compile(
    r"\b(booked|executed|paid|sent|transferred|signed)\b|"
    r"\b(\u0437\u0430\u0431\u0440\u043e\u043d\u0438\u0440\u043e\u0432\u0430\u043b|\u043e\u0442\u043f\u0440\u0430\u0432\u0438\u043b|\u043e\u043f\u043b\u0430\u0442\u0438\u043b|\u043f\u0435\u0440\u0435\u0432\u0451\u043b|\u043f\u043e\u0434\u043f\u0438\u0441\u0430\u043b)\b",
    re.I,
)
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
MEASURE = re.compile(
    r"(?<![\w.])([-+]?\d{1,3}(?:[ ,\u00a0]\d{3})*(?:[.,]\d+)?|[-+]?\d+(?:[.,]\d+)?)"
    r"\s*(AED|days?|hours?|hrs?|minutes?|mins?|degC|\u00b0C|ug/m3|\u00b5g/m3|percent|%)",
    re.I,
)
UNIT_SUFFIXES = {
    "aed": "aed", "day": "days", "days": "days", "hour": "hours", "hours": "hours",
    "minute": "minutes", "minutes": "minutes", "c": "degc", "degc": "degc",
    "ugm3": "ug/m3", "percent": "%", "pct": "%",
}
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "\u044f\u043d\u0432\u0430\u0440\u044f": 1, "\u0444\u0435\u0432\u0440\u0430\u043b\u044f": 2, "\u043c\u0430\u0440\u0442\u0430": 3, "\u0430\u043f\u0440\u0435\u043b\u044f": 4,
    "\u043c\u0430\u044f": 5, "\u0438\u044e\u043d\u044f": 6, "\u0438\u044e\u043b\u044f": 7, "\u0430\u0432\u0433\u0443\u0441\u0442\u0430": 8, "\u0441\u0435\u043d\u0442\u044f\u0431\u0440\u044f": 9,
    "\u043e\u043a\u0442\u044f\u0431\u0440\u044f": 10, "\u043d\u043e\u044f\u0431\u0440\u044f": 11, "\u0434\u0435\u043a\u0430\u0431\u0440\u044f": 12,
}
MONTH_FIRST_DATE = re.compile(
    rf"\b({'|'.join(MONTHS)})\s+(\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(\d{{4}})\b", re.I
)
DAY_FIRST_DATE = re.compile(
    rf"\b(\d{{1,2}})\s+({'|'.join(MONTHS)})\s+(\d{{4}})\b", re.I
)
STATUS_QUALIFIERS = {
    "unsupported_request": re.compile(
        r"\b(unsupported|not supported|cannot handle|outside (?:the )?supported)\b|\u043d\u0435 \u043f\u043e\u0434\u0434\u0435\u0440\u0436", re.I
    ),
    "missing_data": re.compile(
        r"\b(missing|need|provide|unknown|not available)\b|\u043d\u0435 \u0445\u0432\u0430\u0442|\u043d\u0443\u0436\u043d|\u043d\u0435\u0438\u0437\u0432\u0435\u0441\u0442", re.I
    ),
    "insufficient_model": re.compile(
        r"\b(insufficient|no registered model|cannot compute|not modeled)\b|\u043d\u0435\u0442 \u043c\u043e\u0434\u0435\u043b\u0438|\u043d\u0435\u0434\u043e\u0441\u0442\u0430\u0442\u043e\u0447", re.I
    ),
    "clarification_required": re.compile(r"\b(clarif|please specify|what do you mean)\b|\u0443\u0442\u043e\u0447\u043d", re.I),
    "consent_required": re.compile(r"\b(consent|required permission|need permission)\b|\u0441\u043e\u0433\u043b\u0430\u0441", re.I),
    "approval_required": re.compile(r"\b(approval|confirm|pending)\b|\u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0434|\u043e\u0434\u043e\u0431\u0440\u0435\u043d", re.I),
    "policy_blocked": re.compile(r"\b(blocked|not allowed|policy)\b|\u0437\u0430\u043f\u0440\u0435\u0449|\u043f\u043e\u043b\u0438\u0442\u0438\u043a", re.I),
}


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    reasons: list[str]


class NarrationGenerationError(RuntimeError):
    status = "generation_failed"

    def __init__(self, reasons: list[str], attempts: int):
        self.reasons = reasons
        self.attempts = attempts
        super().__init__("Model narration failed evidence validation")


def _forbidden_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                return str(key).lower()
            found = _forbidden_key(child)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = _forbidden_key(child)
            if found:
                return found
    return None


def _normal_number(raw: str) -> float:
    value = raw.replace("\u00a0", "").replace(" ", "")
    value = value.replace(",", "") if re.search(r",\d{3}(?:\D|$)", value) else value.replace(",", ".")
    return float(value)


def _normal_unit(raw: str) -> str:
    unit = raw.lower().replace("\u00b5", "u").replace("\u00b0", "deg")
    return {
        "day": "days", "days": "days", "hour": "hours", "hours": "hours", "hr": "hours", "hrs": "hours",
        "minute": "minutes", "minutes": "minutes", "min": "minutes", "mins": "minutes",
        "degc": "degc", "ug/m3": "ug/m3", "percent": "%", "%": "%", "aed": "aed",
    }.get(unit, unit)


def _measure_pairs(text: str) -> list[tuple[float, str, str]]:
    return [(_normal_number(number), _normal_unit(unit), unit) for number, unit in MEASURE.findall(text)]


def _evidence_units(value: Any, *, key: str = "") -> set[tuple[float, str]]:
    pairs: set[tuple[float, str]] = set()
    if isinstance(value, dict):
        if isinstance(value.get("value"), (int, float)) and isinstance(value.get("unit"), str):
            pairs.add((float(value["value"]), _normal_unit(value["unit"])))
        elif isinstance(value.get("value"), (int, float)) and (unit := _unit_from_key(key)):
            pairs.add((float(value["value"]), unit))
        for child_key, child in value.items():
            pairs.update(_evidence_units(child, key=str(child_key)))
    elif isinstance(value, (list, tuple)):
        for child in value:
            pairs.update(_evidence_units(child, key=key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if unit := _unit_from_key(key):
            pairs.add((float(value), unit))
    elif isinstance(value, str):
        pairs.update((number, unit) for number, unit, _ in _measure_pairs(value))
    return pairs


def _unit_from_key(key: str) -> str | None:
    normalized = key.lower()
    if "aed" in normalized.split("_"):
        return "aed"
    suffix = normalized.rsplit("_", 1)[-1]
    aliases = {**UNIT_SUFFIXES, "min": "minutes", "h": "hours"}
    return aliases.get(suffix)


def _iso_dates(value: Any) -> set[str]:
    dates: set[str] = set()
    if isinstance(value, str):
        dates.update(ISO_DATE.findall(value))
    elif isinstance(value, dict):
        for child in value.values():
            dates.update(_iso_dates(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            dates.update(_iso_dates(child))
    return dates


def _natural_dates(text: str) -> set[str]:
    dates: set[str] = set()
    candidates = [
        (int(year), MONTHS[month.casefold()], int(day))
        for month, day, year in MONTH_FIRST_DATE.findall(text)
    ]
    candidates += [
        (int(year), MONTHS[month.casefold()], int(day))
        for day, month, year in DAY_FIRST_DATE.findall(text)
    ]
    for year, month, day in candidates:
        try:
            dates.add(date(year, month, day).isoformat())
        except ValueError:
            dates.add(f"invalid:{year:04d}-{month:02d}-{day:02d}")
    return dates


def evidence_guard(text: str, result: BackendResult, user_text: str) -> GuardResult:
    reasons: list[str] = []
    payload = result.model_dump()
    forbidden = _forbidden_key(result.facts)
    if forbidden:
        reasons.append(f"forbidden_evidence:{forbidden}")

    allowed = list(_numbers_in(payload)) + [value for value, _ in _tokens(user_text)]
    unsupported_values: set[float] = set()
    for value, decimals in _tokens(text):
        if not any(round(candidate, decimals) == value for candidate in allowed):
            reasons.append(f"unsupported_number:{value:g}")
            unsupported_values.add(value)

    allowed_dates = _iso_dates(payload) | set(ISO_DATE.findall(user_text))
    for claimed_date in ISO_DATE.findall(text):
        if claimed_date not in allowed_dates:
            reasons.append(f"unsupported_date:{claimed_date}")
    for claimed_date in _natural_dates(text):
        if claimed_date not in allowed_dates:
            reasons.append(f"unsupported_date:{claimed_date}")

    evidenced_units = _evidence_units(payload) | {(number, unit) for number, unit, _ in _measure_pairs(user_text)}
    units_by_number: dict[float, set[str]] = {}
    for number, unit in evidenced_units:
        units_by_number.setdefault(number, set()).add(unit)
    for number, unit, display_unit in _measure_pairs(text):
        if number in unsupported_values:
            continue
        known = units_by_number.get(number, set())
        if unit not in known:
            reasons.append(f"unsupported_unit_claim:{number:g}:{display_unit}")

    lowered = text.lower()
    if result.data_mode == "synthetic" and "synthetic" not in lowered and "\u0441\u0438\u043d\u0442\u0435\u0442" not in lowered:
        reasons.append("missing_synthetic_qualifier")
    if result.data_mode == "shadow" and "shadow" not in lowered:
        reasons.append("missing_shadow_qualifier")
    status_qualifier = STATUS_QUALIFIERS.get(result.status)
    if status_qualifier is not None and not status_qualifier.search(text):
        reasons.append(f"missing_status_qualifier:{result.status}")
    if (result.status == "approval_required" or result.pending) and COMPLETION_WORDS.search(text):
        reasons.append("pending_claimed_complete")
    if not result.external_action and EXTERNAL_WORDS.search(text):
        reasons.append("false_external_action")
    return GuardResult(ok=not reasons, reasons=reasons)


class NarrationService:
    def __init__(self, gateway: LanguageGateway):
        self.gateway = gateway

    def render(self, request: LanguageRequest, result: BackendResult) -> NarrationResult:
        payload = result.model_dump()
        first = self.gateway.narrate(request, payload)
        if first.source == "local":
            return first
        checked = evidence_guard(first.text, result, request.text)
        if checked.ok:
            return first
        second = self.gateway.narrate(request, payload, feedback={"reasons": checked.reasons})
        checked_again = evidence_guard(second.text, result, request.text)
        if checked_again.ok:
            return second
        raise NarrationGenerationError(checked_again.reasons, attempts=2)
