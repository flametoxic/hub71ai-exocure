"""Number guard: the language model cannot state a number the engine did not compute.

number_guard(text, trace_ids, user_text): every number in the model's text must match a number from the engine
answers (or from the person's own phrase) at the precision it is written with. Otherwise the text is dropped.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from .core import ANSWERS, Store

NUM = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:[ ,\u00a0]\d{3})+(?:[.,]\d+)?|(?<![\w.])[-+]?\d+(?:[.,]\d+)?")


def _tokens(text: str) -> list[tuple[float, int]]:
    """Numbers in the text and the number of decimals each is written with."""
    out = []
    for m in NUM.findall(text):
        s = m.replace("\u00a0", "").replace(" ", "")
        s = s.replace(",", "") if re.search(r",\d{3}(\D|$)", s) else s.replace(",", ".")
        try:
            out.append((float(s), len(s.split(".")[1]) if "." in s else 0))
        except ValueError:
            pass
    return out


def _numbers_in(obj: Any) -> Iterable[float]:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, str):
        yield from (x for x, _ in _tokens(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _numbers_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _numbers_in(v)


def number_guard(text: str, *, trace_ids: list[str], user_text: str, policy: Store) -> dict:
    decimals = set(policy.get("guard.decimals"))
    allowed = [x for t in trace_ids for k in ("value", "interval") for x in _numbers_in(ANSWERS.get(t, {}).get(k))]
    allowed += [x for x, _ in _tokens(user_text)]
    bad = [x for x, d in _tokens(text) if d not in decimals or not any(round(a, d) == x for a in allowed)]
    return {"ok": not bad, "rejected_numbers": bad, "text": text if not bad else None,
            "note": None if not bad else "Model text dropped: it contains numbers the engine did not compute."}
