"""Fact gateway: the only path by which information from the web (found by agents) enters the CURE core.

An agent (search_public tool) finds a page and returns a card. The gateway accepts the card only if:
  • it has exactly the card fields, nothing extra (free text and commands never pass);
  • the source is on the allow-list of official sites (policy gateway.allowlist);
  • the claim has a known type, is about a known step and has a plausible range.
An accepted card is "pending"; it enters the plan after the person's yes (or a second independent source).
"""
from __future__ import annotations

from datetime import datetime
from urllib.parse import urlparse

from .core import Store

CARD_FIELDS = {"claim", "subject", "low", "high", "unit", "source_url", "retrieved_at"}


def allowed_source(url: str, domains: list[str]) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or '').casefold()
    return parsed.scheme == 'https' and not parsed.username and not parsed.password and any(
        host == domain.casefold() or host.endswith('.' + domain.casefold()) for domain in domains)


def search(corpus: Store, topic: str) -> list[dict]:
    return [p for p in corpus.raw("pages") if p["topic"] == topic]


def read(page: dict, *, now: datetime) -> dict:
    """Card from a page. The environment, not the model, sets the source and time."""
    return {**page["card"], "source_url": page["url"], "retrieved_at": now.isoformat()}


def check_card(card: dict, *, policy: Store, steps: Store) -> dict:
    reasons = []
    extra = sorted(set(card) - CARD_FIELDS)
    if extra:
        reasons.append(f"extra_fields:{extra}")
    if urlparse(str(card.get("source_url", ""))).hostname not in set(policy.get("gateway.allowlist")):
        reasons.append("source_not_official")
    if card.get("claim") not in set(policy.get("gateway.claims")):
        reasons.append("unknown_claim")
    if card.get("unit") != "days":
        reasons.append("unknown_unit")
    if card.get("subject") not in {s["id"] for s in steps.raw("steps")}:
        reasons.append("unknown_subject")
    lo, hi = card.get("low"), card.get("high")
    if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and 0 < lo <= hi <= float(policy.get("gateway.max_days"))):
        reasons.append("implausible_range")
    return {"accepted": not reasons, "reasons": reasons, "status": "pending" if not reasons else "rejected",
            "card": {k: card[k] for k in sorted(CARD_FIELDS & set(card))}}


def to_override(card: dict) -> dict:
    """Confirmed card → step duration distribution for the plan."""
    lo, hi = float(card["low"]), float(card["high"])
    return {"dist": "triangular", "low": lo, "mode": (lo + hi) / 2, "high": hi, "source": card["source_url"]}
