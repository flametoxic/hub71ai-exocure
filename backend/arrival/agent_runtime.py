"""Bounded resident-agent runtimes selected by the backend language mode.

Local mode preserves deterministic templates. OpenAI mode gives each existing
signed role a separate model run, while the CURE core still owns authority,
source identity, validation, state mutation, approvals, and execution.
"""
from __future__ import annotations

from datetime import datetime
import json
import math
import re
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from api_gateway.core.planning.delegation import DelegationDecision

from . import drafts as local_drafts
from . import gateway as fact_gateway
from .config import Settings
from .person import sanitize_for_llm
from .guard import _numbers_in, _tokens
from .language import GatewayConfigurationError, GatewayError


FORBIDDEN_INPUT_KEYS = {"sealed", "diagnosis", "health_record", "condition", "device_only"}
EXTERNAL_COMPLETION = re.compile(
    r"\b(i|we|the agent|cure)\s+(sent|paid|booked|signed|submitted|transferred)\b|"
    r"\b(\u044f|\u043c\u044b|\u0430\u0433\u0435\u043d\u0442|cure)\s+(\u043e\u0442\u043f\u0440\u0430\u0432\u0438\u043b|\u043e\u043f\u043b\u0430\u0442\u0438\u043b|\u0437\u0430\u0431\u0440\u043e\u043d\u0438\u0440\u043e\u0432\u0430\u043b|\u043f\u043e\u0434\u043f\u0438\u0441\u0430\u043b)",
    re.I,
)
MONTH_WORD = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
    r"\u044f\u043d\u0432\u0430\u0440\w*|\u0444\u0435\u0432\u0440\u0430\u043b\w*|\u043c\u0430\u0440\u0442\w*|\u0430\u043f\u0440\u0435\u043b\w*|\u043c\u0430[\u0439\u044f\u0435]|"
    r"\u0438\u044e\u043d\w*|\u0438\u044e\u043b\w*|\u0430\u0432\u0433\u0443\u0441\u0442\w*|\u0441\u0435\u043d\u0442\u044f\u0431\u0440\w*|\u043e\u043a\u0442\u044f\u0431\u0440\w*|"
    r"\u043d\u043e\u044f\u0431\u0440\w*|\u0434\u0435\u043a\u0430\u0431\u0440\w*)\b",
    re.I,
)


class AgentRuntimeError(GatewayError):
    """A specialist agent could not produce a usable backend artifact."""


class AgentOutputError(AgentRuntimeError):
    def __init__(self, message: str):
        super().__init__("invalid_agent_output", message)


class _DraftOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variant: Literal["standard", "concise"]


class _FactCardOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page_index: int = Field(ge=0)
    claim: Literal["step_duration"]
    subject: str
    low: float
    high: float
    unit: Literal["days"]


class _FactCardsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cards: list[_FactCardOutput] = Field(max_length=20)


class AgentRuntime(Protocol):
    def draft(
        self,
        agent: dict[str, Any],
        draft_id: str,
        *,
        facts: dict[str, Any],
        lang: str,
        now: datetime,
        trace_ids: list[str] | None = None,
    ) -> dict[str, Any]: ...

    def extract_fact_cards(
        self,
        agent: dict[str, Any],
        pages: list[dict[str, Any]],
        *,
        allowed_subjects: list[str],
        now: datetime,
    ) -> list[dict[str, Any]]: ...


def _authorize(agent: dict[str, Any], action: str, *, now: datetime) -> None:
    contract = agent["contract"]
    matching_scopes = sorted(contract.scope & contract.authority.allowed_scopes)
    if not matching_scopes:
        raise AgentRuntimeError("agent_authority_denied", "Agent has no authorized scope.")
    decision = contract.authorize(action=action, scope=matching_scopes[0], at=now)
    if decision.decision != DelegationDecision.ALLOWED:
        raise AgentRuntimeError("agent_authority_denied", decision.reason)


def _sanitize(value: Any) -> Any:
    return sanitize_for_llm(value)


def _validate_draft_text(text: str, facts: dict[str, Any]) -> None:
    allowed_numbers = list(_numbers_in(facts))
    for value, decimals in _tokens(text):
        if not any(round(candidate, decimals) == value for candidate in allowed_numbers):
            raise AgentOutputError(f"Specialist invented numeric value: {value:g}")
    evidence_text = json.dumps(facts, ensure_ascii=False, default=str).casefold()
    for month in MONTH_WORD.findall(text):
        if month.casefold() not in evidence_text:
            raise AgentOutputError(f"Specialist invented or changed date context: {month}")
    if EXTERNAL_COMPLETION.search(text):
        raise AgentOutputError("Specialist claimed an external action was completed.")


_SOURCE_NUMBER = r"\d+(?:[.,]\d+)?"
_SOURCE_DURATION_RANGE = re.compile(
    rf"(?<!\w)({_SOURCE_NUMBER})\s*(?:-|\u2013|\u2014|to)\s*({_SOURCE_NUMBER})"
    rf"(?:\s+[a-z]+){{0,3}}\s+days?\b",
    re.I,
)
_SOURCE_DURATION_SINGLE = re.compile(
    rf"(?<!\w)({_SOURCE_NUMBER})(?:\s+[a-z]+){{0,3}}\s+days?\b",
    re.I,
)


def _source_duration_ranges(text: str) -> list[tuple[float, float]]:
    ranges = [
        (float(low.replace(",", ".")), float(high.replace(",", ".")))
        for low, high in _SOURCE_DURATION_RANGE.findall(text)
    ]
    if ranges:
        return ranges
    return [
        (value, value)
        for raw in _SOURCE_DURATION_SINGLE.findall(text)
        for value in [float(raw.replace(",", "."))]
    ]


def _validate_extracted_fact(item: _FactCardOutput, page: dict[str, Any], allowed_subjects: list[str]) -> None:
    if item.subject not in set(allowed_subjects):
        raise AgentOutputError("Specialist returned a subject outside the core allow-list.")
    page_card = page.get("card") if isinstance(page.get("card"), dict) else {}
    trusted_subject = page.get("subject") or page_card.get("subject")
    if trusted_subject not in set(allowed_subjects):
        raise AgentOutputError("Source page has no trusted subject binding.")
    if item.subject != trusted_subject:
        raise AgentOutputError("Specialist subject does not match the selected source page.")
    if not (math.isfinite(item.low) and math.isfinite(item.high) and 0 < item.low <= item.high):
        raise AgentOutputError("Specialist returned a non-finite or unordered duration.")
    source_ranges = _source_duration_ranges(str(page.get("text") or ""))
    grounded = any(
        math.isclose(item.low, low, rel_tol=0, abs_tol=1e-9)
        and math.isclose(item.high, high, rel_tol=0, abs_tol=1e-9)
        for low, high in source_ranges
    )
    if not grounded:
        raise AgentOutputError("Specialist duration is not explicitly grounded in the selected source page.")


class LocalAgentRuntime:
    mode = "local"

    def draft(self, agent, draft_id, *, facts, lang, now, trace_ids=None) -> dict[str, Any]:
        result = local_drafts.draft(
            agent,
            draft_id,
            facts=facts,
            lang=lang,
            now=now,
            trace_ids=trace_ids,
        )
        return {**result, "agent_runtime": self.mode, "model": None}

    def extract_fact_cards(self, agent, pages, *, allowed_subjects, now) -> list[dict[str, Any]]:
        _authorize(agent, "search_public", now=now)
        return [
            {
                "card": fact_gateway.read(page, now=now),
                "agent_runtime": self.mode,
                "model": None,
                "response_id": None,
            }
            for page in pages
        ]


class OpenAIAgentRuntime:
    mode = "openai"

    def __init__(self, settings: Settings, *, client: Any):
        self.settings = settings
        self.client = client

    def _run(self, *, agent: dict[str, Any], task: str, payload: dict[str, Any], output_model: type[BaseModel]) -> tuple[BaseModel, str | None]:
        template = agent.get("template") or {}
        instructions = (
            f"You are the CURE {agent['title']} specialist. Complete exactly one bounded task assigned by the CURE core. "
            "Return only the requested structured output. Use only supplied evidence. Never invent numbers or dates. "
            "Never claim to send, pay, book, sign, submit, or mutate state. The resident sends drafts. "
            f"Forbidden capabilities: {json.dumps(template.get('forbidden_display') or [], ensure_ascii=False)}. "
            f"Task: {task}"
        )
        try:
            response = self.client.responses.create(
                model=self.settings.openai_model,
                store=False,
                reasoning={"effort": self.settings.reasoning_effort},
                instructions=instructions,
                input=json.dumps(_sanitize(payload), ensure_ascii=False, default=str),
                text={"format": {
                    "type": "json_schema",
                    "name": output_model.__name__.lower(),
                    "strict": True,
                    "schema": output_model.model_json_schema(),
                }},
            )
            text = str(getattr(response, "output_text", "")).strip()
            if not text:
                raise ValueError("empty specialist output")
            return output_model.model_validate_json(text), getattr(response, "id", None)
        except AgentRuntimeError:
            raise
        except TimeoutError as error:
            raise AgentRuntimeError("agent_timeout", type(error).__name__) from error
        except Exception as error:
            if isinstance(error, ValueError):
                raise AgentOutputError(type(error).__name__) from error
            raise AgentRuntimeError("agent_provider_error", type(error).__name__) from error

    def draft(self, agent, draft_id, *, facts, lang, now, trace_ids=None) -> dict[str, Any]:
        _authorize(agent, "draft_text", now=now)
        allowed_keys = local_drafts.fact_keys(draft_id)
        clean_facts = _sanitize({key: facts[key] for key in allowed_keys if key in facts})
        output, response_id = self._run(
            agent=agent,
            task=(f"Choose one approved writing variant for the {draft_id} draft in language '{lang}'. "
                  "The core will render the final text from supplied facts."),
            payload={"draft_id": draft_id, "language": lang, "facts": clean_facts,
                     "approved_variants": {
                         "standard": "complete context and request",
                         "concise": "shortest complete version",
                     }},
            output_model=_DraftOutput,
        )
        assert isinstance(output, _DraftOutput)
        rendered = local_drafts.draft(
            agent,
            draft_id,
            facts=clean_facts,
            lang=lang,
            now=now,
            trace_ids=trace_ids,
            variant=output.variant,
        )
        _validate_draft_text(rendered["text"], clean_facts)
        return {**rendered,
            "agent_runtime": self.mode,
            "model": self.settings.openai_model,
            "response_id": response_id,
        }

    def extract_fact_cards(self, agent, pages, *, allowed_subjects, now) -> list[dict[str, Any]]:
        _authorize(agent, "search_public", now=now)
        public_pages = [
            {"page_index": index, "text": str(page.get("text") or "")}
            for index, page in enumerate(pages)
        ]
        output, response_id = self._run(
            agent=agent,
            task=(
                "Extract at most one explicit step-duration fact from each supplied public page. "
                "Do not infer missing values. page_index must identify the supplied page."
            ),
            payload={
                "pages": public_pages,
                "allowed_claims": ["step_duration"],
                "allowed_subjects": list(allowed_subjects),
                "allowed_unit": "days",
            },
            output_model=_FactCardsOutput,
        )
        assert isinstance(output, _FactCardsOutput)
        if len(output.cards) > len(pages):
            raise AgentOutputError("Specialist returned more fact cards than source pages.")
        seen: set[int] = set()
        results = []
        for item in output.cards:
            if item.page_index >= len(pages) or item.page_index in seen:
                raise AgentOutputError("Specialist returned an invalid or duplicate page index.")
            seen.add(item.page_index)
            page = pages[item.page_index]
            _validate_extracted_fact(item, page, allowed_subjects)
            results.append({
                "card": {
                    "claim": item.claim,
                    "subject": item.subject,
                    "low": float(item.low),
                    "high": float(item.high),
                    "unit": item.unit,
                    "source_url": str(page["url"]),
                    "retrieved_at": now.isoformat(),
                },
                "agent_runtime": self.mode,
                "model": self.settings.openai_model,
                "response_id": response_id,
            })
        return results


def build_agent_runtime(settings: Settings, *, client: Any | None = None) -> AgentRuntime:
    if settings.language_mode == "local":
        return LocalAgentRuntime()
    if not settings.openai_api_key:
        raise GatewayConfigurationError("OPENAI_API_KEY is required for OpenAI resident agents")
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise GatewayConfigurationError("The openai package is required for OpenAI resident agents") from error
        client = OpenAI(api_key=settings.openai_api_key, timeout=60, max_retries=0)
    return OpenAIAgentRuntime(settings, client=client)
