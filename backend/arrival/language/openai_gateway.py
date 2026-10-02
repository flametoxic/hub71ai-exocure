"""OpenAI Responses API adapter for strict planning and evidence-only narration."""
from __future__ import annotations

import json
from typing import Any

from arrival.config import Settings
from arrival.person import sanitize_for_llm

from .contracts import BackendResultLike, GatewayError, LanguageRequest, NarrationResult, QueryPlan, ToolCall


PLANNER_PROMPT = """Map the user's request to registered CURE tools only.
Do not calculate, guess values, create variables, create domains, or claim an action happened.
Use only canonical identifiers from the tools. Represent interventions as the strict assignment array declared by
the tool schema. Use simulate with relocation_plan to compute a relocation plan; use read_state to inspect state.
Use create_specialist for unknown public-research tasks outside registered models. Never map a new task
to an unrelated model merely because it involves moving, documents, money or family.
If no registered tool applies, make no tool call."""

NARRATOR_PROMPT = """Write the final answer in the user's language using only the backend tool output.
Do not add numbers, dates, facts, diagnoses, formulas, or completed actions.
Preserve synthetic, shadow, pending, unsupported, and missing-data qualifications."""


class OpenAILanguageGateway:
    def __init__(self, settings: Settings, *, client: Any):
        self.settings = settings
        self.client = client
        self._plan_items = []

    @staticmethod
    def _request_json(request):
        return json.dumps(sanitize_for_llm(request.model_dump()), ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _failure(error: Exception) -> GatewayError:
        return GatewayError("timeout" if isinstance(error, TimeoutError) else "provider_error", type(error).__name__)

    def plan(self, request: LanguageRequest, catalog: list[dict[str, Any]]) -> QueryPlan:
        try:
            response = self.client.responses.create(
                model=self.settings.openai_model,
                store=False,
                include=['reasoning.encrypted_content'],
                reasoning={"effort": self.settings.reasoning_effort},
                instructions=PLANNER_PROMPT,
                input=[{"role": "user", "content": self._request_json(request)}],
                tools=catalog,
            )
            operations = []
            self._plan_items = [item.model_dump(exclude_none=True) for item in getattr(response, "output", [])
                                if hasattr(item, "model_dump")]
            for item in getattr(response, "output", []):
                if getattr(item, "type", None) != "function_call":
                    continue
                arguments = json.loads(item.arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be an object")
                operations.append(ToolCall(name=item.name, arguments=arguments, call_id=getattr(item, "call_id", None)))
            return QueryPlan(source="openai", operations=operations, response_id=getattr(response, "id", None))
        except GatewayError:
            raise
        except Exception as error:
            raise self._failure(error) from error

    def narrate(
        self,
        request: LanguageRequest,
        result: BackendResultLike,
        feedback: dict[str, Any] | None = None,
    ) -> NarrationResult:
        payload = sanitize_for_llm(dict(result))
        previous_response_id = payload.pop("plan_response_id", None)
        tool_outputs = list(payload.pop("tool_outputs", []) or [])
        had_tool_call = bool(payload.get("tool_call_id"))
        call_id = payload.pop("tool_call_id", None) or "backend-result"
        if feedback:
            payload["narration_validation_feedback"] = feedback
        try:
            input_items = ([{
                "type": "function_call_output",
                "call_id": str(item["call_id"]),
                "output": json.dumps(item["output"], ensure_ascii=False, default=str),
            } for item in tool_outputs] if tool_outputs else ([{
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(payload, ensure_ascii=False, default=str),
            }] if had_tool_call else [{
                "role": "user",
                "content": "Backend result: " + json.dumps(payload, ensure_ascii=False, default=str),
            }]))
            input_items.append({
                "role": "user",
                "content": "User request: " + self._request_json(request)
                + ("\nValidation feedback: " + json.dumps(feedback, ensure_ascii=False, default=str) if feedback else ""),
            })
            response = self.client.responses.create(
                model=self.settings.openai_model,
                store=False,
                reasoning={"effort": self.settings.reasoning_effort},
                instructions=NARRATOR_PROMPT,
                input=(self._plan_items + input_items) if previous_response_id and self._plan_items else input_items,
            )
            text = str(getattr(response, "output_text", "")).strip()
            if not text:
                raise ValueError("empty narration")
            return NarrationResult(
                text=text,
                source="openai",
                model=self.settings.openai_model,
                response_id=getattr(response, "id", None),
            )
        except GatewayError:
            raise
        except Exception as error:
            raise self._failure(error) from error
