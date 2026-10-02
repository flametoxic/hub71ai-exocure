"""Execution of a fully validated plan through explicit backend adapters."""
from __future__ import annotations

from .adapters import SessionAdapters
from .results import BackendResult, ValidatedPlan


class QueryExecutor:
    def __init__(self, registry):
        self.adapters = SessionAdapters(registry)

    def execute(self, plan: ValidatedPlan | BackendResult, session) -> BackendResult:
        if isinstance(plan, BackendResult):
            return plan
        results = [
            self.adapters.execute(call.name, call.arguments, session, response_id=plan.response_id, call_id=call.call_id)
            for call in plan.operations
        ]
        if len(results) == 1:
            return results[0]
        failed = next((result for result in results if result.status != "computed"), None)
        if failed:
            return failed
        last = results[-1]
        serialized = [result.model_dump() for result in results]
        tool_outputs = [
            {"call_id": result.tool_call_id, "output": payload}
            for result, payload in zip(results, serialized)
            if result.tool_call_id
        ]
        return last.model_copy(update={"facts": {"operations": serialized}, "tool_outputs": tool_outputs})
