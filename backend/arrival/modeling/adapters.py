"""Explicit adapter map from generic operations to existing CURE session services."""
from __future__ import annotations

from typing import Any, Callable

from arrival.profiles import ProfileService, ProfileValueError

from .registry import RegistryError
from .results import BackendResult


def reply_result(operation: str, reply: dict[str, Any], *, inputs: dict[str, Any], response_id: str | None, call_id: str | None) -> BackendResult:
    cards = list(reply.get("cards") or [])
    traces = list(reply.get("trace_ids") or [])
    refused = next((card for card in cards if card.get("type") == "refused"), None)
    missing = list((refused or {}).get("answer", {}).get("missing") or [])
    status = "missing_data" if refused else "computed"
    return BackendResult(
        status=status,
        operation=operation,
        inputs=inputs,
        facts={"route": reply.get("route"), "cards": cards},
        cards=cards,
        pending=list(reply.get("pending") or []),
        evidence=traces,
        calculation_status="not_computed" if refused else "computed",
        missing=missing,
        external_action=False,
        message=str(reply.get("text") or ""),
        local_text=str(reply.get("text") or ""),
        plan_response_id=response_id,
        tool_call_id=call_id,
    )


class SessionAdapters:
    def __init__(self, registry):
        self.registry = registry
        self.profiles = ProfileService(registry)

    def execute(self, operation: str, args: dict[str, Any], session, *, response_id: str | None, call_id: str | None) -> BackendResult:
        handler: Callable = getattr(self, f"_{operation}")
        result = handler(args, session, response_id=response_id, call_id=call_id)
        return result.model_copy(update={"plan_response_id": response_id, "tool_call_id": call_id})

    def _read_state(self, args, session, **ids) -> BackendResult:
        if args.get('domain') == 'relocation':
            return reply_result('read_state', session.plan(), inputs=args, **ids)
        domain = args["domain"]
        if domain == "profile":
            facts = self.profiles.snapshot(session)
        elif domain == "city":
            city = session.S["city"]
            facts = {
                "districts": sorted(city.raw("rent_aed_month")),
                "sources": city.sources(),
                "data_mode": city.data_mode,
                "clock": session.now.isoformat(),
            }
        elif domain == "cohort":
            facts = {"resident_count": len(session.rt.sessions), "extra_city_families": len(session.rt.city["extra_families"])}
        else:
            facts = {"domain": domain, "available": True}
        return BackendResult(status="computed", operation="read_state", inputs=args, facts=facts, **ids)

    def _record_event(self, args, session, **ids) -> BackendResult:
        observations = args.get("observations") or {}
        status = observations.get("status", "known")
        try:
            self.profiles.record(
                session,
                args["variable_id"],
                value=observations.get("value"),
                status=status,
                source=observations.get("source", "user"),
            )
        except (ProfileValueError, RegistryError) as error:
            return BackendResult(status="unsupported_request", operation="record_event", inputs=args, message=str(error), local_text=str(error), calculation_status="not_computed", **ids)
        facts = {args["variable_id"]: self.profiles.snapshot(session)[args["variable_id"]]}
        return BackendResult(status="computed", operation="record_event", inputs=args, facts=facts, **ids)

    def _simulate(self, args, session, **ids) -> BackendResult:
        model = args["model_id"]
        missing = self.profiles.missing_for_model(session, model)
        if missing:
            message = "Missing required profile values: " + ", ".join(missing)
            return BackendResult(status="missing_data", operation="simulate", inputs=args, missing=missing, message=message, local_text=message, calculation_status="not_computed", **ids)
        interventions = args.get("interventions") or {}
        if model == "relocation_plan":
            if "visa_delay_days" in interventions:
                value = interventions["visa_delay_days"]
                value = value.get("value") if isinstance(value, dict) else value
                reply = session.visa_delay(float(value))
            else:
                reply = session.plan()
        elif model == "thermal_home":
            reply = session.apartment()
        elif model == "outdoor_plan":
            reply = session.m_wed_windows()
        elif model == "weekly_plan":
            reply = session.m_mon_morning()
        elif model == "city_context":
            reply = session.vision_chain()
        else:
            return BackendResult(status="insufficient_model", operation="simulate", inputs=args, message=f"No simulation adapter for {model}.", local_text=f"No simulation adapter for {model}.", calculation_status="not_computed", **ids)
        return reply_result("simulate", reply, inputs=args, **ids)

    def _compare(self, args, session, **ids) -> BackendResult:
        model = args["model_id"]
        if model == "relocation_plan":
            reply = session.visa_counterfactual()
        elif model == "cohort_coordination":
            reply = session.vision_cohort()
        else:
            return BackendResult(status="insufficient_model", operation="compare", inputs=args, message=f"No comparison adapter for {model}.", local_text=f"No comparison adapter for {model}.", calculation_status="not_computed", **ids)
        return reply_result("compare", reply, inputs=args, **ids)

    def _optimize(self, args, session, **ids) -> BackendResult:
        model = args["model_id"]
        handlers = {"district_cost": session.districts, "outdoor_plan": session.m_wed_windows, "weekly_plan": session.m_mon_morning}
        if model not in handlers:
            return BackendResult(status="insufficient_model", operation="optimize", inputs=args, message=f"No optimization adapter for {model}.", local_text=f"No optimization adapter for {model}.", calculation_status="not_computed", **ids)
        return reply_result("optimize", handlers[model](), inputs=args, **ids)

    def _explain(self, args, session, **ids) -> BackendResult:
        output = args["output_id"]
        handlers = {
            "relocation_ready_days": session.visa_why,
            "district_recommendation": session.district_why,
        }
        if output not in handlers:
            return BackendResult(status="insufficient_model", operation="explain", inputs=args,
                                 message=f"No explanation adapter for {output}.",
                                 local_text=f"No explanation adapter for {output}.",
                                 calculation_status="not_computed", **ids)
        reply = handlers[output]()
        return reply_result("explain", reply, inputs=args, **ids)

    def _find_levers(self, args, session, **ids) -> BackendResult:
        if args["output_id"] != "relocation_ready_days":
            return BackendResult(status="insufficient_model", operation="find_levers", inputs=args, message="No registered lever adapter for this output.", local_text="No registered lever adapter for this output.", calculation_status="not_computed", **ids)
        return reply_result("find_levers", session.visa_levers(), inputs=args, **ids)

    def _propose_action(self, args, session, **ids) -> BackendResult:
        message = f"Action '{args['action']}' requires explicit approval."
        return BackendResult(status="approval_required", operation="propose_action", inputs=args, message=message, local_text=message, external_action=False, calculation_status="not_computed", **ids)
