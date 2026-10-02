"""Одно управляемое действие низкого риска: уставка температуры зоны HVAC (ТЗ Sovereign §11 п.4, §3.4 пример).

Проверка: «статус контроллера + ожидаемое направление/изменение телеметрии в допустимом окне»:
  • статус контроллера — обратное чтение уставки |sp_readback − sp| ≤ tolerance;
  • направление — температура зоны сдвинулась к новой уставке минимум на min_detectable_change.
Все пороги (допуск, минимально различимое изменение, окно, пределы) — параметры площадки, не константы кода.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from .contracts import ActionContract, ExpectedEffect, SovereignError, finite, text

HVAC_SETPOINT_ACTION_CLASS = "hvac_setpoint"
HVAC_SETPOINT_COMMAND = "write_setpoint"


def build_hvac_setpoint_action(*, action_id: str, trace_id: str, idempotency_key: str, zone_entity_id: str,
                               setpoint_point: str, setpoint_readback_point: str, zone_temp_point: str,
                               new_setpoint: float, current_zone_temp: float, readback_tolerance: float,
                               min_detectable_change: float, verification_window_s: float,
                               setpoint_limits: tuple[Optional[float], Optional[float]], safe_setpoint: float,
                               capability_version: str, preconditions: tuple[str, ...], expiry_at: datetime,
                               authority_tier: str, policy_refs: tuple[str, ...], tenant: str, site: str,
                               risk_tier: str, approval_ref: Optional[str] = None) -> ActionContract:
    sp, cur = finite("new_setpoint", new_setpoint), finite("current_zone_temp", current_zone_temp)
    tol, dmin = finite("readback_tolerance", readback_tolerance), finite("min_detectable_change", min_detectable_change)
    if tol < 0 or dmin <= 0:
        raise SovereignError("readback tolerance must be ≥ 0 and the detectable change > 0 (site calibration)")
    if abs(sp - cur) < dmin:
        raise SovereignError("setpoint change is below the detectable change: the effect could not be verified")
    zp = text("zone_temp_point", zone_temp_point)
    direction = ExpectedEffect(zp, None, cur - dmin, verification_window_s) if sp < cur \
        else ExpectedEffect(zp, cur + dmin, None, verification_window_s)
    readback = ExpectedEffect(text("setpoint_readback_point", setpoint_readback_point), sp - tol, sp + tol,
                              verification_window_s)
    return ActionContract(
        action_id=action_id, trace_id=trace_id, idempotency_key=idempotency_key, target_entity_id=zone_entity_id,
        action_type=HVAC_SETPOINT_COMMAND, parameters={text("setpoint_point", setpoint_point): sp},
        capability_version=capability_version, preconditions=tuple(preconditions),
        physical_limits={setpoint_point: tuple(setpoint_limits)}, expected_effect=(readback, direction),
        expiry_at=expiry_at, authority_tier=authority_tier, policy_refs=tuple(policy_refs), approval_ref=approval_ref,
        safe_state={setpoint_point: finite("safe_setpoint", safe_setpoint)},
        compensation_plan=(f"restore {setpoint_point} to safe setpoint {safe_setpoint}", "notify operator"),
        tenant=tenant, site=site, risk_tier=risk_tier, action_class=HVAC_SETPOINT_ACTION_CLASS)


__all__ = ["HVAC_SETPOINT_ACTION_CLASS", "HVAC_SETPOINT_COMMAND", "build_hvac_setpoint_action"]
