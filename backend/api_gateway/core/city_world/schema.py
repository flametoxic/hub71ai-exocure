"""EXO City World Model — схема данных (§1–§7): реестр, граф, телеметрия, подграфы города.

    entity_id = UUIDv7,  entity_type ∈ T;   state_key = (entity_id, metric_id, timestamp_ns)
    G_t = (V, E, A_t) — сущности, отношения, атрибуты/состояния
Пять представлений вместо «широкой CSV»: реестр сущностей, пространственно-сетевой граф, телеметрия
временных рядов, журнал событий (append-only), реестр политик и согласий. Реестр/граф/кадры — это
core.spatial_world; здесь — каталог типов и метрик города и контракт телеметрии. Метрики несут единицу
СИ (или каноническую), класс атрибута (для TTL) и класс приватности.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional

from ..world_model.reality_formulas import FormulaReference

DOC = "EXO_City_World_Model_schema_subgraphs_causal_engine"
F_STATE_KEY = FormulaReference("CITY-1-STATE-KEY", DOC, 1, "entity_id = UUIDv7, entity_type ∈ T; "
                                                           "state_key = (entity_id, metric_id, timestamp_ns)")
PRIVACY = ("public", "operational", "personal", "sensitive")
ENTITY_STATUS = ("active", "inactive", "maintenance", "decommissioned")
EDGE_TYPES = ("contains", "connected_to", "supplies", "controls", "observes", "depends_on", "causes", "serves",
              "attracts", "loads")


class CitySchemaError(ValueError):
    pass


def uuid7() -> str:
    """UUIDv7 (RFC 9562): 48 бит миллисекунд Unix + версия 7 + вариант + случайные биты — сортируется по времени."""
    ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rnd = int.from_bytes(os.urandom(10), "big")
    val = (ms << 80) | (0x7 << 76) | ((rnd >> 68) & 0xFFF) << 64 | (0b10 << 62) | (rnd & ((1 << 62) - 1))
    return str(uuid.UUID(int=val))


def is_uuid7(s: str) -> bool:
    try:
        u = uuid.UUID(str(s))
    except ValueError:
        return False
    return u.version == 7


@dataclass(frozen=True)
class Metric:
    metric_id: str
    unit: str
    attribute_class: str        # telemetry | state | config
    privacy: str = "operational"
    kind: str = "num"           # num | text | bool | ref (ссылка/хэш — не сырые данные)


def _m(names: str, unit: str, cls: str = "telemetry", privacy: str = "operational", kind: str = "num") -> dict:
    return {n.strip(): Metric(n.strip(), unit, cls, privacy, kind) for n in names.split(",") if n.strip()}


# §3–§7: подграфы (сокращённо — полный перечень полей из документа; единицы канонические)
CATALOG: dict[str, dict[str, Metric]] = {
    "district": {**_m("total_population_current", "count", "state"), **_m("grid_load_mw", "MW"),
                 **_m("grid_capacity_remaining_pct,district_solar_irradiance_wm2", "1"),
                 **_m("water_main_pressure_bar", "bar"), **_m("district_weather_temp_c", "degC"),
                 **_m("emergency_readiness_level,macro_weather_anomaly", "1", "state", kind="text"),
                 **_m("public_event_active,vip_movement_flag", "1", "state", kind="bool"),
                 **_m("event_expected_attendance", "count", "state"),
                 **_m("dynamic_electricity_tariff_multiplier,holiday_effect_score", "1", "state"),
                 **_m("salik_toll_gate_price", "AED", "config"), **_m("municipal_waste_fleet_active", "count")},
    "building": {**_m("total_floors,total_apartments", "count", "config"), **_m("current_occupancy_pct", "1"),
                 **_m("building_energy_mode,building_security_mode,roof_helipad_status", "1", "state", kind="text"),
                 **_m("total_building_power_draw_kw", "kW"), **_m("structural_health_vibration", "m/s"),
                 **_m("facade_cleaning_robot_active", "1", "state", kind="bool")},
    "chiller_plant": {**_m("chiller_load_pct", "1"), **_m("chilled_water_supply_temp_c,chilled_water_return_temp_c", "degC"),
                      **_m("chilled_water_flow_rate", "kg/s")},
    "utilities": {**_m("water_roof_tank_level_pct,groundwater_level_sensor", "1"), **_m("water_main_pressure_bar", "bar"),
                  **_m("sewage_main_flow_rate", "m3/s"), **_m("central_exhaust_fan_rpm", "Hz"),
                  **_m("fiber_optic_trunk_status,fire_alarm_central_status", "1", "state", kind="text")},
    "floor": {**_m("corridor_lighting_pct", "1"), **_m("floor_air_quality_co2_ppm", "ppm"),
              **_m("trash_chute_door_status", "1", "state", kind="text"),
              **_m("corridor_camera_motion_detected,floor_evacuation_doors_unlocked", "1", kind="bool")},
    "unit": {**_m("smart_meter_power_kw", "kW"), **_m("smart_meter_water_liters", "L"),
             **_m("water_leakage_sensor_active,appliance_heavy_load_active", "1", kind="bool"),
             **_m("smart_thermostat_setpoint_c", "degC", "config"), **_m("current_indoor_temp_c", "degC"),
             **_m("window_blinds_open_pct", "1"), **_m("unit_occupancy_status,smart_lock_status", "1", "state",
                                                       "personal", "text")},
    "retail_unit": {**_m("business_type,operating_status", "1", "state", kind="text"),
                    **_m("current_customer_count", "count"), **_m("hvac_power_draw_kw", "kW"),
                    **_m("outdoor_seating_active,loading_zone_occupied,promotional_screen_active", "1", kind="bool"),
                    **_m("delivery_robot_dock_status", "1", "state", kind="text")},
    "traffic_light": {**_m("tl_current_color", "1", kind="text"), **_m("tl_timer_seconds_left", "s"),
                      **_m("tl_green_phase_duration,tl_red_phase_duration", "s", "config"),
                      **_m("tl_is_smart_adaptive,tl_pedestrian_button_pressed,tl_camera_feed_active,tl_maintenance_mode,"
                           "tl_emergency_override_active", "1", kind="bool")},
    "intersection": {**_m("intersection_type", "1", "config", kind="text"), **_m("intersection_capacity_vph", "count/h", "config"),
                     **_m("intersection_current_load", "count/h")},
    "road_segment": {**_m("lane_count", "count", "config"), **_m("autonomous_lane_exists", "1", "config", kind="bool"),
                     **_m("speed_limit_kmh", "km/h", "config"), **_m("current_average_speed_kmh", "km/h"),
                     **_m("traffic_density_index", "1"), **_m("accident_hazard_flag,pothole_detected_flag", "1", kind="bool"),
                     **_m("weather_surface_condition", "1", kind="text")},
    "vehicle": {**_m("telemetry_gps_lat,telemetry_gps_lon", "deg", privacy="personal"), **_m("altitude_m", "m"),
                **_m("current_speed_kmh", "km/h"), **_m("acceleration_ms2", "m/s2"),
                **_m("v2x_connected_flag,autopilot_engaged_flag,emergency_siren_active,hazard_lights_active,"
                     "route_deviation_flag", "1", kind="bool"),
                **_m("v2v_mesh_peers,passenger_count", "count"), **_m("battery_fuel_level_pct", "1"),
                **_m("tire_traction_status", "1", kind="text"), **_m("carbon_emission_rate_gs", "g/s"),
                **_m("lidar_point_cloud_ref", "1", kind="ref"), **_m("v2x_latency_ms", "ms"),
                **_m("perception_confidence", "1")},
    "pedestrian_cluster": {**_m("mobility_aid_used_probability,carrying_heavy_items_probability,"
                                "sos_gesture_detected_probability,waiting_for_taxi_probability,pet_detected_probability",
                                "1", privacy="personal"),
                           **_m("movement_vector_x,movement_vector_y,estimated_walking_speed_mps", "m/s")},
    "metro_entrance": {**_m("escalator_status,nol_card_reader_status,hvac_station_status,security_scanner_status", "1",
                            "state", kind="text"),
                       **_m("turnstile_crossings_per_min", "count/min"), **_m("platform_crowd_level_pct", "1"),
                       **_m("train_arriving_in_sec", "s"), **_m("pa_system_active", "1", kind="bool"),
                       **_m("platform_safe_capacity", "count", "config"), **_m("platform_occupancy", "count")},
    "parking_zone": {**_m("parking_total_spots", "count", "config"),
                     **_m("parking_available_spots,parking_ev_chargers_available", "count"),
                     **_m("loading_zone_occupied", "1", kind="bool")},
    "airspace": {**_m("evtol_traffic_density", "1"), **_m("delivery_drone_active_count", "count"),
                 **_m("low_altitude_wind_gust_ms", "m/s"), **_m("drone_landing_pad_status", "1", "state", kind="text")},
    "universal_device": {**_m("current_micro_state", "1", "state", kind="text"),
                         **_m("firmware_version_hash,certificate_id,ota_channel", "1", "config", kind="text"),
                         **_m("last_heartbeat_ms", "ms"), **_m("local_compute_capacity_flops", "1", "config"),
                         **_m("device_health_score", "1")},
    "edge_node": {**_m("network_bandwidth_utilization_pct,edge_node_gpu_utilization_pct,packet_loss_pct", "1"),
                  **_m("v2x_latency_ms,edge_inference_queue_ms", "ms")},
}
ENTITY_TYPES = tuple(sorted(CATALOG)) + ("pedestrian_crossing", "elevator_bank", "elevator_car", "water_tank",
                                         "fire_alarm_panel", "exhaust_fan", "water_main", "sewage_main", "fiber_trunk",
                                         "camera", "person_agent", "bus_stop", "metro_station", "evtol_corridor",
                                         "cell_tower", "loading_zone", "cooling_plant")


def metric(entity_type: str, metric_id: str) -> Metric:
    m = CATALOG.get(entity_type, {}).get(metric_id)
    if m is None:
        raise CitySchemaError(f"metric {metric_id!r} is not in the catalog for {entity_type!r}")
    return m


def spatial_world_attribute_classes() -> dict[str, str]:
    """Классы атрибутов для SpatialWorldPolicy.attribute_class (TTL по классу)."""
    out: dict[str, str] = {}
    for ms in CATALOG.values():
        for m in ms.values():
            out.setdefault(m.metric_id, m.attribute_class)
    return out


@dataclass(frozen=True)
class TelemetryObservation:
    """§2.3 telemetry_observations; event time и время поступления разделены (порядок событий)."""

    observation_id: str
    timestamp_ns: int
    ingested_at: datetime
    entity_id: str
    entity_type: str
    metric_id: str
    value: Any
    unit: str
    quality_score: float
    confidence: float
    sensor_id: str
    schema_version: str
    privacy_class: str

    def __post_init__(self) -> None:
        if not is_uuid7(self.observation_id):
            raise CitySchemaError("observation_id must be UUIDv7")
        if not isinstance(self.timestamp_ns, int) or self.timestamp_ns <= 0:
            raise CitySchemaError("timestamp_ns must be a positive integer (source time, ns)")
        m = metric(self.entity_type, self.metric_id)
        if self.unit != m.unit:
            raise CitySchemaError(f"{self.metric_id}: unit {self.unit!r} ≠ catalog {m.unit!r}")
        if m.kind == "num" and (isinstance(self.value, bool) or not isinstance(self.value, (int, float))):
            raise CitySchemaError(f"{self.metric_id} expects a number")
        if m.kind == "bool" and not isinstance(self.value, bool):
            raise CitySchemaError(f"{self.metric_id} expects a bool")
        if m.kind == "ref" and not (isinstance(self.value, str) and self.value.startswith(("sha256:", "ref:", "s3://"))):
            raise CitySchemaError(f"{self.metric_id} is a reference/hash, never a raw blob")
        for n in ("quality_score", "confidence"):
            if not 0.0 <= float(getattr(self, n)) <= 1.0:
                raise CitySchemaError(f"{n} must be in [0, 1]")
        if self.privacy_class not in PRIVACY:
            raise CitySchemaError(f"privacy_class must be one of {PRIVACY}")
        if PRIVACY.index(self.privacy_class) < PRIVACY.index(m.privacy):
            raise CitySchemaError(f"{self.metric_id} is at least {m.privacy!r}; cannot be labelled {self.privacy_class!r}")

    @property
    def state_key(self) -> tuple:
        return (self.entity_id, self.metric_id, self.timestamp_ns)

    def to_event(self, *, source_namespace: str, trace_id: str, data_mode: str,
                 observed: Optional[Mapping[str, str]] = None) -> dict:
        """→ событие для spatial_world.ProjectionPipeline (единственный путь записи)."""
        from datetime import timezone
        t = datetime.fromtimestamp(self.timestamp_ns / 1e9, tz=timezone.utc)
        ev = {"observation_id": self.observation_id, "schema_version": self.schema_version, "data_mode": data_mode,
              "source_namespace": source_namespace, "source_id": self.sensor_id, "attribute": self.metric_id,
              "value": self.value, "unit": self.unit, "event_time": t, "knowledge_time": self.ingested_at,
              "source_ref": f"sensor:{self.sensor_id}", "integrity": "verified", "confidence": self.confidence,
              "uncertainty": 1.0 - self.quality_score, "trace_id": trace_id,
              "context": {"privacy_class": self.privacy_class}}
        if observed:
            ev["observed"] = dict(observed)
        return ev


__all__ = ["CATALOG", "DOC", "EDGE_TYPES", "ENTITY_STATUS", "ENTITY_TYPES", "F_STATE_KEY", "CitySchemaError", "Metric",
           "PRIVACY", "TelemetryObservation", "is_uuid7", "metric", "spatial_world_attribute_classes", "uuid7"]
