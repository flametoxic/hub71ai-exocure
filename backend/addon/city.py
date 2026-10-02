"""Sourced Abu Dhabi context used by the existing CURE formulas."""
from __future__ import annotations

import math
from pathlib import Path

import yaml

DATA = Path(__file__).parent / "data"


def load_city() -> dict:
    return yaml.safe_load((DATA / "city_abudhabi.yaml").read_text(encoding="utf-8"))


def _curve(low: float, high: float, peak_hour: int = 15) -> list[float]:
    return [
        round(low + (high - low) * max(0.0, math.cos((hour - peak_hour) / 24 * 2 * math.pi)), 1)
        for hour in range(24)
    ]


def apply_real_city(runtime) -> dict:
    """Replace only formula inputs backed by the supplied sourced city profile."""
    source = load_city()
    applied = []
    city = runtime.S["city"].doc
    for district, info in source["districts"].items():
        annual = info["rent_2br_aed_year"]
        monthly = round((annual["low"] + annual["high"]) / 2 / 12)
        city["rent_aed_month"][district] = {
            "value": monthly,
            "source": f"{info['source']} — midpoint of range / 12 ({info['kind']})",
        }
        applied.append(f"rent {district} = {monthly} AED/month")
    city["district_names"] = {district: info["name"] for district, info in source["districts"].items()}

    electricity = source["electricity_expat_apartment"]
    runtime.S["tariffs"].doc["electricity_per_kwh"]["residential"] = {
        "value": electricity["red_fils_per_kwh"] / 100,
        "source": (
            f"{electricity['source']} — red band for expat apartments, including VAT; "
            f"green band below {electricity['green_band_kwh_per_day']} kWh/day is not modelled"
        ),
    }
    applied.append(f"electricity = {electricity['red_fils_per_kwh'] / 100} AED/kWh")

    months = runtime.S["climate"].doc["months"]
    climate_source = source["climate"]["source"]
    for month, (low, high, humidity) in source["climate"]["months"].items():
        for key in [month] + (["aug_dust"] if month == "aug" else []):
            if key in months:
                months[key]["temperature_c"] = {"value": _curve(low, high), "source": climate_source}
                months[key]["humidity_pct"] = {"value": [humidity] * 24, "source": climate_source}
    applied.append("climate monthly temperature and humidity")

    for transaction in runtime.S["week"].doc.get("transactions", []):
        if transaction.get("category") == "rent":
            transaction["aed"] = city["rent_aed_month"]["A"]["value"]
    for session in runtime.sessions.values():
        session.cache.clear()
    runtime.city_profile_applied = applied
    return {"applied": applied}


def district_label(district: str, lang: str) -> str:
    info = load_city()["districts"].get(district)
    return f"{district} · {info['name'].get(lang, info['name']['en'])}" if info else district


def profile(lang: str = "en") -> dict:
    """Return display-ready city facts with provenance and data classification."""
    source = load_city()

    def localized(value):
        return value.get(lang, value.get("en")) if isinstance(value, dict) else value

    rows = []
    for district, info in source["districts"].items():
        rent = info["rent_2br_aed_year"]
        value = f"{rent['low']:,}–{rent['high']:,} AED / year (2-bed)".replace(",", " ")
        rows.append({
            "layer": "housing", "title": district_label(district, lang), "detail": localized(info["role"]),
            "value": value, "kind": info["kind"], "source": info["source"], "url": info["url"],
        })
    electricity = source["electricity_expat_apartment"]
    rows.append({
        "layer": "energy", "title": {"en": "Electricity (expat apartment)", "ru": "Электричество (квартира, экспат)"}[lang],
        "value": f"{electricity['green_fils_per_kwh']} / {electricity['red_fils_per_kwh']} fils/kWh",
        "detail": {"en": f"green band up to {electricity['green_band_kwh_per_day']} kWh/day", "ru": f"зелёный тариф до {electricity['green_band_kwh_per_day']} кВт·ч в день"}[lang],
        "kind": electricity["kind"], "source": electricity["source"], "url": electricity["url"],
    })
    taxi = source["taxi"]
    rows.append({
        "layer": "roads", "title": {"en": "Taxi", "ru": "Такси"}[lang],
        "value": f"{taxi['day_flag_fall_aed']:g} AED + {taxi['per_km_aed']} AED/km",
        "detail": {"en": f"minimum {taxi['minimum_fare_aed']} AED", "ru": f"минимум {taxi['minimum_fare_aed']} AED"}[lang],
        "kind": taxi["kind"], "source": taxi["source"], "url": taxi["url"],
    })
    toll = source["darb_toll"]
    rows.append({
        "layer": "roads", "title": "Darb", "value": f"{toll['fee_aed_per_crossing']} AED",
        "detail": localized(toll["peak"]), "kind": toll["kind"], "source": toll["source"], "url": toll["url"],
    })
    air = source["air_quality"]
    rows.extend([
        {
            "layer": "air", "title": {"en": "PM10 national limit (24 h)", "ru": "Предел PM10 (сутки)"}[lang],
            "value": f"{air['pm10_national_limit_24h_ugm3']} µg/m³", "detail": air["limit_source"],
            "kind": air["kind"], "source": air["source"], "url": air["url"],
        },
        {
            "layer": "air", "title": {"en": "Dust event days, 2023", "ru": "Пыльные дни, 2023"}[lang],
            "value": str(air["dust_event_days_2023"]), "detail": localized(air["dust_event_definition"]),
            "kind": air["kind"], "source": air["source"], "url": air["url"],
        },
    ])
    climate = source["climate"]
    hottest = max(climate["months"].items(), key=lambda item: item[1][1])
    rows.append({
        "layer": "climate", "title": {"en": "Hottest month (average max)", "ru": "Самый жаркий месяц (средний максимум)"}[lang],
        "value": f"{hottest[0]} · {hottest[1][1]:g} °C", "detail": "", "kind": climate["kind"],
        "source": climate["source"], "url": climate["url"],
    })
    for rhythm in source["city_rhythm"]:
        rows.append({
            "layer": "rhythm", "title": localized(rhythm), "value": "", "detail": "", "kind": rhythm["kind"],
            "source": rhythm["source"], "url": rhythm.get("url"),
        })
    hub = source["hub71"]
    rows.append({
        "layer": "work", "title": "Hub71", "value": hub["address"], "detail": localized(hub["incentives"]),
        "kind": hub["kind"], "source": "hub71.com", "url": hub["incentives_url"],
    })
    indoor = source["indoor_places"]
    rows.append({
        "layer": "leisure", "title": {"en": "Indoor places for dusty days", "ru": "Места в помещении на пыльные дни"}[lang],
        "value": ", ".join(indoor["items"][:5]) + " …", "detail": "", "kind": indoor["kind"],
        "source": "Visit Abu Dhabi", "url": indoor["url"],
    })
    return {
        "collected": source["meta"]["collected"], "rows": rows, "climate_months": climate["months"],
        "still_synthetic": [localized(item) for item in source["still_synthetic"]],
        "kinds": {
            "official": {"en": "official", "ru": "официальный"}[lang],
            "market": {"en": "market snapshot (not official)", "ru": "рынок, снимок (не официальный)"}[lang],
            "model": {"en": "model data (not official)", "ru": "модельные данные (не официальные)"}[lang],
        },
    }
