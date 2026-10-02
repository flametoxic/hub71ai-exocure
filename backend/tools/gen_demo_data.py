"""Generator of SYNTHETIC demo data for CURE.

Every number below is a placeholder for testing the mechanics, labelled source: synthetic-demo and data_mode: synthetic.
For a pilot they are replaced with real values from sources: NCM, DoE/ADDC, ADREC, ADEK, TAMM, OSM.
Values decided by the team or the person are labelled decided_by.
"*_ru" fields hold Russian titles used only in Russian replies.
Run: python tools/gen_demo_data.py  → rewrites arrival/demo/*.yaml
"""
import math
from pathlib import Path

import yaml

OUT = Path(__file__).parents[1] / "arrival" / "demo"
S = "synthetic-demo: REPLACE with a real sourced value"
SYN = {"data_mode": "synthetic", "source_kind": "synthetic", "source": S}


def v(x, src=S):
    return {"value": x, "source": src}


def team(x):
    return {"value": x, "decided_by": "team"}


def tri(a, b, c):
    return {"dist": "triangular", "low": a, "mode": b, "high": c}


def uni(a, b):
    return {"dist": "uniform", "low": a, "high": b, "source": S}


def curve(lo, hi, peak_h=15):
    return [round(lo + (hi - lo) * max(0.0, math.cos((h - peak_h) / 24 * 2 * math.pi)), 1) for h in range(24)]


def ghi(peak):
    return [round(max(0.0, peak * math.sin((h - 6) / 12 * math.pi)), 0) if 6 <= h <= 18 else 0.0 for h in range(24)]


MONTHS = [("jan", 31, 15, 25, 300, 650), ("feb", 28, 16, 26, 300, 720), ("mar", 31, 19, 30, 350, 800),
          ("apr", 30, 22, 34, 400, 870), ("may", 31, 26, 39, 450, 920), ("jun", 30, 29, 41, 500, 940),
          ("jul", 31, 31, 44, 500, 950), ("aug", 31, 31, 43, 550, 900), ("sep", 30, 28, 40, 450, 860),
          ("oct", 31, 25, 36, 400, 780), ("nov", 30, 20, 31, 350, 690), ("dec", 31, 17, 27, 300, 640)]


def month_doc(d, lo, hi, pm_peak, g, dust=False):
    pm = [40] * 7 + [pm_peak // 5 if not dust else pm_peak] * 10 + [60] * 7
    return {"days": v(d), "temperature_c": v(curve(lo, hi)), "temperature_sd_c": v(1.5),
            "humidity_pct": v([60] * 24), "pm10_ugm3": v(pm), "ghi_wm2": v(ghi(g))}


climate = {"meta": SYN, "months": {m: month_doc(d, lo, hi, pm, g) for m, d, lo, hi, pm, g in MONTHS}}
climate["months"]["aug_dust"] = month_doc(31, 31, 43, 550, 700, dust=True)

policy = {
    "meta": {"data_mode": "synthetic", "source_kind": "policy", "version": "arrival-policy-demo-0.2"},
    "mc": {"runs": team(400), "vision_runs": team(20), "interval_q": team([0.1, 0.9])},
    "walk": {"speed_mps": v(1.2), "alpha": v(60.0), "beta": v(30.0), "dt_crowd_s": v(20.0), "dt_crossing_s": v(40.0)},
    "outdoor_limits": {"temperature_c": v([10, 38]), "humidity_pct": v([20, 80]), "pm10_ugm3_max": v(100)},
    "indoor_limits": {"temperature": v([21, 27]), "humidity": v([30, 70])},
    "comfort": {"t_pref_c": v(24.0), "weight_temperature": team(1.0), "weight_humidity": team(0.0),
                "arrival_band_c": team(1.0)},
    "traffic": {"peak_hours": v([7, 8, 17, 18])},
    "grid": {"peak_hours": v([12, 13, 14, 15, 16, 17])},
    "hvac": {"dt_s": team(600), "mass_flow_kg_s": v(0.3),
             "specific_heat": {"value": 1005.0, "source": "specific heat of air, reference value"},
             "supply_c": v(14.0)},
    "orientation_factor": {k: v(f) for k, f in {"N": 0.3, "E": 0.7, "S": 0.6, "W": 0.9}.items()},
    "thermal_priors": {"R": uni(0.012, 0.03), "C": uni(8.0e6, 2.0e7), "k_solar": uni(0.15, 0.4),
                       "k_occ": uni(70, 110), "COP": uni(2.2, 4.5)},
    "thermal_answers": {"COP": {"old_split": uni(2.2, 3.0), "new_inverter": uni(3.5, 4.5)},
                        "k_solar": {"blackout_film": uni(0.15, 0.22), "no_film": uni(0.3, 0.4)},
                        "R": {"new_building": uni(0.022, 0.03), "old_building": uni(0.012, 0.016)}},
    "questions": {"COP": team("Is the AC an old split unit or a new inverter?"),
                  "k_solar": team("Is there film or blackout curtains on the west windows?"),
                  "R": team("Is the building new or older than 10 years?"),
                  "k_occ": team("How many people are usually home during the day?")},
    "guard": {"decimals": team([0, 1, 2])},
    # personal model
    "personal_params": {
        "lambda_grid": team([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6], [0.34, 0.33, 0.33]]),
        "lambda_names": team(["money", "time", "comfort"]),
        "choice_temperature": team(0.15),
        "setpoint_prior": {"mean": v(24.0), "sd": v(1.5)}, "setpoint_obs_sd": team(0.5)},
    "consent_purposes": {
        "arrival_planning": {"allowed_categories": ["household", "anchors", "plan"], "allowed_roles": ["arrival_twin"],
                             "allowed_fields": ["household", "anchors", "flags", "trips"], "actions": ["read", "compute"]},
        "arrival_cost": {"allowed_categories": ["finance"], "allowed_roles": ["arrival_twin"],
                         "allowed_fields": ["budget", "value_of_time"], "actions": ["read", "compute"]},
        "arrival_vision": {"allowed_categories": ["constraints"], "allowed_roles": ["city_core"],
                           "allowed_fields": ["pm10_ugm3_max", "outdoor_min_max", "indoor_c_max_on_arrival"],
                           "actions": ["propose"]}},
    "vault": {"agent_token_ttl_s": team(3600)},
    # student → teacher
    "sync": {"noise_sigma_multiple": team(1.0), "anomaly_robust_z": team(5.0), "min_window": team(3),
             "pii_keys": team(["pseudonym", "profile_id", "household", "name", "email", "phone"]),
             "pii_patterns": team([r"784-?\d{4}-?\d{7}-?\d", r"[\w.+-]+@[\w-]+\.[\w.]+"]),
             "sigma_from_interval_z": team(1.2816), "prior_pseudo_count": team(5)},
    # daily life and future
    "future": {"workdays_per_month": v(22), "horizon_months": team(12)},
    "business": {"hire_lag_after_license_days": uni(20, 45)},
    # vision
    "vision": {
        "pm10_ugm3_max": v(50), "outdoor_min_max": {"value": 0, "decided_by": "user"},
        "min_p_feasible": team(0.8),
        "links": [{"id": "airport", "uses_av": True, "title": "Plane → covered pick-up zone", "action": "robotaxi.dispatch_covered", "action_class": "fleet_dispatch"},
                  {"id": "car", "uses_av": True, "title": "Robotaxi: cabin filtration", "action": "vehicle.cabin_recirculation", "action_class": "vehicle_comfort"},
                  {"id": "road", "uses_av": True, "title": "Lower-dust route", "action": "route.low_dust", "action_class": "route_plan"},
                  {"id": "building", "title": "Lift waiting, lobby ventilation", "action": "building.lobby_ventilation", "action_class": "building_hvac"},
                  {"id": "home", "title": "Flat pre-cooled overnight", "action": "hvac.preload_offpeak", "action_class": "hvac_setpoint"},
                  {"id": "grid", "title": "Grid: no peak increase", "action": "grid.offpeak_schedule", "action_class": "grid_schedule"}],
        "envelopes": {c: {"max_latency_s": lat, "max_staleness_s": st, "min_confidence": conf, "degraded": deg, "human_approval": ha}
                      for c, lat, st, conf, deg, ha in [("fleet_dispatch", 5.0, 60.0, 0.8, "defer", True),
                                                        ("vehicle_comfort", 2.0, 30.0, 0.7, "defer", False),
                                                        ("route_plan", 5.0, 120.0, 0.7, "shadow_only", False),
                                                        ("building_hvac", 60.0, 600.0, 0.7, "defer", False),
                                                        ("hvac_setpoint", 60.0, 600.0, 0.7, "defer", True),
                                                        ("grid_schedule", 300.0, 900.0, 0.7, "defer", False)]},
        "av": {"tau_v2x_ms": v(100.0), "tau_c": v(0.8), "safety_case_ref": v("ODD-SC-demo (placeholder)")},
        "arbitration": {"policy_version": "arbitration-demo-0.1",
                        "weights": {"dust_exposure": 1.0, "neighbor_discomfort": 1.0, "energy_kwh": 0.2},
                        "emergency_weights": {"dust_exposure": 3.0, "neighbor_discomfort": 1.0, "energy_kwh": 0.0},
                        "inequity_lambda": 0.5, "inequity_measure": "max_gap", "decided_by": "team"}},
    "cohort": {"flex_days": team(5)},
    "scenario": {"today": team("2026-10-01"), "week_start": team("2026-11-22"), "week_month": team("nov"),
                 "move_in": team("2026-11-19")},
    "traffic_windows": {"peak_windows_min": v([[435, 525], [1020, 1110]])},
    "life": {"on_time_probability": team(0.9), "step_min": team(10), "search_back_min": team(90),
             "min_window_h": team(1), "awake_hours": team([6, 21]), "reschedule_slots_min": team([600, 660, 840]),
             "pack_reminder_min": team(45)},
    "habits": {"min_n": team(3), "min_shift_h": team(0.75), "max_sd_h": team(0.5), "min_choices": team(3)},
    "assistant": {"max_questions_per_day": team(3), "quiet_hours": team([21, 7]), "near_tie": team(0.1)},
    "gateway": {"allowlist": team(["tamm.abudhabi.demo", "icp.gov.ae.demo", "adek.gov.ae.demo"]),
                "claims": team(["step_duration"]), "max_days": team(60)},
    # live official-site search — allow-list decided by the team
    "official_search": {"allowlist": team(["tamm.abudhabi", "u.ae", "icp.gov.ae", "adek.gov.ae", "doh.gov.ae",
                                           "addc.ae", "dmt.gov.ae", "adrec.gov.ae"])},
    "plan_levers": team([
        {"id": "medical_first_slot", "variable": "medical_wait", "value": 0.0,
         "title": "medical test on the first free day", "title_ru": "медкомиссия в первый свободный день"},
        {"id": "eid_express", "variable": "emirates_id_dur", "scale": 0.7,
         "title": "express Emirates ID filing (synthetic)", "title_ru": "ускоренная подача Emirates ID (синтетика)"},
        {"id": "bank_early", "variable": "bank_dur", "value": 0.0, "title": "opening the bank account earlier",
         "title_ru": "открыть банковский счёт раньше"}]),
    "causal": {"samples": team(400), "seed": team(7), "scm_version": team("plan-scm-demo-1")},
    # what counts as health and is cut from the phrase BEFORE any language model (RU and EN terms)
    "privacy": {"health_terms": team([r"астм\w*", r"аллерг\w*", r"диабет\w*", r"эпилеп\w*", r"asthm\w*",
                                      r"allerg\w*", r"diabet\w*", r"инвалид\w*", r"беремен\w*",
                                      r"ходунк\w*", r"walker\w*", r"wheelchair\w*", r"коляск\w*"]),
                # derived physical constraints per fragment class (team rule; the person confirms)
                "derived_constraints": team([
                    {"id": "respiratory", "terms": [r"астм\w*", r"аллерг\w*", r"asthm\w*", r"allerg\w*"],
                     "constraints": {"pm10_ugm3_max": 50, "outdoor_harsh_min_max": 0}},
                    {"id": "mobility", "terms": [r"ходунк\w*", r"walker\w*", r"wheelchair\w*", r"инвалид\w*", r"коляск\w*"],
                     "constraints": {"accessible_vehicle": True, "mobility_aid": 1, "outdoor_harsh_min_max": 0}}]),
                "jurisdiction": team("AE-AZ"), "legal_basis": team("consent")},
    # trust ladder: the system OFFERS autonomy after N approvals in a row; only the person grants it
    "trust": {"promote_after_approvals": team(2),
              "autonomy_allowed_actions": team(["propose_visit_window", "precool_home"]),   # everything else always needs approval
              "level_names": team(["shadow", "advises", "proposes", "acts alone"])},
    # indoor air, for comfort at the setpoint
    "indoor_air": {"co2_ppm": v(700), "humidity_pct": v(50), "noise_db": v(35)},
}

leila = {
    "meta": {"data_mode": "synthetic", "source_kind": "synthetic", "source": "fictional family for the demo"},
    "pseudonym": {"value": "p-7f3a", "decided_by": "TokenVault"},
    "household": [{"id": "leila", "role": "adult", "name": {"ru": "Лейла", "en": "Leila"}},
                  {"id": "husband", "role": "adult", "name": {"ru": "муж", "en": "your husband"}},
                  {"id": "daughter", "role": "child", "name": {"ru": "дочь", "en": "your daughter"}},
                  {"id": "son", "role": "child", "name": {"ru": "Адам", "en": "Adam"}}],
    "flags": {"has_school_children": True, "founder": True},
    "params": {"setpoint_c": {"value": 24.0, "decided_by": "user"}, "value_of_time_aed_per_h": {"value": 60, "decided_by": "user"},
               "budget_aed_month": {"value": 18000, "decided_by": "user"}},
    "device_only": {"son": {"condition": "asthma"}},   # sealed health domain in the CURE contour: never leaves
    # derived physical constraints — set by the person; only these are visible to planners and the city
    "constraints": {"son": {"pm10_ugm3_max": {"value": 50, "decided_by": "user"},
                            "outdoor_harsh_min_max": {"value": 0, "decided_by": "user"}}},
    "occupancy_by_hour": [4] * 7 + [1] * 8 + [3] * 3 + [4] * 6,
    "trips": [{"id": "school_am", "who": ["husband", "daughter", "son"], "from": "home", "to": "school", "mode": "car", "depart_h": 7,
               "arrive_by_min": 465, "usual_depart_min": 450},
              {"id": "office_am", "who": ["leila"], "from": "home", "to": "office", "mode": "car", "depart_h": 8,
               "arrive_by_min": 570, "usual_depart_min": 510},
              {"id": "school_pm", "who": ["husband", "daughter", "son"], "from": "school", "to": "home", "mode": "car", "depart_h": 14},
              {"id": "park", "who": ["daughter", "son"], "from": "home", "to": "park", "mode": "walk", "depart_h": 17},
              {"id": "office_pm", "who": ["leila"], "from": "office", "to": "home", "mode": "car", "depart_h": 18}],
    "grants": {"actions": ["recompute", "notify", "propose_visit_window", "propose_listing", "draft_checklist", "request_confirmation"],
               "agent_tools": ["search_public", "draft_text"],
               "cure_actions": ["propose_visit_window", "propose_listing", "precool_home", "calendar_event", "reminder",
                                "budget_rule", "dispatch_ramp_car"],
               "cure_approval_required": ["propose_visit_window", "propose_listing", "precool_home", "calendar_event",
                                          "reminder", "budget_rule", "dispatch_ramp_car"],
               "never_display": ["move money", "sign contracts", "send forms or messages for you",
                                 "book appointments without you"],
               "scopes": ["documents_status", "housing", "finance", "family", "health_local", "business"],
               "approval_required": ["propose_visit_window", "propose_listing"], "vision_consent": True},
    "resident_local_id": {"value": "device:leila", "decided_by": "device"},
    "consents": [{"purpose_id": "arrival_planning", "data_category": "household", "scope": ["arrival"]},
                 {"purpose_id": "arrival_cost", "data_category": "finance", "scope": ["arrival"]},
                 {"purpose_id": "arrival_vision", "data_category": "constraints", "scope": ["arrival"]}],
    "district_choices_demo": [{"options": ["A", "B", "C"], "chosen": "A", "decided_by": "user"}],
    "setpoint_observations_demo": {"value": [23.5, 24.0], "decided_by": "user"},
    "voice_example": "We're moving to Abu Dhabi, four of us. I'm a founder at Hub71, two kids going to school, my son has asthma, housing budget up to 18,000 a month",
    "voice_example_ru": "Мы переезжаем в Абу-Даби вчетвером, я основатель стартапа в Hub71, двое детей в школу, у сына астма, бюджет на жильё до 18 000 в месяц",
}


def od(base):
    return {"home|school|car": {"peak": tri(base, base + 8, base + 20), "offpeak": tri(base - 3, base + 2, base + 8)},
            "school|home|car": {"peak": tri(base, base + 8, base + 20), "offpeak": tri(base - 3, base + 2, base + 8)},
            "home|office|car": {"peak": tri(base + 5, base + 15, base + 30), "offpeak": tri(base, base + 8, base + 15)},
            "office|home|car": {"peak": tri(base + 5, base + 15, base + 30), "offpeak": tri(base, base + 8, base + 15)},
            "home|park|walk": {"peak": tri(0, 0, 0.1), "offpeak": tri(0, 0, 0.1)},
            "home|client|car": {"peak": tri(base + 8, base + 18, base + 32), "offpeak": tri(base, base + 8, base + 16)},
            "home|hub71|car": {"peak": tri(base + 15, base + 25, base + 40), "offpeak": tri(base + 3, base + 8, base + 14)}}


city = {"meta": {**SYN, "source": S + " (OSM routing, ADREC, ADEK)"},
        "od": {"A": od(10), "B": od(22), "C": od(16)},
        "walk_m": {d: {t: v(m) for t, m in zip(["school_am", "office_am", "school_pm", "park", "office_pm"], ms)}
                   for d, ms in {"A": [80, 150, 80, 250, 150], "B": [350, 400, 350, 900, 400], "C": [200, 250, 200, 500, 250]}.items()},
        "rent_aed_month": {"A": v(15500), "B": v(11000), "C": v(13000)},
        "typical_listing": {"A": "apt_west", "B": "apt_north", "C": "apt_north"},
        "trip_cost_aed_per_min": v(1.1),
        "car": {"ownership_aed_month": v(3500), "fuel_aed_per_min": v(0.35)},
        "taxi": {"base_aed": v(12), "aed_per_min": v(1.1), "wait_min": tri(3, 8, 15), "wait_indoor_booked_min": tri(0, 0, 1)},
        "errands": {"medical": {"from_home_min": v(15)}, "pharmacy": {"from_home_min": v(8)},
                    "keys": {"from_home_min": v(10)}, "hop_min": v(7)},
        "school_fees_aed_month": {"A": v(4200), "B": v(3600), "C": v(3900)}}

tariffs = {"meta": {**SYN, "source": S + " (DoE / ADDC)"}, "electricity_per_kwh": {"residential": v(0.3)}}

steps = {"meta": {**SYN, "source": "dependencies and durations — REPLACE from TAMM and agencies"},
         "steps": [
             {"id": "entry_permit", "title": "Entry permit", "duration_days": tri(3, 5, 10), "source_url": "TODO"},
             {"id": "medical", "title": "Medical test", "blocks_on": ["entry_permit"], "requires_visit": True, "center": "medical", "duration_days": tri(1, 2, 4), "source_url": "TODO"},
             {"id": "emirates_id", "title": "Emirates ID and residence visa", "blocks_on": ["medical"], "requires_visit": True, "center": "icp", "duration_days": tri(5, 7, 14), "source_url": "TODO"},
             {"id": "bank", "title": "Bank account", "blocks_on": ["emirates_id"], "duration_days": tri(2, 4, 10), "source_url": "TODO"},
             {"id": "tenancy", "title": "Tenancy contract", "blocks_on": ["emirates_id"], "duration_days": tri(3, 7, 14), "source_url": "TODO"},
             {"id": "tawtheeq", "title": "Tawtheeq registration", "blocks_on": ["tenancy"], "duration_days": tri(1, 2, 5), "source_url": "TODO"},
             {"id": "utilities", "title": "Electricity and water", "blocks_on": ["tawtheeq", "bank"], "duration_days": tri(1, 3, 6), "source_url": "TODO"},
             {"id": "school", "title": "School", "blocks_on": ["emirates_id"], "applies_if": {"has_school_children": True}, "duration_days": tri(5, 10, 21), "source_url": "TODO"},
             {"id": "company_license", "title": "Company licence", "blocks_on": ["emirates_id"], "applies_if": {"founder": True}, "duration_days": tri(5, 10, 20), "source_url": "TODO"}]}

queues = {"meta": {**SYN, "source": "queues — synthetic; an agency aggregate in a pilot"},
          "centers": {c: {"wait_days_by_hour": {"8": tri(0, 1, 3), "10": tri(1, 3, 7), "14": tri(1, 2, 5), "17": tri(0, 2, 6)},
                          "capacity_per_day": v(cap),
                          # daily visits by other residents (not the cohort) — they take capacity
                          "background_per_day": tri(bg[0], bg[1], bg[2])}
                      for c, cap, bg in (("medical", 40, (28, 34, 40)), ("icp", 30, (20, 26, 30)))}}

rates = {"meta": SYN, "temporary_housing_per_day": v(450), "setup_costs_aed": v(25000),
         "company_monthly_cost_aed": v(35000)}


def link(**kw):
    base = {"physical_safe": True, "authorized": True, "human_approved": True, "data_age_s": 10.0, "confidence": 0.9,
            "stages_s": {"sense": 0.2, "validation": 0.1, "network": 0.3, "state": 0.1, "reasoning": 0.5, "policy": 0.1, "actuation": 0.5}}
    base.update(kw)
    return base


def fam(fid, who, day, hour, pm10_max=None, outdoor_max=None, accessible=False, mobility_aid=0, heavy_items=0):
    return {"id": fid, "who": who, "arrival_day": day, "arrival_h": hour,
            "limits": {"pm10_ugm3_max": pm10_max, "outdoor_harsh_min_max": outdoor_max, "accessible_vehicle": accessible},
            "slowest_member": {"mobility_aid": mobility_aid, "heavy_items": heavy_items}}


# demo synthetic: (id, who, landing day, hour, PM10 limit, max minutes in heat/dust, needs a ramp car,
#                  mobility aid, heavy luggage). The "who" label is for viewers only.
FAMILIES = [
    ("p-7f3a", "Leila: four, a child who must avoid dust", 0, 20, 50, 0),
    ("f-02", "Elderly couple, wheelchair", 0, 13, None, 0, True, 1),
    ("f-03", "Family with a newborn", 0, 16, 50, 0),
    ("f-04", "Student, alone", 0, 13),
    ("f-05", "Engineer after a night shift", 1, 23),
    ("f-06", "Three kids and heavy luggage", 0, 13, None, None, False, 0, 1),
    ("f-07", "Grandmother visiting, struggles in heat", 1, 16, None, 0),
    ("f-08", "Person with a guide dog", 1, 9, None, None, False, 1),
    ("f-09", "Couple, one must avoid dust", 1, 20, 50, 5),
    ("f-10", "Doctor, straight to a shift", 1, 6),
    ("f-11", "Child in a wheelchair", 2, 13, None, 0, True, 1),
    ("f-12", "Musician with instruments", 2, 20, None, None, False, 0, 1),
    ("f-13", "Retiree after surgery", 2, 16, None, 0, True, 1),
    ("f-14", "Young couple on a budget", 2, 9),
    ("f-15", "Family expecting a baby", 3, 13, None, 0),
    ("f-16", "Athlete at a training camp", 3, 16),
    ("f-17", "Two toddlers", 3, 20, 50, 0),
    ("f-18", "Scientist with equipment", 4, 23, None, None, False, 0, 1),
    ("f-19", "Elderly man with a cane", 5, 13, None, 5, False, 1),
    ("f-20", "Family with teenagers", 6, 9),
]

FAMILY_RU = {'p-7f3a': 'Лейла: четверо, ребёнку нельзя в пыль', 'f-02': 'Пожилая пара, коляска', 'f-03': 'Семья с новорождённым', 'f-04': 'Студент, один', 'f-05': 'Инженер после ночной смены', 'f-06': 'Трое детей и большой багаж', 'f-07': 'Бабушка к внукам, тяжело в жару', 'f-08': 'Человек с собакой-поводырём', 'f-09': 'Пара, одному нельзя в пыль', 'f-10': 'Врач, сразу на смену', 'f-11': 'Ребёнок на коляске', 'f-12': 'Музыкант с инструментами', 'f-13': 'Пенсионер после операции', 'f-14': 'Молодая пара, экономят', 'f-15': 'Семья, ждут ребёнка', 'f-16': 'Спортсмен на сборы', 'f-17': 'Двое малышей', 'f-18': 'Учёный с оборудованием', 'f-19': 'Пожилой мужчина с тростью', 'f-20': 'Семья с подростками'}

scenario = {
    "meta": {**SYN, "source": "vision scene — demo synthetic"},
    "links": {l: link() for l in ("airport", "car", "road", "building", "home", "grid")},
    "arrival": {"month": "aug", "arrival_h": 20, "flight_delay_h": 3,
                "without": {"walk_to_taxi_m": v(350), "taxi_wait_min": tri(5, 12, 25), "walk_home_m": v(120)},
                # with CURE the car waits at the covered exit BEFORE landing (airport link): no waiting;
                # waiting appears only when cars run short that hour (computed in cohort.py)
                "with": {"walk_to_taxi_m": v(0), "taxi_wait_min": {"dist": "fixed", "value": 0.0}, "walk_home_m": v(0)}},
    "storm": {"pm10_ugm3": v(600), "perception_confidence": v(0.6), "v2x_latency_ms": v(80.0)},
    "av_normal": {"perception_confidence": v(0.95), "v2x_latency_ms": v(40.0)},
    "neighbors": {
        "candidates": [
            {"action_id": "lobby_purge_max", "dust_exposure": 0.0, "neighbor_discomfort": 3.0, "energy_kwh": 6.0,
             "feasibility": {"safety": True, "law": True, "contract": True, "capacity": False}, "note": "noise and draught above the neighbours' limit", "note_ru": "шум и сквозняк выше предела для соседей"},
            {"action_id": "lobby_purge_timed", "dust_exposure": 0.5, "neighbor_discomfort": 0.5, "energy_kwh": 3.0,
             "feasibility": {"safety": True, "law": True, "contract": True, "capacity": True}, "note": "lobby purge 15 minutes before arrival", "note_ru": "продувка за 15 минут до приезда"},
            {"action_id": "no_change", "dust_exposure": 4.0, "neighbor_discomfort": 0.0, "energy_kwh": 0.0,
             "feasibility": {"safety": True, "law": True, "contract": True, "capacity": True}, "note": "do nothing", "note_ru": "ничего не делать"}],
        "stakeholders": [{"stakeholder_id": "p-7f3a", "objectives": ["dust_exposure"], "action_scope": ["building.lobby_ventilation"]},
                         {"stakeholder_id": "floor_residents", "objectives": ["neighbor_discomfort"], "action_scope": ["building.lobby_ventilation"]},
                         {"stakeholder_id": "building_operator", "objectives": ["energy_kwh"], "action_scope": ["building.lobby_ventilation"]}],
        "proposals": [{"stakeholder_id": "p-7f3a", "objective": "dust_exposure", "constraints": {"dust_exposure": 1.0},
                       "priority": 1.0, "evidence_refs": ["constraint:pm10_max"], "action_envelope": ["building.lobby_ventilation"]},
                      {"stakeholder_id": "floor_residents", "objective": "neighbor_discomfort", "constraints": {"neighbor_discomfort": 1.0},
                       "priority": 1.0, "evidence_refs": ["house_rules"], "action_envelope": ["building.lobby_ventilation"]}]},
    # V5: 20 DIFFERENT families in one week. The "who" label is for viewers; the city sees only constraints
    # (pm10_max, outdoor_harsh_min_max, accessible_vehicle) and the slowest member's mobility.
    "cohort": {"families": [{**fam(*f), "who_ru": FAMILY_RU[f[0]]} for f in FAMILIES], "arrival_window_days": v(7), "visits_per_family": v(2),
               "centers": v(["medical", "icp"]),
               "fleet_size": v(25), "accessible_fleet_size": v(1), "rides_per_hour_per_car": v(1.5),
               "ramp_rides_per_hour_per_car": v(1.0),
               "arrival_hours": v(sorted({f[3] for f in FAMILIES})),
               "accessible_taxi_wait_min_without": tri(15, 35, 90),
               "apartment_cooling_kw": uni(2.0, 4.0), "chiller_peak_capacity_kw": v(120)}}

def day_profile(date, pm_morning, pm_day, pm_evening, t_lo=22, t_hi=33):
    t = [round(t_lo + (t_hi - t_lo) * max(0.0, math.cos((h - 15) / 24 * 2 * math.pi)), 1) for h in range(24)]
    pm = [pm_morning if h < 9 else pm_day if h < 17 else pm_evening for h in range(24)]
    return {"date": date, "temperature_c": t, "humidity_pct": [55] * 24, "pm10_ugm3": pm}


week = {"meta": {**SYN, "source": "synthetic week 22–28 Nov 2026: forecast, calendar, logs"},
        # PM10: morning / day / evening. The son's limit is PM10 ≤ 50 → windows: Tue and Sat mornings, Sat evening for the family
        "forecast": {"days": [day_profile("2026-11-22", 60, 90, 70), day_profile("2026-11-23", 55, 180, 80),
                              day_profile("2026-11-24", 40, 120, 70), day_profile("2026-11-25", 60, 150, 90),
                              day_profile("2026-11-26", 55, 140, 80), day_profile("2026-11-27", 65, 160, 85),
                              day_profile("2026-11-28", 35, 110, 45)]},
        "calendar": [{"id": "husband_client", "who": "husband", "date": "2026-11-26", "start_min": 480,
                      "needs_car": True, "od": "home|client|car", "movable": False, "title": "Client", "title_ru": "Клиент"},
                     {"id": "leila_investor", "who": "leila", "date": "2026-11-26", "start_min": 510,
                      "needs_car": True, "od": "home|hub71|car", "movable": True, "title": "Investor at Hub71", "title_ru": "Инвестор в Hub71"}],
        "errands": [{"id": "medical", "title": "Medical test", "title_ru": "Медкомиссия", "center": "medical", "date": "2026-11-24"},
                    {"id": "pharmacy", "title": "Pharmacy", "title_ru": "Аптека", "date": "2026-11-24"},
                    {"id": "keys", "title": "Storage keys", "title_ru": "Ключи от кладовой", "date": "2026-11-24"}],
        # 4 weeks of return-home log (fractional hour): later on Mondays
        "returns_log": [{"weekday": wd, "hour": h} for wk in range(4)
                        for wd, h in (("mon", 19.5 + 0.1 * (wk % 2)), ("tue", 18.5), ("wed", 18.4 + 0.1 * wk),
                                      ("thu", 18.6), ("fri", 18.3))],
        # peak-hour choices: taxi (money) vs waiting for the shared car (time); axes — money AED, time h, comfort min
        "choices_log": [{"date": d, "chosen": "taxi", "peak": True,
                         "options": {"taxi": [45, 0.4, 0], "wait_for_shared_car": [0, 1.3, 0]}}
                        for d in ("2026-12-01", "2026-12-08", "2026-12-15")],
        "transactions": [{"month": "2026-11", "category": c, "aed": a} for c, a in
                         (("rent", 15500), ("cooling", 70), ("commute", 1200), ("school", 4200))],
        "guest": {"id": "mother", "title": "Leila's mother", "title_ru": "Мама Лейлы", "month": "jul", "candidate_hours": [13, 16, 20, 23],
                  "slowest_member": {"mobility_aid": 1, "heavy_items": 0}},
        "summer_months": ["jul", "aug"]}

corpus = {"meta": {"data_mode": "synthetic", "source_kind": "synthetic",
                   "source": "offline page set for the plan update scene (.demo domains are not real sites)"},
          "pages": [{"url": "https://tamm.abudhabi.demo/residency/medical", "topic": "step_durations",
                     "text": "Medical fitness results are usually issued within 2-4 working days.",
                     "card": {"claim": "step_duration", "subject": "medical", "low": 2, "high": 4, "unit": "days"}},
                    {"url": "https://icp.gov.ae.demo/eid/processing", "topic": "step_durations",
                     "text": "Emirates ID is usually issued within 5-14 days after biometrics.",
                     "card": {"claim": "step_duration", "subject": "emirates_id", "low": 5, "high": 14, "unit": "days"}},
                    {"url": "https://fast-visa-blog.demo/eid-in-one-day", "topic": "step_durations",
                     "text": "Get your EID in 1 day!",
                     "card": {"claim": "step_duration", "subject": "emirates_id", "low": 1, "high": 1, "unit": "days"}}]}

cohort = {"meta": {**SYN, "source": "synthetic cohort generator parameters"}, "seed": team(11),
          # facts reported by other residents (synthetic) — all the teacher knows
          "teacher_facts": {"medical": v([2, 3, 2, 4, 3]), "emirates_id": v([11, 12, 10, 13]), "tenancy": v([6, 8, 5])},
          # what the resident reports (three demo cases)
          "demo_facts": [{"id": "signal", "step_id": "emirates_id", "observed_days": v(13), "source_integrity": "verified"},
                         {"id": "noise", "step_id": "bank", "observed_days": v(4), "source_integrity": "verified"},
                         {"id": "forged", "step_id": "medical", "observed_days": v(0.1), "source_integrity": "unsigned_upload"}]}

tpl = lambda **k: k
templates = {"meta": {"data_mode": "synthetic", "source_kind": "designed", "source": "tools/agent_designer (OpenAI) + team review"},
             "templates": [
                 tpl(id="documents", title="Documents", goal_classes=["settle_by_date"], uses=["plan"], scopes=["documents_status"],
                     allowed_actions=["search_public", "draft_text"], approval_required=[],
                     forbidden_display=["see personal data", "act in any system", "send anything for you"],
                     required_evidence=["step_status"], escalation_path=["leila"], completion_verification="all documents steps closed",
                     reactions=[{"event": "visa_delay", "action": "draft_text", "scope": "documents_status", "draft": "employer_letter"}],
                     design_report="designed with agent_designer: public TAMM pages + schema; CURE checks ✓; review ✓"),
                 tpl(id="housing", title="Housing", goal_classes=["settle_by_date"], uses=["cost", "apartment"], scopes=["housing"],
                     allowed_actions=["search_public", "draft_text"], approval_required=[],
                     forbidden_display=["contact the landlord", "pay"],
                     required_evidence=["tenancy_signed"], escalation_path=["leila"], completion_verification="tenancy registered",
                     reactions=[{"event": "visa_delay", "action": "draft_text", "scope": "housing", "draft": "landlord_note"}],
                     design_report="designed with agent_designer; CURE checks ✓; review ✓"),
                 tpl(id="school", title="School and kids", goal_classes=["settle_by_date"], applies_if={"has_school_children": True}, uses=["day", "cost"], scopes=["family"],
                     allowed_actions=["search_public", "draft_text"], approval_required=[],
                     forbidden_display=["submit applications"], required_evidence=["enrollment_confirmed"], escalation_path=["leila"],
                     completion_verification="enrolment confirmed by the school",
                     reactions=[{"event": "visa_delay", "action": "draft_text", "scope": "family", "draft": "school_note"}],
                     design_report="designed with agent_designer; CURE checks ✓; review ✓"),
                 tpl(id="family_health", title="Family health: air and heat", goal_classes=["settle_by_date"], uses=["day"], scopes=["health_local"],
                     allowed_actions=["search_public", "draft_text"], approval_required=[],
                     forbidden_display=["share a diagnosis with anyone", "book doctors"], required_evidence=["none"], escalation_path=["leila"],
                     completion_verification="none",
                     reactions=[{"event": "dust_forecast", "action": "draft_text", "scope": "health_local", "draft": "dust_line"}],
                     design_report="designed with agent_designer; CURE checks ✓; review ✓"),
                 tpl(id="budget", title="Budget", goal_classes=["settle_by_date"], uses=["cost", "plan"], scopes=["finance"],
                     allowed_actions=["search_public", "draft_text"], approval_required=[],
                     forbidden_display=["move money"], required_evidence=["none"], escalation_path=["leila"], completion_verification="none",
                     reactions=[],
                     design_report="designed with agent_designer; CURE checks ✓; review ✓"),
                 tpl(id="business", title="Business", goal_classes=["settle_by_date"], applies_if={"founder": True}, uses=["plan"], scopes=["business"],
                     allowed_actions=["search_public", "draft_text"], approval_required=[],
                     forbidden_display=["sign or submit on her behalf"], required_evidence=["license_issued"], escalation_path=["leila"],
                     completion_verification="licence issued",
                     reactions=[{"event": "car_conflict", "action": "draft_text", "scope": "business", "draft": "investor_reschedule"},
                                {"event": "license", "action": "draft_text", "scope": "business", "draft": "vacancy"}],
                     design_report="designed with agent_designer; CURE checks ✓; review ✓")]}

listings = {"meta": {"data_mode": "synthetic", "source": "flats drawn by the team"},
            "listings": [{"id": "apt_west", "district": "A", "tariff_class": "residential", "equip_w": 300,
                          "windows": [{"orientation": "W", "area_m2": 8.0}, {"orientation": "S", "area_m2": 3.0}]},
                         {"id": "apt_north", "district": "B", "tariff_class": "residential", "equip_w": 300,
                          "windows": [{"orientation": "N", "area_m2": 8.0}, {"orientation": "E", "area_m2": 3.0}]}]}

versions = {"meta": {"data_mode": "synthetic", "source_kind": "config"},
            "teacher": {k: "v1" for k in ("schema_version", "ontology_version", "topology_version", "scm_version", "dynamics_version",
                                          "policy_version", "skill_version", "adapter_version", "model_version", "site_pack_version")},
            "student": {k: "v1" for k in ("schema_version", "ontology_version", "topology_version", "scm_version", "dynamics_version",
                                          "policy_version", "skill_version", "adapter_version", "model_version", "site_pack_version")}}

# Russian step titles — used only in Russian replies (CURE answers in the question's language)
STEP_RU = {"entry_permit": "Въездное разрешение", "medical": "Медкомиссия", "emirates_id": "Emirates ID и резидентская виза",
           "bank": "Банковский счёт", "tenancy": "Договор аренды", "tawtheeq": "Регистрация договора (Tawtheeq)",
           "utilities": "Электричество и вода", "school": "Школа", "company_license": "Лицензия компании"}
for s_ in steps["steps"]:
    s_["title_ru"] = STEP_RU[s_["id"]]

for name, doc in [("climate", climate), ("policy", policy), ("leila", leila), ("city", city), ("tariffs", tariffs),
                  ("steps", steps), ("queues", queues), ("rates", rates), ("vision_scenario", scenario), ("agents", templates),
                  ("cohort", cohort), ("listings", listings), ("versions", versions), ("week", week),
                  ("web_corpus", corpus)]:
    (OUT / f"{name}.yaml").write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
print("demo data written to", OUT)
