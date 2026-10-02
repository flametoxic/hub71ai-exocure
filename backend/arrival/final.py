"""Final: one number for the family and one for the cohort. Only references to computed answers — nothing new is computed."""
from __future__ import annotations

import copy

from .core import ANSWERS, Store, answer


def final_number(*, impact_trace: str, arrival_trace: str, dust_trace: str, cohort_trace: str,
                 stores: list[Store]) -> dict:
    a = {k: ANSWERS[t] for k, t in (("impact", impact_trace), ("arrival", arrival_trace), ("dust", dust_trace),
                                     ("cohort", cohort_trace))}
    bad = [k for k, v in a.items() if v.get("calculation_status") != "computed"]
    if bad:
        return {"calculation_status": "refused", "missing": bad, "message": "compute these scenes first: " + ", ".join(bad)}
    c = a["cohort"]["value"]
    return answer({
        "family": {"days_saved": copy.deepcopy(a["impact"]["value"]["days_saved"]),
                   "son_dust_min_saved_on_arrival": copy.deepcopy(a["arrival"]["value"]["saved"]["dust_min"]),
                   "son_harsh_min_saved_per_dust_day": copy.deepcopy(a["dust"]["value"]["son_harsh_min"]["saved"])},
        "cohort": {"families": c["families"],
                   "families_with_personal_limit": c["limits"]["families_with_personal_limit"],
                   "limits_kept": copy.deepcopy(c["limits"]["kept"]),
                   "harsh_minutes_total": copy.deepcopy(c["harsh_minutes_total"]),
                   "wasted_center_trips_avoided": c["centers"]["wasted_trips"]["without"] - c["centers"]["wasted_trips"]["with"],
                   "hot_home_arrivals": copy.deepcopy(c["grid"]["hot_home_arrivals"]),
                   "arrival_day_peak_kw": copy.deepcopy(c["grid"]["arrival_day_peak_kw"])},
        "refs": {k: v["trace_id"] for k, v in a.items()},
        "not_claimed": ["yearly dust minutes — no data on the number of dusty days",
                        "centre capacity — coordination does not create slots"]},
        unit="summary", stores=stores, formulas=["refs only"], label="Result — model comparison")
