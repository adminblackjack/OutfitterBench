"""Pure V38 proactive-reservation contingency preference kernel."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any


ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v38-proactive-contingency"
TOOL_INTERFACE_VERSION = 33
CORE_RELEASE_ID = "outfitter-v38-shared-proactive-contingency-core"
POLICY_ID = "proactive-reservation-contingency-v1"
RNG_NAMESPACE = 3801

MAX_INQUIRIES_PER_COMMITMENT = 1
MIN_NOTICE_DAYS = 2
MAX_PROPOSAL_LEAD_DAYS = 14
MAX_PROPOSALS = 2
MAX_BATCH_SIZE = 25

ADVERSITY = {
    "sunny": 0.0,
    "cloudy": 0.18,
    "hot": 0.35,
    "cold": 0.45,
    "rainy": 0.82,
    "snowy": 1.0,
}

CALENDAR_VALUES = {
    "family": {
        "weekday": -0.45,
        "weekend": 0.20,
        "public_holiday": 0.50,
        "school_holiday": 0.40,
    },
    "group": {
        "weekday": -0.20,
        "weekend": 0.20,
        "public_holiday": 0.40,
        "school_holiday": 0.15,
    },
    "solo": {
        "weekday": 0.00,
        "weekend": 0.08,
        "public_holiday": 0.15,
        "school_holiday": 0.05,
    },
}

# Selected mechanically from the preregistered grids. The calibration artifact
# records targets and tie-breaking; these values are frozen runtime inputs.
SELECTED_PARAMETERS = {
    "weather_weight": 0.9,
    "shift_weight": 0.07,
    "initial_consideration_intercept": 0.10,
    "initial_consideration_flex_slope": 0.70,
    "initial_tolerance_base": 0.10,
    "initial_tolerance_flex_slope": 0.40,
    "initial_sampling_temperature": 2.0,
    "equipment_scale": 0.5,
    "equipment_weight": 0.5,
    "accept_intercept": -2.0,
    "accept_delta_coefficient": 1.5,
    "accept_flex_coefficient": 0.8,
    "accept_notice_coefficient": 0.16,
    "cancel_intercept": -3.5,
    "cancel_badness_coefficient": 1.2,
    "cancel_late_coefficient": 0.8,
    "cancel_shift_coefficient": 0.2,
    "cancel_flex_protection": 0.8,
    "cancel_probability_cap": 0.15,
    "counteroffer_base": 0.03,
    "counteroffer_flex_coefficient": 0.14,
    "counteroffer_notice_coefficient": 0.06,
    "counteroffer_probability_cap": 0.25,
}

PARAMETER_GRIDS = {
    "weather_weight": [0.6, 0.9, 1.2],
    "shift_weight": [0.04, 0.07, 0.10],
    "initial_consideration_intercept": [0.10, 0.20],
    "initial_consideration_flex_slope": [0.50, 0.70],
    "initial_tolerance_base": [0.0, 0.10],
    "initial_tolerance_flex_slope": [0.20, 0.40],
    "initial_sampling_temperature": [1.0, 2.0],
    "equipment_scale": [0.25, 0.5, 1.0],
    "equipment_weight": [0.5, 1.0],
    "accept_intercept": [-2.4, -2.0, -1.6],
    "accept_delta_coefficient": [1.0, 1.5, 2.0],
    "accept_flex_coefficient": [0.4, 0.8],
    "accept_notice_coefficient": [0.08, 0.16],
    "cancel_intercept": [-4.0, -3.5, -3.0],
    "cancel_badness_coefficient": [0.8, 1.2],
    "cancel_late_coefficient": [0.4, 0.8],
    "cancel_shift_coefficient": [0.2, 0.4],
    "cancel_flex_protection": [0.4, 0.8],
    "counteroffer_base": [0.03, 0.06],
    "counteroffer_flex_coefficient": [0.08, 0.14],
    "counteroffer_notice_coefficient": [0.02, 0.06],
}


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def customer_class(*, children: int, archetype: str) -> str:
    if int(children) > 0:
        return "family"
    if archetype in {"overnight", "picnic"}:
        return "group"
    return "solo"


def date_utility_delta(
    *,
    customer_type: str,
    current_day_type: str,
    candidate_day_type: str,
    current_condition: str,
    candidate_condition: str,
    shift_days: int,
    schedule_flexibility: float,
    forecast_trust: float,
    occasion_importance: float,
    parameters: dict[str, float] = SELECTED_PARAMETERS,
    current_is_school_holiday: bool = False,
    candidate_is_school_holiday: bool = False,
) -> float:
    calendar = CALENDAR_VALUES[customer_type]
    current_value = float(calendar[current_day_type])
    candidate_value = float(calendar[candidate_day_type])
    if current_is_school_holiday:
        current_value = max(current_value, float(calendar['school_holiday']))
    if candidate_is_school_holiday:
        candidate_value = max(candidate_value, float(calendar['school_holiday']))
    flexibility = min(1.0, max(0.0, float(schedule_flexibility)))
    weather_delta = ADVERSITY[candidate_condition] - ADVERSITY[current_condition]
    return (
        candidate_value
        - current_value
        - float(parameters["weather_weight"])
        * min(1.0, max(0.0, float(forecast_trust)))
        * max(0.0, float(occasion_importance))
        * weather_delta
        - float(parameters["shift_weight"])
        * abs(int(shift_days))
        * (1.0 - 0.75 * flexibility)
    )


def normalized_equipment_delta(
    raw_delta: float, parameters: dict[str, float] = SELECTED_PARAMETERS
) -> float:
    return float(parameters["equipment_weight"]) * math.tanh(
        float(raw_delta) / float(parameters["equipment_scale"])
    )


def acceptance_probability(
    *,
    utility_delta: float,
    schedule_flexibility: float,
    notice_days: int,
    parameters: dict[str, float] = SELECTED_PARAMETERS,
) -> float:
    flexibility = min(1.0, max(0.0, float(schedule_flexibility)))
    notice = max(MIN_NOTICE_DAYS, int(notice_days))
    score = (
        float(parameters["accept_intercept"])
        + float(parameters["accept_delta_coefficient"]) * float(utility_delta)
        + float(parameters["accept_flex_coefficient"]) * (flexibility - 0.5)
        + float(parameters["accept_notice_coefficient"])
        * math.log1p(notice - MIN_NOTICE_DAYS)
    )
    return min(0.80, max(0.01, sigmoid(score)))


def cancellation_probability(
    *,
    best_utility_delta: float,
    best_shift_days: int,
    schedule_flexibility: float,
    notice_days: int,
    parameters: dict[str, float] = SELECTED_PARAMETERS,
) -> float:
    flexibility = min(1.0, max(0.0, float(schedule_flexibility)))
    notice = min(MAX_PROPOSAL_LEAD_DAYS, max(MIN_NOTICE_DAYS, int(notice_days)))
    late = 1.0 - (notice - MIN_NOTICE_DAYS) / (
        MAX_PROPOSAL_LEAD_DAYS - MIN_NOTICE_DAYS
    )
    score = (
        float(parameters["cancel_intercept"])
        + float(parameters["cancel_badness_coefficient"])
        * max(0.0, -float(best_utility_delta))
        + float(parameters["cancel_late_coefficient"]) * late
        + float(parameters["cancel_shift_coefficient"])
        * min(1.0, abs(int(best_shift_days)) / 10.0)
        - float(parameters["cancel_flex_protection"]) * flexibility
    )
    return min(float(parameters["cancel_probability_cap"]), sigmoid(score))


def counteroffer_probability(
    *,
    candidate_available: bool,
    schedule_flexibility: float,
    notice_days: int,
    parameters: dict[str, float] = SELECTED_PARAMETERS,
) -> float:
    if not candidate_available:
        return 0.0
    flexibility = min(1.0, max(0.0, float(schedule_flexibility)))
    notice = min(MAX_PROPOSAL_LEAD_DAYS, max(MIN_NOTICE_DAYS, int(notice_days)))
    notice_fraction = (notice - MIN_NOTICE_DAYS) / (
        MAX_PROPOSAL_LEAD_DAYS - MIN_NOTICE_DAYS
    )
    value = (
        float(parameters["counteroffer_base"])
        + float(parameters["counteroffer_flex_coefficient"]) * flexibility
        + float(parameters["counteroffer_notice_coefficient"])
        * notice_fraction
    )
    return min(float(parameters["counteroffer_probability_cap"]), max(0.0, value))


def initial_date_consideration_probability(
    schedule_flexibility: float,
    parameters: dict[str, float] = SELECTED_PARAMETERS,
) -> float:
    value = (
        float(parameters["initial_consideration_intercept"])
        + float(parameters["initial_consideration_flex_slope"])
        * min(1.0, max(0.0, float(schedule_flexibility)))
    )
    return min(1.0, max(0.0, value))


def initial_date_is_acceptable(
    utility_delta: float,
    schedule_flexibility: float,
    parameters: dict[str, float] = SELECTED_PARAMETERS,
) -> bool:
    tolerance = float(parameters["initial_tolerance_base"]) + float(
        parameters["initial_tolerance_flex_slope"]
    ) * min(1.0, max(0.0, float(schedule_flexibility)))
    return float(utility_delta) >= -tolerance


def canonical_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    return {
        "pickup_day": int(proposal["pickup_day"]),
        "return_day": int(proposal["return_day"]),
        "line_changes": sorted(
            (
                {
                    "line_id": str(row["line_id"]),
                    "offer_id": str(row["offer_id"]),
                    "promise_profile_id": str(row["promise_profile_id"]),
                }
                for row in proposal.get("line_changes", [])
            ),
            key=lambda row: row["line_id"],
        ),
    }


def proposal_digest(proposal: dict[str, Any]) -> str:
    encoded = json.dumps(
        canonical_proposal(proposal), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def policy_document() -> dict[str, Any]:
    return {
        "format": POLICY_ID,
        "maximum_successful_inquiries_per_commitment": MAX_INQUIRIES_PER_COMMITMENT,
        "minimum_notice_days": MIN_NOTICE_DAYS,
        "maximum_proposal_lead_days": MAX_PROPOSAL_LEAD_DAYS,
        "maximum_proposals_per_inquiry": MAX_PROPOSALS,
        "maximum_batch_size": MAX_BATCH_SIZE,
        "response_delay_days": 1,
        "report_valid_days": 1,
        "equipment_change_scope": "same_promise_profile_exact_offer",
        "financial_terms": "locked_rent_prepayment_and_deposit_unchanged",
        "counteroffer_scope": "date_only",
        "customer_cancellation_settlement": "manager_cancellation_terms",
        "customer_cancellation_reputation_effect": "none",
        "inventory_feasibility_evaluated": False,
        "parameters": dict(SELECTED_PARAMETERS),
        "parameter_grids": PARAMETER_GRIDS,
        "calendar_values": CALENDAR_VALUES,
        "weather_adversity": ADVERSITY,
    }
