"""V44 customer information: immutable, bounded public trip forecasts."""
from __future__ import annotations

import copy
from typing import Any

ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v44-trip-weather-close-outcomes"
TOOL_INTERFACE_VERSION = 39
CORE_RELEASE_ID = "outfitter-v44-trip-weather-close-outcomes-core"
POLICY_ID = "public-trip-temperature-and-close-outcomes-v1"
FORECAST_DATES = 15


def trip_context(*, issued_day: int, pickup_day: int, duration_days: int,
                 forecast_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """No extrapolation: absence of a forecast is recorded as unknown."""
    if duration_days < 1:
        raise ValueError("trip duration must be positive")
    dates = list(range(pickup_day, pickup_day + duration_days))
    rows = {int(row["day"]): row for row in forecast_rows
            if issued_day <= int(row["day"]) < issued_day + FORECAST_DATES}
    covered = [day for day in dates if day in rows]
    if not covered:
        raise ValueError("customer trip has no published temperature coverage")
    forecasts = [{"day": day, "condition": str(rows[day]["condition"]),
                  "condition_accuracy": float(rows[day]["condition_accuracy"]),
                  **{key: float(rows[day][key]) for key in
                     ("night_low_c", "temp_mean_c", "day_high_c")}}
                 for day in covered]
    return {"issued_day": issued_day, "pickup_day": pickup_day,
            "return_day": pickup_day + duration_days,
            "covered_days": covered, "unknown_days": [d for d in dates if d not in rows],
            "minimum_covered_forecast_night_low_c": min(r["night_low_c"] for r in forecasts),
            "forecasts": forecasts}


def public_coverage(context: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy({k: context[k] for k in (
        "issued_day", "covered_days", "unknown_days",
        "minimum_covered_forecast_night_low_c")})


def contextualize(raw: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(raw)
    result["trip_weather_context"] = copy.deepcopy(context)
    result["pickup_day"] = context["pickup_day"]
    result["booking_lead_days"] = max(0, context["pickup_day"] - context["issued_day"])
    if result["service_key"] == "rental:sleeping_bag":
        cool = context["minimum_covered_forecast_night_low_c"] <= 15.0
        result["preferred_group"] = "three_season_bag" if cool else "warm_weather_bag"
        result["use_case"] = "cool_night" if cool else "warm_night"
    return result
