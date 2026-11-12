"""Frozen V42 customer-weather information policy and pure helpers."""

from __future__ import annotations

import math
from typing import Any


ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v42-weather-information-coherence"
TOOL_INTERFACE_VERSION = 37
CORE_RELEASE_ID = "outfitter-v42-shared-weather-information-coherence-core"
POLICY_ID = "shared-public-weather-customer-pool-v1"

# The pool itself is weather-independent. These multipliers define its size
# relative to the calendar/season base rate and are intentionally separate
# from the subsequent customer consideration probability.
POTENTIAL_POOL_MULTIPLIERS = {
    "overnight": 1.35,
    "picnic": 1.35,
    "lightweight": 1.30,
    "supply": 1.15,
}

CONSIDERATION_PROBABILITIES = {
    "overnight": {
        "sunny": 0.92,
        "cloudy": 0.78,
        "hot": 0.60,
        "cold": 0.50,
        "rainy": 0.25,
        "snowy": 0.12,
    },
    "picnic": {
        "sunny": 0.94,
        "cloudy": 0.75,
        "hot": 0.45,
        "cold": 0.32,
        "rainy": 0.14,
        "snowy": 0.07,
    },
    "lightweight": {
        "sunny": 0.90,
        "cloudy": 0.78,
        "hot": 0.60,
        "cold": 0.55,
        "rainy": 0.30,
        "snowy": 0.16,
    },
    "supply": {
        "sunny": 0.90,
        "cloudy": 0.86,
        "hot": 0.95,
        "cold": 0.82,
        "rainy": 0.88,
        "snowy": 0.75,
    },
}

NATURAL_CANCELLATION_BASE = 0.018
NATURAL_CANCELLATION_PER_LEAD_DAY = 0.006
NATURAL_CANCELLATION_CAP = 0.10
NATURAL_CANCELLATION_FLEX_PROTECTION = 0.40
NATURAL_CANCELLATION_LATE_WEIGHT = 0.25


def forecast_regime(row: dict[str, Any]) -> str:
    """Derive the demand-relevant regime only from one public forecast row."""

    condition = str(row["condition"])
    if condition in {"rainy", "snowy"}:
        return condition
    high = float(row["day_high_c"])
    if high >= 32.0:
        return "hot"
    if high <= 8.0:
        return "cold"
    return condition


def consideration_probability(
    *,
    archetype: str,
    regime: str,
    forecast_trust: float | None,
) -> float:
    """Return request/visit consideration from public weather information."""

    table = CONSIDERATION_PROBABILITIES[archetype]
    target = float(table[regime])
    if forecast_trust is None:
        return target
    trust = min(1.0, max(0.0, float(forecast_trust)))
    neutral = float(table["cloudy"])
    return neutral + trust * (target - neutral)


def natural_cancellation_probability(
    *, booking_lead_days: int, schedule_flexibility: float
) -> float:
    """Cumulative probability of one non-weather shock before pickup."""

    lead = max(0, int(booking_lead_days))
    flexibility = min(1.0, max(0.0, float(schedule_flexibility)))
    raw = min(
        NATURAL_CANCELLATION_CAP,
        NATURAL_CANCELLATION_BASE + NATURAL_CANCELLATION_PER_LEAD_DAY * lead,
    )
    return raw * (1.0 - NATURAL_CANCELLATION_FLEX_PROTECTION * flexibility)


def natural_cancellation_day_weights(
    *, confirmed_day: int, pickup_day: int
) -> list[tuple[int, float]]:
    """Weight later pre-pickup dates more heavily without daily redraws."""

    days = list(range(int(confirmed_day) + 1, int(pickup_day)))
    return [
        (day, 1.0 + NATURAL_CANCELLATION_LATE_WEIGHT * index)
        for index, day in enumerate(days)
    ]


def weighted_index(draw: float, weights: list[float]) -> int:
    if not weights or any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("weights must be finite and positive")
    threshold = min(1.0, max(0.0, float(draw))) * sum(weights)
    cumulative = 0.0
    for index, value in enumerate(weights):
        cumulative += value
        if threshold <= cumulative:
            return index
    return len(weights) - 1


def policy_document() -> dict[str, Any]:
    return {
        "format": POLICY_ID,
        "potential_pool": "calendar-season-archetype-only-v1",
        "potential_pool_multipliers": POTENTIAL_POOL_MULTIPLIERS,
        "channel_and_lead_fixed_before_weather_consideration": True,
        "one_consideration_opportunity_per_customer": True,
        "reservation_information": "request-day-public-forecast",
        "walk_in_information": "arrival-day-observed-weather",
        "consideration_probabilities": CONSIDERATION_PROBABILITIES,
        "reservation_forecast_trust_blend": "toward-cloudy-neutral-v1",
        "weather_review": "incremental-adversity-once-per-contract-version-v1",
        "date_preference": "shared-public-forecast-calendar-distance-customer-v1",
        "natural_cancellation": {
            "model": "single-presampled-absolute-event-day-v1",
            "base": NATURAL_CANCELLATION_BASE,
            "per_booking_lead_day": NATURAL_CANCELLATION_PER_LEAD_DAY,
            "cap": NATURAL_CANCELLATION_CAP,
            "flexibility_protection": NATURAL_CANCELLATION_FLEX_PROTECTION,
            "late_day_weight": NATURAL_CANCELLATION_LATE_WEIGHT,
            "reschedule_redraws_event": False,
        },
        "future_truth_direct_customer_access": False,
    }
