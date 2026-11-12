from __future__ import annotations

import hashlib
import json
import math
import copy
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, ROUND_HALF_UP
from statistics import median
from typing import Any, Iterable

import numpy as np

from ..channel_demand import (
    ChannelDecisionTrace,
    ChannelDemandModel,
    ChannelEvaluationInput,
    ChannelOfferView,
    QUALITY_SCORES as CHANNEL_QUALITY_SCORES,
    evaluate_channel_demand,
)
from ..config import OutfitterConfig
from ..data import OutfitterDataset
from ..demand import (
    DemandGroupParameters,
    DemandModel,
    EpisodeMarketProfile,
    LocalGroupProfile,
    functional_quantity_scale,
    sample_beta_pert,
)
from ..models import (
    CalendarBlock,
    CleaningJob,
    PurchaseOrder,
    RentalAsset,
    RentalContract,
    ReservationAssignment,
    ReservationChangeRequest,
    ReservationCommitment,
    ReservationEvent,
    ReservationLine,
    ReservationRequest,
    SupplierTrialRecord,
    ToolResult,
    WalkInCapacityLimit,
    WorldState,
)
from ..reference_demand import (
    ACCESSIBILITY_ADJUSTMENTS,
    NEED_ADJUSTMENTS,
    QUALITY_SALIENCE,
    QUALITY_SCORES,
    SUBSTITUTE_PENALTIES,
    ReferenceDecisionTrace,
    ReferenceEvaluationInput,
    ReferenceOfferView,
    ReferenceTransactionModel,
    evaluate_reference_demand,
)
from ..v5 import (
    CHOICE_GROUPS_BY_SERVICE,
    CLEANING_WIPE_BOXES_PER_REQUEST,
    V5MarketRules,
    gumbel_from_uniform,
    logsumexp,
    seasonal_index,
    stable_words,
)
from .ledger import _empty_agent_actions, _empty_ledger


PAD_LINES = ("closed_cell_foam_pad", "self_inflating_pad", "air_mattress")
DEPOSIT_RATIOS = {
    "tent": 0.60,
    "sleeping_bag": 0.35,
    "closed_cell_foam_pad": 0.25,
    "self_inflating_pad": 0.40,
    "air_mattress": 0.55,
    "camp_table": 0.30,
    "camp_chair": 0.30,
    "camp_lantern": 0.40,
    "camp_cookware": 0.25,
}
DAMAGE_RATES = {"low": 0.004, "medium": 0.010, "high": 0.020}
WEATHER_MULTIPLIERS = {
    "sunny": 1.15,
    "cloudy": 1.0,
    "hot": 0.9,
    "cold": 0.8,
    "rainy": 0.55,
    "snowy": 0.3,
}
ARCHETYPES = {
    "overnight": (0.7, 1.4),
    "picnic": (0.8, 1.6),
    "lightweight": (0.5, 0.9),
    "supply": (1.5, 2.3),
}
SERVICE_INTERCEPT = 0.0
RENT_PRICE_BETA = 2.5
RETAIL_PRICE_BETA = 3.0
OFFER_CHOICE_BETA = 3.0
QUALITY_ADJUSTMENTS = {"economy": -0.2, "standard": 0.0, "premium": 0.2}
RETAIL_HOLDING_RATE = 0.001
RENTAL_HOLDING_RATE = 0.0005
RETAIL_SALVAGE_RATE = 0.50
MONEY_QUANTUM = Decimal("0.01")
INVENTORY_COST_QUANTUM = Decimal("0.0001")
CLEANING_COST_RULES = {
    "camp_chair": (0.50, 0.03),
    "camp_lantern": (0.50, 0.03),
    "camp_table": (0.75, 0.05),
    "closed_cell_foam_pad": (0.65, 0.08),
    "air_mattress": (0.90, 0.10),
    "self_inflating_pad": (1.00, 0.10),
    "camp_cookware": (1.25, 0.10),
    "sleeping_bag": (2.30, 0.20),
    "tent": (3.20, 0.25),
}
SECONDHAND_GRADE_SCHEDULE = (
    ("excellent", 0.80, 0.55),
    ("good", 0.60, 0.40),
    ("fair", 0.35, 0.25),
    ("worn", 0.00, 0.10),
)
SECONDHAND_PARTS_FLOOR_RATE = 0.05
SECONDHAND_SELLABLE_STATUSES = frozenset({"available", "damaged"})
TERMINAL_HEALTHY_STATUSES = frozenset(
    {"available", "cleaning", "repairing", "maintenance", "maintenance_hold"}
)
RENTAL_ACTIVITY_EVENT_TYPES = (
    "asset_acquired",
    "rental_started",
    "returned_normal",
    "returned_damaged",
    "retired_service_life",
    "cleaning_completed",
    "repair_started",
    "repair_completed",
    "retired_by_agent",
    "asset_sold",
    "maintenance_started",
    "maintenance_completed",
    "maintenance_hold_started",
)


def _catalog_search_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(
        token
        for token in re.split(r"[^\w\u3400-\u9fff]+", normalized)
        if token
    )


def _catalog_query_tokens(query: Any) -> tuple[str, ...]:
    normalized = _catalog_search_text(query)
    return tuple(dict.fromkeys(token for token in normalized.split() if token))


def _catalog_match_score(
    public: dict[str, Any], query: Any
) -> tuple[int, int, int, int, int] | None:
    """Rank deterministic keyword matches over Agent-visible offer fields."""

    phrase = _catalog_search_text(query)
    tokens = _catalog_query_tokens(query)
    if not tokens:
        return (0, 0, 0, 0, 0)
    fields = [
        _catalog_search_text(public.get("offer_id")),
        _catalog_search_text(public.get("name")),
        _catalog_search_text(public.get("source_variant_name")),
        _catalog_search_text(public.get("source_title")),
        _catalog_search_text(public.get("brand")),
        _catalog_search_text(public.get("seller_name")),
        _catalog_search_text(
            " ".join(
                f"{row.get('name')} {row.get('value')} {row.get('evidence')}"
                for row in public.get("specifications", [])
            )
        ),
    ]
    token_hits = [any(token in field for field in fields) for token in tokens]
    matched = sum(token_hits)
    if matched == 0:
        return None
    field_weights = (6, 5, 4, 3, 2, 1, 2)
    best_field = max(
        (
            weight
            for weight, field in zip(field_weights, fields)
            if any(token in field for token in tokens)
        ),
        default=0,
    )
    return (
        int(phrase == fields[0]),
        int(bool(phrase) and any(phrase in field for field in fields)),
        int(all(token_hits)),
        matched,
        best_field,
    )

# Stable integer coordinates for independent random streams. These values are part
# of the benchmark's random-event layout: changing a manager's actions must not
# move unrelated customer or damage draws forward or backward.
_RNG_ARRIVALS = 1
_RNG_CUSTOMER_BASKET = 2
_RNG_REQUEST_ACCEPTANCE = 3
_RNG_REQUEST_OFFER_CHOICE = 4
_RNG_REQUEST_DAMAGE = 5
_RNG_PARTY_COMPOSITION = 6
_RNG_SESSION_PREFERENCE = 7
_RNG_SESSION_BUDGET = 8
_RNG_SUBSTITUTION = 9
_RNG_SERVICE_SHOCK = 10
_RNG_GROUP_SHOCK = 11
_RNG_OFFER_SHOCK = 12
_RNG_FORECAST = 13
_RNG_EPISODE_MARKET = 14
_RNG_CUSTOMER_SEGMENT = 15
_RNG_RESERVATION_PRICE = 16
_RNG_PROFILE_TASTE = 17
_RNG_PROFILE_CHOICE = 18
_RNG_CHANNEL_AFFINITY = 19
_RNG_EXISTING_OPTION = 20
_RNG_SUPPLIER_DELIVERY = 21
# V6+ streams are append-only.  Never renumber the V5 coordinates above.
_RNG_CHANNEL_BOUNDED_TASTE = 22
_RNG_RENTAL_CONTEXT = 23
_RNG_RESERVATION_ROUTE = 24
_RNG_RESERVATION_LEAD = 25
_RNG_RESERVATION_WEATHER_RESPONSE = 26
_RNG_RESERVATION_CHANGE = 27
_RNG_RESERVATION_LATE_RETURN = 28
_RNG_RESERVATION_FALLBACK = 29
_RNG_RESERVATION_QUOTE_CANDIDATES = 30
_RNG_RESERVATION_QUOTE_RECOVERY = 31
# V13+ lifecycle streams are append-only. Damage event occurrence retains the
# existing stream; severity is independent so adding it cannot move old draws.
_RNG_REQUEST_DAMAGE_SEVERITY = 32
_RNG_REQUEST_BUDGET = 33
_RNG_RESERVATION_BUNDLE_CHOICE = 34
# V19 complete-contract feedback streams are append-only.  Response
# availability is request-level; report noise is shared across the request's
# unselected contract options so similar alternatives receive correlated
# customer reports without advancing any older tape.
_RNG_RESERVATION_FEEDBACK_RESPONSE = 35
_RNG_RESERVATION_FEEDBACK_REPORT = 36
_RNG_CUSTOMER_WEATHER_CONSIDERATION = 37
_RNG_NATURAL_CANCELLATION = 38
RESERVATION_STORAGE_CATEGORY_ADJUSTMENT = {
    "rental:tent": 0.15,
    "rental:camp_table": 0.14,
    "rental:camp_chair": 0.08,
    "rental:sleeping_pad": 0.06,
    "rental:sleeping_bag": 0.04,
    "rental:camp_cookware": -0.03,
    "rental:camp_lantern": -0.10,
}
RESERVATION_REUSE_CATEGORY_ADJUSTMENT = {
    "rental:sleeping_bag": 0.08,
    "rental:sleeping_pad": 0.06,
    "rental:tent": 0.04,
    "rental:camp_lantern": 0.02,
    "rental:camp_cookware": 0.0,
    "rental:camp_chair": -0.04,
    "rental:camp_table": -0.06,
}
_ARCHETYPE_INDEX = {
    name: index for index, name in enumerate(ARCHETYPES, start=1)
}


@dataclass(frozen=True, slots=True)
class ChoiceSession:
    interaction_id: str
    service_key: str
    units_requested: int
    duration_days: int | None
    adults: int
    children: int
    size_class: str
    visible_party_context: str
    use_case: str
    preferred_group: str
    budget: float
    price_beta: float
    substitution_tolerance: float
    portability_sensitive: bool
    planned_purchase: bool
    event_address: tuple[int, int, int, int]
    archetype: str = "supply"
    customer_segment: str = "balanced"
    intent_role: str = "optional"
    need_horizon_days: int | None = None
    occasion_importance: float = 1.0
    booking_lead_days: int = 0
    pickup_day: int | None = None
    forecast_adversity: float = 0.0
    forecast_uncertainty: float = 0.0
    forecast_trust: float = 0.5
    future_use_expectation: float = 0.0
    storage_burden: float = 0.0
    return_burden_multiplier: float = 1.0
    request_budget_multiplier: float = 1.0
    trip_weather_context: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ChoiceServiceResult:
    fulfilled_units: int
    request_satisfied: bool
    internal_channel: str
