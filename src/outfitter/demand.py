"""Auditable bounded reservation-price demand parameters for Outfitter V5."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


DEMAND_MODEL_FORMAT = "outfitter-bounded-reservation-demand-v1"

ORDINALS = frozenset({"low", "medium", "high"})
VARIATION_LEVELS = frozenset({"narrow", "medium", "wide"})
VALUE_UNIT_KINDS = frozenset(
    {"retail_sale_unit", "requested_unit_day", "rental_asset_day"}
)


@dataclass(frozen=True, slots=True)
class CustomerSegment:
    share: float
    reservation_multiplier: float
    quality_weight: float
    functional_weight: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CustomerSegment":
        return cls(**value)


@dataclass(frozen=True, slots=True)
class DemandGroupParameters:
    service_key: str
    choice_group: str
    value_unit: str
    value_unit_kind: str
    reservation_low_cny: float
    reservation_central_cny: float
    reservation_high_cny: float
    customer_variation: str
    need_strength: str
    quality_salience: str
    alternative_accessibility: str
    substitution_strength: str
    provenance: str = "operational_scenario_parameter"

    @property
    def key(self) -> str:
        return f"{self.service_key}|{self.choice_group}"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DemandGroupParameters":
        return cls(**value)


@dataclass(frozen=True, slots=True)
class DemandModel:
    format: str
    customer_segments: dict[str, CustomerSegment]
    groups: dict[str, DemandGroupParameters]
    variation_shapes: dict[str, float]
    quality_scores: dict[str, float]
    quality_salience_values: dict[str, float]
    need_salience_values: dict[str, float]
    accessibility_scores: dict[str, float]
    substitution_choice_taste: dict[str, float]
    context_accessibility_scores: dict[str, float]
    context_urgency_scores: dict[str, float]
    local_reservation_range: tuple[float, float]
    local_additive_shift_range: tuple[float, float]
    local_segment_share_max_multiplier: float
    profile_taste_range: tuple[float, float]
    choice_taste_range: tuple[float, float]
    accessibility_factor_range: tuple[float, float]
    retail_margin_floor: float
    initial_price_noise_range: tuple[float, float]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DemandModel":
        raw = dict(value)
        raw["customer_segments"] = {
            str(key): CustomerSegment.from_dict(dict(row))
            for key, row in dict(raw.get("customer_segments") or {}).items()
        }
        rows = list(raw.pop("group_parameters", []))
        raw["groups"] = {
            row.key: row
            for row in (DemandGroupParameters.from_dict(dict(item)) for item in rows)
        }
        for key in (
            "local_reservation_range",
            "local_additive_shift_range",
            "profile_taste_range",
            "choice_taste_range",
            "accessibility_factor_range",
            "initial_price_noise_range",
        ):
            raw[key] = tuple(raw[key])
        model = cls(**raw)
        model.validate()
        return model

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["group_parameters"] = [
            asdict(self.groups[key]) for key in sorted(self.groups)
        ]
        value.pop("groups")
        for key in (
            "local_reservation_range",
            "local_additive_shift_range",
            "profile_taste_range",
            "choice_taste_range",
            "accessibility_factor_range",
            "initial_price_noise_range",
        ):
            value[key] = list(value[key])
        return value

    def validate(self) -> None:
        if self.format != DEMAND_MODEL_FORMAT:
            raise ValueError(f"unsupported demand model format: {self.format}")
        if set(self.customer_segments) != {
            "price_sensitive",
            "balanced",
            "quality_seeking",
            "need_driven",
        }:
            raise ValueError("demand model must define the four fixed customer segments")
        shares = [row.share for row in self.customer_segments.values()]
        if any(not math.isfinite(value) or value <= 0 for value in shares):
            raise ValueError("customer segment shares must be finite and positive")
        if not math.isclose(sum(shares), 1.0, abs_tol=1e-6):
            raise ValueError("customer segment shares must sum to one")
        if not self.groups:
            raise ValueError("demand model must define at least one demand group")
        if set(self.variation_shapes) != VARIATION_LEVELS:
            raise ValueError("variation_shapes must define narrow, medium, and wide")
        for key, row in self.groups.items():
            if key != row.key:
                raise ValueError(f"demand group key mismatch: {key} != {row.key}")
            if row.value_unit_kind not in VALUE_UNIT_KINDS:
                raise ValueError(f"invalid value unit kind for {key}")
            if not (
                0 < row.reservation_low_cny
                <= row.reservation_central_cny
                <= row.reservation_high_cny
            ):
                raise ValueError(f"invalid reservation interval for {key}")
            if row.customer_variation not in VARIATION_LEVELS:
                raise ValueError(f"invalid customer variation for {key}")
            for field in (
                row.need_strength,
                row.quality_salience,
                row.alternative_accessibility,
                row.substitution_strength,
            ):
                if field not in ORDINALS:
                    raise ValueError(f"invalid ordinal demand parameter for {key}")
            if row.provenance != "operational_scenario_parameter":
                raise ValueError(f"invalid demand parameter provenance for {key}")
        for name, bounds in (
            ("local_reservation_range", self.local_reservation_range),
            ("profile_taste_range", self.profile_taste_range),
            ("accessibility_factor_range", self.accessibility_factor_range),
            ("initial_price_noise_range", self.initial_price_noise_range),
        ):
            if len(bounds) != 2 or not 0 < bounds[0] <= bounds[1]:
                raise ValueError(f"invalid {name}")
        if self.choice_taste_range[0] > self.choice_taste_range[1]:
            raise ValueError("invalid choice_taste_range")
        if self.local_additive_shift_range[0] > self.local_additive_shift_range[1]:
            raise ValueError("invalid local_additive_shift_range")
        if self.retail_margin_floor < 1:
            raise ValueError("retail margin floor must be at least one")

    def group(self, service_key: str, choice_group: str) -> DemandGroupParameters:
        key = f"{service_key}|{choice_group}"
        try:
            return self.groups[key]
        except KeyError as error:
            raise ValueError(f"demand model lacks group parameters: {key}") from error


@dataclass(frozen=True, slots=True)
class LocalGroupProfile:
    reservation_multiplier: float
    need_shift: float
    accessibility_shift: float
    request_rate_multiplier: float


@dataclass(frozen=True, slots=True)
class EpisodeMarketProfile:
    segment_shares: dict[str, float]
    groups: dict[str, LocalGroupProfile]

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_shares": dict(sorted(self.segment_shares.items())),
            "groups": {
                key: asdict(self.groups[key]) for key in sorted(self.groups)
            },
        }


def sample_beta_pert(
    rng: np.random.Generator,
    *,
    low: float,
    central: float,
    high: float,
    shape: float,
) -> float:
    """Draw from a bounded Beta-PERT distribution."""

    if math.isclose(low, high):
        return float(low)
    alpha = 1.0 + shape * (central - low) / (high - low)
    beta = 1.0 + shape * (high - central) / (high - low)
    return float(low + (high - low) * rng.beta(alpha, beta))


def functional_quantity_scale(
    category_id: str,
    choice_group: str | None,
    attributes: dict[str, Any],
) -> float:
    """Convert exact sale-unit functional quantity into bounded consumer value."""

    if category_id == "bottled_water":
        ratio, exponent = float(attributes.get("volume_ml", 500.0)) / 500.0, 0.80
    elif category_id == "cleaning_wipes":
        ratio, exponent = float(attributes.get("sheets_per_unit", 100.0)) / 100.0, 0.65
    elif category_id == "garbage_bags":
        ratio, exponent = float(attributes.get("bags_per_unit", 20.0)) / 20.0, 0.65
    elif category_id == "energy_snacks":
        ratio, exponent = float(attributes.get("net_weight_g_per_unit", 50.0)) / 50.0, 0.60
    elif category_id == "repellent" and choice_group == "topical_repellent":
        ratio, exponent = float(attributes.get("net_content_ml_per_unit", 100.0)) / 100.0, 0.65
    elif category_id == "repellent" and choice_group == "wearable_repellent":
        ratio, exponent = float(attributes.get("wearable_items_per_unit", 1.0)), 0.65
    elif category_id == "instant_food":
        ratio, exponent = float(attributes.get("net_weight_g_per_unit", 250.0)) / 250.0, 0.40
    else:
        return 1.0
    if not math.isfinite(ratio) or ratio <= 0:
        return 1.0
    return min(10.0, max(0.5, ratio**exponent))


def default_demand_model_payload(group_parameters: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap audited group rows in the fixed V1 global parameter contract."""

    return {
        "format": DEMAND_MODEL_FORMAT,
        "customer_segments": {
            "price_sensitive": {
                "share": 0.35,
                "reservation_multiplier": 0.80,
                "quality_weight": 0.60,
                "functional_weight": 1.00,
            },
            "balanced": {
                "share": 0.35,
                "reservation_multiplier": 1.00,
                "quality_weight": 1.00,
                "functional_weight": 1.00,
            },
            "quality_seeking": {
                "share": 0.15,
                "reservation_multiplier": 1.05,
                "quality_weight": 1.50,
                "functional_weight": 1.10,
            },
            "need_driven": {
                "share": 0.15,
                "reservation_multiplier": 1.20,
                "quality_weight": 0.80,
                "functional_weight": 1.25,
            },
        },
        "group_parameters": group_parameters,
        "variation_shapes": {"narrow": 8.0, "medium": 4.0, "wide": 2.0},
        "quality_scores": {
            "economy": -0.5,
            "standard": 0.0,
            "premium": 0.5,
            "unknown": 0.0,
        },
        "quality_salience_values": {"low": 0.5, "medium": 1.0, "high": 1.5},
        "need_salience_values": {"low": 0.15, "medium": 0.30, "high": 0.50},
        "accessibility_scores": {"low": -1.0, "medium": 0.0, "high": 1.0},
        "substitution_choice_taste": {"low": 0.02, "medium": 0.035, "high": 0.05},
        "context_accessibility_scores": {
            "supply_stop": 0.25,
            "day_picnic": -0.20,
            "overnight_camping": -0.35,
            "solo_hiking": -0.40,
            "emergency_rain": -0.60,
            "hot_water_need": -0.40,
        },
        "context_urgency_scores": {
            "supply_stop": 0.0,
            "day_picnic": 0.20,
            "overnight_camping": 0.35,
            "solo_hiking": 0.30,
            "emergency_rain": 1.0,
            "hot_water_need": 0.70,
        },
        "local_reservation_range": [0.85, 1.15],
        "local_additive_shift_range": [-0.20, 0.20],
        "local_segment_share_max_multiplier": 1.20,
        "profile_taste_range": [0.95, 1.05],
        "choice_taste_range": [-0.05, 0.05],
        "accessibility_factor_range": [0.65, 1.45],
        "retail_margin_floor": 1.25,
        "initial_price_noise_range": [0.90, 1.10],
    }
