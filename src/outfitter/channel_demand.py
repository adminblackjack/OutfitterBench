"""Exact-offer channel-choice demand model.

The evaluator in this module is deliberately pure.  Production simulation and
``outfitter.demand_lab`` both build the same input structure and call the same
function, which keeps channel mechanics auditable without exposing them to the
shop manager.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any


CHANNEL_DEMAND_FORMAT = "outfitter-exact-offer-channel-demand-v4"
REQUEST_BUNDLE_CHANNEL_DEMAND_FORMAT = (
    "outfitter-exact-offer-channel-demand-v5-request-bundle"
)
LEGACY_CHANNEL_DEMAND_FORMATS = {
    "outfitter-exact-offer-channel-demand-v1",
    "outfitter-exact-offer-channel-demand-v2",
    "outfitter-exact-offer-channel-demand-v3",
}
DEADLINE_AWARE_RETAIL_FORMATS = {
    "outfitter-exact-offer-channel-demand-v3",
    CHANNEL_DEMAND_FORMAT,
    REQUEST_BUNDLE_CHANNEL_DEMAND_FORMAT,
}
CONTEXTUAL_RENTAL_FORMATS = {
    CHANNEL_DEMAND_FORMAT,
    REQUEST_BUNDLE_CHANNEL_DEMAND_FORMAT,
}
CHANNEL_TRACE_FORMAT = "outfitter-exact-offer-channel-trace-v4"

QUALITY_SCORES = {
    "economy": -0.5,
    "standard": 0.0,
    "premium": 0.5,
    "unknown": 0.0,
}
QUALITY_SALIENCE = {"low": 0.5, "medium": 1.0, "high": 1.5}
SUBSTITUTE_PENALTIES = {"low": 1.20, "medium": 0.70, "high": 0.35}


@dataclass(frozen=True, slots=True)
class ChannelWorld:
    world_id: str
    common_price_beta: float
    absolute_price_weight: float
    retail_markup_weight: float
    retail_store_convenience: float
    rental_relative_weight: float
    rental_convenience: float
    online_friction: float
    retail_context_multiplier: float = 1.0
    online_arrival_cdf_day0: float = 0.05
    online_arrival_cdf_day1: float = 0.35
    online_arrival_cdf_day3: float = 0.90

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ChannelWorld":
        row = cls(**value)
        numeric = tuple(
            float(getattr(row, name))
            for name in (
                "common_price_beta",
                "absolute_price_weight",
                "retail_markup_weight",
                "rental_relative_weight",
            )
        )
        if not row.world_id or any(not math.isfinite(item) or item <= 0 for item in numeric):
            raise ValueError("channel world price parameters must be finite and positive")
        if any(
            not math.isfinite(float(getattr(row, name)))
            for name in (
                "retail_store_convenience",
                "rental_convenience",
                "online_friction",
            )
        ):
            raise ValueError("channel world convenience parameters must be finite")
        if not math.isfinite(float(row.retail_context_multiplier)) or (
            row.retail_context_multiplier <= 0
        ):
            raise ValueError("retail_context_multiplier must be finite and positive")
        arrival = (
            float(row.online_arrival_cdf_day0),
            float(row.online_arrival_cdf_day1),
            float(row.online_arrival_cdf_day3),
        )
        if any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in arrival):
            raise ValueError("consumer online arrival CDF must be within [0, 1]")
        if not arrival[0] <= arrival[1] <= arrival[2]:
            raise ValueError("consumer online arrival CDF must be monotone")
        return row


DEFAULT_CHANNEL_WORLDS = {
    row.world_id: row
    for row in (
        ChannelWorld(
            "conservative", 4.0, 2.8, 3.5, 0.08, 1.5, 0.10, 0.05,
            retail_context_multiplier=0.85,
            online_arrival_cdf_day0=0.08,
            online_arrival_cdf_day1=0.50,
            online_arrival_cdf_day3=0.95,
        ),
        ChannelWorld(
            "base", 3.0, 2.6, 3.0, 0.20, 1.2, 0.30, 0.20,
            retail_context_multiplier=1.00,
            online_arrival_cdf_day0=0.05,
            online_arrival_cdf_day1=0.35,
            online_arrival_cdf_day3=0.90,
        ),
        ChannelWorld(
            "convenience-heavy", 2.5, 2.4, 2.5, 0.40, 1.0, 0.70, 0.30,
            retail_context_multiplier=1.15,
            online_arrival_cdf_day0=0.03,
            online_arrival_cdf_day1=0.25,
            online_arrival_cdf_day3=0.80,
        ),
    )
}


@dataclass(frozen=True, slots=True)
class ChannelSegment:
    share: float
    cash_scale_cny: float
    absolute_weight_multiplier: float
    markup_weight_multiplier: float
    functional_weight: float
    quality_weight: float
    rental_weight_multiplier: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ChannelSegment":
        row = cls(**value)
        if any(
            not math.isfinite(float(item)) or float(item) <= 0
            for item in asdict(row).values()
        ):
            raise ValueError("channel segment parameters must be finite and positive")
        return row


DEFAULT_CHANNEL_SEGMENTS = {
    "price_sensitive": ChannelSegment(0.35, 7.0, 1.25, 1.25, 1.00, 0.60, 1.15),
    "balanced": ChannelSegment(0.35, 10.0, 1.00, 1.00, 1.00, 1.00, 1.00),
    "quality_seeking": ChannelSegment(0.15, 13.0, 0.80, 0.80, 1.10, 1.50, 0.85),
    "need_driven": ChannelSegment(0.15, 12.0, 0.85, 0.85, 1.25, 0.80, 0.90),
}


@dataclass(frozen=True, slots=True)
class ChannelGroupPolicy:
    service_key: str
    choice_group: str
    quality_salience: str
    substitution_strength: str

    @property
    def key(self) -> str:
        return f"{self.service_key}|{self.choice_group}"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ChannelGroupPolicy":
        row = cls(**value)
        if row.quality_salience not in QUALITY_SALIENCE:
            raise ValueError("invalid channel quality_salience")
        if row.substitution_strength not in SUBSTITUTE_PENALTIES:
            raise ValueError("invalid channel substitution_strength")
        return row


@dataclass(frozen=True, slots=True)
class ChannelDemandModel:
    format: str
    groups: dict[str, ChannelGroupPolicy]
    segments: dict[str, ChannelSegment]
    worlds: dict[str, ChannelWorld]
    cash_scale_unit: str = "CNY per complete customer request"
    required_need_intercept: float = 6.9
    optional_need_intercept: float = 4.4
    deposit_friction_rate: float = 0.05
    return_burden: float = 1.6
    ownership_burden: float = 0.1
    future_use_credit: float = 0.0
    existing_option_bonus: float = 0.75
    channel_shock_scale: float = 0.6
    profile_shock_scale: float = 0.25
    deadline_value_scale: float = 1.0
    retail_immediate_value_rate: float = 0.35
    retail_unmet_need_loss_rate: float = 1.0
    rental_log_ratio_clip: tuple[float, float] = (-2.302585092994046, 0.6931471805599453)
    # V4 completes the existing buy-versus-rent decomposition with explicit,
    # bounded request context.  The values are synthetic behavioral terms, not
    # claims about observed Beijing elasticities.
    rental_duration_value_rate: float = 0.45
    rental_weather_disutility_rate: float = 0.55
    rental_delivery_lateness_loss_rate: float = 0.08
    rental_immediate_value_rate: float = 0.08
    rental_date_guarantee_rate: float = 0.12
    rental_flexibility_value_rate: float = 0.30
    rental_future_use_credit_rate: float = 0.80
    rental_storage_burden_rate: float = 0.65
    channel_taste_scale: float = 0.55
    # Cash paid for temporary use has additional salience beyond the shared
    # household-budget penalty. This is a rental-demand scale, never a pricing
    # formula and never a function of supplier cost or target payback.
    rental_contract_cash_salience: float = 1.0
    # V5 advance-reservation request-level choice. These parameters are ignored
    # by every legacy format and by lead-zero V5 walk-ins, so the frozen V17
    # walk-in market and old runtime snapshots retain their behavior.
    rental_quantity_value_exponent: float = 0.80
    rental_duration_value_exponent: float = 0.70
    rental_party_budget_exponent: float = 0.50
    rental_duration_budget_exponent: float = 1.00
    request_budget_sigma: float = 0.25
    rental_request_cash_salience: float = 3.80
    rental_budget_tail_coefficient: float = 0.0
    request_budget_bounds: list[float] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ChannelDemandModel":
        raw = dict(value)
        policies = [
            ChannelGroupPolicy.from_dict(dict(row))
            for row in raw.pop("group_policies", [])
        ]
        raw["groups"] = {row.key: row for row in policies}
        raw["segments"] = {
            str(key): ChannelSegment.from_dict(dict(row))
            for key, row in dict(raw.get("segments") or {}).items()
        }
        raw["worlds"] = {
            str(key): ChannelWorld.from_dict(dict(row))
            for key, row in dict(raw.get("worlds") or {}).items()
        }
        raw["rental_log_ratio_clip"] = tuple(
            map(float, raw.get("rental_log_ratio_clip", cls.rental_log_ratio_clip))
        )
        model = cls(**raw)
        model.validate()
        return model

    def validate(self) -> None:
        if not math.isfinite(self.rental_budget_tail_coefficient) or self.rental_budget_tail_coefficient<0:
            raise ValueError("invalid rental budget tail coefficient")
        if self.request_budget_bounds is not None:
            if (len(self.request_budget_bounds)!=2 or not all(math.isfinite(x) for x in self.request_budget_bounds)
                    or not 0<self.request_budget_bounds[0]<self.request_budget_bounds[1] or self.request_budget_sigma<=0):
                raise ValueError("invalid request budget bounds")
        if self.format not in {
            CHANNEL_DEMAND_FORMAT,
            REQUEST_BUNDLE_CHANNEL_DEMAND_FORMAT,
            *LEGACY_CHANNEL_DEMAND_FORMATS,
        }:
            raise ValueError(f"unsupported channel demand format: {self.format}")
        if set(self.segments) != set(DEFAULT_CHANNEL_SEGMENTS):
            raise ValueError("channel model must define the four fixed customer segments")
        if not math.isclose(
            sum(row.share for row in self.segments.values()), 1.0, abs_tol=1e-9
        ):
            raise ValueError("channel segment shares must sum to one")
        if set(self.worlds) != set(DEFAULT_CHANNEL_WORLDS):
            raise ValueError("channel model must define the three formal worlds")
        if len(self.rental_log_ratio_clip) != 2 or self.rental_log_ratio_clip[0] > self.rental_log_ratio_clip[1]:
            raise ValueError("invalid rental log-ratio clip")
        for name in (
            "required_need_intercept",
            "optional_need_intercept",
            "deposit_friction_rate",
            "return_burden",
            "ownership_burden",
            "channel_shock_scale",
            "profile_shock_scale",
            "deadline_value_scale",
            "retail_immediate_value_rate",
            "retail_unmet_need_loss_rate",
            "rental_duration_value_rate",
            "rental_weather_disutility_rate",
            "rental_delivery_lateness_loss_rate",
            "rental_immediate_value_rate",
            "rental_date_guarantee_rate",
            "rental_flexibility_value_rate",
            "rental_future_use_credit_rate",
            "rental_storage_burden_rate",
            "channel_taste_scale",
            "rental_contract_cash_salience",
        ):
            value = float(getattr(self, name))
            if (
                not math.isfinite(value)
                or value < 0
                or (name == "rental_contract_cash_salience" and value <= 0)
            ):
                raise ValueError(f"invalid channel model parameter: {name}")
        if self.format == REQUEST_BUNDLE_CHANNEL_DEMAND_FORMAT:
            for name in (
                "rental_quantity_value_exponent",
                "rental_duration_value_exponent",
                "rental_party_budget_exponent",
                "rental_duration_budget_exponent",
                "request_budget_sigma",
                "rental_request_cash_salience",
            ):
                value = float(getattr(self, name))
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid V5 channel model parameter: {name}")
            if self.rental_request_cash_salience <= 0:
                raise ValueError(
                    "rental_request_cash_salience must be positive"
                )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "format": self.format,
            "group_policies": [asdict(self.groups[key]) for key in sorted(self.groups)],
            "segments": {key: asdict(self.segments[key]) for key in sorted(self.segments)},
            "worlds": {key: asdict(self.worlds[key]) for key in sorted(self.worlds)},
            "cash_scale_unit": self.cash_scale_unit,
            "required_need_intercept": self.required_need_intercept,
            "optional_need_intercept": self.optional_need_intercept,
            "deposit_friction_rate": self.deposit_friction_rate,
            "return_burden": self.return_burden,
            "ownership_burden": self.ownership_burden,
            "future_use_credit": self.future_use_credit,
            "existing_option_bonus": self.existing_option_bonus,
            "channel_shock_scale": self.channel_shock_scale,
            "profile_shock_scale": self.profile_shock_scale,
            "deadline_value_scale": self.deadline_value_scale,
            "retail_immediate_value_rate": self.retail_immediate_value_rate,
            "retail_unmet_need_loss_rate": self.retail_unmet_need_loss_rate,
            "rental_log_ratio_clip": list(self.rental_log_ratio_clip),
            "rental_duration_value_rate": self.rental_duration_value_rate,
            "rental_weather_disutility_rate": self.rental_weather_disutility_rate,
            "rental_delivery_lateness_loss_rate": self.rental_delivery_lateness_loss_rate,
            "rental_immediate_value_rate": self.rental_immediate_value_rate,
            "rental_date_guarantee_rate": self.rental_date_guarantee_rate,
            "rental_flexibility_value_rate": self.rental_flexibility_value_rate,
            "rental_future_use_credit_rate": self.rental_future_use_credit_rate,
            "rental_storage_burden_rate": self.rental_storage_burden_rate,
            "channel_taste_scale": self.channel_taste_scale,
            "rental_contract_cash_salience": self.rental_contract_cash_salience,
        }
        if self.format == REQUEST_BUNDLE_CHANNEL_DEMAND_FORMAT:
            result.update(
                {
                    "rental_quantity_value_exponent": (
                        self.rental_quantity_value_exponent
                    ),
                    "rental_duration_value_exponent": (
                        self.rental_duration_value_exponent
                    ),
                    "rental_party_budget_exponent": (
                        self.rental_party_budget_exponent
                    ),
                    "rental_duration_budget_exponent": (
                        self.rental_duration_budget_exponent
                    ),
                    "request_budget_sigma": self.request_budget_sigma,
                    "rental_request_cash_salience": (
                        self.rental_request_cash_salience
                    ),
                }
            )
        if self.rental_budget_tail_coefficient or self.request_budget_bounds is not None:
            result.update(rental_budget_tail_coefficient=self.rental_budget_tail_coefficient,
                          request_budget_bounds=self.request_budget_bounds)
        return result


@dataclass(frozen=True, slots=True)
class ChannelOfferView:
    offer_id: str
    profile_id: str
    choice_group: str
    kind: str
    hard_compatible: bool
    incompatibility_reasons: tuple[str, ...]
    soft_fit: float
    quality_score: float
    requested_units: int
    available_units: int
    store_unit_price: float | None
    store_cash_price: float | None
    store_deposit: float
    online_package_price: float
    online_package_units: int
    online_unit_price: float
    online_cash_price: float
    requested_customer_units: int = 1
    units_per_resource: int = 1


@dataclass(frozen=True, slots=True)
class ChannelEvaluationInput:
    service_key: str
    preferred_group: str
    customer_segment: str
    world_id: str
    intent_role: str
    existing_option_available: bool
    channel_affinities: dict[str, float]
    channel_gumbels: dict[str, float]
    profile_gumbels: dict[str, float]
    offers: tuple[ChannelOfferView, ...]
    need_horizon_days: int | None = None
    occasion_importance: float = 1.0
    duration_days: int | None = None
    booking_lead_days: int = 0
    forecast_adversity: float = 0.0
    forecast_uncertainty: float = 0.0
    forecast_trust: float = 0.5
    future_use_expectation: float = 0.0
    storage_burden: float = 0.0
    return_burden_multiplier: float = 1.0
    online_delivery_on_time_probability: float | None = None
    channel_tastes: dict[str, float] | None = None
    request_budget_multiplier: float = 1.0
    party_equivalent: float = 1.0


@dataclass(frozen=True, slots=True)
class ChannelDecisionTrace:
    format: str
    world_id: str
    service_key: str
    preferred_group: str
    customer_segment: str
    intent_role: str
    profiles: tuple[dict[str, Any], ...]
    best_store_profile_id: str | None
    best_store_plan_id: str | None
    best_online_profile_id: str | None
    channel_systematic: dict[str, float]
    channel_scores: dict[str, float]
    selected_channel: str
    selected_profile_id: str | None
    selected_store_plan_id: str | None
    selected_offer_id: str | None
    need_horizon_days: int | None = None
    occasion_importance: float = 1.0
    online_late_probability: float = 0.0
    rental_context: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cash_penalty(cash: float, *, world: ChannelWorld, segment: ChannelSegment) -> float:
    return (
        world.absolute_price_weight
        * segment.absolute_weight_multiplier
        * math.log1p(max(0.0, cash) / segment.cash_scale_cny)
    )


def effective_rental_cash_segment(
    model: ChannelDemandModel,
    segment: ChannelSegment,
    *,
    duration_days: int,
    party_equivalent: float,
    request_budget_multiplier: float,
) -> ChannelSegment:
    """Return the V5 request cash scale without mutating model state."""

    if model.format != REQUEST_BUNDLE_CHANNEL_DEMAND_FORMAT:
        return segment
    budget = float(request_budget_multiplier)
    party = float(party_equivalent)
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("request_budget_multiplier must be finite and positive")
    if not math.isfinite(party) or party <= 0:
        raise ValueError("party_equivalent must be finite and positive")
    scale = (
        budget
        * party ** float(model.rental_party_budget_exponent)
        * max(1, int(duration_days))
        ** float(model.rental_duration_budget_exponent)
    )
    return replace(segment, cash_scale_cny=segment.cash_scale_cny * scale)


def _online_late_probability(
    world: ChannelWorld,
    need_horizon_days: int | None,
    *,
    interpolate: bool = False,
) -> float:
    if need_horizon_days is None:
        return 0.0
    arrival = {
        0: world.online_arrival_cdf_day0,
        1: world.online_arrival_cdf_day1,
        3: world.online_arrival_cdf_day3,
    }
    if interpolate:
        lead = max(0, int(need_horizon_days))
        if lead == 0:
            probability = float(world.online_arrival_cdf_day0)
        elif lead == 1:
            probability = float(world.online_arrival_cdf_day1)
        elif lead == 2:
            probability = 0.5 * (
                float(world.online_arrival_cdf_day1)
                + float(world.online_arrival_cdf_day3)
            )
        else:
            residual = 1.0 - float(world.online_arrival_cdf_day3)
            probability = 1.0 - residual * math.exp(-0.55 * (lead - 3))
        return 1.0 - min(1.0, max(0.0, probability))
    try:
        return 1.0 - float(arrival[int(need_horizon_days)])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("retail need_horizon_days must be 0, 1, 3, or null") from error


def _common_product_utility(
    row: ChannelOfferView,
    *,
    value: ChannelEvaluationInput,
    segment: ChannelSegment,
    policy: ChannelGroupPolicy,
) -> tuple[float, float, float]:
    quality = (
        0.18
        * segment.quality_weight
        * QUALITY_SALIENCE[policy.quality_salience]
        * row.quality_score
    )
    substitute = (
        0.0
        if row.choice_group == value.preferred_group
        else SUBSTITUTE_PENALTIES[policy.substitution_strength]
    )
    common = segment.functional_weight * row.soft_fit + quality - substitute
    return common, quality, substitute


def _best(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: (float(row[key]), str(row["profile_id"])))


def evaluate_channel_demand(
    model: ChannelDemandModel,
    value: ChannelEvaluationInput,
) -> ChannelDecisionTrace:
    """Evaluate one articulated request without mutating business state.

    Store plans may combine distinct compatible consumer profiles.  Required
    intents remain all-or-nothing, while optional intents may expose and sell
    the available fraction.  Inventory beyond the requested quantity is
    ignored, and profile shocks enter only after channel entry.
    """

    if value.intent_role not in {"required", "optional"}:
        raise ValueError("channel intent_role must be required or optional")
    try:
        world = model.worlds[value.world_id]
        segment = model.segments[value.customer_segment]
        policy = model.groups[f"{value.service_key}|{value.preferred_group}"]
    except KeyError as error:
        raise ValueError(f"channel evaluation lacks policy: {error}") from error
    need = (
        model.required_need_intercept
        if value.intent_role == "required"
        else model.optional_need_intercept
    )
    by_profile: dict[str, list[ChannelOfferView]] = {}
    for row in value.offers:
        if row.hard_compatible:
            by_profile.setdefault(row.profile_id, []).append(row)
    rental = value.service_key.startswith("rental:")
    contextual_rental = model.format in CONTEXTUAL_RENTAL_FORMATS
    # V5 is a versioned advance-reservation construct.  Applying its
    # request-bundle cash scale to lead-zero walk-ins silently changed the V17
    # walk-in market and confounded reservation evaluation.  Advance inquiries
    # always have a positive booking lead; lead-zero rental/retail behavior
    # therefore remains the frozen V4 control.
    request_bundle_v5 = bool(
        model.format == REQUEST_BUNDLE_CHANNEL_DEMAND_FORMAT
        and int(value.booking_lead_days) > 0
    )
    deadline_aware_retail = (
        model.format in DEADLINE_AWARE_RETAIL_FORMATS and not rental
    )
    if not math.isfinite(float(value.occasion_importance)) or (
        value.occasion_importance <= 0
    ):
        raise ValueError("occasion_importance must be finite and positive")
    online_late_probability = (
        _online_late_probability(world, value.need_horizon_days)
        if deadline_aware_retail
        else 0.0
    )
    context_scale = (
        model.deadline_value_scale
        * world.retail_context_multiplier
        * online_late_probability
        * float(value.occasion_importance)
        if deadline_aware_retail
        else 0.0
    )
    immediate_fulfillment_value = (
        model.retail_immediate_value_rate * context_scale
    )
    expected_lateness_loss = model.retail_unmet_need_loss_rate * context_scale
    rental_context: dict[str, float] | None = None
    rental_shared_scale = 1.0
    rental_online_late_probability = 0.0
    rental_online_penalty = 0.0
    rental_store_service_value = 0.0
    rental_ownership_burden = model.ownership_burden
    rental_future_use_credit = model.future_use_credit
    rental_return_burden = model.return_burden
    target_units = max(
        (int(row.requested_customer_units) for row in value.offers),
        default=1,
    )
    cash_segment = segment
    if rental and contextual_rental:
        duration = max(1, int(value.duration_days or 1))
        lead = max(0, int(value.booking_lead_days))
        adversity = min(1.0, max(0.0, float(value.forecast_adversity)))
        uncertainty = min(1.0, max(0.0, float(value.forecast_uncertainty)))
        trust = min(1.0, max(0.0, float(value.forecast_trust)))
        importance = min(2.0, max(0.25, float(value.occasion_importance)))
        future_use = min(1.0, max(0.0, float(value.future_use_expectation)))
        storage = min(1.0, max(0.0, float(value.storage_burden)))
        return_multiplier = min(
            2.0, max(0.25, float(value.return_burden_multiplier))
        )
        rental_online_late_probability = (
            1.0
            - min(
                1.0,
                max(0.0, float(value.online_delivery_on_time_probability)),
            )
            if value.online_delivery_on_time_probability is not None
            else _online_late_probability(world, lead, interpolate=True)
        )
        duration_value = (
            duration ** float(model.rental_duration_value_exponent)
            if request_bundle_v5
            else 1.0
            + model.rental_duration_value_rate * math.log1p(duration - 1)
        )
        quantity_value = (
            target_units ** float(model.rental_quantity_value_exponent)
            if request_bundle_v5
            and value.service_key
            in {
                "rental:camp_chair",
                "rental:sleeping_bag",
                "rental:sleeping_pad",
            }
            else 1.0
        )
        weather_value = max(
            0.25,
            1.0
            - model.rental_weather_disutility_rate * adversity * trust,
        )
        rental_shared_scale = (
            duration_value * quantity_value * weather_value * importance
        )
        cash_segment = (
            effective_rental_cash_segment(
                model,
                segment,
                duration_days=duration,
                party_equivalent=float(value.party_equivalent),
                request_budget_multiplier=float(
                    value.request_budget_multiplier
                ),
            )
            if request_bundle_v5
            else replace(segment,cash_scale_cny=segment.cash_scale_cny*float(value.request_budget_multiplier))
            if model.request_budget_bounds is not None
            else segment
        )
        rental_online_penalty = (
            world.online_friction
            + model.rental_delivery_lateness_loss_rate
            * rental_online_late_probability
            * importance
        )
        rental_ownership_burden = (
            model.ownership_burden
            + model.rental_storage_burden_rate * storage
        )
        rental_future_use_credit = (
            model.future_use_credit
            + model.rental_future_use_credit_rate * future_use
        )
        rental_return_burden = model.return_burden * return_multiplier
        rental_store_service_value = (
            world.rental_convenience
            + model.rental_immediate_value_rate
            * rental_online_late_probability
            * importance
            + model.rental_date_guarantee_rate
            * math.log1p(lead)
            * importance
            + model.rental_flexibility_value_rate
            * uncertainty
            * trust
            * importance
        )
        rental_context = {
            "duration_days": float(duration),
            "booking_lead_days": float(lead),
            "forecast_adversity": adversity,
            "forecast_uncertainty": uncertainty,
            "forecast_trust": trust,
            "future_use_expectation": future_use,
            "storage_burden": storage,
            "return_burden_multiplier": return_multiplier,
            "online_late_probability": rental_online_late_probability,
            "shared_value_scale": rental_shared_scale,
            "duration_value_scale": duration_value,
            "quantity_value_scale": quantity_value,
            "effective_cash_scale_cny": cash_segment.cash_scale_cny,
            "request_budget_multiplier": float(
                value.request_budget_multiplier
            ),
            "party_equivalent": float(value.party_equivalent),
            "online_penalty": rental_online_penalty,
            "store_service_value": rental_store_service_value,
            "return_burden": rental_return_burden,
        }

    profile_info: dict[str, dict[str, Any]] = {}
    for profile_id, rows in sorted(by_profile.items()):
        online = min(
            rows,
            key=lambda row: (
                row.online_cash_price,
                row.online_unit_price,
                row.offer_id,
            ),
        )
        common, quality_term, substitute = _common_product_utility(
            online, value=value, segment=segment, policy=policy
        )
        online_functional_value = (need + common) * (
            rental_shared_scale if rental and contextual_rental else 1.0
        )
        online_cash_penalty = _cash_penalty(
            online.online_cash_price,
            world=world,
            segment=cash_segment,
        )
        profile_info[profile_id] = {
            "rows": rows,
            "online": online,
            "common": common,
            "quality_term": quality_term,
            "substitute": substitute,
            "online_functional_value": online_functional_value,
            "online_cash_penalty": online_cash_penalty,
            "online_systematic": (
                online_functional_value
                - online_cash_penalty
                - (
                    expected_lateness_loss
                    if deadline_aware_retail
                    else rental_online_penalty
                    if rental and contextual_rental
                    else world.online_friction
                )
                - (rental_ownership_burden if rental else 0.0)
                + (rental_future_use_credit if rental else 0.0)
            ),
        }

    store_items: list[dict[str, Any]] = []
    for profile_id, info in profile_info.items():
        for row in info["rows"]:
            if (
                row.store_cash_price is None
                or row.store_unit_price is None
                or row.available_units <= 0
            ):
                continue
            requested_resources = max(1, int(row.requested_units))
            store_items.append(
                {
                    "row": row,
                    "profile_id": profile_id,
                    "common": float(info["common"]),
                    "per_resource_cash": float(row.store_cash_price)
                    / requested_resources,
                    "per_resource_deposit": float(row.store_deposit)
                    / requested_resources,
                    "units_per_resource": max(1, int(row.units_per_resource)),
                }
            )

    def plan_for_lead(profile_id: str) -> dict[str, Any] | None:
        if not any(item["profile_id"] == profile_id for item in store_items):
            return None
        ordered = sorted(
            store_items,
            key=lambda item: (
                item["profile_id"] != profile_id,
                -float(item["common"]),
                float(item["per_resource_cash"])
                + model.deposit_friction_rate
                * float(item["per_resource_deposit"]),
                item["row"].offer_id,
            ),
        )
        remaining = target_units
        quote_plan: list[dict[str, Any]] = []
        common_total = 0.0
        for item in ordered:
            row = item["row"]
            units_per_resource = int(item["units_per_resource"])
            resources_needed = math.ceil(remaining / units_per_resource)
            resource_units = min(int(row.available_units), resources_needed)
            if resource_units <= 0:
                continue
            covered_units = min(remaining, resource_units * units_per_resource)
            quote_plan.append(
                {
                    "offer_id": row.offer_id,
                    "profile_id": item["profile_id"],
                    "resource_units": resource_units,
                    "covered_units": covered_units,
                    "cash_price": resource_units
                    * float(item["per_resource_cash"]),
                    "deposit": resource_units
                    * float(item["per_resource_deposit"]),
                    "store_unit_price": row.store_unit_price,
                    "online_unit_price": row.online_unit_price,
                }
            )
            common_total += covered_units * float(item["common"])
            remaining -= covered_units
            if remaining <= 0:
                break
        covered = target_units - remaining
        if covered <= 0 or (value.intent_role == "required" and remaining > 0):
            return None
        fraction = covered / target_units
        aggregate_cash = sum(float(line["cash_price"]) for line in quote_plan)
        aggregate_deposit = sum(float(line["deposit"]) for line in quote_plan)
        weighted_common = common_total / covered
        markup_penalty: float | None = None
        markup_reference_weight: float | None = None
        rental_relative_term: float | None = None
        benefit = fraction * (need + weighted_common)
        if rental and contextual_rental:
            benefit *= rental_shared_scale
        if rental:
            generalized = (
                aggregate_cash
                + model.deposit_friction_rate * aggregate_deposit
            )
            online = profile_info[profile_id]["online"]
            raw_log_ratio = math.log(
                max(0.01, generalized)
                / max(0.01, float(online.online_cash_price))
            )
            clipped = (
                raw_log_ratio
                if contextual_rental
                else min(
                    model.rental_log_ratio_clip[1],
                    max(model.rental_log_ratio_clip[0], raw_log_ratio),
                )
            )
            rental_relative_term = (
                -world.rental_relative_weight
                * segment.rental_weight_multiplier
                * clipped
            )
            store_cash_penalty = _cash_penalty(
                generalized, world=world, segment=cash_segment
            )
            store_cash_salience = (
                model.rental_request_cash_salience
                if request_bundle_v5
                else model.rental_contract_cash_salience
                if contextual_rental
                else 1.0
            )
            systematic = (
                benefit
                - store_cash_penalty * store_cash_salience
                - model.rental_budget_tail_coefficient*max(0.,generalized/cash_segment.cash_scale_cny-2.)**2
                + fraction
                * (
                    rental_relative_term
                    + (
                        rental_store_service_value
                        if contextual_rental
                        else world.rental_convenience
                    )
                    - (
                        rental_return_burden
                        if contextual_rental
                        else model.return_burden
                    )
                )
            )
        else:
            store_cash_penalty = _cash_penalty(
                aggregate_cash, world=world, segment=segment
            )
            store_cash_salience = 1.0
            raw_markup_penalty = (
                world.retail_markup_weight
                * segment.markup_weight_multiplier
                * sum(
                    int(line["covered_units"])
                    / target_units
                    * max(
                        0.0,
                        math.log(
                            max(0.01, float(line["store_unit_price"]))
                            / max(0.01, float(line["online_unit_price"]))
                        ),
                    )
                    for line in quote_plan
                )
            )
            # The exact online unit price remains shared evidence about product
            # value even when its delivery would miss this occasion.  Deadline
            # surplus belongs in the explicit immediate/lateness terms above;
            # attenuating this anchor by P(late) gave unrelated routine retail
            # items an almost unbounded urgency markup.
            markup_reference_weight = 1.0
            markup_penalty = raw_markup_penalty * markup_reference_weight
            systematic = (
                benefit
                - _cash_penalty(aggregate_cash, world=world, segment=segment)
                - markup_penalty
                + fraction
                * (
                    immediate_fulfillment_value
                    if deadline_aware_retail
                    else world.retail_store_convenience
                )
            )
        return {
            "plan_id": f"store-plan:{profile_id}",
            "lead_profile_id": profile_id,
            "fulfilled_units": covered,
            "fulfillment_fraction": fraction,
            "store_quote_plan": quote_plan,
            "store_cash_price": aggregate_cash,
            "store_deposit": aggregate_deposit,
            "functional_value": benefit,
            "generalized_store_cash": (
                generalized if rental else aggregate_cash
            ),
            "store_cash_penalty": store_cash_penalty,
            **({"rental_tail_penalty": model.rental_budget_tail_coefficient*max(0.,generalized/cash_segment.cash_scale_cny-2.)**2}
               if rental and model.rental_budget_tail_coefficient else {}),
            "store_cash_salience": store_cash_salience,
            "store_service_value": (
                rental_store_service_value if rental else 0.0
            ),
            "return_burden": rental_return_burden if rental else 0.0,
            "store_unit_price": sum(
                int(line["covered_units"])
                * float(line["store_unit_price"])
                for line in quote_plan
            )
            / covered,
            "markup_penalty": markup_penalty,
            "markup_reference_weight": markup_reference_weight,
            "rental_relative_term": rental_relative_term,
            "immediate_fulfillment_value": (
                fraction * immediate_fulfillment_value
                if deadline_aware_retail
                else None
            ),
            "store_systematic": systematic,
        }

    profiles: list[dict[str, Any]] = []
    for profile_id, info in sorted(profile_info.items()):
        online = info["online"]
        plan = plan_for_lead(profile_id)
        store_systematic = (
            float("-inf") if plan is None else float(plan["store_systematic"])
        )
        profiles.append(
            {
                "profile_id": profile_id,
                "offer_ids": sorted(row.offer_id for row in info["rows"]),
                "choice_group": online.choice_group,
                "common_product_utility": info["common"],
                "quality_term": info["quality_term"],
                "substitution_penalty": info["substitute"],
                "online_representative_offer_id": online.offer_id,
                "online_cash_price": online.online_cash_price,
                "online_unit_price": online.online_unit_price,
                "online_functional_value": info[
                    "online_functional_value"
                ],
                "online_cash_penalty": info["online_cash_penalty"],
                "online_systematic": info["online_systematic"],
                "online_late_probability": (
                    online_late_probability if deadline_aware_retail else None
                ),
                "expected_lateness_loss": (
                    expected_lateness_loss if deadline_aware_retail else None
                ),
                "store_plan_id": None if plan is None else plan["plan_id"],
                "store_representative_offer_id": (
                    None
                    if plan is None
                    else plan["store_quote_plan"][0]["offer_id"]
                ),
                "store_quote_plan": (
                    [] if plan is None else plan["store_quote_plan"]
                ),
                "store_fulfilled_units": (
                    0 if plan is None else plan["fulfilled_units"]
                ),
                "store_fulfillment_fraction": (
                    0.0 if plan is None else plan["fulfillment_fraction"]
                ),
                "store_cash_price": (
                    None if plan is None else plan["store_cash_price"]
                ),
                "store_deposit": (
                    None if plan is None else plan["store_deposit"]
                ),
                "store_functional_value": (
                    None if plan is None else plan["functional_value"]
                ),
                "generalized_store_cash": (
                    None if plan is None else plan["generalized_store_cash"]
                ),
                "store_cash_penalty": (
                    None if plan is None else plan["store_cash_penalty"]
                ),
                "store_cash_salience": (
                    None if plan is None else plan["store_cash_salience"]
                ),
                **({"rental_tail_penalty":0. if plan is None else plan.get("rental_tail_penalty",0.)}
                   if rental and model.rental_budget_tail_coefficient else {}),
                "store_service_value": (
                    None if plan is None else plan["store_service_value"]
                ),
                "return_burden": (
                    None if plan is None else plan["return_burden"]
                ),
                "store_unit_price": (
                    None if plan is None else plan["store_unit_price"]
                ),
                "markup_penalty": (
                    None if plan is None else plan["markup_penalty"]
                ),
                "markup_reference_weight": (
                    None if plan is None else plan["markup_reference_weight"]
                ),
                "immediate_fulfillment_value": (
                    None if plan is None else plan["immediate_fulfillment_value"]
                ),
                "rental_relative_term": (
                    None if plan is None else plan["rental_relative_term"]
                ),
                "store_systematic": store_systematic,
                "store_choice_score": (
                    store_systematic
                    + model.profile_shock_scale
                    * float(value.profile_gumbels.get(profile_id, 0.0))
                ),
            }
        )
    best_store = _best(profiles, "store_systematic")
    if best_store is not None and not math.isfinite(float(best_store["store_systematic"])):
        best_store = None
    best_online = _best(profiles, "online_systematic")
    systematic = {
        "store": float("-inf") if best_store is None else float(best_store["store_systematic"]),
        "online": float("-inf") if best_online is None else float(best_online["online_systematic"]),
        "existing": (
            need
            * (
                rental_shared_scale
                if rental and request_bundle_v5
                else 1.0
            )
            + model.existing_option_bonus
            if value.existing_option_available
            else float("-inf")
        ),
        "abandon": 0.0,
    }
    if contextual_rental:
        tastes = value.channel_tastes or {}
        scores = {
            channel: score
            + model.channel_taste_scale
            * min(1.0, max(-1.0, float(tastes.get(channel, 0.0))))
            for channel, score in systematic.items()
        }
    else:
        scores = {
            channel: score
            + float(value.channel_affinities.get(channel, 0.0))
            + model.channel_shock_scale
            * float(value.channel_gumbels.get(channel, 0.0))
            for channel, score in systematic.items()
        }
    selected_channel = max(scores, key=lambda key: (scores[key], key))
    selected_profile: dict[str, Any] | None = None
    selected_offer_id: str | None = None
    if selected_channel == "store":
        eligible = [row for row in profiles if math.isfinite(float(row["store_systematic"]))]
        selected_profile = _best(eligible, "store_choice_score")
        selected_offer_id = (
            None
            if selected_profile is None
            else str(selected_profile["store_representative_offer_id"])
        )
    elif selected_channel == "online":
        selected_profile = best_online
        selected_offer_id = (
            None
            if selected_profile is None
            else str(selected_profile["online_representative_offer_id"])
        )
    return ChannelDecisionTrace(
        format=CHANNEL_TRACE_FORMAT,
        world_id=world.world_id,
        service_key=value.service_key,
        preferred_group=value.preferred_group,
        customer_segment=value.customer_segment,
        intent_role=value.intent_role,
        profiles=tuple(profiles),
        best_store_profile_id=(None if best_store is None else str(best_store["profile_id"])),
        best_store_plan_id=(
            None if best_store is None else str(best_store["store_plan_id"])
        ),
        best_online_profile_id=(None if best_online is None else str(best_online["profile_id"])),
        channel_systematic=systematic,
        channel_scores=scores,
        selected_channel=selected_channel,
        selected_profile_id=(
            None if selected_profile is None else str(selected_profile["profile_id"])
        ),
        selected_store_plan_id=(
            None
            if selected_channel != "store" or selected_profile is None
            else str(selected_profile["store_plan_id"])
        ),
        selected_offer_id=selected_offer_id,
        need_horizon_days=(
            value.need_horizon_days if deadline_aware_retail else None
        ),
        occasion_importance=(
            float(value.occasion_importance) if deadline_aware_retail else 1.0
        ),
        online_late_probability=online_late_probability,
        rental_context=rental_context,
    )


def default_channel_model_payload(
    group_policies: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format": CHANNEL_DEMAND_FORMAT,
        "group_policies": [
            {
                "service_key": str(row["service_key"]),
                "choice_group": str(row["choice_group"]),
                "quality_salience": str(row["quality_salience"]),
                "substitution_strength": str(row["substitution_strength"]),
            }
            for row in group_policies
        ],
        "segments": {
            key: asdict(row) for key, row in DEFAULT_CHANNEL_SEGMENTS.items()
        },
        "worlds": {
            key: asdict(row) for key, row in DEFAULT_CHANNEL_WORLDS.items()
        },
        "cash_scale_unit": "CNY per complete customer request",
        "required_need_intercept": 6.9,
        "optional_need_intercept": 4.4,
        "deposit_friction_rate": 0.05,
        "return_burden": 1.6,
        "ownership_burden": 0.1,
        "future_use_credit": 0.0,
        "existing_option_bonus": 0.75,
        "channel_shock_scale": 0.6,
        "profile_shock_scale": 0.25,
        "deadline_value_scale": 1.0,
        "retail_immediate_value_rate": 0.35,
        "retail_unmet_need_loss_rate": 1.0,
        "rental_log_ratio_clip": [-2.302585092994046, 0.6931471805599453],
        "rental_duration_value_rate": 0.45,
        "rental_weather_disutility_rate": 0.55,
        "rental_delivery_lateness_loss_rate": 0.08,
        "rental_immediate_value_rate": 0.08,
        "rental_date_guarantee_rate": 0.12,
        "rental_flexibility_value_rate": 0.30,
        "rental_future_use_credit_rate": 0.80,
        "rental_storage_burden_rate": 0.65,
        "channel_taste_scale": 0.55,
    }


def request_bundle_channel_model_payload(
    base: ChannelDemandModel,
    *,
    rental_request_cash_salience: float = 3.80,
) -> dict[str, Any]:
    """Project a frozen V4 model into the V5 request-bundle parameterization."""

    if base.format != CHANNEL_DEMAND_FORMAT:
        raise ValueError("request-bundle projection requires the V4 channel model")
    payload = base.to_dict()
    payload.update(
        {
            "format": REQUEST_BUNDLE_CHANNEL_DEMAND_FORMAT,
            "rental_quantity_value_exponent": 0.80,
            "rental_duration_value_exponent": 0.70,
            "rental_party_budget_exponent": 0.50,
            "rental_duration_budget_exponent": 1.00,
            "request_budget_sigma": 0.25,
            "rental_request_cash_salience": float(
                rental_request_cash_salience
            ),
        }
    )
    ChannelDemandModel.from_dict(payload)
    return payload


def online_rationality_envelope(
    model: ChannelDemandModel,
    world_id: str,
) -> dict[str, Any]:
    """Audit whether the frozen online price is a broad seller-profit optimum.

    This is a diagnostic only.  It never enters a customer's production utility.
    """

    world = model.worlds[world_id]
    rows = []
    grid = tuple(0.5 + 0.005 * index for index in range(301))
    for margin in (0.25, 0.40, 0.60):
        unit_cost = 1.0 - margin
        for reference_conversion in (0.20, 0.40, 0.60):
            intercept = math.log(
                reference_conversion / (1.0 - reference_conversion)
            )

            def quantity(price_multiple: float) -> float:
                logit = intercept - world.common_price_beta * math.log(
                    price_multiple
                )
                return 1.0 / (1.0 + math.exp(-logit))

            optimum = max(
                grid,
                key=lambda multiple: (
                    (multiple - unit_cost) * quantity(multiple),
                    -multiple,
                ),
            )
            rows.append(
                {
                    "gross_margin_at_observed_price": margin,
                    "conversion_at_observed_price": reference_conversion,
                    "profit_optimal_price_multiple": round(optimum, 6),
                }
            )
    broad = sum(0.5 <= row["profit_optimal_price_multiple"] <= 2.0 for row in rows)
    central = sum(0.75 <= row["profit_optimal_price_multiple"] <= 1.5 for row in rows)
    return {
        "world_id": world_id,
        "common_price_beta": world.common_price_beta,
        "assumptions": rows,
        "broad_envelope_count": broad,
        "central_envelope_count": central,
        "passed": broad == 9 and central >= 7,
    }
