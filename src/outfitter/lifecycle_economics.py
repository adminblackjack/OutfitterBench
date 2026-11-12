"""Pure lifecycle economics used to validate the V9 K/A/D/V closure.

The constants below are explicitly candidate benchmark scales.  They are not
real-world estimates and are not loaded by the released runtime until a later
projection stage accepts them through scripted multi-world calibration.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


LIFECYCLE_SCALES_FORMAT = "outfitter-v9-lifecycle-economic-scales-v1"
TIERS = ("low", "medium", "high")


@dataclass(frozen=True, slots=True)
class LifecycleEconomicScales:
    status: str = "candidate_requires_joint_economic_calibration"
    claim_boundary: str = (
        "benchmark free scales; not verified real-world rental-industry parameters"
    )
    cleaning_cash_per_work_point: float = 0.50
    cleaning_extra_work_fraction_per_additional_day: float = 0.10
    damage_daily_rate_by_hazard_tier: dict[str, float] = field(
        default_factory=lambda: {"low": 0.004, "medium": 0.010, "high": 0.020}
    )
    damage_condition_loss_range_by_severity_tier: dict[
        str, tuple[float, float]
    ] = field(
        default_factory=lambda: {
            "low": (0.03, 0.12),
            "medium": (0.08, 0.25),
            "high": (0.15, 0.45),
        }
    )
    damage_wear_hazard_slope: float = 0.75
    damage_condition_hazard_slope: float = 0.75
    repair_cash_per_replacement_condition_loss: float = 1.10
    repair_minimum_cash: float = 5.00
    repair_condition_restore_fraction: float = 0.80
    customer_damage_liability_share: float = 0.75
    customer_damage_minimum_deduction: float = 2.00
    maintenance_interval_wear: float = 0.12
    maintenance_due_window_wear: float = 0.03
    maintenance_hazard_reduction: float = 0.35
    maintenance_cash_rate_by_severity_tier: dict[str, float] = field(
        default_factory=lambda: {"low": 0.008, "medium": 0.012, "high": 0.018}
    )
    maintenance_minimum_cash: float = 2.00
    maintenance_duration_days_by_severity_tier: dict[str, int] = field(
        default_factory=lambda: {"low": 1, "medium": 1, "high": 2}
    )
    annual_capital_rate: float = 0.12
    daily_space_rate_per_storage_unit: float = 0.03
    new_asset_residual_rate: float = 0.55
    residual_wear_exponent: float = 0.65

    def to_dict(self) -> dict[str, Any]:
        return {"format": LIFECYCLE_SCALES_FORMAT, **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LifecycleEconomicScales":
        data = dict(value)
        runtime_format = data.pop("format", LIFECYCLE_SCALES_FORMAT)
        if runtime_format != LIFECYCLE_SCALES_FORMAT:
            raise ValueError("unsupported lifecycle economic scales format")
        return cls(**data)


def _finite_nonnegative(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _positive(value: float, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _tier(value: str, *, name: str) -> str:
    result = str(value)
    if result not in TIERS:
        raise ValueError(f"{name} must be low, medium, or high")
    return result


def combined_wear_fraction(
    *,
    used_rental_days: float,
    completed_rental_cycles: float,
    service_life_limit_days: float,
    service_life_limit_cycles: float,
) -> float:
    """Return additive wear without conflating it with physical condition."""
    days = _finite_nonnegative(used_rental_days, name="used_rental_days")
    cycles = _finite_nonnegative(
        completed_rental_cycles, name="completed_rental_cycles"
    )
    day_limit = _positive(service_life_limit_days, name="service_life_limit_days")
    cycle_limit = _positive(
        service_life_limit_cycles, name="service_life_limit_cycles"
    )
    return days / day_limit + cycles / cycle_limit


def cleaning_terms(
    *,
    one_day_work_points: float,
    duration_days: int,
    return_weather_multiplier: float = 1.0,
    world_work_multiplier: float = 1.0,
    scales: LifecycleEconomicScales = LifecycleEconomicScales(),
) -> dict[str, float]:
    """Derive queue work and cash from one common cleaning quantity."""
    base = _positive(one_day_work_points, name="one_day_work_points")
    if isinstance(duration_days, bool) or int(duration_days) < 1:
        raise ValueError("duration_days must be at least one")
    weather = _positive(
        return_weather_multiplier, name="return_weather_multiplier"
    )
    world = _positive(world_work_multiplier, name="world_work_multiplier")
    work = base * (
        1.0
        + scales.cleaning_extra_work_fraction_per_additional_day
        * (int(duration_days) - 1)
    )
    work *= weather * world
    cash = work * scales.cleaning_cash_per_work_point
    return {
        "work_points": round(work, 6),
        "cash_cost": round(cash, 6),
        "cash_per_work_point": scales.cleaning_cash_per_work_point,
    }


def maintenance_protection(
    *,
    wear_since_maintenance: float,
    scales: LifecycleEconomicScales = LifecycleEconomicScales(),
) -> float:
    """Protection is strongest after maintenance and decays with new wear."""
    exposure = _finite_nonnegative(
        wear_since_maintenance, name="wear_since_maintenance"
    )
    remaining = max(
        0.0, 1.0 - exposure / _positive(
            scales.maintenance_interval_wear, name="maintenance_interval_wear"
        )
    )
    return min(1.0, scales.maintenance_hazard_reduction * remaining)


def maintenance_status(
    *,
    combined_wear: float,
    maintenance_wear_anchor: float,
    condition_ratio: float,
    scales: LifecycleEconomicScales = LifecycleEconomicScales(),
) -> dict[str, Any]:
    wear = _finite_nonnegative(combined_wear, name="combined_wear")
    anchor = _finite_nonnegative(
        maintenance_wear_anchor, name="maintenance_wear_anchor"
    )
    condition = _finite_nonnegative(condition_ratio, name="condition_ratio")
    if condition > 1.0 + 1e-12:
        raise ValueError("condition_ratio cannot exceed one")
    exposure = max(0.0, wear - min(wear, anchor))
    due_at = anchor + scales.maintenance_interval_wear
    overdue_at = due_at + scales.maintenance_due_window_wear
    # Severe condition can open maintenance early, while exposure still remains
    # the ordinary trigger.  Exact thresholds are candidate benchmark scales.
    window_open = wear + 1e-12 >= due_at or condition < 0.80
    overdue = wear > overdue_at + 1e-12 or condition < 0.65
    safety_hold = condition < 0.65 and exposure > 1e-12
    return {
        "wear_since_maintenance": round(exposure, 8),
        "maintenance_due_wear": round(due_at, 8),
        "maintenance_overdue_wear": round(overdue_at, 8),
        "window_open": window_open,
        "overdue": overdue,
        # Exposure overdue is an economic risk signal, not an automatic forced
        # maintenance action. Only unsafe condition requires a capacity hold.
        # Maintenance can certify a low-condition asset at the current wear
        # anchor without pretending to restore its resale condition. Any new
        # wear reopens the safety hold.
        "safety_hold_required": safety_hold,
        "hazard_protection": round(
            maintenance_protection(
                wear_since_maintenance=exposure, scales=scales
            ),
            8,
        ),
    }


def damage_event_probability(
    *,
    hazard_tier: str,
    duration_days: int,
    combined_wear: float,
    condition_ratio: float,
    wear_since_maintenance: float,
    world_hazard_multiplier: float = 1.0,
    daily_hazard_multipliers: list[float] | tuple[float, ...] | None = None,
    scales: LifecycleEconomicScales = LifecycleEconomicScales(),
) -> float:
    hazard = _tier(hazard_tier, name="hazard_tier")
    if isinstance(duration_days, bool) or int(duration_days) < 1:
        raise ValueError("duration_days must be at least one")
    wear = _finite_nonnegative(combined_wear, name="combined_wear")
    condition = _finite_nonnegative(condition_ratio, name="condition_ratio")
    if condition > 1.0 + 1e-12:
        raise ValueError("condition_ratio cannot exceed one")
    world = _positive(world_hazard_multiplier, name="world_hazard_multiplier")
    base = scales.damage_daily_rate_by_hazard_tier[hazard]
    exposure_multiplier = (
        1.0
        + scales.damage_wear_hazard_slope * min(1.0, wear)
        + scales.damage_condition_hazard_slope * (1.0 - condition)
    )
    protection = maintenance_protection(
        wear_since_maintenance=wear_since_maintenance, scales=scales
    )
    daily = max(0.0, base * world * exposure_multiplier * (1.0 - protection))
    if daily_hazard_multipliers is None:
        multipliers = (1.0,) * int(duration_days)
    else:
        multipliers = tuple(
            _positive(value, name="daily_hazard_multiplier")
            for value in daily_hazard_multipliers
        )
        if len(multipliers) != int(duration_days):
            raise ValueError(
                "daily_hazard_multipliers must match duration_days"
            )
    survival = 1.0
    for multiplier in multipliers:
        survival *= 1.0 - min(0.95, daily * multiplier)
    return 1.0 - survival


def condition_loss_from_severity_uniform(
    *,
    severity_tier: str,
    severity_uniform: float,
    world_severity_multiplier: float = 1.0,
    scales: LifecycleEconomicScales = LifecycleEconomicScales(),
) -> float:
    tier = _tier(severity_tier, name="severity_tier")
    uniform = float(severity_uniform)
    if not math.isfinite(uniform) or not 0.0 <= uniform < 1.0:
        raise ValueError("severity_uniform must be in [0, 1)")
    world = _positive(world_severity_multiplier, name="world_severity_multiplier")
    low, high = scales.damage_condition_loss_range_by_severity_tier[tier]
    return min(0.95, (low + (high - low) * uniform) * world)


def damage_terms(
    *,
    replacement_cost: float,
    condition_ratio_before: float,
    security_deposit: float,
    condition_loss: float,
    scales: LifecycleEconomicScales = LifecycleEconomicScales(),
) -> dict[str, float]:
    """Use one severity-derived condition loss for every damage consequence."""
    replacement = _positive(replacement_cost, name="replacement_cost")
    condition = _finite_nonnegative(
        condition_ratio_before, name="condition_ratio_before"
    )
    if condition > 1.0 + 1e-12:
        raise ValueError("condition_ratio_before cannot exceed one")
    deposit = _finite_nonnegative(security_deposit, name="security_deposit")
    loss = min(condition, _finite_nonnegative(condition_loss, name="condition_loss"))
    condition_after_damage = max(0.0, condition - loss)
    repair_cash = max(
        scales.repair_minimum_cash,
        replacement * loss * scales.repair_cash_per_replacement_condition_loss,
    )
    deduction = min(
        deposit,
        max(
            scales.customer_damage_minimum_deduction,
            repair_cash * scales.customer_damage_liability_share,
        ),
    )
    restored = loss * scales.repair_condition_restore_fraction
    condition_after_repair = min(condition, condition_after_damage + restored)
    return {
        "condition_loss": round(loss, 6),
        "condition_ratio_after_damage": round(condition_after_damage, 6),
        "repair_cash_cost": round(repair_cash, 6),
        "deposit_deduction": round(deduction, 6),
        "uncovered_repair_cost": round(max(0.0, repair_cash - deduction), 6),
        "condition_restored_by_repair": round(restored, 6),
        "condition_ratio_after_repair": round(condition_after_repair, 6),
    }


def maintenance_terms(
    *,
    replacement_cost: float,
    severity_tier: str,
    scales: LifecycleEconomicScales = LifecycleEconomicScales(),
) -> dict[str, float | int]:
    replacement = _positive(replacement_cost, name="replacement_cost")
    tier = _tier(severity_tier, name="severity_tier")
    cost = max(
        scales.maintenance_minimum_cash,
        replacement * scales.maintenance_cash_rate_by_severity_tier[tier],
    )
    return {
        "cash_cost": round(cost, 6),
        "duration_days": int(
            scales.maintenance_duration_days_by_severity_tier[tier]
        ),
    }


def holding_terms(
    *,
    replacement_cost: float,
    storage_burden_units: float,
    stored_on_site: bool,
    world_storage_multiplier: float = 1.0,
    scales: LifecycleEconomicScales = LifecycleEconomicScales(),
) -> dict[str, float]:
    replacement = _positive(replacement_cost, name="replacement_cost")
    storage = _finite_nonnegative(
        storage_burden_units, name="storage_burden_units"
    )
    world = _positive(world_storage_multiplier, name="world_storage_multiplier")
    capital = replacement * scales.annual_capital_rate / 365.0
    space = (
        storage * world * scales.daily_space_rate_per_storage_unit
        if stored_on_site
        else 0.0
    )
    return {
        "capital_charge": round(capital, 6),
        "space_charge": round(space, 6),
        "total_charge": round(capital + space, 6),
    }


def nominal_residual_value(
    *,
    replacement_cost: float,
    condition_ratio: float,
    combined_wear: float,
    scales: LifecycleEconomicScales = LifecycleEconomicScales(),
) -> float:
    """Return pre-depth residual value using no demand, price, or payback input."""
    replacement = _positive(replacement_cost, name="replacement_cost")
    condition = _finite_nonnegative(condition_ratio, name="condition_ratio")
    if condition > 1.0 + 1e-12:
        raise ValueError("condition_ratio cannot exceed one")
    wear = _finite_nonnegative(combined_wear, name="combined_wear")
    remaining = max(0.0, 1.0 - wear)
    return round(
        replacement
        * scales.new_asset_residual_rate
        * condition
        * remaining**scales.residual_wear_exponent,
        6,
    )
