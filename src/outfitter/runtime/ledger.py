from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class DemandCounters(TypedDict):
    requested_units: int
    fulfilled_units: int
    unfulfilled_units: int
    requested_line_items: int
    fulfilled_line_items: int
    partial_line_items: int
    unfilled_line_items: int
    outside_rejections: int
    stockout_rejections: int
    service_closed_rejections: int
    capacity_policy_rejections: int
    horizon_rejections: int
    unpriced_rejections: int
    declined_after_quote_units: int
    no_quote_units: int
    preferred_not_carried_units: int
    preferred_stocked_out_units: int
    preferred_fulfilled_units: int
    substitute_fulfilled_units: int
    full_baskets: int
    partial_baskets: int
    unfilled_baskets: int
    operationally_evaluable_baskets: int
    operational_full_baskets: int
    operational_partial_baskets: int
    operational_unfilled_baskets: int


class DailyLedger(TypedDict):
    day: int
    morning: dict[str, float]
    flow_timing_version: NotRequired[int]
    opening_complete: NotRequired[bool]
    intraday: NotRequired[dict[str, float]]
    actions: dict[str, float | int]
    agent_actions: dict[str, Any]
    handoff_note: str | None
    demand: DemandCounters
    service_metrics: dict[str, dict[str, Any]]
    delivery_metrics: NotRequired[dict[str, dict[str, Any]]]
    counter_interactions: list[dict[str, Any]]
    demand_cube: list[dict[str, Any]]
    offer_snapshots: list[dict[str, Any]]
    internal_channel_outcomes: list[dict[str, Any]]
    internal_basket_intents: list[dict[str, Any]]
    arrivals: dict[str, int]
    rental_revenue: float
    retail_revenue: float
    deposits_collected: float
    reservation_prepayments_collected: float
    reservation_prepayment_refunds: float
    reservation_prepayment_recognized: float
    reservation_compensation: float
    retail_holding_cost: float
    rental_holding_cost: float
    rental_idle_holding_cost: float
    opening_rental_holding_cost_basis: float
    opening_rental_idle_holding_cost_basis: float
    operating_fee: float


def empty_agent_actions() -> dict[str, Any]:
    return {
        "price_changes": {},
        "purchase_orders": [],
        "rental_product_status_changes": {},
        "repairs": [],
        "maintenance": [],
        "retirements": [],
        "asset_sales": [],
        "retail_liquidations": [],
    }


def empty_ledger(day: int) -> DailyLedger:
    return {
        "day": day,
        "morning": {
            "refunds": 0.0,
            "damage_deductions": 0.0,
            "cleaning_cost": 0.0,
        },
        "actions": {
            "purchase_cost": 0.0,
            "repair_cost": 0.0,
            "maintenance_cost": 0.0,
            "model_cost": 0.0,
            "asset_sale_proceeds": 0.0,
            "assets_sold": 0,
            "retail_liquidation_proceeds": 0.0,
            "retail_liquidation_book_cost": 0.0,
            "retail_liquidation_loss": 0.0,
            "retail_units_liquidated": 0,
            "liquidation_nominal_proceeds": 0.0,
            "liquidation_depth_discount": 0.0,
        },
        "agent_actions": empty_agent_actions(),
        "handoff_note": None,
        "demand": {
            "requested_units": 0,
            "fulfilled_units": 0,
            "unfulfilled_units": 0,
            "requested_line_items": 0,
            "fulfilled_line_items": 0,
            "partial_line_items": 0,
            "unfilled_line_items": 0,
            "outside_rejections": 0,
            "stockout_rejections": 0,
            "service_closed_rejections": 0,
            "horizon_rejections": 0,
            "unpriced_rejections": 0,
            "declined_after_quote_units": 0,
            "no_quote_units": 0,
            "preferred_not_carried_units": 0,
            "preferred_stocked_out_units": 0,
            "preferred_fulfilled_units": 0,
            "substitute_fulfilled_units": 0,
            "full_baskets": 0,
            "partial_baskets": 0,
            "unfilled_baskets": 0,
            "operationally_evaluable_baskets": 0,
            "operational_full_baskets": 0,
            "operational_partial_baskets": 0,
            "operational_unfilled_baskets": 0,
        },
        "service_metrics": {},
        "counter_interactions": [],
        "demand_cube": [],
        "offer_snapshots": [],
        # Research-only traces are persisted but never included in public views.
        "internal_channel_outcomes": [],
        "internal_basket_intents": [],
        "arrivals": {},
        "rental_revenue": 0.0,
        "retail_revenue": 0.0,
        "deposits_collected": 0.0,
        "reservation_prepayments_collected": 0.0,
        "reservation_prepayment_refunds": 0.0,
        "reservation_prepayment_recognized": 0.0,
        "reservation_compensation": 0.0,
        "retail_holding_cost": 0.0,
        "rental_holding_cost": 0.0,
        "rental_idle_holding_cost": 0.0,
        "opening_rental_holding_cost_basis": 0.0,
        "opening_rental_idle_holding_cost_basis": 0.0,
        "operating_fee": 0.0,
    }


# Compatibility aliases used while persisted WorldState keeps its dict wire shape.
_empty_agent_actions = empty_agent_actions
_empty_ledger = empty_ledger
