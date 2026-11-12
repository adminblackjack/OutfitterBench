"""Cheap deterministic calibration for OutfitterBench runtime data.

This module deliberately does not call a language model.  It exercises the
same purchasing, delivery, demand, accounting, and terminal valuation paths as
an evaluated run with a small base-stock policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import OutfitterConfig
from .data import OutfitterDataset
from .environment import SECONDHAND_GRADE_SCHEDULE, OutfitterEnvironment
from .models import ToolResult
from .service_commitments import ENVIRONMENT_PROTOCOL as V32_SERVICE_PROTOCOL


FORMAL_STRATEGIES = ("naive", "heuristic", "informed_public")
ADVERSARIAL_STRATEGIES = (
    "fixed_public",
    "scale_winner",
    "buy_liquidate",
    "reject_all",
)
AGGRESSIVE_EXPANSION_REFERENCE = "aggressive_expansion_reference"
LEGACY_REFERENCE_ALIASES = {"oracle_reference", "informed_reference"}
ALL_STRATEGIES = (
    *FORMAL_STRATEGIES,
    *ADVERSARIAL_STRATEGIES,
    AGGRESSIVE_EXPANSION_REFERENCE,
)
_PROFILE_SCRIPTED_DIAGNOSTICS: dict[int, Counter[str]] = {}
_PROFILE_SCRIPTED_MISSED_COMMITMENTS: dict[int, set[str]] = {}
PROFILE_SCRIPTED_QUOTE_SCALE = 1.0
PROFILE_SCRIPTED_OPERATIONAL_SUBSET_MODE = "none"
PROFILE_SCRIPTED_LEAD_RISK_PREMIUM = 0.0
PAIRWISE_SCRIPTED_MENU_POLICY = "single"
PUBLIC_DURABLE_PLANNING_DAYS = 56
INFORMED_EXPERIMENT_INTERVAL_DAYS = 28
INFORMED_DURABLE_COOLDOWN_DAYS = 28
INFORMED_MAX_NET_EXPANSION_ASSETS = 3
AGGRESSIVE_EXPANSION_TARGET_PER_GROUP = 3
RETAIL_REORDER_QUOTE_ROUNDING_TOLERANCE_PER_UNIT = 0.005
BASELINE_POLICY_CONTRACTS = {
    "naive": (
        "post a simple low cost-plus price on Day 1; restore opening retail base "
        "stock; repair damaged assets; no seasonal, channel, or capital learning"
    ),
    "heuristic": (
        "post a moderate public-cost price on Day 1; 14-day local conversion "
        "response; recent-sales retail reorder; no durable capacity expansion"
    ),
    "informed_public": (
        "post a stronger public-economics price on Day 1; use 7-day observable "
        "quote response during the first month, weather-aware reorder, controlled "
        "experiments, cent-quantization-stable retail reorder, and public NPV/payback "
        "with at most three net expansion assets"
    ),
    "scale_winner": (
        "uncapped public-information stress policy: repeatedly raise prices with "
        "strong recent conversion and add capacity behind sustained utilization; "
        "cash, supplier lead time, and observed economics are the only expansion caps"
    ),
    "fixed_public": (
        "fixed-price control: post the informed public-economics opening prices, "
        "never update them, and use conservative raw-calendar inventory decisions"
    ),
    "buy_liquidate": (
        "adversarial churn policy: repeatedly buy the cheapest deliverable durable "
        "package and sell each acquired asset to the liquidation dealer immediately"
    ),
    "reject_all": (
        "do-nothing stress control: post prohibitive prices, buy nothing, repair "
        "nothing, and reject all economically responsive demand"
    ),
    "aggressive_expansion_reference": (
        "aggressive hidden-parameter expansion reference; not an oracle or profit "
        "upper bound, and excluded from formal gates and liquidity recommendations"
    ),
}


def scripted_economic_gate(
    rows: list[dict[str, Any]],
    *,
    required_seed_count: int | None = None,
) -> dict[str, Any]:
    """Check survival, policy ordering, and a bounded-loss difficulty guard."""

    formal = [row for row in rows if row.get("strategy") in FORMAL_STRATEGIES]
    world_ids = sorted(
        {str(row.get("demand_world_id") or "base") for row in formal}
    )
    cells: dict[str, dict[str, Any]] = {}
    for world_id in world_ids:
        world_rows = [
            row
            for row in formal
            if str(row.get("demand_world_id") or "base") == world_id
        ]
        by_strategy = {
            strategy: [
                row for row in world_rows if row["strategy"] == strategy
            ]
            for strategy in FORMAL_STRATEGIES
        }
        common_seeds = sorted(
            set.intersection(
                *(
                    {int(row["seed"]) for row in by_strategy[strategy]}
                    for strategy in FORMAL_STRATEGIES
                )
            )
        ) if all(by_strategy.values()) else []
        medians = {
            strategy: round(
                statistics.median(
                    float(row["value_change"])
                    for row in by_strategy[strategy]
                ),
                2,
            )
            for strategy in FORMAL_STRATEGIES
            if by_strategy[strategy]
        }
        bankrupt = {
            strategy: sum(bool(row["bankrupt"]) for row in by_strategy[strategy])
            for strategy in FORMAL_STRATEGIES
        }
        reject_rows = [
            row
            for row in rows
            if str(row.get("demand_world_id") or "base") == world_id
            and row.get("strategy") == "reject_all"
        ]
        reject_median = (
            round(
                statistics.median(
                    float(row["value_change"]) for row in reject_rows
                ),
                2,
            )
            if reject_rows
            else None
        )
        informed_daily_operating_costs = [
            float(row.get("cumulative_costs", {}).get("operating", 0.0))
            / int(row.get("days", 0))
            for row in by_strategy["informed_public"]
            if int(row.get("days", 0)) > 0
            and float(
                row.get("cumulative_costs", {}).get("operating", 0.0)
            )
            > 0
        ]
        ten_day_operating_allowance = (
            10.0 * statistics.median(informed_daily_operating_costs)
            if informed_daily_operating_costs
            else 0.0
        )
        reject_loss_allowance = (
            0.30 * abs(float(reject_median))
            if reject_median is not None and reject_median < 0
            else 0.0
        )
        opening_values = [
            float(row.get("initial_official_liquidation_value", 0.0))
            for row in by_strategy["informed_public"]
            if float(row.get("initial_official_liquidation_value", 0.0)) > 0
        ]
        opening_value_cap = (
            0.10 * statistics.median(opening_values)
            if opening_values
            else math.inf
        )
        bounded_loss_allowance = min(
            max(ten_day_operating_allowance, reject_loss_allowance),
            opening_value_cap,
        )
        if not math.isfinite(bounded_loss_allowance):
            bounded_loss_allowance = max(
                ten_day_operating_allowance, reject_loss_allowance
            )
        informed_near_break_even_floor = round(-bounded_loss_allowance, 2)
        naive_total = len(by_strategy["naive"])
        seed_protocol_pass = bool(common_seeds) and (
            required_seed_count is None
            or len(common_seeds) == required_seed_count
        )
        informed_expansions = [
            int(row.get("lifecycle", {}).get("maximum_net_active_expansion"))
            for row in by_strategy["informed_public"]
            if row.get("lifecycle", {}).get("maximum_net_active_expansion")
            is not None
        ]
        informed_expansion_contract_pass = (
            not informed_expansions
            or max(informed_expansions)
            <= INFORMED_MAX_NET_EXPANSION_ASSETS
        )
        all_bridge_differences = [
            abs(
                float(
                    row.get("official_value_bridge", {}).get(
                        "reconciliation_difference"
                    )
                )
            )
            for row in world_rows
            if row.get("official_value_bridge", {}).get(
                "reconciliation_difference"
            )
            is not None
        ]
        bridge_differences = [
            abs(
                float(
                    row.get("official_value_bridge", {}).get(
                        "reconciliation_difference"
                    )
                )
            )
            for row in world_rows
            if not bool(row.get("bankrupt"))
            and row.get("official_value_bridge", {}).get(
                "reconciliation_difference"
            )
            is not None
        ]
        bankrupt_bridge_runs = [
            {
                "strategy": str(row["strategy"]),
                "seed": int(row["seed"]),
                "absolute_difference": round(
                    abs(
                        float(
                            row.get("official_value_bridge", {}).get(
                                "reconciliation_difference", 0.0
                            )
                        )
                    ),
                    6,
                ),
                "bankruptcy_shortfall": round(
                    float(row.get("bankruptcy_shortfall", 0.0)), 2
                ),
            }
            for row in world_rows
            if bool(row.get("bankrupt"))
        ]
        accounting_bridge_pass = (
            not bridge_differences or max(bridge_differences) <= 0.01
        )
        median_role_pass = (
            set(medians) == set(FORMAL_STRATEGIES)
            and medians["naive"] < 0
            and medians["naive"]
            < medians["heuristic"]
            < medians["informed_public"]
            and medians["informed_public"]
            >= informed_near_break_even_floor
            and (
                reject_median is None
                or medians["informed_public"] > reject_median
            )
        )
        survival_pass = (
            naive_total > 0
            and bankrupt["heuristic"] == 0
            and bankrupt["informed_public"] == 0
        )
        cells[world_id] = {
            "evaluated_seeds": common_seeds,
            "median_value_change": medians,
            "reject_all_median_value_change": reject_median,
            "bankrupt_runs": bankrupt,
            "naive_bankruptcy_allowed_and_uncensored": True,
            "heuristic_and_informed_all_survive": bankrupt["heuristic"] == 0
            and bankrupt["informed_public"] == 0,
            "median_role_order_pass": median_role_pass,
            "informed_bounded_loss_pass": bool(medians)
            and medians.get("informed_public", -math.inf)
            >= informed_near_break_even_floor,
            "informed_beats_reject_all": (
                None
                if reject_median is None
                else medians.get("informed_public", -math.inf) > reject_median
            ),
            "informed_expansion_contract": {
                "evaluated": bool(informed_expansions),
                "maximum_allowed": INFORMED_MAX_NET_EXPANSION_ASSETS,
                "observed_by_run": informed_expansions,
                "passed": informed_expansion_contract_pass,
            },
            "official_value_bridge_contract": {
                "scope": "nonbankrupt formal runs",
                "evaluated": bool(bridge_differences),
                "evaluated_runs": len(bridge_differences),
                "excluded_bankrupt_runs": bankrupt_bridge_runs,
                "maximum_absolute_difference": (
                    None
                    if not bridge_differences
                    else round(max(bridge_differences), 6)
                ),
                "tolerance": 0.01,
                "passed": accounting_bridge_pass,
                "all_formal_runs_maximum_absolute_difference_diagnostic": (
                    None
                    if not all_bridge_differences
                    else round(max(all_bridge_differences), 6)
                ),
            },
            "informed_near_break_even_floor": (
                informed_near_break_even_floor
            ),
            "bounded_loss_allowance_components": {
                "ten_day_operating_cost": round(
                    ten_day_operating_allowance, 2
                ),
                "thirty_percent_of_reject_all_loss": round(
                    reject_loss_allowance, 2
                ),
                "ten_percent_of_opening_value_cap": (
                    None
                    if not opening_values
                    else round(opening_value_cap, 2)
                ),
                "allowed_median_loss": round(bounded_loss_allowance, 2),
            },
            "near_break_even_policy": (
                "informed median may be negative but must stay above the stricter "
                "bounded-loss floor: min(max(10 operating days, 30% of reject-all "
                "loss), 10% of opening official value)"
            ),
            "seed_protocol_pass": seed_protocol_pass,
            "passed": (
                seed_protocol_pass
                and median_role_pass
                and survival_pass
                and informed_expansion_contract_pass
                and accounting_bridge_pass
            ),
        }
    return {
        "format": (
            "outfitter-scripted-economic-gate-v6-accounting-policy-bounded-loss"
        ),
        "absolute_profit_caps": None,
        "bankrupt_runs_are_not_censored": True,
        "required_seed_count": required_seed_count,
        "worlds": cells,
        "passed": bool(cells) and all(row["passed"] for row in cells.values()),
    }


def _public_cleaning_cost(rule: dict[str, Any], duration_days: int) -> float:
    if "one_day_work_points" in rule:
        return (
            float(rule["one_day_work_points"])
            * (
                1.0
                + float(rule["extra_work_fraction_per_additional_day"])
                * (max(1, int(duration_days)) - 1)
            )
            * float(rule["cash_per_work_point"])
        )
    return float(rule["base_cost"]) + float(
        rule["extra_per_additional_day"]
    ) * (max(1, int(duration_days)) - 1)


def _post_missing_opening_prices(
    environment: OutfitterEnvironment,
    state: Any,
    strategy: str,
) -> None:
    """Give every scripted policy an explicit, public-information Day-1 action."""

    if len(state.prices) == len(environment.offers):
        return
    retail_markup = {
        "naive": 1.15,
        "heuristic": 1.35,
        "informed_public": 1.55,
        "scale_winner": 1.55,
        "fixed_public": 1.55,
        "buy_liquidate": 1_000_000.0,
        "reject_all": 1_000_000.0,
        "aggressive_expansion_reference": 1.55,
    }[strategy]
    recovery_days = {
        "naive": 60.0,
        "heuristic": 35.0,
        "informed_public": 10.0,
        "scale_winner": 10.0,
        "fixed_public": 10.0,
        "buy_liquidate": 0.000001,
        "reject_all": 0.000001,
        "aggressive_expansion_reference": 10.0,
    }[strategy]
    updates = []
    for offer in sorted(environment.dataset.offers, key=lambda row: row.offer_id):
        if offer.offer_id in state.prices:
            continue
        if offer.asset_kind == "consumable":
            unit_cost = float(offer.unit_cost)
            if strategy in {
                "informed_public",
                "fixed_public",
                "scale_winner",
                "aggressive_expansion_reference",
            }:
                # Public break-pack economics: keep a defensible unit margin,
                # charge for avoiding a full source package, but never let that
                # convenience heuristic create an unbounded unit markup.
                price = max(
                    1.25 * unit_cost,
                    min(
                        2.50 * unit_cost,
                        0.65 * float(offer.package_price),
                    ),
                )
            else:
                price = unit_cost * retail_markup
        else:
            cleaning = environment.public_offer(offer.offer_id)[
                "normal_return_cleaning"
            ]
            cleaning_per_day = _public_cleaning_cost(cleaning, 1) / 3.0
            price = float(offer.unit_cost) / recovery_days + cleaning_per_day
        updates.append(
            {"offer_id": offer.offer_id, "price": round(max(0.01, price), 2)}
        )
    if updates:
        result = environment.set_prices(state, updates)
        if not result.ok:
            raise RuntimeError(f"scripted Day-1 pricing failed: {result.data}")


def _adjust_scale_winner_prices(
    environment: OutfitterEnvironment, state: Any
) -> None:
    """Run an uncapped observable conversion-price hill climb."""

    if not state.daily_history or state.day % 7 != 1:
        return
    updates = []
    for offer_id, counts in sorted(_recent_offer_metrics(state, 14).items()):
        exposures = int(counts.get("quote_exposures", 0))
        if exposures < 5:
            continue
        conversion = int(counts.get("fulfilled_units", 0)) / exposures
        multiplier = 1.12 if conversion >= 0.55 else 0.90 if conversion < 0.15 else 1.0
        if multiplier != 1.0:
            updates.append(
                {
                    "offer_id": offer_id,
                    "price": round(max(0.01, state.prices[offer_id] * multiplier), 2),
                }
            )
    if updates:
        environment.set_prices(state, updates)


def _buy_liquidate_actions(
    environment: OutfitterEnvironment,
    state: Any,
    *,
    opening_asset_ids: set[str],
) -> None:
    """Exercise purchase-to-dealer churn without consuming the opening fleet."""

    acquired = sorted(
        asset.asset_id
        for asset in state.assets
        if asset.asset_id not in opening_asset_ids
        and asset.status not in {"retired", "sold", "rented"}
    )
    if acquired:
        result = environment.sell_to_liquidation_dealer(state, acquired, [])
        if not result.ok:
            raise RuntimeError(f"buy-liquidate sale failed: {result.data}")
    if any(order.status == "in_transit" for order in state.orders):
        return
    candidates = []
    for offer in environment.dataset.offers:
        if offer.asset_kind != "durable":
            continue
        delivery_days = int(environment.supplier_delivery_estimate(offer.seller_id))
        delivery_cutoff = int(
            state.decision_horizon_day
            if state.decision_horizon_day is not None
            else state.max_days
        )
        if state.day + delivery_days > delivery_cutoff:
            continue
        candidates.append(offer)
    if not candidates:
        return
    chosen = min(candidates, key=lambda row: (float(row.package_price), row.offer_id))
    if float(chosen.package_price) > float(state.cash):
        return
    result = _scripted_place_purchase_order(
        environment,
        state,
        seller_id=chosen.seller_id,
        items=[{"offer_id": chosen.offer_id, "packs": 1}],
    )
    if not result.ok:
        raise RuntimeError(f"buy-liquidate purchase failed: {result.data}")


def _retail_gross_margin(
    environment: OutfitterEnvironment, state: Any
) -> dict[str, float]:
    values: dict[str, float] = defaultdict(float)
    for report in state.daily_history:
        for metric in report.get("service_metrics", {}).values():
            for offer_id, row in metric.get("offer_fulfillment_breakdown", {}).items():
                if environment.offers[offer_id].asset_kind == "consumable":
                    values[offer_id] += float(row.get("revenue", 0.0)) - float(
                        row.get("retail_cogs", 0.0)
                    )
    return {key: round(value, 2) for key, value in sorted(values.items())}


def _initial_offer_state(
    environment: OutfitterEnvironment, state: Any
) -> dict[str, dict[str, float | int]]:
    """Freeze opening units and official terminal values before assets mutate."""

    values: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "initial_retail_units": 0,
            "initial_rental_assets": 0,
            "initial_official_value": 0.0,
        }
    )
    for offer_id, inventory in state.retail_inventory.items():
        quantity = int(inventory["quantity"])
        values[offer_id]["initial_retail_units"] = quantity
        values[offer_id]["initial_official_value"] = (
            environment.retail_liquidation_proceeds(
                float(inventory["avg_unit_cost"]), quantity
            )
        )
    for asset in state.assets:
        row = values[asset.offer_id]
        row["initial_rental_assets"] = int(row["initial_rental_assets"]) + 1
        quote = environment.secondhand_quote(asset, terminal=True)
        if quote is not None:
            row["initial_official_value"] = round(
                float(row["initial_official_value"])
                + float(quote["offer_price"]),
                2,
            )
    return {key: dict(value) for key, value in values.items()}


def _offer_economic_bridge(
    environment: OutfitterEnvironment,
    state: Any,
    initial_offer_state: dict[str, dict[str, float | int]],
    *,
    official_value_change: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """Attribute value creation to offers and reconcile it to the official score.

    Holding costs and the common operating fee do not have a customer transaction
    identity, so they remain explicit shared costs instead of being assigned using
    an arbitrary allocation rule.
    """

    numeric_fields = (
        "fulfilled_units",
        "rental_days",
        "quote_exposures",
        "quoted_units",
        "revenue",
        "retail_cogs",
        "purchase_units",
        "purchase_cost",
        "cleaning_cost_paid",
        "repair_cost",
        "damage_deductions",
        "asset_sale_proceeds",
        "retail_liquidation_units",
        "retail_liquidation_proceeds",
        "retail_liquidation_book_cost",
        "retail_liquidation_loss",
        "pending_cleaning_cost",
        "final_official_value",
    )
    values: dict[str, dict[str, Any]] = defaultdict(
        lambda: {field: 0.0 for field in numeric_fields}
    )
    for offer_id, opening in initial_offer_state.items():
        values[offer_id].update(opening)

    for report in state.daily_history:
        for metric in report.get("service_metrics", {}).values():
            for offer_id, row in metric.get(
                "offer_fulfillment_breakdown", {}
            ).items():
                target = values[offer_id]
                for field in (
                    "fulfilled_units",
                    "rental_days",
                    "quote_exposures",
                    "quoted_units",
                    "revenue",
                    "retail_cogs",
                ):
                    target[field] = float(target[field]) + float(
                        row.get(field, 0.0)
                    )
        for row in report.get("agent_actions", {}).get(
            "retail_liquidations", []
        ):
            target = values[str(row["offer_id"])]
            target["retail_liquidation_units"] = float(
                target["retail_liquidation_units"]
            ) + int(row.get("units", 0))
            for field in (
                "retail_liquidation_proceeds",
                "retail_liquidation_book_cost",
                "retail_liquidation_loss",
            ):
                source = field.removeprefix("retail_liquidation_")
                target[field] = float(target[field]) + float(row.get(source, 0.0))

    for order in state.orders:
        for item in order.items:
            target = values[str(item["offer_id"])]
            target["purchase_units"] = float(target["purchase_units"]) + int(
                item["packs"] * item["package_units"]
            )
            target["purchase_cost"] = float(target["purchase_cost"]) + float(
                item["line_cost"]
            )

    scheduled_cleaning: defaultdict[str, float] = defaultdict(float)
    acquisition_days: defaultdict[str, list[int]] = defaultdict(list)
    for event in state.rental_activity:
        offer_id = str(event["offer_id"])
        event_type = str(event["event_type"])
        target = values[offer_id]
        if event_type == "asset_acquired":
            acquisition_days[offer_id].append(int(event["day"]))
        elif event_type == "returned_normal":
            scheduled_cleaning[offer_id] += float(
                event.get("scheduled_cleaning_cost", 0.0)
            )
        elif event_type == "returned_damaged":
            target["damage_deductions"] = float(
                target["damage_deductions"]
            ) + float(event.get("damage_deduction", 0.0))
        elif event_type == "repair_started":
            target["repair_cost"] = float(target["repair_cost"]) + float(
                event.get("repair_cost", 0.0)
            )
        elif event_type == "asset_sold":
            target["asset_sale_proceeds"] = float(
                target["asset_sale_proceeds"]
            ) + float(event.get("sale_proceeds", 0.0))

    for job in state.cleaning_queue:
        target = values[job.offer_id]
        target["pending_cleaning_cost"] = float(
            target["pending_cleaning_cost"]
        ) + float(job.cleaning_cost)
    for offer_id, scheduled in scheduled_cleaning.items():
        values[offer_id]["cleaning_cost_paid"] = max(
            0.0,
            scheduled - float(values[offer_id]["pending_cleaning_cost"]),
        )

    asset_statuses: defaultdict[str, Counter[str]] = defaultdict(Counter)
    acquired_assets: Counter[str] = Counter()
    for asset in state.assets:
        asset_statuses[asset.offer_id][asset.status] += 1
        if int(asset.acquired_day) > 0:
            acquired_assets[asset.offer_id] += 1
        quote = environment.secondhand_quote(asset, terminal=True)
        if quote is not None:
            values[asset.offer_id]["final_official_value"] = float(
                values[asset.offer_id]["final_official_value"]
            ) + float(quote["offer_price"])
    for offer_id, inventory in state.retail_inventory.items():
        values[offer_id]["final_official_value"] = float(
            values[offer_id]["final_official_value"]
        ) + environment.retail_liquidation_proceeds(
            float(inventory["avg_unit_cost"]), int(inventory["quantity"])
        )
        values[offer_id]["final_retail_units"] = int(inventory["quantity"])
    for offer_id, target in values.items():
        target["final_official_value"] = max(
            0.0,
            float(target["final_official_value"])
            - float(target["pending_cleaning_cost"]),
        )

    snapshots: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in state.daily_history:
        for row in report.get("offer_snapshots", []):
            snapshots[str(row["offer_id"])].append(row)

    result: dict[str, dict[str, Any]] = {}
    for offer_id in sorted(values):
        offer = environment.offers[offer_id]
        target = values[offer_id]
        opening = float(target.get("initial_official_value", 0.0))
        closing = float(target["final_official_value"])
        contribution = (
            float(target["revenue"])
            + float(target["damage_deductions"])
            + float(target["asset_sale_proceeds"])
            + float(target["retail_liquidation_proceeds"])
            - float(target["purchase_cost"])
            - float(target["cleaning_cost_paid"])
            - float(target["repair_cost"])
            + closing
            - opening
        )
        price_rows = snapshots.get(offer_id, [])
        prices = [float(row["posted_price"]) for row in price_rows]
        result[offer_id] = {
            "name": offer.name,
            "asset_kind": offer.asset_kind,
            "service_key": str(offer.fulfills_service_keys[0]),
            "choice_group": offer.choice_group,
            "unit_cost": round(float(offer.unit_cost), 4),
            "initial_retail_units": int(
                target.get("initial_retail_units", 0)
            ),
            "initial_rental_assets": int(
                target.get("initial_rental_assets", 0)
            ),
            "purchased_units": int(target["purchase_units"]),
            "acquired_rental_assets": int(acquired_assets[offer_id]),
            "acquisition_days": acquisition_days.get(offer_id, []),
            "final_retail_units": int(target.get("final_retail_units", 0)),
            "final_rental_asset_statuses": dict(
                sorted(asset_statuses[offer_id].items())
            ),
            "fulfilled_units": int(target["fulfilled_units"]),
            "rental_days": int(target["rental_days"]),
            "quote_exposures": int(target["quote_exposures"]),
            "quoted_units": int(target["quoted_units"]),
            "revenue": round(float(target["revenue"]), 2),
            "retail_cogs": round(float(target["retail_cogs"]), 2),
            "retail_gross_margin": round(
                float(target["revenue"]) - float(target["retail_cogs"]), 2
            )
            if offer.asset_kind == "consumable"
            else None,
            "purchase_cost": round(float(target["purchase_cost"]), 2),
            "cleaning_cost_paid": round(
                float(target["cleaning_cost_paid"]), 2
            ),
            "pending_cleaning_cost": round(
                float(target["pending_cleaning_cost"]), 2
            ),
            "repair_cost": round(float(target["repair_cost"]), 2),
            "damage_deductions": round(
                float(target["damage_deductions"]), 2
            ),
            "asset_sale_proceeds": round(
                float(target["asset_sale_proceeds"]), 2
            ),
            "retail_units_liquidated": int(
                target["retail_liquidation_units"]
            ),
            "retail_liquidation_proceeds": round(
                float(target["retail_liquidation_proceeds"]), 2
            ),
            "retail_liquidation_book_cost": round(
                float(target["retail_liquidation_book_cost"]), 2
            ),
            "retail_liquidation_loss": round(
                float(target["retail_liquidation_loss"]), 2
            ),
            "initial_official_value": round(opening, 2),
            "final_official_value": round(closing, 2),
            "contribution_before_shared_costs": round(contribution, 2),
            "average_posted_price": round(sum(prices) / len(prices), 2)
            if prices
            else None,
            "minimum_posted_price": round(min(prices), 2) if prices else None,
            "maximum_posted_price": round(max(prices), 2) if prices else None,
        }

    attributed = round(
        sum(row["contribution_before_shared_costs"] for row in result.values()),
        2,
    )
    offer_breakdown_revenue = round(
        sum(float(row["revenue"]) for row in result.values()), 2
    )
    recognized_revenue = round(
        state.cumulative_rental_revenue + state.cumulative_retail_revenue,
        2,
    )
    # Approved reservation extensions are recognized in the financial ledger
    # without creating a second fulfillment attempt.  Keep that revenue
    # explicit instead of silently losing it from the offer bridge.
    revenue_not_in_offer_breakdown = round(
        recognized_revenue - offer_breakdown_revenue, 2
    )
    shared_costs = round(
        state.cumulative_retail_holding_cost
        + state.cumulative_rental_holding_cost
        + state.cumulative_maintenance_cost
        + state.cumulative_fees
        + (
            state.cumulative_model_cost
            if environment.config.charge_model_usage_to_cash
            else 0.0
        ),
        2,
    )
    deposit_liability_change = round(float(state.deposit_liability), 2)
    unearned_liability_change = round(
        float(getattr(state, "unearned_revenue_liability", 0.0)), 2
    )
    emergency_principal = round(state.emergency_credit_principal, 2)
    emergency_liability = round(state.emergency_credit_liability, 2)
    reservation_compensation = round(
        state.cumulative_reservation_compensation, 2
    )
    unpaid_bankruptcy_shortfall = round(state.bankruptcy_shortfall, 2)
    per_offer_non_cash_change = round(
        sum(
            float(row["final_official_value"])
            - float(row["initial_official_value"])
            for row in result.values()
        ),
        2,
    )
    ending_liabilities = round(
        deposit_liability_change
        + unearned_liability_change
        + emergency_liability,
        2,
    )
    ending_refundable_cash_receipts = round(
        deposit_liability_change + unearned_liability_change, 2
    )
    official_non_cash_change = round(
        official_value_change
        - (float(state.cash) - float(environment.config.initial_cash))
        + ending_liabilities,
        2,
    )
    # Individual secondhand quotes are nominal; the official score applies
    # portfolio liquidation depth.  This independently calculated adjustment
    # prevents a nominal offer sum from masquerading as the terminal score.
    portfolio_valuation_adjustment = round(
        official_non_cash_change - per_offer_non_cash_change, 2
    )
    reconciled = round(
        attributed
        + revenue_not_in_offer_breakdown
        + portfolio_valuation_adjustment
        + emergency_principal
        + ending_refundable_cash_receipts
        + unpaid_bankruptcy_shortfall
        - shared_costs
        - reservation_compensation
        - deposit_liability_change
        - unearned_liability_change
        - emergency_liability,
        2,
    )
    bridge = {
        "offer_contribution_before_shared_costs": attributed,
        "offer_breakdown_recognized_revenue": offer_breakdown_revenue,
        "recognized_revenue_not_in_offer_breakdown": (
            revenue_not_in_offer_breakdown
        ),
        "portfolio_liquidation_valuation_adjustment": (
            portfolio_valuation_adjustment
        ),
        "retail_holding_cost": round(state.cumulative_retail_holding_cost, 2),
        "rental_holding_cost": round(state.cumulative_rental_holding_cost, 2),
        "maintenance_cost": round(state.cumulative_maintenance_cost, 2),
        "reservation_compensation": reservation_compensation,
        "unpaid_bankruptcy_shortfall": unpaid_bankruptcy_shortfall,
        "operating_fee": round(state.cumulative_fees, 2),
        "charged_model_cost": round(
            state.cumulative_model_cost
            if environment.config.charge_model_usage_to_cash
            else 0.0,
            6,
        ),
        "ending_deposit_liability": deposit_liability_change,
        "ending_unearned_revenue_liability": unearned_liability_change,
        "ending_refundable_cash_receipts": (
            ending_refundable_cash_receipts
        ),
        "emergency_credit_principal": emergency_principal,
        "ending_emergency_credit_liability": emergency_liability,
        "reconciled_official_value_change": reconciled,
        "reported_official_value_change": round(official_value_change, 2),
        "reconciliation_difference": round(
            reconciled - official_value_change, 2
        ),
    }
    return result, bridge


def _seasonal_period_audit(
    reports: list[dict[str, Any]],
    *,
    initial_value: float,
    initial_contract_value: float,
) -> list[dict[str, Any]]:
    """Aggregate the April-to-October episode by real calendar month."""

    by_month: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        month = str(report["calendar"]["date"])[:7]
        by_month[month].append(report)
    official_anchor = float(initial_value)
    contract_anchor = float(initial_contract_value)
    result = []
    for month, rows in sorted(by_month.items()):
        financial_rows = [row["financial"] for row in rows]
        demand_rows = [row["demand"] for row in rows]
        official_end = float(rows[-1]["end"]["official_liquidation_value"])
        contract_end = float(
            rows[-1]["end"]["contract_adjusted_economic_value"]
        )
        requested = sum(int(row["requested_units"]) for row in demand_rows)
        fulfilled = sum(int(row["fulfilled_units"]) for row in demand_rows)
        result.append(
            {
                "month": month,
                "start_day": int(rows[0]["day"]),
                "end_day": int(rows[-1]["day"]),
                "days": len(rows),
                "public_holiday_days": sum(
                    row["calendar"]["day_type"] == "public_holiday"
                    for row in rows
                ),
                "weekend_days": sum(
                    bool(row["calendar"]["is_weekend"]) for row in rows
                ),
                "customers": sum(
                    sum(int(value) for value in row.get("arrivals", {}).values())
                    for row in rows
                ),
                "requested_units": requested,
                "fulfilled_units": fulfilled,
                "fulfillment_rate": round(fulfilled / max(1, requested), 6),
                "outside_rejections": sum(
                    int(row.get("outside_rejections", 0)) for row in demand_rows
                ),
                "stockout_rejections": sum(
                    int(row.get("stockout_rejections", 0)) for row in demand_rows
                ),
                "rental_revenue": round(
                    sum(float(row["rental_revenue"]) for row in financial_rows),
                    2,
                ),
                "retail_revenue": round(
                    sum(float(row["retail_revenue"]) for row in financial_rows),
                    2,
                ),
                "retail_liquidation_proceeds": round(
                    sum(
                        float(row.get("retail_liquidation_proceeds", 0.0))
                        for row in financial_rows
                    ),
                    2,
                ),
                "retail_liquidation_loss": round(
                    sum(
                        float(row.get("retail_liquidation_loss", 0.0))
                        for row in financial_rows
                    ),
                    2,
                ),
                "retail_units_liquidated": sum(
                    int(row.get("retail_units_liquidated", 0))
                    for row in financial_rows
                ),
                "purchase_cost": round(
                    sum(float(row["purchase_cost"]) for row in financial_rows),
                    2,
                ),
                "cleaning_and_repair_cost": round(
                    sum(
                        float(row["cleaning_cost"])
                        + float(row["repair_cost"])
                        + float(row.get("maintenance_cost", 0.0))
                        + float(row.get("reservation_compensation", 0.0))
                        for row in financial_rows
                    ),
                    2,
                ),
                "holding_cost": round(
                    sum(
                        float(row["retail_holding_cost"])
                        + float(row["rental_holding_cost"])
                        + float(row.get("rental_idle_holding_cost", 0.0))
                        for row in financial_rows
                    ),
                    2,
                ),
                "operating_fee": round(
                    sum(float(row["operating_fee"]) for row in financial_rows),
                    2,
                ),
                "official_value_change": round(
                    official_end - official_anchor, 2
                ),
                "contract_value_change": round(
                    contract_end - contract_anchor, 2
                ),
                "official_value_end": round(official_end, 2),
                "contract_value_end": round(contract_end, 2),
            }
        )
        official_anchor = official_end
        contract_anchor = contract_end
    return result


def _weighted_median_pairs(values: list[tuple[float, float]]) -> float | None:
    rows = sorted((float(value), float(weight)) for value, weight in values if weight > 0)
    if not rows:
        return None
    threshold = sum(weight for _value, weight in rows) / 2.0
    cumulative = 0.0
    for value, weight in rows:
        cumulative += weight
        if cumulative + 1e-12 >= threshold:
            return value
    return rows[-1][0]


def _lifecycle_payback_diagnostics(
    environment: OutfitterEnvironment,
    state: Any,
    *,
    completed_days: int,
) -> dict[str, Any]:
    """Report benchmark payback outcomes without using them to set prices or WTP."""

    if not environment.config.lifecycle_economics_enabled:
        return {"enabled": False}
    assert environment.lifecycle_scales is not None
    contracts_by_offer: defaultdict[str, list[Any]] = defaultdict(list)
    contracts_by_asset: defaultdict[str, list[Any]] = defaultdict(list)
    events_by_offer: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    events_by_asset: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for contract in state.contracts:
        contracts_by_offer[str(contract.offer_id)].append(contract)
        contracts_by_asset[str(contract.asset_id)].append(contract)
    for event in state.rental_activity:
        events_by_offer[str(event.get("offer_id") or "")].append(event)
        events_by_asset[str(event.get("asset_id") or "")].append(event)

    offer_rows: list[dict[str, Any]] = []
    gross_pairs: list[tuple[float, float]] = []
    contribution_pairs: list[tuple[float, float]] = []
    for offer_id in sorted(contracts_by_offer):
        contracts = contracts_by_offer[offer_id]
        if not contracts:
            continue
        events = events_by_offer[offer_id]
        rental_cash = sum(float(row.rent_paid) for row in contracts)
        cleaning = sum(
            float(row.get("scheduled_cleaning_cost", 0.0))
            for row in events
            if row.get("event_type") == "returned_normal"
        )
        uncovered_damage = sum(
            float(row.get("uncovered_repair_cost", 0.0))
            for row in events
            if row.get("event_type") == "returned_damaged"
        )
        count = len(contracts)
        gross_cash = rental_cash / count
        variable_contribution = (
            rental_cash - cleaning - uncovered_damage
        ) / count
        replacement = float(environment.offers[offer_id].unit_cost)
        gross_payback = replacement / gross_cash if gross_cash > 0 else None
        variable_payback = (
            replacement / variable_contribution
            if variable_contribution > 0
            else None
        )
        row = {
            "offer_id": offer_id,
            "name": environment.offers[offer_id].name,
            "service_line": environment.offers[offer_id].service_line,
            "realized_contracts": count,
            "realized_rental_cash": round(rental_cash, 2),
            "mean_rental_cash_per_contract": round(gross_cash, 6),
            "mean_variable_contribution_per_contract": round(
                variable_contribution, 6
            ),
            "scheduled_cleaning_cost": round(cleaning, 2),
            "uncovered_damage_cost": round(uncovered_damage, 2),
            "gross_payback_contracts": (
                None if gross_payback is None else round(gross_payback, 6)
            ),
            "variable_contribution_payback_contracts": (
                None if variable_payback is None else round(variable_payback, 6)
            ),
        }
        offer_rows.append(row)
        if gross_payback is not None:
            gross_pairs.append((gross_payback, count))
        if variable_payback is not None:
            contribution_pairs.append((variable_payback, count))

    terminal_quotes = [
        (asset, environment.secondhand_quote(asset, terminal=True))
        for asset in state.assets
        if asset.status != "sold"
    ]
    fleet_nominal = sum(
        float(quote["offer_price"])
        for _asset, quote in terminal_quotes
        if quote is not None
    )
    fleet_actual, _depth = environment._depth_adjusted_pool_proceeds(
        state, "rental_asset", fleet_nominal
    )
    terminal_depth_rate = (
        fleet_actual / fleet_nominal if fleet_nominal > 1e-12 else 0.0
    )
    purchased_assets = [
        asset for asset in state.assets if int(asset.acquired_day) > 0
    ]
    cohort_rows: list[dict[str, Any]] = []
    for asset in purchased_assets:
        events = events_by_asset[asset.asset_id]
        contracts = contracts_by_asset[asset.asset_id]
        sale_events = [
            row for row in events if row.get("event_type") == "asset_sold"
        ]
        billed_end = (
            min(int(row["day"]) for row in sale_events)
            if sale_events
            else completed_days
        )
        holding_days = max(0, billed_end - int(asset.acquired_day) + 1)
        rental_days = sum(int(row.duration_days) for row in contracts)
        on_site_days = max(0, holding_days - rental_days)
        projection = environment._lifecycle_projection(asset.offer_id)
        capital_holding = (
            holding_days
            * float(asset.unit_cost)
            * environment.lifecycle_scales.annual_capital_rate
            / 365.0
        )
        space_holding = (
            on_site_days
            * float(projection["storage_burden_units"])
            * environment.lifecycle_scales.daily_space_rate_per_storage_unit
        )
        rental_cash = sum(float(row.rent_paid) for row in contracts)
        damage_deductions = sum(
            float(row.get("damage_deduction", 0.0))
            for row in events
            if row.get("event_type") == "returned_damaged"
        )
        cleaning = sum(
            float(row.get("scheduled_cleaning_cost", 0.0))
            for row in events
            if row.get("event_type") == "returned_normal"
        )
        repairs = sum(
            float(row.get("repair_cost", 0.0))
            for row in events
            if row.get("event_type") == "repair_started"
        )
        maintenance = sum(
            float(row.get("maintenance_cost", 0.0))
            for row in events
            if row.get("event_type") == "maintenance_started"
        )
        sale_cash = float(asset.sale_proceeds or 0.0) if asset.status == "sold" else 0.0
        terminal_quote = environment.secondhand_quote(asset, terminal=True)
        terminal_value = (
            float(terminal_quote["offer_price"]) * terminal_depth_rate
            if terminal_quote is not None and asset.status != "sold"
            else 0.0
        )
        operating_cash_recovery = (
            rental_cash
            + damage_deductions
            - cleaning
            - repairs
            - maintenance
            - capital_holding
            - space_holding
        )
        realized_cash_recovery = operating_cash_recovery + sale_cash
        economic_recovery = realized_cash_recovery + terminal_value
        acquisition = float(asset.unit_cost)
        cohort_rows.append(
            {
                "asset_id": asset.asset_id,
                "offer_id": asset.offer_id,
                "acquired_day": int(asset.acquired_day),
                "final_status": asset.status,
                "contracts": len(contracts),
                "rental_days": rental_days,
                "acquisition_cost": round(acquisition, 2),
                "rental_cash": round(rental_cash, 2),
                "damage_deductions": round(damage_deductions, 2),
                "cleaning_cost": round(cleaning, 2),
                "repair_cost": round(repairs, 2),
                "maintenance_cost": round(maintenance, 2),
                "capital_holding_cost_approx": round(capital_holding, 2),
                "space_holding_cost_approx": round(space_holding, 2),
                "sale_cash": round(sale_cash, 2),
                "terminal_dealer_value": round(terminal_value, 2),
                "terminal_depth_adjusted_value": round(terminal_value, 2),
                "realized_cash_recovery": round(realized_cash_recovery, 2),
                "economic_recovery": round(economic_recovery, 2),
                "realized_cash_recovery_ratio": round(
                    realized_cash_recovery / acquisition, 6
                ),
                "economic_recovery_ratio": round(
                    economic_recovery / acquisition, 6
                ),
                "unrecovered_economic_share": round(
                    max(0.0, 1.0 - economic_recovery / acquisition), 6
                ),
            }
        )
    total_acquisition = sum(float(row["acquisition_cost"]) for row in cohort_rows)
    total_cash_recovery = sum(
        float(row["realized_cash_recovery"]) for row in cohort_rows
    )
    total_economic_recovery = sum(
        float(row["economic_recovery"]) for row in cohort_rows
    )
    return {
        "enabled": True,
        "interpretation": (
            "descriptive benchmark outcomes; never used to generate customer WTP, "
            "inquiries, or contract prices"
        ),
        "realized_offer_payback": offer_rows,
        "gross_payback_contracts_weighted_median": (
            None
            if (value := _weighted_median_pairs(gross_pairs)) is None
            else round(value, 6)
        ),
        "variable_contribution_payback_contracts_weighted_median": (
            None
            if (value := _weighted_median_pairs(contribution_pairs)) is None
            else round(value, 6)
        ),
        "purchased_asset_cohorts": cohort_rows,
        "purchased_assets": len(cohort_rows),
        "purchased_assets_unrecovered": sum(
            float(row["economic_recovery_ratio"]) < 1.0 - 1e-9
            for row in cohort_rows
        ),
        "purchased_asset_unrecovered_share": round(
            sum(
                float(row["economic_recovery_ratio"]) < 1.0 - 1e-9
                for row in cohort_rows
            )
            / max(1, len(cohort_rows)),
            6,
        ),
        "cohort_total_acquisition_cost": round(total_acquisition, 2),
        "cohort_total_realized_cash_recovery": round(total_cash_recovery, 2),
        "cohort_total_economic_recovery": round(total_economic_recovery, 2),
        "cohort_realized_cash_recovery_ratio": (
            None
            if total_acquisition <= 0
            else round(total_cash_recovery / total_acquisition, 6)
        ),
        "cohort_economic_recovery_ratio": (
            None
            if total_acquisition <= 0
            else round(total_economic_recovery / total_acquisition, 6)
        ),
        "terminal_depth_allocation_rate": round(terminal_depth_rate, 8),
    }


def _daily_performance_curve(
    reports: list[dict[str, Any]],
    *,
    initial_value: float,
    initial_contract_value: float | None = None,
) -> list[dict[str, Any]]:
    contract_anchor = (
        initial_value if initial_contract_value is None else initial_contract_value
    )
    cumulative_revenue = 0.0
    cumulative_non_deposit_cash_flow = 0.0
    curve = []
    for report in reports:
        financial = report["financial"]
        daily_revenue = float(financial["rental_revenue"]) + float(
            financial["retail_revenue"]
        )
        daily_non_deposit_cash_flow = (
            daily_revenue
            + float(financial.get("reservation_prepayments_collected", 0.0))
            - float(financial.get("reservation_prepayment_refunds", 0.0))
            - float(financial.get("reservation_prepayment_recognized", 0.0))
            + float(financial.get("damage_deductions", 0.0))
            + float(financial.get("asset_sale_proceeds", 0.0))
            + float(financial.get("retail_liquidation_proceeds", 0.0))
            - float(financial.get("purchase_cost", 0.0))
            - float(financial.get("cleaning_cost", 0.0))
            - float(financial.get("repair_cost", 0.0))
            - float(financial.get("maintenance_cost", 0.0))
            - float(financial.get("reservation_compensation", 0.0))
            - float(financial.get("retail_holding_cost", 0.0))
            - float(financial.get("rental_holding_cost", 0.0))
            - float(financial.get("rental_idle_holding_cost", 0.0))
            - float(financial.get("operating_fee", 0.0))
            - float(financial.get("model_cost", 0.0))
        )
        cumulative_revenue += daily_revenue
        cumulative_non_deposit_cash_flow += daily_non_deposit_cash_flow
        value = float(report["end"]["official_liquidation_value"])
        curve.append(
            {
                "day": int(report["day"]),
                "daily_revenue": round(daily_revenue, 2),
                "cumulative_revenue": round(cumulative_revenue, 2),
                "daily_non_deposit_cash_flow": round(
                    daily_non_deposit_cash_flow, 2
                ),
                "cumulative_non_deposit_cash_flow": round(
                    cumulative_non_deposit_cash_flow, 2
                ),
                "official_liquidation_value": round(value, 2),
                "official_value_change": round(value - initial_value, 2),
                "contract_adjusted_economic_value": round(
                    float(report["end"]["contract_adjusted_economic_value"]), 2
                ),
                "contract_adjusted_value_change": round(
                    float(report["end"]["contract_adjusted_economic_value"])
                    - contract_anchor,
                    2,
                ),
                "cash": round(float(report["end"]["cash"]), 2),
                "deposit_liability": round(
                    float(report["end"]["deposit_liability"]), 2
                ),
                "requested_units": int(report["demand"]["requested_units"]),
                "fulfilled_units": int(report["demand"]["fulfilled_units"]),
            }
        )
    return curve


def _in_transit_units(state: Any) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for order in state.orders:
        if order.status != "in_transit":
            continue
        for item in order.items:
            result[str(item["offer_id"])] += int(
                item["packs"] * item["package_units"]
            ) - int(item.get("received_units", 0))
    return result


def _place_affordable_orders(
    environment: OutfitterEnvironment,
    state: Any,
    desired_packs: dict[str, int],
    *,
    cash_reserve: float,
) -> None:
    by_seller: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for offer_id, packs in sorted(desired_packs.items()):
        if packs <= 0:
            continue
        offer = environment.offers[offer_id]
        by_seller[offer.seller_id].append(
            {"offer_id": offer_id, "packs": min(100, int(packs))}
        )
    for seller_id, items in sorted(by_seller.items()):
        affordable: list[dict[str, Any]] = []
        for item in items:
            offer = environment.offers[item["offer_id"]]
            if state.cash - offer.package_price * item["packs"] < cash_reserve:
                continue
            affordable.append(item)
        if affordable:
            _scripted_place_purchase_order(
                environment,
                state,
                seller_id=seller_id,
                items=affordable,
            )
        if state.done:
            return


def _scripted_place_purchase_order(
    environment: OutfitterEnvironment,
    state: Any,
    *,
    seller_id: str,
    items: list[dict[str, Any]],
) -> ToolResult:
    """Apply one optional diagnostic fleet cap to every scripted order path."""

    cap = getattr(environment, "_scripted_fleet_expansion_cap", None)
    if cap is None:
        return environment.place_purchase_order(
            state, seller_id=seller_id, items=items
        )
    opening = len(environment.dataset.initial_assets)
    active = sum(
        asset.status not in {"retired", "sold"} for asset in state.assets
    )
    in_transit = sum(
        int(item["packs"]) * int(item["package_units"])
        for order in state.orders
        if order.status == "in_transit"
        for item in order.items
        if environment.offers[str(item["offer_id"])].asset_kind == "durable"
    )
    room = max(0, opening + int(cap) - active - in_transit)
    filtered: list[dict[str, Any]] = []
    for item in items:
        offer_id = str(item.get("offer_id", ""))
        offer = environment.offers.get(offer_id)
        if offer is None or offer.asset_kind != "durable":
            filtered.append(dict(item))
            continue
        allowed = min(
            int(item.get("packs", 0)),
            room // max(1, int(offer.package_units)),
        )
        if allowed <= 0:
            continue
        filtered.append({**item, "packs": allowed})
        room -= allowed * int(offer.package_units)
    if not filtered:
        return ToolResult(
            False,
            {
                "error": "scripted diagnostic fleet expansion cap reached",
                "maximum_net_active_expansion": int(cap),
            },
        )
    return environment.place_purchase_order(
        state, seller_id=seller_id, items=filtered
    )


def _seed_scripted_opening_fleet(
    environment: OutfitterEnvironment,
    state: Any,
    *,
    units_per_offer: int,
) -> None:
    """Create an intentionally over-capitalized Day-1 control fleet.

    This diagnostic is not an agent policy: supplier orders are charged through
    the ordinary ledger, then treated as a pre-opening delivery so the control
    isolates whether abundant exact-offer capacity can eliminate failures.
    """

    if units_per_offer <= 0:
        return
    owned = Counter(
        asset.offer_id
        for asset in state.assets
        if asset.status not in {"retired", "sold"}
    )
    by_seller: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for offer in sorted(environment.dataset.offers, key=lambda row: row.offer_id):
        if offer.asset_kind != "durable":
            continue
        missing = max(0, int(units_per_offer) - int(owned[offer.offer_id]))
        packs = math.ceil(missing / max(1, int(offer.package_units)))
        if packs:
            by_seller[offer.seller_id].append(
                {"offer_id": offer.offer_id, "packs": packs}
            )
    placed_ids: set[str] = set()
    for seller_id, items in sorted(by_seller.items()):
        result = environment.place_purchase_order(
            state, seller_id=seller_id, items=items
        )
        if not result.ok:
            raise RuntimeError(
                f"opening capacity-control order failed: {result.data}"
            )
        placed_ids.add(str(result.data["order"]["order_id"]))
    for order in state.orders:
        if order.order_id in placed_ids:
            order.delivery_day = state.day
    environment._process_morning(state)


def _order_scripted_forced_expansion(
    environment: OutfitterEnvironment,
    state: Any,
    *,
    additional_assets: int,
) -> None:
    """Place a nested public-economics Day-1 scale-stress order."""

    if additional_assets <= 0:
        return
    candidates = []
    for offer in environment.dataset.offers:
        if offer.asset_kind != "durable":
            continue
        cleaning = _public_cleaning_cost(
            environment.public_offer(offer.offer_id)["normal_return_cleaning"],
            1,
        )
        posted = float(state.prices[offer.offer_id])
        contribution = max(0.01, posted - cleaning)
        candidates.append(
            (
                float(offer.unit_cost) / contribution,
                float(offer.unit_cost),
                offer.offer_id,
            )
        )
    candidates.sort()
    if not candidates:
        return
    desired: Counter[str] = Counter(
        candidates[index % len(candidates)][2]
        for index in range(int(additional_assets))
    )
    by_seller: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for offer_id, units in sorted(desired.items()):
        offer = environment.offers[offer_id]
        packs = math.ceil(units / max(1, int(offer.package_units)))
        by_seller[offer.seller_id].append(
            {"offer_id": offer_id, "packs": packs}
        )
    for seller_id, items in sorted(by_seller.items()):
        result = environment.place_purchase_order(
            state, seller_id=seller_id, items=items
        )
        if not result.ok:
            raise RuntimeError(f"forced scale-stress order failed: {result.data}")


def _cap_informed_durable_orders(
    environment: OutfitterEnvironment,
    state: Any,
    desired_packs: dict[str, int],
    *,
    priority_offer_ids: set[str],
) -> dict[str, int]:
    """Apply the fleet contract to actual supplier pack units before ordering."""

    capped = {
        offer_id: int(packs)
        for offer_id, packs in desired_packs.items()
        if environment.offers[offer_id].asset_kind != "durable"
    }
    remaining_units = _informed_expansion_capacity(environment, state)
    durable = sorted(
        (
            (offer_id, int(packs))
            for offer_id, packs in desired_packs.items()
            if environment.offers[offer_id].asset_kind == "durable"
            and int(packs) > 0
        ),
        key=lambda row: (
            0 if row[0] in priority_offer_ids else 1,
            row[0],
        ),
    )
    for offer_id, packs in durable:
        units_per_pack = int(environment.offers[offer_id].package_units)
        allowed_packs = min(packs, remaining_units // units_per_pack)
        if allowed_packs <= 0:
            continue
        capped[offer_id] = allowed_packs
        remaining_units -= allowed_packs * units_per_pack
    return capped


def _repair_damaged(
    environment: OutfitterEnvironment, state: Any, *, cash_reserve: float
) -> None:
    repair = []
    for asset in state.assets:
        if asset.status != "damaged":
            continue
        if state.cash - environment.repair_cost(asset) >= cash_reserve:
            repair.append(asset.asset_id)
    if repair:
        environment.repair_or_retire(state, repair, [])


def _durable_fleet_counts(
    environment: OutfitterEnvironment, state: Any
) -> tuple[int, int, int]:
    """Return opening, active, and prepaid in-transit durable asset units."""

    opening = sum(int(asset.acquired_day) == 0 for asset in state.assets)
    active = sum(
        asset.status not in {"retired", "sold"} for asset in state.assets
    )
    in_transit = sum(
        int(item["packs"] * item["package_units"])
        for order in state.orders
        if order.status == "in_transit"
        for item in order.items
        if environment.offers[str(item["offer_id"])].asset_kind == "durable"
    )
    return opening, active, in_transit


def _informed_expansion_capacity(
    environment: OutfitterEnvironment, state: Any
) -> int:
    """Bound active plus incoming fleet, while still allowing retiree replacement."""

    opening, active, in_transit = _durable_fleet_counts(environment, state)
    return max(
        0,
        opening
        + INFORMED_MAX_NET_EXPANSION_ASSETS
        - active
        - in_transit,
    )


def _pairwise_public_context_key(line: Any) -> str:
    """Canonical public context used by the scripted pairwise learner."""

    return json.dumps(
        {
            "service_key": str(
                line.service_key or f"rental:{line.service_line}"
            ),
            "hard_requirements": dict(line.public_hard_requirements),
            "stated_preferences": [dict(row) for row in line.public_preferences],
            "duration_days": int(line.duration_days),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _pairwise_visible_choice_evidence(
    state: Any,
) -> dict[str, Counter[tuple[str, str]]]:
    """Reconstruct only choice evidence available in public quote history.

    A one-option accepted quote identifies acceptance versus outside, but it is
    not a relative exact-offer comparison.  It must therefore not inflate the
    pairwise win rate of the usual primary offer.  Equal-price counters isolate
    ongoing preference probes from subsequent price-treatment observations.
    """

    exposures: Counter[tuple[str, str]] = Counter()
    choices: Counter[tuple[str, str]] = Counter()
    equal_exposures: Counter[tuple[str, str]] = Counter()
    equal_choices: Counter[tuple[str, str]] = Counter()
    equal_context_exposures: Counter[tuple[str, str]] = Counter()
    equal_context_choices: Counter[tuple[str, str]] = Counter()
    for request in state.reservation_requests:
        if (
            request.reservation_contract_mode
            != "adaptive_pairwise_exact_quotes_v1"
            or request.commitment_id is None
        ):
            continue
        for line in request.lines:
            options = list(line.submitted_quote_options)
            if len(options) != 2:
                continue
            service_key = str(line.service_key or f"rental:{line.service_line}")
            context_key = _pairwise_public_context_key(line)
            equal_price = abs(
                float(options[0]["daily_unit_price"])
                - float(options[1]["daily_unit_price"])
            ) < 0.005
            for option in options:
                exposures[(service_key, str(option["offer_id"]))] += 1
                if equal_price:
                    equal_exposures[(service_key, str(option["offer_id"]))] += 1
                    equal_context_exposures[
                        (context_key, str(option["offer_id"]))
                    ] += 1
            if line.selected_quote_option_id is None:
                continue
            selected = next(
                (
                    option
                    for option in options
                    if option["option_id"] == line.selected_quote_option_id
                ),
                None,
            )
            if selected is not None:
                choices[(service_key, str(selected["offer_id"]))] += 1
                if equal_price:
                    equal_choices[(service_key, str(selected["offer_id"]))] += 1
                    equal_context_choices[
                        (context_key, str(selected["offer_id"]))
                    ] += 1
    return {
        "exposures": exposures,
        "choices": choices,
        "equal_exposures": equal_exposures,
        "equal_choices": equal_choices,
        "equal_context_exposures": equal_context_exposures,
        "equal_context_choices": equal_context_choices,
    }


def _pairwise_visible_choice_counts(
    state: Any,
) -> tuple[Counter[tuple[str, str]], Counter[tuple[str, str]]]:
    evidence = _pairwise_visible_choice_evidence(state)
    return evidence["exposures"], evidence["choices"]


def _respond_scripted_profile_reservations(
    environment: OutfitterEnvironment,
    state: Any,
    strategy: str,
    pending: list[Any],
) -> None:
    """Exercise V16 using only facts available through its public surfaces."""

    pairwise_mode = (
        environment.config.reservation_contract_mode
        == "adaptive_pairwise_exact_quotes_v1"
    )
    factorized_plan_mode = (
        environment.config.reservation_contract_mode
        == "factorized_line_options_plan_set_v1"
    )
    line_menu_mode = pairwise_mode or factorized_plan_mode
    complete_contract_mode = (
        environment.config.reservation_contract_mode
        == "complete_contract_options_v1"
    )
    pairwise_policies = {
        "single",
        "random_pair",
        "public_pair",
        "adaptive_pair",
        "oracle_pair",
        "owned_public_pair",
        "owned_random_pair",
        "owned_adaptive_pair",
        "owned_oracle_pair",
        "owned_random_price_pair",
        "owned_public_price_pair",
        "owned_adaptive_price_pair",
        "owned_oracle_price_pair",
        "owned_delayed_random_price_pair",
        "owned_delayed_public_price_pair",
        "owned_matched_equal_price_pair",
        "owned_matched_random_price_pair",
        "owned_matched_public_price_pair",
        "owned_matched_contextual_price_pair",
        "owned_matched_contextual_public_prior_price_pair",
    }
    if line_menu_mode and PAIRWISE_SCRIPTED_MENU_POLICY not in pairwise_policies:
        raise ValueError(
            f"unsupported pairwise scripted policy: {PAIRWISE_SCRIPTED_MENU_POLICY}"
        )
    history_evidence = _pairwise_visible_choice_evidence(state)
    history_exposures = history_evidence["exposures"]
    history_choices = history_evidence["choices"]
    diagnostics = _PROFILE_SCRIPTED_DIAGNOSTICS.setdefault(id(state), Counter())

    if strategy in {"reject_all", "buy_liquidate"}:
        environment.quote_reservation_inquiries(
            state,
            [
                {"request_id": row.request_id, "action": "decline"}
                for row in pending
            ],
        )
        return
    capacity = Counter(
        asset.offer_id
        for asset in state.assets
        if asset.status not in {"retired", "sold"}
    )
    in_transit = [
        (
            environment.estimated_order_delivery_day(order),
            str(item["offer_id"]),
            int(item["packs"] * item["package_units"]),
        )
        for order in state.orders
        if order.status == "in_transit"
        for item in order.items
        if environment.offers[str(item["offer_id"])].asset_kind == "durable"
    ]
    committed: Counter[tuple[str, int]] = Counter()
    for block in state.reservation_calendar_blocks:
        if block.status in {"active", "bound"}:
            committed[(block.offer_id, block.day)] += int(block.quantity)

    for request in sorted(pending, key=lambda row: (row.pickup_day, row.request_id)):
        if line_menu_mode and (
            "adaptive" in PAIRWISE_SCRIPTED_MENU_POLICY
            or "delayed" in PAIRWISE_SCRIPTED_MENU_POLICY
            or "matched" in PAIRWISE_SCRIPTED_MENU_POLICY
        ):
            history_evidence = _pairwise_visible_choice_evidence(state)
            history_exposures = history_evidence["exposures"]
            history_choices = history_evidence["choices"]
        selected_terms: dict[str, dict[str, Any]] = {}
        accept = True

        for line in request.lines:
            option_sheet = environment.view_reservation_options(
                state,
                request_id=request.request_id,
                line_id=line.line_id,
                view="profiles",
                limit=50,
            )
            candidates: list[tuple[int, float, str, str, int, int]] = []
            for profile in option_sheet["profiles"]:
                profile_id = str(profile["promise_profile_id"])
                quantity = int(profile["asset_quantity"])
                member_ids = request.inquiry_profile_member_ids_by_line[
                    line.line_id
                ][profile_id]
                for offer_id in member_ids:
                    peak_needed = max(
                        committed[(offer_id, day)] + quantity
                        for day in range(request.pickup_day, request.return_day)
                    )
                    arriving = sum(
                        units
                        for delivery_day, arriving_offer_id, units in in_transit
                        if arriving_offer_id == offer_id
                        and delivery_day <= request.pickup_day
                    )
                    deficit = max(0, peak_needed - capacity[offer_id] - arriving)
                    offer = environment.offers[offer_id]
                    estimate = environment.supplier_delivery_estimate(
                        offer.seller_id
                    )
                    packs = math.ceil(deficit / offer.package_units)
                    cost = packs * float(offer.package_price)
                    public_procurement = (
                        deficit > 0
                        and state.day + estimate <= request.pickup_day
                        and state.cash - cost
                        >= max(500.0, 7.0 * environment.config.daily_fee)
                    )
                    rank = 0 if deficit == 0 else 1 if public_procurement else 2
                    candidates.append(
                        (
                            rank,
                            float(offer.unit_cost),
                            profile_id,
                            offer_id,
                            quantity,
                            packs,
                        )
                    )
            if not candidates:
                if line.required:
                    accept = False
                continue
            if strategy == "naive":
                selected = min(candidates, key=lambda row: (row[2], row[3]))
            elif strategy in {"fixed_public", "heuristic"}:
                operational = [row for row in candidates if row[0] == 0]
                if not operational:
                    if line.required:
                        accept = False
                    selected = min(candidates, key=lambda row: (row[2], row[3]))
                else:
                    selected = min(
                        operational, key=lambda row: (row[1], row[2], row[3])
                    )
            else:
                selected = min(candidates)
                if selected[0] == 2 and line.required:
                    accept = False
            _rank, _cost, profile_id, offer_id, quantity, packs = selected
            price = state.prices.get(offer_id)
            if price is None:
                cleaning = environment.public_offer(offer_id)[
                    "normal_return_cleaning"
                ]
                price = float(environment.offers[offer_id].unit_cost) / 10.0 + (
                    _public_cleaning_cost(cleaning, 1) / 3.0
                )
            selected_terms[line.line_id] = {
                "profile_id": profile_id,
                "offer_id": offer_id,
                "quantity": quantity,
                "packs": packs,
                "daily_unit_price": round(
                    max(
                        0.01,
                        float(price)
                        * PROFILE_SCRIPTED_QUOTE_SCALE
                        * (
                            1.0
                            + PROFILE_SCRIPTED_LEAD_RISK_PREMIUM
                            * float(request.booking_lead_days)
                            / 14.0
                        ),
                    ),
                    2,
                ),
            }
            quote_options = [
                {
                    "offer_id": offer_id,
                    "profile_id": profile_id,
                    "daily_unit_price": selected_terms[line.line_id][
                        "daily_unit_price"
                    ],
                }
            ]
            if line_menu_mode and (
                factorized_plan_mode
                or PAIRWISE_SCRIPTED_MENU_POLICY != "single"
            ):
                candidates_by_offer: dict[
                    str, tuple[int, float, str, str, int, int]
                ] = {}
                for candidate in candidates:
                    candidate_rank, candidate_cost, _profile, candidate_offer, *_ = (
                        candidate
                    )
                    if candidate_offer == offer_id or candidate_rank > 1:
                        continue
                    if (
                        PAIRWISE_SCRIPTED_MENU_POLICY.startswith("owned_")
                        and candidate_rank != 0
                    ):
                        continue
                    if (
                        max(float(_cost), float(candidate_cost))
                        / min(float(_cost), float(candidate_cost))
                        > 2.0
                    ):
                        continue
                    previous = candidates_by_offer.get(candidate_offer)
                    if previous is None or candidate < previous:
                        candidates_by_offer[candidate_offer] = candidate
                pool = list(candidates_by_offer.values())
                secondary = None
                if pool:
                    if factorized_plan_mode:
                        secondary = min(pool)
                    elif PAIRWISE_SCRIPTED_MENU_POLICY.startswith(
                        "owned_matched_"
                    ):
                        # All matched arms use the identical exact menu.  Only
                        # the later price-treatment rule may differ.
                        secondary = min(
                            pool,
                            key=lambda row: hashlib.sha256(
                                (
                                    f"{request.request_id}|{line.line_id}|{row[3]}"
                                ).encode()
                            ).hexdigest(),
                        )
                    elif "random" in PAIRWISE_SCRIPTED_MENU_POLICY:
                        secondary = min(
                            pool,
                            key=lambda row: hashlib.sha256(
                                (
                                    f"{request.request_id}|{line.line_id}|{row[3]}"
                                ).encode()
                            ).hexdigest(),
                        )
                    elif "public" in PAIRWISE_SCRIPTED_MENU_POLICY:
                        secondary = max(
                            pool,
                            key=lambda row: (
                                float(
                                    environment.offers[row[3]].consumer_online_offer[
                                        "package_price_cny"
                                    ]
                                ),
                                -float(row[1]),
                                row[3],
                            ),
                        )
                    elif "adaptive" in PAIRWISE_SCRIPTED_MENU_POLICY:
                        service_key = str(
                            line.service_key or f"rental:{line.service_line}"
                        )

                        def adaptive_score(
                            row: tuple[int, float, str, str, int, int]
                        ) -> tuple[float, float, str]:
                            key = (service_key, row[3])
                            seen = history_exposures[key]
                            rate = (history_choices[key] + 1.0) / (seen + 2.0)
                            exploration = 0.75 / math.sqrt(seen + 1.0)
                            online = float(
                                environment.offers[row[3]].consumer_online_offer[
                                    "package_price_cny"
                                ]
                            )
                            return rate + exploration, online, row[3]

                        secondary = max(pool, key=adaptive_score)
                    else:
                        secondary = max(
                            pool,
                            key=lambda row: (
                                float(
                                    request.inquiry_profile_price_ceiling_by_line[
                                        line.line_id
                                    ][row[2]].get(row[3], -1.0)
                                ),
                                row[3],
                            ),
                        )
                if secondary is not None:
                    quote_options.append(
                        {
                            "offer_id": secondary[3],
                            "profile_id": secondary[2],
                            "daily_unit_price": selected_terms[line.line_id][
                                "daily_unit_price"
                            ],
                        }
                    )
                    diagnostics["pairwise_two_option_lines_submitted"] += 1
                else:
                    diagnostics["pairwise_no_comparable_second_option"] += 1
            selected_terms[line.line_id]["quote_options"] = quote_options
            selected_terms[line.line_id]["pairwise_prediction_offer_id"] = None
            selected_terms[line.line_id]["pairwise_balanced_probe"] = False
            if (
                line_menu_mode
                and len(quote_options) == 2
                and "price_pair" in PAIRWISE_SCRIPTED_MENU_POLICY
            ):
                service_key = str(line.service_key or f"rental:{line.service_line}")
                first, second = quote_options
                pair_seen = min(
                    history_exposures[(service_key, str(first["offer_id"]))],
                    history_exposures[(service_key, str(second["offer_id"]))],
                )
                preferred_offer_id: str | None
                if PAIRWISE_SCRIPTED_MENU_POLICY.startswith(
                    "owned_matched_"
                ):
                    offer_ids = (
                        str(first["offer_id"]),
                        str(second["offer_id"]),
                    )
                    equal_exposures = history_evidence["equal_exposures"]
                    equal_pair_seen = min(
                        equal_exposures[(service_key, candidate)]
                        for candidate in offer_ids
                    )
                    probe_bucket = int(
                        hashlib.sha256(
                            (
                                f"balanced-probe|{request.request_id}|"
                                f"{line.line_id}"
                            ).encode()
                        ).hexdigest()[:8],
                        16,
                    ) % 5
                    balanced_probe = (
                        "equal" in PAIRWISE_SCRIPTED_MENU_POLICY
                        or equal_pair_seen < 5
                        or probe_bucket == 0
                    )
                    prediction_offer_id: str | None = None
                    if "contextual" in PAIRWISE_SCRIPTED_MENU_POLICY:
                        context_key = _pairwise_public_context_key(line)
                        public_favorite = max(
                            offer_ids,
                            key=lambda candidate: (
                                float(
                                    environment.offers[
                                        candidate
                                    ].consumer_online_offer[
                                        "package_price_cny"
                                    ]
                                ),
                                candidate,
                            ),
                        )
                        use_public_prior = (
                            "public_prior" in PAIRWISE_SCRIPTED_MENU_POLICY
                        )

                        def contextual_score(
                            candidate: str,
                        ) -> tuple[float, int, float, str]:
                            global_seen = equal_exposures[
                                (service_key, candidate)
                            ]
                            prior_strength = 6.0 if use_public_prior else 2.0
                            prior_rate = (
                                (0.68 if candidate == public_favorite else 0.32)
                                if use_public_prior
                                else 0.5
                            )
                            global_rate = (
                                history_evidence["equal_choices"][
                                    (service_key, candidate)
                                ]
                                + prior_strength * prior_rate
                            ) / (global_seen + prior_strength)
                            local_seen = history_evidence[
                                "equal_context_exposures"
                            ][(context_key, candidate)]
                            local_strength = 6.0 if use_public_prior else 3.0
                            local_rate = (
                                history_evidence["equal_context_choices"][
                                    (context_key, candidate)
                                ]
                                + local_strength * global_rate
                            ) / (local_seen + local_strength)
                            online = float(
                                environment.offers[
                                    candidate
                                ].consumer_online_offer["package_price_cny"]
                            )
                            return local_rate, global_seen, online, candidate

                        prediction_offer_id = max(
                            offer_ids, key=contextual_score
                        )
                    elif "public" in PAIRWISE_SCRIPTED_MENU_POLICY:
                        prediction_offer_id = max(
                            offer_ids,
                            key=lambda candidate: (
                                float(
                                    environment.offers[
                                        candidate
                                    ].consumer_online_offer[
                                        "package_price_cny"
                                    ]
                                ),
                                candidate,
                            ),
                        )
                    elif "random" in PAIRWISE_SCRIPTED_MENU_POLICY:
                        prediction_offer_id = min(
                            offer_ids,
                            key=lambda candidate: hashlib.sha256(
                                (
                                    f"price|{request.request_id}|"
                                    f"{line.line_id}|{candidate}"
                                ).encode()
                            ).hexdigest(),
                        )
                    selected_terms[line.line_id][
                        "pairwise_prediction_offer_id"
                    ] = prediction_offer_id
                    selected_terms[line.line_id][
                        "pairwise_balanced_probe"
                    ] = balanced_probe
                    if balanced_probe:
                        preferred_offer_id = None
                        diagnostics["pairwise_balanced_probe_lines"] += 1
                    else:
                        preferred_offer_id = prediction_offer_id
                    if preferred_offer_id is not None:
                        diagnostics["pairwise_matched_exploit_lines"] += 1
                elif (
                    "delayed" in PAIRWISE_SCRIPTED_MENU_POLICY
                    and pair_seen < 3
                ):
                    preferred_offer_id = None
                elif "random" in PAIRWISE_SCRIPTED_MENU_POLICY:
                    preferred_offer_id = min(
                        (str(first["offer_id"]), str(second["offer_id"])),
                        key=lambda candidate: hashlib.sha256(
                            (
                                f"price|{request.request_id}|{line.line_id}|"
                                f"{candidate}"
                            ).encode()
                        ).hexdigest(),
                    )
                elif "public" in PAIRWISE_SCRIPTED_MENU_POLICY:
                    preferred_offer_id = max(
                        (str(first["offer_id"]), str(second["offer_id"])),
                        key=lambda candidate: (
                            float(
                                environment.offers[
                                    candidate
                                ].consumer_online_offer["package_price_cny"]
                            ),
                            candidate,
                        ),
                    )
                elif "adaptive" in PAIRWISE_SCRIPTED_MENU_POLICY:
                    preferred_offer_id = (
                        max(
                            (str(first["offer_id"]), str(second["offer_id"])),
                            key=lambda candidate: (
                                (
                                    history_choices[(service_key, candidate)] + 1.0
                                )
                                / (
                                    history_exposures[(service_key, candidate)]
                                    + 2.0
                                ),
                                candidate,
                            ),
                        )
                        if pair_seen >= 3
                        else None
                    )
                else:
                    ceilings_by_profile = (
                        request.inquiry_profile_price_ceiling_by_line[
                            line.line_id
                        ]
                    )
                    preferred_offer_id = max(
                        (str(first["offer_id"]), str(second["offer_id"])),
                        key=lambda candidate: max(
                            (
                                float(ceilings.get(candidate, -1.0))
                                for ceilings in ceilings_by_profile.values()
                            ),
                            default=-1.0,
                        ),
                    )
                if preferred_offer_id is not None:
                    base_price = float(
                        selected_terms[line.line_id]["daily_unit_price"]
                    )
                    for option in quote_options:
                        multiplier = (
                            1.10
                            if option["offer_id"] == preferred_offer_id
                            else 0.90
                        )
                        option["daily_unit_price"] = round(
                            max(0.01, base_price * multiplier), 2
                        )
                    diagnostics["pairwise_price_tilt_lines"] += 1

        if accept and complete_contract_mode:
            full_contract_lines = [
                {
                    "line_id": line.line_id,
                    "disposition": "quote",
                    "offer_id": selected_terms[line.line_id]["offer_id"],
                    "promise_profile_id": selected_terms[line.line_id][
                        "profile_id"
                    ],
                    "daily_unit_price": selected_terms[line.line_id][
                        "daily_unit_price"
                    ],
                }
                for line in request.lines
            ]
            has_optional = any(not line.required for line in request.lines)
            alternative_lines = []
            for line in request.lines:
                if not line.required:
                    alternative_lines.append(
                        {"line_id": line.line_id, "disposition": "omit"}
                    )
                    continue
                term = selected_terms[line.line_id]
                alternative_lines.append(
                    {
                        "line_id": line.line_id,
                        "disposition": "quote",
                        "offer_id": term["offer_id"],
                        "promise_profile_id": term["profile_id"],
                        "daily_unit_price": (
                            term["daily_unit_price"]
                            if has_optional
                            else round(
                                max(
                                    0.01,
                                    float(term["daily_unit_price"]) + 0.01,
                                    1.10 * float(term["daily_unit_price"]),
                                ),
                                2,
                            )
                        ),
                    }
                )
            decision = {
                "request_id": request.request_id,
                "action": "quote",
                "contract_options": [
                    {"lines": full_contract_lines},
                    {"lines": alternative_lines},
                ],
            }
        elif accept and factorized_plan_mode:
            decision = {
                "request_id": request.request_id,
                "action": "quote",
                "lines": [
                    {
                        "line_id": line.line_id,
                        "options": [
                            {
                                "disposition": "quote",
                                "offer_id": option["offer_id"],
                                "promise_profile_id": option["profile_id"],
                                "daily_unit_price": option[
                                    "daily_unit_price"
                                ],
                            }
                            for option in selected_terms[line.line_id][
                                "quote_options"
                            ]
                        ]
                        + (
                            [{"disposition": "omit"}]
                            if not line.required
                            else []
                        ),
                    }
                    for line in request.lines
                ],
            }
        else:
            decision = {
            "request_id": request.request_id,
            "action": "quote" if accept else "decline",
            **(
                {
                    "lines": [
                        (
                            {
                                "line_id": line.line_id,
                                "quote_options": [
                                    {
                                        "offer_id": option["offer_id"],
                                        "promise_profile_id": option[
                                            "profile_id"
                                        ],
                                        "daily_unit_price": option[
                                            "daily_unit_price"
                                        ],
                                    }
                                    for option in selected_terms[line.line_id][
                                        "quote_options"
                                    ]
                                ],
                            }
                            if pairwise_mode
                            else {
                                "line_id": line.line_id,
                                "promise_profile_id": selected_terms[line.line_id][
                                    "profile_id"
                                ],
                                "daily_unit_price": selected_terms[line.line_id][
                                    "daily_unit_price"
                                ],
                            }
                        )
                        for line in request.lines
                    ]
                }
                if accept
                else {}
            ),
            }
        if (
            accept
            and not line_menu_mode
            and PROFILE_SCRIPTED_OPERATIONAL_SUBSET_MODE == "anchor_only"
        ):
            for line in request.lines:
                term = selected_terms[line.line_id]
                request.inquiry_profile_member_ids_by_line[line.line_id][
                    str(term["profile_id"])
                ] = [str(term["offer_id"])]
        response = environment.quote_reservation_inquiries(state, [decision])[
            "decisions"
        ][0]
        if (
            factorized_plan_mode
            and response.get("status") == "awaiting_manager_confirmation"
        ):
            plans = list(response["customer_authorized_plans"])
            if strategy == "naive":
                selected_plan_id = str(response["customer_preferred_plan_id"])
            else:
                def plan_score(plan: dict[str, Any]) -> tuple[Any, ...]:
                    uncovered = 0
                    procurement_cost = 0.0
                    for plan_line in plan["lines"]:
                        if plan_line["disposition"] == "omit":
                            continue
                        offer_id = str(plan_line["offer_id"])
                        quantity = int(plan_line["asset_quantity"])
                        peak_needed = max(
                            committed[(offer_id, day)] + quantity
                            for day in range(request.pickup_day, request.return_day)
                        )
                        arriving = sum(
                            units
                            for delivery_day, arriving_offer_id, units in in_transit
                            if arriving_offer_id == offer_id
                            and delivery_day <= request.pickup_day
                        )
                        deficit = max(
                            0, peak_needed - capacity[offer_id] - arriving
                        )
                        uncovered += deficit
                        if deficit:
                            offer = environment.offers[offer_id]
                            procurement_cost += math.ceil(
                                deficit / offer.package_units
                            ) * float(offer.package_price)
                    return (
                        uncovered,
                        procurement_cost,
                        -float(plan["total_rent"]),
                        str(plan["plan_id"]),
                    )

                selected_plan_id = str(min(plans, key=plan_score)["plan_id"])
            response = environment.confirm_reservation_plans(
                state,
                [
                    {
                        "request_id": request.request_id,
                        "plan_id": selected_plan_id,
                    }
                ],
            )["confirmations"][0]
        if response.get("status") != "accepted":
            continue
        commitment = next(
            row
            for row in state.reservation_commitments
            if row.commitment_id == response["commitment_id"]
        )
        diagnostics = _PROFILE_SCRIPTED_DIAGNOSTICS.setdefault(
            id(state), Counter()
        )
        for committed_line in commitment.lines:
            authorized = list(committed_line.authorized_offer_ids)
            quantity = int(committed_line.quantity)
            diagnostics["confirmed_lines"] += 1
            planned_offer_id = (
                str(committed_line.offer_id)
                if complete_contract_mode or factorized_plan_mode
                else str(selected_terms[committed_line.line_id]["offer_id"])
            )
            if pairwise_mode and committed_line.offer_id != planned_offer_id:
                diagnostics["pairwise_customer_selected_secondary"] += 1
            prediction_offer_id = selected_terms[committed_line.line_id].get(
                "pairwise_prediction_offer_id"
            )
            if pairwise_mode and prediction_offer_id is not None:
                diagnostics["pairwise_prediction_visible_choices"] += 1
                prediction_correct = (
                    committed_line.offer_id == prediction_offer_id
                )
                diagnostics["pairwise_prediction_correct"] += int(
                    prediction_correct
                )
                phase = (
                    "probe"
                    if selected_terms[committed_line.line_id].get(
                        "pairwise_balanced_probe", False
                    )
                    else "exploit"
                )
                diagnostics[
                    f"pairwise_{phase}_prediction_visible_choices"
                ] += 1
                diagnostics[f"pairwise_{phase}_prediction_correct"] += int(
                    prediction_correct
                )
            if (
                not pairwise_mode
                and not complete_contract_mode
                and not factorized_plan_mode
                and planned_offer_id not in authorized
            ):
                diagnostics["planned_offer_not_authorized"] += 1
                _PROFILE_SCRIPTED_MISSED_COMMITMENTS.setdefault(
                    id(state), set()
                ).add(commitment.commitment_id)

            def deficit(offer_id: str) -> int:
                peak = max(
                    committed[(offer_id, day)] + quantity
                    for day in range(commitment.pickup_day, commitment.return_day)
                )
                arriving = sum(
                    units
                    for delivery_day, arriving_offer_id, units in in_transit
                    if arriving_offer_id == offer_id
                    and delivery_day <= commitment.pickup_day
                )
                return max(0, peak - capacity[offer_id] - arriving)

            available = [offer_id for offer_id in authorized if deficit(offer_id) == 0]
            selected_offer_id = min(
                available,
                key=lambda offer_id: (
                    float(environment.offers[offer_id].unit_cost), offer_id
                ),
                default=None,
            )
            if selected_offer_id is None and strategy in {
                "informed_public",
                "scale_winner",
                "aggressive_expansion_reference",
            }:
                procurement = []
                for offer_id in authorized:
                    missing = deficit(offer_id)
                    offer = environment.offers[offer_id]
                    packs = math.ceil(missing / offer.package_units)
                    units = packs * int(offer.package_units)
                    cost = packs * float(offer.package_price)
                    estimate = environment.supplier_delivery_estimate(offer.seller_id)
                    evidence_pass = (
                        strategy in {"scale_winner", "aggressive_expansion_reference"}
                        or (
                            len(state.daily_history) >= 28
                            and _durable_purchase_npv(
                                environment, state, offer, "informed_public"
                            )["npv"]
                            > 0
                        )
                    )
                    expansion_pass = (
                        strategy != "informed_public"
                        or units <= _informed_expansion_capacity(environment, state)
                    )
                    if (
                        missing > 0
                        and evidence_pass
                        and expansion_pass
                        and state.day + estimate <= commitment.pickup_day
                        and state.cash - cost
                        >= max(500.0, 7.0 * environment.config.daily_fee)
                    ):
                        procurement.append((cost, offer_id, packs, units, estimate))
                if procurement:
                    _cost, offer_id, packs, units, estimate = min(procurement)
                    offer = environment.offers[offer_id]
                    placed = _scripted_place_purchase_order(
                        environment,
                        state,
                        seller_id=offer.seller_id,
                        items=[{"offer_id": offer_id, "packs": packs}],
                    )
                    if placed.ok:
                        selected_offer_id = offer_id
                        in_transit.append(
                            (state.day + estimate, offer_id, units)
                        )
            selected_offer_id = selected_offer_id or committed_line.offer_id
            if (
                not pairwise_mode
                and not complete_contract_mode
                and not factorized_plan_mode
                and state.day < commitment.pickup_day
            ):
                environment.stage_reservation_line(
                    state,
                    commitment_id=commitment.commitment_id,
                    line_id=committed_line.line_id,
                    authorized_offer_id=selected_offer_id,
                )
            for day in range(commitment.pickup_day, commitment.return_day):
                committed[(selected_offer_id, day)] += quantity


def _proactively_cancel_scripted_conflicts(
    environment: OutfitterEnvironment,
    state: Any,
    *,
    policy: str,
) -> None:
    """Use only public exact-offer assets, deliveries, and calendar blocks."""

    if (
        policy == "none"
        or not environment._v29_reservation_accountability()
        or state.done
    ):
        return

    def capacity(offer_id: str, day: int) -> int:
        owned = sum(
            asset.offer_id == offer_id
            and asset.status not in {"retired", "sold"}
            for asset in state.assets
        )
        arriving = sum(
            int(item["packs"]) * int(item["package_units"])
            for order in state.orders
            if order.status == "in_transit"
            and environment.estimated_order_delivery_day(order) <= day
            for item in order.items
            if str(item["offer_id"]) == offer_id
        )
        return owned + arriving

    def conflicts() -> set[str]:
        occupancy: Counter[tuple[str, int]] = Counter()
        owners: dict[tuple[str, int], set[str]] = defaultdict(set)
        for commitment in state.reservation_commitments:
            if commitment.status != "confirmed" or commitment.assignments:
                continue
            for line in commitment.lines:
                if not line.required:
                    continue
                offer_id = str(line.selected_offer_id or line.offer_id)
                for day in range(commitment.pickup_day, commitment.return_day):
                    occupancy[(offer_id, day)] += int(line.quantity)
                    owners.setdefault((offer_id, day), set()).add(
                        commitment.commitment_id
                    )
        result: set[str] = set()
        for (offer_id, day), units in occupancy.items():
            if units > capacity(offer_id, day):
                result.update(owners[(offer_id, day)])
        return result

    while True:
        implicated = conflicts()
        if not implicated:
            return
        candidates = [
            commitment
            for commitment in state.reservation_commitments
            if commitment.commitment_id in implicated
            and commitment.status == "confirmed"
        ]
        if not candidates:
            return

        def cancellation_rank(commitment: Any) -> tuple[Any, ...]:
            service_days = max(
                1,
                sum(
                    int(line.service_units or line.customer_units)
                    * int(line.duration_days)
                    for line in commitment.lines
                    if line.required
                ),
            )
            contribution = float(commitment.total_locked_rent) / service_days
            # The intentionally abusive arm preserves the most advance notice;
            # the selective arm first drops the weakest economic commitment.
            if policy == "low_price_cancel":
                return (
                    commitment.pickup_day,
                    contribution,
                    commitment.commitment_id,
                )
            return (
                contribution,
                -max(0, commitment.pickup_day - state.day),
                commitment.commitment_id,
            )

        chosen = min(candidates, key=cancellation_rank)
        environment.cancel_reservation_commitment(
            state, commitment_id=chosen.commitment_id
        )
        if state.done:
            return


def _respond_scripted_reservations(
    environment: OutfitterEnvironment,
    state: Any,
    strategy: str,
    *,
    reservation_change_policy: str = "default",
    reservation_commitment_policy: str = "none",
) -> None:
    """Make baseline decisions by reconciling only public raw calendar facts."""

    if not environment.config.reservations_enabled:
        return
    _proactively_cancel_scripted_conflicts(
        environment,
        state,
        policy=reservation_commitment_policy,
    )
    pending = [
        row
        for row in state.reservation_requests
        if row.status == "pending_quote" and row.created_day == state.day
    ]
    changes = [
        row
        for row in state.reservation_change_requests
        if row.status == "pending" and row.created_day == state.day
    ]
    if changes:
        def change_action(row: Any) -> str:
            if row.change_type == "reschedule":
                if reservation_change_policy == "accept_all":
                    return "accept"
                if reservation_change_policy == "accept_first":
                    request_count = sum(
                        candidate.commitment_id == row.commitment_id
                        and candidate.change_type == "reschedule"
                        for candidate in state.reservation_change_requests
                    )
                    return "accept" if request_count == 1 else "decline"
                if reservation_change_policy == "reject_all":
                    return "decline"
            return (
                "accept"
                if strategy
                in {
                    "naive",
                    "informed_public",
                    "scale_winner",
                    "aggressive_expansion_reference",
                }
                or row.change_type == "reschedule"
                else "decline"
            )

        environment.respond_reservation_changes(
            state,
            [
                {
                    "change_id": row.change_id,
                    "action": change_action(row),
                }
                for row in changes
            ],
        )
    risks = [
        row
        for row in state.reservation_commitments
        if row.status == "at_risk"
    ]
    for commitment in risks:
        if strategy == "naive":
            continue
        if strategy in {"informed_public", "scale_winner", "aggressive_expansion_reference"}:
            inquiry = environment.resolve_reservation_risks(
                state,
                [
                    {
                        "commitment_id": commitment.commitment_id,
                        "action": "inquire_fallback",
                    }
                ],
            )
            inquiry_result = inquiry["actions"][0]
            options = inquiry_result.get(
                "authorized_line_fallbacks",
                inquiry_result.get("authorized_fallbacks", []),
            )
            available = Counter(
                asset.offer_id
                for asset in state.assets
                if asset.status == "available"
            )
            complete_recovery = None
            for plan in inquiry_result.get(
                "preauthorized_recovery_plans", []
            ):
                remaining = Counter(available)
                feasible = True
                for line in plan.get("lines", []):
                    if line.get("disposition") == "omit" or not bool(
                        line.get("required")
                    ):
                        continue
                    offer_id = str(line["offer_id"])
                    quantity = int(line["asset_quantity"])
                    if remaining[offer_id] < quantity:
                        feasible = False
                        break
                    remaining[offer_id] -= quantity
                if feasible:
                    complete_recovery = plan
                    break
            if complete_recovery is not None:
                environment.resolve_reservation_risks(
                    state,
                    [
                        {
                            "commitment_id": commitment.commitment_id,
                            "action": "select_preauthorized_plan",
                            "plan_id": complete_recovery["plan_id"],
                        }
                    ],
                )
                continue
            line_quantity = {
                line.line_id: line.quantity for line in commitment.lines
            }
            if (
                environment.config.reservation_contract_mode
                in {
                    "promise_profiles_v1",
                    "adaptive_pairwise_exact_quotes_v1",
                    "complete_contract_options_v1",
                    "factorized_line_options_plan_set_v1",
                }
            ):
                options_by_line: dict[str, list[dict[str, Any]]] = {}
                for option in options:
                    options_by_line.setdefault(
                        str(option.get("line_id")), []
                    ).append(option)
                for line in commitment.lines:
                    if not line.required:
                        continue
                    normal_candidates = (
                        [line.selected_offer_id]
                        if line.selected_offer_id is not None
                        else list(line.authorized_offer_ids or [line.offer_id])
                    )
                    if any(
                        available[offer_id] >= line.quantity
                        for offer_id in normal_candidates
                    ):
                        continue
                    selected_for_line = next(
                        (
                            row
                            for row in options_by_line.get(line.line_id, [])
                            if available[str(row["offer_id"])]
                            >= line.quantity
                        ),
                        None,
                    )
                    if selected_for_line is not None:
                        environment.resolve_reservation_risks(
                            state,
                            [
                                {
                                    "commitment_id": commitment.commitment_id,
                                    "action": "select_authorized_fallback",
                                    "offer_id": selected_for_line["offer_id"],
                                    "line_id": line.line_id,
                                }
                            ],
                        )
                possible = all(
                    not line.required
                    or any(
                        available[offer_id] >= line.quantity
                        for offer_id in (
                            [line.selected_offer_id]
                            if line.selected_offer_id is not None
                            else list(
                                line.authorized_offer_ids or [line.offer_id]
                            )
                        )
                    )
                    for line in commitment.lines
                )
                if possible:
                    continue
                if environment._v29_reservation_accountability():
                    environment.cancel_reservation_commitment(
                        state, commitment_id=commitment.commitment_id
                    )
                else:
                    environment.resolve_reservation_risks(
                        state,
                        [
                            {
                                "commitment_id": commitment.commitment_id,
                                "action": "cancel_early",
                            }
                        ],
                    )
                continue
            selected = next(
                (
                    row
                    for row in options
                    if available[str(row["offer_id"])]
                    >= line_quantity.get(str(row.get("line_id")), 1)
                ),
                None,
            )
            if selected is not None:
                environment.resolve_reservation_risks(
                    state,
                    [
                        {
                            "commitment_id": commitment.commitment_id,
                            "action": "select_authorized_fallback",
                            "offer_id": selected["offer_id"],
                            "line_id": selected.get("line_id"),
                        }
                    ],
                )
                continue
        if environment._v29_reservation_accountability():
            environment.cancel_reservation_commitment(
                state, commitment_id=commitment.commitment_id
            )
        else:
            environment.resolve_reservation_risks(
                state,
                [
                    {
                        "commitment_id": commitment.commitment_id,
                        "action": "cancel_early",
                    }
                ],
            )
    if not pending:
        return
    if environment.config.reservation_contract_mode in {
        "promise_profiles_v1",
        "adaptive_pairwise_exact_quotes_v1",
        "complete_contract_options_v1",
        "factorized_line_options_plan_set_v1",
    }:
        _respond_scripted_profile_reservations(
            environment, state, strategy, pending
        )
        return
    if strategy in {"reject_all", "buy_liquidate"}:
        environment.quote_reservation_inquiries(
            state,
            [
                {"request_id": row.request_id, "action": "decline"}
                for row in pending
            ],
        )
        return
    if strategy == "naive":
        environment.quote_reservation_inquiries(
            state,
            [
                {
                    "request_id": row.request_id,
                    "action": "quote",
                    "line_prices": [
                        {
                            "line_id": line.line_id,
                            "daily_unit_price": float(
                                state.prices[line.offer_id]
                            ),
                        }
                        for line in row.lines
                    ],
                }
                for row in pending
            ],
        )
        return
    capacity = Counter(
        asset.offer_id
        for asset in state.assets
        if asset.status not in {"retired", "sold"}
    )
    in_transit = [
        (
            environment.estimated_order_delivery_day(order),
            str(item["offer_id"]),
            int(item["packs"] * item["package_units"]),
        )
        for order in state.orders
        if order.status == "in_transit"
        for item in order.items
        if environment.offers[str(item["offer_id"])].asset_kind == "durable"
    ]
    committed: Counter[tuple[str, int]] = Counter()
    for block in state.reservation_calendar_blocks:
        if block.status in {"active", "bound"}:
            committed[(block.offer_id, block.day)] += int(block.quantity)
    for request in sorted(pending, key=lambda row: (row.pickup_day, row.request_id)):
        # Quote sequentially so a rejected customer quote cannot trigger a
        # speculative asset purchase or consume calendar capacity.
        accept = True
        provisional_committed = committed.copy()
        provisional_in_transit = list(in_transit)
        provisional_cost = 0.0
        provisional_expansion_units = 0
        for line in request.lines:
            candidate_ids = list(line.candidate_offer_ids or [line.offer_id])

            def candidate_deficit(offer_id: str) -> int:
                peak_needed = max(
                    provisional_committed[(offer_id, day)] + line.quantity
                    for day in range(request.pickup_day, request.return_day)
                )
                arriving = sum(
                    units
                    for delivery_day, arriving_offer_id, units in provisional_in_transit
                    if arriving_offer_id == offer_id
                    and delivery_day <= request.pickup_day
                )
                return max(0, peak_needed - capacity[offer_id] - arriving)

            selected_offer_id = next(
                (
                    offer_id
                    for offer_id in candidate_ids
                    if candidate_deficit(offer_id) == 0
                ),
                None,
            )
            if (
                selected_offer_id is None
                and line.required
                and strategy
                in {"informed_public", "scale_winner", "aggressive_expansion_reference"}
            ):
                procurement_options = []
                for offer_id in candidate_ids:
                    deficit = candidate_deficit(offer_id)
                    offer = environment.offers[offer_id]
                    npv_strategy = (
                        "aggressive_expansion_reference"
                        if strategy == "aggressive_expansion_reference"
                        else "informed_public"
                    )
                    evidence_pass = (
                        strategy == "scale_winner"
                        or (
                            len(state.daily_history) >= 28
                            and _durable_purchase_npv(
                                environment, state, offer, npv_strategy
                            )["npv"]
                            > 0
                        )
                    )
                    estimate = environment.supplier_delivery_estimate(
                        offer.seller_id
                    )
                    packs = math.ceil(deficit / offer.package_units)
                    units = packs * int(offer.package_units)
                    cost = packs * float(offer.package_price)
                    expansion_pass = (
                        strategy != "informed_public"
                        or provisional_expansion_units + units
                        <= _informed_expansion_capacity(environment, state)
                    )
                    if (
                        evidence_pass
                        and state.day + estimate <= request.pickup_day
                        and state.cash - provisional_cost - cost
                        >= max(500.0, 7.0 * environment.config.daily_fee)
                        and expansion_pass
                    ):
                        procurement_options.append(
                            (
                                cost,
                                candidate_ids.index(offer_id),
                                offer_id,
                                units,
                                estimate,
                            )
                        )
                if procurement_options:
                    cost, _order, offer_id, units, estimate = min(
                        procurement_options
                    )
                    provisional_cost += cost
                    provisional_expansion_units += units
                    provisional_in_transit.append(
                        (state.day + estimate, offer_id, units)
                    )
                    selected_offer_id = offer_id
            if selected_offer_id is None:
                if line.required:
                    accept = False
                continue
            for day in range(request.pickup_day, request.return_day):
                provisional_committed[(selected_offer_id, day)] += line.quantity

        decision = {
            "request_id": request.request_id,
            "action": "quote" if accept else "decline",
            **(
                {
                    "line_prices": [
                        {
                            "line_id": line.line_id,
                            "daily_unit_price": float(
                                state.prices[line.offer_id]
                            ),
                        }
                        for line in request.lines
                    ]
                }
                if accept
                else {}
            ),
        }
        response = environment.quote_reservation_inquiries(state, [decision])[
            "decisions"
        ][0]
        if response.get("status") != "accepted":
            continue

        commitment = next(
            row
            for row in state.reservation_commitments
            if row.commitment_id == response["commitment_id"]
        )
        for line in commitment.lines:
            candidate_ids = list(line.authorized_offer_ids or [line.offer_id])

            def accepted_candidate_deficit(offer_id: str) -> int:
                peak_needed = max(
                    committed[(offer_id, day)] + line.quantity
                    for day in range(commitment.pickup_day, commitment.return_day)
                )
                arriving = sum(
                    units
                    for delivery_day, arriving_offer_id, units in in_transit
                    if arriving_offer_id == offer_id
                    and delivery_day <= commitment.pickup_day
                )
                return max(0, peak_needed - capacity[offer_id] - arriving)

            selected_offer_id = next(
                (
                    offer_id
                    for offer_id in candidate_ids
                    if accepted_candidate_deficit(offer_id) == 0
                ),
                None,
            )
            if selected_offer_id is None:
                procurement_options = []
                for offer_id in candidate_ids:
                    deficit = accepted_candidate_deficit(offer_id)
                    offer = environment.offers[offer_id]
                    estimate = environment.supplier_delivery_estimate(
                        offer.seller_id
                    )
                    packs = math.ceil(deficit / offer.package_units)
                    units = packs * int(offer.package_units)
                    cost = packs * float(offer.package_price)
                    evidence_pass = (
                        strategy in {"scale_winner", "aggressive_expansion_reference"}
                        or (
                            len(state.daily_history) >= 28
                            and _durable_purchase_npv(
                                environment,
                                state,
                                offer,
                                "informed_public",
                            )["npv"]
                            > 0
                        )
                    )
                    expansion_pass = (
                        strategy != "informed_public"
                        or units
                        <= _informed_expansion_capacity(environment, state)
                    )
                    if (
                        evidence_pass
                        and state.day + estimate <= commitment.pickup_day
                        and state.cash - cost
                        >= max(500.0, 7.0 * environment.config.daily_fee)
                        and expansion_pass
                    ):
                        procurement_options.append(
                            (cost, candidate_ids.index(offer_id), offer_id, packs)
                        )
                if procurement_options:
                    _cost, _order, offer_id, packs = min(procurement_options)
                    offer = environment.offers[offer_id]
                    placed = _scripted_place_purchase_order(
                        environment,
                        state,
                        seller_id=offer.seller_id,
                        items=[{"offer_id": offer_id, "packs": packs}],
                    )
                    if placed.ok:
                        units = packs * int(offer.package_units)
                        in_transit.append(
                            (
                                state.day
                                + environment.supplier_delivery_estimate(
                                    offer.seller_id
                                ),
                                offer_id,
                                units,
                            )
                        )
                        selected_offer_id = offer_id
            # If procurement became infeasible only after acceptance, keep the
            # primary reserved here and let the ordinary risk path resolve it.
            selected_offer_id = selected_offer_id or line.offer_id
            for day in range(commitment.pickup_day, commitment.return_day):
                committed[(selected_offer_id, day)] += line.quantity


def _set_scripted_walk_in_capacity_limits(
    environment: OutfitterEnvironment,
    state: Any,
    strategy: str,
    *,
    reservation_commitment_policy: str = "none",
) -> None:
    """Exercise V24's public capacity policy without hidden feasibility."""

    if (
        not environment._v24_reservation_operations()
        or strategy
        not in {"heuristic", "informed_public", "scale_winner", "aggressive_expansion_reference"}
        and reservation_commitment_policy == "none"
    ):
        return
    total_assets = Counter(
        asset.offer_id
        for asset in state.assets
        if asset.status not in {"retired", "sold"}
    )
    committed: Counter[tuple[str, int]] = Counter()
    for block in state.reservation_calendar_blocks:
        if block.status not in {"active", "bound"} or block.day < state.day:
            continue
        committed[(block.offer_id, int(block.day))] += int(block.quantity)
    desired: dict[tuple[str, int], int] = {}
    for (offer_id, day), quantity in committed.items():
        desired[(offer_id, day)] = max(
            0, int(total_assets[offer_id]) - int(quantity)
        )
    # A return on pickup morning joins cleaning only after that morning's
    # cleaning work. The informed baseline deliberately protects three prior
    # occupied days as its own conservative service buffer; the environment
    # itself still infers no cleaning dates from the limit.
    for commitment in state.reservation_commitments:
        if commitment.status not in {"confirmed", "at_risk"}:
            continue
        for buffer_day in range(
            max(state.day, int(commitment.pickup_day) - 3),
            int(commitment.pickup_day),
        ):
            for line in commitment.lines:
                key = (line.offer_id, buffer_day)
                desired[key] = min(
                    desired.get(key, 10_000),
                    max(
                        0,
                        int(total_assets[line.offer_id])
                        - int(line.quantity),
                    ),
                )
    existing = {
        (row.offer_id, int(row.day)): int(row.max_walk_in_occupied_units)
        for row in state.walk_in_capacity_limits
        if int(row.day) >= state.day
    }
    point_actions: list[tuple[str, str, int, int | None]] = []
    for key in sorted(set(existing) | set(desired)):
        if key in desired and existing.get(key) != desired[key]:
            point_actions.append(("set", key[0], key[1], desired[key]))
        elif key in existing and key not in desired:
            point_actions.append(("clear", key[0], key[1], None))
    grouped: list[dict[str, Any]] = []
    for action in ("clear", "set"):
        offers = sorted(
            {row[1] for row in point_actions if row[0] == action}
        )
        for offer_id in offers:
            values = sorted(
                (day, maximum)
                for row_action, row_offer, day, maximum in point_actions
                if row_action == action and row_offer == offer_id
            )
            start = previous = values[0][0]
            maximum = values[0][1]
            for day, value in values[1:] + [(values[-1][0] + 2, None)]:
                if day == previous + 1 and value == maximum:
                    previous = day
                    continue
                grouped.append(
                    {
                        "action": action,
                        "offer_id": offer_id,
                        "from_day": start,
                        "to_day": previous,
                        **(
                            {"max_walk_in_occupied_units": int(maximum)}
                            if action == "set"
                            else {}
                        ),
                    }
                )
                start = previous = day
                maximum = value
    for offset in range(0, len(grouped), 50):
        result = environment.set_walk_in_capacity_limits(
            state, grouped[offset : offset + 50]
        )
        if not result.ok:
            raise AssertionError(
                f"scripted walk-in capacity policy failed: {result.data}"
            )


def _schedule_scripted_maintenance(
    environment: OutfitterEnvironment, state: Any, strategy: str
) -> None:
    if (
        not environment.config.preventive_maintenance_enabled
        or strategy in {"naive", "buy_liquidate", "reject_all"}
    ):
        return
    maintenance_horizon = (
        max(environment.lifecycle_scales.maintenance_duration_days_by_severity_tier.values())
        if bool(getattr(environment.config, "lifecycle_economics_enabled", False))
        and environment.lifecycle_scales is not None
        else int(environment.config.maintenance_duration_days)
    )
    blocked_offers = {
        offer_id
        for block in state.reservation_calendar_blocks
        for offer_id in (block.authorized_offer_ids or [block.offer_id])
        if block.status in {"active", "bound"}
        and state.day
        <= block.day
        <= state.day + maintenance_horizon
    }
    candidates = [
        asset.asset_id
        for asset in state.assets
        if asset.status in {"available", "maintenance_hold"}
        and (
            bool(environment._lifecycle_maintenance_status(asset)["window_open"])
            if bool(getattr(environment.config, "lifecycle_economics_enabled", False))
            else asset.next_inspection_due_day is not None
            and state.day
            >= asset.next_inspection_due_day
            - int(environment.config.maintenance_due_window_days)
        )
        and asset.offer_id not in blocked_offers
    ]
    if candidates:
        environment.schedule_maintenance(state, candidates)


def _recent_group_demand(state: Any, horizon: int) -> dict[str, int]:
    start = max(1, state.day - horizon)
    values: dict[str, int] = defaultdict(int)
    for report in state.daily_history:
        if int(report["day"]) < start:
            continue
        for row in report.get("demand_cube", []):
            values[str(row["choice_group"])] += int(row["requested_units"])
    return values


def _trailing_group_utilization(
    environment: OutfitterEnvironment,
    state: Any,
    service_key: str,
    group: str,
    *,
    horizon: int = 14,
) -> float | None:
    reports = state.daily_history[-horizon:]
    if not reports:
        return None
    offer_ids = {
        offer.offer_id for offer in environment.by_group[(service_key, group)]
    }
    rental_days = 0
    for report in reports:
        metric = report.get("service_metrics", {}).get(service_key, {})
        rental_days += sum(
            int(row.get("rental_days", 0))
            for offer_id, row in metric.get(
                "offer_fulfillment_breakdown", {}
            ).items()
            if offer_id in offer_ids
        )
    active_assets = sum(
        asset.offer_id in offer_ids and asset.status not in {"retired", "sold"}
        for asset in state.assets
    )
    if active_assets <= 0:
        return None
    return rental_days / (len(reports) * active_assets)


def _recent_offer_metrics(state: Any, horizon: int = 14) -> dict[str, Counter[str]]:
    values: dict[str, Counter[str]] = defaultdict(Counter)
    for report in state.daily_history[-horizon:]:
        for metric in report.get("service_metrics", {}).values():
            for offer_id, row in metric.get(
                "offer_fulfillment_breakdown", {}
            ).items():
                target = values[str(offer_id)]
                for field in (
                    "fulfilled_units",
                    "rental_days",
                    "quote_exposures",
                    "quoted_units",
                ):
                    target[field] += int(row.get(field, 0))
                target["revenue_cents"] += round(
                    100 * float(row.get("revenue", 0.0))
                )
                target["retail_cogs_cents"] += round(
                    100 * float(row.get("retail_cogs", 0.0))
                )
    return values


def _trailing_offer_utilization(state: Any, offer_id: str, horizon: int = 14) -> float | None:
    reports = state.daily_history[-horizon:]
    if not reports:
        return None
    metrics = _recent_offer_metrics(state, horizon).get(offer_id, Counter())
    active = sum(
        asset.offer_id == offer_id and asset.status not in {"retired", "sold"}
        for asset in state.assets
    )
    if active <= 0:
        return None
    return int(metrics.get("rental_days", 0)) / (len(reports) * active)


def _forecast_offer_sales_rate(
    environment: OutfitterEnvironment, state: Any, offer_id: str
) -> float:
    reports = state.daily_history[-60:]
    if not reports:
        return 0.0
    totals: defaultdict[str, int] = defaultdict(int)
    day_counts: Counter[str] = Counter()
    for report in reports:
        weather = str(report["weather"])
        day_counts[weather] += 1
        totals[weather] += sum(
            int(values.get("fulfilled_units", 0))
            for metric in report.get("service_metrics", {}).values()
            for candidate_id, values in metric.get(
                "offer_fulfillment_breakdown", {}
            ).items()
            if candidate_id == offer_id
        )
    forecast = environment.weather_forecast(
        state, min(7, state.max_days - state.day + 1)
    )
    estimates = [
        totals[str(row["condition"])] / day_counts[str(row["condition"])]
        for row in forecast
        if day_counts[str(row["condition"])] > 0
    ]
    if estimates:
        return sum(estimates) / len(estimates)
    return sum(totals.values()) / len(reports)


def _terminal_rate_for_remaining(remaining: float) -> float:
    for _grade, threshold, rate in SECONDHAND_GRADE_SCHEDULE:
        if remaining >= threshold:
            return float(rate)
    return float(SECONDHAND_GRADE_SCHEDULE[-1][2])


def _durable_purchase_npv(
    environment: OutfitterEnvironment,
    state: Any,
    offer: Any,
    strategy: str,
) -> dict[str, float]:
    """Conservative, observable short-window estimate for one new rental asset."""

    if strategy in LEGACY_REFERENCE_ALIASES:
        strategy = "aggressive_expansion_reference"
    lifecycle_enabled = bool(
        getattr(environment.config, "lifecycle_economics_enabled", False)
    )

    seller_id = str(getattr(offer, "seller_id", ""))
    supplier_delivery_days = (
        int(environment.supplier_delivery_estimate(seller_id))
        if seller_id and hasattr(environment, "supplier_delivery_estimate")
        else int(environment.config.delivery_days)
    )
    remaining_service_days = max(
        0,
        int(state.max_days - state.day - supplier_delivery_days + 1),
    )
    service_key = (
        str(offer.fulfills_service_keys[0])
        if strategy == "aggressive_expansion_reference"
        else ""
    )
    if strategy == "aggressive_expansion_reference":
        default_duration = (
            2
            if service_key
            in {"rental:tent", "rental:sleeping_bag", "rental:sleeping_pad"}
            else 1
        )
    else:
        public_metrics = _recent_offer_metrics(state, 28).get(
            offer.offer_id, Counter()
        )
        fulfilled = int(public_metrics.get("fulfilled_units", 0))
        default_duration = max(
            1,
            round(int(public_metrics.get("rental_days", 0)) / fulfilled),
        ) if fulfilled else 1
    if strategy in {"heuristic", "informed_public"}:
        observed = _trailing_offer_utilization(state, offer.offer_id)
        utilization = (
            0.0
            if observed is None
            else min(0.40 if strategy == "heuristic" else 0.45, observed)
        )
    elif strategy == "aggressive_expansion_reference":
        neutral = float(
            environment.neutral_demand_price(
                state,
                offer.offer_id,
                duration_days=default_duration,
            )
            or 0.0
        )
        price = float(state.prices[offer.offer_id])
        utilization = 0.0 if neutral <= 0 or price >= 1.5 * neutral else min(
            0.55, max(0.05, 0.35 * (1.5 - price / neutral))
        )
    else:
        raise ValueError(f"unsupported calibration strategy: {strategy}")
    if lifecycle_enabled:
        public_profile = environment.public_offer(offer.offer_id)[
            "lifecycle_profile"
        ]
        life_days = float(public_profile["service_life_limit_days"])
        life_cycles = float(public_profile["service_life_limit_cycles"])
        maximum_contracts = 1.0 / (
            1.0 / life_cycles + default_duration / life_days
        )
        maximum_billed_days = default_duration * maximum_contracts
    elif strategy == "aggressive_expansion_reference":
        life_days = float(offer.service_life_limit_days or remaining_service_days)
        life_cycles = float(
            getattr(offer, "service_life_limit_cycles", None) or 1_000_000_000
        )
        maximum_contracts = 1.0 / (
            1.0 / life_cycles + default_duration / life_days
        )
        maximum_billed_days = default_duration * maximum_contracts
    else:
        # Public scripted baselines do not receive hidden lifecycle priors and do
        # not extrapolate a short utilization streak across the full season.
        planning_days = min(PUBLIC_DURABLE_PLANNING_DAYS, remaining_service_days)
        life_days = float(max(1, planning_days))
        life_cycles = 1_000_000_000.0
        maximum_billed_days = float(planning_days)
    expected_billed_days = min(
        (
            float(remaining_service_days)
            if strategy == "aggressive_expansion_reference"
            else maximum_billed_days
        )
        * utilization,
        maximum_billed_days,
    )
    expected_contracts = expected_billed_days / default_duration
    cleaning_rule = environment.public_offer(offer.offer_id)[
        "normal_return_cleaning"
    ]
    cleaning_per_contract = _public_cleaning_cost(
        cleaning_rule, default_duration
    )
    rental_revenue = expected_billed_days * float(state.prices[offer.offer_id])
    cleaning_cost = expected_contracts * cleaning_per_contract
    if lifecycle_enabled:
        assert environment.lifecycle_scales is not None
        public_profile = environment.public_offer(offer.offer_id)[
            "lifecycle_profile"
        ]
        public_storage_units = {
            "low": 0.5,
            "medium": 1.0,
            "high": 2.0,
        }[str(public_profile["storage_burden_tier"])]
        holding_cost = remaining_service_days * (
            float(offer.unit_cost)
            * environment.lifecycle_scales.annual_capital_rate
            / 365.0
            + (1.0 - utilization)
            * public_storage_units
            * environment.lifecycle_scales.daily_space_rate_per_storage_unit
        )
    else:
        holding_cost = (
            remaining_service_days
            * (
                float(getattr(environment.config, "rental_holding_rate", 0.0005))
                + float(
                    getattr(environment.config, "rental_idle_holding_rate", 0.0)
                )
                * (1.0 - utilization)
            )
            * float(offer.unit_cost)
        )
    if lifecycle_enabled:
        assert environment.lifecycle_scales is not None
        used_wear = min(
            1.0,
            expected_contracts / life_cycles
            + expected_billed_days / life_days,
        )
        terminal_rate = (
            environment.lifecycle_scales.new_asset_residual_rate
            * max(0.0, 1.0 - used_wear)
            ** environment.lifecycle_scales.residual_wear_exponent
        )
    elif strategy == "aggressive_expansion_reference":
        used_wear = expected_contracts / life_cycles + expected_billed_days / life_days
        terminal_rate = _terminal_rate_for_remaining(max(0.0, 1.0 - used_wear))
    else:
        terminal_rate = 0.10
    terminal_value = terminal_rate * float(offer.unit_cost)
    acquisition_cost = float(offer.unit_cost)
    npv = (
        rental_revenue
        - cleaning_cost
        - holding_cost
        + terminal_value
        - acquisition_cost
    )
    return {
        "remaining_service_days": float(remaining_service_days),
        "expected_utilization": round(utilization, 6),
        "expected_duration_days": float(default_duration),
        "expected_billed_days": round(expected_billed_days, 6),
        "expected_contracts": round(expected_contracts, 6),
        "rental_revenue": round(rental_revenue, 6),
        "cleaning_cost": round(cleaning_cost, 6),
        "holding_cost": round(holding_cost, 6),
        "terminal_value": round(terminal_value, 6),
        "acquisition_cost": round(acquisition_cost, 6),
        "npv": round(npv, 6),
    }


def _adjust_public_prices(
    environment: OutfitterEnvironment, state: Any, *, informed: bool = False
) -> None:
    interval = 7 if informed else 14
    if informed and state.day > 29:
        return
    if not state.daily_history or state.day % interval != 1:
        return
    by_offer = _recent_offer_metrics(state, 28)
    updates = []
    for offer_id, counts in sorted(by_offer.items()):
        exposures = int(counts["quote_exposures"])
        minimum_exposures = 5 if informed else 20
        if exposures < minimum_exposures:
            continue
        conversion = int(counts["fulfilled_units"]) / exposures
        if informed:
            multiplier = (
                0.75
                if conversion < 0.30
                else 1.12
                if conversion > 0.70
                else 1.0
            )
        else:
            multiplier = 0.95 if conversion < 0.20 else 1.05 if conversion > 0.80 else 1.0
        if multiplier == 1.0:
            continue
        updates.append(
            {
                "offer_id": offer_id,
                "price": round(max(0.01, state.prices[offer_id] * multiplier), 2),
            }
        )
    if updates:
        environment.set_prices(state, updates)


def _apply_oracle_margin_floors(
    environment: OutfitterEnvironment, state: Any
) -> None:
    updates = []
    for offer in environment.dataset.offers:
        if offer.asset_kind != "consumable":
            continue
        floor = round(offer.unit_cost * 1.25, 2)
        if state.prices[offer.offer_id] + 1e-9 < floor:
            updates.append({"offer_id": offer.offer_id, "price": floor})
    if updates:
        environment.set_prices(state, updates)


def _apply_oracle_reservation_prices(
    environment: OutfitterEnvironment, state: Any
) -> None:
    if (
        environment.demand_model is None
        and environment.reference_demand_model is None
    ) or state.day % 7 != 1:
        return
    updates = []
    for offer in environment.dataset.offers:
        service_key = str(offer.fulfills_service_keys[0])
        duration = (
            2
            if offer.asset_kind == "durable"
            and service_key
            in {"rental:tent", "rental:sleeping_bag", "rental:sleeping_pad"}
            else 1
        )
        neutral = environment.neutral_demand_price(
            state, offer.offer_id, duration_days=duration
        )
        if neutral is None:
            continue
        floor = (
            offer.unit_cost
            * (
                environment.demand_model.retail_margin_floor
                if environment.demand_model is not None
                else 1.25
            )
            if offer.asset_kind == "consumable"
            else 0.01
        )
        target = max(floor, 0.90 * neutral)
        if abs(float(state.prices[offer.offer_id]) - target) >= 0.01:
            updates.append(
                {"offer_id": offer.offer_id, "price": round(target, 2)}
            )
    if updates:
        environment.set_prices(state, updates)


def _retail_pack_projection(
    state: Any,
    offer: Any,
    metrics: Counter[str],
    *,
    horizon_days: int = 14,
) -> tuple[float, float]:
    observed_days = max(1, min(horizon_days, len(state.daily_history)))
    sales_rate = int(metrics.get("fulfilled_units", 0)) / observed_days
    selling_days = max(
        0,
        min(horizon_days, state.max_days - state.day - 1),
    )
    expected_sold = min(float(offer.package_units), sales_rate * selling_days)
    expected_left = float(offer.package_units) - expected_sold
    return expected_sold, expected_left


def _retail_pack_expected_value(
    environment: OutfitterEnvironment,
    state: Any,
    offer: Any,
    metrics: Counter[str],
    *,
    horizon_days: int = 14,
) -> float:
    expected_sold, expected_left = _retail_pack_projection(
        state, offer, metrics, horizon_days=horizon_days
    )
    return (
        expected_sold * float(state.prices[offer.offer_id])
        + expected_left
        * environment.retail_liquidation_unit_quote(float(offer.unit_cost))
        - float(offer.package_price)
    )


def _retail_reorder_rounding_tolerance(
    state: Any,
    offer: Any,
    metrics: Counter[str],
    *,
    horizon_days: int = 14,
) -> float:
    """Return the pack-level tie band induced by cent-quantized unit quotes."""

    _, expected_left = _retail_pack_projection(
        state, offer, metrics, horizon_days=horizon_days
    )
    return expected_left * RETAIL_REORDER_QUOTE_ROUNDING_TOLERANCE_PER_UNIT


def _public_experiment_order(
    environment: OutfitterEnvironment, state: Any
) -> dict[str, int]:
    if (
        state.day <= INFORMED_EXPERIMENT_INTERVAL_DAYS
        or state.day % INFORMED_EXPERIMENT_INTERVAL_DAYS != 1
        or any(order.status == "in_transit" for order in state.orders)
        or state.max_days - state.day + 1
        < INFORMED_EXPERIMENT_INTERVAL_DAYS
    ):
        return {}
    block = (state.day - 1) // INFORMED_EXPERIMENT_INTERVAL_DAYS
    target_kind = "durable" if block % 2 else "consumable"
    carried = set(state.ever_carried_offer_ids)
    budget = 0.03 * float(state.cash)
    public_kind = "rental_asset" if target_kind == "durable" else "retail_item"
    durable_capacity = (
        _informed_expansion_capacity(environment, state)
        if target_kind == "durable"
        else math.inf
    )
    candidates = [
        public
        for offer_id in sorted(environment.offers)
        if (public := environment.public_offer(offer_id))["kind"] == public_kind
        and offer_id not in carried
        and float(public["package_price"]) <= budget
        and int(environment.offers[offer_id].package_units) <= durable_capacity
    ]
    if not candidates:
        return {}
    chosen = min(
        candidates,
        key=lambda offer: (float(offer["package_price"]), offer["offer_id"]),
    )
    return {str(chosen["offer_id"]): 1}


def _last_asset_acquisition_day(state: Any, offer_id: str) -> int | None:
    days = [
        int(event["day"])
        for event in state.rental_activity
        if event.get("event_type") == "asset_acquired"
        and str(event.get("offer_id")) == offer_id
    ]
    return max(days) if days else None


def _reschedule_request_tail_diagnostics(state: Any) -> dict[str, Any]:
    counts_by_commitment = Counter(
        change.commitment_id
        for change in state.reservation_change_requests
        if change.change_type == "reschedule"
    )
    request_by_id = {
        request.request_id: request for request in state.reservation_requests
    }

    def summarize(commitments: list[Any]) -> dict[str, Any]:
        counts = [
            int(counts_by_commitment.get(commitment.commitment_id, 0))
            for commitment in commitments
        ]
        total = len(counts)
        return {
            "commitments": total,
            "request_count_distribution": dict(sorted(Counter(counts).items())),
            "at_least_one": sum(value >= 1 for value in counts),
            "at_least_two": sum(value >= 2 for value in counts),
            "at_least_three": sum(value >= 3 for value in counts),
            "p_at_least_one": (
                round(sum(value >= 1 for value in counts) / total, 6)
                if total
                else None
            ),
            "p_at_least_two": (
                round(sum(value >= 2 for value in counts) / total, 6)
                if total
                else None
            ),
            "p_at_least_three": (
                round(sum(value >= 3 for value in counts) / total, 6)
                if total
                else None
            ),
            "maximum": max(counts, default=0),
        }

    result = summarize(list(state.reservation_commitments))
    result["by_booking_lead"] = {
        label: summarize(
            [
                commitment
                for commitment in state.reservation_commitments
                if low
                <= int(request_by_id[commitment.request_id].booking_lead_days)
                <= high
            ]
        )
        for label, (low, high) in {
            "1-3": (1, 3),
            "4-7": (4, 7),
            "8-10": (8, 10),
            "11-14": (11, 14),
        }.items()
    }
    return result


def run_policy(
    config: OutfitterConfig,
    strategy: str,
    *,
    reservation_change_policy: str = "default",
    scripted_fleet_expansion_cap: int | None = None,
    reservation_commitment_policy: str = "none",
    scripted_opening_durable_units_per_offer: int = 0,
    scripted_day1_forced_durable_assets: int = 0,
) -> dict[str, Any]:
    if strategy in LEGACY_REFERENCE_ALIASES:
        strategy = "aggressive_expansion_reference"
    if strategy not in ALL_STRATEGIES:
        raise ValueError(f"strategy must be one of {ALL_STRATEGIES}")
    if reservation_change_policy not in {
        "default",
        "accept_all",
        "accept_first",
        "reject_all",
    }:
        raise ValueError(
            "reservation_change_policy must be default, accept_all, "
            "accept_first, or reject_all"
        )
    if reservation_commitment_policy not in {
        "none",
        "selective_conflict_cancel",
        "low_price_cancel",
    }:
        raise ValueError(
            "reservation_commitment_policy must be none, "
            "selective_conflict_cancel, or low_price_cancel"
        )
    if (
        scripted_fleet_expansion_cap is not None
        and (
            isinstance(scripted_fleet_expansion_cap, bool)
            or int(scripted_fleet_expansion_cap) < 0
        )
    ):
        raise ValueError("scripted_fleet_expansion_cap must be non-negative")
    if (
        isinstance(scripted_opening_durable_units_per_offer, bool)
        or int(scripted_opening_durable_units_per_offer) < 0
    ):
        raise ValueError(
            "scripted_opening_durable_units_per_offer must be non-negative"
        )
    if (
        isinstance(scripted_day1_forced_durable_assets, bool)
        or int(scripted_day1_forced_durable_assets) < 0
    ):
        raise ValueError(
            "scripted_day1_forced_durable_assets must be non-negative"
        )
    dataset = OutfitterDataset.load(config.data_dir)
    if dataset.metadata.get("environment_protocol") == V32_SERVICE_PROTOCOL:
        raise ValueError(
            "Legacy calibration merchants do not implement V32 fixed offers and date choices; "
            "use python -m outfitter.run_scripted_merchant instead."
        )
    dataset.require_weather_coverage(
        config.max_days + config.runoff_days,
        start_day=config.weather_start_day,
    )
    environment = OutfitterEnvironment(config, dataset)
    environment._scripted_fleet_expansion_cap = (
        None
        if scripted_fleet_expansion_cap is None
        else int(scripted_fleet_expansion_cap)
    )
    state = environment.reset()
    _PROFILE_SCRIPTED_DIAGNOSTICS.pop(id(state), None)
    _PROFILE_SCRIPTED_MISSED_COMMITMENTS.pop(id(state), None)
    initial_values = environment.business_value(state)
    initial_value = initial_values["official_liquidation_value"]
    initial_contract_value = initial_values["contract_adjusted_economic_value"]
    opening_offer_state = _initial_offer_state(environment, state)
    opening_asset_ids = {asset.asset_id for asset in state.assets}
    _post_missing_opening_prices(environment, state, strategy)
    initial_retail_target = {
        row["offer_id"]: max(2, int(row["quantity"]))
        for row in environment.dataset.initial_retail
        if int(row["quantity"]) > 0
    }
    initial_asset_target = Counter(asset.offer_id for asset in state.assets)
    _seed_scripted_opening_fleet(
        environment,
        state,
        units_per_offer=int(scripted_opening_durable_units_per_offer),
    )
    _order_scripted_forced_expansion(
        environment,
        state,
        additional_assets=int(scripted_day1_forced_durable_assets),
    )
    cash_reserve = max(500.0, 7.0 * config.daily_fee)

    while not state.done:
        if state.day > config.max_days:
            environment.close_business_day(state)
            continue
        _respond_scripted_reservations(
            environment,
            state,
            strategy,
            reservation_change_policy=reservation_change_policy,
            reservation_commitment_policy=reservation_commitment_policy,
        )
        _set_scripted_walk_in_capacity_limits(
            environment,
            state,
            strategy,
            reservation_commitment_policy=reservation_commitment_policy,
        )
        _schedule_scripted_maintenance(environment, state, strategy)
        if strategy == "reject_all":
            environment.close_business_day(state)
            continue
        if strategy == "buy_liquidate":
            _buy_liquidate_actions(
                environment,
                state,
                opening_asset_ids=opening_asset_ids,
            )
            environment.close_business_day(state)
            continue
        _repair_damaged(environment, state, cash_reserve=cash_reserve)
        if strategy in {"heuristic", "informed_public"}:
            _adjust_public_prices(
                environment, state, informed=strategy == "informed_public"
            )
        if strategy == "scale_winner":
            _adjust_scale_winner_prices(environment, state)
        if strategy == "aggressive_expansion_reference":
            _apply_oracle_margin_floors(environment, state)
            _apply_oracle_reservation_prices(environment, state)
        transit = _in_transit_units(state)
        desired_packs: dict[str, int] = (
            _public_experiment_order(environment, state)
            if strategy == "informed_public"
            else {}
        )
        recent_offer = _recent_offer_metrics(state, 14)

        if strategy in (*FORMAL_STRATEGIES, "fixed_public", "scale_winner"):
            if strategy == "naive":
                retail_offer_ids = set(initial_retail_target)
            else:
                retail_offer_ids = {
                    offer_id
                    for offer_id in state.ever_carried_offer_ids
                    if environment.public_offer(offer_id)["kind"] == "retail_item"
                }
            for offer_id in sorted(retail_offer_ids):
                offer = environment.offers[offer_id]
                current = int(
                    state.retail_inventory.get(offer_id, {}).get("quantity", 0)
                ) + transit.get(offer_id, 0)
                if strategy == "naive":
                    target = initial_retail_target.get(offer_id, 0)
                else:
                    metrics = recent_offer.get(offer_id, Counter())
                    fulfilled = int(metrics.get("fulfilled_units", 0))
                    if fulfilled < (2 if strategy == "heuristic" else 3):
                        continue
                    if strategy == "heuristic":
                        sales_rate = fulfilled / max(
                            1, min(14, len(state.daily_history))
                        )
                        target = max(2, math.ceil(5 * sales_rate))
                    else:
                        sales_rate = _forecast_offer_sales_rate(
                            environment, state, offer_id
                        )
                        target = max(2, math.ceil(7 * sales_rate))
                    expected_value = _retail_pack_expected_value(
                        environment, state, offer, metrics
                    )
                    rounding_tolerance = _retail_reorder_rounding_tolerance(
                        state, offer, metrics
                    )
                    # Within the maximum half-cent quote quantization band,
                    # preserve the carried SKU instead of flipping a discrete
                    # reorder because of monetary representation alone.
                    if expected_value < -rounding_tolerance:
                        continue
                if target <= 0 or current > math.ceil(target / 2):
                    continue
                packs = max(1, math.ceil((target - current) / offer.package_units))
                if strategy != "naive":
                    packs = 1
                desired_packs[offer_id] = max(
                    desired_packs.get(offer_id, 0), packs
                )
        else:
            recent = _recent_group_demand(state, 14)
            for (service_key, group), offers in sorted(environment.by_group.items()):
                if not service_key.startswith("retail:"):
                    continue
                candidates = list(offers)
                if not candidates:
                    continue
                chosen = max(
                    candidates,
                    key=lambda offer: (
                        float(
                            environment.neutral_demand_price(
                                state, offer.offer_id
                            )
                            or 0.0
                        )
                        - offer.unit_cost,
                        -offer.unit_cost,
                        offer.offer_id,
                    ),
                )
                target = max(3, math.ceil(recent.get(group, 0) / 14 * 5))
                current = int(
                    state.retail_inventory.get(chosen.offer_id, {}).get("quantity", 0)
                ) + transit.get(chosen.offer_id, 0)
                if current <= math.ceil(target / 2):
                    desired_packs[chosen.offer_id] = max(
                        1,
                        math.ceil((target - current) / chosen.package_units),
                    )

        durable_targets: dict[str, int] = {}
        if strategy in {"heuristic", "informed_public", "scale_winner"}:
            planned_net_expansion = sum(
                int(packs) * int(environment.offers[offer_id].package_units)
                for offer_id, packs in desired_packs.items()
                if environment.offers[offer_id].asset_kind == "durable"
            )
            carried_durable_ids = {
                offer_id
                for offer_id in state.ever_carried_offer_ids
                if environment.public_offer(offer_id)["kind"] == "rental_asset"
            }
            for offer_id in sorted(carried_durable_ids):
                offer = environment.offers[offer_id]
                active = sum(
                    asset.offer_id == offer_id
                    and asset.status not in {"retired", "sold"}
                    for asset in state.assets
                ) + transit.get(offer_id, 0)
                opening_target = initial_asset_target.get(offer_id, 0)
                was_acquired = _last_asset_acquisition_day(state, offer_id) is not None
                replacement_target = opening_target or int(was_acquired)
                target = replacement_target
                observed = _trailing_offer_utilization(state, offer_id)
                metrics = _recent_offer_metrics(state, 28).get(
                    offer_id, Counter()
                )
                evidence_ready = (
                    len(state.daily_history) >= (14 if strategy == "scale_winner" else 28)
                    and int(metrics.get("quote_exposures", 0)) >= 8
                    and int(metrics.get("fulfilled_units", 0)) >= 4
                    and int(metrics.get("rental_days", 0)) >= 4
                )
                if (
                    strategy in {"informed_public", "scale_winner"}
                    and evidence_ready
                    and observed is not None
                    and observed > (0.35 if strategy == "scale_winner" else 0.40)
                    and active >= target
                ):
                    last_acquired = _last_asset_acquisition_day(state, offer_id)
                    cooldown_ready = (
                        last_acquired is None
                        or state.day - last_acquired
                        >= (
                            7
                            if strategy == "scale_winner"
                            else INFORMED_DURABLE_COOLDOWN_DAYS
                        )
                    )
                    maximum = (
                        math.inf
                        if strategy == "scale_winner"
                        else max(1, opening_target) + 2
                    )
                    opening_fleet = sum(initial_asset_target.values())
                    active_fleet = sum(
                        asset.status not in {"retired", "sold"}
                        for asset in state.assets
                    ) + sum(
                        units
                        for candidate_id, units in transit.items()
                        if environment.offers[candidate_id].asset_kind == "durable"
                    )
                    expansion_capacity = (
                        math.inf
                        if strategy == "scale_winner"
                        else max(
                            0,
                            opening_fleet
                            + INFORMED_MAX_NET_EXPANSION_ASSETS
                            - active_fleet
                            - planned_net_expansion,
                        )
                    )
                    if (
                        cooldown_ready
                        and active < maximum
                        and expansion_capacity > 0
                    ):
                        target = active + 1
                        planned_net_expansion += 1
                if (
                    evidence_ready
                    and active < target
                    and _durable_purchase_npv(
                        environment,
                        state,
                        offer,
                        "informed_public"
                        if strategy == "scale_winner"
                        else strategy,
                    )["npv"] > 0
                ):
                    durable_targets[offer_id] = target
        elif strategy == "aggressive_expansion_reference":
            target_per_group = int(AGGRESSIVE_EXPANSION_TARGET_PER_GROUP)
            for (service_key, _group), offers in sorted(environment.by_group.items()):
                if not service_key.startswith("rental:"):
                    continue
                evaluated = [
                    (
                        _durable_purchase_npv(environment, state, offer, strategy),
                        offer,
                    )
                    for offer in offers
                ]
                estimate, chosen = max(
                    evaluated,
                    key=lambda item: (item[0]["npv"], -item[1].unit_cost, item[1].offer_id),
                )
                if estimate["npv"] > 0:
                    group_offer_ids = {offer.offer_id for offer in offers}
                    active_in_group = sum(
                        asset.offer_id in group_offer_ids
                        and asset.status not in {"retired", "sold"}
                        for asset in state.assets
                    ) + sum(transit.get(offer_id, 0) for offer_id in group_offer_ids)
                    additional = max(0, target_per_group - active_in_group)
                    if additional:
                        chosen_active = sum(
                            asset.offer_id == chosen.offer_id
                            and asset.status not in {"retired", "sold"}
                            for asset in state.assets
                        ) + transit.get(chosen.offer_id, 0)
                        durable_targets[chosen.offer_id] = chosen_active + additional
        for offer_id, desired in sorted(durable_targets.items()):
            active = sum(
                asset.offer_id == offer_id and asset.status not in {"retired", "sold"}
                for asset in state.assets
            ) + transit.get(offer_id, 0)
            if active >= desired:
                continue
            offer = environment.offers[offer_id]
            desired_packs[offer_id] = max(
                desired_packs.get(offer_id, 0),
                math.ceil((desired - active) / offer.package_units),
            )
        if strategy == "informed_public":
            desired_packs = _cap_informed_durable_orders(
                environment,
                state,
                desired_packs,
                priority_offer_ids=set(durable_targets),
            )
        _place_affordable_orders(
            environment, state, desired_packs, cash_reserve=cash_reserve
        )
        if not state.done:
            environment.close_business_day(state)

    values = environment.business_value(state)
    margins = _retail_gross_margin(environment, state)
    positive_margin = sum(max(0.0, value) for value in margins.values())
    concentration = {
        offer_id: round(max(0.0, margin) / positive_margin, 6)
        for offer_id, margin in margins.items()
        if positive_margin > 0
    }
    high_concentration = sorted(
        offer_id for offer_id, share in concentration.items() if share > 0.40
    )
    purchased_rental_by_offer = Counter(
        asset.offer_id for asset in state.assets if int(asset.acquired_day) > 0
    )
    purchased_rental_by_product = Counter(
        asset.service_line for asset in state.assets if int(asset.acquired_day) > 0
    )
    performance_curve = _daily_performance_curve(
        state.daily_history,
        initial_value=initial_value,
        initial_contract_value=initial_contract_value,
    )
    official_value_change = round(
        values["official_liquidation_value"] - initial_value, 2
    )
    offer_economics, value_bridge = _offer_economic_bridge(
        environment,
        state,
        opening_offer_state,
        official_value_change=official_value_change,
    )
    internal_channel_counts = Counter(
        str(row.get("selected_channel") or "unknown")
        for report in state.daily_history
        for row in report.get("internal_channel_outcomes", [])
    )
    basket_intent_status_counts = Counter(
        str(row.get("status") or "unknown")
        for report in state.daily_history
        for row in report.get("internal_basket_intents", [])
    )
    lifecycle_event_counts = Counter(
        str(row.get("event_type") or "unknown")
        for row in state.rental_activity
    )
    lifecycle_assets = [
        asset for asset in state.assets if asset.status != "sold"
    ]
    opening_active_assets = int(sum(initial_asset_target.values()))
    ending_active_assets = sum(
        asset.status not in {"retired", "sold"} for asset in state.assets
    )
    maximum_active_assets = max(
        [opening_active_assets]
        + [
            sum(
                int(count)
                for status, count in report.get("end", {})
                .get("asset_status_counts", {})
                .items()
                if status not in {"retired", "sold"}
            )
            for report in state.daily_history
        ]
    )
    lifecycle_diagnostics = (
        {
            "enabled": True,
            "event_counts": dict(sorted(lifecycle_event_counts.items())),
            "damage_event_count": int(
                lifecycle_event_counts.get("returned_damaged", 0)
            ),
            "condition_loss_total": round(
                sum(
                    float(row.get("condition_loss", 0.0))
                    for row in state.rental_activity
                    if row.get("event_type") == "returned_damaged"
                ),
                6,
            ),
            "uncovered_repair_cost_total": round(
                sum(
                    float(row.get("uncovered_repair_cost", 0.0))
                    for row in state.rental_activity
                    if row.get("event_type") == "returned_damaged"
                ),
                2,
            ),
            "unsold_asset_count": len(lifecycle_assets),
            "opening_active_asset_count": opening_active_assets,
            "ending_active_asset_count": ending_active_assets,
            "maximum_active_asset_count": maximum_active_assets,
            "maximum_net_active_expansion": max(
                0, maximum_active_assets - opening_active_assets
            ),
            "condition_ratio_minimum": round(
                min((asset.condition_ratio for asset in lifecycle_assets), default=1.0),
                6,
            ),
            "condition_ratio_mean": round(
                sum(asset.condition_ratio for asset in lifecycle_assets)
                / max(1, len(lifecycle_assets)),
                6,
            ),
            "combined_wear_maximum": round(
                max(
                    (environment._combined_wear(asset) for asset in lifecycle_assets),
                    default=0.0,
                ),
                6,
            ),
            "capital_holding_cost": round(
                state.cumulative_rental_capital_holding_cost, 2
            ),
            "space_holding_cost": round(
                state.cumulative_rental_space_holding_cost, 2
            ),
        }
        if config.lifecycle_economics_enabled
        else {"enabled": False}
    )
    payback_diagnostics = _lifecycle_payback_diagnostics(
        environment,
        state,
        completed_days=len(state.daily_history),
    )
    scripted_profile_diagnostics = dict(
        sorted(_PROFILE_SCRIPTED_DIAGNOSTICS.pop(id(state), Counter()).items())
    )
    missed_commitments = _PROFILE_SCRIPTED_MISSED_COMMITMENTS.pop(
        id(state), set()
    )
    scripted_profile_diagnostics.update(
        {
            "confirmed_requests_with_planned_offer_miss": len(
                missed_commitments
            ),
            "manager_cancellations_with_planned_offer_miss": sum(
                commitment.commitment_id in missed_commitments
                and commitment.status == "cancelled_by_manager"
                for commitment in state.reservation_commitments
            ),
        }
    )
    pairwise_periods = []
    if config.reservation_contract_mode == "adaptive_pairwise_exact_quotes_v1":
        midpoint = max(1, int(config.max_days) // 2)
        for label, start_day, end_day in (
            ("early", 1, midpoint),
            ("late", midpoint + 1, int(config.max_days)),
        ):
            period_requests = [
                request
                for request in state.reservation_requests
                if start_day <= request.created_day <= end_day
            ]
            period_lines = [
                line for request in period_requests for line in request.lines
            ]
            pair_lines = [
                line
                for line in period_lines
                if len(line.submitted_quote_options) == 2
            ]
            tilted_lines = [
                line
                for line in pair_lines
                if len(
                    {
                        float(option["daily_unit_price"])
                        for option in line.submitted_quote_options
                    }
                )
                > 1
            ]
            selected_offers = Counter()
            for line in pair_lines:
                if line.selected_quote_option_id is None:
                    continue
                selected = next(
                    (
                        option
                        for option in line.submitted_quote_options
                        if option["option_id"]
                        == line.selected_quote_option_id
                    ),
                    None,
                )
                if selected is not None:
                    selected_offers[str(selected["offer_id"])] += 1
            spreads = [
                round(
                    max(
                        float(option["daily_unit_price"])
                        for option in line.submitted_quote_options
                    )
                    - min(
                        float(option["daily_unit_price"])
                        for option in line.submitted_quote_options
                    ),
                    2,
                )
                for line in pair_lines
            ]
            pairwise_periods.append(
                {
                    "period": label,
                    "start_day": start_day,
                    "end_day": end_day,
                    "requests": len(period_requests),
                    "quoted_requests": sum(
                        request.quote_action == "quote"
                        for request in period_requests
                    ),
                    "accepted_requests": sum(
                        request.commitment_id is not None
                        for request in period_requests
                    ),
                    "customer_rejected_requests": sum(
                        request.status == "customer_declined_quote"
                        for request in period_requests
                    ),
                    "submitted_lines": sum(
                        bool(line.submitted_quote_options)
                        for line in period_lines
                    ),
                    "two_option_lines": len(pair_lines),
                    "tilted_price_lines": len(tilted_lines),
                    "selected_pair_lines": sum(selected_offers.values()),
                    "selected_offer_counts": dict(
                        sorted(selected_offers.items())
                    ),
                    "mean_pair_price_spread": (
                        round(statistics.mean(spreads), 2)
                        if spreads
                        else None
                    ),
                }
            )
    recognized_reservation_prepayments = round(
        sum(
            float(
                report.get("financial", {}).get(
                    "reservation_prepayment_recognized", 0.0
                )
            )
            for report in state.daily_history
        ),
        2,
    )
    calibration_cash_inflows = (
        float(state.cumulative_rental_revenue)
        - recognized_reservation_prepayments
        + float(state.cumulative_reservation_prepayments)
        + float(state.cumulative_retail_revenue)
        + float(state.cumulative_deposits_collected)
        + float(state.cumulative_asset_sale_proceeds)
        + float(state.cumulative_retail_liquidation_proceeds)
        + float(state.emergency_credit_principal)
    )
    calibration_cash_outflows = (
        float(state.cumulative_refunds)
        + float(state.cumulative_purchase_cost)
        + float(state.cumulative_fees)
        + float(state.cumulative_cleaning_cost)
        + float(state.cumulative_repair_cost)
        + float(state.cumulative_maintenance_cost)
        + float(state.cumulative_rental_holding_cost)
        + float(state.cumulative_retail_holding_cost)
        + float(state.cumulative_reservation_compensation)
        + (
            float(state.cumulative_model_cost)
            if config.charge_model_usage_to_cash
            else 0.0
        )
    )
    calibration_expected_cash = round(
        float(config.initial_cash)
        + calibration_cash_inflows
        - calibration_cash_outflows
        + float(state.bankruptcy_shortfall),
        6,
    )
    return {
        "strategy": strategy,
        "reservation_change_policy": reservation_change_policy,
        "reservation_commitment_policy": reservation_commitment_policy,
        "scripted_fleet_expansion_cap": scripted_fleet_expansion_cap,
        "scripted_opening_durable_units_per_offer": int(
            scripted_opening_durable_units_per_offer
        ),
        "scripted_day1_forced_durable_assets": int(
            scripted_day1_forced_durable_assets
        ),
        "policy_contract": BASELINE_POLICY_CONTRACTS[strategy],
        "reservation_contract_mode": config.reservation_contract_mode,
        "reservation_profile_deposit_policy": (
            config.reservation_profile_deposit_policy
        ),
        "scripted_profile_diagnostics": scripted_profile_diagnostics,
        "scripted_profile_quote_scale": PROFILE_SCRIPTED_QUOTE_SCALE,
        "scripted_profile_lead_risk_premium": (
            PROFILE_SCRIPTED_LEAD_RISK_PREMIUM
        ),
        "scripted_profile_operational_subset_mode": (
            PROFILE_SCRIPTED_OPERATIONAL_SUBSET_MODE
        ),
        "scripted_pairwise_menu_policy": (
            PAIRWISE_SCRIPTED_MENU_POLICY
            if config.reservation_contract_mode
            == "adaptive_pairwise_exact_quotes_v1"
            else None
        ),
        "seed": config.seed,
        "demand_world_id": config.demand_world_id,
        "initial_cash": config.initial_cash,
        "weather_start_day": config.weather_start_day,
        "days": config.max_days,
        "bankrupt": state.bankrupt,
        "bankruptcy_shortfall": round(state.bankruptcy_shortfall, 2),
        "initial_official_liquidation_value": initial_value,
        "final_official_liquidation_value": values["official_liquidation_value"],
        **({"warehouse_capacity": environment.warehouse_diagnostics(state)}
           if environment._v31_warehouse_capacity() else {}),
        "value_change": official_value_change,
        "initial_contract_adjusted_value": initial_contract_value,
        "final_contract_adjusted_value": values["contract_adjusted_economic_value"],
        "contract_value_change": round(
            values["contract_adjusted_economic_value"] - initial_contract_value, 2
        ),
        "rental_revenue": state.cumulative_rental_revenue,
        "retail_revenue": state.cumulative_retail_revenue,
        "purchase_cost": state.cumulative_purchase_cost,
        "damage_deductions": state.cumulative_damage_deductions,
        "cash_flow_audit": {
            "ending_cash": round(float(state.cash), 6),
            "expected_ending_cash": calibration_expected_cash,
            "reconciliation_difference": round(
                float(state.cash) - calibration_expected_cash, 6
            ),
        },
        "lifecycle": lifecycle_diagnostics,
        "payback": payback_diagnostics,
        "asset_sale_proceeds": state.cumulative_asset_sale_proceeds,
        "retail_liquidation_proceeds": (
            state.cumulative_retail_liquidation_proceeds
        ),
        "retail_liquidation_book_cost": (
            state.cumulative_retail_liquidation_book_cost
        ),
        "retail_liquidation_loss": state.cumulative_retail_liquidation_loss,
        "retail_units_liquidated": state.cumulative_retail_units_liquidated,
        "reservations": {
            "enabled": bool(config.reservations_enabled),
            "contract_mode": config.reservation_contract_mode,
            "profile_deposit_policy": (
                config.reservation_profile_deposit_policy
            ),
            "configured_share": float(config.reservation_share),
            "pairwise_early_late": pairwise_periods,
            "complete_contract_option_count_counts": dict(
                sorted(
                    Counter(
                        len(request.submitted_contract_options)
                        for request in state.reservation_requests
                        if request.reservation_contract_mode
                        == "complete_contract_options_v1"
                        and request.quote_action == "quote"
                    ).items()
                )
            ),
            "complete_contract_selected_option_counts": dict(
                sorted(
                    Counter(
                        str(request.selected_contract_option_id)
                        for request in state.reservation_requests
                        if request.selected_contract_option_id is not None
                    ).items()
                )
            ),
            "complete_contract_feedback_response_requests": sum(
                request.feedback_response_available is True
                for request in state.reservation_requests
            ),
            "complete_contract_assessment_category_counts": dict(
                sorted(
                    Counter(
                        str(
                            assessment["reported_assessment"]["category"]
                        )
                        for request in state.reservation_requests
                        for assessment in request.reported_contract_assessments
                        if assessment.get("reported_assessment", {}).get(
                            "available"
                        )
                    ).items()
                )
            ),
            "factorized_internal_plan_count_counts": dict(
                sorted(
                    Counter(
                        int(request.internal_plan_count)
                        for request in state.reservation_requests
                        if request.reservation_contract_mode
                        == "factorized_line_options_plan_set_v1"
                        and request.quote_action == "quote"
                    ).items()
                )
            ),
            "factorized_authorized_plan_count_counts": dict(
                sorted(
                    Counter(
                        len(request.customer_authorized_plans)
                        for request in state.reservation_requests
                        if request.reservation_contract_mode
                        == "factorized_line_options_plan_set_v1"
                        and request.quote_action == "quote"
                    ).items()
                )
            ),
            "factorized_confirmation_origin_counts": dict(
                sorted(
                    Counter(
                        str(request.plan_confirmation_origin)
                        for request in state.reservation_requests
                        if request.plan_confirmation_origin is not None
                    ).items()
                )
            ),
            "factorized_nonpreferred_confirmations": sum(
                request.manager_confirmed_plan_id is not None
                and request.manager_confirmed_plan_id
                != request.customer_preferred_plan_id
                for request in state.reservation_requests
                if request.reservation_contract_mode
                == "factorized_line_options_plan_set_v1"
            ),
            "factorized_decline_feedback_category_counts": dict(
                sorted(
                    Counter(
                        str(request.reported_decline_feedback["category"])
                        for request in state.reservation_requests
                        if request.reported_decline_feedback.get("available")
                        and "category" in request.reported_decline_feedback
                    ).items()
                )
            ),
            "factorized_decline_feedback_tag_counts": dict(
                sorted(
                    Counter(
                        str(tag)
                        for request in state.reservation_requests
                        if request.reported_decline_feedback.get("available")
                        for tag in request.reported_decline_feedback.get(
                            "reported_concerns", []
                        )
                    ).items()
                )
            ),
            "factorized_decline_feedback_concern_set_counts": dict(
                sorted(
                    Counter(
                        "+".join(
                            str(tag)
                            for tag in request.reported_decline_feedback.get(
                                "reported_concerns", []
                            )
                        )
                        for request in state.reservation_requests
                        if request.reported_decline_feedback.get("available")
                        and request.reported_decline_feedback.get(
                            "reported_concerns"
                        )
                    ).items()
                )
            ),
            "authorization_mode": str(config.reservation_authorization_mode),
            "accepted_line_authorization_set_size_counts": dict(
                sorted(
                    Counter(
                        len(line.authorized_offer_ids)
                        for commitment in state.reservation_commitments
                        for line in commitment.lines
                        if line.authorized_offer_ids
                    ).items()
                )
            ),
            "inquiry_latent_set_size_counts": dict(
                sorted(
                    Counter(
                        len(
                            request.inquiry_latent_offer_ids_by_line.get(
                                line.line_id, [line.offer_id]
                            )
                        )
                        for request in state.reservation_requests
                        for line in request.lines
                    ).items()
                )
            ),
            "recovery_set_size_counts": dict(
                sorted(
                    Counter(
                        len(
                            request.recovery_offer_ids_by_line.get(
                                line.line_id, []
                            )
                        )
                        for request in state.reservation_requests
                        for line in request.lines
                        if request.commitment_id is not None
                    ).items()
                )
            ),
            "fulfillment_mode_counts": dict(
                sorted(
                    Counter(
                        str(line.fulfillment_mode)
                        for commitment in state.reservation_commitments
                        for line in commitment.lines
                        if line.fulfillment_mode is not None
                    ).items()
                )
            ),
            "profile_selection_counts": dict(
                sorted(
                    Counter(
                        str(line.promise_profile_id)
                        for request in state.reservation_requests
                        for line in request.lines
                        if line.promise_profile_id is not None
                    ).items()
                )
            ),
            "profile_family_selection_counts": dict(
                sorted(
                    Counter(
                        str(line.profile_family_id)
                        for request in state.reservation_requests
                        for line in request.lines
                        if line.profile_family_id is not None
                    ).items()
                )
            ),
            "staging_actions": state.cumulative_reservation_staging_actions,
            "staged_authorized_fallbacks": (
                state.cumulative_reservation_staged_fallbacks
            ),
            "requests": state.cumulative_reservation_requests,
            "quotes": state.cumulative_reservation_quotes,
            "decision_counts": {
                "explicit_quote": state.cumulative_reservation_quotes,
                "decline": state.cumulative_reservation_manager_quote_declines,
                "expired_unquoted": state.cumulative_reservations_expired,
            },
            "customer_quote_declines": (
                state.cumulative_reservation_customer_quote_declines
            ),
            "manager_quote_declines": (
                state.cumulative_reservation_manager_quote_declines
            ),
            "optional_lines_removed_at_quote": (
                state.cumulative_reservation_optional_lines_removed_at_quote
            ),
            "confirmed": state.cumulative_reservations_confirmed,
            "declined": state.cumulative_reservations_declined,
            "expired": state.cumulative_reservations_expired,
            "pickups": state.cumulative_reservation_pickups,
            "pickup_failures": state.cumulative_reservation_pickup_failures,
            "pickup_failure_cause_counts": dict(
                sorted(
                    Counter(
                        str(commitment.failure_cause or "unknown")
                        for commitment in state.reservation_commitments
                        if commitment.status == "pickup_failed"
                    ).items()
                )
            ),
            "pickup_failure_days": [
                int(event.day)
                for event in state.reservation_events
                if event.event_type == "pickup_failed"
            ],
            "pickup_failures_after_decision_horizon": sum(
                event.event_type == "pickup_failed"
                and int(event.day) > int(config.max_days)
                for event in state.reservation_events
            ),
            "manager_cancellations": sum(
                commitment.status == "cancelled_by_manager"
                for commitment in state.reservation_commitments
            ),
            "proactive_manager_cancellations": sum(
                event.event_type == "cancelled_by_manager"
                and event.details.get("cancellation_origin") == "proactive"
                for event in state.reservation_events
            ),
            "risk_alert_manager_cancellations": sum(
                event.event_type == "cancelled_by_manager"
                and event.details.get("cancellation_origin") == "risk_alert"
                for event in state.reservation_events
            ),
            "same_day_manager_cancellations": sum(
                event.event_type == "cancelled_by_manager"
                and int(event.details.get("notice_days", 0)) == 0
                for event in state.reservation_events
            ),
            "manager_failure_required_service_unit_days": sum(
                int(event.details.get("required_service_unit_days", 0))
                for event in state.reservation_events
                if event.event_type
                in {"cancelled_by_manager", "pickup_failed"}
            ),
            "manager_cancellation_profile_line_counts": dict(
                sorted(
                    Counter(
                        str(
                            line.promise_profile_id
                            or f"exact:{line.offer_id}"
                        )
                        for commitment in state.reservation_commitments
                        if commitment.status == "cancelled_by_manager"
                        for line in commitment.lines
                        if line.required
                    ).items()
                )
            ),
            "manager_cancellation_service_line_counts": dict(
                sorted(
                    Counter(
                        str(line.service_key or line.service_line)
                        for commitment in state.reservation_commitments
                        if commitment.status == "cancelled_by_manager"
                        for line in commitment.lines
                        if line.required
                    ).items()
                )
            ),
            "risk_exposures": sum(
                event.event_type == "pickup_risk_exposed"
                for event in state.reservation_events
            ),
            "fallback_inquiries": sum(
                event.event_type == "fallback_inquired"
                for event in state.reservation_events
            ),
            "preauthorized_recovery_plan_candidates_frozen": sum(
                len(commitment.preauthorized_recovery_plans)
                for commitment in state.reservation_commitments
            ),
            "preauthorized_recovery_plan_selections": sum(
                event.event_type == "preauthorized_recovery_plan_selected"
                for event in state.reservation_events
            ),
            "recovery_price_credits": round(
                sum(
                    float(line.recovery_price_credit)
                    for commitment in state.reservation_commitments
                    for line in commitment.lines
                ),
                2,
            ),
            "prepayments_collected": state.cumulative_reservation_prepayments,
            "prepayments_refunded": (
                state.cumulative_reservation_prepayment_refunds
            ),
            "customer_cancellations": (
                state.cumulative_reservation_customer_cancellations
            ),
            "weather_customer_cancellations": sum(
                event.event_type == "customer_cancelled_weather"
                for event in state.reservation_events
            ),
            "non_weather_customer_cancellations": sum(
                event.event_type == "customer_cancelled_non_weather"
                for event in state.reservation_events
            ),
            "reschedules": state.cumulative_reservation_reschedules,
            "reschedule_request_tail": _reschedule_request_tail_diagnostics(
                state
            ),
            "extensions_requested": (
                state.cumulative_reservation_extensions_requested
            ),
            "extensions_approved": (
                state.cumulative_reservation_extensions_approved
            ),
            "late_returns": state.cumulative_reservation_late_returns,
            "compensation": state.cumulative_reservation_compensation,
            "optional_line_credits": state.cumulative_optional_line_credits,
            **(
                {
                    "rent_cash_audit": {
                        "rent_collected_at_pickup": round(
                            sum(
                                commitment.rent_collected_at_pickup
                                for commitment in state.reservation_commitments
                            ),
                            2,
                        ),
                        "post_pickup_rent_collected": round(
                            sum(
                                commitment.post_pickup_rent_collected
                                for commitment in state.reservation_commitments
                            ),
                            2,
                        ),
                        "gross_rent_cash_collected": round(
                            sum(
                                environment._v27_reservation_cash_history(
                                    commitment
                                )["gross_rent_cash_collected"]
                                for commitment in state.reservation_commitments
                            ),
                            2,
                        ),
                        "net_rent_cash_collected": round(
                            sum(
                                environment._v27_reservation_cash_history(
                                    commitment
                                )["net_rent_cash_collected"]
                                for commitment in state.reservation_commitments
                            ),
                            2,
                        ),
                        "extension_rent_collected_from_events": round(
                            sum(
                                float(
                                    event.details.get(
                                        "additional_rent_collected", 0.0
                                    )
                                )
                                for event in state.reservation_events
                                if event.event_type == "extension_accepted"
                            ),
                            2,
                        ),
                        "extension_accepted_events": sum(
                            event.event_type == "extension_accepted"
                            for event in state.reservation_events
                        ),
                        "zero_rent_extension_line_deltas": sum(
                            float(line.get("additional_locked_rent", 0.0))
                            == 0.0
                            for event in state.reservation_events
                            if event.event_type == "extension_accepted"
                            for line in event.details.get(
                                "line_rent_deltas", []
                            )
                        ),
                    }
                }
                if environment._v27_extension_cash_audit()
                else {}
            ),
            "maintenance_actions": state.cumulative_maintenance_actions,
            "emergency_credit_principal": state.emergency_credit_principal,
            "emergency_credit_liability": state.emergency_credit_liability,
            "emergency_credit_fees": state.cumulative_emergency_credit_fees,
            "unearned_revenue_liability_end": (
                state.unearned_revenue_liability
            ),
        },
        "cumulative_costs": {
            "purchases": state.cumulative_purchase_cost,
            "cleaning": state.cumulative_cleaning_cost,
            "repairs": state.cumulative_repair_cost,
            "maintenance": state.cumulative_maintenance_cost,
            "reservation_compensation": (
                state.cumulative_reservation_compensation
            ),
            "retail_holding": state.cumulative_retail_holding_cost,
            "rental_holding": state.cumulative_rental_holding_cost,
            "rental_idle_holding": float(
                getattr(state, "cumulative_rental_idle_holding_cost", 0.0)
            ),
            **(
                {
                    "rental_capital_holding": state.cumulative_rental_capital_holding_cost,
                    "rental_space_holding": state.cumulative_rental_space_holding_cost,
                }
                if config.lifecycle_economics_enabled
                else {}
            ),
            "operating": state.cumulative_fees,
            "model": state.cumulative_model_cost,
        },
        "liquidation_market": {
            "rolling_depth_enabled": bool(config.liquidation_depth_enabled),
            "nominal_proceeds": round(
                sum(
                    float(
                        report.get("financial", {}).get(
                            "liquidation_nominal_proceeds", 0.0
                        )
                    )
                    for report in state.daily_history
                ),
                2,
            ),
            "depth_discount": round(
                sum(
                    float(
                        report.get("financial", {}).get(
                            "liquidation_depth_discount", 0.0
                        )
                    )
                    for report in state.daily_history
                ),
                2,
            ),
        },
        "initial_value_components": initial_values,
        "final_value_components": values,
        "offer_economics": offer_economics,
        "official_value_bridge": value_bridge,
        "seasonal_periods": _seasonal_period_audit(
            state.daily_history,
            initial_value=initial_value,
            initial_contract_value=initial_contract_value,
        ),
        "retail_gross_margin_by_offer": margins,
        "retail_gross_margin_share_by_offer": concentration,
        "customers": state.cumulative_customers,
        "requested_units": state.cumulative_requested_units,
        "fulfilled_units": state.cumulative_fulfilled_units,
        "outside_rejections": state.cumulative_outside_rejections,
        "stockout_rejections": state.cumulative_stockout_rejections,
        "service_closed_rejections": state.cumulative_service_closed_rejections,
        "capacity_policy_rejections": state.cumulative_capacity_policy_rejections,
        "horizon_rejections": state.cumulative_horizon_rejections,
        "internal_channel_counts": dict(sorted(internal_channel_counts.items())),
        "basket_intent_status_counts": dict(
            sorted(basket_intent_status_counts.items())
        ),
        "peak_cleaning_queue_work": max(
            (
                float(report.get("end", {}).get("cleaning_queue_work_points", 0.0))
                for report in state.daily_history
            ),
            default=0.0,
        ),
        "postseason_cleaning_work_points": values.get(
            "postseason_cleaning_work_points", 0.0
        ),
        "postseason_cleaning_cost": values.get(
            "postseason_cleaning_cost", values.get("pending_cleaning_cost", 0.0)
        ),
        "minimum_cash": min(
            (float(point["cash"]) for point in performance_curve),
            default=float(state.cash),
        ),
        "purchased_rental_assets_by_offer": dict(
            sorted(purchased_rental_by_offer.items())
        ),
        "purchased_rental_assets_by_product": dict(
            sorted(purchased_rental_by_product.items())
        ),
        "daily_value_curve": [
            {
                "day": report["day"],
                "official_liquidation_value": report["end"]["official_liquidation_value"],
                "contract_adjusted_economic_value": report["end"]["contract_adjusted_economic_value"],
            }
            for report in state.daily_history
        ],
        "daily_performance_curve": performance_curve,
        "refunds_by_day": [
            round(float(report["financial"].get("refunds", 0.0)), 2)
            for report in state.daily_history
        ],
        "review_flags": {
            "bankruptcy": state.bankrupt,
            "offers_above_40pct_retail_margin": high_concentration,
        },
    }


def run_base_stock_policy(config: OutfitterConfig) -> dict[str, Any]:
    return run_policy(config, "naive")



def main(argv: list[str] | None = None) -> int:
    from .calibration_cli import main as run

    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
