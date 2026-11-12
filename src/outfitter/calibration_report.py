"""Aggregate multi-seed calibration runs into strategy and curve comparisons."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


STRATEGY_ORDER = ("naive", "heuristic", "informed_public")
REPORT_STRATEGIES = (
    *STRATEGY_ORDER,
    "fixed_public",
    "scale_winner",
    "buy_liquidate",
    "reject_all",
    "aggressive_expansion_reference",
    # Historical reports are read as written; this legacy label is never
    # interpreted as an attainable upper bound.
    "oracle_reference",
)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "minimum": round(min(values), 2),
        "median": round(statistics.median(values), 2),
        "maximum": round(max(values), 2),
    }


def _number(value: Any) -> float:
    return 0.0 if value is None else float(value)


def _reference_policy_metadata(payload: dict[str, Any]) -> dict[str, Any] | None:
    current = payload.get("reference_policy")
    if current is not None:
        return dict(current)
    legacy = payload.get("oracle_policy")
    if legacy is None:
        return None
    return {
        **dict(legacy),
        "strategy": "aggressive_expansion_reference",
        "source_strategy_label": dict(legacy).get("strategy"),
        "is_oracle": False,
        "is_attainable_profit_upper_bound": False,
        "migrated_from_legacy_metadata": True,
    }


def _run_metrics(row: dict[str, Any]) -> dict[str, float]:
    requested = int(row["requested_units"])
    fulfilled = int(row["fulfilled_units"])
    curve = list(row["daily_performance_curve"])
    costs = dict(row.get("cumulative_costs") or {})
    bridge = dict(row.get("official_value_bridge") or {})
    final_components = dict(row.get("final_value_components") or {})
    lifecycle = dict(row.get("lifecycle") or {})
    payback = dict(row.get("payback") or {})
    reservations = dict(row.get("reservations") or {})
    return {
        "value_change": float(row["value_change"]),
        "contract_value_change": float(row["contract_value_change"]),
        "bankruptcy_shortfall": float(row.get("bankruptcy_shortfall", 0)),
        "total_revenue": float(row["rental_revenue"])
        + float(row["retail_revenue"]),
        "rental_revenue": float(row["rental_revenue"]),
        "retail_revenue": float(row["retail_revenue"]),
        "retail_liquidation_proceeds": float(
            row.get("retail_liquidation_proceeds", 0)
        ),
        "retail_liquidation_book_cost": float(
            row.get("retail_liquidation_book_cost", 0)
        ),
        "retail_liquidation_loss": float(
            row.get("retail_liquidation_loss", 0)
        ),
        "retail_units_liquidated": float(
            row.get("retail_units_liquidated", 0)
        ),
        "purchase_cost": float(row.get("purchase_cost", costs.get("purchases", 0))),
        "cleaning_cost": float(costs.get("cleaning", 0)),
        "repair_cost": float(costs.get("repairs", 0)),
        "maintenance_cost": float(costs.get("maintenance", 0)),
        "reservation_compensation_cost": float(
            costs.get(
                "reservation_compensation",
                reservations.get("compensation", 0),
            )
        ),
        "retail_holding_cost": float(costs.get("retail_holding", 0)),
        "rental_holding_cost": float(costs.get("rental_holding", 0)),
        "rental_capital_holding_cost": float(
            costs.get("rental_capital_holding", 0)
        ),
        "rental_space_holding_cost": float(
            costs.get("rental_space_holding", 0)
        ),
        "operating_fee": float(costs.get("operating", 0)),
        "offer_contribution_before_shared_costs": float(
            bridge.get("offer_contribution_before_shared_costs", 0)
        ),
        "value_bridge_reconciliation_difference": float(
            bridge.get("reconciliation_difference", 0)
        ),
        "ending_cash": float(final_components.get("cash", curve[-1]["cash"])),
        "ending_rental_secondhand_value": float(
            final_components.get("rental_secondhand_value", 0)
        ),
        "ending_retail_inventory_value": float(
            final_components.get("retail_inventory_value", 0)
        ),
        "ending_in_transit_value": float(
            final_components.get("in_transit_value", 0)
        ),
        "fulfillment_rate": fulfilled / max(1, requested),
        "minimum_cash": float(
            row.get(
                "minimum_cash",
                min(float(point["cash"]) for point in curve),
            )
        ),
        "peak_cleaning_queue_work": float(row["peak_cleaning_queue_work"]),
        "postseason_cleaning_work_points": float(
            row.get("postseason_cleaning_work_points", 0.0)
        ),
        "purchased_rental_assets": float(
            sum(
                int(value)
                for value in row.get(
                    "purchased_rental_assets_by_product", {}
                ).values()
            )
        ),
        "gross_payback_contracts_weighted_median": _number(
            payback.get("gross_payback_contracts_weighted_median")
        ),
        "variable_payback_contracts_weighted_median": _number(
            payback.get(
                "variable_contribution_payback_contracts_weighted_median"
            )
        ),
        "purchased_cohort_economic_recovery_ratio": _number(
            payback.get("cohort_economic_recovery_ratio")
        ),
        "purchased_asset_unrecovered_share": _number(
            payback.get("purchased_asset_unrecovered_share")
        ),
        "damage_event_count": _number(lifecycle.get("damage_event_count")),
        "maximum_net_active_expansion": _number(
            lifecycle.get("maximum_net_active_expansion")
        ),
        "reservation_pickup_failures": _number(
            reservations.get("pickup_failures")
        ),
        "reservation_pickup_failures_after_decision_horizon": _number(
            reservations.get("pickup_failures_after_decision_horizon")
        ),
    }


def _seasonal_summaries(
    by_strategy: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    fields = (
        "official_value_change",
        "contract_value_change",
        "rental_revenue",
        "retail_revenue",
        "retail_liquidation_proceeds",
        "retail_liquidation_loss",
        "retail_units_liquidated",
        "purchase_cost",
        "cleaning_and_repair_cost",
        "holding_cost",
        "operating_fee",
        "customers",
        "requested_units",
        "fulfilled_units",
        "fulfillment_rate",
        "outside_rejections",
        "stockout_rejections",
    )
    result: dict[str, list[dict[str, Any]]] = {}
    for strategy in REPORT_STRATEGIES:
        month_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for run in by_strategy.get(strategy, []):
            for period in run.get("seasonal_periods", []):
                month_rows[str(period["month"])].append(period)
        if not month_rows:
            continue
        result[strategy] = [
            {
                "month": month,
                "start_day": min(int(row["start_day"]) for row in rows),
                "end_day": max(int(row["end_day"]) for row in rows),
                "metrics": {
                    field: _summary([_number(row.get(field, 0)) for row in rows])
                    for field in fields
                },
            }
            for month, rows in sorted(month_rows.items())
        ]
    return result


def _offer_economic_summaries(
    by_strategy: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    fields = (
        "revenue",
        "purchase_cost",
        "retail_cogs",
        "retail_gross_margin",
        "cleaning_cost_paid",
        "repair_cost",
        "pending_cleaning_cost",
        "asset_sale_proceeds",
        "retail_units_liquidated",
        "retail_liquidation_proceeds",
        "retail_liquidation_book_cost",
        "retail_liquidation_loss",
        "final_official_value",
        "contribution_before_shared_costs",
        "purchased_units",
        "acquired_rental_assets",
        "rental_days",
        "fulfilled_units",
        "quote_exposures",
    )
    result: dict[str, dict[str, Any]] = {}
    for strategy in REPORT_STRATEGIES:
        runs = by_strategy.get(strategy, [])
        if not runs or not any(run.get("offer_economics") for run in runs):
            continue
        offer_ids = sorted(
            {
                str(offer_id)
                for run in runs
                for offer_id in (run.get("offer_economics") or {})
            }
        )
        offers = []
        for offer_id in offer_ids:
            source_rows = [
                (run.get("offer_economics") or {}).get(offer_id, {})
                for run in runs
            ]
            identity = next((row for row in source_rows if row), {})
            metrics = {
                field: _summary([_number(row.get(field, 0)) for row in source_rows])
                for field in fields
            }
            if not any(
                metrics[field]["maximum"] != 0
                for field in (
                    "revenue",
                    "purchase_cost",
                    "final_official_value",
                    "contribution_before_shared_costs",
                )
            ):
                continue
            offers.append(
                {
                    "offer_id": offer_id,
                    "name": identity.get("name"),
                    "asset_kind": identity.get("asset_kind"),
                    "service_key": identity.get("service_key"),
                    "choice_group": identity.get("choice_group"),
                    "metrics": metrics,
                }
            )
        ranked = sorted(
            offers,
            key=lambda row: (
                -row["metrics"]["contribution_before_shared_costs"]["median"],
                row["offer_id"],
            ),
        )
        positive_total = sum(
            max(0.0, row["metrics"]["contribution_before_shared_costs"]["median"])
            for row in ranked
        )
        result[strategy] = {
            "offers": ranked,
            "top_positive_contributors": ranked[:10],
            "top_negative_contributors": sorted(
                ranked,
                key=lambda row: (
                    row["metrics"]["contribution_before_shared_costs"]["median"],
                    row["offer_id"],
                ),
            )[:10],
            "largest_positive_contribution_share": round(
                (
                    max(
                        0.0,
                        ranked[0]["metrics"]["contribution_before_shared_costs"][
                            "median"
                        ],
                    )
                    / positive_total
                    if ranked and positive_total > 0
                    else 0.0
                ),
                6,
            ),
        }
    return result


def _adversarial_policy_checks(runs: list[dict[str, Any]]) -> dict[str, Any]:
    worlds: dict[str, dict[str, Any]] = {}
    for world_id in sorted(
        {str(row.get("demand_world_id") or "base") for row in runs}
    ):
        selected = [
            row
            for row in runs
            if str(row.get("demand_world_id") or "base") == world_id
        ]
        medians = {}
        for strategy in (
            "informed_public",
            "fixed_public",
            "scale_winner",
            "buy_liquidate",
            "reject_all",
        ):
            values = [
                float(row["value_change"])
                for row in selected
                if row.get("strategy") == strategy
            ]
            if values:
                medians[strategy] = round(statistics.median(values), 2)
        scale_check = (
            medians["scale_winner"] < medians["informed_public"]
            if {"scale_winner", "informed_public"} <= set(medians)
            else None
        )
        churn_check = (
            medians["buy_liquidate"] < medians["reject_all"]
            if {"buy_liquidate", "reject_all"} <= set(medians)
            else None
        )
        fixed_comparison = (
            medians["informed_public"] - medians["fixed_public"]
            if {"informed_public", "fixed_public"} <= set(medians)
            else None
        )
        # Profitable demand-supported scaling is not an exploit.  Finite-scale
        # behavior is checked by the dedicated marginal-scale panel; this
        # compact report retains the old comparison only as a diagnostic.
        required = [value for value in (churn_check,) if value is not None]
        worlds[world_id] = {
            "median_value_change": medians,
            "scale_winner_below_informed": scale_check,
            "buy_liquidate_below_reject_all": churn_check,
            "informed_minus_fixed_public": (
                None if fixed_comparison is None else round(fixed_comparison, 2)
            ),
            "passed": bool(required) and all(required),
        }
    return {
        "format": "outfitter-adversarial-policy-check-v2-churn-gated-scale-diagnostic",
        "worlds": worlds,
        "passed": bool(worlds) and all(row["passed"] for row in worlds.values()),
    }


def build_calibration_report(payload: dict[str, Any]) -> dict[str, Any]:
    runs = list(payload["runs"])
    world_ids = sorted(
        {str(row.get("demand_world_id") or "base") for row in runs}
    )
    if len(world_ids) > 1:
        return {
            "format": "outfitter-calibration-comparison-v2",
            "by_demand_world": {
                world_id: build_calibration_report(
                    {
                        **payload,
                        "runs": [
                            row
                            for row in runs
                            if str(row.get("demand_world_id") or "base")
                            == world_id
                        ],
                    }
                )
                for world_id in world_ids
            },
            "liquidity_recommendation": payload.get("liquidity_recommendation"),
            "scripted_profit_gate": payload.get("scripted_profit_gate"),
            "adversarial_policy_checks": _adversarial_policy_checks(runs),
            "reference_policy": _reference_policy_metadata(payload),
        }
    by_strategy: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        by_strategy[str(row["strategy"])].append(row)

    strategy_summaries = {}
    median_curves = {}
    for strategy in REPORT_STRATEGIES:
        rows = sorted(by_strategy.get(strategy, []), key=lambda row: int(row["seed"]))
        if not rows:
            continue
        metrics = [_run_metrics(row) for row in rows]
        strategy_summaries[strategy] = {
            "seeds": [int(row["seed"]) for row in rows],
            "bankrupt_runs": sum(bool(row["bankrupt"]) for row in rows),
            **{
                name: _summary([metric[name] for metric in metrics])
                for name in metrics[0]
            },
            "purchased_rental_assets_by_product": {
                product: _summary(
                    [
                        float(
                            row.get("purchased_rental_assets_by_product", {}).get(
                                product, 0
                            )
                        )
                        for row in rows
                    ]
                )
                for product in sorted(
                    {
                        product
                        for row in rows
                        for product in row.get(
                            "purchased_rental_assets_by_product", {}
                        )
                    }
                )
            },
            "internal_channel_share": {
                channel: _summary(
                    [
                        float(row.get("internal_channel_counts", {}).get(channel, 0))
                        / max(
                            1,
                            sum(
                                int(value)
                                for value in row.get(
                                    "internal_channel_counts", {}
                                ).values()
                            ),
                        )
                        for row in rows
                    ]
                )
                for channel in sorted(
                    {
                        channel
                        for row in rows
                        for channel in row.get("internal_channel_counts", {})
                    }
                )
            },
        }
        by_day: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            for point in row["daily_performance_curve"]:
                normalized = dict(point)
                normalized.setdefault(
                    "official_liquidation_value",
                    float(row.get("initial_official_liquidation_value", 0.0))
                    + float(point["official_value_change"]),
                )
                normalized.setdefault(
                    "contract_adjusted_value_change",
                    float(point["official_value_change"]),
                )
                normalized.setdefault(
                    "contract_adjusted_economic_value",
                    float(row.get("initial_contract_adjusted_value", 0.0))
                    + float(normalized["contract_adjusted_value_change"]),
                )
                by_day[int(point["day"])].append(normalized)
        median_curves[strategy] = []
        for day, points in sorted(by_day.items()):
            median_curves[strategy].append(
                {
                    "day": day,
                    "surviving_runs": len(points),
                    **{
                        field: round(
                            statistics.median(
                                float(point[field]) for point in points
                            ),
                            2,
                        )
                        for field in (
                            "official_liquidation_value",
                            "official_value_change",
                            "contract_adjusted_economic_value",
                            "contract_adjusted_value_change",
                            "cumulative_revenue",
                            "cumulative_non_deposit_cash_flow",
                            "cash",
                        )
                    },
                }
            )

    by_strategy_seed = {
        (str(row["strategy"]), int(row["seed"])): row for row in runs
    }
    pairwise = []
    for lower, upper in zip(STRATEGY_ORDER, STRATEGY_ORDER[1:]):
        seeds = sorted(
            seed
            for strategy, seed in by_strategy_seed
            if strategy == lower and (upper, seed) in by_strategy_seed
        )
        differences = [
            float(by_strategy_seed[(upper, seed)]["value_change"])
            - float(by_strategy_seed[(lower, seed)]["value_change"])
            for seed in seeds
        ]
        if not differences:
            continue
        pairwise.append(
            {
                "lower_strategy": lower,
                "upper_strategy": upper,
                "seeds": seeds,
                "upper_value_change_minus_lower": _summary(differences),
                "upper_wins": sum(value > 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
            }
        )

    common_seeds = sorted(
        set.intersection(
            *(
                {int(row["seed"]) for row in by_strategy.get(strategy, [])}
                for strategy in STRATEGY_ORDER
            )
        )
    )
    strict_order_seeds = [
        seed
        for seed in common_seeds
        if all(
            float(by_strategy_seed[(upper, seed)]["value_change"])
            > float(by_strategy_seed[(lower, seed)]["value_change"])
            for lower, upper in zip(STRATEGY_ORDER, STRATEGY_ORDER[1:])
        )
    ]
    return {
        "format": "outfitter-calibration-comparison-v1",
        "strategy_summaries": strategy_summaries,
        "median_performance_curves": median_curves,
        "seasonal_period_summaries": _seasonal_summaries(by_strategy),
        "offer_economic_summaries": _offer_economic_summaries(by_strategy),
        "pairwise_value_change": pairwise,
        "strict_strategy_order": {
            "evaluated_seeds": common_seeds,
            "passing_seeds": strict_order_seeds,
            "pass_count": len(strict_order_seeds),
        },
        "liquidity_recommendation": payload.get("liquidity_recommendation"),
        "scripted_profit_gate": payload.get("scripted_profit_gate"),
        "adversarial_policy_checks": _adversarial_policy_checks(runs),
        "reference_policy": _reference_policy_metadata(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Outfitter calibration")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = build_calibration_report(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
