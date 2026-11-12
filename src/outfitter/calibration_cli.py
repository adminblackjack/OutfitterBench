"""Command-line orchestration for scripted Outfitter calibration."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .calibrate import (
    AGGRESSIVE_EXPANSION_REFERENCE,
    ALL_STRATEGIES,
    FORMAL_STRATEGIES,
    LEGACY_REFERENCE_ALIASES,
    run_policy,
    scripted_economic_gate,
)
from .config import OutfitterConfig
from .data import OutfitterDataset
from .environment import OutfitterEnvironment


def _run_task(task: tuple[dict[str, object], str]) -> dict[str, object]:
    config_values, strategy = task
    return run_policy(OutfitterConfig.from_dict(config_values), strategy)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic Outfitter calibration")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "runtime_data"))
    parser.add_argument("--days", type=int, default=198)
    parser.add_argument("--weather-start-day", type=int, default=91)
    parser.add_argument("--daily-fee", type=float, default=60.0)
    parser.add_argument("--arrival-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--rental-holding-rate", type=float, default=0.0005)
    parser.add_argument("--rental-idle-holding-rate", type=float, default=0.0)
    parser.add_argument("--retail-holding-rate", type=float, default=0.001)
    parser.add_argument(
        "--liquidation-depth-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--liquidation-depth-window-days", type=int, default=7)
    parser.add_argument(
        "--liquidation-depth-capacity-fraction", type=float, default=0.20
    )
    parser.add_argument("--liquidation-depth-end-multiple", type=float, default=4.0)
    parser.add_argument("--liquidation-depth-floor-rate", type=float, default=0.55)
    parser.add_argument(
        "--reservations-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--reservation-share", type=float, default=0.65)
    parser.add_argument(
        "--reservation-contract-mode",
        choices=(
            "exact_candidates_v15",
            "promise_profiles_v1",
            "adaptive_pairwise_exact_quotes_v1",
            "complete_contract_options_v1",
            "factorized_line_options_plan_set_v1",
        ),
        default="exact_candidates_v15",
    )
    parser.add_argument(
        "--reservation-profile-deposit-policy",
        choices=("canonical_reference_v1", "flat20_v1", "flat40_v1"),
    )
    parser.add_argument("--reservation-booking-lead-min-days", type=int, default=1)
    parser.add_argument("--reservation-booking-lead-max-days", type=int, default=14)
    parser.add_argument("--reservation-prepayment-rate", type=float, default=0.25)
    parser.add_argument(
        "--reservation-authorization-mode",
        choices=("credible_assortment", "risk_only", "all_latent"),
        default="credible_assortment",
    )
    parser.add_argument(
        "--reservation-authorized-alternative-max", type=int, default=2
    )
    parser.add_argument(
        "--reservation-disruptions-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--late-return-probability-after-extension-decline",
        type=float,
        default=0.12,
    )
    parser.add_argument(
        "--reservation-failure-compensation-rate", type=float, default=0.50
    )
    parser.add_argument(
        "--weather-wear-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--reservation-bundles-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--optional-line-missing-credit-rate", type=float, default=1.0)
    parser.add_argument(
        "--preventive-maintenance-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--lifecycle-economics-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--maintenance-interval-days", type=int, default=28)
    parser.add_argument("--maintenance-due-window-days", type=int, default=3)
    parser.add_argument("--maintenance-duration-days", type=int, default=1)
    parser.add_argument("--maintenance-cost-rate", type=float, default=0.02)
    parser.add_argument("--runoff-days", type=int, default=0)
    parser.add_argument("--emergency-credit-limit", type=float, default=0.0)
    parser.add_argument("--emergency-credit-fee-rate", type=float, default=0.25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 42, 59, 73])
    parser.add_argument(
        "--demand-worlds",
        nargs="+",
        choices=("conservative", "base", "convenience-heavy"),
        default=["conservative", "base", "convenience-heavy"],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=(*ALL_STRATEGIES, *sorted(LEGACY_REFERENCE_ALIASES)),
        default=ALL_STRATEGIES,
    )
    args = parser.parse_args(argv)
    tasks = [
        (
            {
                "data_dir": args.data_dir,
                "max_days": args.days,
                "weather_start_day": args.weather_start_day,
                "seed": seed,
                "daily_fee": args.daily_fee,
                "arrival_scale_multiplier": args.arrival_scale_multiplier,
                "initial_cash": args.initial_cash,
                "demand_world_id": demand_world,
                "charge_model_usage_to_cash": False,
                "rental_holding_rate": args.rental_holding_rate,
                "rental_idle_holding_rate": args.rental_idle_holding_rate,
                "retail_holding_rate": args.retail_holding_rate,
                "liquidation_depth_enabled": args.liquidation_depth_enabled,
                "liquidation_depth_window_days": args.liquidation_depth_window_days,
                "liquidation_depth_capacity_fraction": (
                    args.liquidation_depth_capacity_fraction
                ),
                "liquidation_depth_end_multiple": args.liquidation_depth_end_multiple,
                "liquidation_depth_floor_rate": args.liquidation_depth_floor_rate,
                "reservations_enabled": args.reservations_enabled,
                "reservation_contract_mode": args.reservation_contract_mode,
                "reservation_profile_deposit_policy": (
                    args.reservation_profile_deposit_policy
                ),
                "reservation_share": args.reservation_share,
                "reservation_booking_lead_min_days": (
                    args.reservation_booking_lead_min_days
                ),
                "reservation_booking_lead_max_days": (
                    args.reservation_booking_lead_max_days
                ),
                "reservation_prepayment_rate": args.reservation_prepayment_rate,
                "reservation_authorization_mode": (
                    args.reservation_authorization_mode
                ),
                "reservation_authorized_alternative_max": (
                    args.reservation_authorized_alternative_max
                ),
                "reservation_disruptions_enabled": (
                    args.reservation_disruptions_enabled
                ),
                "late_return_probability_after_extension_decline": (
                    args.late_return_probability_after_extension_decline
                ),
                "reservation_failure_compensation_rate": (
                    args.reservation_failure_compensation_rate
                ),
                "weather_wear_enabled": args.weather_wear_enabled,
                "reservation_bundles_enabled": args.reservation_bundles_enabled,
                "optional_line_missing_credit_rate": (
                    args.optional_line_missing_credit_rate
                ),
                "preventive_maintenance_enabled": (
                    args.preventive_maintenance_enabled
                ),
                "lifecycle_economics_enabled": args.lifecycle_economics_enabled,
                "maintenance_interval_days": args.maintenance_interval_days,
                "maintenance_due_window_days": args.maintenance_due_window_days,
                "maintenance_duration_days": args.maintenance_duration_days,
                "maintenance_cost_rate": args.maintenance_cost_rate,
                "runoff_days": args.runoff_days,
                "emergency_credit_limit": args.emergency_credit_limit,
                "emergency_credit_fee_rate": args.emergency_credit_fee_rate,
            },
            strategy,
        )
        for demand_world in args.demand_worlds
        for strategy in args.strategies
        for seed in args.seeds
    ]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(_run_task, tasks))
    else:
        rows = [_run_task(task) for task in tasks]
    dataset = OutfitterDataset.load(args.data_dir)
    starter_costs = []
    environment = OutfitterEnvironment(
        OutfitterConfig(
            data_dir=args.data_dir,
            max_days=args.days,
            weather_start_day=args.weather_start_day,
            daily_fee=args.daily_fee,
            arrival_scale_multiplier=args.arrival_scale_multiplier,
            initial_cash=args.initial_cash,
            charge_model_usage_to_cash=False,
            rental_holding_rate=args.rental_holding_rate,
            rental_idle_holding_rate=args.rental_idle_holding_rate,
            retail_holding_rate=args.retail_holding_rate,
            liquidation_depth_enabled=args.liquidation_depth_enabled,
            liquidation_depth_window_days=args.liquidation_depth_window_days,
            liquidation_depth_capacity_fraction=(
                args.liquidation_depth_capacity_fraction
            ),
            liquidation_depth_end_multiple=args.liquidation_depth_end_multiple,
            liquidation_depth_floor_rate=args.liquidation_depth_floor_rate,
            reservations_enabled=args.reservations_enabled,
            reservation_contract_mode=args.reservation_contract_mode,
            reservation_profile_deposit_policy=(
                args.reservation_profile_deposit_policy
            ),
            reservation_share=args.reservation_share,
            reservation_booking_lead_min_days=(
                args.reservation_booking_lead_min_days
            ),
            reservation_booking_lead_max_days=(
                args.reservation_booking_lead_max_days
            ),
            reservation_prepayment_rate=args.reservation_prepayment_rate,
            reservation_authorization_mode=args.reservation_authorization_mode,
            reservation_authorized_alternative_max=(
                args.reservation_authorized_alternative_max
            ),
            reservation_disruptions_enabled=(
                args.reservation_disruptions_enabled
            ),
            late_return_probability_after_extension_decline=(
                args.late_return_probability_after_extension_decline
            ),
            reservation_failure_compensation_rate=(
                args.reservation_failure_compensation_rate
            ),
            weather_wear_enabled=args.weather_wear_enabled,
            reservation_bundles_enabled=args.reservation_bundles_enabled,
            optional_line_missing_credit_rate=(
                args.optional_line_missing_credit_rate
            ),
            preventive_maintenance_enabled=args.preventive_maintenance_enabled,
            lifecycle_economics_enabled=args.lifecycle_economics_enabled,
            maintenance_interval_days=args.maintenance_interval_days,
            maintenance_due_window_days=args.maintenance_due_window_days,
            maintenance_duration_days=args.maintenance_duration_days,
            maintenance_cost_rate=args.maintenance_cost_rate,
            runoff_days=args.runoff_days,
            emergency_credit_limit=args.emergency_credit_limit,
            emergency_credit_fee_rate=args.emergency_credit_fee_rate,
        ),
        dataset,
    )
    for offers in environment.by_group.values():
        starter_costs.append(min(float(offer.package_price) for offer in offers))
    ordered_starters = sorted(starter_costs)
    middle = max(0, len(ordered_starters) // 2 - 1)
    experiment_allowance = sum(ordered_starters[middle : middle + 3])
    naive_refund_windows = []
    for row in rows:
        if row["strategy"] != "naive":
            continue
        refunds = list(map(float, row["refunds_by_day"]))
        naive_refund_windows.extend(
            sum(refunds[max(0, index - 6) : index + 1])
            for index in range(len(refunds))
        )
    refund_pressure = (
        sorted(naive_refund_windows)[
            min(
                len(naive_refund_windows) - 1,
                math.floor(0.95 * len(naive_refund_windows)),
            )
        ]
        if naive_refund_windows
        else 0.0
    )
    component_cash = 14 * args.daily_fee + refund_pressure + experiment_allowance
    formal_rows = [row for row in rows if row["strategy"] in FORMAL_STRATEGIES]
    liquidity_rows = [
        row
        for row in formal_rows
        if row["strategy"] in {"heuristic", "informed_public"}
    ]
    completed_drawdowns = [
        args.initial_cash
        - min(
            float(point["cash"])
            for point in row["daily_performance_curve"]
        )
        for row in liquidity_rows
        if not row["bankrupt"] and row["daily_performance_curve"]
    ]
    observed_drawdown = (
        sorted(completed_drawdowns)[
            min(
                len(completed_drawdowns) - 1,
                math.floor(0.95 * len(completed_drawdowns)),
            )
        ]
        if completed_drawdowns
        else 0.0
    )
    bankrupt_runs = sum(bool(row["bankrupt"]) for row in formal_rows)
    liquidity_policy_bankrupt_runs = sum(
        bool(row["bankrupt"]) for row in liquidity_rows
    )
    survival_cash = (
        args.initial_cash + 1_000
        if liquidity_policy_bankrupt_runs
        else observed_drawdown
    )
    recommended_cash = max(
        1_000,
        int(math.ceil(max(component_cash, survival_cash) / 1000.0) * 1000),
    )
    profit_gate = scripted_economic_gate(
        rows,
        required_seed_count=5,
    )
    result = {
        "format": "outfitter-calibration-v6",
        "strategies": list(dict.fromkeys(row["strategy"] for row in rows)),
        "runs": rows,
        "calibration_scale": {
            "arrival_scale_multiplier": args.arrival_scale_multiplier,
            "resolved_arrival_scale": round(
                environment.rules.arrival_scale * args.arrival_scale_multiplier,
                6,
            ),
            "daily_fee": args.daily_fee,
        },
        "scripted_profit_gate": profit_gate,
        "reference_policy": {
            "strategy": AGGRESSIVE_EXPANSION_REFERENCE,
            "is_oracle": False,
            "is_attainable_profit_upper_bound": False,
            "legacy_cli_aliases": sorted(LEGACY_REFERENCE_ALIASES),
            "participates_in_formal_comparison": False,
            "participates_in_profit_gate": False,
            "participates_in_liquidity_recommendation": False,
        },
        "adversarial_policies": {
            strategy: {
                "participates_in_formal_comparison": False,
                "participates_in_profit_gate": False,
                "participates_in_liquidity_recommendation": False,
            }
            for strategy in (
                "fixed_public",
                "scale_winner",
                "buy_liquidate",
                "reject_all",
            )
        },
        "liquidity_recommendation": {
            "fourteen_day_fee_reserve": round(14 * args.daily_fee, 2),
            "p95_seven_day_refund_pressure": round(refund_pressure, 2),
            "three_median_group_experiments": round(experiment_allowance, 2),
            "observed_p95_peak_cash_drawdown": round(observed_drawdown, 2),
            "naive_survival_not_required": True,
            "baseline_bankrupt_runs": bankrupt_runs,
            "minimum_rounding_floor": 1_000,
            "recommended_initial_cash_rounded": recommended_cash,
            "policy": (
                "liquidity pressure only: max(fee reserve + p95 refund window + "
                "three median choice-group experiments, p95 heuristic/informed "
                "drawdown); naive survival is not required and the result is not "
                "a score denominator"
            ),
        },
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0
