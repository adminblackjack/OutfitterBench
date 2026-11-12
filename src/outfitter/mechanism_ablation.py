"""Run a fixed, reproducible Outfitter reservation-mechanism ablation panel."""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from .calibrate import run_policy
from .config import OutfitterConfig


STRATEGIES = ("fixed_public", "informed_public", "scale_winner")


def _run_task(task: tuple[dict[str, Any], str, str]) -> dict[str, Any]:
    config_values, condition, strategy = task
    row = run_policy(OutfitterConfig.from_dict(config_values), strategy)
    row["ablation_condition"] = condition
    return row


def _conditions(*, lifecycle: bool = False) -> dict[str, dict[str, Any]]:
    conditions = {
        "full_main": {},
        "full_stress_80pct": {"reservation_share": 0.80},
        "risk_only_exact_lock": {
            "reservation_authorization_mode": "risk_only",
        },
        "all_latent_visible": {
            "reservation_authorization_mode": "all_latent",
        },
        "no_reservations": {
            "reservations_enabled": False,
            "reservation_disruptions_enabled": False,
            "reservation_bundles_enabled": False,
        },
        "reservations_without_disruptions": {
            "reservation_disruptions_enabled": False,
        },
        "reservations_without_bundles": {
            "reservation_bundles_enabled": False,
        },
        "no_preventive_maintenance": {
            "preventive_maintenance_enabled": False,
        },
        "legacy_flat_exit_and_holding": {
            "rental_holding_rate": 0.0005,
            "rental_idle_holding_rate": 0.0,
            "liquidation_depth_enabled": False,
        },
    }
    if lifecycle:
        conditions["no_preventive_maintenance"][
            "lifecycle_mechanism_ablation"
        ] = True
        conditions["legacy_flat_exit_and_holding"] = {
            "lifecycle_mechanism_ablation": True,
            "lifecycle_holding_enabled": False,
            "rental_holding_rate": 0.0005,
            "rental_idle_holding_rate": 0.0,
            "liquidation_depth_enabled": False,
        }
    return conditions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--days", type=int, default=200)
    parser.add_argument("--runoff-days", type=int, default=14)
    parser.add_argument("--weather-start-day", type=int, default=91)
    parser.add_argument("--daily-fee", type=float, default=20.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 42, 59, 73])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--lifecycle-economics-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args(argv)
    common: dict[str, Any] = {
        "data_dir": args.data_dir,
        "max_days": args.days,
        "runoff_days": args.runoff_days,
        "weather_start_day": args.weather_start_day,
        "daily_fee": args.daily_fee,
        "initial_cash": 5000.0,
        "demand_world_id": "base",
        "charge_model_usage_to_cash": False,
        "rental_holding_rate": 0.001,
        "rental_idle_holding_rate": 0.002,
        "retail_holding_rate": 0.001,
        "liquidation_depth_enabled": True,
        "liquidation_depth_window_days": 7,
        "liquidation_depth_capacity_fraction": 0.15,
        "liquidation_depth_end_multiple": 4.0,
        "liquidation_depth_floor_rate": 0.55,
        "reservations_enabled": True,
        "reservation_share": 0.65,
        "reservation_booking_lead_min_days": 1,
        "reservation_booking_lead_max_days": 14,
        "reservation_prepayment_rate": 0.25,
        "reservation_authorization_mode": "credible_assortment",
        "reservation_authorized_alternative_max": 2,
        "reservation_disruptions_enabled": True,
        "late_return_probability_after_extension_decline": 0.12,
        "reservation_failure_compensation_rate": 0.50,
        "weather_wear_enabled": True,
        "reservation_bundles_enabled": True,
        "optional_line_missing_credit_rate": 1.0,
        "preventive_maintenance_enabled": True,
        "lifecycle_economics_enabled": args.lifecycle_economics_enabled,
        "maintenance_interval_days": 28,
        "maintenance_due_window_days": 3,
        "maintenance_duration_days": 1,
        "maintenance_cost_rate": 0.02,
        "emergency_credit_limit": 300.0,
        "emergency_credit_fee_rate": 0.25,
    }
    tasks = []
    conditions = _conditions(lifecycle=args.lifecycle_economics_enabled)
    for condition, overrides in conditions.items():
        for strategy in STRATEGIES:
            for seed in args.seeds:
                values = {**common, **overrides, "seed": seed}
                tasks.append((values, condition, strategy))
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        rows = list(executor.map(_run_task, tasks))
    summaries = []
    for condition in conditions:
        for strategy in STRATEGIES:
            selected = [
                row
                for row in rows
                if row["ablation_condition"] == condition
                and row["strategy"] == strategy
            ]
            summaries.append(
                {
                    "condition": condition,
                    "strategy": strategy,
                    "seeds": [int(row["seed"]) for row in selected],
                    "median_value_change": round(
                        statistics.median(
                            float(row["value_change"]) for row in selected
                        ),
                        2,
                    ),
                    "median_contract_value_change": round(
                        statistics.median(
                            float(row["contract_value_change"])
                            for row in selected
                        ),
                        2,
                    ),
                    "bankrupt_runs": sum(bool(row["bankrupt"]) for row in selected),
                    "median_confirmed_reservations": statistics.median(
                        int(row["reservations"]["confirmed"]) for row in selected
                    ),
                    "median_pickup_failures": statistics.median(
                        int(row["reservations"]["pickup_failures"])
                        for row in selected
                    ),
                }
            )
    result = {
        "format": "outfitter-v6-mechanism-ablation-v1",
        "days": args.days,
        "runoff_days": args.runoff_days,
        "daily_fee": args.daily_fee,
        "conditions": conditions,
        "strategies": list(STRATEGIES),
        "summaries": summaries,
        "runs": rows,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(
        json.dumps(
            {"format": result["format"], "summaries": summaries},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
