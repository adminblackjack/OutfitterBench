"""Deterministic micro-diagnostics for the bounded reservation-price model."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import OutfitterConfig
from .data import OutfitterDataset
from .environment import OutfitterEnvironment
from .demand_lab import (
    SCENARIO_FORMAT,
    _default_request,
    curve as lab_curve,
    invariance as lab_invariance,
)


PRICE_RATIOS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)


def _session_shape(service_key: str) -> tuple[str, int, int, int, str]:
    if service_key in {
        "rental:tent",
        "rental:sleeping_bag",
        "rental:sleeping_pad",
        "rental:camp_lantern",
        "rental:camp_cookware",
    }:
        return "overnight", 2, 0, 2, "adult_regular"
    if service_key in {"rental:camp_chair", "rental:camp_table"}:
        return "picnic", 3, 0, 1, "n/a"
    if service_key == "retail:rain_gear":
        return "supply", 1, 0, 0, "adult"
    return "supply", 1, 0, 0, "n/a"


def _sessions(
    environment: OutfitterEnvironment,
    state: Any,
    service_key: str,
    group: str,
    *,
    count: int,
) -> list[Any]:
    archetype, adults, children, duration, size = _session_shape(service_key)
    units = adults + children if service_key in {
        "rental:sleeping_bag",
        "rental:sleeping_pad",
        "rental:camp_chair",
    } else 1
    return [
        replace(
            environment._make_choice_session(
                state,
                service_key=service_key,
                units_requested=units,
                duration_days=duration or None,
                adults=adults,
                children=children,
                size_class=size,
                archetype=archetype,
                weather=environment._weather(1),
                request_index=0,
                event_address=(1, 9, index),
            ),
            preferred_group=group,
        )
        for index in range(count)
    ]


def _conversion(
    environment: OutfitterEnvironment,
    state: Any,
    sessions: list[Any],
    offer_ids: list[str],
) -> float:
    bought = 0
    for session in sessions:
        selection = environment._select_quoted_offer(
            state, session, offer_ids
        )
        bought += selection[3] > selection[4]
    return round(bought / len(sessions), 6)


def build_diagnostics(
    data_dir: Path,
    *,
    seed: int = 42,
    sessions_per_group: int = 1024,
    world_id: str = "base",
) -> dict[str, Any]:
    dataset = OutfitterDataset.load(data_dir)
    if dataset.channel_demand_model is not None:
        raise ValueError(
            "legacy demand_diagnostics does not audit exact-offer channel demand; "
            "use outfitter.demand_lab"
        )
    config = OutfitterConfig(
        data_dir=str(data_dir),
        max_days=28,
        weather_start_day=261,
        seed=seed,
        initial_cash=6_000.0,
        charge_model_usage_to_cash=False,
        demand_world_id=world_id,
        daily_fee=float(dataset.metadata.get("formal_daily_fee") or 60.0),
    )
    environment = OutfitterEnvironment(config, dataset)
    if environment.reference_demand_model is not None:
        groups = []
        for service_key, group in sorted(environment.by_group):
            request = _default_request(service_key, group)
            scenario = {
                "format": SCENARIO_FORMAT,
                "request": request,
                "available_units": {
                    offer.offer_id: max(1, int(request["units_requested"]))
                    for offer in environment.by_service[service_key]
                },
            }
            curve_result = lab_curve(
                data_dir,
                scenario,
                world=world_id,
                seed=seed,
                sessions=sessions_per_group,
                grid_step=0.5,
            )
            invariant_result = lab_invariance(
                data_dir,
                scenario,
                world=world_id,
                seed=seed,
            )
            checkpoints = curve_result["standard_checkpoints"]
            groups.append(
                {
                    "key": f"{service_key}|{group}",
                    "sessions": sessions_per_group,
                    "conversion_curve": checkpoints,
                    "revenue_optimum": curve_result["revenue_optimum"],
                    "invariance": invariant_result,
                    "price_monotone": all(
                        left["conversion"] >= right["conversion"]
                        for left, right in zip(checkpoints, checkpoints[1:])
                    ),
                }
            )
        passed = all(
            row["price_monotone"] and row["invariance"]["passed"]
            for row in groups
        )
        return {
            "format": "outfitter-reference-demand-diagnostics-v1",
            "data_dir": str(data_dir.resolve()),
            "seed": seed,
            "world_id": world_id,
            "groups": groups,
            "checks": {
                "all_price_curves_monotone_nonincreasing": all(
                    row["price_monotone"] for row in groups
                ),
                "all_invariance_checks_pass": all(
                    row["invariance"]["passed"] for row in groups
                ),
            },
            "passed": passed,
        }
    if environment.demand_model is None:
        raise ValueError("demand diagnostics require demand_model.json")
    state = environment.reset()
    original_prices = dict(state.prices)
    groups = []
    for (service_key, group), offers in sorted(environment.by_group.items()):
        offer_ids = sorted(offer.offer_id for offer in offers)
        sessions = _sessions(
            environment,
            state,
            service_key,
            group,
            count=sessions_per_group,
        )
        neutral = {
            offer_id: float(environment.neutral_reservation_value(offer_id) or 0.01)
            for offer_id in offer_ids
        }
        curve = []
        for ratio in PRICE_RATIOS:
            for offer_id in offer_ids:
                state.prices[offer_id] = round(neutral[offer_id] * ratio, 4)
            curve.append(
                {
                    "price_to_neutral_ratio": ratio,
                    "conversion": _conversion(
                        environment, state, sessions, offer_ids
                    ),
                }
            )
        for offer_id in offer_ids:
            offer = environment.offers[offer_id]
            state.prices[offer_id] = (
                offer.unit_cost
                if offer.asset_kind == "consumable"
                else float(offer.reference_daily_rate or 0.01)
            )
        conversion_at_cost = _conversion(
            environment, state, sessions, offer_ids
        )
        for offer_id in offer_ids:
            state.prices[offer_id] = original_prices[offer_id]
        conversion_at_opening = _conversion(
            environment, state, sessions, offer_ids
        )
        demand_key = f"{service_key}|{group}"
        original_group = environment.demand_model.groups[demand_key]
        accessibility_counterfactual = {}
        for accessibility in ("low", "medium", "high"):
            environment.demand_model.groups[demand_key] = replace(
                original_group,
                alternative_accessibility=accessibility,
            )
            accessibility_counterfactual[accessibility] = _conversion(
                environment, state, sessions, offer_ids
            )
        environment.demand_model.groups[demand_key] = original_group
        special_prices = {}
        if demand_key == "retail:cleaning_wipes|cleaning_wipes":
            for offer_id in offer_ids:
                state.prices[offer_id] = 9.41
            special_prices["conversion_at_9_41_cny"] = _conversion(
                environment, state, sessions, offer_ids
            )
            for offer_id in offer_ids:
                state.prices[offer_id] = original_prices[offer_id]
        groups.append(
            {
                "key": demand_key,
                "sessions": sessions_per_group,
                "conversion_curve": curve,
                "conversion_at_cost_recovery": conversion_at_cost,
                "conversion_at_opening_price": conversion_at_opening,
                "accessibility_counterfactual_at_opening": accessibility_counterfactual,
                **special_prices,
            }
        )
    monotone = all(
        all(
            left["conversion"] >= right["conversion"]
            for left, right in zip(
                row["conversion_curve"], row["conversion_curve"][1:]
            )
        )
        for row in groups
    )
    accessibility_ordered = all(
        row["accessibility_counterfactual_at_opening"]["low"]
        >= row["accessibility_counterfactual_at_opening"]["medium"]
        >= row["accessibility_counterfactual_at_opening"]["high"]
        for row in groups
    )
    return {
        "format": "outfitter-bounded-demand-diagnostics-v1",
        "data_dir": str(data_dir.resolve()),
        "seed": seed,
        "groups": groups,
        "checks": {
            "all_price_curves_monotone_nonincreasing": monotone,
            "all_accessibility_counterfactuals_ordered": accessibility_ordered,
        },
        "passed": monotone and accessibility_ordered,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit bounded Outfitter demand curves")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sessions-per-group", type=int, default=1024)
    parser.add_argument("--world", default="base")
    args = parser.parse_args()
    result = build_diagnostics(
        args.data_dir.expanduser().resolve(),
        seed=args.seed,
        sessions_per_group=args.sessions_per_group,
        world_id=args.world,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
