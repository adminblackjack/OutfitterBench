"""Build frozen runtime data from an audited V5 release catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .build_runtime_data import (
    INITIAL_RETAIL_TARGET_UNITS,
    WEATHER_SOURCE,
    build_weather_rows,
    read_jsonl,
    write_jsonl,
)
from .config import OutfitterConfig
from .data import OutfitterDataset
from .demand import DemandModel
from .environment import OutfitterEnvironment
from .v5 import (
    CHOICE_GROUPS_BY_SERVICE,
    RUNTIME_FORMAT,
    SCHEMA_VERSION,
    V5MarketRules,
    normalize_tent_capacity_offer,
)
from .visibility import write_product_visibility_exports


def _service_key(row: dict[str, Any]) -> str:
    keys = list(row.get("fulfills_service_keys") or [])
    if len(keys) != 1:
        raise ValueError(f"offer {row.get('offer_id')} must fulfill exactly one service")
    return str(keys[0])


def _validate_catalog(
    catalog: list[dict[str, Any]], *, bounded_demand: bool = False
) -> dict[str, int]:
    if not 66 <= len(catalog) <= 110:
        raise ValueError("formal V5 catalog must contain 66 through 110 offers")
    if len({str(row["offer_id"]) for row in catalog}) != len(catalog):
        raise ValueError("formal V5 catalog contains duplicate offer IDs")
    counts = Counter((_service_key(row), str(row["choice_group"])) for row in catalog)
    expected = {
        (service, group)
        for service, groups in CHOICE_GROUPS_BY_SERVICE.items()
        for group in groups
    }
    if set(counts) != expected:
        raise ValueError(
            f"formal V5 catalog choice groups differ: missing={sorted(expected-set(counts))}, "
            f"extra={sorted(set(counts)-expected)}"
        )
    invalid = {key: value for key, value in counts.items() if not 3 <= value <= 5}
    if invalid:
        raise ValueError(f"formal V5 groups must contain 3 through 5 offers: {invalid}")
    for row in catalog:
        if row.get("unresolved_decision_attributes"):
            raise ValueError(f"release offer {row['offer_id']} has unresolved decisions")
        if row.get("quality_tier") != "unknown":
            raise ValueError(f"release offer {row['offer_id']} has sourced quality")
        if row.get("perceived_quality_tier") not in {
            "economy",
            "standard",
            "premium",
            "unknown",
        }:
            raise ValueError(f"release offer {row['offer_id']} has invalid perceived quality")
        if row["asset_kind"] == "durable":
            if (
                int(row.get("service_life_limit_days") or 0) < 20
                or int(row.get("service_life_limit_cycles") or 0) < 5
                or row.get("damage_risk_band") not in {"low", "medium", "high"}
                or float(row.get("cleaning_work_points") or 0) <= 0
            ):
                raise ValueError(f"release durable offer {row['offer_id']} lacks priors")
        elif (
            not bounded_demand
            and float(row.get("reference_retail_price") or 0)
            < 1.5 * float(row["unit_cost"]) - 0.011
        ):
            raise ValueError(f"release retail offer {row['offer_id']} violates margin floor")
        if bounded_demand:
            required = {
                ("energy_snacks", "compressed_ration"): "net_weight_g_per_unit",
                ("energy_snacks", "snack_portion"): "net_weight_g_per_unit",
                ("instant_food", "instant_food"): "net_weight_g_per_unit",
                ("repellent", "topical_repellent"): "net_content_ml_per_unit",
                ("repellent", "wearable_repellent"): "wearable_items_per_unit",
            }.get((str(row["category_id"]), str(row["choice_group"])))
            if required and row.get("consumer_value_attributes", {}).get(required) is None:
                raise ValueError(
                    f"bounded-demand offer {row['offer_id']} lacks {required}"
                )
    return {f"{service}|{group}": counts[(service, group)] for service, group in sorted(counts)}


def _initial_assets(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_line: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for offer in catalog:
        if offer["asset_kind"] == "durable":
            by_line[str(offer["service_line"])].append(offer)
    rows = []
    next_id = 1
    for service_line, offers in sorted(by_line.items()):
        cheapest = min(offers, key=lambda row: (float(row["unit_cost"]), row["offer_id"]))
        life_days = int(cheapest["service_life_limit_days"])
        life_cycles = int(cheapest["service_life_limit_cycles"])
        typical_duration = 2 if service_line in {"tent", "sleeping_bag", "air_mattress", "self_inflating_pad", "closed_cell_foam_pad"} else 1
        for cohort, ratio in (("young", 0.12), ("midlife", 0.46), ("mature", 0.82)):
            wear_per_cycle = 1.0 / life_cycles + typical_duration / life_days
            used_cycles = max(0, round(ratio / wear_per_cycle))
            used_days = used_cycles * typical_duration
            while (
                used_cycles / life_cycles + used_days / life_days
                >= 1.0
                and used_cycles > 0
            ):
                used_cycles -= 1
                used_days = used_cycles * typical_duration
            rows.append(
                {
                    "asset_id": f"asset-{next_id:06d}",
                    "offer_id": cheapest["offer_id"],
                    "service_line": service_line,
                    "used_life_days": used_days,
                    "service_life_limit_days": life_days,
                    "used_rental_cycles": used_cycles,
                    "service_life_limit_cycles": life_cycles,
                    "damage_risk_band": cheapest["damage_risk_band"],
                    "unit_cost": cheapest["unit_cost"],
                    "age_cohort": cohort,
                }
            )
            next_id += 1
    for service_line in ("tent", "sleeping_bag"):
        candidates = [row for row in rows if row["service_line"] == service_line]
        if candidates:
            max(candidates, key=lambda row: (row["used_life_days"], row["asset_id"]))[
                "status"
            ] = "damaged"
    return rows


def _initial_retail(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    multipliers = {
        "bottled_water": 0.45,
        "energy_snacks": 1.75,
        "instant_food": 0.50,
        "rain_gear": 0.0,
        "repellent": 1.60,
        "cleaning_wipes": 1.40,
        "garbage_bags": 1.80,
    }
    for category, base_target in INITIAL_RETAIL_TARGET_UNITS.items():
        service_key = f"retail:{category}"
        offers = sorted(
            (row for row in catalog if _service_key(row) == service_key),
            key=lambda row: (
                str(row["choice_group"]),
                str(row["perceived_quality_tier"]),
                float(row["unit_cost"]),
                row["offer_id"],
            ),
        )
        target = round(base_target * multipliers[category])
        quotient, remainder = divmod(target, len(offers))
        for index, offer in enumerate(offers):
            rows.append(
                {
                    "service_line": offer["service_line"],
                    "offer_id": offer["offer_id"],
                    "quantity": quotient + int(index < remainder),
                    "unit_cost": offer["unit_cost"],
                }
            )
    return rows


def _initial_prices(
    catalog: list[dict[str, Any]], demand_model: DemandModel | None = None
) -> list[dict[str, Any]]:
    if demand_model is not None:
        return _bounded_initial_prices(catalog, demand_model)
    rows = []
    rules = V5MarketRules()
    by_group: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for offer in catalog:
        by_group[(_service_key(offer), str(offer["choice_group"]))].append(offer)
    rental_multipliers = (0.80, 0.95, 1.10, 1.25)
    retail_multipliers = (0.85, 1.00, 1.15, 1.30)
    for key, offers in sorted(by_group.items()):
        ordered = sorted(offers, key=lambda row: (float(row["unit_cost"]), row["offer_id"]))
        for index, offer in enumerate(ordered):
            if offer["asset_kind"] == "durable":
                cost_reference = float(offer["reference_daily_rate"])
                market_budget_anchor = 0.85 * float(rules.base_budgets[key[0]])
                opening_anchor = max(cost_reference, market_budget_anchor)
                multiplier = rental_multipliers[index]
                policy = (
                    "max(cost-recovery reference, 0.85x service base budget), "
                    "then within-group cost-order multiplier"
                )
            else:
                cost_reference = float(offer["reference_retail_price"])
                market_budget_anchor = None
                opening_anchor = cost_reference
                multiplier = retail_multipliers[index]
                policy = "retail reference with within-group cost-order multiplier"
            rows.append(
                {
                    "offer_id": offer["offer_id"],
                    "price": round(max(0.01, opening_anchor * multiplier), 2),
                    "cost_recovery_reference": round(cost_reference, 4),
                    "market_budget_anchor": (
                        round(market_budget_anchor, 4)
                        if market_budget_anchor is not None
                        else None
                    ),
                    "opening_anchor": round(opening_anchor, 4),
                    "reference_multiplier": multiplier,
                    "initial_price_policy": policy,
                }
            )
    return sorted(rows, key=lambda row: row["offer_id"])


def _bounded_initial_prices(
    catalog: list[dict[str, Any]], _demand_model: DemandModel
) -> list[dict[str, Any]]:
    """Set predecessor prices from public operations, never hidden customer WTP.

    Acquisition cost may inform a manager's posted price. It must not alter the
    customer's reservation distribution or utility calculation.
    """

    rows = []
    for offer in sorted(catalog, key=lambda row: row["offer_id"]):
        service_key = _service_key(offer)
        digest = hashlib.sha256(
            f"outfitter-v5-public-opening-price-v1:{offer['offer_id']}".encode(
                "utf-8"
            )
        ).digest()
        fraction = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        noise_low, noise_high = (0.85, 1.15)
        noise = noise_low + (noise_high - noise_low) * fraction
        if offer["asset_kind"] == "durable":
            duration = 2 if service_key in {
                "rental:tent",
                "rental:sleeping_bag",
                "rental:sleeping_pad",
            } else 1
            cleaning_rule = OutfitterEnvironment.public_cleaning_rule(
                str(offer["service_line"])
            )
            cleaning_per_contract = float(cleaning_rule["base_cost"]) + float(
                cleaning_rule["extra_per_additional_day"]
            ) * (duration - 1)
            cleaning_per_billed_day = cleaning_per_contract / duration
            acquisition_recovery_per_day = float(offer["unit_cost"]) / 30.0
            public_anchor = 1.15 * (
                acquisition_recovery_per_day + cleaning_per_billed_day
            )
            cost_floor = cleaning_per_billed_day + float(offer["unit_cost"]) / 60.0
            anchor_policy = (
                "public supplier cost recovered over 30 billed days plus public "
                "normal-return cleaning, with 15% operating markup"
            )
        else:
            reference = offer.get("reference_retail_price")
            public_anchor = (
                float(reference)
                if reference is not None
                else 1.50 * float(offer["unit_cost"])
            )
            cost_floor = 1.25 * float(offer["unit_cost"])
            cleaning_per_contract = None
            cleaning_per_billed_day = None
            acquisition_recovery_per_day = None
            duration = None
            anchor_policy = (
                "dimension-safe public peer/cost-plus reference with a 1.25x "
                "supplier-cost floor"
            )
        opening = max(cost_floor, public_anchor * noise)
        rows.append(
            {
                "offer_id": offer["offer_id"],
                "price": round(max(0.01, opening), 2),
                "public_operating_anchor": round(public_anchor, 4),
                "public_typical_duration_days": duration,
                "public_cleaning_per_contract": (
                    None
                    if cleaning_per_contract is None
                    else round(cleaning_per_contract, 4)
                ),
                "public_cleaning_per_billed_day": (
                    None
                    if cleaning_per_billed_day is None
                    else round(cleaning_per_billed_day, 4)
                ),
                "public_acquisition_recovery_per_day": (
                    None
                    if acquisition_recovery_per_day is None
                    else round(acquisition_recovery_per_day, 4)
                ),
                "deterministic_predecessor_noise": round(noise, 6),
                "cost_recovery_floor": round(cost_floor, 4),
                "cost_floor_policy": anchor_policy,
                "initial_price_policy": (
                    "max(public operating floor, public operating anchor times "
                    "demand-independent deterministic predecessor noise)"
                ),
            }
        )
    return rows


def _cash_recommendation(catalog: list[dict[str, Any]], daily_fee: float = 60.0) -> dict[str, Any]:
    by_group: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in catalog:
        by_group[(_service_key(row), str(row["choice_group"]))].append(
            float(row["package_price"])
        )
    starters = sorted(min(values) for values in by_group.values())
    middle = max(0, len(starters) // 2 - 1)
    experiments = sum(starters[middle : middle + 3])
    raw = 14 * daily_fee + experiments
    recommended = max(1_000, int(math.ceil(raw / 1000.0) * 1000))
    return {
        "fourteen_day_fee_reserve": round(14 * daily_fee, 2),
        "three_median_group_experiments": round(experiments, 2),
        "refund_pressure": "pending 200-day multi-seed calibration",
        "minimum_rounding_floor": 1_000,
        "recommended_initial_cash_rounded": recommended,
        "policy": (
            "liquidity pressure only; initial cash is not a score denominator or "
            "profitability calibration parameter"
        ),
    }


def build(
    *,
    release_catalog: Path,
    output_dir: Path,
    calibration_result: Path | None = None,
    demand_model_path: Path | None = None,
) -> dict[str, Any]:
    release_rows = [
        normalize_tent_capacity_offer(row) for row in read_jsonl(release_catalog)
    ]
    demand_model = (
        DemandModel.from_dict(
            json.loads(demand_model_path.read_text(encoding="utf-8"))
        )
        if demand_model_path is not None
        else None
    )
    group_counts = _validate_catalog(
        release_rows, bounded_demand=demand_model is not None
    )
    catalog = []
    for source in release_rows:
        row = dict(source)
        if demand_model is not None and row["asset_kind"] == "consumable":
            row.pop("reference_retail_price", None)
            row.pop("reference_retail_price_method", None)
        catalog.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "catalog.jsonl", catalog)
    write_jsonl(output_dir / "initial_assets.jsonl", _initial_assets(catalog))
    write_jsonl(output_dir / "initial_retail_inventory.jsonl", _initial_retail(catalog))
    write_jsonl(
        output_dir / "initial_prices.jsonl",
        _initial_prices(
            # Bounded runtime catalogs deliberately omit retail reference prices;
            # the predecessor-price projection consumes the audited release copy
            # before that internal QA field is removed.
            release_rows if demand_model is not None else catalog,
            demand_model,
        ),
    )
    weather = build_weather_rows()
    write_jsonl(output_dir / "weather.jsonl", weather)
    calibration = None
    scale_multiplier = 1.0
    calibrated_daily_fee = None
    if calibration_result is not None:
        calibration = json.loads(calibration_result.read_text(encoding="utf-8"))
        profit_gate = calibration.get("scripted_profit_gate")
        if not isinstance(profit_gate, dict) or profit_gate.get("passed") is not True:
            raise ValueError("calibration result has not passed the scripted profit gate")
        scale = calibration.get("calibration_scale") or {}
        scale_multiplier = float(scale.get("arrival_scale_multiplier", 1.0))
        calibrated_daily_fee = float(scale["daily_fee"])
    rules_payload = V5MarketRules().to_dict()
    rules_payload["arrival_scale"] = round(
        float(rules_payload["arrival_scale"]) * scale_multiplier, 6
    )
    rules = V5MarketRules.from_dict(rules_payload)
    (output_dir / "market_rules.json").write_text(
        json.dumps(rules.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if demand_model is not None:
        (output_dir / "demand_model.json").write_text(
            json.dumps(
                demand_model.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
    weather_destination = output_dir / WEATHER_SOURCE.name
    if (
        WEATHER_SOURCE.is_file()
        and WEATHER_SOURCE.resolve() != weather_destination.resolve()
    ):
        shutil.copy2(WEATHER_SOURCE, weather_destination)
    hash_names = (
        "catalog.jsonl",
        "initial_assets.jsonl",
        "initial_retail_inventory.jsonl",
        "initial_prices.jsonl",
        "market_rules.json",
        "weather.jsonl",
    ) + (("demand_model.json",) if demand_model is not None else ())
    hashes = {
        name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        for name in hash_names
    }
    liquidity = _cash_recommendation(catalog)
    release_status = "v5-formal-candidate-pending-200-day-calibration"
    if calibration_result is not None:
        assert calibration is not None
        calibrated_liquidity = calibration.get("liquidity_recommendation")
        if not isinstance(calibrated_liquidity, dict) or int(
            calibrated_liquidity.get("recommended_initial_cash_rounded", 0)
        ) <= 0:
            raise ValueError("calibration result lacks a valid liquidity recommendation")
        liquidity = {
            **calibrated_liquidity,
            "calibration_source": str(calibration_result.resolve()),
        }
        release_status = "v5-formal-calibrated-candidate"
    metadata = {
        "format": RUNTIME_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "catalog_offers": len(catalog),
        "choice_groups": len(group_counts),
        "offers_by_choice_group": group_counts,
        "service_lines": len({row["service_line"] for row in catalog}),
        "source": str(release_catalog.resolve()),
        "source_quality_policy": "unknown/source_missing",
        "perceived_quality_policy": "operational_estimate panel majority",
        "retail_reference_price_method": (
            "not used by bounded customer demand; acquisition-price statistics remain "
            "release QA only"
            if demand_model is not None
            else "same-group dimension-safe peer median when available, otherwise fixed cost-plus; minimum 1.50x unit cost"
        ),
        "initial_price_policy": (
            "bounded demand: predecessor prices use public supplier cost, public "
            "cleaning charges, a fixed billed-day recovery horizon, dimension-safe "
            "retail references, and demand-independent offer noise; no reservation "
            "value, hidden quality, accessibility, or price sensitivity"
            if demand_model is not None
            else "rental: max(reference_daily_rate, 0.85x service base budget) times 0.80/0.95/1.10/1.25 in within-group cost order; retail retains 0.85/1.00/1.15/1.30 reference multipliers"
        ),
        "initial_liquidity": liquidity,
        "formal_weather_start_day": 91,
        "formal_daily_fee": calibrated_daily_fee,
        "calibrated_arrival_scale": rules.arrival_scale,
        "formal_max_days": 200,
        "benchmark_calendar_year": 2026,
        "weather_source_year": 2023,
        "weather_mapping": "same_month_day_historical_replay",
        "weather_source": "Beijing Visual Crossing daily weather, 2023-01-01 through 2023-12-31",
        "weather_days": len(weather),
        "content_sha256": hashes,
        "release_status": release_status,
        "demand_model": (
            None
            if demand_model is None
            else {
                "format": demand_model.format,
                "source": str(demand_model_path.resolve()),
                "supplier_cost_enters_customer_demand": False,
            }
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    visibility_environment = OutfitterEnvironment(
        OutfitterConfig(
            data_dir=str(output_dir),
            max_days=200,
            weather_start_day=91,
            daily_fee=(60.0 if calibrated_daily_fee is None else calibrated_daily_fee),
        ),
        OutfitterDataset.load(output_dir),
    )
    sales_path, public_path = write_product_visibility_exports(
        visibility_environment, output_dir
    )
    for path in (sales_path, public_path):
        metadata["content_sha256"][path.name] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    metadata["product_visibility_exports"] = {
        "internal_sales_model": sales_path.name,
        "agent_visible_projection": public_path.name,
        "agent_visible_source": "ToolExecutor.inspect_offer over initial runtime state",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V5 formal runtime candidate")
    parser.add_argument("--release-catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-result", type=Path)
    parser.add_argument("--demand-model", type=Path)
    args = parser.parse_args()
    metadata = build(
        release_catalog=args.release_catalog.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        calibration_result=(
            None
            if args.calibration_result is None
            else args.calibration_result.expanduser().resolve()
        ),
        demand_model_path=(
            None
            if args.demand_model is None
            else args.demand_model.expanduser().resolve()
        ),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
