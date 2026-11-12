"""Build the compact, frozen catalog used by OutfitterBench.

This is intentionally a one-way projection of the validation artifacts.  The
benchmark never visits the web and never calls an auxiliary LLM at run time.
"""

from __future__ import annotations

from .jsonio import read_jsonl, write_jsonl

import json
import hashlib
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from .v5 import (
    MARKET_RULES_VERSION,
    RUNTIME_FORMAT,
    SCHEMA_VERSION,
    V5MarketRules,
    calendar_for_source_day,
    decision_signature,
    infer_v5_offer_fields,
)


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "data_validation" / "artifacts"
FULL = ARTIFACTS / "expanded-20260810"
PADS = ARTIFACTS / "sleeping-pad-subtypes-v8-20260810"
RETAIL = ARTIFACTS / "retail-units-v2-20260817"
OUTPUT = ROOT / "runtime_data"
WEATHER_SOURCE = OUTPUT / "weather_finalinfo.pkl"
SOURCE_WEATHER_CONDITIONS = {"sunny", "cloudy", "rainy", "hot", "cold", "snowy"}
PUBLIC_WEATHER_CONDITIONS = {"sunny", "cloudy", "rainy", "snowy"}

RENTAL_CATEGORIES = (
    "tent",
    "sleeping_bag",
    "camp_table",
    "camp_chair",
    "camp_lantern",
    "camp_cookware",
)
RETAIL_CATEGORIES = (
    "bottled_water",
    "energy_snacks",
    "instant_food",
    "rain_gear",
    "repellent",
    "cleaning_wipes",
    "garbage_bags",
)
INITIAL_RETAIL_TARGET_UNITS = {
    "bottled_water": 24,
    "energy_snacks": 18,
    "instant_food": 12,
    "rain_gear": 6,
    "repellent": 6,
    "cleaning_wipes": 12,
    "garbage_bags": 6,
}
PAD_LABELS = {
    "closed_cell_foam_pad": "睡垫（闭孔泡沫）",
    "self_inflating_pad": "睡垫（自充气）",
    "air_mattress": "睡垫（纯充气）",
}






def _public_weather_condition(row: Any) -> str:
    source_condition = str(row.weather_true or "").casefold()
    raw_icon = str(row.raw_icon or "").casefold()
    precipitation = float(row.precip or 0.0)
    snow = float(row.snow or 0.0)
    if source_condition == "snowy" or snow > 0 or "snow" in raw_icon:
        return "snowy"
    if source_condition == "rainy" or precipitation > 0 or any(
        token in raw_icon for token in ("rain", "showers", "thunder")
    ):
        return "rainy"
    if "clear" in raw_icon:
        return "sunny"
    return "cloudy"


def build_weather_rows() -> list[dict[str, Any]]:
    """Replay frozen Beijing 2023 weather values on the real 2026 calendar."""
    import pandas as pd

    if not WEATHER_SOURCE.is_file():
        raise FileNotFoundError(f"frozen weather source is missing: {WEATHER_SOURCE}")
    frame = pd.read_pickle(WEATHER_SOURCE)
    required = {
        "day_index",
        "date",
        "weather_true",
        "temp",
        "raw_icon",
        "precip",
        "snow",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"weather source is missing columns: {missing_columns}")
    if len(frame) != 365:
        raise ValueError(f"weather source must contain 365 days, got {len(frame)}")

    day_indices = [int(value) for value in frame["day_index"].tolist()]
    if day_indices != list(range(365)):
        raise ValueError("weather day_index must be contiguous from 0 through 364")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any() or dates.nunique() != 365:
        raise ValueError("weather dates must contain 365 distinct valid dates")
    if any(delta.days != 1 for delta in dates.diff().dropna()):
        raise ValueError("weather dates must be consecutive")

    conditions = [str(value) for value in frame["weather_true"].tolist()]
    unsupported = sorted(set(conditions) - SOURCE_WEATHER_CONDITIONS)
    if unsupported:
        raise ValueError(f"weather source contains unsupported conditions: {unsupported}")
    means = [float(value) for value in frame["temp"].tolist()]
    base_ranges = {1: 10, 2: 11, 3: 12, 4: 12, 5: 11, 6: 10,
                   7: 8, 8: 8, 9: 10, 10: 11, 11: 11, 12: 10}
    adjustments = {"sunny": 2, "cloudy": -1, "rainy": -3, "snowy": -2}
    rows = []
    for row, day_index, source_date, source_condition, mean in zip(
        frame.itertuples(index=False), day_indices, dates, conditions, means, strict=True
    ):
        day = day_index + 1
        calendar = calendar_for_source_day(day)
        condition = _public_weather_condition(row)
        if condition not in PUBLIC_WEATHER_CONDITIONS:
            raise AssertionError(f"invalid public weather condition: {condition}")
        span = base_ranges[int(source_date.month)] + adjustments[condition]
        rows.append(
            {
                "day": day,
                **calendar,
                "condition": condition,
                "temp_mean_c": round(mean, 2),
                "night_low_c": round(mean - 0.5 * span, 2),
                "day_high_c": round(mean + 0.5 * span, 2),
                "weather_source_date": source_date.date().isoformat(),
                "weather_source_condition": source_condition,
            }
        )
    return rows


def upgrade_runtime_calendar(output: Path) -> dict[str, Any]:
    """Upgrade an existing V5 runtime without rebuilding its product catalog."""

    output = output.expanduser().resolve()
    metadata_path = output / "metadata.json"
    rules_path = output / "market_rules.json"
    if not metadata_path.is_file() or not rules_path.is_file():
        raise FileNotFoundError("calendar upgrade requires metadata.json and market_rules.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_rules = V5MarketRules.from_dict(
        json.loads(rules_path.read_text(encoding="utf-8"))
    )
    defaults = V5MarketRules()
    rules = replace(
        source_rules,
        version=MARKET_RULES_VERSION,
        holiday_arrival_multipliers=dict(defaults.holiday_arrival_multipliers),
    )
    write_jsonl(output / "weather.jsonl", build_weather_rows())
    rules_path.write_text(
        json.dumps(rules.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    metadata.update(
        {
            "benchmark_calendar_year": 2026,
            "weather_source_year": 2023,
            "weather_mapping": "same_month_day_historical_replay",
            "weather_source": (
                "Beijing Visual Crossing daily weather, "
                "2023-01-01 through 2023-12-31"
            ),
            "weather_days": 365,
        }
    )
    hash_names = tuple(metadata.get("content_sha256") or ())
    metadata["content_sha256"] = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in hash_names
        if (output / name).is_file()
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def compact_offer(
    row: dict[str, Any],
    *,
    service_line: str,
    category_label: str | None = None,
    lifecycle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    units = int(row["package_units"])
    offer = {
        "offer_id": row["offer_id"],
        "service_line": service_line,
        "category_id": row["category_id"],
        "category_label_zh": category_label or row["category_label_zh"],
        "asset_kind": row["asset_kind"],
        "name": row["normalized_name"],
        "package_price": round(float(row["package_price"]), 2),
        "package_units": units,
        "unit_cost": round(float(row["package_price"]) / units, 4),
        "seller_id": str(row.get("seller_id") or ""),
        "seller_name": row.get("seller_name") or "未命名供应商",
        "source_title": row.get("source_title") or "",
        "source_variant_name": row.get("source_variant_name") or "",
        "evidence": [
            {"name": f["name"], "value": f.get("value"), "evidence": f.get("evidence", "")}
            for f in row.get("facts", [])
            if f.get("evidence_kind") == "explicit"
        ],
    }
    for field in (
        "consumer_unit",
        "product_subtype",
        "fulfills_service_keys",
        "reference_retail_price",
        "quality_tier",
    ):
        if row.get(field) is not None:
            offer[field] = row[field]
    if lifecycle is not None:
        offer.update(
            service_life_limit_days=int(
                lifecycle["service_life_limit_equivalent_rental_days"]
            ),
            damage_risk_band=lifecycle["damage_risk_band"],
            lifecycle_prior_scope=lifecycle.get("prior_scope", "exact_offer"),
        )
        offer["reference_daily_rate"] = round(
            offer["unit_cost"]
            / max(1.0, float(offer["service_life_limit_days"]) * 0.35)
            * 1.35,
            4,
        )
    group, decision, descriptive, tier = infer_v5_offer_fields(
        category_id=str(offer["category_id"]),
        service_line=service_line,
        name=str(offer["name"]),
        product_subtype=offer.get("product_subtype"),
        quality_tier=offer.get("quality_tier"),
        evidence=list(offer["evidence"]),
    )
    offer.update(
        choice_group=group,
        decision_attributes=decision,
        descriptive_attributes=descriptive,
        perceived_quality_tier=tier,
        decision_signature=decision_signature(group, decision, tier),
    )
    return offer


def quantile_three(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row["package_price"]) / int(row["package_units"]))
    indices = [round((len(ordered) - 1) * q) for q in (0.25, 0.5, 0.75)]
    chosen: list[dict[str, Any]] = []
    for index in indices:
        candidate = ordered[index]
        if candidate["offer_id"] not in {row["offer_id"] for row in chosen}:
            chosen.append(candidate)
    for candidate in ordered:
        if len(chosen) == 3:
            break
        if candidate["offer_id"] not in {row["offer_id"] for row in chosen}:
            chosen.append(candidate)
    if len(chosen) != 3:
        raise ValueError("category does not contain three distinct accepted offers")
    return chosen


def build() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    full_offers = {
        row["offer_id"]: row
        for row in read_jsonl(FULL / "normalized_offers.jsonl")
        if row.get("accepted")
    }
    retail_source_offers = dict(full_offers)
    enriched_source = RETAIL / "normalized_offers.jsonl"
    if enriched_source.is_file():
        retail_source_offers.update(
            {
                row["offer_id"]: row
                for row in read_jsonl(enriched_source)
                if row.get("accepted")
            }
        )
    full_priors = [
        row
        for row in read_jsonl(FULL / "lifecycle_priors.jsonl")
        if row.get("asset_kind") == "durable"
        and row.get("successful_repeats") == 3
        and not row.get("needs_review")
    ]
    catalog: list[dict[str, Any]] = []
    for prior in full_priors:
        if prior["category_id"] not in RENTAL_CATEGORIES:
            continue
        catalog.append(
            compact_offer(
                full_offers[prior["offer_id"]],
                service_line=prior["category_id"],
                lifecycle=prior,
            )
        )

    pad_offers = {
        row["offer_id"]: row
        for row in read_jsonl(PADS / "normalized_offers.jsonl")
        if row.get("accepted")
    }
    pad_priors = {
        row["product_subtype"]: row
        for row in read_jsonl(PADS / "lifecycle_priors.jsonl")
        if row.get("asset_kind") == "durable" and not row.get("needs_review")
    }
    for subtype, prior in pad_priors.items():
        for offer_id in prior["representative_offer_ids"]:
            catalog.append(
                compact_offer(
                    pad_offers[offer_id],
                    service_line=subtype,
                    category_label=PAD_LABELS[subtype],
                    lifecycle=prior,
                )
            )

    selection = json.loads((RETAIL / "selection.json").read_text(encoding="utf-8"))
    if not isinstance(selection, list):
        raise ValueError("retail selection must be a JSON array")
    retail_rows: list[dict[str, Any]] = []
    for override in selection:
        offer_id = str(override["offer_id"])
        source = retail_source_offers.get(offer_id)
        if source is None:
            raise ValueError(f"audited retail offer is not accepted in source data: {offer_id}")
        retail_rows.append({**source, **override})
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in retail_rows:
        by_category[row["category_id"]].append(row)
    for category in RETAIL_CATEGORIES:
        rows = by_category[category]
        if len(rows) != 3:
            raise ValueError(
                f"retail category {category} must contain exactly three audited offers"
            )
        for row in rows:
            if int(row.get("reference_peer_count", 0)) < 5:
                raise ValueError(
                    f"retail offer {row.get('offer_id')} lacks five independent price peers"
                )
            if row.get("fulfills_service_keys") != [f"retail:{category}"]:
                raise ValueError(
                    f"retail offer {row.get('offer_id')} has an invalid service mapping"
                )
            # Retail goods deliberately have no lifespan or damage prior.
            offer = compact_offer(
                row,
                service_line=str(row.get("product_subtype") or category),
            )
            offer["asset_kind"] = "consumable"
            catalog.append(offer)

    catalog.sort(key=lambda row: (row["asset_kind"] != "durable", row["service_line"], row["unit_cost"]))
    if len(catalog) != 48:
        raise ValueError(f"expected 48 runtime offers, got {len(catalog)}")
    write_jsonl(OUTPUT / "catalog.jsonl", catalog)

    ages = read_jsonl(FULL / "initial_age_cohorts.jsonl")
    pad_ages = read_jsonl(PADS / "initial_age_cohorts.jsonl")
    initial_assets: list[dict[str, Any]] = []
    durable_by_line: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for offer in catalog:
        if offer["asset_kind"] == "durable":
            durable_by_line[offer["service_line"]].append(offer)
    next_initial_asset_id = 1
    for service_line, offers in durable_by_line.items():
        cheapest = min(offers, key=lambda row: row["unit_cost"])
        if service_line in PAD_LABELS:
            prior_id = pad_priors[service_line]["offer_id"]
            cohort_rows = [row for row in pad_ages if row["offer_id"] == prior_id]
        else:
            cohort_rows = [row for row in ages if row["offer_id"] == cheapest["offer_id"]]
        selected = {row["cohort"]: row for row in cohort_rows}
        for cohort in ("young", "midlife", "mature"):
            row = selected[cohort]
            initial_assets.append(
                {
                    "asset_id": f"asset-{next_initial_asset_id:06d}",
                    "offer_id": cheapest["offer_id"],
                    "service_line": service_line,
                    "used_life_days": int(row["used_life_equivalent_rental_days"]),
                    "service_life_limit_days": int(row["service_life_limit_equivalent_rental_days"]),
                    "damage_risk_band": cheapest["damage_risk_band"],
                    "unit_cost": cheapest["unit_cost"],
                }
            )
            next_initial_asset_id += 1
    damaged_lines: set[str] = set()
    for row in sorted(
        initial_assets,
        key=lambda value: (
            -int(value["used_life_days"]) / max(1, int(value["service_life_limit_days"])),
            value["asset_id"],
        ),
    ):
        if row["service_line"] not in {"tent", "sleeping_bag"}:
            continue
        if row["service_line"] in damaged_lines:
            continue
        row["status"] = "damaged"
        damaged_lines.add(str(row["service_line"]))
    write_jsonl(OUTPUT / "initial_assets.jsonl", initial_assets)

    initial_retail = []
    for category in RETAIL_CATEGORIES:
        service_key = f"retail:{category}"
        offers = [
            row
            for row in catalog
            if service_key in row.get("fulfills_service_keys", [])
        ]
        target_multiplier = {
            "bottled_water": 0.45,
            "energy_snacks": 1.75,
            "instant_food": 0.50,
            "rain_gear": 0.0,
            "repellent": 1.60,
            "cleaning_wipes": 1.40,
            "garbage_bags": 1.80,
        }[category]
        target = round(INITIAL_RETAIL_TARGET_UNITS[category] * target_multiplier)
        quotient, remainder = divmod(target, len(offers))
        for index, offer in enumerate(
            sorted(offers, key=lambda row: (row["quality_tier"], row["unit_cost"]))
        ):
            initial_retail.append(
                {
                    "service_line": offer["service_line"],
                    "offer_id": offer["offer_id"],
                    "quantity": quotient + int(index < remainder),
                    "unit_cost": offer["unit_cost"],
                }
            )
    write_jsonl(OUTPUT / "initial_retail_inventory.jsonl", initial_retail)

    price_multipliers = (0.70, 0.85, 1.00, 1.25, 1.45)
    groups = sorted({str(row["choice_group"]) for row in catalog})
    multiplier_by_group = {
        group: price_multipliers[index % len(price_multipliers)]
        for index, group in enumerate(groups)
    }
    initial_prices = []
    for offer in sorted(catalog, key=lambda row: row["offer_id"]):
        reference = (
            float(offer["reference_retail_price"])
            if offer["asset_kind"] == "consumable"
            else float(offer["reference_daily_rate"])
        )
        initial_prices.append(
            {
                "offer_id": offer["offer_id"],
                "price": round(reference * multiplier_by_group[str(offer["choice_group"])], 2),
                "reference_multiplier": multiplier_by_group[str(offer["choice_group"])],
            }
        )
    write_jsonl(OUTPUT / "initial_prices.jsonl", initial_prices)

    weather = build_weather_rows()
    write_jsonl(OUTPUT / "weather.jsonl", weather)
    (OUTPUT / "market_rules.json").write_text(
        json.dumps(V5MarketRules().to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    hash_names = (
        "catalog.jsonl",
        "initial_assets.jsonl",
        "initial_retail_inventory.jsonl",
        "initial_prices.jsonl",
        "market_rules.json",
        "weather.jsonl",
    )
    content_hashes = {
        name: hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest()
        for name in hash_names
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(
            {
                "format": RUNTIME_FORMAT,
                "schema_version": SCHEMA_VERSION,
                "catalog_offers": len(catalog),
                "service_lines": len({row["service_line"] for row in catalog}),
                "source": "validated rental artifacts (2026-08-10) and audited retail units (2026-08-17)",
                "retail_source": "data_validation/artifacts/retail-units-v2-20260817",
                "retail_reference_price_method": (
                    "1.20 times a leave-one-listing-out median of at least five "
                    "subtype peers, normalized by consumer quantity where applicable"
                ),
                "formal_weather_start_day": 91,
                "formal_max_days": 200,
                "benchmark_calendar_year": 2026,
                "weather_source_year": 2023,
                "weather_mapping": "same_month_day_historical_replay",
                "weather_source": "Beijing Visual Crossing daily weather, 2023-01-01 through 2023-12-31",
                "weather_days": len(weather),
                "content_sha256": content_hashes,
                "release_status": "v5-transitional-catalog-pending-expanded-audit",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    build()
