"""Build a reusable exact-offer channel-demand runtime from a validated V5 runtime."""

from __future__ import annotations

from .jsonio import (
    read_json_object as _read_json,
    read_jsonl as _read_jsonl,
    write_json as _write_json,
    write_jsonl as _write_jsonl,
)

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from .channel_demand import ChannelDemandModel, default_channel_model_payload
from .channel_calibration import calibrate_channel_model
from .build_runtime_data import build_weather_rows
from .config import OutfitterConfig
from .data import OutfitterDataset
from .environment import OutfitterEnvironment
from .visibility import write_product_visibility_exports
from .v5 import (
    MARKET_RULES_VERSION,
    V5MarketRules,
    decision_signature,
    normalize_tent_capacity_offer,
)


BUILDER_VERSION = "outfitter-exact-offer-channel-runtime-builder-v13"
RAINWEAR_SIZE_RESOLUTION_FORMAT = "outfitter-rainwear-size-resolution-v1"
OFFER_ID_REGISTRY_FORMAT = "outfitter-offer-id-registry-v1"
SHORT_OFFER_ID = re.compile(r"offer-([0-9]{3,})$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact_identity(row: dict[str, Any]) -> str:
    payload = {
        "offer_id": row["offer_id"],
        "seller_id": row["seller_id"],
        "source_title": row["source_title"],
        "source_variant_name": row["source_variant_name"],
        "package_units": row["package_units"],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _offer_registry_identity(row: dict[str, Any]) -> str:
    payload = {
        "seller_id": str(row["seller_id"]),
        "source_title": str(row["source_title"]),
        "source_variant_name": str(row["source_variant_name"]),
        "package_units": int(row["package_units"]),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assign_short_offer_ids(
    source: Path,
    catalog: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, Any]]:
    registry_path = source / "offer_id_registry.json"
    if registry_path.is_file():
        registry = _read_json(registry_path)
        if registry.get("format") != OFFER_ID_REGISTRY_FORMAT:
            raise ValueError("unsupported offer ID registry format")
        entries = [dict(row) for row in registry.get("entries", [])]
    else:
        entries = []
    identities = [_offer_registry_identity(row) for row in catalog]
    if len(set(identities)) != len(identities):
        raise ValueError("catalog contains duplicate exact listing identities")
    by_identity: dict[str, dict[str, Any]] = {}
    used_ids: set[str] = set()
    maximum = 0
    for entry in entries:
        identity = str(entry.get("listing_identity_sha256") or "")
        offer_id = str(entry.get("offer_id") or "")
        match = SHORT_OFFER_ID.fullmatch(offer_id)
        if not identity or identity in by_identity or match is None or offer_id in used_ids:
            raise ValueError("offer ID registry contains duplicate or invalid entries")
        by_identity[identity] = entry
        used_ids.add(offer_id)
        maximum = max(maximum, int(match.group(1)))
        entry["active"] = False
    source_to_short: dict[str, str] = {}
    for row in sorted(catalog, key=lambda value: str(value["offer_id"])):
        source_offer_id = str(row["offer_id"])
        identity = _offer_registry_identity(row)
        entry = by_identity.get(identity)
        if entry is None:
            maximum += 1
            offer_id = f"offer-{maximum:03d}"
            entry = {
                "offer_id": offer_id,
                "listing_identity_sha256": identity,
                "first_seen_source_offer_id": source_offer_id,
                "active": True,
            }
            entries.append(entry)
            by_identity[identity] = entry
            used_ids.add(offer_id)
        else:
            entry["active"] = True
        source_to_short[source_offer_id] = str(entry["offer_id"])
    entries.sort(key=lambda row: int(SHORT_OFFER_ID.fullmatch(str(row["offer_id"])).group(1)))
    return source_to_short, {
        "format": OFFER_ID_REGISTRY_FORMAT,
        "policy": "append_only; removed numbers are never reused; catalog reorder never renumbers",
        "entries": entries,
    }


def _with_online_offer(
    row: dict[str, Any],
    *,
    verified_at: str,
    override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(row)
    package_price = float(row["package_price"])
    package_units = int(row["package_units"])
    if package_price <= 0 or package_units <= 0:
        raise ValueError(f"invalid exact package economics: {row['offer_id']}")
    variant = str(row.get("source_variant_name") or "").strip()
    if not variant:
        raise ValueError(f"missing exact source variant: {row['offer_id']}")
    online = dict(
        override
        or row.get("consumer_online_offer")
        or {
            "package_price_cny": round(package_price, 6),
            "consumer_units_per_package": package_units,
            "currency": "CNY",
            "provenance": "verified_exact_source_listing",
            "verified_at": verified_at,
            "exact_sku_identity_sha256": _exact_identity(row),
            "evidence_fields": [
                "seller_id",
                "source_title",
                "source_variant_name",
                "package_price",
                "package_units",
            ],
        }
    )
    online_price = float(online.get("package_price_cny", 0.0))
    online_units = int(online.get("consumer_units_per_package", 0))
    if online_price <= 0 or online_units <= 0 or online_units != package_units:
        raise ValueError(f"invalid consumer online offer: {row['offer_id']}")
    online["unit_price_cny"] = round(online_price / online_units, 6)
    online.setdefault("currency", "CNY")
    online.setdefault("verified_at", verified_at)
    expected_identity = _exact_identity(row)
    if online.get("exact_sku_identity_sha256") not in {None, expected_identity}:
        raise ValueError(f"online offer exact SKU mismatch: {row['offer_id']}")
    online["exact_sku_identity_sha256"] = expected_identity
    if not str(online.get("provenance") or "").strip():
        raise ValueError(f"online offer lacks provenance: {row['offer_id']}")
    if not online.get("evidence_fields"):
        raise ValueError(f"online offer lacks evidence fields: {row['offer_id']}")
    result["consumer_online_offer"] = online
    return result


def _with_rainwear_size_resolution(
    row: dict[str, Any], resolution: dict[str, Any]
) -> dict[str, Any]:
    offer_id = str(row["offer_id"])
    if resolution.get("format") != RAINWEAR_SIZE_RESOLUTION_FORMAT:
        raise ValueError(f"unsupported rainwear size resolution: {offer_id}")
    if str(resolution.get("offer_id")) != offer_id:
        raise ValueError(f"rainwear size resolution offer mismatch: {offer_id}")
    if row.get("category_id") != "rain_gear":
        raise ValueError(f"rainwear size resolution targets another category: {offer_id}")
    if resolution.get("exact_sku_identity_sha256") != _exact_identity(row):
        raise ValueError(f"rainwear size resolution exact SKU mismatch: {offer_id}")
    size_class = str(resolution.get("size_class") or "")
    if size_class not in {"adult", "child", "adult_child_universal"}:
        raise ValueError(f"invalid resolved rainwear size: {offer_id}")
    evidence_sources = resolution.get("evidence_sources")
    if not isinstance(evidence_sources, list) or len(evidence_sources) < 2:
        raise ValueError(f"rainwear size resolution lacks evidence: {offer_id}")
    if not all(
        isinstance(source, dict)
        and str(source.get("evidence") or "").strip()
        and str(source.get("source_key") or "").strip()
        and str(source.get("source_kind") or "").strip()
        for source in evidence_sources
    ):
        raise ValueError(f"rainwear size resolution has invalid evidence: {offer_id}")
    result = dict(row)
    attributes = dict(result.get("decision_attributes") or {})
    attributes["size_class"] = size_class
    result["decision_attributes"] = attributes
    evidence = dict(result.get("decision_evidence") or {})
    evidence["size_class"] = {
        "value": size_class,
        "derivation": str(resolution["derivation"]),
        "sources": [dict(source) for source in evidence_sources],
        "adjudication": {
            "decision": str(resolution["decision"]),
            "reviewed_at": str(resolution["reviewed_at"]),
            "source_artifact": str(resolution["source_artifact"]),
            "source_artifact_sha256": str(
                resolution["source_artifact_sha256"]
            ),
        },
    }
    result["decision_evidence"] = evidence
    result["unresolved_decision_attributes"] = [
        name
        for name in result.get("unresolved_decision_attributes") or []
        if name != "size_class"
    ]
    tier = str(result.get("perceived_quality_tier") or "unknown")
    result["decision_signature"] = decision_signature(
        str(result["choice_group"]), attributes, tier
    )
    return result


def _group_policies(source: Path) -> list[dict[str, Any]]:
    demand_path = source / "demand_model.json"
    if not demand_path.is_file():
        raise ValueError(
            "source runtime must retain demand_model.json so quality/substitution "
            "policies can be audited and the control world remains reproducible"
        )
    demand = _read_json(demand_path)
    rows = list(demand.get("group_parameters") or [])
    if not rows:
        raise ValueError("source demand model has no group parameters")
    return [
        {
            "service_key": str(row["service_key"]),
            "choice_group": str(row["choice_group"]),
            "quality_salience": str(row["quality_salience"]),
            "substitution_strength": str(row["substitution_strength"]),
        }
        for row in rows
    ]


def _content_hashes(output: Path) -> dict[str, str]:
    names = (
        "agent_visible_product_data.csv",
        "catalog.jsonl",
        "channel_demand_model.json",
        "channel_parameter_calibration.json",
        "demand_model.json",
        "initial_assets.jsonl",
        "initial_prices.jsonl",
        "initial_retail_inventory.jsonl",
        "market_rules.json",
        "offer_id_registry.json",
        "rainwear_size_resolutions.jsonl",
        "report.md",
        "result.json",
        "sales_model_product_data.csv",
        "weather.jsonl",
    )
    return {
        name: _sha256(output / name)
        for name in names
        if (output / name).is_file()
    }


def build_channel_runtime(
    source: Path,
    output: Path,
    *,
    verified_at: str,
    recommended_initial_cash: int | None = None,
    formal_daily_fee: float | None = None,
    consumer_online_offers_path: Path | None = None,
    rainwear_size_resolutions_path: Path | None = None,
    calibration_result_paths: list[Path] | None = None,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if output == source or source in output.parents:
        raise ValueError("output must not be the source runtime or a child of it")
    output.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir()):
        if not item.is_file() or item.name in {
            "metadata.json",
            "catalog.jsonl",
            "initial_prices.jsonl",
            "agent_visible_product_data.csv",
            "sales_model_product_data.csv",
            "channel_demand_model.json",
        }:
            continue
        shutil.copy2(item, output / item.name)

    # Calendar and weather are benchmark infrastructure, not catalog evidence.
    # Rebuild them on every channel projection so an older validated catalog can
    # still be promoted onto the current public calendar contract.
    _write_jsonl(output / "weather.jsonl", build_weather_rows())
    source_rules = V5MarketRules.from_dict(_read_json(source / "market_rules.json"))
    defaults = V5MarketRules()
    projected_rules = replace(
        source_rules,
        version=MARKET_RULES_VERSION,
        holiday_arrival_multipliers=dict(defaults.holiday_arrival_multipliers),
    )
    _write_json(output / "market_rules.json", projected_rules.to_dict())

    overrides = (
        {
            str(row["offer_id"]): {
                key: value for key, value in row.items() if key != "offer_id"
            }
            for row in _read_jsonl(consumer_online_offers_path.expanduser().resolve())
        }
        if consumer_online_offers_path is not None
        else {}
    )
    source_catalog = [
        normalize_tent_capacity_offer(row)
        for row in _read_jsonl(source / "catalog.jsonl")
    ]
    source_ids = {str(row["offer_id"]) for row in source_catalog}
    if extra := sorted(set(overrides) - source_ids):
        raise ValueError(f"online-offer overrides contain unknown offer IDs: {extra}")
    inherited_size_resolutions = source / "rainwear_size_resolutions.jsonl"
    resolution_rows = (
        _read_jsonl(rainwear_size_resolutions_path.expanduser().resolve())
        if rainwear_size_resolutions_path is not None
        else (
            _read_jsonl(inherited_size_resolutions)
            if inherited_size_resolutions.is_file()
            else []
        )
    )
    repo_root = Path(__file__).resolve().parents[2]
    for resolution in resolution_rows:
        artifact = Path(str(resolution.get("source_artifact") or ""))
        artifact = artifact if artifact.is_absolute() else repo_root / artifact
        if not artifact.is_file():
            raise ValueError(
                f"rainwear size resolution source artifact is missing: {artifact}"
            )
        if resolution.get("source_artifact_sha256") != _sha256(artifact):
            raise ValueError(
                "rainwear size resolution source artifact hash mismatch: "
                f"{resolution.get('offer_id')}"
            )
    size_resolutions = {
        str(row["offer_id"]): row for row in resolution_rows
    }
    if extra := sorted(set(size_resolutions) - source_ids):
        raise ValueError(
            f"rainwear size resolutions contain unknown offer IDs: {extra}"
        )
    if len(size_resolutions) != len(resolution_rows):
        raise ValueError("rainwear size resolutions contain duplicate offer IDs")
    resolved_catalog = [
        _with_rainwear_size_resolution(row, size_resolutions[str(row["offer_id"])])
        if str(row["offer_id"]) in size_resolutions
        else row
        for row in source_catalog
    ]
    catalog = [
        _with_online_offer(
            row,
            verified_at=verified_at,
            override=overrides.get(str(row["offer_id"])),
        )
        for row in resolved_catalog
    ]
    if len({str(row["offer_id"]) for row in catalog}) != len(catalog):
        raise ValueError("source runtime has duplicate offer IDs")
    source_to_short, offer_registry = _assign_short_offer_ids(source, catalog)
    mapped_catalog = []
    for row in catalog:
        mapped = {
            **row,
            "offer_id": source_to_short[str(row["offer_id"])],
        }
        if isinstance(mapped.get("consumer_online_offer"), dict):
            mapped["consumer_online_offer"] = {
                **mapped["consumer_online_offer"],
                "exact_sku_identity_sha256": _exact_identity(mapped),
            }
        mapped_catalog.append(mapped)
    catalog = mapped_catalog
    _write_jsonl(output / "catalog.jsonl", catalog)
    _write_json(output / "offer_id_registry.json", offer_registry)
    initial_assets = [
        {
            **row,
            "offer_id": source_to_short[str(row["offer_id"])],
        }
        for row in _read_jsonl(source / "initial_assets.jsonl")
    ]
    initial_retail = [
        {
            **row,
            "offer_id": source_to_short[str(row["offer_id"])],
        }
        for row in _read_jsonl(source / "initial_retail_inventory.jsonl")
    ]
    _write_jsonl(output / "initial_assets.jsonl", initial_assets)
    _write_jsonl(output / "initial_retail_inventory.jsonl", initial_retail)
    if size_resolutions:
        _write_jsonl(
            output / "rainwear_size_resolutions.jsonl",
            [
                {
                    **size_resolutions[key],
                    "offer_id": source_to_short[key],
                    "exact_sku_identity_sha256": _exact_identity(
                        next(
                            row
                            for row in catalog
                            if row["offer_id"] == source_to_short[key]
                        )
                    ),
                }
                for key in sorted(size_resolutions)
            ],
        )
    (output / "initial_prices.jsonl").write_text("", encoding="utf-8")

    channel_payload = default_channel_model_payload(_group_policies(source))
    channel_model, calibration = calibrate_channel_model(
        ChannelDemandModel.from_dict(channel_payload)
    )
    _write_json(output / "channel_demand_model.json", channel_model.to_dict())
    _write_json(output / "channel_parameter_calibration.json", calibration)
    _write_json(
        output / "result.json",
        {
            "format": BUILDER_VERSION,
            "source_runtime": str(source),
            "catalog_offers": len(catalog),
            "initial_priced_offers": 0,
            "channel_model_format": channel_model.format,
            "behavior_calibration_passed": calibration[
                "passing_candidate_count"
            ]
            > 0,
            "selected_parameters": calibration["selected_parameters"],
            "rainwear_size_resolutions": len(size_resolutions),
            "offer_id_registry_entries": len(offer_registry["entries"]),
        },
    )
    (output / "report.md").write_text(
        "# Exact-offer channel runtime build\n\n"
        f"- Source: `{source}`\n"
        f"- Exact offers: {len(catalog)}\n"
        "- Opening priced offers: 0\n"
        f"- Demand model: `{channel_model.format}`\n"
        f"- Behavior candidates: {calibration['candidate_count']} "
        f"({calibration['passing_candidate_count']} passed)\n",
        encoding="utf-8",
    )

    metadata = _read_json(source / "metadata.json")
    # Economic QA is behavior-version specific.  Never inherit a passing gate
    # from the source runtime after replacing its formal demand model.
    metadata.pop("scripted_economic_qa", None)
    metadata.update(
        {
            "format": "outfitter-runtime-v11-compact-ids-pagination",
            "schema_version": "outfitter-decision-schema-v11",
            "release_status": "v12-compact-ids-pagination-candidate",
            "channel_runtime_builder": BUILDER_VERSION,
            "source_runtime": str(source),
            "benchmark_calendar_year": 2026,
            "weather_source_year": 2023,
            "weather_mapping": "same_month_day_historical_replay",
            "formal_max_days": 200,
            "formal_runoff_days": 14,
            "environment_protocol": "outfitter-seasonal-v12-compact-ids-pagination",
            "tool_interface_version": 10,
            "offer_id_policy": {
                "format": OFFER_ID_REGISTRY_FORMAT,
                "registry_file": "offer_id_registry.json",
                "identifier_format": "offer-NNN",
                "append_only": True,
                "removed_numbers_reused": False,
                "catalog_reorder_renumbers": False,
                "random_tape_identity": (
                    "internal first_seen_source_offer_id; never exposed by tools"
                ),
                "public_id_projection_changes_random_tape": False,
            },
            "business_id_policy": {
                "request_id": "stable trip-derived rr-DDD-AA-NNNN",
                "line_id": "request-local line-NN",
                "commitment_id": "global sequential reservation-NNN",
                "change_id": "global sequential change-NNN",
                "block_id_visibility": "internal only",
                "assignment_id_visibility": "internal only",
                "historical_runs_rewritten": False,
            },
            "reservation_mechanism": {
                "main_reservation_share": 0.65,
                "stress_reservation_share": 0.80,
                "booking_lead_days": [1, 14],
                "prepayment_rate": 0.25,
                "inquiry_generation": "routed_trip_before_channel_choice_inventory_independent",
                "quote_rounds": 1,
                "quote_unit": "per_asset_per_day_by_bundle_line",
                "quote_actions": ["explicit_complete_quote", "decline"],
                "quote_line_completeness": "all_required_and_optional_lines",
                "walk_in_price_separation": "posted_prices_never_fill_contract_lines",
                "channel_choice": "after_manager_quote",
                "asset_binding": "pickup_morning_primary_first_disjunction",
                "authorized_offer_count": [1, 3],
                "line_mixing": False,
                "security_deposit": "confirmed_primary_terms_locked_after_quote",
                "latent_acceptance": "hidden_customer_price_ceiling",
                "public_authorization": "quoted_candidate_survivors",
                "bundle_lines": [2, 4],
                "bundle_customer_latents": (
                    "request-level segment, forecast trust, price sensitivity, "
                    "and return burden; shared storage/reuse bases with category "
                    "adjustments; line-level product preferences"
                ),
                "latent_snapshots": (
                    "immutable inquiry truth separated from post-quote public "
                    "authorization and hidden recovery sets"
                ),
                "required_pickup_atomic": True,
                "optional_missing_credit_rate": 1.0,
                "public_calendar": "raw facts without feasibility verdict",
                "no_show_in_main": False,
            },
            "supplier_delivery_policy": {
                "minimum_days": 2,
                "maximum_days": 7,
                "actual_may_arrive_days_before_estimate": 1,
                "actual_may_arrive_days_after_estimate": 2,
                "seller_specific_concrete_estimate": True,
                "agent_observes_variability_range": False,
            },
            "initial_price_policy": (
                "all offers start unpriced; the manager must post positive offer-level "
                "prices for walk-ins; every advance inquiry requires a separate, "
                "complete explicit one-time line quote or whole-request decline"
            ),
            "channel_demand_model": {
                "format": channel_model.format,
                "formal_worlds": sorted(channel_model.worlds),
                "primary_world": "base",
                "supplier_cost_enters_customer_demand": False,
                "online_price_generates_natural_demand": False,
                "agent_observes_hidden_channel": False,
                "deadline_context": {
                    "scope": "retail deadlines and contextual rental channel choice",
                    "need_horizon_days": [0, 1, 3, None],
                    "rental_booking_lead_days": [0, 14],
                    "consumer_online_delivery_model": "independent frozen CDF by demand world",
                    "agent_observes_internal_deadline_parameters": False,
                    "rental_formula_changed": True,
                    "rental_context": (
                        "duration, pickup lead, forecast adversity/uncertainty, trip "
                        "importance, future-use expectation, storage burden, return burden"
                    ),
                    "channel_heterogeneity": (
                        "bounded customer-level channel tastes; profile Gumbel remains "
                        "conditional on store entry only"
                    ),
                    "online_unit_price_anchor": (
                        "retained at full weight as shared product-value evidence; "
                        "deadline surplus enters only immediate fulfillment and lateness loss"
                    ),
                    "mapping_policy": {
                        "emergency_rain": (
                            "day 0; rainy/snowy missed-occasion loss calibrated to "
                            "a bounded positive-margin disposable-poncho curve"
                        ),
                        "required_supply_water": "day 0",
                        "other_required_unplanned_retail": "day 1 with service-specific importance",
                        "required_planned_retail": "day 3",
                        "optional_retail": "flexible; no deadline surplus",
                    },
                },
                "parameter_calibration": "channel_parameter_calibration.json",
                "multi_unit_policy": {
                    "required": "full fulfillment with compatible-profile mixing",
                    "optional": "partial fulfillment with compatible-profile mixing",
                    "inventory_utility_cap": "requested customer units",
                },
            },
            "tent_capacity_policy": {
                "comfortable_capacity_persons": "lower sourced range endpoint",
                "capacity_persons": "upper sourced range endpoint used for hard compatibility",
                "legacy_runtime_projection": "deterministic builder migration",
            },
            "rainwear_size_policy": {
                "formal_request_sizes": ["adult", "child"],
                "universal_offer_size": "adult_child_universal",
                "unknown_request_size_allowed": False,
                "compatible_inventory_projection": True,
                "resolution_file": (
                    "rainwear_size_resolutions.jsonl"
                    if size_resolutions
                    else None
                ),
            },
            "consumer_online_offer_policy": {
                "scope": "verified exact source SKU",
                "transaction_unit": "complete source package",
                "retail_unit_comparison": "complete consumer unit",
                "verified_at": verified_at,
                "agent_visibility": (
                    "the manager sees ordinary source listing facts, never their hidden "
                    "role in customer channel choice"
                ),
            },
            "retail_reference_price_method": (
                "legacy field retained for control/audit only; formal channel demand "
                "uses verified exact-offer package cash and complete-unit prices"
            ),
            "scripted_economic_qa": {
                "passed": False,
                "status": "pending_for_current_channel_demand_version",
            },
        }
    )
    if isinstance(metadata.get("liquidation_dealer"), dict):
        metadata["liquidation_dealer"] = {
            **metadata["liquidation_dealer"],
            "environment_protocol": "outfitter-seasonal-v12-compact-ids-pagination",
            "tool_interface_version": 10,
            "rolling_depth_market": True,
            "transaction_policy": (
                "sales validate and commit atomically; a read-only quote with "
                "invalid input returns all invalid-item diagnostics at once"
            ),
        }
    if isinstance(metadata.get("demand_model"), dict):
        metadata["demand_model"] = {
            **metadata["demand_model"],
            "formal_use": "control world only",
        }
    metadata["initial_liquidity"] = {
        **dict(metadata.get("initial_liquidity") or {}),
        "refund_pressure": (
            "pending 200-day recalibration after seller-specific delivery times"
        ),
    }
    calibration_results = [
        (path.expanduser().resolve(), _read_json(path.expanduser().resolve()))
        for path in calibration_result_paths or []
    ]
    if calibration_results:
        if any(
            not bool(payload.get("scripted_profit_gate", {}).get("passed"))
            for _, payload in calibration_results
        ):
            raise ValueError("all supplied scripted calibration results must pass")
        fees = {
            float(payload["calibration_scale"]["daily_fee"])
            for _, payload in calibration_results
        }
        if len(fees) != 1:
            raise ValueError("scripted calibration results disagree on daily fee")
        formal_daily_fee = fees.pop()
        liquidity_source = max(
            calibration_results,
            key=lambda item: max(
                (int(run["days"]) for run in item[1].get("runs", [])),
                default=0,
            ),
        )[1]
        metadata["initial_liquidity"] = dict(
            liquidity_source["liquidity_recommendation"]
        )
        recommended_initial_cash = max(
            int(
                payload["liquidity_recommendation"][
                    "recommended_initial_cash_rounded"
                ]
            )
            for _, payload in calibration_results
        )
        metadata["scripted_economic_qa"] = {
            "passed": True,
            "results": [
                {"path": str(path), "sha256": _sha256(path)}
                for path, _ in calibration_results
            ],
            "horizons_days": sorted(
                {
                    int(run["days"])
                    for _, payload in calibration_results
                    for run in payload.get("runs", [])
                }
            ),
            "worlds": sorted(
                {
                    str(run.get("demand_world_id") or "base")
                    for _, payload in calibration_results
                    for run in payload.get("runs", [])
                }
            ),
            "seeds": sorted(
                {
                    int(run["seed"])
                    for _, payload in calibration_results
                    for run in payload.get("runs", [])
                }
            ),
        }
        metadata["release_status"] = "v6-reservations-scripted-qa-passed-candidate"
    if recommended_initial_cash is not None:
        if recommended_initial_cash <= 0:
            raise ValueError("recommended_initial_cash must be positive")
        metadata["initial_liquidity"] = {
            **dict(metadata.get("initial_liquidity") or {}),
            "recommended_initial_cash_rounded": int(recommended_initial_cash),
            "policy": (
                "liquidity pressure only; selected from frozen-demand scripted "
                "survival/drawdown QA, not a score denominator"
            ),
        }
    if formal_daily_fee is not None:
        if formal_daily_fee < 0:
            raise ValueError("formal_daily_fee must be non-negative")
        metadata["formal_daily_fee"] = round(float(formal_daily_fee), 2)
        metadata["initial_liquidity"] = {
            **dict(metadata.get("initial_liquidity") or {}),
            "fourteen_day_fee_reserve": round(14 * float(formal_daily_fee), 2),
        }
    metadata["content_sha256"] = _content_hashes(output)
    _write_json(output / "metadata.json", metadata)

    config = OutfitterConfig(
        data_dir=str(output),
        max_days=min(28, int(metadata.get("formal_max_days") or 200)),
        weather_start_day=int(metadata.get("formal_weather_start_day") or 91),
        daily_fee=float(metadata.get("formal_daily_fee") or 60.0),
        charge_model_usage_to_cash=False,
    )
    environment = OutfitterEnvironment(config, OutfitterDataset.load(output))
    write_product_visibility_exports(environment, output)
    metadata["content_sha256"] = _content_hashes(output)
    _write_json(output / "metadata.json", metadata)
    # Re-load after final hashing so a generated runtime is immediately usable.
    OutfitterDataset.load(output)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verified-at", default="2026-08-21")
    parser.add_argument("--recommended-initial-cash", type=int)
    parser.add_argument("--formal-daily-fee", type=float)
    parser.add_argument("--consumer-online-offers", type=Path)
    parser.add_argument("--rainwear-size-resolutions", type=Path)
    parser.add_argument("--calibration-results", type=Path, nargs="+")
    args = parser.parse_args()
    metadata = build_channel_runtime(
        args.source,
        args.output,
        verified_at=str(args.verified_at),
        recommended_initial_cash=args.recommended_initial_cash,
        formal_daily_fee=args.formal_daily_fee,
        consumer_online_offers_path=args.consumer_online_offers,
        rainwear_size_resolutions_path=args.rainwear_size_resolutions,
        calibration_result_paths=args.calibration_results,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
