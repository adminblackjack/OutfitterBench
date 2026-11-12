"""Build content-addressed synthetic reference-demand runtimes for mechanism tests."""

from __future__ import annotations

from .jsonio import read_jsonl as _read_jsonl, write_jsonl as _write_jsonl

import hashlib
import json
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import OutfitterConfig
from .data import OutfitterDataset
from .environment import OutfitterEnvironment
from .reference_demand import (
    DEFAULT_REFERENCE_WORLDS,
    ReferenceTransactionModel,
    default_reference_model_payload,
)
from .visibility import write_product_visibility_exports
from .data_validation.v5_reference_transactions import CHILD_CHAIR_EXCLUSIONS


SYNTHETIC_REFERENCE_BUILDER_VERSION = "v2-formal-catalog-scope"


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _class_payload(offer: Any, service_key: str) -> dict[str, Any]:
    return {
        "service_key": service_key,
        "choice_group": offer.choice_group,
        "asset_kind": offer.asset_kind,
        "decision_attributes": offer.decision_attributes,
        "consumer_value_attributes": offer.consumer_value_attributes,
        "consumer_unit": offer.consumer_unit,
    }


def build_synthetic_reference_model(
    environment: OutfitterEnvironment,
    state: Any,
) -> tuple[ReferenceTransactionModel, list[dict[str, Any]]]:
    """Use opening prices as synthetic anchors; never claim market provenance."""

    if environment.demand_model is None:
        raise ValueError("synthetic reference shadow requires control demand_model.json")
    grouped: defaultdict[str, list[Any]] = defaultdict(list)
    class_payloads: dict[str, dict[str, Any]] = {}
    assignments: dict[str, str] = {}
    for offer in environment.dataset.offers:
        if offer.offer_id in CHILD_CHAIR_EXCLUSIONS:
            continue
        service_key = str(offer.fulfills_service_keys[0])
        payload = _class_payload(offer, service_key)
        cell_id = "syn-" + hashlib.sha256(
            _canonical(payload).encode("utf-8")
        ).hexdigest()[:20]
        grouped[cell_id].append(offer)
        class_payloads[cell_id] = payload
        assignments[offer.offer_id] = cell_id
    cells = []
    assignments_rows = []
    for cell_id, offers in sorted(grouped.items()):
        first = offers[0]
        daily_or_sale = statistics.median(
            float(state.prices[offer.offer_id]) for offer in offers
        )
        deposit = (
            0.0
            if first.asset_kind == "consumable"
            else statistics.median(
                float(state.deposits[offer.offer_id]) for offer in offers
            )
        )
        schedule = (
            {"0": round(daily_or_sale, 6)}
            if first.asset_kind == "consumable"
            else {
                str(duration): round(daily_or_sale * duration, 6)
                for duration in range(1, 6)
            }
        )
        payload = class_payloads[cell_id]
        cells.append(
            {
                "cell_id": cell_id,
                "service_key": payload["service_key"],
                "choice_group": payload["choice_group"],
                "transaction_kind": (
                    "retail_sale"
                    if first.asset_kind == "consumable"
                    else "rental_contract"
                ),
                "cash_price_by_duration": schedule,
                "deposit": round(deposit, 6),
                "quote_ids": [f"synthetic-opening:{cell_id}"],
                "independent_source_count": 0,
                "confidence": "low",
                "synthetic": True,
            }
        )
        for offer in sorted(offers, key=lambda row: row.offer_id):
            assignments_rows.append(
                {
                    "offer_id": offer.offer_id,
                    "cell_id": cell_id,
                    "class_payload": payload,
                    "provenance": "synthetic_opening_price_mechanism_fixture",
                }
            )
    policies = [
        {
            "service_key": row.service_key,
            "choice_group": row.choice_group,
            "need_strength": row.need_strength,
            "quality_salience": row.quality_salience,
            "alternative_accessibility": row.alternative_accessibility,
            "substitution_strength": row.substitution_strength,
        }
        for row in environment.demand_model.groups.values()
    ]
    model = ReferenceTransactionModel.from_dict(
        default_reference_model_payload(
            reference_cells=cells,
            offer_assignments=assignments,
            group_policies=policies,
        )
    )
    return model, assignments_rows


def build_synthetic_shadow_runtime(source: Path, output: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if source == output or source in output.parents:
        raise ValueError("synthetic shadow output must not overwrite the source runtime")
    dataset = OutfitterDataset.load(source)
    environment = OutfitterEnvironment(
        OutfitterConfig(
            data_dir=str(source),
            max_days=28,
            weather_start_day=91,
            initial_cash=6_000,
            charge_model_usage_to_cash=False,
        ),
        dataset,
    )
    model, assignments = build_synthetic_reference_model(
        environment, environment.reset()
    )
    dataset.snapshot_to(output)
    allowed = set(model.offer_assignments)
    for name in (
        "catalog.jsonl",
        "initial_assets.jsonl",
        "initial_retail_inventory.jsonl",
        "initial_prices.jsonl",
    ):
        path = output / name
        _write_jsonl(
            path,
            [
                row
                for row in _read_jsonl(path)
                if str(row["offer_id"]) in allowed
            ],
        )
    model_path = output / "reference_transaction_model.json"
    model_path.write_text(
        json.dumps(model.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    worlds_path = output / "demand_worlds.json"
    worlds_path.write_text(
        json.dumps(
            {
                "format": "outfitter-reference-demand-worlds-v1",
                "worlds": {
                    key: asdict(row)
                    for key, row in sorted(DEFAULT_REFERENCE_WORLDS.items())
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    assignments_path = output / "reference_class_assignments.jsonl"
    with assignments_path.open("w", encoding="utf-8") as handle:
        for row in assignments:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    aggregation_path = output / "aggregation_report.json"
    aggregation_path.write_text(
        json.dumps(
            {
                "format": "outfitter-reference-aggregation-report-v1",
                "synthetic": True,
                "economic_use_allowed": False,
                "builder_version": SYNTHETIC_REFERENCE_BUILDER_VERSION,
                "cells": len(model.cells),
                "offers": len(model.offer_assignments),
                "unresolved_cells": 0,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hashes = dict(metadata.get("content_sha256") or {})
    for path in (
        output / "catalog.jsonl",
        output / "initial_assets.jsonl",
        output / "initial_retail_inventory.jsonl",
        output / "initial_prices.jsonl",
        model_path,
        worlds_path,
        assignments_path,
        aggregation_path,
    ):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata["content_sha256"] = hashes
    metadata["catalog_offers"] = len(allowed)
    metadata["release_status"] = "synthetic-reference-shadow-mechanism-only"
    metadata["reference_transaction_demand"] = {
        "synthetic": True,
        "economic_use_allowed": False,
        "formal_scoring_allowed": False,
        "builder_version": SYNTHETIC_REFERENCE_BUILDER_VERSION,
        "cells": len(model.cells),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shadow_environment = OutfitterEnvironment(
        OutfitterConfig(
            data_dir=str(output),
            max_days=28,
            weather_start_day=91,
            initial_cash=6_000,
            demand_world_id="base",
            charge_model_usage_to_cash=False,
        ),
        OutfitterDataset.load(output),
    )
    sales_path, public_path = write_product_visibility_exports(
        shadow_environment, output
    )
    for path in (sales_path, public_path):
        metadata["content_sha256"][path.name] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "format": "outfitter-synthetic-reference-shadow-v1",
        "source": str(source),
        "output": str(output),
        "cells": len(model.cells),
        "offers": len(model.offer_assignments),
        "economic_use_allowed": False,
    }






def build_formal_shadow_runtime(
    source: Path,
    model_dir: Path,
    output: Path,
) -> dict[str, Any]:
    """Build a non-formal shadow runtime only after reference coverage passes."""

    source = source.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    if "shadow" not in output.name.lower():
        raise ValueError("reference runtime output must be explicitly named shadow")
    report = json.loads(
        (model_dir / "aggregation_report.json").read_text(encoding="utf-8")
    )
    if report.get("passed") is not True:
        raise ValueError("reference aggregation coverage has not passed")
    model = ReferenceTransactionModel.from_dict(
        json.loads(
            (model_dir / "reference_transaction_model.json").read_text(
                encoding="utf-8"
            )
        )
    )
    allowed = set(model.offer_assignments)
    dataset = OutfitterDataset.load(source)
    dataset.snapshot_to(output)
    file_keys = {
        "catalog.jsonl": "offer_id",
        "initial_assets.jsonl": "offer_id",
        "initial_retail_inventory.jsonl": "offer_id",
        "initial_prices.jsonl": "offer_id",
    }
    for name, key in file_keys.items():
        path = output / name
        _write_jsonl(
            path,
            [row for row in _read_jsonl(path) if str(row[key]) in allowed],
        )
    for name in (
        "reference_transaction_model.json",
        "reference_class_assignments.jsonl",
        "reference_contract_evidence.jsonl",
        "aggregation_report.json",
    ):
        (output / name).write_bytes((model_dir / name).read_bytes())
    worlds_path = output / "demand_worlds.json"
    worlds_path.write_text(
        json.dumps(
            {
                "format": "outfitter-reference-demand-worlds-v1",
                "worlds": {
                    key: asdict(row) for key, row in sorted(model.worlds.items())
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hash_names = [
        "catalog.jsonl",
        "initial_assets.jsonl",
        "initial_retail_inventory.jsonl",
        "initial_prices.jsonl",
        "market_rules.json",
        "weather.jsonl",
        "demand_model.json",
        "reference_transaction_model.json",
        "reference_class_assignments.jsonl",
        "reference_contract_evidence.jsonl",
        "aggregation_report.json",
        "demand_worlds.json",
    ]
    metadata["content_sha256"] = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in hash_names
        if (output / name).is_file()
    }
    metadata["catalog_offers"] = len(allowed)
    metadata["release_status"] = "reference-transaction-shadow-pending-economics"
    metadata["reference_transaction_demand"] = {
        "synthetic": False,
        "economic_use_allowed": True,
        "formal_scoring_allowed": False,
        "cells": len(model.cells),
        "primary_world": "base",
        "mandatory_robustness_worlds": ["conservative", "convenience-heavy"],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    environment = OutfitterEnvironment(
        OutfitterConfig(
            data_dir=str(output),
            max_days=28,
            weather_start_day=91,
            initial_cash=6_000,
            demand_world_id="base",
            charge_model_usage_to_cash=False,
        ),
        OutfitterDataset.load(output),
    )
    sales_path, public_path = write_product_visibility_exports(environment, output)
    for path in (sales_path, public_path):
        metadata["content_sha256"][path.name] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "format": "outfitter-reference-transaction-shadow-runtime-v1",
        "source": str(source),
        "model_dir": str(model_dir),
        "output": str(output),
        "offers": len(allowed),
        "cells": len(model.cells),
    }
