from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from .data import V11_RUNTIME_FORMAT, V12_RUNTIME_FORMAT
from .channel_demand import ChannelDemandModel
from .jsonio import read_jsonl
from .lifecycle_economics import LifecycleEconomicScales


BUILDER_FORMAT = "outfitter-lifecycle-runtime-builder-v3"
ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v13-lifecycle-closure"
PRICING_ENVIRONMENT_PROTOCOL = (
    "outfitter-seasonal-v14-lifecycle-cash-salience"
)
BALANCED_ENVIRONMENT_PROTOCOL = (
    "outfitter-seasonal-v15-lifecycle-economic-balance"
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_lifecycle_runtime(
    source_runtime: str | Path,
    lifecycle_artifact: str | Path,
    output_dir: str | Path,
    *,
    rental_contract_cash_salience: float = 1.0,
    formal_daily_fee: float | None = None,
) -> dict[str, Any]:
    source = Path(source_runtime).expanduser().resolve()
    lifecycle = Path(lifecycle_artifact).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    source_metadata = json.loads(
        (source / "metadata.json").read_text(encoding="utf-8")
    )
    if source_metadata.get("format") != V11_RUNTIME_FORMAT:
        raise ValueError("lifecycle projection requires the frozen V12 compact-ID runtime")
    salience = float(rental_contract_cash_salience)
    if not 0.1 <= salience <= 5.0:
        raise ValueError("rental_contract_cash_salience must be in [0.1, 5.0]")
    source_daily_fee = float(source_metadata.get("formal_daily_fee", 60.0))
    daily_fee = (
        source_daily_fee
        if formal_daily_fee is None
        else float(formal_daily_fee)
    )
    if not 0.0 <= daily_fee <= 1_000.0:
        raise ValueError("formal_daily_fee must be in [0, 1000]")
    registry = json.loads(
        (source / "offer_id_registry.json").read_text(encoding="utf-8")
    )
    source_to_public = {
        str(row["first_seen_source_offer_id"]): str(row["offer_id"])
        for row in registry.get("entries", [])
        if bool(row.get("active"))
    }
    catalog = list(read_jsonl(source / "catalog.jsonl"))
    durable_ids = {
        str(row["offer_id"])
        for row in catalog
        if str(row.get("asset_kind")) == "durable"
    }
    factors = list(read_jsonl(lifecycle / "offer_lifecycle_factors.jsonl"))
    projected: list[dict[str, Any]] = []
    for raw in factors:
        source_offer_id = str(raw.get("offer_id") or "")
        public_offer_id = source_to_public.get(source_offer_id)
        if public_offer_id is None:
            raise ValueError(
                f"lifecycle factor has no active registry identity: {source_offer_id}"
            )
        row = dict(raw)
        row["offer_id"] = public_offer_id
        row["internal_factor_source_sha256"] = hashlib.sha256(
            json.dumps(raw, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        projected.append(row)
    projected.sort(key=lambda row: str(row["offer_id"]))
    if {str(row["offer_id"]) for row in projected} != durable_ids:
        raise ValueError("projected lifecycle factors do not exactly cover durable catalog")

    scales_raw = json.loads(
        (lifecycle / "candidate_economic_scales.json").read_text(encoding="utf-8")
    )
    scales = LifecycleEconomicScales.from_dict(scales_raw)
    if scales.status != "candidate_requires_joint_economic_calibration":
        raise ValueError("lifecycle scales must remain explicitly unreleased")
    worlds = json.loads((lifecycle / "worlds.json").read_text(encoding="utf-8"))
    world_ids = {
        str(row.get("world_id")) for row in list(worlds.get("worlds") or [])
    }
    if world_ids != {"conservative", "base", "convenience-heavy"}:
        raise ValueError("lifecycle artifact must contain the three registered worlds")

    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty or absent")
    output.mkdir(parents=True, exist_ok=True)
    for source_file in sorted(source.iterdir()):
        if source_file.is_file() and source_file.name != "metadata.json":
            shutil.copy2(source_file, output / source_file.name)
    _write_jsonl(output / "lifecycle_factors.jsonl", projected)
    _write_json(output / "lifecycle_economic_scales.json", scales.to_dict())
    _write_json(output / "lifecycle_worlds.json", worlds)
    channel_model_path = output / "channel_demand_model.json"
    channel_model = ChannelDemandModel.from_dict(
        json.loads(channel_model_path.read_text(encoding="utf-8"))
    )
    channel_model = replace(
        channel_model, rental_contract_cash_salience=salience
    )
    channel_model.validate()
    _write_json(channel_model_path, channel_model.to_dict())

    metadata = dict(source_metadata)
    pricing_recalibrated = abs(salience - 1.0) > 1e-12
    economic_rebalanced = abs(daily_fee - source_daily_fee) > 1e-12
    metadata.update(
        {
            "format": V12_RUNTIME_FORMAT,
            "schema_version": "outfitter-decision-schema-v12",
            "environment_protocol": (
                BALANCED_ENVIRONMENT_PROTOCOL
                if economic_rebalanced
                else (
                    PRICING_ENVIRONMENT_PROTOCOL
                    if pricing_recalibrated
                    else ENVIRONMENT_PROTOCOL
                )
            ),
            "release_status": (
                "v15-lifecycle-economic-balance-candidate"
                if economic_rebalanced
                else (
                    "v14-lifecycle-cash-salience-candidate"
                    if pricing_recalibrated
                    else "v13-lifecycle-closure-candidate"
                )
            ),
            "formal_daily_fee": daily_fee,
            "lifecycle_runtime_builder": BUILDER_FORMAT,
            "source_runtime": str(source),
            "lifecycle_economics": {
                "enabled_config_required": True,
                "factor_file": "lifecycle_factors.jsonl",
                "scales_file": "lifecycle_economic_scales.json",
                "worlds_file": "lifecycle_worlds.json",
                "durable_offer_count": len(projected),
                "registered_worlds": sorted(world_ids),
                "scale_status": scales.status,
                "claim_boundary": scales.claim_boundary,
                "condition_is_independent_of_wear": True,
                "initial_maintenance_anchor_policy": (
                    "stable asset-ID phase across interval plus due window; no wear reset"
                ),
                "initial_age_cohort_policy": (
                    "preserve base day/cycle usage ratios when projecting world-specific "
                    "service-life limits"
                ),
            },
            "rental_pricing_calibration": {
                "rental_contract_cash_salience": salience,
                "supplier_cost_enters_customer_demand": False,
                "target_payback_enters_customer_demand": False,
                "scope": "rental store-contract cash penalty only; retail unchanged",
                "claim_boundary": (
                    "candidate benchmark difficulty scale; not observed market truth"
                ),
            },
            "economic_balance_calibration": {
                "formal_daily_fee": daily_fee,
                "source_formal_daily_fee": source_daily_fee,
                "changed": economic_rebalanced,
                "selection_rule": (
                    "joint 28-day and 200-day scripted bounded-loss QA; "
                    "not inferred from hidden customer willingness to pay"
                ),
                "claim_boundary": (
                    "candidate benchmark difficulty control, not an observed "
                    "real-world operating cost"
                ),
            },
        }
    )
    metadata["content_sha256"] = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "metadata.json"
    }
    _write_json(output / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project the audited lifecycle factor ledger into a compact-ID runtime"
    )
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--lifecycle-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rental-contract-cash-salience", type=float, default=1.0
    )
    parser.add_argument("--formal-daily-fee", type=float)
    args = parser.parse_args()
    metadata = build_lifecycle_runtime(
        args.source_runtime,
        args.lifecycle_artifact,
        args.output,
        rental_contract_cash_salience=args.rental_contract_cash_salience,
        formal_daily_fee=args.formal_daily_fee,
    )
    print(json.dumps(metadata["lifecycle_economics"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
