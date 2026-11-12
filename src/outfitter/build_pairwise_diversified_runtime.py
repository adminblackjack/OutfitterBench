"""Create a fixed-size, SKU-diversified pairwise quote runtime ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .build_pairwise_quote_runtime import (
    CONTRACT_MODE,
    TOOL_INTERFACE_VERSION,
)
from .data import OutfitterDataset, V13_RUNTIME_FORMAT


BUILDER_FORMAT = "outfitter-pairwise-diversified-opening-builder-v1"
SOURCE_PROTOCOL = "outfitter-seasonal-v16-adaptive-pairwise-exact-quotes"
ENVIRONMENT_PROTOCOL = (
    "outfitter-seasonal-v16-adaptive-pairwise-diversified-opening"
)
REPLACEMENTS = {
    "offer-009": "offer-019",
    "offer-020": "offer-052",
    "offer-022": "offer-078",
    "offer-048": "offer-064",
    "offer-073": "offer-023",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_pairwise_diversified_runtime(
    source_runtime: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source = Path(source_runtime).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format") != V13_RUNTIME_FORMAT:
        raise ValueError("diversification requires a V16 pairwise runtime")
    if metadata.get("environment_protocol") != SOURCE_PROTOCOL:
        raise ValueError("unexpected pairwise source protocol")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty or absent")
    output.mkdir(parents=True, exist_ok=True)
    for source_file in sorted(source.iterdir()):
        if source_file.is_file() and source_file.name not in {
            "metadata.json",
            "initial_assets.jsonl",
        }:
            shutil.copy2(source_file, output / source_file.name)

    dataset = OutfitterDataset.load(source)
    offers = {offer.offer_id: offer for offer in dataset.offers}
    assets = [dict(row) for row in dataset.initial_assets]
    original_cost = sum(float(row["unit_cost"]) for row in assets)
    replacement_rows = []
    for source_offer_id, target_offer_id in REPLACEMENTS.items():
        candidates = sorted(
            (
                row
                for row in assets
                if str(row["offer_id"]) == source_offer_id
            ),
            key=lambda row: (
                str(row.get("age_cohort")) != "young",
                str(row["asset_id"]),
            ),
        )
        if len(candidates) < 2:
            raise ValueError(
                f"replacement would remove opening coverage: {source_offer_id}"
            )
        row = candidates[0]
        source_offer = offers[source_offer_id]
        target_offer = offers[target_offer_id]
        day_fraction = float(row.get("used_life_days", 0)) / max(
            1, int(source_offer.service_life_limit_days)
        )
        source_cycles = int(source_offer.service_life_limit_cycles or 1)
        cycle_fraction = float(row.get("used_rental_cycles", 0)) / max(
            1, source_cycles
        )
        row.update(
            {
                "offer_id": target_offer_id,
                "service_line": target_offer.service_line,
                "unit_cost": float(target_offer.unit_cost),
                "service_life_limit_days": int(
                    target_offer.service_life_limit_days
                ),
                "service_life_limit_cycles": (
                    target_offer.service_life_limit_cycles
                ),
                "damage_risk_band": target_offer.damage_risk_band,
                "used_life_days": round(
                    day_fraction * int(target_offer.service_life_limit_days)
                ),
                "used_rental_cycles": round(
                    cycle_fraction
                    * int(target_offer.service_life_limit_cycles or 1)
                ),
            }
        )
        replacement_rows.append(
            {
                "asset_id": str(row["asset_id"]),
                "source_offer_id": source_offer_id,
                "target_offer_id": target_offer_id,
                "source_unit_cost": float(source_offer.unit_cost),
                "target_unit_cost": float(target_offer.unit_cost),
            }
        )

    asset_path = output / "initial_assets.jsonl"
    asset_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in assets
        ),
        encoding="utf-8",
    )
    diversified_cost = sum(float(row["unit_cost"]) for row in assets)
    profile_metadata = dict(metadata.get("reservation_promise_profiles") or {})
    profile_metadata["contract_mode"] = CONTRACT_MODE
    metadata.update(
        {
            "environment_protocol": ENVIRONMENT_PROTOCOL,
            "tool_interface_version": TOOL_INTERFACE_VERSION,
            "release_status": "v16-pairwise-diversified-opening-ablation",
            "pairwise_diversified_opening_builder": BUILDER_FORMAT,
            "source_runtime": str(source),
            "reservation_promise_profiles": profile_metadata,
            "opening_fleet_diversification": {
                "asset_count_before": len(dataset.initial_assets),
                "asset_count_after": len(assets),
                "opening_cost_before": round(original_cost, 2),
                "opening_cost_after": round(diversified_cost, 2),
                "opening_cost_change": round(diversified_cost - original_cost, 2),
                "replacement_count": len(replacement_rows),
                "replacements": replacement_rows,
                    "policy": (
                        "replace one young asset in five three-unit exact cohorts; "
                        "retain two units of every source offer and preserve "
                        "used-life fractions"
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
    return {
        "format": BUILDER_FORMAT,
        "source_runtime": str(source),
        "output_runtime": str(output),
        "environment_protocol": ENVIRONMENT_PROTOCOL,
        "tool_interface_version": TOOL_INTERFACE_VERSION,
        "contract_mode": CONTRACT_MODE,
        "asset_count": len(assets),
        "replacement_count": len(replacement_rows),
        "opening_cost_before": round(original_cost, 2),
        "opening_cost_after": round(diversified_cost, 2),
        "opening_cost_change": round(diversified_cost - original_cost, 2),
        "metadata_sha256": _sha256(output / "metadata.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = build_pairwise_diversified_runtime(
        args.source_runtime, args.output_dir
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
