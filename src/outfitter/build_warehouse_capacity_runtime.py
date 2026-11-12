"""Create an independently versioned V31 runtime from unchanged V30 economics."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .build_pickup_priority_runtime import ENVIRONMENT_PROTOCOL as SOURCE_PROTOCOL
from .warehouse_capacity import WAREHOUSE_POLICY, validate_weights

ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v31-return-reserved-warehouse-capacity"
TOOL_INTERFACE_VERSION = 27


def build_warehouse_capacity_runtime(source_runtime: str | Path, weights_file: str | Path,
                                     output_dir: str | Path) -> dict:
    source, output, weights_path = Path(source_runtime).resolve(), Path(output_dir).resolve(), Path(weights_file).resolve()
    metadata = json.loads((source / "metadata.json").read_text())
    if (metadata.get("environment_protocol"), metadata.get("tool_interface_version")) != (SOURCE_PROTOCOL, 26):
        raise ValueError("V31 projection requires the frozen V30 runtime/interface 26")
    # Also verify source integrity before copying, not only when the result loads.
    from .data import OutfitterDataset
    dataset = OutfitterDataset.load(source)
    weights = json.loads(weights_path.read_text())
    mapping = validate_weights(weights, source / "catalog.jsonl",
        {offer.offer_id for offer in dataset.offers if offer.asset_kind == "durable"})
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty or absent")
    output.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if path.is_file() and path.name not in {"metadata.json", "report.md", "result.json"}:
            shutil.copy2(path, output / path.name)
    shutil.copy2(weights_path, output / "warehouse_capacity_weights.json")
    initial_units = sum(mapping[row["offer_id"]] for row in dataset.initial_assets)
    metadata.update({"environment_protocol": ENVIRONMENT_PROTOCOL, "tool_interface_version": TOOL_INTERFACE_VERSION,
        "release_status": "v31-warehouse-capacity-candidate-awaiting-limit-selection",
        "warehouse_capacity_policy": WAREHOUSE_POLICY, "source_runtime": str(source)})
    result = {"format": "outfitter-warehouse-capacity-runtime-builder-v1",
        "environment_protocol": ENVIRONMENT_PROTOCOL, "tool_interface_version": TOOL_INTERFACE_VERSION,
        "source_runtime": str(source), "output_runtime": str(output),
        "rental_offer_count": len(mapping), "initial_occupied_units": initial_units,
        "formal_capacity_limit": None, "formal_capacity_selected": False}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "report.md").write_text(
        "# V31 return-reserved warehouse capacity\n\n"
        "All held rental assets and confirmed unreceived units occupy fixed relative integer capacity. "
        "Only physical sale/disposal releases it. N is fixed by configuration, not purchasable. "
        "Null N is an uncapped calibration control, not a formal selected capacity. "
        "Catalog/evidence, customer model, fees, liquidation and reservation scheduling are unchanged. "
        "Service-life retirement now records physical disposal on actual return. No additional fee. "
        "V30 and older runtimes preserve their behavior.\n")
    metadata["content_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir()) if path.is_file() and path.name != "metadata.json"}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--weights-file", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build_warehouse_capacity_runtime(args.source_runtime, args.weights_file, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
