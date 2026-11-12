"""Project frozen V29 data into V30's reservation-before-walk-in cutoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .build_liquidation_accountability_runtime import ENVIRONMENT_PROTOCOL as SOURCE_PROTOCOL
from .data import V13_RUNTIME_FORMAT


ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v30-reservation-pickup-priority"
TOOL_INTERFACE_VERSION = 26
CONTRACT_MODE = "factorized_line_options_plan_set_v1"
PICKUP_ORDER = "finalize_due_pickups_before_walk_ins_v1"
BUILDER_FORMAT = "outfitter-pickup-priority-runtime-builder-v1"


def build_pickup_priority_runtime(source_runtime: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source, output = Path(source_runtime).resolve(), Path(output_dir).resolve()
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("format"), metadata.get("environment_protocol"), metadata.get("tool_interface_version")) != (
        V13_RUNTIME_FORMAT, SOURCE_PROTOCOL, 25,
    ):
        raise ValueError("V30 projection requires the frozen V29 runtime/interface 25")
    if metadata.get("reservation_promise_profiles", {}).get("contract_mode") != CONTRACT_MODE:
        raise ValueError("V30 requires factorized plan-set contracts")
    if metadata.get("liquidation_policy") != "flat_nominal_per_unit_v1":
        raise ValueError("V30 requires V29 flat liquidation")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty or absent")
    output.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if path.is_file() and path.name not in {"metadata.json", "report.md", "result.json"}:
            shutil.copy2(path, output / path.name)
    metadata.update({
        "environment_protocol": ENVIRONMENT_PROTOCOL,
        "tool_interface_version": TOOL_INTERFACE_VERSION,
        "release_status": "v30-pickup-priority-candidate",
        "pickup_priority_runtime_builder": BUILDER_FORMAT,
        "source_runtime": str(source),
        "reservation_pickup_order": PICKUP_ORDER,
    })
    result = {
        "format": BUILDER_FORMAT, "source_runtime": str(source), "output_runtime": str(output),
        "environment_protocol": ENVIRONMENT_PROTOCOL, "tool_interface_version": TOOL_INTERFACE_VERSION,
        "reservation_pickup_order": PICKUP_ORDER,
    }
    (output / "report.md").write_text(
        "# V30 reservation pickup priority\n\n"
        "Due reservation pickups, including explicitly selected recoveries, are finalized "
        "before same-day walk-ins at close. Recovery selection does not itself bind assets. "
        "Future reservations still require manager capacity planning. No feasibility verdict "
        "or automatic fallback selection is added.\n\n"
        "V29 and earlier runtimes retain their original close order. Catalog, demand, "
        "feedback, weather, lifecycle economics and settlement rates are unchanged.\n",
        encoding="utf-8",
    )
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata["content_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir()) if path.is_file() and path.name != "metadata.json"
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build_pickup_priority_runtime(args.source_runtime, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
