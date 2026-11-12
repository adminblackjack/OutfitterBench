"""Project V36 into the shared V37 supplier-trial runtime candidate."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from .acquisition_budget import ENVIRONMENT_PROTOCOL as SOURCE_PROTOCOL
from .acquisition_budget import TOOL_INTERFACE_VERSION as SOURCE_TOOL_INTERFACE_VERSION
from .assortment import SUPPORTED_ASSORTMENT_MODES
from .data import OutfitterDataset
from .supplier_trial import (
    CORE_RELEASE_ID, ENVIRONMENT_PROTOCOL, HANDLING_FEE, MAX_ACCEPTED_TRIALS,
    MAX_ACTIVE_TRIALS, MAX_CONTRACT_STARTS, MAX_UNIT_COST, POLICY_ID,
    TOOL_INTERFACE_VERSION, USAGE_FEE_MULTIPLIER, WINDOW_DAYS,
)


def build_supplier_trial_runtime(source_runtime: str | Path, output_dir: str | Path) -> dict:
    source = Path(source_runtime).resolve()
    output = Path(output_dir).resolve()
    data = OutfitterDataset.load(source)
    if (data.metadata.get("environment_protocol"), data.metadata.get("tool_interface_version")) != (
        SOURCE_PROTOCOL, SOURCE_TOOL_INTERFACE_VERSION
    ):
        raise ValueError("V37 requires the shared V36 durable-budget runtime")
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("source and output must be independent directories")
    if output.exists():
        raise ValueError("output must not already exist; never overwrite a runtime")
    output.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.is_file() and path.name not in {"metadata.json", "result.json", "report.md"}:
            shutil.copy2(path, output / path.name)
    policy = {
        "maximum_accepted_trials": MAX_ACCEPTED_TRIALS,
        "maximum_active_trials": MAX_ACTIVE_TRIALS,
        "maximum_contract_starts": MAX_CONTRACT_STARTS,
        "window_days": WINDOW_DAYS,
        "maximum_unit_cost": MAX_UNIT_COST,
        "handling_fee": HANDLING_FEE,
        "usage_fee_multiplier": USAGE_FEE_MULTIPLIER,
        "supplier_cancellation_generated": False,
        "terminal_value": 0.0,
        "optional_omit": "implicit_singleton_with_explicit_singleton_compatibility",
    }
    metadata = copy.deepcopy(data.metadata)
    metadata.update(
        environment_protocol=ENVIRONMENT_PROTOCOL,
        tool_interface_version=TOOL_INTERFACE_VERSION,
        core_release_id=CORE_RELEASE_ID,
        supported_assortment_modes=list(SUPPORTED_ASSORTMENT_MODES),
        supplier_trial_policy=POLICY_ID,
        supplier_trial=policy,
        release_status="v37-supplier-trial-candidate-awaiting-validation",
        source_runtime=str(source),
    )
    result = {
        "format": "outfitter-supplier-trial-runtime-builder-v1",
        "environment_protocol": ENVIRONMENT_PROTOCOL,
        "tool_interface_version": TOOL_INTERFACE_VERSION,
        "core_release_id": CORE_RELEASE_ID,
        "supplier_trial_policy": POLICY_ID,
        "supplier_trial": policy,
        "supported_assortment_modes": list(SUPPORTED_ASSORTMENT_MODES),
        "source_runtime": str(source),
        "output_runtime": str(output),
        "underlying_catalog_customer_and_economic_files_changed": False,
    }
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "report.md").write_text(
        "# V37 supplier-trial candidate\n\nV37 adds bounded supplier trials and implicit optional-line omit to the shared V36 rental core without changing catalog, demand, weather, or economic source files.\n",
        encoding="utf-8",
    )
    metadata["content_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir()) if path.is_file() and path.name != "metadata.json"
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loaded = OutfitterDataset.load(output)
    if loaded.metadata.get("supplier_trial_policy") != POLICY_ID:
        raise AssertionError("V37 supplier-trial metadata did not round-trip")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build_supplier_trial_runtime(args.source_runtime, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

