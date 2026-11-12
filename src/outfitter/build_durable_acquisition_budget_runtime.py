"""Project the shared V35 runtime into a V36 durable-budget candidate."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from .acquisition_budget import (
    CORE_RELEASE_ID,
    ENVIRONMENT_PROTOCOL,
    POLICY_ID,
    TOOL_INTERFACE_VERSION,
)
from .assortment import (
    ENVIRONMENT_PROTOCOL as SOURCE_PROTOCOL,
    SUPPORTED_ASSORTMENT_MODES,
    TOOL_INTERFACE_VERSION as SOURCE_TOOL_INTERFACE_VERSION,
)
from .data import OutfitterDataset


def build_durable_acquisition_budget_runtime(
    source_runtime: str | Path, output_dir: str | Path
) -> dict:
    source = Path(source_runtime).resolve()
    output = Path(output_dir).resolve()
    data = OutfitterDataset.load(source)
    if (
        data.metadata.get("environment_protocol"),
        data.metadata.get("tool_interface_version"),
    ) != (SOURCE_PROTOCOL, SOURCE_TOOL_INTERFACE_VERSION):
        raise ValueError("V36 requires the shared V35 optional-retail runtime")
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("source and output must be independent directories")
    if output.exists():
        raise ValueError("output must not already exist; never overwrite a runtime")

    output.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.is_file() and path.name not in {
            "metadata.json",
            "result.json",
            "report.md",
        }:
            shutil.copy2(path, output / path.name)

    metadata = copy.deepcopy(data.metadata)
    metadata.update(
        environment_protocol=ENVIRONMENT_PROTOCOL,
        tool_interface_version=TOOL_INTERFACE_VERSION,
        core_release_id=CORE_RELEASE_ID,
        supported_assortment_modes=list(SUPPORTED_ASSORTMENT_MODES),
        durable_acquisition_budget_policy=POLICY_ID,
        durable_acquisition_budget={
            "initial_assets_consume_budget": False,
            "accepted_durable_order_lines_consume_budget": True,
            "retail_order_lines_consume_budget": False,
            "replenishment": "none",
            "supplier_cancellation_or_refund_mechanism": False,
            "unused_budget_has_terminal_value": False,
            "accounting": "fully delivered durable value is used; accepted in-transit durable value is committed",
        },
        release_status="v36-durable-budget-candidate-awaiting-B-calibration",
        source_runtime=str(source),
    )
    result = {
        "format": "outfitter-durable-acquisition-budget-runtime-builder-v1",
        "environment_protocol": ENVIRONMENT_PROTOCOL,
        "tool_interface_version": TOOL_INTERFACE_VERSION,
        "core_release_id": CORE_RELEASE_ID,
        "durable_acquisition_budget_policy": POLICY_ID,
        "supported_assortment_modes": list(SUPPORTED_ASSORTMENT_MODES),
        "source_runtime": str(source),
        "output_runtime": str(output),
        "underlying_catalog_customer_and_economic_files_changed": False,
        "separate_mode_runtime_copies_created": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "report.md").write_text(
        "# V36 durable acquisition budget candidate\n\n"
        "Both assortment modes use the same non-replenishing durable acquisition "
        "budget implementation. Retail lines are exempt. No supplier cancellation "
        "or budget restoration path is introduced. The runtime is awaiting a "
        "finite-B calibration decision.\n",
        encoding="utf-8",
    )
    metadata["content_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "metadata.json"
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded = OutfitterDataset.load(output)
    if (
        loaded.metadata.get("durable_acquisition_budget_policy") != POLICY_ID
        or tuple(loaded.metadata["supported_assortment_modes"])
        != SUPPORTED_ASSORTMENT_MODES
    ):
        raise AssertionError("V36 durable-budget metadata did not round-trip")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_durable_acquisition_budget_runtime(
                args.source_runtime, args.output_dir
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
