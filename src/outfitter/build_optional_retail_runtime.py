"""Project V34 into one shared V35 runtime supporting two assortment modes."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

from .assortment import (
    CORE_RELEASE_ID,
    ENVIRONMENT_PROTOCOL,
    SUPPORTED_ASSORTMENT_MODES,
    TOOL_INTERFACE_VERSION,
)
from .capital_balance_tuning import (
    ENVIRONMENT_PROTOCOL as SOURCE_PROTOCOL,
    TOOL_INTERFACE_VERSION as SOURCE_TOOL_INTERFACE_VERSION,
)
from .data import OutfitterDataset


def build_optional_retail_runtime(
    source_runtime: str | Path, output_dir: str | Path
) -> dict:
    source = Path(source_runtime).resolve()
    output = Path(output_dir).resolve()
    data = OutfitterDataset.load(source)
    if (
        data.metadata.get("environment_protocol"),
        data.metadata.get("tool_interface_version"),
    ) != (SOURCE_PROTOCOL, SOURCE_TOOL_INTERFACE_VERSION):
        raise ValueError("V35 requires the V34 capital/reservation runtime")
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("source and output must be independent directories")
    if output.exists():
        raise ValueError("output must not already exist; never overwrite a runtime")

    output.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.is_file() and path.name not in {
            "metadata.json", "result.json", "report.md",
        }:
            shutil.copy2(path, output / path.name)

    metadata = copy.deepcopy(data.metadata)
    metadata.update(
        environment_protocol=ENVIRONMENT_PROTOCOL,
        tool_interface_version=TOOL_INTERFACE_VERSION,
        core_release_id=CORE_RELEASE_ID,
        supported_assortment_modes=list(SUPPORTED_ASSORTMENT_MODES),
        assortment_projection={
            "source_catalog": "one shared full catalog",
            "rental_only": (
                "durable offers and rental demand only; retail add-ons and the "
                "retail-only supply stream are absent"
            ),
            "rental_and_retail": "complete shared catalog and existing full-store demand",
            "rental_kernel": "shared",
        },
        release_status="v35-optional-retail-candidate-awaiting-validation",
        source_runtime=str(source),
    )
    result = {
        "format": "outfitter-optional-retail-runtime-builder-v1",
        "environment_protocol": ENVIRONMENT_PROTOCOL,
        "tool_interface_version": TOOL_INTERFACE_VERSION,
        "core_release_id": CORE_RELEASE_ID,
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
        "# V35 optional-retail shared runtime\n\n"
        "Both assortment modes load this same runtime and execute the same rental "
        "kernel. Rental-only projects out retail offers, opening inventory, retail "
        "walk-in add-ons, and the retail-only supply stream. Full-store mode keeps "
        "the current behavior. Future fixes advance the shared core rather than "
        "freezing V34 as a secondary environment.\n",
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
    if tuple(loaded.metadata["supported_assortment_modes"]) != (
        SUPPORTED_ASSORTMENT_MODES
    ):
        raise AssertionError("V35 assortment metadata did not round-trip")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(
        build_optional_retail_runtime(args.source_runtime, args.output_dir),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

