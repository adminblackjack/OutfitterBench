"""Project V39 into the shared V40 revolving-capital runtime."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from .contingency_actor_interface import (
    ENVIRONMENT_PROTOCOL as SOURCE_PROTOCOL,
    TOOL_INTERFACE_VERSION as SOURCE_TOOL_INTERFACE_VERSION,
)
from .data import OutfitterDataset
from .revolving_durable_capital import (
    CORE_RELEASE_ID,
    ENVIRONMENT_PROTOCOL,
    POLICY_ID,
    TOOL_INTERFACE_VERSION,
    policy_document,
)


def build_revolving_durable_capital_runtime(
    source_runtime: str | Path, output_dir: str | Path
) -> dict:
    source = Path(source_runtime).resolve()
    output = Path(output_dir).resolve()
    data = OutfitterDataset.load(source)
    if (
        data.metadata.get("environment_protocol"),
        data.metadata.get("tool_interface_version"),
    ) != (SOURCE_PROTOCOL, SOURCE_TOOL_INTERFACE_VERSION):
        raise ValueError("V40 requires the shared V39 actor-interface runtime")
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

    policy = policy_document()
    metadata = copy.deepcopy(data.metadata)
    metadata.update(
        environment_protocol=ENVIRONMENT_PROTOCOL,
        tool_interface_version=TOOL_INTERFACE_VERSION,
        core_release_id=CORE_RELEASE_ID,
        durable_acquisition_budget_policy=POLICY_ID,
        durable_capital_policy=policy,
        release_status="v40-revolving-durable-capital-candidate",
        source_runtime=str(source),
    )
    result = {
        "format": "outfitter-revolving-durable-capital-runtime-builder-v1",
        "environment_protocol": ENVIRONMENT_PROTOCOL,
        "tool_interface_version": TOOL_INTERFACE_VERSION,
        "core_release_id": CORE_RELEASE_ID,
        "durable_acquisition_budget_policy": POLICY_ID,
        "durable_capital_policy": policy,
        "source_runtime": str(source),
        "output_runtime": str(output),
        "underlying_customer_world_and_economic_files_changed": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "report.md").write_text(
        "# V40 revolving durable capital candidate\n\n"
        "V40 replaces only the non-replenishing durable acquisition budget with "
        "a fixed gross deployed-capital envelope. V39 customer behavior, actor "
        "clarity, demand, economics, assortment projections and scoring remain "
        "otherwise unchanged.\n",
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
    if loaded.metadata.get("durable_acquisition_budget_policy") != POLICY_ID:
        raise AssertionError("V40 durable-capital metadata did not round-trip")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_revolving_durable_capital_runtime(
                args.source_runtime, args.output_dir
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
