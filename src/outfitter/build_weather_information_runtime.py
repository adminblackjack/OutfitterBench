"""Project V41 into the V42 weather-information-coherent runtime."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from .actor_interface_slimming import (
    ENVIRONMENT_PROTOCOL as SOURCE_PROTOCOL,
    TOOL_INTERFACE_VERSION as SOURCE_TOOL_INTERFACE_VERSION,
)
from .data import OutfitterDataset
from .weather_information import (
    CORE_RELEASE_ID,
    ENVIRONMENT_PROTOCOL,
    POLICY_ID,
    TOOL_INTERFACE_VERSION,
    policy_document,
)


def build_weather_information_runtime(
    source_runtime: str | Path, output_dir: str | Path
) -> dict:
    source = Path(source_runtime).resolve()
    output = Path(output_dir).resolve()
    data = OutfitterDataset.load(source)
    if (
        data.metadata.get("environment_protocol"),
        data.metadata.get("tool_interface_version"),
    ) != (SOURCE_PROTOCOL, SOURCE_TOOL_INTERFACE_VERSION):
        raise ValueError("V42 requires the shared V41 runtime")
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
        weather_information_policy=POLICY_ID,
        weather_information=policy,
        release_status="v42-weather-information-coherence-candidate",
        source_runtime=SOURCE_PROTOCOL,
    )
    result = {
        "format": "outfitter-weather-information-runtime-builder-v1",
        "environment_protocol": ENVIRONMENT_PROTOCOL,
        "tool_interface_version": TOOL_INTERFACE_VERSION,
        "core_release_id": CORE_RELEASE_ID,
        "weather_information_policy": POLICY_ID,
        "weather_information": policy,
        "source_runtime": SOURCE_PROTOCOL,
        "output_runtime": str(output),
        "weather_source_files_changed": False,
        "customer_demand_process_changed": True,
        "economics_or_scoring_changed": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "report.md").write_text(
        "# V42 weather-information-coherence candidate\n\n"
        "V42 replaces the weather-leaking arrival tape with a weather-independent "
        "potential-customer pool, routes each customer once, and lets reservation "
        "and walk-in customers act only on information available to them. It also "
        "separates non-weather cancellation and prevents repeated reactions to one "
        "forecast deterioration. Economic and scoring rules are inherited from "
        "V41.\n",
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
    if loaded.metadata.get("weather_information_policy") != POLICY_ID:
        raise AssertionError("V42 policy metadata did not round-trip")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_weather_information_runtime(
                args.source_runtime, args.output_dir
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
