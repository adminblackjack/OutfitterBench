"""Project V37 into the shared V38 proactive-contingency runtime."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from .data import OutfitterDataset
from .reservation_contingency import (
    CORE_RELEASE_ID,
    ENVIRONMENT_PROTOCOL,
    POLICY_ID,
    TOOL_INTERFACE_VERSION,
    policy_document,
)
from .supplier_trial import (
    ENVIRONMENT_PROTOCOL as SOURCE_PROTOCOL,
    TOOL_INTERFACE_VERSION as SOURCE_TOOL_INTERFACE_VERSION,
)


def build_proactive_contingency_runtime(
    source_runtime: str | Path, output_dir: str | Path
) -> dict:
    source = Path(source_runtime).resolve()
    output = Path(output_dir).resolve()
    data = OutfitterDataset.load(source)
    if (
        data.metadata.get("environment_protocol"),
        data.metadata.get("tool_interface_version"),
    ) != (SOURCE_PROTOCOL, SOURCE_TOOL_INTERFACE_VERSION):
        raise ValueError("V38 requires the shared V37 supplier-trial runtime")
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
    service = dict(metadata.get("service_commitment_policy") or {})
    service.pop("date_flexibility_probability", None)
    service["initial_date_alternative_policy"] = (
        "customer_preference_weighted_v1"
    )
    metadata.update(
        environment_protocol=ENVIRONMENT_PROTOCOL,
        tool_interface_version=TOOL_INTERFACE_VERSION,
        core_release_id=CORE_RELEASE_ID,
        proactive_contingency_policy=POLICY_ID,
        proactive_contingency=policy,
        service_commitment_policy=service,
        release_status="v38-proactive-contingency-candidate-awaiting-validation",
        source_runtime=str(source),
    )
    result = {
        "format": "outfitter-proactive-contingency-runtime-builder-v1",
        "environment_protocol": ENVIRONMENT_PROTOCOL,
        "tool_interface_version": TOOL_INTERFACE_VERSION,
        "core_release_id": CORE_RELEASE_ID,
        "proactive_contingency_policy": POLICY_ID,
        "proactive_contingency": policy,
        "source_runtime": str(source),
        "output_runtime": str(output),
        "underlying_catalog_customer_and_economic_files_changed": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "report.md").write_text(
        "# V38 proactive contingency candidate\n\n"
        "V38 adds preference-weighted initial dates and one delayed proactive "
        "contingency inquiry per commitment without changing catalog, demand, "
        "weather, supplier-trial, or economic source files.\n",
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
    if loaded.metadata.get("proactive_contingency_policy") != POLICY_ID:
        raise AssertionError("V38 contingency metadata did not round-trip")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_proactive_contingency_runtime(
                args.source_runtime, args.output_dir
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
