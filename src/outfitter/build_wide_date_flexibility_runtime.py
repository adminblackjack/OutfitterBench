"""Project a frozen V32 runtime into V33's wider initial date alternatives."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

from .build_service_commitments_runtime import ENVIRONMENT_PROTOCOL as SOURCE_PROTOCOL
from .data import OutfitterDataset
from .service_commitments import POLICY_ID
from .wide_date_flexibility import DATE_OFFSETS, ENVIRONMENT_PROTOCOL, TOOL_INTERFACE_VERSION


def build_wide_date_flexibility_runtime(
    source_runtime: str | Path, output_dir: str | Path
) -> dict:
    source = Path(source_runtime).resolve()
    output = Path(output_dir).resolve()
    data = OutfitterDataset.load(source)
    if (
        data.metadata.get("environment_protocol"),
        data.metadata.get("tool_interface_version"),
    ) != (SOURCE_PROTOCOL, 28):
        raise ValueError("V33 requires a frozen V32 runtime with interface 28")
    source_policy = copy.deepcopy(data.metadata.get("service_commitment_policy") or {})
    if source_policy.get("format") != POLICY_ID:
        raise ValueError("V33 requires the frozen V32 service commitment policy")
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("source and output must be independent directories")
    if output.exists():
        raise ValueError("output must not already exist; never overwrite a runtime")

    output.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.is_file() and path.name not in {"metadata.json", "result.json", "report.md"}:
            shutil.copy2(path, output / path.name)

    policy = {**source_policy, "date_offsets": list(DATE_OFFSETS)}
    metadata = copy.deepcopy(data.metadata)
    metadata.update(
        environment_protocol=ENVIRONMENT_PROTOCOL,
        tool_interface_version=TOOL_INTERFACE_VERSION,
        release_status="v33-candidate-awaiting-deterministic-and-static-checks",
        source_runtime=str(source),
        service_commitment_policy=policy,
    )
    result = {
        "format": "outfitter-wide-date-flexibility-runtime-builder-v1",
        "environment_protocol": ENVIRONMENT_PROTOCOL,
        "tool_interface_version": TOOL_INTERFACE_VERSION,
        "source_runtime": str(source),
        "output_runtime": str(output),
        "alternative_pickup_date_absolute_offset_min_days": 3,
        "alternative_pickup_date_absolute_offset_max_days": 10,
        "alternative_pickup_date_count": 1,
        "duration_days_unchanged": True,
        "underlying_customer_and_economic_parameters_changed": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "report.md").write_text(
        "# V33 wider initial date flexibility\n\n"
        "A date-flexible inquiry still advertises exactly two discrete pickup dates: "
        "the original date and one customer-acceptable alternative. The alternative "
        "is now 3--10 days earlier or later, subject to the unchanged booking-lead, "
        "decision-horizon, and scheduled-return boundaries. Contract duration is "
        "unchanged. Inventory never influences date generation. V32 and older runtimes "
        "retain their original behavior.\n",
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
    OutfitterDataset.load(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_wide_date_flexibility_runtime(args.source_runtime, args.output_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

