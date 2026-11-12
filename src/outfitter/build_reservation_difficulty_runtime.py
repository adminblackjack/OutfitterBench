"""Build an independent V45 dataset from the frozen V44 runtime."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from .data import OutfitterDataset
from .trip_weather import ENVIRONMENT_PROTOCOL as SOURCE_PROTOCOL
from .reservation_difficulty import (
    CORE_RELEASE_ID, ENVIRONMENT_PROTOCOL, POLICY_ID, TOOL_INTERFACE_VERSION,
)


def build_reservation_difficulty_runtime(source_runtime, output_dir, *, peak_multiplier=1.0):
    source, output = Path(source_runtime).resolve(), Path(output_dir).resolve()
    dataset = OutfitterDataset.load(source)
    if dataset.metadata.get("environment_protocol") != SOURCE_PROTOCOL:
        raise ValueError("V45 requires the V44 runtime")
    if not 1.0 <= float(peak_multiplier) <= 2.2:
        raise ValueError("peak_multiplier must be in [1.0, 2.2]")
    if output.exists() or output == source or source in output.parents or output in source.parents:
        raise ValueError("output must be a new independent directory")
    output.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.is_file() and path.name not in {"metadata.json", "report.md", "result.json"}:
            shutil.copy2(path, output / path.name)
    metadata = copy.deepcopy(dataset.metadata)
    metadata.update(
        environment_protocol=ENVIRONMENT_PROTOCOL,
        tool_interface_version=TOOL_INTERFACE_VERSION,
        core_release_id=CORE_RELEASE_ID,
        reservation_difficulty_policy=POLICY_ID,
        reservation_calendar_peak_multiplier=float(peak_multiplier),
        source_runtime=SOURCE_PROTOCOL,
        release_status="v45-reservation-difficulty-candidate",
    )
    result = {
        "format": "outfitter-reservation-difficulty-runtime-builder-v1",
        "environment_protocol": ENVIRONMENT_PROTOCOL,
        "tool_interface_version": TOOL_INTERFACE_VERSION,
        "reservation_difficulty_policy": POLICY_ID,
        "reservation_calendar_peak_multiplier": float(peak_multiplier),
        "source_runtime": SOURCE_PROTOCOL,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "report.md").write_text(
        "# V45 reservation difficulty\n\n"
        "Shared rental core for the main and retail tracks. "
        "Optional promised lines require customer consent before reduced pickup; "
        "trip demand and first-day tariffs are versioned in code.\n"
    )
    metadata["content_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file() and p.name != "metadata.json"
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    OutfitterDataset.load(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--peak-multiplier", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(build_reservation_difficulty_runtime(
        args.source_runtime, args.output_dir, peak_multiplier=args.peak_multiplier), indent=2))


if __name__ == "__main__":
    main()
