"""Build an independent V44 runtime from V43; prices and random rates unchanged."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from .actor_manual import ENVIRONMENT_PROTOCOL as SOURCE_PROTOCOL
from .trip_weather import ENVIRONMENT_PROTOCOL, TOOL_INTERFACE_VERSION, CORE_RELEASE_ID, POLICY_ID
from .data import OutfitterDataset


def build_trip_weather_runtime(source_runtime, output_dir):
    source, output = Path(source_runtime).resolve(), Path(output_dir).resolve()
    dataset = OutfitterDataset.load(source)
    if dataset.metadata.get("environment_protocol") != SOURCE_PROTOCOL:
        raise ValueError("V44 requires the V43 runtime")
    if output.exists() or output == source or source in output.parents or output in source.parents:
        raise ValueError("output must be a new independent directory")
    output.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.is_file() and path.name not in {"metadata.json", "report.md", "result.json"}:
            shutil.copy2(path, output / path.name)
    metadata = copy.deepcopy(dataset.metadata)
    metadata.update(environment_protocol=ENVIRONMENT_PROTOCOL, tool_interface_version=TOOL_INTERFACE_VERSION,
                    core_release_id=CORE_RELEASE_ID, trip_weather_policy=POLICY_ID,
                    source_runtime=SOURCE_PROTOCOL, release_status="v44-trip-weather-close-outcomes-candidate")
    result = {"format": "outfitter-trip-weather-runtime-builder-v1",
              "environment_protocol": ENVIRONMENT_PROTOCOL, "tool_interface_version": TOOL_INTERFACE_VERSION,
              "trip_weather_policy": POLICY_ID, "public_forecast_dates": 15,
              "uncovered_dates": "unknown-no-extrapolation",
              "temperature_evaluation": "minimum-covered-trip-forecast-night-low",
              "customer_temperature_evaluation_changed": True,
              "customer_arrival_and_cancellation_parameters_changed": False,
              "economics_changed": False, "scoring_changed": False,
              "close_outcomes": "frozen-actual-results-with-payment-status"}
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "report.md").write_text(
        "# V44 trip weather and close outcomes\n\n"
        "Customer temperature evaluation uses a decision-time public forecast of the target trip, "
        "with unknown dates beyond the existing 15-date window. Closing receipts and the next "
        "morning report include actual reservation outcomes and payment status.\n")
    metadata["content_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file() and p.name != "metadata.json"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    OutfitterDataset.load(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build_trip_weather_runtime(args.source_runtime, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
