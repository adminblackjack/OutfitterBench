"""Project V33 into V34 while changing only two frozen generation probabilities."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

from .capital_balance_tuning import (
    DATE_FLEXIBILITY_PROBABILITY,
    ENVIRONMENT_PROTOCOL,
    SERVICE_OFFER_ATTEMPT_PROBABILITY,
    TOOL_INTERFACE_VERSION,
)
from .data import OutfitterDataset
from .service_commitments import POLICY_ID
from .wide_date_flexibility import (
    DATE_OFFSETS,
    ENVIRONMENT_PROTOCOL as SOURCE_PROTOCOL,
    TOOL_INTERFACE_VERSION as SOURCE_TOOL_INTERFACE_VERSION,
)


def build_capital_balance_runtime(
    source_runtime: str | Path, output_dir: str | Path
) -> dict:
    source = Path(source_runtime).resolve()
    output = Path(output_dir).resolve()
    data = OutfitterDataset.load(source)
    if (
        data.metadata.get("environment_protocol"),
        data.metadata.get("tool_interface_version"),
    ) != (SOURCE_PROTOCOL, SOURCE_TOOL_INTERFACE_VERSION):
        raise ValueError("V34 requires the frozen V33 wide-date runtime")
    source_policy = copy.deepcopy(data.metadata.get("service_commitment_policy") or {})
    if (
        source_policy.get("format") != POLICY_ID
        or source_policy.get("date_offsets") != DATE_OFFSETS
    ):
        raise ValueError("V34 requires V33's frozen service policy and date offsets")
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("source and output must be independent directories")
    if output.exists():
        raise ValueError("output must not already exist; never overwrite a runtime")

    output.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.is_file() and path.name not in {"metadata.json", "result.json", "report.md"}:
            shutil.copy2(path, output / path.name)

    policy = {
        **source_policy,
        "service_offer_attempt_probability": SERVICE_OFFER_ATTEMPT_PROBABILITY,
        "date_flexibility_probability": DATE_FLEXIBILITY_PROBABILITY,
    }
    metadata = copy.deepcopy(data.metadata)
    metadata.update(
        environment_protocol=ENVIRONMENT_PROTOCOL,
        tool_interface_version=TOOL_INTERFACE_VERSION,
        release_status="v34-candidate-awaiting-deterministic-checks-and-seed42-llm-runs",
        source_runtime=str(source),
        service_commitment_policy=policy,
    )
    result = {
        "format": "outfitter-capital-reservation-balance-runtime-builder-v1",
        "environment_protocol": ENVIRONMENT_PROTOCOL,
        "tool_interface_version": TOOL_INTERFACE_VERSION,
        "source_runtime": str(source),
        "output_runtime": str(output),
        "service_offer_attempt_probability": SERVICE_OFFER_ATTEMPT_PROBABILITY,
        "date_flexibility_probability": DATE_FLEXIBILITY_PROBABILITY,
        "date_offsets_unchanged_from_v33": True,
        "underlying_customer_and_economic_parameter_files_changed": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "report.md").write_text(
        "# V34 capital/reservation balance candidate\n\n"
        "This independent projection retains V33's 3--10 day alternative-date "
        "space and all catalog, customer-choice, price, cost, compensation, lifecycle, "
        "and valuation files. It changes only the fixed-service-offer attempt probability "
        "from 1/3 to 0.40 and the two-date attempt probability from 1/3 to 0.50. "
        "The formal run configuration separately uses warehouse N=190 and reservation "
        "share 0.70. Observed successful shares may be lower after unchanged legality and "
        "customer-coherence filters.\n",
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
    print(json.dumps(build_capital_balance_runtime(args.source_runtime, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
