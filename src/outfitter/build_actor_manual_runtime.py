"""Build V43 from an immutable V42 runtime without changing world data."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from .actor_manual import ENVIRONMENT_PROTOCOL, TOOL_INTERFACE_VERSION, CORE_RELEASE_ID, POLICY_ID
from .weather_information import ENVIRONMENT_PROTOCOL as SOURCE_PROTOCOL
from .data import OutfitterDataset


def build_actor_manual_runtime(source_runtime: str | Path, output_dir: str | Path) -> dict:
    source, output = Path(source_runtime).resolve(), Path(output_dir).resolve()
    data = OutfitterDataset.load(source)
    if data.metadata.get("environment_protocol") != SOURCE_PROTOCOL or data.metadata.get("tool_interface_version") != 37:
        raise ValueError("V43 requires a V42 runtime")
    if output == source or source in output.parents or output in source.parents or output.exists():
        raise ValueError("output must be a new independent directory")
    output.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.is_file() and path.name not in {"metadata.json", "report.md", "result.json"}:
            shutil.copy2(path, output / path.name)
    metadata = copy.deepcopy(data.metadata)
    metadata.update(environment_protocol=ENVIRONMENT_PROTOCOL, tool_interface_version=TOOL_INTERFACE_VERSION,
                    core_release_id=CORE_RELEASE_ID, business_manual_policy=POLICY_ID,
                    source_runtime=SOURCE_PROTOCOL, release_status="v43-public-business-manual-candidate")
    result = {
        "format": "outfitter-public-business-manual-runtime-builder-v1",
        "environment_protocol": ENVIRONMENT_PROTOCOL, "tool_interface_version": TOOL_INTERFACE_VERSION,
        "core_release_id": CORE_RELEASE_ID, "source_runtime": SOURCE_PROTOCOL,
        "business_manual_policy": POLICY_ID, "customer_world_changed": False,
        "economics_changed": False, "scoring_changed": False,
        "actor_information_changed": True,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "report.md").write_text(
        "# V43 public business manual\n\n"
        "A concise system prompt, queryable public terms, current tool descriptions, "
        "public reservation-event projection and explicit response budgets. "
        "V42 customer processes, random tape, economics and scoring are unchanged.\n"
    )
    metadata["content_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file() and p.name != "metadata.json"
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    OutfitterDataset.load(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build_actor_manual_runtime(args.source_runtime, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
