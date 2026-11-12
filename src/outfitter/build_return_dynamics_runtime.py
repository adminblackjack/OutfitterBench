"""Build V46 independently of the source V45 dataset; never overwrite source."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from .data import OutfitterDataset
from .reservation_difficulty import ENVIRONMENT_PROTOCOL as SOURCE_PROTOCOL
from .return_dynamics import ENVIRONMENT_PROTOCOL, CORE_RELEASE_ID, TOOL_INTERFACE_VERSION, POLICY_ID, delay_weights


def build_return_dynamics_runtime(source_runtime, output_dir, *, delay_scale=1.):
    source, output = Path(source_runtime).resolve(), Path(output_dir).resolve()
    dataset = OutfitterDataset.load(source)
    if dataset.metadata.get("environment_protocol") != SOURCE_PROTOCOL:
        raise ValueError("V46 builder requires V45 source")
    delay_weights("dependent_transport", delay_scale)
    if output.exists() or output == source or source in output.parents or output in source.parents:
        raise ValueError("output must be a new independent directory")
    output.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.is_file() and path.name not in {"metadata.json", "report.md", "result.json"}:
            shutil.copy2(path, output / path.name)
    metadata = copy.deepcopy(dataset.metadata)
    metadata.update(environment_protocol=ENVIRONMENT_PROTOCOL, core_release_id=CORE_RELEASE_ID,
        tool_interface_version=TOOL_INTERFACE_VERSION, return_dynamics_policy=POLICY_ID,
        return_delay_scale=float(delay_scale), source_runtime=SOURCE_PROTOCOL,
        release_status="v46-return-dynamics-candidate")
    result = {"format": "outfitter-return-dynamics-builder-v1", "environment_protocol": ENVIRONMENT_PROTOCOL,
              "return_delay_scale": float(delay_scale), "source_runtime": str(source)}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    metadata["content_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir()) if path.is_file() and path.name != "metadata.json"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    OutfitterDataset.load(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--delay-scale", type=float, default=1.)
    args = parser.parse_args()
    print(json.dumps(build_return_dynamics_runtime(args.source_runtime, args.output_dir,
                                                  delay_scale=args.delay_scale), indent=2))


if __name__ == "__main__":
    main()
