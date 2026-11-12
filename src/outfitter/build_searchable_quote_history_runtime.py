"""Project the frozen V16 pairwise runtime into the V17 query-only runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .data import V13_RUNTIME_FORMAT


BUILDER_FORMAT = "outfitter-searchable-quote-history-runtime-builder-v1"
SOURCE_PROTOCOL = "outfitter-seasonal-v16-adaptive-pairwise-exact-quotes"
ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v17-searchable-quote-history"
TOOL_INTERFACE_VERSION = 13
CONTRACT_MODE = "adaptive_pairwise_exact_quotes_v1"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_searchable_quote_history_runtime(
    source_runtime: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Copy a V16 pairwise runtime and change only versioned interface metadata."""

    source = Path(source_runtime).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    source_metadata = json.loads(
        (source / "metadata.json").read_text(encoding="utf-8")
    )
    if source_metadata.get("format") != V13_RUNTIME_FORMAT:
        raise ValueError("V17 projection requires a V16 pairwise runtime")
    if source_metadata.get("environment_protocol") != SOURCE_PROTOCOL:
        raise ValueError(
            "V17 projection requires the frozen V16 pairwise protocol"
        )
    profile_metadata = dict(
        source_metadata.get("reservation_promise_profiles") or {}
    )
    if profile_metadata.get("contract_mode") != CONTRACT_MODE:
        raise ValueError(
            "V17 projection requires adaptive_pairwise_exact_quotes_v1"
        )
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty or absent")
    output.mkdir(parents=True, exist_ok=True)
    for source_file in sorted(source.iterdir()):
        if source_file.is_file() and source_file.name != "metadata.json":
            shutil.copy2(source_file, output / source_file.name)

    metadata = dict(source_metadata)
    profile_metadata.update(
        {
            "choice_history": "immutable_raw_searchable_paginated",
            "choice_history_sort_default": "newest_first",
            "choice_history_default_limit": 10,
        }
    )
    metadata.update(
        {
            "environment_protocol": ENVIRONMENT_PROTOCOL,
            "tool_interface_version": TOOL_INTERFACE_VERSION,
            "release_status": "v17-searchable-quote-history-candidate",
            "searchable_quote_history_runtime_builder": BUILDER_FORMAT,
            "source_runtime": str(source),
            "reservation_promise_profiles": profile_metadata,
        }
    )
    metadata["content_sha256"] = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "metadata.json"
    }
    _write_json(output / "metadata.json", metadata)
    return {
        "format": BUILDER_FORMAT,
        "source_runtime": str(source),
        "output_runtime": str(output),
        "environment_protocol": ENVIRONMENT_PROTOCOL,
        "tool_interface_version": TOOL_INTERFACE_VERSION,
        "contract_mode": CONTRACT_MODE,
        "metadata_only_projection": True,
        "metadata_sha256": _sha256(output / "metadata.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = build_searchable_quote_history_runtime(
        args.source_runtime, args.output_dir
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
