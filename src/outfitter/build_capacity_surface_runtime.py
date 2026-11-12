"""Project the frozen V21 runtime into the V22 capacity/interface runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .data import V13_RUNTIME_FORMAT


BUILDER_FORMAT = "outfitter-capacity-surface-runtime-builder-v1"
SOURCE_PROTOCOL = "outfitter-seasonal-v21-noisy-line-mention"
ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v22-capacity-surface"
TOOL_INTERFACE_VERSION = 18
CONTRACT_MODE = "factorized_line_options_plan_set_v1"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_capacity_surface_runtime(
    source_runtime: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source = Path(source_runtime).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format") != V13_RUNTIME_FORMAT:
        raise ValueError("V22 projection requires a V21 V13-format runtime")
    if metadata.get("environment_protocol") != SOURCE_PROTOCOL:
        raise ValueError("V22 projection requires the frozen V21 protocol")
    if int(metadata.get("tool_interface_version", 0)) != 17:
        raise ValueError("V22 projection requires V21 tool interface 17")
    profile_metadata = dict(metadata.get("reservation_promise_profiles") or {})
    if profile_metadata.get("contract_mode") != CONTRACT_MODE:
        raise ValueError("V22 projection requires the factorized plan-set mode")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty or absent")
    output.mkdir(parents=True, exist_ok=True)
    for source_file in sorted(source.iterdir()):
        if source_file.is_file() and source_file.name not in {
            "metadata.json",
            "report.md",
        }:
            shutil.copy2(source_file, output / source_file.name)

    (output / "report.md").write_text(
        "# V22 raw rental-capacity and compact reservation-interface runtime\n\n"
        f"- Source: `{source}`\n"
        f"- Environment protocol: `{ENVIRONMENT_PROTOCOL}`\n"
        f"- Tool interface: {TOOL_INTERFACE_VERSION}\n"
        f"- Contract mode: `{CONTRACT_MODE}` (unchanged).\n"
        "- Capacity: `view_rental_capacity` replaces the Agent-visible legacy "
        "calendar with raw asset counts, binding intervals, public delivery "
        "timing, and current service jobs; it computes no feasibility.\n"
        "- Delivery privacy: in-transit orders expose ETA only; actual arrival "
        "is public only after delivery.\n"
        "- Reservation records: maximum page size 10, normalized compact quote "
        "history, and sectioned exact-ID detail.\n"
        "- Quote validation: one atomic rejected batch reports all invalid "
        "request/line/option items before any customer evaluation.\n"
        "- Customer choice, V21 line mentions, economics, inventory, and random "
        "tapes are unchanged.\n",
        encoding="utf-8",
    )

    capacity_metadata = {
        "tool": "view_rental_capacity",
        "views": ["timeline", "service_jobs"],
        "legacy_calendar_agent_visible": False,
        "in_transit_actual_delivery_visible": False,
        "timeline_computes_feasibility": False,
        "cleaning_projection": "current_global_fifo_jobs_only",
        "future_return_jobs_included": False,
    }
    history_metadata = {
        "page_limit_default": 10,
        "page_limit_maximum": 10,
        "quote_history": "normalized_line_options_and_plan_references",
        "detail_sections": [
            "current_terms",
            "submitted_quote",
            "customer_authorization",
            "fulfillment",
            "changes",
            "risk",
        ],
        "detail_default_sections": ["current_terms", "fulfillment"],
    }
    mechanism = dict(metadata.get("reservation_mechanism") or {})
    mechanism.update(
        {
            "capacity_surface": capacity_metadata,
            "reservation_record_surface": history_metadata,
            "quote_batch_validation": "atomic_all_invalid_items",
        }
    )
    metadata.update(
        {
            "environment_protocol": ENVIRONMENT_PROTOCOL,
            "tool_interface_version": TOOL_INTERFACE_VERSION,
            "release_status": "v22-capacity-surface-candidate",
            "capacity_surface_runtime_builder": BUILDER_FORMAT,
            "source_runtime": str(source),
            "reservation_mechanism": mechanism,
            "rental_capacity_surface": capacity_metadata,
            "reservation_record_surface": history_metadata,
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
        "metadata_sha256": _sha256(output / "metadata.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = build_capacity_surface_runtime(
        args.source_runtime, args.output_dir
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
