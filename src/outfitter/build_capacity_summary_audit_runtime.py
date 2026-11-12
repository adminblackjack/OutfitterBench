"""Project frozen V27 data into the V28 capacity/summary audit interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .data import V13_RUNTIME_FORMAT

BUILDER_FORMAT = "outfitter-capacity-summary-audit-runtime-builder-v1"
SOURCE_PROTOCOL = "outfitter-seasonal-v27-extension-cash-audit"
ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v28-capacity-summary-audit"
TOOL_INTERFACE_VERSION = 24
CONTRACT_MODE = "factorized_line_options_plan_set_v1"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_capacity_summary_audit_runtime(
    source_runtime: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source = Path(source_runtime).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format") != V13_RUNTIME_FORMAT:
        raise ValueError("V28 projection requires a V27 V13-format runtime")
    if metadata.get("environment_protocol") != SOURCE_PROTOCOL:
        raise ValueError("V28 projection requires the frozen V27 protocol")
    if int(metadata.get("tool_interface_version", 0)) != 23:
        raise ValueError("V28 projection requires V27 tool interface 23")
    profiles = dict(metadata.get("reservation_promise_profiles") or {})
    if profiles.get("contract_mode") != CONTRACT_MODE:
        raise ValueError("V28 projection requires factorized plan-set contracts")
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
        "# V28 walk-in capacity and summary audit runtime\n\n"
        f"- Source: `{source}`\n"
        f"- Environment protocol: `{ENVIRONMENT_PROTOCOL}`.\n"
        f"- Tool interface: {TOOL_INTERFACE_VERSION}.\n"
        "- Customer behavior, reservation contracts, capacity enforcement, "
        "economics, product data, and random tapes are unchanged from V27.\n"
        "- Walk-in capacity set is explicitly an upsert, set/clear use a "
        "discriminated schema, invalid batches return every item/conflict, and "
        "clear safely ignores historical prefixes without rewriting history.\n"
        "- Summary output adds reconciled cash_flow, reservation_rent_cash, and "
        "value_change blocks instead of more loose accounting fields.\n",
        encoding="utf-8",
    )
    mechanism = dict(metadata.get("reservation_mechanism") or {})
    mechanism.update(
        {
            "walk_in_capacity_set_semantics": "daily_upsert",
            "walk_in_capacity_item_schema": "discriminated_set_clear_v1",
            "walk_in_capacity_batch_validation": "all_items_and_conflicts_v1",
            "walk_in_capacity_batch_order_dependency": False,
            "walk_in_capacity_clear_historical_prefix": (
                "ignored_without_history_rewrite"
            ),
            "summary_format": "outfitter-summary-v2-auditable",
            "summary_structured_blocks": [
                "cash_flow",
                "reservation_rent_cash",
                "value_change",
            ],
        }
    )
    metadata.update(
        {
            "environment_protocol": ENVIRONMENT_PROTOCOL,
            "tool_interface_version": TOOL_INTERFACE_VERSION,
            "release_status": "v28-capacity-summary-audit-candidate",
            "capacity_summary_audit_runtime_builder": BUILDER_FORMAT,
            "source_runtime": str(source),
            "reservation_mechanism": mechanism,
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
    result = build_capacity_summary_audit_runtime(
        args.source_runtime, args.output_dir
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
