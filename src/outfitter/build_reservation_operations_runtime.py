"""Project the frozen V23 runtime into V24 reservation operations."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .data import V13_RUNTIME_FORMAT


BUILDER_FORMAT = "outfitter-reservation-operations-runtime-builder-v1"
SOURCE_PROTOCOL = "outfitter-seasonal-v23-factorized-feedback-tags"
ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v24-reservation-operations"
TOOL_INTERFACE_VERSION = 20
CONTRACT_MODE = "factorized_line_options_plan_set_v1"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_reservation_operations_runtime(
    source_runtime: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source = Path(source_runtime).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format") != V13_RUNTIME_FORMAT:
        raise ValueError("V24 projection requires a V23 V13-format runtime")
    if metadata.get("environment_protocol") != SOURCE_PROTOCOL:
        raise ValueError("V24 projection requires the frozen V23 protocol")
    if int(metadata.get("tool_interface_version", 0)) != 19:
        raise ValueError("V24 projection requires V23 tool interface 19")
    profiles = dict(metadata.get("reservation_promise_profiles") or {})
    if profiles.get("contract_mode") != CONTRACT_MODE:
        raise ValueError("V24 projection requires factorized plan-set contracts")
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
        "# V24 reservation operations runtime\n\n"
        f"- Source: `{source}`\n"
        f"- Environment protocol: `{ENVIRONMENT_PROTOCOL}`.\n"
        f"- Tool interface: {TOOL_INTERFACE_VERSION}.\n"
        "- Customer choice, feedback, catalog, lifecycle economics, and random tapes "
        "are unchanged from V23.\n"
        "- The manager decision horizon is distinct from automatic runoff and final "
        "scoring. Decision-period rentals may return during runoff.\n"
        "- Dormant customer-authorized complete plans support explicit whole-plan risk "
        "recovery with line-level price and deposit protection.\n"
        "- Exact-offer walk-in occupied-asset limits can be set for current or future "
        "dates without limiting reservation contracts or inferring cleaning buffers.\n"
        "- Operationally lost walk-in demand separates service closure, manager capacity "
        "policy, physical stockout, and genuine outside choice.\n",
        encoding="utf-8",
    )
    mechanism = dict(metadata.get("reservation_mechanism") or {})
    mechanism.update(
        {
            "decision_horizon_semantics": "last_manager_action_demand_and_pickup_day",
            "runoff_semantics": "returns_cleaning_settlement_and_final_scoring",
            "decision_to_runoff_rentals_allowed": True,
            "preauthorized_complete_plan_recovery": "explicit_manager_selection_only",
            "recovery_plan_line_price_protection": "lower_of_two_authorized_amounts",
            "walk_in_capacity_limits": "exact_offer_occupied_asset_days",
            "walk_in_capacity_excludes_reservations": True,
            "walk_in_capacity_infers_cleaning_buffer": False,
        }
    )
    metadata.update(
        {
            "environment_protocol": ENVIRONMENT_PROTOCOL,
            "tool_interface_version": TOOL_INTERFACE_VERSION,
            "release_status": "v24-reservation-operations-candidate",
            "reservation_operations_runtime_builder": BUILDER_FORMAT,
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
    result = build_reservation_operations_runtime(
        args.source_runtime, args.output_dir
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
