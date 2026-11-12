"""Project frozen V28 data into the V29 accountability interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .data import V13_RUNTIME_FORMAT


BUILDER_FORMAT = "outfitter-liquidation-accountability-runtime-builder-v1"
SOURCE_PROTOCOL = "outfitter-seasonal-v28-capacity-summary-audit"
ENVIRONMENT_PROTOCOL = (
    "outfitter-seasonal-v29-flat-liquidation-reservation-accountability"
)
TOOL_INTERFACE_VERSION = 25
CONTRACT_MODE = "factorized_line_options_plan_set_v1"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_liquidation_accountability_runtime(
    source_runtime: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source = Path(source_runtime).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format") != V13_RUNTIME_FORMAT:
        raise ValueError("V29 projection requires a V28 V13-format runtime")
    if metadata.get("environment_protocol") != SOURCE_PROTOCOL:
        raise ValueError("V29 projection requires the frozen V28 protocol")
    if int(metadata.get("tool_interface_version", 0)) != 24:
        raise ValueError("V29 projection requires V28 tool interface 24")
    profiles = dict(metadata.get("reservation_promise_profiles") or {})
    if profiles.get("contract_mode") != CONTRACT_MODE:
        raise ValueError("V29 projection requires factorized plan-set contracts")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty or absent")
    output.mkdir(parents=True, exist_ok=True)
    for source_file in sorted(source.iterdir()):
        if source_file.is_file() and source_file.name not in {
            "metadata.json",
            "report.md",
        }:
            shutil.copy2(source_file, output / source_file.name)

    settlement = {
        "format": "outfitter-reservation-failure-settlement-v1",
        "manager_cancellation": {
            "locked_rent_rate": 0.50,
            "rent_notice_floor": 0.25,
            "service_base": 10.0,
            "service_unit_day_rate": 2.0,
            "service_notice_floor": 0.35,
            "notice_decay_days": 7,
        },
        "pickup_failure": {
            "locked_rent_rate": 1.0,
            "service_base": 15.0,
            "service_unit_day_rate": 3.0,
        },
        "prepayment_refund_is_separate": True,
        "rounding": "cent_half_up_after_max",
    }
    reputation = {
        "format": "outfitter-reservation-service-reputation-v1",
        "formal_default_enabled": False,
        "outcome_window": 50,
        "cold_start_pseudo_outcomes": 50,
        "choice_penalty_slope": 20.0,
        "choice_penalty_cap": 2.5,
        "reliability_score_burden_multiplier": 2.0,
        "failure_weights": {
            "manager_cancel_notice_8_plus": 0.15,
            "manager_cancel_notice_4_7": 0.25,
            "manager_cancel_notice_2_3": 0.45,
            "manager_cancel_notice_1": 0.60,
            "manager_cancel_same_day": 0.75,
            "pickup_failure": 1.0,
            "pickup_fulfilled": 0.0,
        },
        "excluded_outcomes": [
            "customer_cancellation",
            "optional_line_missing_credit",
        ],
    }
    (output / "report.md").write_text(
        "# V29 flat liquidation and reservation accountability runtime\n\n"
        f"- Source: `{source}`\n"
        f"- Environment protocol: `{ENVIRONMENT_PROTOCOL}`.\n"
        f"- Tool interface: {TOOL_INTERFACE_VERSION}.\n"
        "- Rental and retail dealer values are flat per-unit nominal quotes; "
        "there is no rolling cross-offer liquidation-depth pool.\n"
        "- Managers may cancel confirmed pre-pickup commitments using a dedicated "
        "tool. Refund and public compensation terms are shared by preview and "
        "execution.\n"
        "- Bounded service reputation is available only as an explicit experiment "
        "and is disabled in formal V29 configurations.\n"
        "- Catalog, customer demand, quote construction, feedback, weather, and "
        "random tapes are otherwise frozen from V28.\n",
        encoding="utf-8",
    )
    mechanism = dict(metadata.get("reservation_mechanism") or {})
    mechanism.update(
        {
            "proactive_manager_cancellation": True,
            "manager_cancellation_tool": "cancel_reservation_commitment",
            "failure_settlement_policy": settlement["format"],
            "service_reputation_experiment": reputation["format"],
            "service_reputation_formal_default_enabled": False,
        }
    )
    metadata.update(
        {
            "environment_protocol": ENVIRONMENT_PROTOCOL,
            "tool_interface_version": TOOL_INTERFACE_VERSION,
            "release_status": "v29-accountability-candidate",
            "liquidation_accountability_runtime_builder": BUILDER_FORMAT,
            "source_runtime": str(source),
            "liquidation_policy": "flat_nominal_per_unit_v1",
            "reservation_failure_settlement": settlement,
            "reservation_service_reputation": reputation,
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
        "liquidation_policy": "flat_nominal_per_unit_v1",
        "metadata_sha256": _sha256(output / "metadata.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = build_liquidation_accountability_runtime(
        args.source_runtime, args.output_dir
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
