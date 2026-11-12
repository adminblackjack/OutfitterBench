"""Project the frozen V25 runtime into V26 contract-state semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .data import V13_RUNTIME_FORMAT

BUILDER_FORMAT = "outfitter-contract-state-runtime-builder-v1"
SOURCE_PROTOCOL = "outfitter-seasonal-v25-request-quote-workspace"
ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v26-contract-state-reschedule-tail"
TOOL_INTERFACE_VERSION = 22
CONTRACT_MODE = "factorized_line_options_plan_set_v1"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_contract_state_runtime(
    source_runtime: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source = Path(source_runtime).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format") != V13_RUNTIME_FORMAT:
        raise ValueError("V26 projection requires a V25 V13-format runtime")
    if metadata.get("environment_protocol") != SOURCE_PROTOCOL:
        raise ValueError("V26 projection requires the frozen V25 protocol")
    if int(metadata.get("tool_interface_version", 0)) != 21:
        raise ValueError("V26 projection requires V25 tool interface 21")
    profiles = dict(metadata.get("reservation_promise_profiles") or {})
    if profiles.get("contract_mode") != CONTRACT_MODE:
        raise ValueError("V26 projection requires factorized plan-set contracts")
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
        "# V26 reservation contract-state and reschedule-tail runtime\n\n"
        f"- Source: `{source}`\n"
        f"- Environment protocol: `{ENVIRONMENT_PROTOCOL}`.\n"
        f"- Tool interface: {TOOL_INTERFACE_VERSION}.\n"
        "- Inquiry generation, factorized quote choice, customer feedback, quote "
        "workspace, capacity, economics, product data, and random tapes are "
        "unchanged from V25.\n"
        "- Reservation views separate immutable inquiry terms from current "
        "commitment terms and separate historical cash facts from live balances.\n"
        "- The first reschedule-request probability is unchanged. Repeated "
        "non-weather/uncertainty pressure decays, and only incremental weather "
        "deterioration receives the full weather response.\n",
        encoding="utf-8",
    )
    mechanism = dict(metadata.get("reservation_mechanism") or {})
    mechanism.update(
        {
            "reservation_term_projection": (
                "immutable_inquiry_plus_current_commitment_v1"
            ),
            "reservation_financial_projection": (
                "gross_history_plus_live_balances_v1"
            ),
            "reschedule_probability_model": (
                "first_unchanged_incremental_weather_repeat_decay_v1"
            ),
            "reschedule_nondeterioration_decay_after_one": 0.30,
            "reschedule_nondeterioration_decay_after_two": 0.075,
            "reschedule_weather_component_decay": 1.0,
            "reschedule_same_forecast_counts_as_new_deterioration": False,
            "accepted_reschedule_resets_forecast_baseline": True,
            "hard_reschedule_request_cap": None,
        }
    )
    metadata.update(
        {
            "environment_protocol": ENVIRONMENT_PROTOCOL,
            "tool_interface_version": TOOL_INTERFACE_VERSION,
            "release_status": "v26-contract-state-reschedule-tail-candidate",
            "contract_state_runtime_builder": BUILDER_FORMAT,
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
    result = build_contract_state_runtime(args.source_runtime, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
