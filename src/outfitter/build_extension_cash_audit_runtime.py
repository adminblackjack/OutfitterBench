"""Project the frozen V26 runtime into V27 extension-cash audit semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .data import V13_RUNTIME_FORMAT

BUILDER_FORMAT = "outfitter-extension-cash-audit-runtime-builder-v1"
SOURCE_PROTOCOL = "outfitter-seasonal-v26-contract-state-reschedule-tail"
ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v27-extension-cash-audit"
TOOL_INTERFACE_VERSION = 23
CONTRACT_MODE = "factorized_line_options_plan_set_v1"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_extension_cash_audit_runtime(
    source_runtime: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source = Path(source_runtime).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format") != V13_RUNTIME_FORMAT:
        raise ValueError("V27 projection requires a V26 V13-format runtime")
    if metadata.get("environment_protocol") != SOURCE_PROTOCOL:
        raise ValueError("V27 projection requires the frozen V26 protocol")
    if int(metadata.get("tool_interface_version", 0)) != 22:
        raise ValueError("V27 projection requires V26 tool interface 22")
    profiles = dict(metadata.get("reservation_promise_profiles") or {})
    if profiles.get("contract_mode") != CONTRACT_MODE:
        raise ValueError("V27 projection requires factorized plan-set contracts")
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
        "# V27 reservation extension-cash audit runtime\n\n"
        f"- Source: `{source}`\n"
        f"- Environment protocol: `{ENVIRONMENT_PROTOCOL}`.\n"
        f"- Tool interface: {TOOL_INTERFACE_VERSION}.\n"
        "- Customer choice, feedback, reservation generation, reschedule-tail "
        "probabilities, inventory, capacity, economics, and random tapes are "
        "unchanged from V26.\n"
        "- Extension requests freeze and disclose their deterministic contract "
        "delta before decision; acceptance returns and records the exact per-line "
        "rent and immediate cash collected.\n"
        "- Lines not fulfilled at pickup, including credited optional lines, have "
        "zero extension rent and are not extended.\n"
        "- Commitment history separates pickup and post-pickup rent receipts and "
        "computes gross/net rent cash without persisting redundant totals.\n"
        "- New summaries omit the ambiguous max_days field and retain explicit "
        "decision_horizon_day and runoff_end_day.\n",
        encoding="utf-8",
    )
    mechanism = dict(metadata.get("reservation_mechanism") or {})
    mechanism.update(
        {
            "extension_rent_delta": "frozen_at_request_creation_v1",
            "extension_rent_collection_timing": "immediate_on_acceptance",
            "extension_line_rent_deltas_public": True,
            "extension_billing_scope": "fulfilled_lines_only",
            "reservation_rent_cash_history": (
                "gross_components_plus_computed_gross_and_net_v1"
            ),
            "summary_horizon_fields": [
                "decision_horizon_day",
                "runoff_end_day",
            ],
            "summary_max_days_omitted": True,
        }
    )
    metadata.update(
        {
            "environment_protocol": ENVIRONMENT_PROTOCOL,
            "tool_interface_version": TOOL_INTERFACE_VERSION,
            "release_status": "v27-extension-cash-audit-candidate",
            "extension_cash_audit_runtime_builder": BUILDER_FORMAT,
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
    result = build_extension_cash_audit_runtime(
        args.source_runtime, args.output_dir
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
