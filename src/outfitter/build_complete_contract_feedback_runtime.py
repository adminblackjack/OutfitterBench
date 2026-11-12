"""Project the frozen V18 runtime into a V19 complete-contract runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .channel_demand import REQUEST_BUNDLE_CHANNEL_DEMAND_FORMAT
from .data import V13_RUNTIME_FORMAT
from .reservation_feedback import (
    CONTRACT_FEEDBACK_FORMAT,
    ContractFeedbackPolicy,
)


BUILDER_FORMAT = "outfitter-complete-contract-feedback-runtime-builder-v1"
SOURCE_PROTOCOL = "outfitter-seasonal-v18-request-level-reservation-choice"
ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v19-complete-contract-feedback"
TOOL_INTERFACE_VERSION = 15
SOURCE_CONTRACT_MODE = "adaptive_pairwise_exact_quotes_v1"
CONTRACT_MODE = "complete_contract_options_v1"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_complete_contract_feedback_runtime(
    source_runtime: str | Path,
    output_dir: str | Path,
    *,
    response_probability: float,
    price_discount_fraction: float,
    reporting_fidelity: float,
    price_gain_scale: float = 1.0,
    configuration_gain_scale: float = 1.0,
) -> dict[str, Any]:
    source = Path(source_runtime).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    metadata = json.loads(
        (source / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("format") != V13_RUNTIME_FORMAT:
        raise ValueError("V19 projection requires a V18 V13-format runtime")
    if metadata.get("environment_protocol") != SOURCE_PROTOCOL:
        raise ValueError("V19 projection requires the frozen V18 protocol")
    if int(metadata.get("tool_interface_version", 0)) != 14:
        raise ValueError("V19 projection requires V18 tool interface 14")
    profile_metadata = dict(metadata.get("reservation_promise_profiles") or {})
    if profile_metadata.get("contract_mode") != SOURCE_CONTRACT_MODE:
        raise ValueError("V19 projection requires the frozen V18 contract mode")
    channel = json.loads(
        (source / "channel_demand_model.json").read_text(encoding="utf-8")
    )
    if channel.get("format") != REQUEST_BUNDLE_CHANNEL_DEMAND_FORMAT:
        raise ValueError("V19 projection requires channel-demand v5")
    policy = ContractFeedbackPolicy(
        response_probability=float(response_probability),
        price_discount_fraction=float(price_discount_fraction),
        reporting_fidelity=float(reporting_fidelity),
        price_gain_scale=float(price_gain_scale),
        configuration_gain_scale=float(configuration_gain_scale),
    )
    policy.validate()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty or absent")
    output.mkdir(parents=True, exist_ok=True)
    for source_file in sorted(source.iterdir()):
        if source_file.is_file() and source_file.name not in {
            "metadata.json",
            "report.md",
            "reservation_contract_feedback.json",
        }:
            shutil.copy2(source_file, output / source_file.name)
    feedback_payload = policy.to_dict()
    if feedback_payload["format"] != CONTRACT_FEEDBACK_FORMAT:
        raise AssertionError("feedback policy format mismatch")
    _write_json(output / "reservation_contract_feedback.json", feedback_payload)
    (output / "report.md").write_text(
        "# V19 explicit complete-contract feedback runtime\n\n"
        f"- Source: `{source}`\n"
        f"- Environment protocol: `{ENVIRONMENT_PROTOCOL}`\n"
        f"- Tool interface: {TOOL_INTERFACE_VERSION}\n"
        f"- Contract mode: `{CONTRACT_MODE}`\n"
        "- Store choice: one or two explicitly submitted complete contracts; "
        "no implicit Cartesian product.\n"
        "- Optional lines: explicitly quoted or omitted in each contract.\n"
        "- Feedback: one request-shared response-availability event; every "
        "unselected contract receives a coarse assessment with independent, "
        "stable content noise.\n"
        f"- Action-gain scales: price={policy.price_gain_scale:.9g}, "
        f"configuration={policy.configuration_gain_scale:.9g}; these are "
        "same-tape calibration constants, not public customer thresholds.\n"
        "- Outside resolution, customer utility, thresholds, failed lines, and "
        "deterministic counterfactuals remain private.\n",
        encoding="utf-8",
    )
    profile_metadata.update(
        {
            "contract_mode": CONTRACT_MODE,
            "complete_contract_option_count": [1, 2],
            "line_option_count": None,
            "optional_line_disposition": ["quote", "omit"],
            "implicit_cartesian_product": False,
            "customer_rejection_feedback": "noisy_per_unselected_contract",
            "feedback": "reservation_contract_feedback.json",
        }
    )
    mechanism = dict(metadata.get("reservation_mechanism") or {})
    mechanism.update(
        {
            "customer_choice": (
                "one or two explicit complete store contracts versus one "
                "complete outside plan versus abandonment"
            ),
            "quote_unit": "complete_contract_options_with_explicit_line_terms",
            "quote_line_completeness": (
                "every contract resolves every required and optional line"
            ),
            "optional_quote_acceptance": "explicit_quote_or_omit_per_contract",
            "implicit_cartesian_product": False,
            "risk_recovery": "selected_full_locked_contract_replay",
            "reported_assessment_is_true_reason": False,
            "outside_resolution_disclosed": False,
        }
    )
    metadata.update(
        {
            "environment_protocol": ENVIRONMENT_PROTOCOL,
            "tool_interface_version": TOOL_INTERFACE_VERSION,
            "release_status": "v19-complete-contract-feedback-candidate",
            "complete_contract_feedback_runtime_builder": BUILDER_FORMAT,
            "source_runtime": str(source),
            "reservation_promise_profiles": profile_metadata,
            "reservation_mechanism": mechanism,
            "contract_feedback_model": {
                **feedback_payload,
                "file": "reservation_contract_feedback.json",
                "claim_boundary": (
                    "synthetic benchmark information budget; not observed "
                    "customer survey frequency or ground-truth reasons"
                ),
            },
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
        "feedback_policy": feedback_payload,
        "metadata_sha256": _sha256(output / "metadata.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--response-probability", type=float, required=True)
    parser.add_argument("--price-discount-fraction", type=float, required=True)
    parser.add_argument("--reporting-fidelity", type=float, required=True)
    parser.add_argument("--price-gain-scale", type=float, default=1.0)
    parser.add_argument("--configuration-gain-scale", type=float, default=1.0)
    args = parser.parse_args(argv)
    result = build_complete_contract_feedback_runtime(
        args.source_runtime,
        args.output_dir,
        response_probability=args.response_probability,
        price_discount_fraction=args.price_discount_fraction,
        reporting_fidelity=args.reporting_fidelity,
        price_gain_scale=args.price_gain_scale,
        configuration_gain_scale=args.configuration_gain_scale,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
