"""Project a frozen V20 plan-set runtime into the V21 feedback runtime."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .data import V13_RUNTIME_FORMAT
from .reservation_feedback import (
    PLAN_SET_FEEDBACK_FORMAT,
    PLAN_SET_FEEDBACK_FORMAT_V2,
    PlanSetFeedbackPolicy,
)


BUILDER_FORMAT = "outfitter-noisy-line-mention-runtime-builder-v1"
SOURCE_PROTOCOL = "outfitter-seasonal-v20-factorized-plan-set"
ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v21-noisy-line-mention"
TOOL_INTERFACE_VERSION = 17
CONTRACT_MODE = "factorized_line_options_plan_set_v1"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_noisy_line_mention_runtime(
    source_runtime: str | Path,
    output_dir: str | Path,
    *,
    line_mention_probability: float = 0.40,
    line_mention_content_fidelity: float = 0.65,
    line_mention_contrast_sharpness: float = 3.0,
    line_mention_salience_scale: float = 0.12927675864947474,
) -> dict[str, Any]:
    """Copy V20 and add only a versioned, low-frequency feedback field."""

    source = Path(source_runtime).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format") != V13_RUNTIME_FORMAT:
        raise ValueError("V21 projection requires a V20 V13-format runtime")
    if metadata.get("environment_protocol") != SOURCE_PROTOCOL:
        raise ValueError("V21 projection requires the frozen V20 protocol")
    if int(metadata.get("tool_interface_version", 0)) != 16:
        raise ValueError("V21 projection requires V20 tool interface 16")
    profile_metadata = dict(metadata.get("reservation_promise_profiles") or {})
    if profile_metadata.get("contract_mode") != CONTRACT_MODE:
        raise ValueError("V21 projection requires the factorized plan-set mode")
    source_policy = PlanSetFeedbackPolicy.from_dict(
        json.loads(
            (source / "reservation_plan_set_feedback.json").read_text(
                encoding="utf-8"
            )
        )
    )
    if source_policy.feedback_format != PLAN_SET_FEEDBACK_FORMAT:
        raise ValueError("V21 projection requires the frozen V20 feedback v1 policy")
    policy = replace(
        source_policy,
        feedback_format=PLAN_SET_FEEDBACK_FORMAT_V2,
        line_mention_probability=float(line_mention_probability),
        line_mention_content_fidelity=float(line_mention_content_fidelity),
        line_mention_contrast_sharpness=float(
            line_mention_contrast_sharpness
        ),
        line_mention_salience_scale=float(line_mention_salience_scale),
    )
    policy.validate()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty or absent")
    output.mkdir(parents=True, exist_ok=True)
    for source_file in sorted(source.iterdir()):
        if source_file.is_file() and source_file.name not in {
            "metadata.json",
            "report.md",
            "reservation_plan_set_feedback.json",
        }:
            shutil.copy2(source_file, output / source_file.name)

    feedback_payload = policy.to_dict()
    _write_json(output / "reservation_plan_set_feedback.json", feedback_payload)
    (output / "report.md").write_text(
        "# V21 noisy customer line-mention runtime\n\n"
        f"- Source: `{source}`\n"
        f"- Environment protocol: `{ENVIRONMENT_PROTOCOL}`\n"
        f"- Tool interface: {TOOL_INTERFACE_VERSION}\n"
        f"- Contract mode: `{CONTRACT_MODE}` (unchanged from V20).\n"
        "- Scope: only an empty authorized plan set with available "
        "equipment-fit feedback can mention at most one request-local line.\n"
        "- Public field: `customer_mentioned_line_id`. It is a subjective, "
        "noisy, non-exhaustive customer comment, never a certified failed line.\n"
        f"- Mention probability: {policy.line_mention_probability:.2f}.\n"
        f"- Content fidelity: {policy.line_mention_content_fidelity:.2f}.\n"
        f"- Contrast sharpness: {policy.line_mention_contrast_sharpness:.1f}.\n"
        f"- Frozen salience scale: {policy.line_mention_salience_scale:.17g}.\n"
        "- Excluded inputs: price, deposit, outside resolution, inventory, "
        "commitments, capacity, supplier cost, cash, profit, and one-off "
        "Profile shock.\n",
        encoding="utf-8",
    )

    profile_metadata.update(
        {
            "customer_rejection_feedback": (
                "empty_set_noisy_request_concern_with_optional_subjective_line_mention"
            ),
            "feedback": "reservation_plan_set_feedback.json",
        }
    )
    mechanism = dict(metadata.get("reservation_mechanism") or {})
    mechanism.update(
        {
            "negative_feedback_scope": "empty_authorized_set_only",
            "equipment_feedback_line_mention": (
                "optional_single_subjective_request_local_line"
            ),
            "line_mention_is_failed_line_truth": False,
            "line_mention_guarantees_repair_or_transaction": False,
            "line_mention_reads_inventory_or_economics": False,
        }
    )
    plan_feedback = dict(metadata.get("plan_set_feedback_model") or {})
    plan_feedback.update(
        {
            **feedback_payload,
            "file": "reservation_plan_set_feedback.json",
            "scope": "empty_customer_authorized_plan_set_only",
            "customer_mentioned_line_id_scope": (
                "available equipment_fit_concern only; zero or one line"
            ),
            "claim_boundary": (
                "a noisy subjective customer mention, not a failed-line truth, "
                "specific repair, threshold, or transaction guarantee"
            ),
        }
    )
    metadata.update(
        {
            "environment_protocol": ENVIRONMENT_PROTOCOL,
            "tool_interface_version": TOOL_INTERFACE_VERSION,
            "release_status": "v21-noisy-line-mention-candidate",
            "noisy_line_mention_runtime_builder": BUILDER_FORMAT,
            "source_runtime": str(source),
            "reservation_promise_profiles": profile_metadata,
            "reservation_mechanism": mechanism,
            "plan_set_feedback_model": plan_feedback,
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
    parser.add_argument(
        "--line-mention-probability", type=float, default=0.40
    )
    parser.add_argument(
        "--line-mention-content-fidelity", type=float, default=0.65
    )
    parser.add_argument(
        "--line-mention-contrast-sharpness", type=float, default=3.0
    )
    parser.add_argument(
        "--line-mention-salience-scale",
        type=float,
        default=0.12927675864947474,
    )
    args = parser.parse_args(argv)
    result = build_noisy_line_mention_runtime(
        args.source_runtime,
        args.output_dir,
        line_mention_probability=args.line_mention_probability,
        line_mention_content_fidelity=args.line_mention_content_fidelity,
        line_mention_contrast_sharpness=(
            args.line_mention_contrast_sharpness
        ),
        line_mention_salience_scale=args.line_mention_salience_scale,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
