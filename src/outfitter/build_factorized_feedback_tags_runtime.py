"""Project the frozen V22 runtime into V23 independent concern tags."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .data import V13_RUNTIME_FORMAT
from .reservation_feedback import (
    PLAN_SET_FEEDBACK_FORMAT_V2,
    PLAN_SET_FEEDBACK_FORMAT_V3,
    PlanSetFeedbackPolicy,
)


BUILDER_FORMAT = "outfitter-factorized-feedback-tags-runtime-builder-v1"
SOURCE_PROTOCOL = "outfitter-seasonal-v22-capacity-surface"
ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v23-factorized-feedback-tags"
TOOL_INTERFACE_VERSION = 19
CONTRACT_MODE = "factorized_line_options_plan_set_v1"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_factorized_feedback_tags_runtime(
    source_runtime: str | Path,
    output_dir: str | Path,
    *,
    response_probability: float = 0.60,
    price_action_scale: float = 2.398877906212523,
    equipment_action_scale: float = 1.9265646974835349,
    tag_probability_floor: float = 0.10,
    tag_probability_ceiling: float = 0.85,
    tag_relevance_threshold_ratio: float = 1.25,
    tag_sharpness: float = 8.0,
    line_mention_probability: float = 0.75,
    line_mention_content_fidelity: float = 0.50,
    line_mention_contrast_sharpness: float = 3.0,
    line_mention_salience_scale: float = 0.9681183032966039,
) -> dict[str, Any]:
    source = Path(source_runtime).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format") != V13_RUNTIME_FORMAT:
        raise ValueError("V23 projection requires a V22 V13-format runtime")
    if metadata.get("environment_protocol") != SOURCE_PROTOCOL:
        raise ValueError("V23 projection requires the frozen V22 protocol")
    if int(metadata.get("tool_interface_version", 0)) != 18:
        raise ValueError("V23 projection requires V22 tool interface 18")
    profiles = dict(metadata.get("reservation_promise_profiles") or {})
    if profiles.get("contract_mode") != CONTRACT_MODE:
        raise ValueError("V23 projection requires factorized plan-set contracts")
    source_policy = PlanSetFeedbackPolicy.from_dict(
        json.loads(
            (source / "reservation_plan_set_feedback.json").read_text(
                encoding="utf-8"
            )
        )
    )
    if source_policy.feedback_format != PLAN_SET_FEEDBACK_FORMAT_V2:
        raise ValueError("V23 projection requires the frozen V22 v2 policy")
    policy = PlanSetFeedbackPolicy(
        response_probability=float(response_probability),
        contrast_sharpness=None,
        mixed_signal_weight=None,
        feedback_format=PLAN_SET_FEEDBACK_FORMAT_V3,
        price_curve_discount_fractions=(0.10, 0.20, 0.30),
        price_action_scale=float(price_action_scale),
        equipment_action_scale=float(equipment_action_scale),
        equipment_action_top_k=2,
        tag_probability_floor=float(tag_probability_floor),
        tag_probability_ceiling=float(tag_probability_ceiling),
        tag_relevance_threshold_ratio=float(tag_relevance_threshold_ratio),
        tag_sharpness=float(tag_sharpness),
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

    payload = policy.to_dict()
    _write_json(output / "reservation_plan_set_feedback.json", payload)
    (output / "report.md").write_text(
        "# V23 factorized customer concern-tag runtime\n\n"
        f"- Source: `{source}`\n"
        f"- Environment protocol: `{ENVIRONMENT_PROTOCOL}`.\n"
        f"- Tool interface: {TOOL_INTERFACE_VERSION}.\n"
        "- Customer choice, authorization, capacity, economics, and request "
        "tapes are unchanged from V22.\n"
        "- Empty authorized sets may report independent `price_value` and "
        "`equipment_configuration` concerns together; `other_or_unclear` is "
        "an exclusive residual, not a third action axis.\n"
        "- Price evidence averages complete 10/20/30% repricing responses; "
        "equipment evidence averages the top two public line replacements "
        "with approximate line rent and deposit frozen.\n"
        "- A customer line mention is possible only with the equipment tag and "
        "remains subjective, noisy, and non-exhaustive.\n"
        "- Feedback never reads inventory, commitments, deliveries, capacity, "
        "supplier cost, cash, profit, or the realized outside resolution.\n",
        encoding="utf-8",
    )

    profiles.update(
        {
            "customer_rejection_feedback": (
                "empty_set_independent_concern_tags_with_subjective_line_comment"
            ),
            "feedback": "reservation_plan_set_feedback.json",
        }
    )
    mechanism = dict(metadata.get("reservation_mechanism") or {})
    mechanism.update(
        {
            "negative_feedback_scope": "empty_authorized_set_only",
            "reported_concern_tags": [
                "price_value",
                "equipment_configuration",
            ],
            "reported_concern_tags_can_cooccur": True,
            "residual_concern": "other_or_unclear",
            "residual_concern_can_cooccur": False,
            "equipment_feedback_line_mention": (
                "optional_single_subjective_request_local_line"
            ),
            "line_mention_is_failed_line_truth": False,
            "line_mention_guarantees_repair_or_transaction": False,
            "feedback_reads_inventory_or_economics": False,
        }
    )
    feedback_model = {
        **payload,
        "file": "reservation_plan_set_feedback.json",
        "scope": "empty_customer_authorized_plan_set_only",
        "price_evidence": "mean_complete_repricing_gain_10_20_30pct",
        "equipment_evidence": (
            "top2_mean_public_line_replacement_gain_approx_fixed_rent_deposit"
        ),
        "claim_boundary": (
            "noisy coarse reported concerns, not true reasons, thresholds, best "
            "actions, transaction guarantees, or manager advice"
        ),
    }
    metadata.update(
        {
            "environment_protocol": ENVIRONMENT_PROTOCOL,
            "tool_interface_version": TOOL_INTERFACE_VERSION,
            "release_status": "v23-factorized-feedback-tags-candidate",
            "factorized_feedback_tags_runtime_builder": BUILDER_FORMAT,
            "source_runtime": str(source),
            "reservation_promise_profiles": profiles,
            "reservation_mechanism": mechanism,
            "plan_set_feedback_model": feedback_model,
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
        "feedback_policy": payload,
        "metadata_sha256": _sha256(output / "metadata.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = build_factorized_feedback_tags_runtime(
        args.source_runtime, args.output_dir
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
