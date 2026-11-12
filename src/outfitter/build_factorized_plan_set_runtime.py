"""Project the frozen V19 runtime into a V20 factorized-plan-set runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .channel_demand import REQUEST_BUNDLE_CHANNEL_DEMAND_FORMAT
from .data import V13_RUNTIME_FORMAT
from .reservation_feedback import PlanSetFeedbackPolicy


BUILDER_FORMAT = "outfitter-factorized-plan-set-runtime-builder-v1"
SOURCE_PROTOCOL = "outfitter-seasonal-v19-complete-contract-feedback"
ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v20-factorized-plan-set"
TOOL_INTERFACE_VERSION = 16
SOURCE_CONTRACT_MODE = "complete_contract_options_v1"
CONTRACT_MODE = "factorized_line_options_plan_set_v1"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_factorized_plan_set_runtime(
    source_runtime: str | Path,
    output_dir: str | Path,
    *,
    response_probability: float,
    reporting_fidelity: float,
    price_burden_scale: float,
    equipment_fit_regret_scale: float,
    contrast_sharpness: float = 3.0,
) -> dict[str, Any]:
    source = Path(source_runtime).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format") != V13_RUNTIME_FORMAT:
        raise ValueError("V20 projection requires a V19 V13-format runtime")
    if metadata.get("environment_protocol") != SOURCE_PROTOCOL:
        raise ValueError("V20 projection requires the frozen V19 protocol")
    if int(metadata.get("tool_interface_version", 0)) != 15:
        raise ValueError("V20 projection requires V19 tool interface 15")
    profile_metadata = dict(metadata.get("reservation_promise_profiles") or {})
    if profile_metadata.get("contract_mode") != SOURCE_CONTRACT_MODE:
        raise ValueError("V20 projection requires the frozen V19 contract mode")
    channel = json.loads(
        (source / "channel_demand_model.json").read_text(encoding="utf-8")
    )
    if channel.get("format") != REQUEST_BUNDLE_CHANNEL_DEMAND_FORMAT:
        raise ValueError("V20 projection requires channel-demand v5")
    policy = PlanSetFeedbackPolicy(
        response_probability=float(response_probability),
        reporting_fidelity=float(reporting_fidelity),
        price_burden_scale=float(price_burden_scale),
        equipment_fit_regret_scale=float(equipment_fit_regret_scale),
        contrast_sharpness=float(contrast_sharpness),
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
            "reservation_plan_set_feedback.json",
        }:
            shutil.copy2(source_file, output / source_file.name)
    feedback_payload = policy.to_dict()
    _write_json(output / "reservation_plan_set_feedback.json", feedback_payload)
    (output / "report.md").write_text(
        "# V20 factorized line-option customer plan-set runtime\n\n"
        f"- Source: `{source}`\n"
        f"- Environment protocol: `{ENVIRONMENT_PROTOCOL}`\n"
        f"- Tool interface: {TOOL_INTERFACE_VERSION}\n"
        f"- Contract mode: `{CONTRACT_MODE}`\n"
        "- Required lines: one or two exact/Profile/price options.\n"
        "- Optional lines: zero to two quote options plus one explicit omit.\n"
        "- Private expansion: at most 24 complete plans. Each is evaluated "
        "independently against one frozen outside plan and abandonment.\n"
        "- Disclosure: the highest-ranked at most three customer-authorized "
        "plans; the set is intentionally non-exhaustive.\n"
        "- Confirmation: one plan confirms immediately; multiple plans await "
        "manager selection and default to the customer preferred plan at close.\n"
        "- Feedback: only an empty authorized set may receive one noisy "
        "request-level price/equipment/multiple concern.\n"
        "- The feedback scales are synthetic information-budget constants, not "
        "observed survey frequencies, WTP, true reasons, or action guarantees.\n",
        encoding="utf-8",
    )
    profile_metadata.update(
        {
            "contract_mode": CONTRACT_MODE,
            "complete_contract_option_count": None,
            "required_line_option_count": [1, 2],
            "optional_line_quote_option_count": [0, 2],
            "optional_line_disposition": ["quote", "omit"],
            "maximum_internal_plan_count": 24,
            "maximum_returned_authorized_plan_count": 3,
            "authorization_is_exhaustive": False,
            "implicit_cartesian_product": "private_bounded_expansion",
            "customer_rejection_feedback": "empty_set_noisy_request_concern",
            "feedback": "reservation_plan_set_feedback.json",
        }
    )
    mechanism = dict(metadata.get("reservation_mechanism") or {})
    mechanism.update(
        {
            "customer_choice": (
                "each internally formed complete store plan independently versus "
                "one frozen outside plan and abandonment"
            ),
            "quote_unit": "factorized_line_exact_profile_price_options",
            "quote_line_completeness": (
                "required one-or-two quote options; optional zero-to-two quote "
                "options plus exactly one omit"
            ),
            "maximum_internal_plan_count": 24,
            "maximum_returned_authorized_plan_count": 3,
            "authorization_is_exhaustive": False,
            "confirmation": (
                "single auto; multiple manager choice; day-end customer-preferred auto"
            ),
            "cash_and_calendar_timing": "confirmed_plan_only",
            "unconfirmed_plan_fallback": False,
            "risk_recovery": "manager_confirmed_full_locked_plan_replay",
            "reported_assessment_is_true_reason": False,
            "negative_feedback_scope": "empty_authorized_set_only",
            "outside_resolution_disclosed": False,
        }
    )
    metadata.pop("contract_feedback_model", None)
    metadata.update(
        {
            "environment_protocol": ENVIRONMENT_PROTOCOL,
            "tool_interface_version": TOOL_INTERFACE_VERSION,
            "release_status": "v20-factorized-plan-set-candidate",
            "factorized_plan_set_runtime_builder": BUILDER_FORMAT,
            "source_runtime": str(source),
            "reservation_promise_profiles": profile_metadata,
            "reservation_mechanism": mechanism,
            "plan_set_feedback_model": {
                **feedback_payload,
                "file": "reservation_plan_set_feedback.json",
                "public_categories": [
                    "price_value_concern",
                    "equipment_fit_concern",
                    "multiple_or_unclear_concerns",
                ],
                "scope": "empty_customer_authorized_plan_set_only",
                "claim_boundary": (
                    "noisy subjective concern with held-out causal validation; "
                    "not observed truth, a threshold, or a guaranteed intervention"
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
    parser.add_argument("--reporting-fidelity", type=float, required=True)
    parser.add_argument("--price-burden-scale", type=float, required=True)
    parser.add_argument("--equipment-fit-regret-scale", type=float, required=True)
    parser.add_argument("--contrast-sharpness", type=float, default=3.0)
    args = parser.parse_args(argv)
    result = build_factorized_plan_set_runtime(
        args.source_runtime,
        args.output_dir,
        response_probability=args.response_probability,
        reporting_fidelity=args.reporting_fidelity,
        price_burden_scale=args.price_burden_scale,
        equipment_fit_regret_scale=args.equipment_fit_regret_scale,
        contrast_sharpness=args.contrast_sharpness,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
