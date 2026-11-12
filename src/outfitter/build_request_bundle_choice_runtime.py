"""Project the frozen V17 runtime into the V18 request-bundle runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .channel_demand import (
    ChannelDemandModel,
    REQUEST_BUNDLE_CHANNEL_DEMAND_FORMAT,
    request_bundle_channel_model_payload,
)
from .data import V13_RUNTIME_FORMAT


BUILDER_FORMAT = "outfitter-request-bundle-choice-runtime-builder-v1"
SOURCE_PROTOCOL = "outfitter-seasonal-v17-searchable-quote-history"
ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v18-request-level-reservation-choice"
TOOL_INTERFACE_VERSION = 14
CONTRACT_MODE = "adaptive_pairwise_exact_quotes_v1"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_request_bundle_choice_runtime(
    source_runtime: str | Path,
    output_dir: str | Path,
    *,
    rental_request_cash_salience: float = 3.8,
) -> dict[str, Any]:
    """Copy V17 and replace only versioned customer-choice data/metadata."""

    source = Path(source_runtime).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    source_metadata = json.loads(
        (source / "metadata.json").read_text(encoding="utf-8")
    )
    if source_metadata.get("format") != V13_RUNTIME_FORMAT:
        raise ValueError("V18 projection requires a V17 runtime")
    if source_metadata.get("environment_protocol") != SOURCE_PROTOCOL:
        raise ValueError(
            "V18 projection requires the frozen V17 searchable-history protocol"
        )
    profile_metadata = dict(
        source_metadata.get("reservation_promise_profiles") or {}
    )
    if profile_metadata.get("contract_mode") != CONTRACT_MODE:
        raise ValueError(
            "V18 projection requires adaptive_pairwise_exact_quotes_v1"
        )
    source_channel_path = source / "channel_demand_model.json"
    source_channel = json.loads(source_channel_path.read_text(encoding="utf-8"))
    if source_channel.get("format") != "outfitter-exact-offer-channel-demand-v4":
        raise ValueError("V18 projection requires the frozen channel-demand v4")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty or absent")
    output.mkdir(parents=True, exist_ok=True)
    for source_file in sorted(source.iterdir()):
        if (
            source_file.is_file()
            and source_file.name
            not in {"metadata.json", "channel_demand_model.json"}
        ):
            shutil.copy2(source_file, output / source_file.name)

    channel_payload = request_bundle_channel_model_payload(
        ChannelDemandModel.from_dict(source_channel),
        rental_request_cash_salience=rental_request_cash_salience,
    )
    _write_json(output / "channel_demand_model.json", channel_payload)
    (output / "report.md").write_text(
        "# V18 request-level reservation choice runtime\n\n"
        f"- Source: `{source}`\n"
        f"- Environment protocol: `{ENVIRONMENT_PROTOCOL}`\n"
        f"- Tool interface: {TOOL_INTERFACE_VERSION}\n"
        f"- Channel-demand model: `{REQUEST_BUNDLE_CHANNEL_DEMAND_FORMAT}`\n"
        "- Contract mode: `adaptive_pairwise_exact_quotes_v1`\n"
        f"- Frozen request cash salience: {rental_request_cash_salience:.1f}\n"
        "- Customer choice: complete store bundle versus complete outside "
        "alternatives or abandonment; required lines are atomic and optional "
        "lines may be skipped.\n"
        "- Walk-in scope: lead-zero rental and retail channel choice remains "
        "the frozen V17 behavior.\n"
        "- Claim boundary: synthetic benchmark behavior; no observed market "
        "WTP, supplier cost, lifecycle floor, or payback target enters customer "
        "choice.\n",
        encoding="utf-8",
    )

    metadata = dict(source_metadata)
    channel_metadata = dict(metadata.get("channel_demand_model") or {})
    channel_metadata.update(
        {
            "format": REQUEST_BUNDLE_CHANNEL_DEMAND_FORMAT,
            "rental_customer_choice_scope": "request_bundle",
            "lead_zero_walk_in_behavior": "frozen_v17_v4",
            "request_budget": "mean_one_lognormal_shared_within_request",
            "quantity_value_scope": [
                "rental:camp_chair",
                "rental:sleeping_bag",
                "rental:sleeping_pad",
            ],
            "supplier_cost_enters_customer_demand": False,
        }
    )
    reservation_metadata = dict(metadata.get("reservation_mechanism") or {})
    reservation_metadata.update(
        {
            "customer_choice": (
                "one complete store bundle versus one complete outside plan "
                "versus abandonment"
            ),
            "required_quote_acceptance": "atomic_complete_bundle",
            "optional_quote_acceptance": "independent_skip_within_store_bundle",
            "request_level_terms": [
                "absolute contract cash",
                "aggregate deposit friction",
                "store service/date guarantee",
                "return burden",
                "outside delivery friction/risk",
            ],
            "inquiry_latent_price_ceiling": "not_generated",
            "risk_recovery": "full_locked_bundle_replay",
        }
    )
    metadata.update(
        {
            "environment_protocol": ENVIRONMENT_PROTOCOL,
            "tool_interface_version": TOOL_INTERFACE_VERSION,
            "release_status": "v18-request-level-choice-candidate",
            "request_bundle_choice_runtime_builder": BUILDER_FORMAT,
            "source_runtime": str(source),
            "channel_demand_model": channel_metadata,
            "reservation_mechanism": reservation_metadata,
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
        "channel_demand_format": REQUEST_BUNDLE_CHANNEL_DEMAND_FORMAT,
        "rental_request_cash_salience": rental_request_cash_salience,
        "metadata_sha256": _sha256(output / "metadata.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--rental-request-cash-salience", type=float, default=3.8
    )
    args = parser.parse_args(argv)
    result = build_request_bundle_choice_runtime(
        args.source_runtime,
        args.output_dir,
        rental_request_cash_salience=args.rental_request_cash_salience,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
