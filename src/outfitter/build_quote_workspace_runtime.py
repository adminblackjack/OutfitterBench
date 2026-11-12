"""Project the frozen V24 runtime into V25 request-level quote workspaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .data import V13_RUNTIME_FORMAT


BUILDER_FORMAT = "outfitter-quote-workspace-runtime-builder-v1"
SOURCE_PROTOCOL = "outfitter-seasonal-v24-reservation-operations"
ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v25-request-quote-workspace"
TOOL_INTERFACE_VERSION = 21
CONTRACT_MODE = "factorized_line_options_plan_set_v1"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_quote_workspace_runtime(
    source_runtime: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source = Path(source_runtime).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format") != V13_RUNTIME_FORMAT:
        raise ValueError("V25 projection requires a V24 V13-format runtime")
    if metadata.get("environment_protocol") != SOURCE_PROTOCOL:
        raise ValueError("V25 projection requires the frozen V24 protocol")
    if int(metadata.get("tool_interface_version", 0)) != 20:
        raise ValueError("V25 projection requires V24 tool interface 20")
    profiles = dict(metadata.get("reservation_promise_profiles") or {})
    if profiles.get("contract_mode") != CONTRACT_MODE:
        raise ValueError("V25 projection requires factorized plan-set contracts")
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
        "# V25 request-level reservation quote workspace runtime\n\n"
        f"- Source: `{source}`\n"
        f"- Environment protocol: `{ENVIRONMENT_PROTOCOL}`.\n"
        f"- Tool interface: {TOOL_INTERFACE_VERSION}.\n"
        "- Customer choice, feedback, reservation generation, inventory, capacity, "
        "economics, and random tapes are byte-identical to V24.\n"
        "- Factorized inquiry option discovery is normalized by complete request into "
        "request-line Profile terms, deduplicated static Profile definitions, and "
        "deduplicated compact exact-offer definitions.\n"
        "- Dynamic prices, assets, contracts, orders, cleaning, maintenance, repairs, "
        "acceptance predictions, recommendation, and feasibility are excluded.\n"
        "- Frozen exact-member sets remain request-line-Profile scoped; all business "
        "actions continue to use stable global Profile and offer IDs.\n",
        encoding="utf-8",
    )
    mechanism = dict(metadata.get("reservation_mechanism") or {})
    mechanism.update(
        {
            "quote_option_surface": "normalized_request_level_workspace",
            "quote_workspace_pagination": "complete_request_keyset",
            "quote_workspace_default_offer_projection": "compact",
            "quote_workspace_dynamic_operations_included": False,
            "quote_workspace_recommendation_or_feasibility": False,
            "quote_workspace_frozen_membership_scope": (
                "request_line_promise_profile"
            ),
            "post_quote_fulfillment_surface_separate": True,
        }
    )
    profiles.update(
        {
            "quote_option_surface": "view_reservation_quote_workspace",
            "quote_workspace_offer_projections": [
                "compact",
                "references_only",
            ],
            "quote_workspace_request_page_limit": 5,
        }
    )
    metadata.update(
        {
            "environment_protocol": ENVIRONMENT_PROTOCOL,
            "tool_interface_version": TOOL_INTERFACE_VERSION,
            "release_status": "v25-request-quote-workspace-candidate",
            "quote_workspace_runtime_builder": BUILDER_FORMAT,
            "source_runtime": str(source),
            "reservation_promise_profiles": profiles,
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
    result = build_quote_workspace_runtime(args.source_runtime, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
