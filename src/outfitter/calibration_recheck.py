"""Recompute derived calibration gates without rerunning deterministic episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .calibrate import scripted_economic_gate


RECHECK_FORMAT = "outfitter-calibration-derived-gate-recheck-v1"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def recheck_calibration_payload(
    payload: dict[str, Any],
    *,
    source_result_sha256: str,
) -> dict[str, Any]:
    runs = list(payload["runs"])
    source_gate = dict(payload.get("scripted_profit_gate") or {})
    required_seed_count = source_gate.get("required_seed_count")
    result = dict(payload)
    result["scripted_profit_gate"] = scripted_economic_gate(
        runs,
        required_seed_count=(
            None
            if required_seed_count is None
            else int(required_seed_count)
        ),
    )
    result["scripted_profit_gate_recheck_provenance"] = {
        "format": RECHECK_FORMAT,
        "source_result_sha256": source_result_sha256,
        "source_stored_gate_format": source_gate.get("format"),
        "source_stored_gate_passed": source_gate.get("passed"),
        "runs_canonical_sha256": _canonical_sha256(runs),
        "runs_unchanged": True,
        "scope": (
            "derived gate fields only; no episode row, business state, random "
            "outcome, or score was rerun or modified"
        ),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.input.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    result = recheck_calibration_payload(
        payload,
        source_result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "format": RECHECK_FORMAT,
                "output": str(output),
                "passed": result["scripted_profit_gate"]["passed"],
                "runs": len(result["runs"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
