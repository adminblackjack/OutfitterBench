"""Project a frozen V15 runtime into a V16 promise-profile prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .data import OutfitterDataset, V12_RUNTIME_FORMAT, V13_RUNTIME_FORMAT
from .reservation_profiles import (
    PROFILE_CATALOG_FORMAT,
    PROFILE_SCHEMA_VERSION,
    ReservationPromiseCatalog,
    build_reservation_promise_catalog,
    catalog_semantics_sha256,
)


BUILDER_FORMAT = "outfitter-reservation-promise-runtime-builder-v1"
ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v16-reservation-promise-profiles"
TOOL_INTERFACE_VERSION = 11


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_previous(value: str | Path | None) -> ReservationPromiseCatalog | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "reservation_promise_profiles.json"
    return ReservationPromiseCatalog.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


def build_reservation_profile_runtime(
    source_runtime: str | Path,
    output_dir: str | Path,
    *,
    previous_profile_catalog: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(source_runtime).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    source_metadata = json.loads(
        (source / "metadata.json").read_text(encoding="utf-8")
    )
    if source_metadata.get("format") != V12_RUNTIME_FORMAT:
        raise ValueError("V16 profile projection requires a frozen V15 lifecycle runtime")
    if source_metadata.get("environment_protocol") != (
        "outfitter-seasonal-v15-lifecycle-economic-balance"
    ):
        raise ValueError("V16 profile projection requires the V15 balanced protocol")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty or absent")
    output.mkdir(parents=True, exist_ok=True)
    for source_file in sorted(source.iterdir()):
        if source_file.is_file() and source_file.name != "metadata.json":
            shutil.copy2(source_file, output / source_file.name)

    dataset = OutfitterDataset.load(source)
    source_catalog_sha256 = _sha256(source / "catalog.jsonl")
    profiles = build_reservation_promise_catalog(
        dataset.offers,
        source_catalog_sha256=source_catalog_sha256,
        previous=_load_previous(previous_profile_catalog),
    )
    profile_path = output / "reservation_promise_profiles.json"
    _write_json(profile_path, profiles.to_dict())

    metadata = dict(source_metadata)
    metadata.update(
        {
            "format": V13_RUNTIME_FORMAT,
            "schema_version": "outfitter-decision-schema-v13",
            "environment_protocol": ENVIRONMENT_PROTOCOL,
            "tool_interface_version": TOOL_INTERFACE_VERSION,
            "release_status": "v16-reservation-promise-profile-prototype",
            "reservation_profile_runtime_builder": BUILDER_FORMAT,
            "source_runtime": str(source),
            "reservation_promise_profiles": {
                "enabled_config_required": True,
                "file": profile_path.name,
                "format": PROFILE_CATALOG_FORMAT,
                "schema_version": PROFILE_SCHEMA_VERSION,
                "schema_semantics_sha256": catalog_semantics_sha256(profiles),
                "profile_count": len(profiles.profiles),
                "active_profile_count": sum(
                    bool(row.member_offer_ids) for row in profiles.profiles
                ),
                "durable_offer_coverage": len(
                    {
                        offer_id
                        for row in profiles.profiles
                        for offer_id in row.member_offer_ids
                    }
                ),
                "contract_mode": "promise_profiles_v1",
                "line_mixing": False,
                "feedback": "none",
                "claim_boundary": (
                    "benchmark contract representation; not an observed market taxonomy"
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

    result = {
        "format": BUILDER_FORMAT,
        "source_runtime": str(source),
        "output_runtime": str(output),
        "environment_protocol": ENVIRONMENT_PROTOCOL,
        "tool_interface_version": TOOL_INTERFACE_VERSION,
        "profile_count": len(profiles.profiles),
        "active_profile_count": sum(
            bool(row.member_offer_ids) for row in profiles.profiles
        ),
        "profile_catalog_sha256": _sha256(profile_path),
        "profile_semantics_sha256": catalog_semantics_sha256(profiles),
        "metadata_sha256": _sha256(output / "metadata.json"),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--previous-profile-catalog")
    args = parser.parse_args(argv)
    result = build_reservation_profile_runtime(
        args.source_runtime,
        args.output_dir,
        previous_profile_catalog=args.previous_profile_catalog,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

