"""Build independent V53 data from V52; never rewrite existing runs or input data."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import yaml

from .config import OutfitterConfig
from .data import OutfitterDataset
from .environment import OutfitterEnvironment
from .models import Offer
from .operating_history import ENVIRONMENT_PROTOCOL, CORE_RELEASE_ID, TOOL_INTERFACE_VERSION, POLICY_ID
from .resale_disclosure import ENVIRONMENT_PROTOCOL as SOURCE_PROTOCOL, CORE_RELEASE_ID as SOURCE_CORE
from .reservation_profiles import ReservationPromiseCatalog, _reference_exact_deposit, catalog_semantics_sha256
from .visibility import write_product_visibility_exports

PRICES = {"offer-021": 500.0, "offer-050": 350.0}


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source, output, config):
    source, output = Path(source).resolve(), Path(output).resolve()
    data = OutfitterDataset.load(source)
    if data.metadata.get("environment_protocol") != SOURCE_PROTOCOL or data.metadata.get("core_release_id") != SOURCE_CORE:
        raise ValueError("V53 requires the V52 core-r1 source")
    if output.exists() or source in output.parents or output in source.parents:
        raise ValueError("Output must be a new independent directory")
    shutil.copytree(source, output)
    catalog = [json.loads(line) for line in (source / "catalog.jsonl").read_text().splitlines() if line.strip()]
    revisions = []
    for row in catalog:
        if row["offer_id"] not in PRICES:
            continue
        price = PRICES[row["offer_id"]]
        online = row["consumer_online_offer"]
        revisions.append({"offer_id": row["offer_id"], "old_unit_cost": row["unit_cost"],
                          "old_external_offer": copy.deepcopy(online), "new_unit_price": price})
        row["unit_cost"] = price
        row["package_price"] = round(price * row["package_units"], 2)
        row["reference_daily_rate"] = round(price / max(1.0, row["service_life_limit_days"] * .35) * 1.35, 4)
        online.update(unit_price_cny=price, package_price_cny=row["package_price"],
                      provenance="benchmark_product_fact_revision_v53")
        online.pop("verified_at", None)
    (output / "catalog.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in catalog), encoding="utf-8")
    catalog_hash = _hash(output / "catalog.jsonl")
    offers = {row["offer_id"]: Offer.from_dict(row) for row in catalog}
    profiles = copy.deepcopy(data.reservation_promise_catalog.to_dict())
    profiles["source_catalog_sha256"] = catalog_hash
    for profile in profiles["profiles"]:
        members = profile["member_offer_ids"]
        if set(members) & PRICES.keys():
            cheapest = min((offers[key] for key in members), key=lambda o: (o.unit_cost, o.offer_id))
            profile["unit_deposit_cny_by_policy"]["canonical_reference_v1"] = _reference_exact_deposit(cheapest)
    _write(output / "reservation_promise_profiles.json", profiles)
    weights = json.loads((output / "warehouse_capacity_weights.json").read_text())
    weights["catalog_sha256"] = catalog_hash
    _write(output / "warehouse_capacity_weights.json", weights)
    metadata = copy.deepcopy(data.metadata)
    metadata.update(environment_protocol=ENVIRONMENT_PROTOCOL, core_release_id=CORE_RELEASE_ID,
                    tool_interface_version=TOOL_INTERFACE_VERSION, operating_history_policy=POLICY_ID,
                    package_version="0.53.0", source_runtime=SOURCE_PROTOCOL,
                    release_status="v53-operating-history", benchmark_price_revisions=revisions)
    metadata["reservation_promise_profiles"]["schema_semantics_sha256"] = catalog_semantics_sha256(ReservationPromiseCatalog.from_dict(profiles))

    def write_metadata():
        metadata["content_sha256"] = {p.name: _hash(p) for p in sorted(output.iterdir())
                                     if p.is_file() and p.name != "metadata.json"}
        _write(output / "metadata.json", metadata)

    # Internal audit, not a claim of new profitability/market calibration.
    _write(output / "result.json", {"environment_protocol": ENVIRONMENT_PROTOCOL,
            "source_runtime": SOURCE_PROTOCOL, "price_revisions": PRICES,
            "profitability_validation_performed": False})
    write_metadata()
    # Export the complete catalog, including the optional retail track.
    cfg = replace(config, data_dir=str(output), assortment_mode="rental_and_retail")
    env = OutfitterEnvironment(cfg, OutfitterDataset.load(output))
    write_product_visibility_exports(env, output)
    write_metadata()
    OutfitterDataset.load(output)
    return {"runtime": str(output), "environment_protocol": ENVIRONMENT_PROTOCOL,
            "core_release_id": CORE_RELEASE_ID, "price_revisions": PRICES}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    configuration = yaml.safe_load(Path(args.config).read_text())
    configuration["data_dir"] = str(Path(args.source).resolve())
    print(json.dumps(build(args.source, args.output, OutfitterConfig.from_dict(configuration))))
