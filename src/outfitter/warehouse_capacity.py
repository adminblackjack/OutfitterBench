"""Frozen relative storage units, independent of prices and holding fees."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

WAREHOUSE_POLICY = "owned_and_inbound_return_reserved_v1"
WEIGHTS_FORMAT = "outfitter-warehouse-capacity-weights-v1"
ANCHORS = {"offer-020": 1, "offer-031": 2, "offer-073": 4,
           "offer-009": 6, "offer-050": 8, "offer-048": 10}


def positive_integer(value: Any) -> bool:
    return type(value) is int and value > 0


def validate_weights(document: dict[str, Any], catalog_path: Path,
                     rental_offer_ids: set[str], *, require_approved: bool = True) -> dict[str, int]:
    if document.get("format") != WEIGHTS_FORMAT:
        raise ValueError("unsupported warehouse weights format")
    if document.get("catalog_sha256") != hashlib.sha256(catalog_path.read_bytes()).hexdigest():
        raise ValueError("warehouse weights catalog hash mismatch")
    if require_approved and document.get("review_status") != "approved":
        raise ValueError("warehouse weights require explicit completed review")
    rows = document.get("offers")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("warehouse weights require offer rows")
    ids = [row.get("offer_id") for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != rental_offer_ids:
        raise ValueError("warehouse weights must cover each rental offer exactly once")
    weights = {}
    for row in rows:
        weight = row.get("warehouse_capacity_units_per_asset")
        if not positive_integer(weight):
            raise ValueError("warehouse weight must be a positive integer (not boolean)")
        if not row.get("rationale") or row.get("confidence") not in {"low", "medium", "high"}:
            raise ValueError("warehouse weight needs rationale and uncertainty")
        weights[row["offer_id"]] = weight
    for offer_id, weight in ANCHORS.items():
        if offer_id in weights and weights[offer_id] != weight:
            raise ValueError(f"fixed warehouse anchor changed: {offer_id}")
    return weights
