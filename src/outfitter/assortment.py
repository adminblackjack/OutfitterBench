"""Shared V35 optional-retail protocol identity."""
from __future__ import annotations


ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v35-optional-retail"
TOOL_INTERFACE_VERSION = 30
CORE_RELEASE_ID = "outfitter-v35-shared-rental-core"
SUPPORTED_ASSORTMENT_MODES = ("rental_only", "rental_and_retail")


def benchmark_track_id(assortment_mode: str) -> str:
    if assortment_mode == "rental_only":
        return "outfitter-rental-main"
    if assortment_mode == "rental_and_retail":
        return "outfitter-rental-retail-secondary"
    raise ValueError(f"unsupported assortment mode: {assortment_mode}")

