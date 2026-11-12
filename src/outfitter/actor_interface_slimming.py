"""V41 compact, hash-free actor-interface protocol identity."""
from __future__ import annotations


ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v41-compact-hash-free-actor-interface"
TOOL_INTERFACE_VERSION = 36
CORE_RELEASE_ID = "outfitter-v41-shared-compact-actor-interface-core"
POLICY_ID = "compact-hash-free-actor-interface-v1"


def policy_document() -> dict:
    return {
        "format": POLICY_ID,
        "capacity_mutation_result": "touched-offers-current-and-future-only-v1",
        "capacity_query_result": "requested-offers-and-date-window-only-v1",
        "capacity_intervals": "merge-adjacent-equal-values-v1",
        "actor_integrity_identifiers": "omit-hash-sha256-and-fingerprint-keys-v1",
        "internal_integrity_checks_changed": False,
        "customer_model_changed": False,
        "economics_changed": False,
        "scoring_changed": False,
    }
