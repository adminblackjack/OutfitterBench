"""V39 actor-interface identity for proactive reservation contingency."""
from __future__ import annotations


ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v39-contingency-actor-clarity"
TOOL_INTERFACE_VERSION = 34
CORE_RELEASE_ID = "outfitter-v39-shared-contingency-interface-core"
POLICY_ID = "contingency-actor-interface-v1"


def policy_document() -> dict:
    return {
        "format": POLICY_ID,
        "reservation_view": "public_version_eligibility_and_same_profile_space",
        "inbox": "updates_separate_from_actionable_reports",
        "maximum_public_forecast_inclusive_dates": 15,
        "maximum_public_capacity_inclusive_dates": 15,
        "rental_asset_schema": "explicit_mutually_exclusive_view_v1",
        "customer_model_changed": False,
        "economics_changed": False,
        "scoring_changed": False,
    }
