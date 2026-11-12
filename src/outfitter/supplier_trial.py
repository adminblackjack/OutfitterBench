"""Shared V37 supplier-trial protocol identity and fixed public policy."""
from __future__ import annotations


ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v37-supplier-trial"
TOOL_INTERFACE_VERSION = 32
CORE_RELEASE_ID = "outfitter-v37-shared-supplier-trial-core"
POLICY_ID = "bounded-supplier-trial-v1"

MAX_ACCEPTED_TRIALS = 4
MAX_ACTIVE_TRIALS = 1
MAX_CONTRACT_STARTS = 3
WINDOW_DAYS = 21
MAX_UNIT_COST = 1500.0
HANDLING_FEE = 10.0
USAGE_FEE_MULTIPLIER = 1.0
ACTIVITY_EVENT_TYPES = (
    "supplier_trial_delivered",
    "supplier_trial_return_requested",
    "supplier_trial_return_pending",
    "supplier_trial_returned",
    "supplier_trial_usage_fee_charged",
)
