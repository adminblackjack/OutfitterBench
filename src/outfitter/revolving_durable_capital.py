"""V40 revolving gross deployed durable-capital protocol identity."""
from __future__ import annotations


ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v40-revolving-durable-capital"
TOOL_INTERFACE_VERSION = 35
CORE_RELEASE_ID = "outfitter-v40-shared-revolving-durable-capital-core"
POLICY_ID = "revolving-gross-durable-capital-envelope-v1"


def policy_document() -> dict:
    return {
        "format": POLICY_ID,
        "configured_budget_semantics": "additional_durable_capital_headroom",
        "capital_limit": "initial_owned_durable_basis_plus_additional_headroom",
        "deployed_scope": "owned_durable_assets_until_release_effective_day",
        "committed_scope": "accepted_unreceived_owned_durable_units",
        "asset_basis": "historical_purchase_basis_integer_cents",
        "release_paths": [
            "service_life_retirement_and_disposal",
            "manager_retirement_and_disposal",
            "dealer_sale",
        ],
        "release_delay_days": 1,
        "retail_order_lines_in_scope": False,
        "supplier_trial_assets_in_scope": False,
        "revenue_increases_limit": False,
        "sale_proceeds_increase_limit": False,
        "unused_headroom_has_terminal_value": False,
        "supplier_cancellation_mechanism_added": False,
    }
