"""V52 publishes the existing durable-resale formula, without quoting products."""
ENVIRONMENT_PROTOCOL = 'outfitter-seasonal-v52-public-resale-formula'
CORE_RELEASE_ID = 'outfitter-v52-public-resale-formula-core-r1'
TOOL_INTERFACE_VERSION = 50
POLICY_ID = 'public-durable-resale-formula-v1'


def enabled(metadata):
    from .operating_history import ENVIRONMENT_PROTOCOL as V53_ENVIRONMENT_PROTOCOL
    from .recovery_state import ENVIRONMENT_PROTOCOL as V54_ENVIRONMENT_PROTOCOL
    return (metadata.get('environment_protocol') in {ENVIRONMENT_PROTOCOL, V53_ENVIRONMENT_PROTOCOL, V54_ENVIRONMENT_PROTOCOL}
            and metadata.get('resale_formula_disclosure_policy') == POLICY_ID
            and int(metadata.get('tool_interface_version', 0)) >= TOOL_INTERFACE_VERSION)


def public_rule(rate, exponent):
    # Render the actual loaded economic scales rather than duplicating pricing
    # constants. This text does not change nominal_residual_value or quotes.
    return (
        f'Owned durable-asset nominal dealer value is V = P * {rate:g} * c * '
        f'max(0, 1 - w) ** {exponent:g}. P is that asset\'s recorded unit_cost '
        '(acquisition/replacement-cost basis), not its rental price or a later supplier price. '
        'c is its current condition_ratio between 0 and 1; '
        'w is the current combined_wear reported in asset details. '
        f'A newly delivered, unused asset at c=1 and w=0 therefore has a {rate * 100:g}% '
        'nominal recovery baseline, before monetary rounding, even before its first rental. '
        'Monetary rounding applies; use the actual quote for settlement. '
        'This is a value formula, not permission '
        'to sell: ordinary sales still require an owned, physically held, available '
        'or damaged asset. A terminal repairing-asset valuation uses its recorded '
        'post-repair target condition; prepaid repair is not deducted again. '
        'Final cleaning deductions and all other status/terminal exclusions still apply. '
        'This is the nominal value before any separately enabled liquidation-depth '
        'discount; default benchmark tracks disable that discount. '
        'Supplier trials are not owned and have no terminal asset value. '
        'No future condition, wear, availability or resale amount is guaranteed.'
    )
