"""V53 information layering only; no business transitions or customer policy."""
from __future__ import annotations

import copy


def system_prompt(t, *, tool_call_protocol, max_read_only_batch_size):
    retail = " and travel-supply retail" if t["retail_enabled"] else ""
    retail_value = " + final retail inventory dealer value" if t["retail_enabled"] else ""
    usage = ("Model usage is charged to shop cash; see scoring rules for rates."
             if t["usage_charged_to_cash"] else
             "Model usage is recorded separately and does not reduce shop cash or the official score.")
    if tool_call_protocol == "read_only_batch_v1":
        protocol = (
            f"You may batch at most {max_read_only_batch_size} independent read-only tools in one response. "
            "State changes, memory writes/deletions and close_business_day must each be the only tool "
            "request in a response. Invalid batches execute nothing. Tool descriptions identify read-only "
            "and state-changing operations."
        )
    elif tool_call_protocol == "first_only":
        protocol = "Submit one tool request per response; if several are submitted, only the first executes."
    else:
        raise ValueError(f"unsupported tool-call protocol: {tool_call_protocol}")
    return f"""You manage a seasonal outdoor-equipment rental{retail} shop in Beijing. All money is CNY. Keep the shop solvent and maximize its official final value. Use business tools to perform actions; ordinary text and notes do not alter contracts or complete business operations. Detailed public rules are available through read_business_rules; omit topics for its directory or request relevant topics.

Manager decisions, new demand and pickups end on Day {t['decision_horizon_day']}. Automatic runoff continues through Day {t['runoff_end_day']}, when final value is scored; there are no manager actions during runoff. For a solvent shop, final value = cash + final owned-equipment net dealer value{retail_value} - refundable deposits - unearned reservation revenue - refund-protection credit liabilities. An unpaid required payment ends the run in insolvency after any applicable refund protection. See scoring rules for terminal eligibility, deductions and insolvency scoring. {usage}

A newly delivered, unused, intact owned asset has a {t['new_asset_residual_rate'] * 100:g}% nominal recovery baseline relative to its recorded acquisition cost. Actual value depends on condition and wear; the full formula and settlement rules are in the scoring handbook. Deposits and unearned prepayments are liabilities, not profit. Cash, durable-capital headroom and warehouse space are separate constraints; capital headroom is not additional cash.

Each day opens after automatic morning processing, including deliveries, returns, service jobs and eligible reservation pickups. Closing resolves pending reservation defaults and pickup risks before walk-ins, charges daily costs and advances the day. Contractual return dates and supplier delivery estimates are not guarantees of physical availability. Customers can continue using rented equipment without approval. Normal returned equipment must pass through cleaning and cannot be rented again on its return day. Current asset status and service terms determine whether it can be used now.

A quotation may create a binding commitment immediately or at close. Customer authorization and successful confirmation do not certify inventory or scheduling feasibility. Every committed line, including a selected optional line, is binding; changes require the applicable customer-authorization process. Merchant cancellation or nonfulfillment incurs refunds/credits and compensation under the disclosed rules. Initial authorization, later modification and actual pickup are different events; current contract terms and settlement receipts record what has occurred. Walk-ins use posted prices, while reservations use their own locked terms. Offers without a posted price do not serve walk-ins. Pricing units and each operation's permissions, deadlines and default consequences are described by its tools and the handbook.

Every business day starts a fresh model session. Business state and named memory documents persist; previous days' reasoning, ordinary assistant text and transcripts are not supplied. Retrieve saved notes through memory tools; they are not automatically loaded. A closing handoff is supplied only on the next day. Older same-day messages may leave the context window.

There are at most {t['daily_model_rounds']} model responses per day. Each response consumes a round, including responses without tools or with errors. Budget notices state the remaining rounds, including the next response. At the daily limit, permitted tools from the last response execute before automatic close if the day remains open. {protocol} Call close_business_day when today's decisions are complete.
"""


def resale_rule(rate, exponent):
    return (
        f"Owned durable-asset nominal dealer value is V = P * {rate:g} * c * max(0, 1 - w) ** {exponent:g}. "
        "P is that asset's recorded unit_cost (acquisition cost), not its rental price or a later "
        "supplier price; c is current condition_ratio in [0, 1]; w is reported combined_wear. "
        f"An unused intact owned asset at c=1 and w=0 has a {rate * 100:g}% nominal recovery baseline. "
        "Monetary rounding applies. Sale eligibility and terminal deductions are separate from this formula."
    )


PARTIAL_PICKUP_RULE = (
    "Reduced pickup may omit only optional committed lines and requires inquire_partial_fulfillment, "
    "customer acceptance, then explicit select_partial_fulfillment on the pickup day. Inquiry is allowed "
    "once per commitment version and is recorded even if refused. Refusal or no selection leaves the "
    "original obligations in force. Actual reduced delivery credits the missing lines' full rent and "
    "pays compensation on their original first-day guarantee bases; inquiry and selection pay nothing. "
    "Unresolved pickup risk fails at close. Other authorized substitutions remain available."
)


def business_manual(env, original):
    """Finalize after inherited mixins so they cannot reappend removed material."""
    from .operating_history import HOLDING_RULE
    from .school_holiday import PUBLIC_RULES as SCHOOL_RULES

    manual = copy.deepcopy(original)
    resale_prefix = "Owned durable-asset nominal dealer value is V = "
    for topic, chapter in manual.items():
        rules = []
        for rule in chapter["rules"]:
            if rule == SCHOOL_RULES:
                # Preserve the independent weather-information fact, not the
                # holiday-specific customer-generation explanation or disclaimers.
                rules.append("Forecasts affect advance requests; observed weather affects walk-in visits.")
                continue
            if rule.startswith(resale_prefix):
                if topic == "scoring":
                    rules.append(resale_rule(env.lifecycle_scales.new_asset_residual_rate,
                                             env.lifecycle_scales.residual_wear_exponent))
                else:
                    rules.append("See scoring rules for the owned-equipment resale formula and terminal valuation; "
                                 "current holdings can be quoted through view_liquidation_quotes.")
                continue
            if topic == "assets" and rule == HOLDING_RULE:
                rules.append("Offer/asset holding_costs give per-item rates. Billing rules and the distinction "
                             "between storage billing units and warehouse capacity units are in finance.")
                continue
            if rule.startswith("Reduced pickup requires explicit customer consent"):
                rules.append(PARTIAL_PICKUP_RULE)
                continue
            if rule.startswith(("Historical quotation alternatives are not permission to omit",
                                "A selected optional reservation line is a binding promise")):
                continue
            if rule.startswith("Historical preauthorized plans are not directly selectable"):
                rule += " Complete-plan recovery must retain every currently committed line; historical omit options cannot waive a commitment."
            if rule.startswith("service_history revenue includes continuation rent"):
                rule = rule.replace("Offer rental_days includes", "Offer rental_asset_days includes")
            rule = rule.replace("The historical field names ", "The fields ")
            rule = rule.replace("Equipment still out is unavailable now; future availability must be judged using current evidence. ", "")
            rules.append(rule)
        chapter["rules"] = rules

    manual["workspace"]["rules"][0] = (
        f"Each business day starts a fresh model session with at most {env.config.max_iterations_per_day} "
        "model responses. A response consumes one round even without a tool or with an error. "
        + (f"Independent read-only requests may be batched, at most {env.config.max_read_only_batch_size} per response. "
           "State changes, memory writes/deletions and close_business_day must each be submitted alone. "
           "Invalid batches execute nothing."
           if env.config.parallel_tool_calls else
           "Submit one tool per response; if several are submitted, only the first executes.")
    )
    # V53's frozen flat-nominal runtime rejects liquidation_depth_enabled.
    # Do not document an unavailable alternative here; legacy versions retain theirs.
    return manual


def tool_description(name, description, *, read_only):
    # Apply classification after all version-specific description overrides.
    for suffix in (
        " Read-only; may be batched with independent read-only requests.",
        " Read-only; submit one tool per response.", " Read-only.",
        " Submit this tool alone in the response.",
    ):
        description = description.replace(suffix, "")
    if name == "inspect_offer":
        description += " Includes per-item holding_costs at the current procurement cost."
    elif name == "close_business_day":
        description = (
            "Finish today's decisions. Settle reservation defaults and pickup risks before walk-ins, "
            "charge daily costs and advance the day. reservation_close_outcomes reports actual close-time "
            "pickups, failures, cancellations and refund/compensation payment status. A handoff is supplied "
            "on the next day only."
        )
    return description + (" Read-only." if read_only else " State-changing; submit this tool alone in a response.")
