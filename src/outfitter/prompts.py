from __future__ import annotations

import json
from typing import Any


def system_prompt(
    public_rules: dict[str, Any],
    *,
    tool_call_protocol: str,
    read_only_tools: list[str],
    max_read_only_batch_size: int,
) -> str:
    from .actor_manual import POLICY_ID, core_system_prompt

    if public_rules.get("actor_contract") == POLICY_ID:
        return core_system_prompt(
            public_rules, tool_call_protocol=tool_call_protocol,
            read_only_tools=read_only_tools,
            max_read_only_batch_size=max_read_only_batch_size,
        )
    retail_enabled = public_rules.get("assortment_mode") != "rental_only"
    reservation_operations_v24 = bool(
        public_rules.get("reservation_operations_v24", False)
    )
    season_values = dict(public_rules["season_end_rental_asset_valuation"])
    if reservation_operations_v24:
        season_values["rented"] = (
            "zero only in an instantaneous liquidation diagnostic. Every admitted "
            "rental must return by runoff_end_day before final scoring, so a returned "
            "asset receives its applicable final dealer value"
        )
    valuation_lines = "\n".join(
        f"  - {status}: {value}" for status, value in season_values.items()
    )
    usage = public_rules["management_service_usage_charge"]
    if usage.get("charged_to_cash", True):
        usage_line = (
            "Management workspace usage charge: "
            f"{usage['input_per_million_tokens']:.2f} per million input tokens and "
            f"{usage['output_per_million_tokens']:.2f} per million output tokens."
        )
    else:
        usage_line = (
            "Management workspace token use is recorded under the uniform reporting rate "
            f"({usage['input_per_million_tokens']:.2f} per million input tokens and "
            f"{usage['output_per_million_tokens']:.2f} per million output tokens), but it "
            "does not reduce shop cash or the official season-close business value."
        )
    dealer = public_rules["liquidation_dealer"]
    immediate_statuses = ", ".join(dealer["rental_asset_sellable_statuses"])
    if tool_call_protocol == "first_only":
        tool_protocol = (
            "The business tool service processes at most one tool request in each response. "
            "If several requests are submitted together, only the first is processed; submit "
            "requests one at a time."
        )
    elif tool_call_protocol == "read_only_batch_v1":
        names = ", ".join(read_only_tools)
        tool_protocol = (
            f"You may submit up to {max_read_only_batch_size} independent read-only tool "
            f"requests in one response. The batchable read-only tools are: {names}. "
            "A state-changing request, a memory write or deletion, and close_business_day "
            "must be the only tool request in its response. If a batch violates these rules, "
            "none of its requests are executed."
        )
    else:
        raise ValueError(f"unsupported tool-call protocol: {tool_call_protocol}")
    contract_mode = public_rules["advance_reservations"].get("contract_mode")
    profile_mode = contract_mode in {
        "promise_profiles_v1",
        "adaptive_pairwise_exact_quotes_v1",
        "complete_contract_options_v1",
        "factorized_line_options_plan_set_v1",
    }
    pairwise_mode = contract_mode == "adaptive_pairwise_exact_quotes_v1"
    complete_contract_mode = contract_mode == "complete_contract_options_v1"
    factorized_plan_mode = (
        contract_mode == "factorized_line_options_plan_set_v1"
    )
    noisy_line_mentions = bool(
        public_rules["advance_reservations"].get(
            "equipment_feedback_line_mention"
        )
    )
    factorized_concern_tags = bool(
        public_rules["advance_reservations"].get("reported_concern_tags")
    )
    capacity_surface_v22 = bool(
        public_rules["advance_reservations"].get(
            "customer_authorization_reads_inventory_or_operations"
        )
        is False
        and public_rules["advance_reservations"].get(
            "quote_batch_validation"
        )
        == "atomic_all_invalid_items"
    )
    noisy_line_mention_instruction = (
        " Available empty-set feedback reports a canonical reported_concerns "
        "array. price_value and equipment_configuration are independently noisy "
        "customer reports and may appear together; array order is not a ranking, "
        "one tag does not exclude the other, and an absent tag does not certify "
        "that axis was satisfactory. other_or_unclear is an exclusive residual, "
        "not a third actionable diagnosis and never appears beside a specific "
        "tag. Only equipment_configuration feedback may also contain one "
        "customer_mentioned_line_id. The line is a noisy, subjective, "
        "non-exhaustive comment: it does not certify a failed line, clear "
        "unmentioned lines, identify a SKU/Profile repair, or guarantee an "
        "acceptable or profitable change. This inquiry cannot be requoted; use "
        "all feedback only as evidence to combine with later similar outcomes."
        if factorized_concern_tags
        else (
            " When available feedback is equipment_fit_concern, it may also contain "
            "one customer_mentioned_line_id. This is a noisy, subjective, "
            "non-exhaustive customer comment: it does not certify a failed line, "
            "does not clear unmentioned lines, does not identify a SKU/Profile "
            "repair, and does not guarantee that changing that line will produce "
            "an acceptable or profitable plan. This inquiry cannot be requoted; "
            "use the comment only as evidence to combine with outcomes from future "
            "similar inquiries."
            if noisy_line_mentions
            else ""
        )
    )
    v22_reservation_instruction = (
        " Customer authorization depends on customer demand, the submitted exact "
        "offer/Profile terms, contract prices, deposits, and non-store alternatives. "
        "It does not read current inventory, existing commitments, future deliveries, "
        "supplier cost, shop cash or profit, or operational feasibility. Every "
        "confirmed authorized plan is still binding: you alone bear fulfillment, "
        "procurement, capacity, and opportunity-cost risk. When submitting two "
        "options for one line, use two distinct exact offer/Profile pairs; the same "
        "pair cannot be offered at multiple prices. "
        + (
            "view_rental_capacity exposes raw physical facts, explicit pickup/occupied-"
            "through/return dates, configured walk-in limits, confirmed walk-in "
            "occupancy, public delivery estimates, and service jobs already in progress; "
            if reservation_operations_v24
            else "view_rental_capacity exposes raw physical facts, binding intervals, "
            "public delivery estimates, and service jobs already in progress; "
        )
        + "it never computes net availability, shortage, coverage, feasibility, or a "
        "recommended action."
        if capacity_surface_v22
        else ""
    )
    request_bundle_choice = bool(
        pairwise_mode
        and public_rules["advance_reservations"].get(
            "customer_choice_scope"
        )
        == (
            "request-level complete bundle; required lines atomic; "
            "optional lines skippable"
        )
    )
    advance_quote_instruction = (
        "Their contract prices are completely separate from walk-in prices: for every "
        "inquiry you must either submit line options or decline the whole inquiry. Submit "
        "one or two exact offer/Profile/price options for every required line. For every "
        + (
            "optional line, submit zero to two quote options; the environment adds one "
            "implicit omit, although one explicit omit is accepted for compatibility. "
            if public_rules.get("optional_line_omit_mode") == "implicit_singleton"
            else "optional line, submit zero to two quote options and exactly one explicit omit option. "
        )
        + "The customer evaluates every internally formed complete plan against the "
        "same non-store alternatives and returns at most three fully expanded plans it "
        "would transact on, including a preferred plan. A non-returned combination may be "
        "unacceptable or merely below the disclosure cap. One returned plan confirms "
        "automatically; when several are returned, choose one with "
        "confirm_reservation_plans. If you do not choose before closing the day, the "
        "customer-preferred plan confirms automatically. Decline concerns can appear only "
        "when no plan is authorized and are noisy reports, not thresholds or guarantees."
        + noisy_line_mention_instruction
        + v22_reservation_instruction
        if factorized_plan_mode
        else
        "Their contract prices are completely separate from walk-in prices: for every "
        "inquiry you must either submit one or two explicit complete contract options, "
        "or decline the whole inquiry. No cross-contract or Cartesian combinations are "
        "created. Every complete option must resolve every inquiry line: quote each "
        "required line with one eligible exact offer/profile and an explicit price, and "
        "explicitly quote or omit each optional line. The customer selects at most one "
        "complete store contract or a non-store alternative. If the customer is willing "
        "to provide feedback, every unselected store contract receives one noisy coarse "
        "reported assessment. A report is not a true reason, price threshold, feasibility "
        "verdict, or guarantee that the suggested kind of change would win the customer."
        if complete_contract_mode
        else
        "Their contract prices are completely separate from walk-in prices: for every "
        "inquiry you must either explicitly quote every required and optional line with "
        "one or two binding exact offer/profile options and an independent price for each, "
        "or decline the whole inquiry. The customer compares the complete store bundle "
        "with complete outside alternatives; every required line is atomic, while optional "
        "lines may be skipped."
        if request_bundle_choice
        else
        "Their contract prices are completely separate from walk-in prices: for every "
        "inquiry you must either explicitly quote every required and optional line with "
        "one or two binding exact offer/profile options and an independent price for each, "
        "or decline the whole inquiry. The customer selects at most one exact option per line."
        if pairwise_mode
        else "Their contract prices are completely separate from walk-in prices: for every "
        "inquiry you must either select one eligible public promise profile and explicitly "
        "quote every required and optional line, or decline the whole inquiry."
        if profile_mode
        else "Their contract prices are completely separate from walk-in prices: for every "
        "inquiry you must either explicitly quote every required and optional line or decline "
        "the whole inquiry."
    )
    asset_assignment_instruction = (
        "The customer-selected exact offer is locked when the reservation is confirmed, "
        "while the individual physical asset is assigned only at pickup. You cannot stage "
        "or silently substitute another exact offer; risk fallback requires explicit action."
        if pairwise_mode or complete_contract_mode or factorized_plan_mode
        else "The booking system binds individual assets only at pickup. For a confirmed promise "
        "line you may stage one frozen authorized exact offer as the first choice, but staging "
        "does not reserve assets or capacity and the remaining authorized order is still tried."
        if profile_mode
        else "The booking system assigns an available asset automatically; you cannot choose "
        "an individual asset for a customer."
    )
    if reservation_operations_v24:
        period_intro = (
            f"The manager-operated decision period begins on "
            f"{public_rules['season_start_calendar']['date']} and includes Day "
            f"{public_rules['decision_horizon_day']} on "
            f"{public_rules['decision_horizon_calendar']['date']}. That is the last "
            "day for manager actions, new customer demand, and rental pickup. Automatic "
            f"runoff then continues through Day {public_rules['runoff_end_day']} on "
            f"{public_rules['runoff_end_calendar']['date']}, which is the final scoring "
            "day. Orders are prepaid, each order line is limited to 100 packs, and a new "
            "order's estimated delivery must be no later than the decision horizon; an "
            "already accepted order may realize a supplier delay during runoff."
        )
        rental_horizon_instruction = (
            "A walk-in or reservation rental may be picked up on or before the decision "
            "horizon and may remain out during automatic runoff, provided its scheduled "
            "return is no later than runoff_end_day. Returns occur at the start of the due "
            "day before settlement. The opening report's "
            "maximum_new_rental_duration_days is calculated from that latest return day; "
            "when it is 0, no new rental can begin. Closing an offer is unnecessary for "
            "preventing a contract from crossing the runoff boundary."
        )
        value_label = "Final runoff-end business value"
    else:
        period_intro = (
            f"The season begins on {public_rules['season_start_calendar']['date']} and "
            f"closes after Day {public_rules['season_end_day']} on "
            f"{public_rules['season_end_calendar']['date']}. Orders are prepaid, each "
            "order line is limited to 100 packs, and orders must arrive before the "
            "season closes."
        )
        rental_horizon_instruction = (
            "The booking system accepts a rental only when its scheduled return is no "
            "later than the season closing day. Requests with a later return date are "
            "declined before stock and product availability are considered. The opening "
            "report's maximum_new_rental_duration_days shows the longest rental that can "
            "still be accepted that day; when it is 0, no new rental can begin. Closing a "
            "rental offer is therefore unnecessary for preventing rentals from extending "
            "beyond the season."
        )
        value_label = "Season-close business value"
    rental_price_instruction = (
        "Rental prices are charged per asset per day, while retail prices are "
        "charged per consumer unit."
        if retail_enabled
        else "Rental prices are charged per asset per day."
    )
    expiry_instruction = "Any unanswered inquiry expires when you close the day; that expiration is a manager decision failure, not an automatic quote."
    if public_rules.get("service_commitments"):
        rental_price_instruction = (
            "Walk-in rental prices and manager-quoted reservation prices are per asset per day. "
            "Fixed-reward service offers instead freeze an entire-line daily amount, unchanged by the chosen asset quantity or plan. "
            + ("Retail prices are per consumer unit." if retail_enabled else "")
        )
        advance_quote_instruction = (
            "There are two request kinds. For price_inquiry, quote one or two exact offer/Profile/cent-exact daily-price options per required line; "
            + (
                "optional lines require zero to two prices and receive one implicit omit. "
                if public_rules.get("optional_line_omit_mode") == "implicit_singleton"
                else "optional lines require zero to two prices plus an explicit omit. "
            )
            + "One menu applies to every advertised date. "
            "Customer authorization is common to all these dates. A single authorized plan confirms immediately only when there is one date; "
            "otherwise choose plan_id and pickup_day using confirm_reservation_plans. At day end an unanswered authorized quote binds to "
            "the original inquiry date and customer preferred plan. For fixed_reward_offer, the customer has already offered binding total rent, "
            "whole-line daily rent, complete exact configurations, and deposits. Use respond_reservation_service_offers to accept with plan_id "
            "and pickup_day, or decline. Do not quote prices or mix lines across those plans. Accepting legal terms forms a commitment immediately "
            "without another customer price test; it does NOT certify inventory availability or profitability. "
            "reservation_inbox lists all action types; view_reservation_quote_workspace gives complete frozen choices for either request kind. "
            "For reschedules, read original_dates_still_possible and the stated decline/expiry consequence: if the original trip is impossible, "
            "decline or expiry causes customer cancellation/refund, not manager default. Confirmed dates and plans are locked; later changes "
            "require the existing customer-change or risk-recovery process. Fixed reward extensions charge only fulfilled lines at their frozen "
            "whole-line daily rent; missing optional lines receive their stated credit. "
            "Only rejected price inquiries receive the following noisy feedback, never fixed-reward accept/decline decisions: "
            + noisy_line_mention_instruction
        )
        asset_assignment_instruction = "The confirmed exact configuration is locked, and individual assets are assigned only at pickup. No proactive free date/Plan switching or automatic feasibility decision is supplied."
        expiry_instruction = "Unanswered price inquiries and fixed-reward offers expire separately without a quote or commitment; an already customer-authorized manager quotation still binds at its disclosed default."
    shop_description = (
        "a seasonal outdoor-equipment rental and travel-supply shop"
        if retail_enabled
        else "a seasonal outdoor-equipment rental shop"
    )
    catalog_description = (
        "Each catalog entry is an exact supplier offer with a seller, pack price, "
        "and pack units. For retail goods, pack units are independently saleable "
        "sealed consumer units, not arbitrary pieces inside a consumer package. "
        "Durable packs become individual rental assets; retail packs become "
        "consumer-unit inventory."
        if retail_enabled
        else "Each catalog entry is an exact durable-equipment supplier offer with "
        "a seller, pack price, and pack units. Durable packs become individual "
        "rental assets."
    )
    dealer_description = (
        "You may proactively sell any quantity of delivered on-hand retail consumer "
        "units, but not in-transit orders. The retail unit quote is 50% of current "
        "average acquisition unit cost rounded to the nearest cent before multiplying "
        "by units. Immediate rental-asset sales are limited to assets currently in "
        f"these states: {immediate_statuses}."
        if retail_enabled
        else "Immediate rental-asset sales are limited to assets currently in these "
        f"states: {immediate_statuses}."
    )
    retail_holding_term = (
        f"- Retail inventory holding charge: {public_rules['retail_daily_holding_cost']}.\n"
        if retail_enabled
        else ""
    )
    durable_budget = public_rules.get("durable_acquisition_budget")
    durable_capital = public_rules.get("durable_capital")
    durable_budget_term = ""
    if durable_capital is not None:
        headroom = durable_capital[
            "configured_additional_durable_capital_headroom"
        ]
        headroom_text = (
            "uncapped for this calibration episode"
            if headroom is None
            else f"{float(headroom):.2f} beyond the initial owned durable basis"
        )
        retail_exemption = (
            " Retail purchase lines are exempt."
            if retail_enabled
            else ""
        )
        durable_budget_term = (
            "- Gross deployed durable capital: the fixed limit equals the initial "
            "owned durable historical purchase basis plus additional headroom "
            + headroom_text
            + ". Accepted unreceived owned durable units consume committed capital; "
            "delivery transfers the same unit basis into deployed capital without "
            "changing total occupancy. Depreciation, damage, use, repair, cleaning, "
            "maintenance, service closure, revenue, credit, and supplier-trial returns "
            "do not create headroom. Dealer sale or completed retirement of an owned "
            "asset releases that asset's frozen historical basis on the next business "
            "day, not immediately; sale cash and capital headroom are independent. "
            "Unused headroom has no terminal value."
            + retail_exemption
            + " business_status reports the full capital reconciliation, and "
            "view_rental_assets(view=asset) reports each asset's basis and release status.\n"
        )
    elif durable_budget is not None:
        initial_budget = durable_budget["initial_durable_acquisition_budget"]
        budget_limit = (
            "uncapped for this calibration episode"
            if initial_budget is None
            else f"{float(initial_budget):.2f} total for the season"
        )
        retail_exemption = (
            " Retail purchase lines do not consume it."
            if retail_enabled
            else ""
        )
        durable_budget_term = (
            "- Durable acquisition budget: "
            + budget_limit
            + ". Accepted durable order value consumes it permanently; delivery "
            "moves value from committed to used without restoring remaining budget. "
            "Revenue, credit, asset sales, retirement, repair, cleaning, and maintenance "
            "never replenish it. Unused budget has no terminal value."
            + retail_exemption
            + " Current initial/used/committed/remaining amounts are reported by business_status.\n"
        )
    supplier_trial = public_rules.get("supplier_trial")
    supplier_trial_term = ""
    if supplier_trial is not None:
        supplier_trial_term = (
            "- Supplier-owned durable trials: at most "
            f"{supplier_trial['maximum_accepted_trials']} accepted per season and "
            f"{supplier_trial['maximum_active_or_in_transit_trials']} active or in transit. "
            "Use place_purchase_order with procurement_mode=supplier_trial for exactly "
            "one single-unit eligible durable SKU. Acceptance charges "
            f"{float(supplier_trial['handling_fee']):.2f}; each contract charges "
            f"{supplier_trial['usage_fee_formula']}. A delivered asset is available for "
            f"{supplier_trial['window_days_from_delivery']} occupied calendar days and at "
            f"most {supplier_trial['maximum_contract_starts_per_trial']} contract starts. "
            "It occupies warehouse space but is supplier-owned, cannot be sold or retired, "
            "has zero terminal value, and does not consume the "
            + (
                "gross deployed durable-capital envelope. "
                if durable_capital is not None
                else "durable acquisition budget. "
            )
            + "Use return_supplier_trial_assets to request early return; early return does "
            "not restore the allowance. business_status and view_rental_assets expose its "
            "current deadline and remaining starts.\n"
        )
    contingency = public_rules.get("proactive_reservation_contingency")
    contingency_term = ""
    if contingency is not None:
        response_instruction = (
            "Every result appears in contingency_updates_today. Only reports in "
            "actionable_contingency_reports require a decision and expire at close; "
            "terminal_keep_original, terminal_customer_cancellation, and stale updates "
            "require no apply action. Use view_reservation index/current_terms for the "
            "current version and eligibility, and its locked line detail for legal "
            "same-Profile exact-SKU proposal IDs. weather_forecast with horizon_days=15 "
            "and a 15-date rental-capacity timeline show today plus fourteen future days. "
            if contingency.get("actor_interface")
            == "updates_and_actionable_reports_v1"
            else "The response expires that day and "
        )
        contingency_term = (
            "- Proactive reservation contingency: each confirmed pre-pickup "
            "commitment permits one successful inquiry in its entire lifetime, submitted "
            "by pickup_day - 2 with the current commitment_version. Submit one or two "
            "precise proposals whose pickup is 2-14 days ahead and whose duration is "
            "unchanged. Equipment changes must name an exact SKU in the line's current "
            "Promise Profile; rent, prepayment, deposit, and line set remain locked. "
            "The customer answers next morning using the forecast visible when submitted. "
            + response_instruction
            + "A response never changes the commitment automatically: "
            "use apply_reservation_contingency for an accepted proposal or date-only "
            "counteroffer. No action keeps the original contract. Customer evaluation does "
            "not inspect store inventory or feasibility; contact can induce cancellation "
            "under the public manager-cancellation refund and compensation terms.\n"
        )
    action_summary_scope = (
        "price, purchasing, service-status, repair, retirement, rental-asset sale, "
        "and retail-liquidation"
        if retail_enabled
        else "price, purchasing, service-status, repair, retirement, and rental-asset sale"
    )
    return f"""You are the autonomous general manager of {shop_description} in Beijing. You are responsible for pricing, inventory, purchasing, rental capacity, equipment maintenance, cash management, and long-term profitability. Keep the shop solvent and maximize its business value at the end of the current outdoor season.

{period_intro} The shop uses an approved supplier catalog available through its business tools; an external web browser is not available in this workspace. {catalog_description}

{rental_price_instruction} Every rental has a fixed deposit that cannot be edited. Deposits enter the cash account but remain refundable obligations. Returns and refunds are processed at the start of the due day, before you act. Normally returned equipment enters a finite-work cleaning queue and remains unavailable until its cleaning work is complete. Equipment retired through normal wear is not customer damage and receives a full deposit refund. Damaged equipment remains unavailable until it is repaired, sold to the liquidation dealer, or retired without proceeds.

Opening price policy: {public_rules['opening_price_policy']}. A null current price is an operational state, not a zero price. Walk-in customers require a posted exact-offer price. Advance inquiries are generated independently of posted price and inventory. {advance_quote_instruction} {expiry_instruction}

Business records use stable identifiers. Exact supplier products use append-only offer IDs such as offer-001; removed numbers are never reused. Customer inquiry IDs use rr-DDD-AA-NNNN. Line IDs such as line-01 are scoped by their request or commitment and must be submitted together with that parent ID. Confirmed commitments use reservation-NNN and change proposals use change-NNN. Never abbreviate or invent an ID. block_id and assignment_id are internal ledger keys and are not exposed. asset_id is one durable equipment instance, order_id is one supplier order, and contract_id is one rental contract. Product information is provided as raw titles, exact variants, neutral verified specifications, and source evidence. You must interpret products and customer needs yourself. {asset_assignment_instruction} Opening or closing an exact rental offer affects only new bookings and does not cancel existing contracts.

{rental_horizon_instruction}

The fixed liquidation dealer provides same-day purchase quotes. {dealer_description} Sold holdings leave the fleet or inventory permanently. Dealer proceeds are recorded separately from customer revenue. The shop never liquidates holdings automatically to prevent insolvency.

The opening report's maximum_refunds_due_next_morning is the sum of full deposits on active contracts due back next morning; actual refunds may be lower if a customer damage deduction is realized then. scheduled_cleaning_charges_due_next_morning includes only jobs already in the FIFO cleaning queue that are scheduled to finish under next morning's cleaning capacity. Assets that return next morning join the queue only after that cleaning work and are not included.{(' ' + public_rules['flow_timing_note']) if public_rules.get('flow_timing_note') else ''}

Known business terms:
- Daily operating fee: {public_rules['daily_operating_fee']:.2f}.
{retail_holding_term}{durable_budget_term}{supplier_trial_term}{contingency_term}- Rental equipment holding charge: {public_rules['rental_daily_holding_cost']}.
- Idle rental equipment charge: {public_rules['rental_idle_daily_holding_cost']}.
- Advance reservations: {public_rules['advance_reservations']}.
- Normal-return cleaning charge: {public_rules['normal_return_cleaning_cost']}.
- Cleaning capacity: {json.dumps(public_rules['cleaning_capacity'], ensure_ascii=False, sort_keys=True)}.
- Damaged-equipment repair charge: {public_rules['damaged_asset_repair_cost']}.
- Customer deposit deduction for damage: {public_rules['damage_deposit_deduction']}.
- Supplier inventory: {public_rules['supplier_inventory']}.
- Supplier delivery: {public_rules['purchase_timeline']}.
- Normal returns: {public_rules['normal_return_timeline']}.
- Repairs: {public_rules['repair_timeline']}.
- Liquidation market depth: {json.dumps(public_rules['liquidation_dealer']['market_depth'], ensure_ascii=False, sort_keys=True)}.
- {usage_line}
- {value_label}: {public_rules['season_end_business_value']}.
- Final rental-equipment values:
{valuation_lines}
- Insolvency: {public_rules['insolvency_rule']}.

At the start of each business day you receive one compact opening report. It includes yesterday's operating outcome and a summary of successful {action_summary_scope} actions. If you left a handoff note when closing yesterday, it is available during the next business day only.

Use counter_log for natural customer requests and exact interactions from the two most recently completed days, demand_report for longer observable demand patterns, catalog_report for natural-language and specification-based supplier filtering, and exact offer IDs to connect assortment, holdings, and outcomes. Named memory documents persist across business days, while ordinary conversation from previous days is not kept in the active workspace. {tool_protocol} Call close_business_day when today's decisions are complete. Do not invent business records or claim to have used unavailable web access."""


def day_message(observation: dict[str, Any]) -> str:
    workspace_notice = ""
    if int(observation["day"]) > 1:
        workspace_notice = (
            " The active workspace contains only today's conversation. Existing persistent "
            "memory documents are unaffected; yesterday's authoritative outcome, actions, "
            "and any one-day handoff are included in the opening report."
        )
    return (
        f"Good morning. Day {observation['day']} ({observation['calendar']['date']}, {observation['calendar']['weekday']}) of the current outdoor season has started. "
        + workspace_notice
        + " Here is the opening business report:\n"
        + json.dumps(observation, ensure_ascii=False, sort_keys=True)
        + "\nReview the shop's obligations and records, make today's decisions, then call "
        "close_business_day."
    )


def continuation_message(reason: str) -> str:
    return (
        f"The current business day is still open because {reason}. Continue with a business "
        "tool request, and call close_business_day when today's work is complete."
    )


def generation_truncation_message(*, malformed_tool_names: list[str]) -> str:
    target = (
        " for " + ", ".join(malformed_tool_names)
        if malformed_tool_names
        else ""
    )
    return (
        "The previous model generation hit its output limit before completing a "
        f"business tool request{target}. No incomplete request was executed. Do not "
        "repeat the analysis. Resubmit only the compact JSON tool call needed for the "
        "next action (or at most five compact read-only calls), using exact IDs."
    )
