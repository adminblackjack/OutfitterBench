"""V43 public business contract. No customer-model metadata is rendered here."""
from __future__ import annotations

from typing import Any
from .reservation_difficulty import TARIFF_RULE
from .service_guarantee import RULE as GUARANTEE_RULE
from .resale_disclosure import enabled as resale_disclosure_enabled, public_rule as resale_public_rule
from . import operating_history

ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v43-public-business-manual"
TOOL_INTERFACE_VERSION = 38
CORE_RELEASE_ID = "outfitter-v43-shared-public-business-manual-core"
POLICY_ID = "public-business-manual-v1"
TOPICS = {
    "finance": "Cash, charges, refunds, credit and durable capital",
    "reservations": "Quotes, commitments, changes and cancellation responsibility",
    "assets": "Availability, cleaning, repairs, maintenance and warehouse space",
    "procurement": "Purchases, delivery, supplier-owned trials and disposal",
    "scoring": "Final valuation, insolvency and automatic runoff",
    "workspace": "Tools, record units, dates, memory and interaction limits",
}


def public_terms(env: Any) -> dict[str, Any]:
    """Select only contractual parameters; never pass through a rules dict."""
    c, scales = env.config, env.lifecycle_scales
    settlement = env.dataset.metadata["reservation_failure_settlement"]
    resale_rule = resale_public_rule
    if operating_history.enabled(env.dataset.metadata):
        from .actor_contract_v53 import resale_rule
    v48 = env._v48_service_guarantee()
    manager, failure = ({}, {}) if v48 else (settlement["manager_cancellation"], settlement["pickup_failure"])
    return {
        "actor_contract": POLICY_ID,
        **({"operating_history_v53": True,
            "new_asset_residual_rate": scales.new_asset_residual_rate}
           if operating_history.enabled(env.dataset.metadata) else {}),
        **({'durable_resale_formula': resale_rule(
            scales.new_asset_residual_rate, scales.residual_wear_exponent)}
           if resale_disclosure_enabled(env.dataset.metadata) else {}),
        **({'service_guarantee_v48':True} if v48 else {}),
        **({'school_holiday_calendar': True} if env._v49_school_holiday() else {}),
        **({"return_process_enabled": True} if env._v46_return_dynamics() else {}),
        **({"autonomous_returns": True} if env._v47_autonomous_rentals() else {}),
        "retail_enabled": c.retail_enabled,
        "start_date": str(env._calendar(1)["date"]),
        "decision_horizon_day": c.max_days,
        "decision_end_date": str(env._calendar(c.max_days)["date"]),
        "runoff_end_day": c.max_days + c.runoff_days,
        "runoff_end_date": str(env._calendar(c.max_days + c.runoff_days)["date"]),
        "daily_fee": c.daily_fee,
        "prepayment_rate": c.reservation_prepayment_rate,
        "optional_credit_rate": c.optional_line_missing_credit_rate,
        "credit_limit": c.emergency_credit_limit,
        "credit_fee_rate": c.emergency_credit_fee_rate,
        "additional_headroom": c.durable_acquisition_budget,
        "warehouse_limit": c.warehouse_capacity_limit,
        "cleaning_capacity": env.rules.cleaning_capacity_points,
        "cleaning_cash_per_point": scales.cleaning_cash_per_work_point,
        "annual_capital_rate": scales.annual_capital_rate,
        "daily_space_rate": scales.daily_space_rate_per_storage_unit,
        "repair_minimum": scales.repair_minimum_cash,
        "repair_loss_rate": scales.repair_cash_per_replacement_condition_loss,
        "repair_restore_fraction": scales.repair_condition_restore_fraction,
        "customer_damage_share": scales.customer_damage_liability_share,
        "customer_damage_minimum": scales.customer_damage_minimum_deduction,
        "maintenance_minimum": scales.maintenance_minimum_cash,
        "maintenance_cash_rates": dict(scales.maintenance_cash_rate_by_severity_tier),
        "maintenance_durations": dict(scales.maintenance_duration_days_by_severity_tier),
        "trial_limit": c.supplier_trial_max_accepted,
        "trial_active_limit": c.supplier_trial_max_active,
        "trial_start_limit": c.supplier_trial_max_contract_starts,
        "trial_window": c.supplier_trial_window_days,
        "trial_cost_limit": c.supplier_trial_max_unit_cost,
        "trial_handling": c.supplier_trial_handling_fee,
        "trial_usage_multiplier": c.supplier_trial_usage_fee_multiplier,
        "retail_holding_rate": c.retail_holding_rate,
        "daily_model_rounds": c.max_iterations_per_day,
        "run_model_rounds": c.max_model_calls,
        "read_only_batch_size": c.max_read_only_batch_size,
        "parallel_tool_calls": c.parallel_tool_calls,
        "usage_charged_to_cash": c.charge_model_usage_to_cash,
        "input_token_cost": c.input_token_cost_per_million,
        "output_token_cost": c.output_token_cost_per_million,
        "manager_compensation": (
            "original affected first-day guarantee basis * (1 - 0.75 * min(max(N, 0)/7, 1))" if v48 else
            f"max({manager['locked_rent_rate']} * R * max({manager['rent_notice_floor']}, "
            f"1 - N/{manager['notice_decay_days']}), "
            f"({manager['service_base']} + {manager['service_unit_day_rate']} * U) * "
            f"max({manager['service_notice_floor']}, 1 - N/{manager['notice_decay_days']}))"
        ),
        "failure_compensation": (
            "original affected first-day guarantee basis" if v48 else
            f"max({failure['locked_rent_rate']} * R, "
            f"{failure['service_base']} + {failure['service_unit_day_rate']} * U)"
        ),
    }


def build_business_manual(env: Any) -> dict[str, dict[str, Any]]:
    t = public_terms(env)
    v45 = env._v45_reservation_difficulty()
    quoted_price_unit = "first-day" if v45 else "daily"
    line_price_unit = "whole-line first-day" if v45 else "daily"
    capital = (
        "No finite durable-capital limit applies."
        if t["additional_headroom"] is None else
        f"The fixed durable-capital limit is the initial owned fleet's historical purchase basis "
        f"plus {t['additional_headroom']:.2f} additional headroom. This headroom is not cash."
    )
    warehouse = (
        "No finite warehouse limit applies." if t["warehouse_limit"] is None else
        f"Warehouse capacity is {t['warehouse_limit']} space units."
    )
    finance = [
        "All money is CNY. Deposits enter cash and remain refundable liabilities. Reservation "
        f"prepayment is {t['prepayment_rate']:.0%} of locked rent, received at confirmation and "
        "recorded as unearned revenue until pickup. Pickup collects the remaining rent and "
        "security deposit. Changing posted prices does not change an existing contract.",
        f"The operating fee is {t['daily_fee']:.2f} per decision day, charged at close. It is not "
        "charged during runoff. Holding costs and already incurred service obligations continue.",
        f"Owned equipment pays replacement_cost * {t['annual_capital_rate']} / 365 per day, "
        f"using the opening holdings snapshot. On-site non-rented equipment also pays "
        f"storage_burden_units * {t['daily_space_rate']} per day. A sale later that day does not "
        "remove that day's opening charge. Supplier-owned trials have no capital holding charge.",
        capital + " Accepted unreceived owned purchases occupy committed capital. Delivery "
        "moves the same basis into deployed capital. Revenue, depreciation, damage and servicing "
        "do not increase the limit. Sale or completed disposal releases historical capital basis "
        "on the next business day. Sale proceeds enter cash immediately; unused headroom has no final value.",
        f"Refund protection provides at most {t['credit_limit']:.2f} principal, with a fee of "
        f"{t['credit_fee_rate']:.0%} on amounts drawn. It is drawn automatically for a cash "
        "shortfall on customer refunds or reservation compensation; it is not procurement "
        "credit. Principal and fees remain liabilities deducted from scored value.",
        "maximum_refunds_due_next_morning covers full deposits on rentals currently due to "
        "return next morning, before possible damage deductions. It excludes other future "
        "refunds or compensation. scheduled_cleaning_charges_due_next_morning covers only "
        "existing queued jobs projected to finish next morning; that morning's new returns "
        "have not entered this queue yet. Neither field is a total cash-obligation forecast.",
        "today_morning_flows is fixed after morning processing; later refunds appear under "
        "today_intraday_flows. Daily totals contain both. finance_history provides recorded "
        "cash flows, including refund-protection draws and fees.",
    ]
    if t["retail_enabled"]:
        finance.append(f"Retail inventory pays {t['retail_holding_rate']} times its positive "
                       "closing book cost each day. Retail revenue does not expand durable capital.")
    reservations = [
        "reservation_inbox lists today's inquiries, fixed offers, pending confirmations, "
        "customer changes, pickup risks and negotiation updates. The quote workspace provides "
        "request-specific permissible products, profiles and dates. Dates offered initially "
        "are the customer's choices; the manager cannot add another initial date.",
        "For price_inquiry, submit one binding menu or decline. Required lines each need one "
        f"or two distinct offer/Profile pairs with positive cent-exact per-asset {quoted_price_unit} prices. "
        "Optional lines accept zero to two priced choices; one omit choice is added automatically. "
        "Resolve every request line. Do not offer the same offer/Profile pair at multiple prices. "
        "The product of the numbers of choices, including each optional omit, must not exceed 24.",
        "One quoted menu and its prices apply to every advertised date. The response contains "
        "at most three acceptable complete plans, valid on every advertised date; these need "
        "not exhaust all acceptable combinations. One plan and one date confirm immediately. "
        "Otherwise select an authorized plan and date before close. Without that selection, "
        "the original date and customer-preferred plan confirm at close. A quote cannot be "
        "revised or freely withdrawn after customer authorization.",
        "For fixed_reward_offer, accept one entire offered plan and advertised date or decline. "
        f"Do not submit prices or mix parts of plans. Each line's {line_price_unit} rent is fixed regardless "
        "of the selected asset count. Acceptance immediately forms the contract without a second "
        "price negotiation. Unquoted inquiries and unaccepted fixed offers expire without a contract.",
        "Each line uses one exact offer/Profile pair, not a mixture. Asset quantity is "
        "ceil(service_units / capacity_per_asset); the quote workspace supplies this quantity. "
        "Confirmation locks the date, exact product configuration, price and deposit; physical "
        "assets are assigned at pickup. Customer authorization and successful confirmation do "
        "not certify inventory or scheduling feasibility. All required lines must be supplied "
        "together; only explicitly authorized substitutions may be selected. "
        + (
            "A selected optional line is also promised; shortage exposes a pickup risk. "
            "Partial delivery needs the customer's explicit agreement and separate manager selection."
            if v45 else
            f"An unavailable optional line receives {t['optional_credit_rate']:.0%} "
            "of its locked rent as credit."
        ),
        "A customer may request a reschedule or extension, or cancel a trip. The current request "
        "states its deadline and on_decline/on_expiry consequence. If original_dates_still_possible "
        "is false, rejecting or ignoring that reschedule causes a customer cancellation and "
        "prepayment refund without manager compensation. Accepting an extension immediately "
        "collects its disclosed extra rent for fulfilled lines. Refusing or ignoring an extension "
        "does not guarantee an on-time return; an estimated late return can still be recorded.",
        "Proactive contingency permits one valid submission per commitment for its entire "
        "lifetime. Rejection, expiry or a new commitment version does not restore it; invalid "
        "submissions do not consume it. Submit by pickup_day - 2 with the current version. "
        "Propose one or two distinct changes, with pickup 2–14 days after submission and "
        "unchanged duration. Equipment replacements must be exact offers within the locked "
        "Promise Profile. Rent, prepayment, deposits and the line set remain unchanged.",
        "Contingency replies arrive next morning using the forecast available at submission. "
        "Accepted proposals and a possible date-only counteroffer require explicit application "
        "on the response day. Without application the proposal expires and the current contract "
        "remains, unless the customer has cancelled. Customer cancellation terminates the "
        "contract automatically and is settled as manager cancellation when induced by this "
        "contact. Only current-version actionable reports can be applied. Customer agreement "
        "does not establish the shop's ability to supply the proposed dates or equipment.",
        "An unresolved pickup risk remains open for today's actions and is finalized at close, "
        "before walk-ins. Recovery inquiry returns authorized alternatives but does not select "
        "or reserve them. Explicitly select one complete-plan path or authorized line replacements. "
        "Complete plans cannot be mixed. A complete-plan recovery caps each line's rent and "
        "deposit at the lower of its original and newly selected authorized amounts. "
        "A selected recovery is still subject to physical "
        "availability at pickup; an unresolved failure incurs refund and compensation.",
        GUARANTEE_RULE if env._v48_service_guarantee() else f"Manager cancellation compensation is {t['manager_compensation']}. Pickup-failure "
        f"compensation is {t['failure_compensation']}. R is current locked total rent, N is "
        "nonnegative days until pickup, and U is required service units times rental days, "
        "summed over required lines. Compensation is rounded half-up to cents after taking "
        "the maximum. Refund of unearned prepayment is additional. view_reservation exposes "
        "current cancellation terms; cancel_reservation_commitment does not require a risk alert.",
        "Walk-in occupied-asset limits apply per exact offer on every occupied date, excluding "
        "reservation rentals. They do not cancel existing rentals or reserve individual assets. "
        "The default is unlimited. set overwrites the specified dates; clear removes limits. "
        "Ranges include both endpoints. No additional cleaning buffer is automatically added.",
        "A declined price inquiry may include customer-reported price_value or "
        "equipment_configuration concerns, possibly both; array order is not a ranking. "
        "other_or_unclear appears alone. An equipment concern may name one line. Reports may "
        "omit or misstate concerns and do not provide a complete rejection diagnosis.",
    ]
    if v45:
        reservations.insert(1, TARIFF_RULE)
        reservations.append(
            "Reduced pickup requires explicit customer consent and selection; credit the missing lines' "
            "full rent and compensate only their original locked first-day guarantee bases. "
            "Inquiry or selection pays nothing; actual reduced delivery settles the compensation. "
            "Only optional lines may be omitted. Inquiry is allowed once per commitment version on "
            "the pickup day; a reply requires explicit same-day selection. Rejection or no selection "
            "leaves the original obligations in force." if env._v48_service_guarantee() else
            "A reduced pickup credits the full locked rent of each omitted optional line. "
            f"Its separate compensation uses {t['failure_compensation']}, with R equal to "
            "the omitted lines' locked rent and U equal to those lines' service-unit-days. "
            "Only optional lines may be omitted. Inquiry is allowed once per commitment "
            "version on the pickup day; a reply requires explicit same-day selection. "
            "Rejection or no selection leaves the original obligations in force."
        )
    if env._v48_service_guarantee():
        reservations.append(
            "Historical preauthorized plans are not directly selectable before inquire_fallback. "
            "Read recovery_authorization.stage and available_actions for the current workflow. "
            "Inquiry then selection may be submitted in order in one atomic batch. "
            "Authorization and selection do not verify inventory. Current-version rescheduling refreshes "
            "recovery authorization; selected line replacements keep their existing recovery path.")
    if env._v51_recovery_integrity():
        reservations = [rule.replace('Inquiry then selection may be submitted in order in one atomic batch.',
            'Each customer inquiry must be submitted alone; read its result before selection in a later call.')
            for rule in reservations]
        reservations.append(
            'Historical quotation alternatives are not permission to omit an item already committed. '
            'Complete-plan recovery must retain all currently committed lines. To omit missing optional '
            'items, use inquire_partial_fulfillment, obtain acceptance, and select_partial_fulfillment; '
            'actual reduced delivery credits rent and pays the original guarantee. Refusal leaves the '
            'original obligations intact. An inquiry is recorded even when the customer refuses.')
    assets = [
        "offer_id identifies an exact supplier product; asset_id identifies one physical item. "
        "Available equipment can be assigned; rented, cleaning, damaged, repairing, maintenance "
        "and maintenance_hold equipment cannot start a rental. Disposed or sold equipment "
        "does not rejoin the fleet. Only owned equipment can be sold or retired by the manager.",
        f"Existing FIFO cleaning jobs receive up to {t['cleaning_capacity']} work points each "
        "morning before that morning's returns join the queue. Normal returned equipment "
        "therefore cannot be rented again on its return day. Jobs can take several mornings. "
        f"Cleaning costs {t['cleaning_cash_per_point']} per work point and is paid when the "
        "job finishes; view_rental_capacity service_jobs shows current jobs and charges.",
        f"A damaged owned asset may be repaired, sold at its damaged quote or retired. Repair "
        f"is prepaid and takes two days. The repair amount is max({t['repair_minimum']}, "
        f"replacement_cost * damage_condition_loss * {t['repair_loss_rate']}). Repair restores "
        f"{t['repair_restore_fraction']:.0%} of that damage loss, not all accumulated wear. "
        "Current asset details disclose the charge and resulting condition.",
        f"Customer damage deduction is min(contract deposit, max({t['customer_damage_minimum']}, "
        f"frozen damage repair amount * {t['customer_damage_share']})). Ordinary service-life "
        "retirement receives a full deposit refund. Deposit receipts are not repair revenue "
        "until the corresponding deduction is actually made.",
        "Preventive maintenance is available when the asset's published maintenance window "
        "opens. Ordinary overdue wear increases risk without itself forcing a hold. Unsafe "
        "condition can cause maintenance_hold. Maintenance provides renewed protection; it "
        "does not reset lifetime wear or restore resale condition. Asset details show "
        "eligibility, cost and duration. Cash rates by severity tier: "
        f"{t['maintenance_cash_rates']}; cost = max({t['maintenance_minimum']}, "
        "replacement_cost * tier rate). Durations in days: "
        f"{t['maintenance_durations']}. Scheduling prepays the charge and removes availability "
        "until the reported completion day.",
        warehouse + " Per-asset weights measure space consumed. All held equipment, including "
        "rented assets with reserved return space and supplier trials, plus accepted inbound "
        "units occupy warehouse capacity. Customer commitments do not add physical occupancy. "
        "Actual sale or disposal releases space; capital basis releases separately next day. "
        "Use view_warehouse_capacity for totals and weights.",
    ]
    if env._v48_service_guarantee():
        assets.append(
            "Each asset has repair and maintenance terms. maintenance_window_open only describes the wear "
            "window: rented, cleaning or repairing assets cannot be maintained now. eligible_now includes "
            "current operational conditions and single-asset cash affordability; a batch also needs enough "
            "cash for the sum. completion_day_if_started_today is conditional on starting today, not a "
            "promise of when an unavailable item can start. Fees are rounded per asset to cents before summing. "
            "Supplier-owned damaged trials can be repaired but cannot be retired or sold by the manager.")
    if env._v51_recovery_integrity():
        assets.append('Dealer quote queries return unavailable_items for ineligible holdings and quotes '
            'for the eligible subset. Totals cover only that subset; actual sale remains atomic and '
            'requires every submitted item to be sellable.')
    procurement = [
        "Purchase orders are prepaid. Each line is limited to 100 packs; each pack contains "
        "package_units individual units. Supplier unit cost and pack price are different units. "
        "Listed suppliers have no stock ceiling; purchases still require cash, durable capital "
        "where applicable, and warehouse space. Delivery does not collect a second purchase payment.",
        "Supplier listings give estimated delivery times, not guaranteed arrival dates. A new "
        "owned order is rejected if its estimated delivery exceeds the decision horizon. "
        "Previously accepted orders may be delayed into runoff. Deliveries arrive in morning processing.",
        f"Supplier trials permit {t['trial_limit']} accepted orders per season, with at most "
        f"{t['trial_active_limit']} not yet returned. Request exactly one single-unit durable "
        f"SKU with unit cost at most {t['trial_cost_limit']:.2f}, never previously owned, ordered "
        f"as owned, or trialed. Acceptance costs {t['trial_handling']:.2f} and consumes allowance "
        "even if the trial is returned early. A trial must have enough remaining season time "
        "for its latest permitted delivery and full trial window; order validation reports "
        "that boundary if the request is too late.",
        f"Each trial lasts {t['trial_window']} consecutive calendar days from delivery, "
        f"including idle and service days, and permits at most {t['trial_start_limit']} individual "
        "asset rental starts. The delivery day is usable; return_deadline is the exclusive "
        "new-start boundary. Each admitted rental must return by that deadline. A pending "
        "trial return stops further rental starts; it does not end an active customer rental.",
        f"Trial usage charge per asset rental is {t['trial_usage_multiplier']} * supplier_unit_cost "
        "* (rental_days / service_life_limit_days + 1 / service_life_limit_cycles). Extensions "
        "and late days add only the daily term. Catalog lifecycle profiles provide the life "
        "limits. Trials use warehouse space but no owned capital headroom and have zero "
        "terminal value; they cannot be sold or retired by the manager.",
        "return_supplier_trial_assets requests early return. Available items can leave "
        "immediately; rented or servicing items wait. Damaged trial items can trigger automatic "
        "paid repair before return. Returning an item does not restore the seasonal trial "
        "allowance; an active slot is freed only on physical return.",
        "The dealer buys owned available or damaged assets at current quotes. A completed "
        "sale removes the item permanently and credits cash immediately. The system does "
        "not sell assets automatically to meet a payment shortfall.",
    ]
    if t.get('durable_resale_formula'):
        procurement.append(t['durable_resale_formula'])
        assets.append(t['durable_resale_formula'])
    if t["retail_enabled"]:
        procurement.append("Retail packs contain sealed saleable consumer units. Retail replenishment "
                           "uses cash but no durable capital or warehouse units. Delivered units can "
                           "be sold to the dealer at 50% of their current average acquisition unit "
                           "cost, rounded per unit to cents; in-transit orders cannot be sold.")
    scoring = [
        "For a solvent shop, the official final score is cash plus final owned-asset net dealer "
        "value" + (" plus remaining retail inventory dealer value" if t["retail_enabled"] else "") +
        " minus refundable deposits, unearned reservation revenue and refund-protection credit "
        "liabilities. business_status accounting distinguishes these obligations from cash.",
        "Final dealer value uses each owned asset's state: available at its quote; damaged "
        "at its damaged quote; repairing at post-repair condition (repair is already prepaid); cleaning at its "
        "healthy quote before unpaid cleaning charges. Remaining cleaning charges are deducted "
        "once from the owned-equipment dealer pool, floored at zero. Post-season cleaning has "
        "no extra queue-capacity discount. Sold, retired and supplier-owned items add no final "
        "asset value; sale proceeds are already cash. In-transit purchases have no final value.",
        "Insolvency ends the run when a required payment cannot be met after any applicable "
        "refund-protection credit. The bankruptcy score is the negative recorded cash "
        "shortfall. A rejected discretionary purchase or repair is not itself insolvency.",
        f"Manager decisions, new demand and pickups end on Day {t['decision_horizon_day']}; "
        f"automatic runoff ends on Day {t['runoff_end_day']}. Returns, already scheduled "
        "services, previously accepted deliveries, refunds, holding costs and settlement "
        "continue during runoff. No new manager actions are available. Rentals and approved "
        "extensions cannot be scheduled beyond the runoff return boundary.",
        (f"Model usage is charged to cash at {t['input_token_cost']} per million input and "
         f"{t['output_token_cost']} per million output tokens." if t["usage_charged_to_cash"] else
         "Model usage is recorded separately and does not reduce shop cash or the official score."),
    ]
    if t.get('durable_resale_formula'):
        scoring.insert(1, t['durable_resale_formula'])
    workspace = [
        f"Each business day starts a fresh model session with at most {t['daily_model_rounds']} "
        "model responses. A response consumes one round even if it has no tool or produces "
        "an error. When batching is enabled, independent read-only requests can share a response; state changes, memory "
        "writes/deletions and close_business_day must each be the sole request in a response. "
        f"Read-only batches contain at most {t['read_only_batch_size']} tools when batching is "
        "enabled. Invalid batches execute nothing. first_only mode processes only the first tool.",
        f"The run has a total limit of {t['run_model_rounds']} model responses. The per-request "
        "budget notice includes the response you are about to produce. The daily limit executes "
        "the last response's permitted tools and then closes the day automatically if still "
        "open. The total limit pauses execution instead of silently completing the season.",
        "Business state and named memory documents persist; previous days' reasoning, ordinary "
        "assistant text and transcripts are not automatically included. Documents are retrieved "
        "through memory tools. A closing handoff is included on the next day only. Older "
        "same-day messages can leave the context window. Saved notes are your own records, "
        "not updates to customer contracts or operating rules.",
        "All dates are business-day indices. Rental occupancy is pickup_day through "
        "return_day - 1; scheduled return occurs on the morning of return_day. Capacity query ranges "
        "and capacity-limit ranges include both endpoints. weather_forecast horizon_days "
        "counts today, so 15 includes today and fourteen future days. A forecast is an "
        "estimate; later updates can differ.",
        "Use exact returned IDs. line_id is scoped by its request or commitment. A plan is "
        "an entire configuration; a Promise Profile states public service guarantees and "
        "capacity per asset. It is not permission to replace a locked exact offer. Multiple "
        "physical rentals can belong to one reservation commitment.",
        "counter_log retains exact walk-in interactions from the two most recently completed "
        "days. demand_report and other history tools provide their documented aggregates. "
        "Historical quotes preserve earlier terms; current_terms gives the current contract. "
        "Rule queries are read-only and may be batched; no separate fee is charged for them.",
    ]
    if t.get("operating_history_v53"):
        finance.append(operating_history.HOLDING_RULE)
        assets.extend([operating_history.HOLDING_RULE, operating_history.WEAR_RULE])
        workspace[-1] = operating_history.HISTORY_RULE
    return {key: {"title": TOPICS[key], "rules": rules} for key, rules in (
        ("finance", finance), ("reservations", reservations), ("assets", assets),
        ("procurement", procurement), ("scoring", scoring), ("workspace", workspace),
    )}


def core_system_prompt(t: dict[str, Any], *, tool_call_protocol: str,
                       read_only_tools: list[str], max_read_only_batch_size: int) -> str:
    if t.get("operating_history_v53"):
        from .actor_contract_v53 import system_prompt
        return system_prompt(t, tool_call_protocol=tool_call_protocol,
                             max_read_only_batch_size=max_read_only_batch_size)
    retail = " and travel-supply" if t["retail_enabled"] else ""
    return_process_note = (
        " Trip/transport descriptions and later return notices are informative, not guarantees. "
        "A reservation commitment's return_day remains the approved contractual deadline; "
        "estimated_return_day describes expected physical return and can be later even if no extension "
        "was requested. Notices do not amend the contract or commitment_version. Check reservation_inbox "
        "and the reservation rules for extension intent, warning and ETA timing."
        if t.get("return_process_enabled") else ""
    )
    if t.get("autonomous_returns"):
        from .autonomous_rentals import PUBLIC_RULES
        return_process_note=" "+PUBLIC_RULES
    if t.get('school_holiday_calendar'):
        from .school_holiday import PUBLIC_RULES as SCHOOL_RULES
        return_process_note += ' ' + SCHOOL_RULES
    retail_value = " + final retail inventory dealer value" if t["retail_enabled"] else ""
    retail_prices = " Retail prices are per saleable consumer unit." if t["retail_enabled"] else ""
    if tool_call_protocol == "read_only_batch_v1":
        batch = (f"You may batch at most {max_read_only_batch_size} independent read-only tools "
                 "in one response. State changes, memory writes/deletions and close_business_day "
                 "must each be the only tool request in a response. Invalid batches execute nothing. "
                 "Batchable tools: " + ", ".join(read_only_tools) + ".")
    elif tool_call_protocol == "first_only":
        batch = "Submit one tool request per response; if several are submitted, only the first executes."
    else:
        raise ValueError(f"unsupported tool-call protocol: {tool_call_protocol}")
    usage = ("Model usage charges are deducted from cash; see the scoring rules for rates."
             if t["usage_charged_to_cash"] else
             "Model usage is recorded separately and does not reduce cash or the official score.")
    optional_rule = (
        "Selected optional lines are binding. A missing line exposes pickup risk; "
        "reduced delivery requires the customer's agreement and explicit manager selection, "
        "with full missing-line rent credit and disclosed compensation."
        if t.get("reservation_difficulty_v45") else
        "Missing optional lines receive their specified credit."
    )
    price_rule = (
        "Posted walk-in rental prices and daily_unit_price in manager quotations are "
        "first-day prices per asset. Each additional day costs half that price, rounded "
        "half-up to cents per asset before multiplying by quantity. Fixed-reward offers "
        "specify a whole-line first-day rent; apply and round the half-price additional "
        "day amount per whole line, independent of asset quantity. Locked rent is the "
        "complete term amount. See reservation rules for formulas and historical field names."
        if t.get("reservation_difficulty_v45") else
        "Walk-in prices and manager-quoted reservation prices are per asset per day. "
        "Fixed-reward offers instead specify a whole-line daily rent, unchanged by the chosen asset quantity."
    )
    if t.get('durable_resale_formula'):
        usage += '\n\n' + t['durable_resale_formula']
    if t.get("reservation_close_outcomes_enabled"):
        usage += ("\n\nThe closing receipt's reservation_close_outcomes records actual "
                  "close-time reservation outcomes and refund/compensation payment status; "
                  "the next morning repeats these facts in yesterday. A handoff note "
                  "does not certify fulfillment or settlement.")
    if t.get('service_guarantee_v48'):
        usage += ("\n\nReservation guarantees lock original first-day amounts at confirmation. "
                  "Additional compensation is the affected original first-day amount times "
                  "(1 - 0.75 * min(max(days_until_pickup, 0) / 7, 1)); unearned-rent refunds are separate. "
                  "Same-day cancellation and pickup failure have equal terms. Recovery discounts or "
                  "contract versions do not reduce that guarantee. Consult current cancellation terms "
                  "and recovery_authorization; historical preauthorization does not skip inquiry.")
    return f"""You manage a seasonal outdoor-equipment rental{retail} shop in Beijing. All money is in CNY. Keep the shop solvent and maximize its official final value. Use the available business tools to make operational decisions. Detailed public terms are available through read_business_rules; omit topics for its directory or request several topics together.

The decision period runs from Day 1 ({t['start_date']}) through Day {t['decision_horizon_day']} ({t['decision_end_date']}). This is the last day for manager actions, new demand and pickups. Automatic runoff continues through Day {t['runoff_end_day']} ({t['runoff_end_date']}), when final value is scored. During runoff, returns, previously scheduled services and deliveries, refunds, holding costs and settlement continue without manager actions.

For a solvent shop, final value = cash + final owned-equipment net dealer value{retail_value} - refundable deposit liabilities - unearned reservation revenue - refund-protection credit liabilities. Final values follow the scoring rules, including unpaid cleaning costs. Supplier-owned trial assets and purchases still in transit have no final value. Insolvency ends the run with a score equal to the negative recorded payment shortfall. {usage}

Each day opens after automatic morning processing. Deliveries and existing service jobs are processed, customer rentals return and deposits are refunded, and ready equipment can fulfill today's reservations before you act. A pickup that cannot be fulfilled remains at risk for today's recovery actions. Closing the day resolves pending defaults and remaining pickup risks before walk-ins, charges daily costs and advances time. The operating fee is {t['daily_fee']:.2f} per decision day.

A rental occupies equipment from pickup_day through return_day - 1. Returns occur in morning processing. Today's returns join the cleaning queue after that morning's cleaning work, so normally returned equipment cannot be rented again on its return day. Equipment is unavailable while rented, cleaning, damaged, repairing or held for maintenance. Repair and preventive maintenance have different effects; asset records disclose their eligibility, costs and completion times. Scheduled returns must fit within the runoff boundary.{return_process_note}

An offer_id identifies an exact supplier product; an asset_id identifies one physical item. A commitment_id identifies an entire reservation, which can generate multiple physical rental contract_id records. A line_id belongs to its request or commitment. A Promise Profile specifies service guarantees and asset capacity. Use the exact IDs and quantities returned by the tools.

{price_rule}{retail_prices} Offers start without a posted walk-in price; a null price does not mean free. Post prices to serve walk-ins. Posted prices do not determine reservation quotations or change existing contracts. Deposits are prescribed by the offer or contract and cannot be edited.

There are two reservation request kinds. For price_inquiry, submit one menu of exact product/Profile/price options or decline. The menu applies to all advertised dates. Customers return up to three authorized complete plans, which may not exhaust all acceptable combinations. A single plan with a single date confirms immediately; otherwise select an authorized plan and date. For fixed_reward_offer, accept one entire offered plan and date or decline; do not submit prices or mix plans. Acceptance immediately creates the commitment.

| Pending item at close | Default consequence |
| --- | --- |
| Unquoted price inquiry or unaccepted fixed-reward offer | Expires without a commitment |
| Authorized quote awaiting manager selection | Confirms on the original date with the customer-preferred plan |
| Customer {'pre-pickup reschedule request' if t.get('autonomous_returns') else 'reschedule or extension request'} | {'Applies the disclosed rescheduling expiry consequence. Continued possession is not an approval request.' if t.get('autonomous_returns') else 'Applies its disclosed expiry consequence; refusing an extension does not guarantee an on-time return'} |
| Unresolved pickup risk | Final pickup attempt, then failure settlement if still unfulfilled |
| Actionable contingency proposal | Proposal expires; current contract remains unless the customer has cancelled |

Confirmation locks dates, exact product configuration, rent and deposits; physical assets are assigned at pickup. Authorization does not check store inventory or scheduling feasibility. All required lines must be fulfilled together. {optional_rule} Changes require the appropriate customer-authorization process. Closing an offer or limiting walk-ins does not cancel existing contracts.

reservation_inbox is the entry point for today's requests and updates. view_reservation_quote_workspace provides quotation choices and fixed offers; view_reservation provides current terms and history. Customer feedback may omit or misstate concerns; it is not a complete rejection diagnosis. Forecasts and customers' travel plans can change.

Before pickup, a confirmed commitment permits one valid proactive contingency submission in its entire lifetime, subject to its stated deadline and current version. Replies arrive next morning. Accepted modifications require apply_reservation_contingency on the response day and remain subject to store availability; a customer cancellation instead terminates and settles the contract automatically. Contact-induced cancellations use manager-cancellation compensation. Contract changes can invalidate earlier reports. See the reservation rules for proposal limits and other recovery paths.

Cash and resource limits are separate. Deposits and reservation prepayments increase cash but create obligations. Owned purchases require cash, durable-capital headroom and warehouse space. Capital headroom is not additional cash; selling an owned asset credits cash immediately but releases its historical capital basis next business day. Supplier trials are fee-paying, supplier-owned equipment with deadlines and return responsibilities; see procurement rules before ordering. The dealer does not automatically sell holdings to prevent insolvency. Applicable refund-protection credit and its fees are described in finance rules.

Every business day starts a fresh model session. Business state and named memory documents persist, but previous days' reasoning, ordinary assistant text and transcripts are not supplied. Retrieve notes through memory tools; they are not automatically loaded. A closing handoff lasts only until the next day. Older messages from today may leave the context window.

There are at most {t['daily_model_rounds']} model responses per day. Each response consumes a round, including responses without tools or with errors. A budget notice states remaining rounds, including the next response. At the daily limit, permitted tools from the last response execute before an automatic close if the day remains open. {batch} Call close_business_day when today's decisions are complete. The workspace provides business tools, not external web browsing.
"""
