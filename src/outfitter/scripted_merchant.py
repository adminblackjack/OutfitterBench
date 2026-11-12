"""V32 public-field merchants with free in-process cross-day memory.

Business facts enter through public tools, but catalog/IDs/targets/caps persist
without memory tool calls. Use strict_visible_merchant for context-faithful
agent-visible experiments. Forecasts are policy approximations, not verdicts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
from decimal import Decimal, ROUND_HALF_UP
from itertools import islice, product
import math

POLICIES = ("stock_only", "calendar", "fixed_buffer", "decline_reservations")
CAPACITY_CALIBRATION_POLICIES = ("owned_calendar", "calendar_capital_guard")
SUPPORTED_POLICIES = (*POLICIES, *CAPACITY_CALIBRATION_POLICIES)
PROTOCOL = "v32-public-fields-free-cross-day-memory-scripted-merchant-v2"


def money(value):
    return float(Decimal(str(value)).quantize(Decimal(".01"), rounding=ROUND_HALF_UP))


def cleaning(offer, days):
    rule = offer.get("normal_return_cleaning", {})
    return (float(rule.get("one_day_work_points", 0)) * float(rule.get("cash_per_work_point", 0))
            * (1 + max(0, days-1)*float(rule.get("extra_work_fraction_per_additional_day", 0))))


def tariff(offer):
    cost = float(offer["supplier_unit_cost"])
    return money(max(.01, cost/10 + cleaning(offer, 1)/3 if offer["kind"] == "rental_asset" else cost*1.5))


def job(identifier, pickup, returned, lines):
    return {"id": identifier, "pickup": int(pickup), "return": int(returned), "lines": [
        {**line, "offer_id": line.get("selected_offer_id") or line.get("offer_id"),
         "asset_quantity": int(line.get("asset_quantity", line.get("quantity", 0))),
         "line_rent": float(line.get("customer_payable_rent", line.get("line_rent", line.get("locked_rent", 0))))}
        for line in lines if line.get("disposition") != "omit" and line.get("status") != "optional_credited"]}


def future_jobs(snapshot):
    return [job(r["commitment_id"], r["current_terms"]["pickup_day"], r["current_terms"]["return_day"],
                r["current_terms"]["locked_lines"]) for r in snapshot["commitments"]
            if r["current_terms"]["status"] in {"confirmed", "at_risk"}]


def slots(snapshot, buffer=0, purchases=None, rental_return_overrides=None):
    """Current asset identity is retained; overdue shipments receive no invented ETA."""
    today = snapshot["status"]["day"]
    rentals = {r["asset_id"]: r for r in snapshot["rentals"]}
    jobs = {j["asset_id"]: j for j in snapshot["jobs"]}
    result = defaultdict(list)
    for asset in snapshot["assets"]:
        if not asset.get("physically_held", True):
            continue
        # The general reservation baseline remains deliberately conservative:
        # supplier-trial assets may serve walk-ins, but this policy does not
        # promise them across dates without modelling their contract/deadline cap.
        if asset.get("ownership_type") == "supplier_trial":
            continue
        status, aid = asset["status"], asset["asset_id"]
        if status == "available":
            ready = today
        elif status == "rented" and aid in rentals:
            r = rentals[aid]
            returned = (rental_return_overrides or {}).get(aid,
                r.get("estimated_return_day") or r.get("approved_return_day") or r["return_day"])
            ready = max(today, int(returned)) + 1 + buffer
        elif status in {"cleaning", "repairing", "maintenance"}:
            j = jobs.get(aid, {})
            ready = j.get("projected_completion_day") or j.get("ready_day") or asset.get("ready_day")
            if ready is None:
                continue
        else:
            continue
        result[asset["offer_id"]].append((int(ready), aid))
    for order in snapshot["orders"]:
        eta = order.get("estimated_delivery_day")
        if eta is None or eta <= today:
            continue
        for line in order["items"]:
            oid = line["offer_id"]
            if snapshot["offers"][oid]["kind"] != "rental_asset":
                continue
            for unit in range(int(line["remaining_arrival_units"])):
                result[oid].append((eta+buffer, f"{order['order_id']}:{oid}:{unit}"))
    for oid, packs in (purchases or {}).items():
        offer = snapshot["offers"][oid]
        for unit in range(packs*int(offer["package_units"])):
            result[oid].append((today+int(offer["estimated_delivery_days"])+buffer, f"new:{oid}:{unit}"))
    return result


def project(snapshot, jobs, *, buffer=0, purchases=None, rental_return_overrides=None):
    """Serial whole-required-bundle allocator; optional credits don't consume partial bundles."""
    supply = slots(snapshot, buffer, purchases, rental_return_overrides)
    failures, missing, credits, assignments = [], [], 0., {}
    for contract in sorted(jobs, key=lambda j: (j["pickup"], j["id"])):
        tentative = copy.deepcopy(supply)
        selected, short, optional_credit = [], [], 0.
        for line in sorted(contract["lines"], key=lambda l: (not l["required"], l["line_id"])):
            oid, quantity = line["offer_id"], line["asset_quantity"]
            eligible = sorted(x for x in tentative[oid] if x[0] <= contract["pickup"])
            taken = eligible[:quantity]
            if len(taken) < quantity:
                if line["required"]:
                    short.append({"offer_id": oid, "units": quantity-len(taken), "pickup": contract["pickup"],
                                  "contract_id": contract["id"]})
                    # Consume available slots temporarily so repeated-SKU lines don't reuse them.
                else:
                    optional_credit += line["line_rent"]
                    continue
            for slot in taken:
                tentative[oid].remove(slot)
                selected.append({"asset_id": slot[1], "offer_id": oid, "line_id": line["line_id"],
                                 "pickup": contract["pickup"], "return": contract["return"]})
            tentative[oid].extend((contract["return"]+1+buffer, x[1]) for x in taken)
        if short:
            failures.append(contract["id"])
            missing.extend(short)
        else:
            supply = tentative
            assignments[contract["id"]] = selected
            credits += optional_credit
    return {"failures": failures, "missing": missing, "optional_credit": money(credits), "assignments": assignments}


def assess(snapshot, candidate, policy, *, exclude=None, allow_purchase=True):
    buffer = int(policy == "fixed_buffer")
    existing = [] if policy == "stock_only" else [j for j in future_jobs(snapshot) if j["id"] != exclude]
    baseline = project(snapshot, existing, buffer=buffer)
    combined = existing+[candidate]
    trial = project(snapshot, combined, buffer=buffer)
    wanted = Counter()
    policy_allows_purchase = policy != "owned_calendar"
    if allow_purchase and policy_allows_purchase:
        for row in trial["missing"]:
            if row["contract_id"] in baseline["failures"]:
                continue
            offer = snapshot["offers"][row["offer_id"]]
            if snapshot["status"]["day"]+offer["estimated_delivery_days"]+buffer <= row["pickup"]:
                wanted[row["offer_id"]] += row["units"]
    packs = {oid: math.ceil(q/snapshot["offers"][oid]["package_units"]) for oid,q in wanted.items()}
    if packs:
        trial = project(snapshot, combined, buffer=buffer, purchases=packs)
    cost = sum(snapshot["offers"][oid]["package_price"]*q for oid,q in packs.items())
    space = sum(snapshot["offers"][oid]["warehouse_capacity_units_per_asset"]
                *snapshot["offers"][oid]["package_units"]*q for oid,q in packs.items())
    status = snapshot["status"]
    spendable = status["cash"]-500-status["deposit_liability"]-status["unearned_revenue_liability"]
    delivered = {r["line_id"] for r in trial["assignments"].get(candidate["id"],[])}
    candidate_credit = sum(l["line_rent"] for l in candidate["lines"] if not l["required"] and l["line_id"] not in delivered)
    margin = sum(l["line_rent"]-l["asset_quantity"]*cleaning(snapshot["offers"][l["offer_id"]],
                  candidate["return"]-candidate["pickup"]) for l in candidate["lines"]
                 if l["required"] or l["line_id"] in delivered)
    # Conservatively debit any additional optional credit on OTHER commitments.
    margin -= max(0, trial["optional_credit"]-baseline["optional_credit"]-candidate_credit)
    # Declared bounded reinvestment heuristic, NOT a demand estimate or real payback claim.
    maximum_reuse_allowance = 1 if policy == "calendar_capital_guard" else 4
    uses = min(maximum_reuse_allowance, max(1, (status["decision_horizon_day"]-candidate["pickup"]+1)//30+1))
    score = margin-.45*cost/uses
    feasible = candidate["id"] not in trial["failures"] and not(set(trial["failures"])-set(baseline["failures"]))
    free = snapshot["warehouse"]["remaining_units"]
    cash_allows = cost <= max(0, spendable)
    capacity_allows = free is None or space <= free
    budget = snapshot.get("durable_capital") or snapshot.get("durable_budget") or {}
    budget_remaining = budget.get(
        "remaining_durable_capital_headroom",
        budget.get("remaining_durable_acquisition_budget"),
    )
    budget_allows = (
        budget_remaining is None or cost <= float(budget_remaining) + 1e-9
    )
    capital_guard_allows = score > 0
    okay = (
        feasible
        and cash_allows
        and capacity_allows
        and budget_allows
        and capital_guard_allows
    )
    return {"accept": okay, "score_proxy": money(score), "margin_proxy": money(margin), "procurement_cost": money(cost),
            "procurement_packs": packs, "procurement_capacity": space, "reuse_allowance": uses,
            "maximum_reuse_allowance": maximum_reuse_allowance,
            "policy_allows_purchase": policy_allows_purchase,
            "schedule_feasible": feasible, "cash_allows_procurement": cash_allows,
            "warehouse_remaining_units": free,
            "warehouse_allows_procurement": capacity_allows,
            "durable_acquisition_budget_remaining": budget_remaining,
            "durable_acquisition_budget_allows_procurement": budget_allows,
            "capital_guard_allows": capital_guard_allows,
            "predicted_failures": trial["failures"], "candidate": candidate}


class PublicMerchant:
    def __init__(self, policy, rpc, *, retail_enabled=True):
        if policy not in SUPPORTED_POLICIES:
            raise ValueError(f"policy must be one of {SUPPORTED_POLICIES}")
        self.policy, self.rpc = policy, rpc
        self.retail_enabled = bool(retail_enabled)
        self.offers, self.commitment_ids, self.retail_targets, self.caps = {}, set(), {}, {}
        self.decisions = []

    def call(self, name, args=None):
        result = self.rpc(name, args or {})
        if not result["ok"]:
            raise RuntimeError(f"scripted tool failed: {name}: {result['data']}")
        return result["data"]

    def pages(self, name, args, key):
        rows, cursor = [], 0
        while True:
            page = self.call(name, {**args, "cursor": cursor})
            rows.extend(page[key])
            if not page.get("has_more"):
                return rows
            next_cursor = page["next_cursor"]
            if next_cursor == cursor:
                raise RuntimeError("pagination did not advance")
            cursor = next_cursor

    def start(self):
        self.offers = {o["offer_id"]: o for o in self.pages("search_catalog", {"view":"offers", "limit":20}, "offers")}
        self.call("set_prices", {"prices": [{"offer_id": oid, "price": tariff(o)} for oid,o in sorted(self.offers.items())]})
        self.retail_targets = (
            {
                row["offer_id"]: row["on_hand_units"]
                for row in self.call("view_retail_inventory")["offers"]
            }
            if self.retail_enabled
            else {}
        )

    def snapshot(self):
        status = self.call("business_status")
        records, ids = [], sorted(self.commitment_ids)
        for offset in range(0,len(ids),3):
            records.extend(self.call("view_reservation", {"view":"detail", "identifiers":ids[offset:offset+3],
                "sections":["current_terms","fulfillment","risk","cancellation_terms"]})["records"])
        records = [r for r in records if r["current_terms"]["status"] in {"confirmed","active","at_risk"}]
        self.commitment_ids = {r["commitment_id"] for r in records}
        # Orders are time-partitioned through a public filter; don't silently truncate >50.
        orders = []
        def load_orders(first, last):
            page = self.call("purchase_order_status", {"status":"in_transit", "placed_day_from":first,
                "placed_day_to":last,"limit":50})
            if page["summary"]["matching_order_count"] > 50:
                if first == last:
                    raise RuntimeError("more than 50 orders in one day; refuse silent truncation")
                mid = (first+last)//2
                load_orders(first,mid); load_orders(mid+1,last)
            else:
                orders.extend(page["orders"])
        load_orders(1,status["day"])
        return {"status":status,"offers":self.offers,"commitments":records,"orders":orders,
                "assets":self.pages("view_rental_assets",{"view":"asset","limit":50},"assets"),
                "jobs":self.pages("view_rental_capacity",{"view":"service_jobs","limit":10},"jobs"),
                "rentals":self.call("view_active_rentals")["rentals"],
                "warehouse":self.call("view_warehouse_capacity",{"limit":100}),
                "durable_budget": status.get("durable_acquisition_budget"),
                "durable_capital": status.get("durable_capital")}

    def log(self, day, category, **fields):
        self.decisions.append({"day":day,"category":category,**fields})

    def purchase(self, packs):
        groups = defaultdict(list)
        for oid,n in sorted(packs.items()):
            if n:
                groups[self.offers[oid]["seller_id"]].append({"offer_id":oid,"packs":n})
        for seller,items in sorted(groups.items()):
            self.call("place_purchase_order",{"seller_id":seller,"items":items})

    def choose(self, snapshot, request, plans, *, exclude=None, allow_purchase=True):
        choices = []
        for plan, date in product(plans, request["pickup_date_options"]):
            proposal = job(exclude or request["request_id"], date["pickup_day"], date["scheduled_return_day"], plan["lines"])
            outcome = assess(snapshot,proposal,self.policy,exclude=exclude,allow_purchase=allow_purchase)
            choices.append({**outcome,"plan_id":plan["plan_id"],"pickup_day":date["pickup_day"]})
        return max(choices,key=lambda c:(c["accept"],c["score_proxy"],-c["procurement_cost"],-c["pickup_day"],c["plan_id"]))

    def quote_plan(self, snapshot, request):
        menus = []
        for line in request["lines"]:
            variants = []
            for terms in line["profile_terms"]:
                for oid in terms["frozen_member_offer_ids"]:
                    quantity = terms["asset_quantity"]
                    price = tariff(self.offers[oid])
                    variants.append({"line_id":line["line_id"],"offer_id":oid,"required":line["required"],
                        "asset_quantity":quantity,"promise_profile_id":terms["promise_profile_id"],
                        "daily_unit_price":price,"line_rent":money(price*quantity*line["duration_days"]),
                        "service_units":line["service_units"],"disposition":"quote"})
            available = Counter(a["offer_id"] for a in snapshot["assets"] if a["status"] == "available")
            variants.sort(key=lambda v:(max(0,v["asset_quantity"]-available[v["offer_id"]])
                *self.offers[v["offer_id"]]["supplier_unit_cost"],
                v["asset_quantity"]*self.offers[v["offer_id"]]["warehouse_capacity_units_per_asset"],
                v["offer_id"],v["promise_profile_id"]))
            choices = variants[:2]
            if not line["required"]:
                choices.append({"line_id":line["line_id"],"disposition":"omit","required":False})
            menus.append(choices)
        # Bounded deterministic candidate search, not a hidden-utility/menu oracle.
        plans = [{"plan_id":f"candidate-{i:03d}","lines":list(lines)} for i,lines in enumerate(islice(product(*menus),24))]
        return self.choose(snapshot,request,plans) if plans else None

    def respond_request(self, request):
        rid, fixed = request["request_id"], request["request_kind"] == "fixed_reward_offer"
        snapshot = self.snapshot()
        selected = None if self.policy == "decline_reservations" else (
            self.choose(snapshot,request,request["service_offer_terms"]["complete_plans"]) if fixed else self.quote_plan(snapshot,request))
        accept = selected is not None and selected["accept"]
        decision = {"request_id":rid,"action":"accept" if fixed and accept else "quote" if accept else "decline"}
        if accept and fixed:
            decision.update(plan_id=selected["plan_id"],pickup_day=selected["pickup_day"])
        if accept and not fixed:
            chosen = {l["line_id"]:l for l in selected["candidate"]["lines"]}
            decision["lines"] = []
            for line in request["lines"]:
                options = []
                if line["line_id"] in chosen:
                    value = chosen[line["line_id"]]
                    options.append({k:value[k] for k in ("disposition","offer_id","promise_profile_id","daily_unit_price")})
                if not line["required"]:
                    options.append({"disposition":"omit"})
                decision["lines"].append({"line_id":line["line_id"],"options":options})
        response = self.call("respond_reservation_service_offers" if fixed else "quote_reservation_inquiries",
                             {"decisions":[decision]})["decisions"][0]
        self.log(snapshot["status"]["day"],"request",request_id=rid,request_kind=request["request_kind"],
                 action=decision["action"],result_status=response["status"],assessment=selected)
        if response["status"] == "awaiting_manager_confirmation":
            selected = self.choose(self.snapshot(),request,response["customer_authorized_plans"])
            # A quote is binding even if the projection is pessimistic; explicitly select the least-bad legal terms.
            response = self.call("confirm_reservation_plans",{"selections":[{"request_id":rid,
                "plan_id":selected["plan_id"],"pickup_day":selected["pickup_day"]}]})["confirmations"][0]
        if response.get("commitment_id"):
            self.commitment_ids.add(response["commitment_id"])
            # No new customer or hidden acceptance query and no speculative purchase before acceptance.
            if selected and selected["accept"]:
                self.purchase(selected["procurement_packs"])

    def change(self, change):
        snapshot = self.snapshot()
        record = next(r for r in snapshot["commitments"] if r["commitment_id"] == change["commitment_id"])
        terms, cid = record["current_terms"], record["commitment_id"]
        if change["change_type"] == "reschedule":
            proposal = job(cid,change["proposed_pickup_day"],change["proposed_return_day"],terms["locked_lines"])
            delta = {r["line_id"]:r["additional_locked_rent"] for r in change.get("projected_line_rent_deltas",[])}
            for line in proposal["lines"]:
                line["line_rent"] += delta.get(line["line_id"],0)
            yes = assess(snapshot,proposal,self.policy,exclude=cid,allow_purchase=False)
            unavailable = change.get("original_dates_still_possible") is False
            old = assess(snapshot,job(cid,terms["pickup_day"],terms["return_day"],terms["locked_lines"]),
                         self.policy,exclude=cid,allow_purchase=False)
            accept = yes["accept"] and (unavailable or yes["score_proxy"] >= old["score_proxy"]-.01)
            detail = {"accept_assessment":yes,"decline_customer_cancels":unavailable}
        else:
            assigned = {a["asset_id"] for a in record["fulfillment"]["assignments"]}
            overrides = {aid:change["proposed_return_day"] for aid in assigned}
            buffer = int(self.policy == "fixed_buffer")
            future = [] if self.policy == "stock_only" else future_jobs(snapshot)
            no = project(snapshot,future,buffer=buffer)
            yes = project(snapshot,future,buffer=buffer,rental_return_overrides=overrides)
            incremental = float(change.get("projected_additional_rent_collected",0))
            delta_days = change["duration_delta_days"]
            for r in snapshot["rentals"]:
                if r["asset_id"] in assigned:
                    incremental -= cleaning(self.offers[r["offer_id"]],r["duration_days"]+delta_days)-cleaning(self.offers[r["offer_id"]],r["duration_days"])
            accept = not(set(yes["failures"])-set(no["failures"])) and incremental > 0
            detail = {"incremental_margin_proxy":money(incremental),"new_predicted_failures":sorted(set(yes["failures"])-set(no["failures"]))}
        self.call("respond_reservation_changes",{"decisions":[{"change_id":change["change_id"],"action":"accept" if accept else "decline"}]})
        self.log(snapshot["status"]["day"],"change",change_id=change["change_id"],change_type=change["change_type"],
                 action="accept" if accept else "decline",**detail)

    def risk(self, alert):
        cid = alert["commitment_id"]
        self.commitment_ids.add(cid)
        data = self.call("resolve_reservation_risks",{"actions":[{"commitment_id":cid,"action":"inquire_fallback"}]})["actions"][0]
        snapshot = self.snapshot()
        record = next(r for r in snapshot["commitments"] if r["commitment_id"] == cid)
        terms = record["current_terms"]
        request = {"request_id":record["request_id"],"pickup_date_options":[{
            "pickup_day":terms["pickup_day"],"scheduled_return_day":terms["return_day"]}]}
        plans = data.get("preauthorized_recovery_plans",[])
        recovered = False
        if plans and not record["risk"].get("recovery_path"):
            choice = self.choose(snapshot,request,plans,exclude=cid,allow_purchase=False)
            if choice["accept"]:
                self.call("resolve_reservation_risks",{"actions":[{"commitment_id":cid,
                    "action":"select_preauthorized_plan","plan_id":choice["plan_id"]}]})
                recovered = True
        if not recovered:
            # Line fallback is a separate authorized path, never mixed with dormant complete plans.
            proposal = job(cid,terms["pickup_day"],terms["return_day"],terms["locked_lines"])
            actions = []
            for line in proposal["lines"]:
                options = [v for v in data.get("authorized_line_fallbacks",[]) if v["line_id"] == line["line_id"]]
                candidates = [line["offer_id"]]+[v["offer_id"] for v in options]
                available = Counter(a["offer_id"] for a in snapshot["assets"] if a["status"] == "available")
                oid = max(candidates,key=lambda oid:(available[oid],oid))
                if oid != line["offer_id"]:
                    actions.append({"commitment_id":cid,"action":"select_authorized_fallback","line_id":line["line_id"],"offer_id":oid})
                    line["offer_id"] = oid
            decision = assess(snapshot,proposal,self.policy,exclude=cid,allow_purchase=False)
            if decision["accept"]:
                for action in actions:
                    self.call("resolve_reservation_risks",{"actions":[action]})
                recovered = True
        if not recovered:
            self.call("cancel_reservation_commitment",{"commitment_id":cid})
        self.log(snapshot["status"]["day"],"risk",commitment_id=cid,action="selected_recovery_or_original_ready" if recovered else "cancel")

    def upkeep(self):
        snapshot = self.snapshot()
        day, horizon = snapshot["status"]["day"],snapshot["status"]["decision_horizon_day"]
        spendable = snapshot["status"]["cash"]-500-snapshot["status"]["deposit_liability"]-snapshot["status"]["unearned_revenue_liability"]
        near = {l["offer_id"] for j in future_jobs(snapshot) if j["pickup"] <= day+2 for l in j["lines"]}
        for asset in snapshot["assets"]:
            cost = self.offers[asset["offer_id"]]["supplier_unit_cost"]
            if asset["status"] == "damaged":
                if asset["combined_wear"] >= .98 or horizon-day < 3:
                    self.call("sell_to_liquidation_dealer",{"rental_asset_ids":[asset["asset_id"]]})
                elif spendable >= cost:
                    self.call("repair_or_retire_assets",{"repair_asset_ids":[asset["asset_id"]]})
                    spendable -= cost  # Conservative bound, not asserted actual repair cost.
            elif (asset["status"] == "available" and asset.get("maintenance_window_open")
                  and asset["offer_id"] not in near and horizon-day >= 2 and spendable >= cost):
                self.call("schedule_maintenance",{"asset_ids":[asset["asset_id"]]})
                spendable -= cost
        if self.retail_enabled:
            for row in self.call("view_retail_inventory")["offers"]:
                offer = self.offers[row["offer_id"]]
                target = self.retail_targets.get(row["offer_id"],0)
                current = row["on_hand_units"]+row["in_transit_units"]
                if current < target/2 and horizon-day >= offer["estimated_delivery_days"]+3:
                    packs = math.ceil((target-current)/offer["package_units"])
                    cost = packs*offer["package_price"]
                    if cost <= spendable:
                        self.purchase({row["offer_id"]:packs})
                        spendable -= cost

    def protect(self):
        snapshot = self.snapshot()
        today, end = snapshot["status"]["day"],snapshot["status"]["runoff_end_day"]
        reserved = Counter()
        buffer = int(self.policy == "fixed_buffer")
        for contract in future_jobs(snapshot):
            if self.policy == "stock_only" and contract["pickup"] != today:
                continue
            for line in contract["lines"]:
                for day in range(max(today,contract["pickup"]-1-buffer),min(end+1,contract["return"]+1+buffer)):
                    reserved[(line["offer_id"],day)] += line["asset_quantity"]
        held = Counter(a["offer_id"] for a in snapshot["assets"] if a.get("physically_held",True)
                       and a["status"] not in {"damaged","retired","sold"})
        new = {key:max(0,held[key[0]]-quantity) for key,quantity in reserved.items()}
        updates = []
        for oid,day in sorted(set(self.caps)|set(new)):
            if day < today or self.caps.get((oid,day)) == new.get((oid,day)):
                continue
            item = {"offer_id":oid,"from_day":day,"to_day":day,"action":"set" if (oid,day) in new else "clear"}
            if item["action"] == "set":
                item["max_walk_in_occupied_units"] = new[(oid,day)]
            updates.append(item)
        for start in range(0,len(updates),30):
            self.call("set_walk_in_capacity_limits",{"items":updates[start:start+30]})
        self.caps = new

    def day(self):
        inbox = self.call("reservation_inbox")
        for change in inbox["changes"]:
            self.commitment_ids.add(change["commitment_id"])
            self.change(change)
        self.upkeep()
        for alert in inbox["risk_alerts"]:
            self.risk(alert)
        ids = sorted([r["request_id"] for r in inbox["requests"]+inbox.get("service_offers",[])])
        for rid in ids:
            workspace = self.call("view_reservation_quote_workspace",{"mode":"requests","request_ids":[rid]})
            self.respond_request(workspace["requests"][0])
        self.protect()
