"""Fixed, deliberately small V44 merchants. Facts enter only via public JSON RPC.

This is a retrospective adversarial diagnostic, not an independent test set or
an optimal merchant. No imports of the environment, dataset or customer kernel.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
from itertools import product
import math

from .scripted_merchant import cleaning, job, money


POLICIES = (
    "picnic_calendar", "core_myopic", "core_calendar", "core_no_extension",
    "core_optional_credit", "core_displace",
)
PROTOCOL = "v44-fixed-simple-policy-stress-v1"
# Insertion order is the declared purchase priority, not fitted during execution.
PRICES = {
    "offer-009": 3.5, "offer-026": 4., "offer-022": 5., "offer-056": 3.5,
    "offer-073": 5.5, "offer-051": 6., "offer-066": 3., "offer-030": 3.5,
    "offer-038": 3.5, "offer-020": 3., "offer-048": 7., "offer-064": 8.,
}
CORE_TARGETS = dict(zip(PRICES, (10, 4, 5, 3, 5, 2, 5, 3, 3, 3, 3, 2)))
PICNIC_TARGETS = {"offer-009": 24, "offer-026": 8, "offer-022": 8, "offer-056": 3}
READ_TOOLS = frozenset({
    "business_status", "search_catalog", "reservation_inbox", "view_reservation",
    "view_reservation_quote_workspace", "view_rental_assets", "view_rental_capacity",
    "view_active_rentals", "purchase_order_status", "view_liquidation_quotes",
    "weather_forecast",
})
WRITE_TOOLS = frozenset({
    "set_prices", "set_rental_offer_status", "place_purchase_order",
    "quote_reservation_inquiries", "respond_reservation_service_offers",
    "confirm_reservation_plans", "respond_reservation_changes",
    "resolve_reservation_risks", "cancel_reservation_commitment",
    "repair_or_retire_assets", "schedule_maintenance", "sell_to_liquidation_dealer",
    "set_walk_in_capacity_limits", "close_business_day",
})


def parameters():
    return {
        "protocol": PROTOCOL, "policies": list(POLICIES), "prices": PRICES,
        "core_targets": CORE_TARGETS, "picnic_targets": PICNIC_TARGETS,
        "cash_reserve_net_of_liabilities": 500, "stop_purchase_last_days": 30,
        "repair_wear_threshold": .9, "return_turnaround_buffer": 1,
        "candidate_limit_per_required_line": 2, "quote_combination_limit": 24,
        "calendar_window_dates": 15, "free_cross_day_state": False,
        "new_customer_quote_retries": 0, "supplier_trials": False,
        "proactive_contingency": False,
    }


def project(snapshot, jobs, *, strict_optional=True, overrides=None, myopic=False):
    """Deterministic greedy asset calendar, never a feasibility oracle.

Known service jobs have their published ready dates. Unknown future returns
get one day for cleaning. This approximation can fail under actual backlog,
damage or shipment delays; those failures remain real scored consequences.
"""
    today = snapshot["status"]["day"]
    rentals = {r["asset_id"]: r for r in snapshot["rentals"]}
    service = {r["asset_id"]: r for r in snapshot["jobs"]}
    supply = defaultdict(list)
    for a in snapshot["assets"]:
        if not a.get("physically_held", True) or a.get("ownership_type") != "owned":
            continue
        status, aid = a["status"], a["asset_id"]
        if status == "available":
            ready = today
        elif myopic:
            continue
        elif status == "rented" and aid in rentals:
            r = rentals[aid]
            ready = (overrides or {}).get(aid, r.get("estimated_return_day") or r["return_day"]) + 1
        elif status in {"cleaning", "repairing", "maintenance"}:
            s = service.get(aid, {})
            ready = s.get("projected_completion_day") or s.get("ready_day") or a.get("ready_day")
            if ready is None:
                continue
        else:
            continue
        supply[a["offer_id"]].append((int(ready), aid))
    if not myopic:
        for order in snapshot["orders"]:
            eta = order.get("estimated_delivery_day")
            if eta is None or eta <= today:
                continue  # An overdue estimate is not an invented delivery promise.
            for line in order["items"]:
                for unit in range(int(line["remaining_arrival_units"])):
                    supply[line["offer_id"]].append((int(eta), f"{order['order_id']}:{line['offer_id']}:{unit}"))
    failures, assignments, optional_missing = [], {}, {}
    for j in sorted(jobs, key=lambda x: (x["pickup"], x["id"])):
        tentative = copy.deepcopy(supply)
        selected, missing, short = [], [], False
        for line in sorted(j["lines"], key=lambda x: (not x["required"], x["line_id"])):
            oid, quantity = line["offer_id"], int(line["asset_quantity"])
            eligible = sorted(s for s in tentative[oid] if s[0] <= j["pickup"])
            if len(eligible) < quantity:
                if line["required"] or strict_optional:
                    short = True
                else:
                    missing.append(line["line_id"])
                    continue
            taken = eligible[:quantity]
            for slot in taken:
                tentative[oid].remove(slot)
                selected.append({"asset_id": slot[1], "offer_id": oid, "line_id": line["line_id"]})
            tentative[oid].extend((j["return"] + 1, s[1]) for s in taken)
        if short:
            failures.append(j["id"])
        else:
            supply = tentative
            assignments[j["id"]] = selected
            optional_missing[j["id"]] = missing
    return {"failures": failures, "assignments": assignments, "optional_missing": optional_missing}


def direct_margin(snapshot, candidate, allocation):
    delivered = {a["line_id"] for a in allocation["assignments"].get(candidate["id"], [])}
    return money(sum(
        line["line_rent"] - line["asset_quantity"] * cleaning(
            snapshot["offers"][line["offer_id"]], candidate["return"] - candidate["pickup"])
        for line in candidate["lines"] if line["line_id"] in delivered
    ))


def cancellation_cost(record):
    """Foregone future rent is economic loss; prepaid refunds aren't a second loss."""
    terms = record["cancellation_terms"]
    return float(terms["compensation"]), float(terms["total_cash_outflow"])


def displacement_choice(snapshot, future, overrides, extra_margin, records):
    """Greedy cancellation, no search over future demand or simulated outcomes."""
    original = project(snapshot, future)
    old_bad = set(original["failures"])
    remaining, cancelled, cost, cash_out = list(future), [], 0., 0.
    while True:
        extended = project(snapshot, remaining, overrides=overrides)
        bad = set(extended["failures"]) - old_bad
        if not bad:
            return {
                "accept": extra_margin > cost and cash_out <= spendable(snapshot["status"]),
                "cancel_ids": cancelled, "economic_loss": money(cost),
                "cash_required": money(cash_out), "extension_margin": money(extra_margin),
            }
        options = []
        for j in remaining:
            if j["id"] in bad:
                comp, cash = cancellation_cost(records[j["id"]])
                forgone = max(0., direct_margin(snapshot, j, original))
                options.append((comp + forgone, j["id"], cash))
        if not options:
            return {"accept": False, "cancel_ids": cancelled, "reason": "unresolved_projection"}
        loss, identifier, cash = min(options)
        cost += loss
        cash_out += cash
        cancelled.append(identifier)
        remaining = [j for j in remaining if j["id"] != identifier]


def spendable(status):
    return max(0., status["cash"] - status["deposit_liability"] - status["unearned_revenue_liability"] - 500.)


class SimpleMerchant:
    def __init__(self, policy, rpc):
        if policy not in POLICIES:
            raise ValueError(policy)
        self.policy, self.rpc = policy, rpc
        self.myopic = policy == "core_myopic"
        self.strict_optional = policy != "core_optional_credit"
        self.targets = PICNIC_TARGETS if policy == "picnic_calendar" else CORE_TARGETS
        self.prices = dict(PRICES)
        self.price_scale = 1.0
        self.rent_formula = lambda price, days, quantity: money(price * days * quantity)
        self.offers, self.records, self.decisions = {}, {}, []

    def call(self, name, args=None):
        result = self.rpc(name, args or {})
        if not result["ok"]:
            raise RuntimeError(f"{name}: {result['data']}")
        return result["data"]

    def many(self, queries):
        results = self.rpc.batch(queries)
        for (name, _args), result in zip(queries, results):
            if not result["ok"]:
                raise RuntimeError(f"{name}: {result['data']}")
        return [r["data"] for r in results]

    def pages(self, name, args, key):
        rows, cursor = [], 0
        while True:
            data = self.call(name, {**args, "cursor": cursor})
            rows.extend(data[key])
            if not data.get("has_more"):
                return rows
            next_cursor = data["next_cursor"]
            if next_cursor == cursor:
                raise RuntimeError("pagination did not advance")
            cursor = next_cursor

    def log(self, category, **values):
        self.decisions.append({"day": self.status["day"], "category": category, **values})

    def load_record(self, identifiers):
        queries = [("view_reservation", {"view": "detail", "identifiers": identifiers[i:i+3],
                    "sections": ["current_terms", "fulfillment", "risk", "cancellation_terms"]})
                   for i in range(0, len(identifiers), 3)]
        for data in self.many(queries):
            for row in data["records"]:
                if row.get("commitment_id"):
                    self.records[row["commitment_id"]] = row

    def physical(self):
        self.status, rented, self.inbox = self.many([
            ("business_status", {}), ("view_active_rentals", {}), ("reservation_inbox", {})])
        self.rentals = rented["rentals"]
        self.assets = self.pages("view_rental_assets", {"view": "asset", "limit": 50}, "assets")
        self.jobs = self.pages("view_rental_capacity", {"view": "service_jobs", "limit": 10}, "jobs")

    def load(self):
        self.offers = {r["offer_id"]: r for r in self.pages(
            "search_catalog", {"view": "offers", "limit": 20}, "offers")}
        self.physical()
        today = self.status["day"]
        self.timeline = self.call("view_rental_capacity", {
            "view": "timeline", "from_day": today,
            "to_day": min(self.status["runoff_end_day"], today + 14)})
        ids = {r["commitment_id"] for r in self.timeline["commitment_intervals"]}
        ids.update(r["commitment_id"] for r in self.inbox["changes"] + self.inbox["risk_alerts"])
        self.load_record(sorted(ids))
        self.orders = []
        def orders(first, last):
            data = self.call("purchase_order_status", {"status": "in_transit",
                "placed_day_from": first, "placed_day_to": last, "limit": 50})
            if data["summary"]["matching_order_count"] > 50:
                if first == last:
                    raise RuntimeError("cannot truncate same-day orders")
                middle = (first + last) // 2
                orders(first, middle)
                orders(middle + 1, last)
            else:
                self.orders.extend(data["orders"])
        orders(1, today)

    def snapshot(self):
        return {"status": self.status, "offers": self.offers, "assets": self.assets,
                "rentals": self.rentals, "jobs": self.jobs, "orders": self.orders}

    def future(self, exclude=None):
        if self.myopic:
            return []
        return [job(cid, r["current_terms"]["pickup_day"], r["current_terms"]["return_day"],
                    r["current_terms"]["locked_lines"])
                for cid, r in self.records.items() if cid != exclude
                and r["current_terms"]["status"] in {"confirmed", "at_risk"}]

    def assess(self, candidate, exclude=None):
        future = self.future(exclude)
        snapshot = self.snapshot()
        baseline = project(snapshot, future, strict_optional=self.strict_optional, myopic=self.myopic)
        trial = project(snapshot, future + [candidate], strict_optional=self.strict_optional, myopic=self.myopic)
        okay = candidate["id"] not in trial["failures"] and not (
            set(trial["failures"]) - set(baseline["failures"]))
        margin = direct_margin(snapshot, candidate, trial)
        return {"accept": okay and margin > 0, "margin": margin,
                "missing_optional": trial["optional_missing"].get(candidate["id"], []),
                "failures": trial["failures"]}

    def setup_prices(self):
        updates = [{"offer_id": oid, "price": self.prices[oid]} for oid in self.targets
                   if self.offers[oid].get("current_price") != self.prices[oid]]
        if updates:
            self.call("set_prices", {"prices": updates})
        switches = [{"offer_id": oid, "open": oid in self.targets}
                    for oid, row in self.offers.items()
                    if row.get("accepting_new_rentals") != (oid in self.targets)]
        if switches:
            self.call("set_rental_offer_status", {"items": switches})

    def upkeep(self):
        snapshot = self.snapshot()
        future = self.future()
        baseline = project(snapshot, future, strict_optional=self.strict_optional)
        budget = spendable(self.status)
        sell, repair, maintain = [], [], []
        for asset in sorted(self.assets, key=lambda x: x["asset_id"]):
            if not asset.get("physically_held", True):
                continue
            oid, aid, status = asset["offer_id"], asset["asset_id"], asset["status"]
            if oid not in self.targets and status in {"available", "damaged"}:
                sell.append(aid)
            elif status == "damaged":
                if asset["combined_wear"] >= .9:
                    sell.append(aid)
                elif budget >= self.offers[oid]["supplier_unit_cost"]:
                    repair.append(aid)
                    budget -= self.offers[oid]["supplier_unit_cost"]
            elif status == "available" and asset.get("maintenance_window_open"):
                if budget < self.offers[oid]["supplier_unit_cost"]:
                    continue
                altered = copy.deepcopy(snapshot)
                for row in altered["assets"]:
                    if row["asset_id"] in maintain + [aid]:
                        row["status"], row["ready_day"] = "maintenance", self.status["day"] + 2
                blocked = project(altered, future, strict_optional=self.strict_optional)
                if self.myopic or not (set(blocked["failures"]) - set(baseline["failures"])):
                    maintain.append(aid)
                    budget -= self.offers[oid]["supplier_unit_cost"]
        changed = False
        for ids, tool, key, batch in [
            (sell, "sell_to_liquidation_dealer", "rental_asset_ids", 50),
            (repair, "repair_or_retire_assets", "repair_asset_ids", 50),
            (maintain, "schedule_maintenance", "asset_ids", 50),
        ]:
            for i in range(0, len(ids), batch):
                self.call(tool, {key: ids[i:i+batch]})
                changed = True
        if changed:
            self.physical()

    def replenish(self):
        if self.status["decision_horizon_day"] - self.status["day"] < 30:
            return
        held = Counter(a["offer_id"] for a in self.assets if a.get("physically_held", True)
                       and a["status"] not in {"retired", "sold"})
        for order in self.orders:
            for item in order["items"]:
                held[item["offer_id"]] += item["remaining_arrival_units"]
        cash, capital = spendable(self.status), self.status["durable_capital"]["remaining_durable_capital_headroom"]
        space = self.status["warehouse_capacity"]["remaining_units"]
        purchases = defaultdict(list)
        for oid, target in self.targets.items():
            o = self.offers[oid]
            packs = 0
            size = int(o["package_units"])
            while held[oid] + size <= target:
                price = float(o["package_price"])
                weight = size * o["warehouse_capacity_units_per_asset"]
                if price > cash + 1e-9 or price > capital + 1e-9 or weight > space:
                    break
                held[oid] += size
                cash -= price
                capital -= price
                space -= weight
                packs += 1
            if packs:
                purchases[o["seller_id"]].append({"offer_id": oid, "packs": packs})
        for seller, items in purchases.items():
            self.call("place_purchase_order", {"seller_id": seller, "items": items})
        if purchases:
            data = self.call("purchase_order_status", {"status": "in_transit", "limit": 50})
            if data["summary"]["matching_order_count"] > 50:
                raise RuntimeError("unexpected >50 active orders")
            self.orders = data["orders"]
            self.status = self.call("business_status")

    def quote_options(self, request):
        menus = []
        for line in request["lines"]:
            if not line["required"] and self.strict_optional:
                menus.append([{"line_id": line["line_id"], "required": False, "disposition": "omit"}])
                continue
            by_offer = {}
            for term in sorted(line["profile_terms"], key=lambda t: t["promise_profile_id"]):
                for oid in term["frozen_member_offer_ids"]:
                    if oid not in self.targets:
                        if line["required"] or line["service_key"] not in {"rental:camp_lantern", "rental:camp_cookware"}:
                            continue
                        price = money((3. if line["service_key"] == "rental:camp_lantern" else 3.5)
                                      * self.price_scale)
                    else:
                        price = self.prices[oid]
                    qty = int(term["asset_quantity"])
                    variant = {"line_id": line["line_id"], "required": line["required"],
                        "offer_id": oid, "asset_quantity": qty, "promise_profile_id": term["promise_profile_id"],
                        "daily_unit_price": price,
                        "line_rent": self.rent_formula(price, line["duration_days"], qty),
                        "disposition": "quote"}
                    # Stable first public Profile per exact SKU, not a hidden
                    # taste/temperature oracle or a claim about Profile strength.
                    by_offer.setdefault(oid, variant)
            variants = sorted(by_offer.values(), key=lambda v: (v["daily_unit_price"] * v["asset_quantity"],
                              self.offers[v["offer_id"]]["supplier_unit_cost"], v["offer_id"]))
            legal = []
            for v in variants:
                if not line["required"]:
                    legal.append(v)
                else:
                    if all(self.assess(job(request["request_id"], date["pickup_day"],
                        date["scheduled_return_day"], [v]))["accept"] for date in request["pickup_date_options"]):
                        legal.append(v)
                if len(legal) == 2:
                    break
            if line["required"] and not legal:
                return None
            if not line["required"]:
                legal.append({"line_id": line["line_id"], "required": False, "disposition": "omit"})
            menus.append(legal)
        # If the full menu can bind an unsafe combination, narrow deterministically
        # by removing each second option from the last line first. No trial quotes.
        while True:
            combinations = list(product(*menus))
            safe = len(combinations) <= 24 and all(
                self.assess(job(request["request_id"], date["pickup_day"], date["scheduled_return_day"], lines))["accept"]
                for date, lines in product(request["pickup_date_options"], combinations))
            if safe:
                return menus
            reduced = False
            for index in reversed(range(len(menus))):
                quoted = [v for v in menus[index] if v["disposition"] != "omit"]
                if len(quoted) >= 2:
                    menus[index].remove(quoted[-1])
                    reduced = True
                    break
            if not reduced:
                return None

    def choose(self, request, plans, exclude=None):
        candidates = []
        for plan, date in product(plans, request["pickup_date_options"]):
            if any(line["required"] and line["offer_id"] not in self.targets for line in plan["lines"]):
                continue
            candidate = job(exclude or request["request_id"], date["pickup_day"], date["scheduled_return_day"], plan["lines"])
            assessment = self.assess(candidate, exclude)
            candidates.append({**assessment, "plan_id": plan["plan_id"], "pickup_day": date["pickup_day"],
                               "candidate": candidate})
        return max(candidates, key=lambda c: (c["accept"], c["margin"], -c["pickup_day"], c["plan_id"])) if candidates else None

    def request(self, request):
        fixed = request["request_kind"] == "fixed_reward_offer"
        selected = None
        decision = {"request_id": request["request_id"], "action": "decline"}
        if fixed:
            selected = self.choose(request, request["service_offer_terms"]["complete_plans"])
            if selected and selected["accept"]:
                decision.update(action="accept", plan_id=selected["plan_id"], pickup_day=selected["pickup_day"])
        else:
            menus = self.quote_options(request)
            if menus:
                decision.update(action="quote", lines=[{"line_id": line["line_id"], "options": [
                    {k: v[k] for k in ("disposition", "offer_id", "promise_profile_id", "daily_unit_price") if k in v}
                    for v in menu]} for line, menu in zip(request["lines"], menus)])
        result = self.call("respond_reservation_service_offers" if fixed else "quote_reservation_inquiries",
                           {"decisions": [decision]})["decisions"][0]
        self.log("request", request_id=request["request_id"], action=decision["action"],
                 status=result["status"], assessment=selected)
        if result["status"] == "awaiting_manager_confirmation":
            selected = self.choose(request, result["customer_authorized_plans"])
            if selected is None:
                raise RuntimeError("binding quote produced no selectable authorized plan")
            result = self.call("confirm_reservation_plans", {"selections": [{"request_id": request["request_id"],
                "plan_id": selected["plan_id"], "pickup_day": selected["pickup_day"]}]})["confirmations"][0]
        if result.get("commitment_id"):
            cid = result["commitment_id"]
            self.load_record([cid])
            rec = self.records[cid]
            t = rec["current_terms"]
            assessment = self.assess(job(cid, t["pickup_day"], t["return_day"], t["locked_lines"]), exclude=cid)
            self.log("confirmed", commitment_id=cid, request_id=request["request_id"],
                     planned_missing_optional=assessment["missing_optional"])
            self.status = self.call("business_status")

    def cancel(self, cid, reason):
        result = self.call("cancel_reservation_commitment", {"commitment_id": cid})
        self.log("cancel", commitment_id=cid, reason=reason, settlement=result)
        self.records.pop(cid, None)
        self.status = self.call("business_status")

    def change(self, change):
        cid = change["commitment_id"]
        record = self.records[cid]
        terms = record["current_terms"]
        detail, cancel_ids = {}, []
        if change["change_type"] == "reschedule":
            candidate = job(cid, change["proposed_pickup_day"], change["proposed_return_day"], terms["locked_lines"])
            detail = self.assess(candidate, cid)
            accept = detail["accept"]
        else:
            assigned = {r["asset_id"] for r in record["fulfillment"]["assignments"]}
            overrides = {aid: change["proposed_return_day"] for aid in assigned}
            incremental = float(change["projected_additional_rent_collected"])
            for rental in self.rentals:
                if rental["asset_id"] in assigned:
                    o = self.offers[rental["offer_id"]]
                    days = int(change["duration_delta_days"])
                    incremental -= cleaning(o, rental["duration_days"] + days) - cleaning(o, rental["duration_days"])
            future, snapshot = self.future(), self.snapshot()
            before = project(snapshot, future, strict_optional=self.strict_optional)
            after = project(snapshot, future, strict_optional=self.strict_optional, overrides=overrides)
            failures = sorted(set(after["failures"]) - set(before["failures"]))
            accept = incremental > 0 and not failures and self.policy != "core_no_extension"
            detail = {"incremental_margin_proxy": money(incremental), "new_failures": failures}
            if self.policy == "core_displace" and failures and incremental > 0:
                detail = displacement_choice(snapshot, future, overrides, incremental, self.records)
                accept = detail["accept"]
                cancel_ids = detail.get("cancel_ids", []) if accept else []
        self.call("respond_reservation_changes", {"decisions": [{"change_id": change["change_id"],
            "action": "accept" if accept else "decline"}]})
        self.log("change", change_id=change["change_id"], change_type=change["change_type"],
                 action="accept" if accept else "decline", diagnostic=detail)
        for identifier in cancel_ids:
            self.cancel(identifier, "profitable_extension_displacement")
        self.load_record([cid])
        self.physical()

    def risk(self, alert):
        cid = alert["commitment_id"]
        data = self.call("resolve_reservation_risks", {"actions": [{"commitment_id": cid, "action": "inquire_fallback"}]})["actions"][0]
        terms = self.records[cid]["current_terms"]
        recovery_path = self.records[cid].get("risk", {}).get("recovery_path")
        req = {"request_id": self.records[cid]["request_id"], "pickup_date_options": [
            {"pickup_day": terms["pickup_day"], "scheduled_return_day": terms["return_day"]}]}
        choice = self.choose(req, data.get("preauthorized_recovery_plans", []), exclude=cid) if not recovery_path else None
        if choice and choice["accept"]:
            self.call("resolve_reservation_risks", {"actions": [{"commitment_id": cid,
                "action": "select_preauthorized_plan", "plan_id": choice["plan_id"]}]})
            self.load_record([cid])
            return
        candidate = job(cid, terms["pickup_day"], terms["return_day"], terms["locked_lines"])
        available = Counter(a["offer_id"] for a in self.assets if a["status"] == "available")
        actions = []
        for line in candidate["lines"]:
            variants = [line["offer_id"]]
            if recovery_path in {None, "profile_line_fallback"}:
                variants += [x["offer_id"] for x in data.get("authorized_line_fallbacks", [])
                             if x["line_id"] == line["line_id"]]
            oid = max(variants, key=lambda o: (available[o], o))
            if oid != line["offer_id"]:
                line["offer_id"] = oid
                actions.append({"commitment_id": cid, "line_id": line["line_id"],
                                "action": "select_authorized_fallback", "offer_id": oid})
            available[oid] -= line["asset_quantity"]
        if self.assess(candidate, cid)["accept"]:
            if actions:
                self.call("resolve_reservation_risks", {"actions": actions})
            self.load_record([cid])
        else:
            self.cancel(cid, "unrecoverable_pickup_risk")

    def protect(self):
        today, end = self.status["day"], min(self.status["day"] + 14, self.status["decision_horizon_day"])
        if today > end:
            return
        held = Counter(a["offer_id"] for a in self.assets if a.get("physically_held", True)
                       and a["status"] not in {"retired", "sold", "damaged"})
        reserved = Counter()
        future = [job(cid, r["current_terms"]["pickup_day"], r["current_terms"]["return_day"], r["current_terms"]["locked_lines"])
                  for cid, r in self.records.items() if r["current_terms"]["status"] in {"confirmed", "at_risk"}]
        for j in future:
            if self.myopic and j["pickup"] != today:
                continue
            for l in j["lines"]:
                for day in range(max(today, j["pickup"] - 1), min(end + 1, j["return"] + 1)):
                    reserved[(l["offer_id"], day)] += l["asset_quantity"]
        previous = {(r["offer_id"], d): r["max_walk_in_occupied_units"]
                    for r in self.timeline["walk_in_capacity_limit_intervals"]
                    for d in range(max(today, r["from_day"]), min(end, r["to_day"]) + 1)}
        desired = {key: max(0, held[key[0]] - qty) for key, qty in reserved.items() if key[0] in self.targets}
        updates = []
        # Merge adjacent equal actions before submitting, so history never makes
        # the response or daily action count grow without bound.
        for oid in sorted({k[0] for k in set(previous) | set(desired)}):
            run = None
            for day in range(today, end + 1):
                key = (oid, day)
                if previous.get(key) == desired.get(key):
                    if run:
                        updates.append(run)
                        run = None
                    continue
                value = desired.get(key)
                action = "clear" if value is None else "set"
                if run and run["action"] == action and run.get("max_walk_in_occupied_units") == value and run["to_day"] == day - 1:
                    run["to_day"] = day
                else:
                    if run:
                        updates.append(run)
                    run = {"offer_id": oid, "from_day": day, "to_day": day, "action": action}
                    if value is not None:
                        run["max_walk_in_occupied_units"] = value
            if run:
                updates.append(run)
        for i in range(0, len(updates), 50):
            self.call("set_walk_in_capacity_limits", {"items": updates[i:i+50]})

    def day(self):
        self.load()
        self.setup_prices()
        for change in list(self.inbox["changes"]):
            if change["commitment_id"] in self.records:
                self.change(change)
        for alert in list(self.inbox["risk_alerts"]):
            if alert["commitment_id"] in self.records:
                self.risk(alert)
        self.upkeep()
        self.replenish()
        requests = sorted(self.inbox["requests"] + self.inbox.get("service_offers", []),
                          key=lambda r: (r.get("default_pickup_day", r["pickup_day"]), r["request_id"]))
        for r in requests:
            workspace = self.call("view_reservation_quote_workspace", {"mode": "requests",
                "request_ids": [r["request_id"]], "offer_projection": "references_only"})
            self.request(workspace["requests"][0])
        self.protect()
