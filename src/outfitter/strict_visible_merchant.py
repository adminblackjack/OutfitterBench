"""Daily-stateless V32 merchants using only actual agent tool results.

The driver must create a new instance every day.  No environment, state,
dataset, customer kernel, future realization, or evaluator object enters here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import product
import math

from .scripted_merchant import PublicMerchant, cleaning, future_jobs, job, money, project, tariff


POLICIES = (
    "visible_owned_myopic",
    "visible_owned_calendar",
    "visible_owned_buffer",
    "visible_observed_reinvest",
)
PROTOCOL = "v32-strict-daily-stateless-agent-visible-v1"
ALLOWED_FACT_TOOLS = frozenset({
    "business_status", "search_catalog", "reservation_inbox",
    "view_reservation_quote_workspace", "view_reservation",
    "view_rental_assets", "view_rental_capacity", "view_active_rentals",
    "purchase_order_status", "view_warehouse_capacity", "view_retail_inventory",
    "view_rental_activity", "view_liquidation_quotes",
})
ALLOWED_ACTION_TOOLS = frozenset({
    "set_prices", "quote_reservation_inquiries", "respond_reservation_service_offers",
    "confirm_reservation_plans", "respond_reservation_changes", "resolve_reservation_risks",
    "cancel_reservation_commitment", "place_purchase_order", "set_walk_in_capacity_limits",
    "sell_to_liquidation_dealer",
})


def _candidate_margin(snapshot, candidate, trial, baseline):
    delivered = {row["line_id"] for row in trial["assignments"].get(candidate["id"], [])}
    own_credit = sum(line["line_rent"] for line in candidate["lines"]
                     if not line["required"] and line["line_id"] not in delivered)
    margin = sum(
        line["line_rent"] - line["asset_quantity"] * cleaning(
            snapshot["offers"][line["offer_id"]], candidate["return"] - candidate["pickup"]
        )
        for line in candidate["lines"]
        if line["required"] or line["line_id"] in delivered
    )
    return margin - max(0, trial["optional_credit"] - baseline["optional_credit"] - own_credit)


def assess_visible(snapshot, candidate, policy, *, exclude=None, allow_purchase=True):
    """Public-evidence scheduling heuristic; never returns environment feasibility."""
    if policy not in POLICIES:
        raise ValueError("unknown strict-visible policy")
    schedule = "stock_only" if policy == "visible_owned_myopic" else "calendar"
    buffer = int(policy == "visible_owned_buffer")
    existing = [] if schedule == "stock_only" else [row for row in future_jobs(snapshot) if row["id"] != exclude]
    baseline = project(snapshot, existing, buffer=buffer)
    combined = existing + [candidate]
    trial = project(snapshot, combined, buffer=buffer)
    packs = {}
    evidence_used = {}
    if policy == "visible_observed_reinvest" and allow_purchase:
        wanted = Counter()
        for shortfall in trial["missing"]:
            if shortfall["contract_id"] == candidate["id"] and shortfall["contract_id"] not in baseline["failures"]:
                wanted[shortfall["offer_id"]] += shortfall["units"]
        proposed = {oid: math.ceil(units / snapshot["offers"][oid]["package_units"])
                    for oid, units in wanted.items()}
        for oid, count in proposed.items():
            offer = snapshot["offers"][oid]
            history = {"rental_started_count": 0, "realized_rent_less_normal_cleaning": 0.0,
                       "maximum_current_same_offer_dealer_quote": None,
                       **snapshot["observable_reinvestment_evidence"].get(oid, {})}
            units = count * offer["package_units"]
            quote = history.get("maximum_current_same_offer_dealer_quote")
            loss = None if quote is None else max(0.0, count * offer["package_price"] - units * quote)
            enough = (quote is not None and history.get("rental_started_count", 0) >= 2 * units
                      and history.get("realized_rent_less_normal_cleaning", 0.0) >= loss)
            evidence_used[oid] = {**history, "packs": count, "physical_units": units,
                "observable_capital_loss_proxy": None if loss is None else money(loss),
                "required_recent_rental_starts": 2 * units, "passes": enough}
            if enough and snapshot["status"]["day"] + offer["estimated_delivery_days"] <= candidate["pickup"]:
                packs[oid] = count
        # Never buy a partial required bundle when one proposed SKU lacks evidence.
        if set(packs) != set(proposed):
            packs = {}
        if packs:
            trial = project(snapshot, combined, buffer=buffer, purchases=packs)
    cost = sum(snapshot["offers"][oid]["package_price"] * count for oid, count in packs.items())
    space = sum(snapshot["offers"][oid]["warehouse_capacity_units_per_asset"]
                * snapshot["offers"][oid]["package_units"] * count for oid, count in packs.items())
    margin = _candidate_margin(snapshot, candidate, trial, baseline)
    obligations = snapshot["status"]["deposit_liability"] + snapshot["status"]["unearned_revenue_liability"]
    spendable = snapshot["status"]["cash"] - 500.0 - obligations
    failures = set(trial["failures"])
    feasible = candidate["id"] not in failures and not (failures - set(baseline["failures"]))
    free = snapshot["warehouse"]["remaining_units"]
    budget = snapshot.get("durable_capital") or snapshot.get("durable_budget") or {}
    budget_remaining = budget.get(
        "remaining_durable_capital_headroom",
        budget.get("remaining_durable_acquisition_budget"),
    )
    budget_allows = (
        budget_remaining is None or cost <= float(budget_remaining) + 1e-9
    )
    accepted = (
        feasible
        and margin > 0
        and cost <= max(0, spendable)
        and (free is None or space <= free)
        and budget_allows
    )
    return {"accept": accepted, "margin_proxy": money(margin), "procurement_cost": money(cost),
        "procurement_capacity": space, "procurement_packs": packs,
        "durable_acquisition_budget_remaining": budget_remaining,
        "durable_acquisition_budget_allows_procurement": budget_allows,
        "observable_reinvestment_evidence_used": evidence_used,
        "predicted_failures": sorted(failures), "candidate": candidate}


class StrictVisibleMerchant(PublicMerchant):
    """One-day policy instance.  The driver intentionally discards it after close."""
    def __init__(self, policy, rpc):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        mapped = "stock_only" if policy == "visible_owned_myopic" else (
            "fixed_buffer" if policy == "visible_owned_buffer" else "calendar")
        super().__init__(mapped, rpc)
        self.visible_policy = policy

    def call(self, name, args=None):
        if name not in ALLOWED_FACT_TOOLS | ALLOWED_ACTION_TOOLS:
            raise RuntimeError(f"strict-visible policy attempted unregistered tool: {name}")
        return super().call(name, args)

    def start(self):
        self.offers = {row["offer_id"]: row for row in self.pages(
            "search_catalog", {"view": "offers", "limit": 20}, "offers")}
        missing = [{"offer_id": oid, "price": tariff(row)} for oid, row in sorted(self.offers.items())
                   if row.get("current_price") is None]
        if missing:
            self.call("set_prices", {"prices": missing})

    def _orders(self, day):
        rows = []
        def load(first, last):
            page = self.call("purchase_order_status", {"status": "in_transit", "placed_day_from": first,
                "placed_day_to": last, "limit": 50})
            if page["summary"]["matching_order_count"] > 50:
                if first == last:
                    raise RuntimeError("refuse to truncate more than 50 same-day orders")
                middle = (first + last) // 2
                load(first, middle)
                load(middle + 1, last)
            else:
                rows.extend(page["orders"])
        load(1, day)
        return rows

    def _activity(self):
        rows, cursor = [], None
        while True:
            args = {"horizon_days": 30, "event_types": ["rental_started"], "limit": 50}
            if cursor is not None:
                args["cursor"] = cursor
            page = self.call("view_rental_activity", args)
            rows.extend(page["events"])
            if not page["has_more"]:
                return rows
            cursor = page["next_cursor"]

    def snapshot(self):
        status = self.call("business_status")
        today = status["day"]
        assets = self.pages("view_rental_assets", {"view": "asset", "limit": 50}, "assets")
        jobs = self.pages("view_rental_capacity", {"view": "service_jobs", "limit": 10}, "jobs")
        timeline = self.call("view_rental_capacity", {"view": "timeline", "from_day": today,
            "to_day": min(status["runoff_end_day"], today + 13)})
        identifiers = sorted({row["commitment_id"] for row in timeline["commitment_intervals"]})
        records = []
        for offset in range(0, len(identifiers), 3):
            records.extend(self.call("view_reservation", {"view": "detail", "identifiers": identifiers[offset:offset+3],
                "sections": ["current_terms", "fulfillment", "risk", "cancellation_terms"]})["records"])
        quotes = {}
        quotable = [row["asset_id"] for row in assets if row["status"] in {"available", "damaged"}]
        for offset in range(0, len(quotable), 25):
            page = self.call("view_liquidation_quotes", {"rental_asset_ids": quotable[offset:offset+25]})
            for row in page["rental_assets"]:
                quotes[row["asset_id"]] = row
        evidence = defaultdict(lambda: {"rental_started_count": 0, "realized_rent_less_normal_cleaning": 0.0,
                                        "maximum_current_same_offer_dealer_quote": None})
        if self.visible_policy == "visible_observed_reinvest":
            for event in self._activity():
                oid = event["offer_id"]
                evidence[oid]["rental_started_count"] += 1
                evidence[oid]["realized_rent_less_normal_cleaning"] += (
                    float(event["rent_paid"]) - cleaning(self.offers[oid], int(event["duration_days"])))
            for row in quotes.values():
                oid = row["offer_id"]
                previous = evidence[oid]["maximum_current_same_offer_dealer_quote"]
                evidence[oid]["maximum_current_same_offer_dealer_quote"] = max(
                    float(row["offer_price"]), float(previous) if previous is not None else 0.0)
            for value in evidence.values():
                value["realized_rent_less_normal_cleaning"] = money(value["realized_rent_less_normal_cleaning"])
        return {"status": status, "offers": self.offers,
            "commitments": [row for row in records if row["current_terms"]["status"] in {"confirmed", "active", "at_risk"}],
            "orders": self._orders(today), "assets": assets, "jobs": jobs,
            "rentals": self.call("view_active_rentals")["rentals"],
            "warehouse": self.call("view_warehouse_capacity", {"limit": 100}),
            "durable_budget": status.get("durable_acquisition_budget"),
            "durable_capital": status.get("durable_capital"),
            "timeline": timeline, "liquidation_quotes_by_asset": quotes,
            "observable_reinvestment_evidence": dict(evidence)}

    def choose(self, snapshot, request, plans, *, exclude=None, allow_purchase=True):
        choices = []
        for plan, date in product(plans, request["pickup_date_options"]):
            proposal = job(exclude or request["request_id"], date["pickup_day"], date["scheduled_return_day"], plan["lines"])
            result = assess_visible(snapshot, proposal, self.visible_policy, exclude=exclude,
                                    allow_purchase=allow_purchase)
            choices.append({**result, "plan_id": plan["plan_id"], "pickup_day": date["pickup_day"]})
        return max(choices, key=lambda row: (row["accept"], row["margin_proxy"],
            -row["procurement_cost"], -row["pickup_day"], row["plan_id"]))

    def upkeep(self):
        snapshot = self.snapshot()
        for asset_id, quote in sorted(snapshot["liquidation_quotes_by_asset"].items()):
            asset = next(row for row in snapshot["assets"] if row["asset_id"] == asset_id)
            if asset["status"] == "damaged" and float(quote["offer_price"]) > 0:
                self.call("sell_to_liquidation_dealer", {"rental_asset_ids": [asset_id]})

    @staticmethod
    def _expand_limits(rows, today, end):
        return {(row["offer_id"], day): int(row["max_walk_in_occupied_units"])
                for row in rows for day in range(max(today, row["from_day"]), min(end, row["to_day"]) + 1)}

    def protect(self):
        snapshot = self.snapshot()
        today = snapshot["status"]["day"]
        end = min(snapshot["status"]["runoff_end_day"], today + 13)
        reserved = Counter()
        buffer = int(self.visible_policy == "visible_owned_buffer")
        for contract in future_jobs(snapshot):
            if self.visible_policy == "visible_owned_myopic" and contract["pickup"] != today:
                continue
            for line in contract["lines"]:
                for day in range(max(today, contract["pickup"] - buffer), min(end + 1, contract["return"] + buffer)):
                    reserved[(line["offer_id"], day)] += line["asset_quantity"]
        held = Counter(row["offer_id"] for row in snapshot["assets"] if row.get("physically_held", True)
                       and row["status"] not in {"damaged", "retired", "sold"})
        desired = {key: max(0, held[key[0]] - quantity) for key, quantity in reserved.items()}
        current = self._expand_limits(snapshot["timeline"]["walk_in_capacity_limit_intervals"], today, end)
        changes = []
        for oid, day in sorted(set(current) | set(desired)):
            if current.get((oid, day)) == desired.get((oid, day)):
                continue
            row = {"action": "set" if (oid, day) in desired else "clear", "offer_id": oid,
                   "from_day": day, "to_day": day}
            if row["action"] == "set":
                row["max_walk_in_occupied_units"] = desired[(oid, day)]
            changes.append(row)
        for offset in range(0, len(changes), 50):
            self.call("set_walk_in_capacity_limits", {"items": changes[offset:offset+50]})

    def change(self, change):
        snapshot = self.snapshot()
        record = next(row for row in snapshot["commitments"] if row["commitment_id"] == change["commitment_id"])
        terms = record["current_terms"]
        cid = record["commitment_id"]
        if change["change_type"] == "reschedule":
            proposal = job(cid, change["proposed_pickup_day"], change["proposed_return_day"], terms["locked_lines"])
            deltas = {row["line_id"]: row["additional_locked_rent"]
                      for row in change.get("projected_line_rent_deltas", [])}
            for line in proposal["lines"]:
                line["line_rent"] += deltas.get(line["line_id"], 0)
            yes = assess_visible(snapshot, proposal, self.visible_policy, exclude=cid, allow_purchase=False)
            old = assess_visible(snapshot, job(cid, terms["pickup_day"], terms["return_day"], terms["locked_lines"]),
                                 self.visible_policy, exclude=cid, allow_purchase=False)
            customer_cancels = change.get("original_dates_still_possible") is False
            accept = yes["accept"] and (customer_cancels or yes["margin_proxy"] >= old["margin_proxy"] - .01)
            details = {"accept_assessment": yes, "decline_customer_cancels": customer_cancels}
        else:
            assigned = {row["asset_id"] for row in record["fulfillment"]["assignments"]}
            overrides = {asset_id: change["proposed_return_day"] for asset_id in assigned}
            buffer = int(self.visible_policy == "visible_owned_buffer")
            existing = [] if self.visible_policy == "visible_owned_myopic" else future_jobs(snapshot)
            no = project(snapshot, existing, buffer=buffer)
            yes = project(snapshot, existing, buffer=buffer, rental_return_overrides=overrides)
            incremental = float(change.get("projected_additional_rent_collected", 0))
            for rental in snapshot["rentals"]:
                if rental["asset_id"] in assigned:
                    old_days = int(rental["duration_days"])
                    new_days = old_days + int(change["duration_delta_days"])
                    incremental -= cleaning(self.offers[rental["offer_id"]], new_days) - cleaning(
                        self.offers[rental["offer_id"]], old_days)
            new_failures = sorted(set(yes["failures"]) - set(no["failures"]))
            accept = incremental > 0 and not new_failures
            details = {"incremental_margin_proxy": money(incremental), "new_predicted_failures": new_failures}
        action = "accept" if accept else "decline"
        self.call("respond_reservation_changes", {"decisions": [{"change_id": change["change_id"], "action": action}]})
        self.log(snapshot["status"]["day"], "change", change_id=change["change_id"],
                 change_type=change["change_type"], action=action, **details)

    def risk(self, alert):
        cid = alert["commitment_id"]
        revealed = self.call("resolve_reservation_risks", {"actions": [
            {"commitment_id": cid, "action": "inquire_fallback"}]})["actions"][0]
        snapshot = self.snapshot()
        record = next(row for row in snapshot["commitments"] if row["commitment_id"] == cid)
        terms = record["current_terms"]
        request = {"request_id": record["request_id"], "pickup_date_options": [{
            "pickup_day": terms["pickup_day"], "scheduled_return_day": terms["return_day"]}]}
        recovered = False
        plans = revealed.get("preauthorized_recovery_plans", [])
        if plans and not record["risk"].get("recovery_path"):
            choice = self.choose(snapshot, request, plans, exclude=cid, allow_purchase=False)
            if choice["accept"]:
                self.call("resolve_reservation_risks", {"actions": [{"commitment_id": cid,
                    "action": "select_preauthorized_plan", "plan_id": choice["plan_id"]}]})
                recovered = True
        if not recovered:
            proposal = job(cid, terms["pickup_day"], terms["return_day"], terms["locked_lines"])
            actions = []
            available = Counter(row["offer_id"] for row in snapshot["assets"] if row["status"] == "available")
            for line in proposal["lines"]:
                options = [row for row in revealed.get("authorized_line_fallbacks", [])
                           if row["line_id"] == line["line_id"]]
                candidates = [line["offer_id"], *[row["offer_id"] for row in options]]
                chosen = max(candidates, key=lambda oid: (available[oid], oid))
                if chosen != line["offer_id"]:
                    actions.append({"commitment_id": cid, "action": "select_authorized_fallback",
                                    "line_id": line["line_id"], "offer_id": chosen})
                    line["offer_id"] = chosen
            assessment = assess_visible(snapshot, proposal, self.visible_policy, exclude=cid, allow_purchase=False)
            if assessment["accept"]:
                for action in actions:
                    self.call("resolve_reservation_risks", {"actions": [action]})
                recovered = True
        if not recovered:
            self.call("cancel_reservation_commitment", {"commitment_id": cid})
        self.log(snapshot["status"]["day"], "risk", commitment_id=cid,
                 action="selected_recovery_or_original_ready" if recovered else "cancel")
