"""V53 public operating facts; no customer utility or evaluator projections."""
from __future__ import annotations

import copy

ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v53-operating-history"
CORE_RELEASE_ID = "outfitter-v53-operating-history-core-r1"
TOOL_INTERFACE_VERSION = 51
POLICY_ID = "operating-history-v1"
WEAR_RULE = (
    "combined_wear measures accumulated usage wear; higher values mean more wear. "
    "It is not the probability of damage or retirement on the next rental. "
    "Maintenance does not reduce this measure or restore consumed service life. "
    "Use current asset status and eligibility for operating decisions."
)
HISTORY_RULE = (
    "counter_log reads all completed-day walk-in records (default: latest seven completed days). "
    "Use view=records for original quotes and customer choices, or view=price_summary for "
    "observed quotes grouped by offer, actual unit price and duration. offer_ids matches quoted "
    "offers only; use service_keys for requests without quotes. No sample is not zero acceptance; "
    "declined_after_quote does not diagnose excessive price. Example: counter_log(view='price_summary', from_day=1, "
    "to_day=20, offer_ids=['offer-064']). service_history reports actual delivery and revenue, "
    "with channel=all|walk_in|reservation. Reservation quote terms remain in "
    "view_reservation(view='quote_history'); asset events remain in view_rental_activity. "
    "Keep the returned date range and filters fixed when following pagination cursors."
)
HOLDING_RULE = (
    "Offer and asset holding_costs are CNY per physical item per day: capital charge plus "
    "on-site storage charge. Storage billing units are NOT warehouse capacity units. "
    "Offer estimates use the current procurement price; owned assets use their recorded cost. "
    "Rented assets pay capital but no on-site storage charge. Supplier-owned trials pay no "
    "capital charge. Opening holdings determine that day's charge; a same-day sale does not "
    "refund it. Per-item estimates retain six decimals; the aggregated bill is rounded."
)


def enabled(metadata):
    from .recovery_state import ENVIRONMENT_PROTOCOL as V54_PROTOCOL
    return (metadata.get("environment_protocol") in {ENVIRONMENT_PROTOCOL, V54_PROTOCOL}
            and metadata.get("operating_history_policy") == POLICY_ID
            and int(metadata.get("tool_interface_version", 0)) >= TOOL_INTERFACE_VERSION)


def tool_parameters(name, original):
    result = copy.deepcopy(original)
    if name == "counter_log":
        result["properties"].update({
            "view": {"type": "string", "enum": ["records", "price_summary"], "default": "records"},
            "from_day": {"type": "integer", "minimum": 1},
            "to_day": {"type": "integer", "minimum": 1},
            "offer_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 20, "uniqueItems": True,
                          "description": "Match actual quoted offers, not requests with no quote."},
            "service_keys": {"type": "array", "items": {"type": "string"}, "maxItems": 20, "uniqueItems": True,
                             "description": "Public service categories, e.g. rental:tent; also matches unquoted requests."},
        })
        result["properties"]["day"]["description"] = "Single completed day; cannot combine with from_day/to_day."
        result.setdefault("allOf", []).append({"not": {"required": ["day"], "anyOf": [
            {"required": ["from_day"]}, {"required": ["to_day"]}]}})
    elif name == "service_history":
        result["properties"]["channel"] = {"type": "string", "enum": ["all", "walk_in", "reservation"], "default": "all"}
    return result


TOOL_DESCRIPTIONS = {
    "counter_log": "Read any completed-day walk-in records or price_summary; defaults to the latest seven completed days and ten rows. Filter by from_day/to_day or single day. offer_ids matches quoted offers only; service_keys also finds unquoted demand. Summaries cover all matches, not just the page. Historical terms are frozen; declined_after_quote does not mean price too high. Fix the returned date range when paginating.",
    "service_history": "Read actual delivery and revenue by channel=all (default), walk_in, or reservation. Initial rental, continuation rental and retail revenue are separate. Reservations count at pickup. Defaults to last 30 completed days; supports period=season_to_date and totals/daily/offer granularity. Use either period or horizon_days, never both. Full-period totals; at most 30 daily rows; offer view is paginated. For price exposure use counter_log; for reservation quotes use view_reservation quote_history.",
}


def holding_costs(env, offer_id, asset=None):
    from .lifecycle_economics import holding_terms
    cost = env.offers[offer_id].unit_cost if asset is None else asset.unit_cost
    units = float(env._lifecycle_projection(offer_id)["storage_burden_units"])
    terms = (holding_terms(replacement_cost=cost, storage_burden_units=units,
                           stored_on_site=True, scales=env.lifecycle_scales)
             if asset is None else env._lifecycle_holding_terms(asset, stored_on_site=True))
    capital = terms["capital_charge"] if asset is None or asset.ownership_type == "owned" else 0.0
    space = terms["space_charge"]
    active = bool(env.config.lifecycle_holding_enabled)
    if not active:
        capital = space = 0.0
    return {"currency": "CNY", "unit": "per_physical_item_per_day", "enabled": active,
            "cost_basis": "current_procurement_price" if asset is None else "recorded_asset_cost",
            "cost_basis_cny": cost, "storage_billing_units": units,
            "capital_charge": capital, "on_site_storage_charge": space,
            "total_when_on_site": round(capital + space, 6), "total_when_rented": capital}


QUOTE_FIELDS = ("physical_quantity", "accepted_physical_quantity", "duration_days",
                "pricing_basis", "quoted_total_rent", "quoted_total_retail_price",
                "quoted_total_deposit", "initial_rental_revenue", "retail_revenue", "contract_ids")


def freeze_quote(env, state, session, line):
    """Called when the interaction happens, never during historical queries."""
    from .reservation_difficulty import rental_amount
    result = copy.deepcopy(line)
    rental = env.offers[line["offer_id"]].asset_kind == "durable"
    quantity = int(line.get("asset_units", 0) if rental else line.get("offered_units", 0))
    duration = max(1, int(session.duration_days or 1)) if rental else None
    price = float(line["quoted_unit_price"])
    accepted = quantity if float(line.get("revenue", 0)) > 0 else 0
    result.update(physical_quantity=quantity, accepted_physical_quantity=accepted,
                  duration_days=duration, pricing_basis="first_day_per_asset" if rental else "retail_per_unit",
                  initial_rental_revenue=float(line.get("revenue", 0)) if rental else 0.0,
                  retail_revenue=float(line.get("revenue", 0)) if not rental else 0.0,
                  contract_ids=list(line.get("contract_ids", [])))
    if rental:
        result.update(quoted_total_rent=round(rental_amount(price, duration) * quantity, 2),
                      quoted_total_deposit=round(float(state.deposits[line["offer_id"]]) * quantity, 2))
    else:
        result.update(quoted_total_retail_price=round(price * quantity, 2), quoted_total_deposit=0.0)
    return result


PRICE_METRICS = ("quote_count", "accepted_quote_count", "quoted_physical_units",
                 "accepted_physical_units", "initial_rental_revenue", "retail_revenue")


def _totals(rows, fields):
    return {key: round(sum(row.get(key, 0) for row in rows), 6) for key in fields}


def counter_log(env, state, *, view="records", day=None, from_day=None, to_day=None,
                offer_ids=None, service_keys=None, cursor=0, limit=10):
    if view not in {"records", "price_summary"}:
        raise ValueError("view must be records or price_summary")
    if cursor < 0 or not 1 <= limit <= 50:
        raise ValueError("cursor must be non-negative and limit must be in [1, 50]")
    if day is not None and (from_day is not None or to_day is not None):
        raise ValueError("day cannot be combined with from_day or to_day")
    offers, services = set(offer_ids or []), set(service_keys or [])
    for values, known, name in ((offer_ids or [], env.offers, "offer_ids"),
                               (service_keys or [], {env.service_key_for_product_type(o.service_line)
                                                    for o in env.offers.values()}, "service_keys")):
        if len(values) > 20 or len(values) != len(set(values)) or set(values) - set(known):
            raise ValueError(f"{name} must contain at most 20 unique known values")
    completed = {int(row["day"]) for row in state.daily_history}
    last = max(completed, default=0)
    end = int(day if day is not None else to_day if to_day is not None else last)
    start = int(day if day is not None else from_day if from_day is not None else max(1, end - 6))
    if start < 1 or (end < start and (day is not None or from_day is not None or to_day is not None)):
        raise ValueError("date range must be positive and ordered")
    available = bool(last) and end <= last and all(d in completed for d in range(start, end + 1))
    selected = []
    if available:
        for report in state.daily_history:
            if not start <= int(report["day"]) <= end:
                continue
            for raw in report.get("counter_interactions", []):
                if services and raw["request"]["service_key"] not in services:
                    continue
                if offers and not any(line["offer_id"] in offers for line in raw.get("quote_lines", [])):
                    continue
                selected.append(raw)
    selected.sort(key=lambda row: (int(row["day"]), str(row["interaction_id"])))
    groups = {}
    for raw in selected:
        # Deduplicate at interaction/offer/price/duration level, not by physical units.
        per_interaction = {}
        for line in raw.get("quote_lines", []):
            if offers and line["offer_id"] not in offers:
                continue
            key = (line["offer_id"], line["quoted_unit_price"], line["duration_days"], line["pricing_basis"])
            target = per_interaction.setdefault(key, {field: 0 for field in PRICE_METRICS})
            target["quote_count"] = 1
            target["accepted_quote_count"] = max(target["accepted_quote_count"], int(line["accepted_physical_quantity"] > 0))
            for out, source in (("quoted_physical_units", "physical_quantity"),
                                ("accepted_physical_units", "accepted_physical_quantity"),
                                ("initial_rental_revenue", "initial_rental_revenue"),
                                ("retail_revenue", "retail_revenue")):
                target[out] += line[source]
        for key, values in per_interaction.items():
            target = groups.setdefault(key, {"offer_id": key[0], "quoted_unit_price": key[1],
                    "duration_days": key[2], "pricing_basis": key[3], **{f: 0 for f in PRICE_METRICS}})
            for field in PRICE_METRICS:
                target[field] = round(target[field] + values[field], 6)
    summary = [groups[key] for key in sorted(groups, key=lambda key: (key[0], float(key[1]), key[2] or 0, key[3]))]
    rows = [env._public_counter_interaction(raw) for raw in selected] if view == "records" else summary
    page = rows[cursor:cursor + limit]
    next_cursor = cursor + len(page) if cursor + len(page) < len(rows) else None
    empty_default = not last and day is None and from_day is None and to_day is None
    result = {"view": view, "available": available,
              "unavailable_reason": None if available else "no_completed_day_available" if not last else "requested_days_not_completed",
              "period": {"from_day": None if empty_default else start,
                         "to_day": None if empty_default else end, "last_completed_day": last},
              "filters": {"offer_ids": sorted(offers), "service_keys": sorted(services)},
              "totals": {"interaction_count": len(selected), **_totals(summary, PRICE_METRICS)},
              "interaction_count": len(selected), "record_count": len(rows),
              "returned_record_count": len(page), "cursor": cursor, "limit": limit,
              "next_cursor": next_cursor, "has_more": next_cursor is not None,
              "quantity_basis": "physical_quantity counts items; offered_units and response.accepted_units count service units, not people. duration_days is the rental term; rental_days is physical item-days.",
              "interpretation": "Quote counts are per interaction and offer; totals cover all matching records, not just the current page. Quote counts across offers are not customer counts."}
    result["interactions" if view == "records" else "price_groups"] = page
    if view == "records":
        result["returned_interaction_count"] = len(page)
    return result


SERVICE_FIELDS = ("fulfilled_units", "initial_rental_revenue", "retail_revenue",
                  "continuation_rental_revenue", "revenue", "rental_asset_days",
                  "continuation_rental_days", "retail_cogs")


def record_delivery(state, offer_id, channel, *, quantity=0, revenue=0.0, rental_days=0,
                    retail_cogs=0.0, rental=True, continuation=False):
    rows = state.current_day_ledger.setdefault("delivery_metrics", {})
    row = rows.setdefault(f"{channel}:{offer_id}", {"offer_id": offer_id, "channel": channel,
                         **{field: 0 for field in SERVICE_FIELDS}})
    row["fulfilled_units"] += quantity
    row["revenue"] = round(row["revenue"] + revenue, 2)
    key = "continuation_rental_revenue" if continuation else "initial_rental_revenue" if rental else "retail_revenue"
    row[key] = round(row[key] + revenue, 2)
    row["rental_asset_days"] += rental_days
    if continuation:
        row["continuation_rental_days"] += rental_days
    row["retail_cogs"] = round(row["retail_cogs"] + retail_cogs, 2)


def service_history(env, state, horizon_days, *, channel="all", offer_ids=None,
                    end_day=None, granularity="totals", cursor=0, limit=20):
    if channel not in {"all", "walk_in", "reservation"}:
        raise ValueError("channel must be all, walk_in, or reservation")
    if granularity not in {"totals", "daily", "offer"}:
        raise ValueError("granularity must be totals, daily, or offer")
    if cursor < 0 or not 1 <= limit <= 50:
        raise ValueError("invalid pagination")
    identifiers = set(offer_ids or [])
    if identifiers - set(env.offers):
        raise ValueError("unknown offer_ids")
    period, reports = env._history_window(state, horizon_days, end_day=end_day)
    by_day, all_rows = [], []
    for report in reports:
        rows = [row for row in report.get("delivery_metrics", {}).values()
                if (channel == "all" or channel == row["channel"])
                and (not identifiers or row["offer_id"] in identifiers)]
        all_rows.extend(rows)
        by_day.append({"day": report["day"], **_totals(rows, SERVICE_FIELDS)})
    result = {"channel": channel, "granularity": granularity, "period": period,
              "totals": _totals(all_rows, SERVICE_FIELDS),
              "units": "fulfilled_units are customer service units, not necessarily physical items; rental_asset_days are physical item-days",
              "revenue_basis": "Actual delivery and collected continuation rent, excluding deposits and reservation prepayments; no quote exposure statistics"}
    if granularity == "daily":
        result.update(days=by_day[-30:], daily_rows_truncated=len(by_day) > 30)
    elif granularity == "offer":
        rows = [{"offer_id": offer_id, "name": env.offers[offer_id].name,
                 **_totals([row for row in all_rows if row["offer_id"] == offer_id], SERVICE_FIELDS)}
                for offer_id in sorted(identifiers or set(env.offers))]
        page = rows[cursor:cursor + limit]
        next_cursor = cursor + len(page) if cursor + len(page) < len(rows) else None
        result.update(offers=page, total_offers=len(rows), returned_offer_count=len(page),
                      cursor=cursor, limit=limit, has_more=next_cursor is not None, next_cursor=next_cursor)
    return result
