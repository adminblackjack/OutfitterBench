from __future__ import annotations

from .common import (
    Any,
    RENTAL_ACTIVITY_EVENT_TYPES,
    RentalAsset,
    WorldState,
    _RNG_FORECAST,
    _catalog_match_score,
    defaultdict,
)
from ..supplier_trial import ACTIVITY_EVENT_TYPES as SUPPLIER_TRIAL_ACTIVITY_EVENT_TYPES
from .. import operating_history


class QueryMixin:
    def counter_log(
        self,
        state: WorldState,
        *,
        day: int | None = None,
        cursor: int = 0,
        limit: int = 10,
        view: str = "records",
        from_day: int | None = None,
        to_day: int | None = None,
        offer_ids: list[str] | None = None,
        service_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        if operating_history.enabled(self.dataset.metadata):
            return operating_history.counter_log(self, state, day=day, cursor=cursor, limit=limit,
                view=view, from_day=from_day, to_day=to_day, offer_ids=offer_ids, service_keys=service_keys)
        completed_days = sorted({int(row["day"]) for row in state.recent_counter_interactions})
        available = True
        unavailable_reason = None
        if day is not None:
            if int(day) not in completed_days:
                available = False
                unavailable_reason = (
                    "current_day_not_completed"
                    if int(day) >= state.day
                    else "completed_day_not_retained"
                )
                selected = []
            else:
                selected = [
                    row
                    for row in state.recent_counter_interactions
                    if int(row["day"]) == int(day)
                ]
        else:
            selected = list(state.recent_counter_interactions)
            if not completed_days:
                available = False
                unavailable_reason = "no_completed_day_available"
        page = selected[cursor : cursor + limit]
        next_cursor = cursor + len(page) if cursor + len(page) < len(selected) else None
        return {
            "available": available,
            "unavailable_reason": unavailable_reason,
            "retained_completed_days": completed_days,
            "interactions": [
                self._public_counter_interaction(row) for row in page
            ],
            "interaction_count": len(selected),
            "returned_interaction_count": len(page),
            "cursor": cursor,
            "limit": limit,
            "has_more": next_cursor is not None,
            "next_cursor": next_cursor,
        }

    def demand_report(
        self,
        state: WorldState,
        *,
        group_by: list[str],
        filters: dict[str, list[Any]] | None,
        start_day: int,
        end_day: int,
        cursor: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        dimensions = {
            "request_text",
            "rental_or_retail",
            "requested_size",
            "disposition",
            "day_type",
            "holiday_name",
            "weather",
            "visible_party_context",
            "day",
            "week",
        }
        if not 1 <= len(group_by) <= 3 or len(set(group_by)) != len(group_by):
            raise ValueError("group_by must contain 1-3 unique dimensions")
        unknown = sorted(set(group_by) - dimensions)
        if unknown:
            raise ValueError(f"unknown group_by dimensions: {unknown}")
        resolved_filters = dict(filters or {})
        unknown_filters = sorted(set(resolved_filters) - dimensions)
        if unknown_filters:
            raise ValueError(f"unknown filter dimensions: {unknown_filters}")
        rows = [
            dict(cube)
            for report in state.daily_history
            if start_day <= int(report["day"]) <= end_day
            for cube in report.get("demand_cube", [])
        ]
        for row in rows:
            row["week"] = (int(row["day"]) - 1) // 7 + 1
        for dimension, accepted in resolved_filters.items():
            if not isinstance(accepted, list) or not accepted:
                raise ValueError(f"filter {dimension} must be a non-empty list")
            known_values = {row.get(dimension) for row in rows}
            domain_name = "this period"
            invalid = sorted(
                (value for value in accepted if value not in known_values), key=str
            )
            if invalid and rows:
                raise ValueError(
                    f"filter {dimension} contains values outside {domain_name}: {invalid}"
                )
            rows = [row for row in rows if row.get(dimension) in accepted]
        metric_names = (
            "requested_units",
            "fulfilled_units",
            "unfulfilled_units",
            "interactions",
            "revenue",
            "retail_cogs",
            "rental_days",
        )
        grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
        for source in rows:
            key = tuple(source.get(dimension) for dimension in group_by)
            target = grouped.setdefault(
                key,
                {
                    **{dimension: source.get(dimension) for dimension in group_by},
                    **{metric: 0 for metric in metric_names},
                },
            )
            for metric in metric_names:
                target[metric] += source.get(metric, 0)
        result_rows = []
        for key in sorted(grouped, key=lambda values: tuple(map(str, values))):
            row = grouped[key]
            for metric in ("revenue", "retail_cogs"):
                row[metric] = round(float(row[metric]), 2)
            for metric in set(metric_names) - {"revenue", "retail_cogs"}:
                row[metric] = int(row[metric])
            result_rows.append(row)
        page = result_rows[cursor : cursor + limit]
        next_cursor = (
            cursor + len(page)
            if cursor + len(page) < len(result_rows)
            else None
        )
        return {
            "period": (
                {"start_day": start_day, "end_day": end_day}
                if end_day >= start_day
                else {"start_day": None, "end_day": None}
            ),
            "group_by": list(group_by),
            "filters": resolved_filters,
            "cells": page,
            "cell_count": len(result_rows),
            "returned_cell_count": len(page),
            "cursor": cursor,
            "limit": limit,
            "has_more": next_cursor is not None,
            "next_cursor": next_cursor,
        }

    def catalog_report(
        self,
        *,
        filters: dict[str, Any] | None = None,
        attribute_filters: list[dict[str, Any]] | None = None,
        offset: int = 0,
        limit: int = 10,
    ) -> dict[str, Any]:
        filters = dict(filters or {})
        allowed = {"query", "kind"}
        unknown = sorted(set(filters) - allowed)
        if unknown:
            raise ValueError(f"unknown catalog filters: {unknown}")
        scored_rows: list[tuple[tuple[int, int, int, int, int], dict[str, Any]]] = []
        for offer in sorted(self.dataset.offers, key=lambda value: value.offer_id):
            public = self.public_offer(offer.offer_id)
            if filters.get("kind") and public["kind"] != filters["kind"]:
                continue
            score = _catalog_match_score(public, filters.get("query"))
            if score is None:
                continue
            matched = True
            specifications = {
                str(row["name"]): row.get("value")
                for row in public["specifications"]
            }
            for condition in attribute_filters or []:
                name = str(condition.get("name") or "")
                operator = str(condition.get("op") or "eq")
                expected = condition.get("value")
                actual = specifications.get(name)
                if name not in specifications:
                    matched = False
                    break
                if operator == "eq":
                    ok = actual == expected
                elif operator == "lte":
                    expected_number = float(expected)
                    normalized = offer.decision_attributes.get(name)
                    try:
                        actual_number = float(normalized)
                    except (TypeError, ValueError):
                        try:
                            actual_number = float(actual)
                        except (TypeError, ValueError):
                            # Mixed public source specifications legitimately
                            # include ranges such as "4-6".  One such row must
                            # not abort the complete catalog query.
                            ok = False
                        else:
                            ok = actual_number <= expected_number
                    else:
                        ok = actual_number <= expected_number
                elif operator == "gte":
                    expected_number = float(expected)
                    normalized = offer.decision_attributes.get(name)
                    try:
                        actual_number = float(normalized)
                    except (TypeError, ValueError):
                        try:
                            actual_number = float(actual)
                        except (TypeError, ValueError):
                            ok = False
                        else:
                            ok = actual_number >= expected_number
                    else:
                        ok = actual_number >= expected_number
                elif operator == "contains":
                    ok = expected in actual
                else:
                    raise ValueError(f"unsupported specification filter op: {operator}")
                if not ok:
                    matched = False
                    break
            if matched:
                scored_rows.append((score, public))
        scored_rows.sort(
            key=lambda item: (
                tuple(-value for value in item[0]),
                str(item[1]["offer_id"]),
            )
        )
        rows = [public for _score, public in scored_rows]
        page = rows[offset : offset + limit]
        next_offset = offset + len(page) if offset + len(page) < len(rows) else None
        return {
            "offers": page,
            "matched_offer_count": len(rows),
            "returned_offer_count": len(page),
            "offset": offset,
            "next_offset": next_offset,
            "cursor": offset,
            "limit": limit,
            "has_more": next_offset is not None,
            "next_cursor": next_offset,
        }

    def business_history(
        self,
        state: WorldState,
        horizon_days: int,
        *,
        end_day: int | None = None,
        granularity: str = "totals",
    ) -> dict[str, Any]:
        if granularity not in {"totals", "daily"}:
            raise ValueError("granularity must be totals or daily")
        period, reports = self._history_window(state, horizon_days, end_day=end_day)
        days = []
        for report in reports:
            row = self._public_daily_report(report)
            row.pop("agent_actions", None)
            days.append(row)
        total_fields = (
            "visitor_count",
            "completed_transactions",
            "rental_revenue",
            "retail_revenue",
            "total_revenue",
            "asset_sale_proceeds",
            "assets_sold",
            "retail_liquidation_proceeds",
            "retail_liquidation_book_cost",
            "retail_liquidation_loss",
            "retail_units_liquidated",
            "requested_units",
            "fulfilled_units",
            "requested_line_items",
            "fulfilled_line_items",
            "partial_line_items",
            "unfilled_line_items",
            "quote_declined_units",
            "stockout_request_units",
            "closed_service_request_units",
            "season_end_declined_units",
        )
        totals = {
            key: round(sum(float(row[key]) for row in days), 6)
            for key in total_fields
        }
        for key in (
            "visitor_count",
            "completed_transactions",
            "assets_sold",
            "retail_units_liquidated",
            "requested_units",
            "fulfilled_units",
            "requested_line_items",
            "fulfilled_line_items",
            "partial_line_items",
            "unfilled_line_items",
            "quote_declined_units",
            "stockout_request_units",
            "closed_service_request_units",
            "season_end_declined_units",
        ):
            totals[key] = int(totals[key])
        result: dict[str, Any] = {
            "view": "summary",
            "granularity": granularity,
            "period": period,
            "totals": totals,
        }
        if granularity == "daily":
            result["days"] = days[-30:]
            result["daily_rows_truncated"] = len(days) > 30
        return result

    def business_actions(self, state: WorldState, day: int) -> dict[str, Any]:
        if day < 1:
            raise ValueError("day must be positive")
        if day > state.day:
            raise ValueError(f"day {day} is in the future; current day is {state.day}")
        report = next(
            (row for row in state.daily_history if int(row["day"]) == day),
            None,
        )
        if report is not None:
            status = "completed"
            raw_actions = report.get("agent_actions")
        elif day == state.day and not state.done:
            status = "in_progress"
            raw_actions = state.current_day_ledger.get("agent_actions")
        else:
            raise ValueError(f"no business action record is available for day {day}")
        return {
            "view": "actions",
            "day": day,
            "day_status": status,
            "actions": self._public_agent_actions(raw_actions),
        }

    def finance_history(
        self,
        state: WorldState,
        horizon_days: int,
        *,
        end_day: int | None = None,
        granularity: str = "totals",
    ) -> dict[str, Any]:
        if granularity not in {"totals", "daily"}:
            raise ValueError("granularity must be totals or daily")
        period, reports = self._history_window(state, horizon_days, end_day=end_day)
        finance_fields = (
            ("rental_revenue", "rental_revenue"),
            ("retail_revenue", "retail_revenue"),
            ("asset_sale_proceeds", "asset_sale_proceeds"),
            ("assets_sold", "assets_sold"),
            ("retail_liquidation_proceeds", "retail_liquidation_proceeds"),
            ("retail_liquidation_book_cost", "retail_liquidation_book_cost"),
            ("retail_liquidation_loss", "retail_liquidation_loss"),
            ("retail_units_liquidated", "retail_units_liquidated"),
            ("deposits_collected", "deposits_collected"),
            ("refunds", "refunds"),
            ("damage_deductions", "damage_deductions"),
            ("purchase_cost", "purchase_cost"),
            ("cleaning_cost", "cleaning_cost"),
            ("repair_cost", "repair_cost"),
            *(
                (("maintenance_cost", "maintenance_cost"),)
                if self.config.preventive_maintenance_enabled
                else ()
            ),
            ("retail_holding_cost", "retail_holding_cost"),
            ("rental_holding_cost", "rental_holding_cost"),
            (
                "opening_rental_holding_cost_basis",
                "opening_rental_holding_cost_basis",
            ),
            ("operating_fee", "operating_fee"),
            ("management_service_cost", "model_cost"),
        ) + (
            (
                (
                    "reservation_prepayments_collected",
                    "reservation_prepayments_collected",
                ),
                (
                    "reservation_prepayment_refunds",
                    "reservation_prepayment_refunds",
                ),
                (
                    "reservation_prepayment_recognized",
                    "reservation_prepayment_recognized",
                ),
                ("reservation_compensation", "reservation_compensation"),
            )
            if self.config.reservations_enabled
            else ()
        ) + (
            (
                (
                    "rental_idle_holding_cost",
                    "rental_idle_holding_cost",
                ),
                (
                    "opening_rental_idle_holding_cost_basis",
                    "opening_rental_idle_holding_cost_basis",
                ),
            )
            if self.config.rental_idle_holding_rate > 0
            else ()
        ) + (
            (
                ("liquidation_nominal_proceeds", "liquidation_nominal_proceeds"),
                ("liquidation_depth_discount", "liquidation_depth_discount"),
            )
            if self.config.liquidation_depth_enabled
            else ()
        )
        days = []
        for report in reports:
            financial = report["financial"]
            row = {"day": report["day"]}
            row.update(
                {
                    public_key: financial.get(source_key, 0)
                    for public_key, source_key in finance_fields
                }
            )
            row["end_cash"] = report["end"]["cash"]
            row["end_deposit_liability"] = report["end"]["deposit_liability"]
            days.append(row)
        totals = {
            public_key: round(sum(float(row[public_key]) for row in days), 6)
            for public_key, _ in finance_fields
        }
        totals["assets_sold"] = int(totals["assets_sold"])
        totals["retail_units_liquidated"] = int(
            totals["retail_units_liquidated"]
        )
        result: dict[str, Any] = {
            "granularity": granularity,
            "period": period,
            "totals": totals,
        }
        if granularity == "daily":
            result["days"] = days[-30:]
            result["daily_rows_truncated"] = len(days) > 30
        return result

    def service_history(
        self,
        state: WorldState,
        horizon_days: int,
        *,
        offer_ids: list[str] | None = None,
        end_day: int | None = None,
        granularity: str = "totals",
        cursor: int = 0,
        limit: int = 20,
        channel: str = "all",
    ) -> dict[str, Any]:
        if operating_history.enabled(self.dataset.metadata):
            return operating_history.service_history(self, state, horizon_days, channel=channel,
                offer_ids=offer_ids, end_day=end_day, granularity=granularity, cursor=cursor, limit=limit)
        if granularity not in {"totals", "daily", "offer"}:
            raise ValueError("granularity must be totals, daily, or offer")
        if cursor < 0 or not 1 <= limit <= 50:
            raise ValueError(
                "cursor must be non-negative and limit must be in [1, 50]"
            )
        period, reports = self._history_window(state, horizon_days, end_day=end_day)
        selected_offer_ids = set(offer_ids or [])
        unknown = sorted(selected_offer_ids - set(self.offers))
        if unknown:
            raise ValueError(f"unknown offer_ids: {unknown}")
        metric_fields = (
            ("requested_units", "requested_units"),
            ("fulfilled_units", "fulfilled_units"),
            ("quote_declined_units", "outside_rejections"),
            ("stockout_request_units", "stockout_rejections"),
            ("closed_service_request_units", "service_closed_rejections"),
            ("season_end_declined_units", "horizon_rejections"),
            ("revenue", "revenue"),
        )
        if self._v47_autonomous_rentals():
            metric_fields += (("continuation_rental_revenue", "continuation_rental_revenue"),
                              ("continuation_rental_days", "continuation_rental_days"))
        count_fields = [key for key, _ in metric_fields
                        if key not in {"revenue", "continuation_rental_revenue"}]
        days = []
        for report in reports:
            row: dict[str, Any] = {"day": report["day"]}
            for public_key, source_key in metric_fields:
                row[public_key] = round(
                    sum(
                        float(metric.get(source_key, 0))
                        for metric in report.get("service_metrics", {}).values()
                    ),
                    6,
                )
            for public_key in count_fields:
                row[public_key] = int(row[public_key])
            days.append(row)
        totals = {
            public_key: round(sum(float(row[public_key]) for row in days), 6)
            for public_key, _ in metric_fields
        }
        for public_key in count_fields:
            totals[public_key] = int(totals[public_key])
        if granularity == "daily":
            return {
                "granularity": granularity,
                "period": period,
                "totals": totals,
                "days": days[-30:],
                "daily_rows_truncated": len(days) > 30,
            }
        if granularity == "offer":
            identifiers = sorted(selected_offer_ids or set(self.offers))
            flat = []
            for offer_id in identifiers:
                offer = self.offers[offer_id]
                raw_rows = []
                for report in reports:
                    for metric in report.get("service_metrics", {}).values():
                        raw = metric.get("offer_fulfillment_breakdown", {}).get(
                            offer_id
                        )
                        if raw:
                            raw_rows.append(raw)
                flat.append(
                    {
                        "offer_id": offer_id,
                        "name": offer.name,
                        "kind": (
                            "rental_asset"
                            if offer.asset_kind == "durable"
                            else "retail_item"
                        ),
                        "quote_exposures": sum(
                            int(row.get("quote_exposures", 0)) for row in raw_rows
                        ),
                        "quoted_units": sum(
                            int(row.get("quoted_units", 0)) for row in raw_rows
                        ),
                        "fulfilled_units": sum(
                            int(row.get("fulfilled_units", 0)) for row in raw_rows
                        ),
                        "revenue": round(
                            sum(float(row.get("revenue", 0.0)) for row in raw_rows),
                            2,
                        ),
                        "retail_cogs": round(
                            sum(
                                float(row.get("retail_cogs", 0.0))
                                for row in raw_rows
                            ),
                            2,
                        ),
                        "rental_days": sum(
                            int(row.get("rental_days", 0)) for row in raw_rows
                        ),
                        **({
                            "continuation_rental_revenue": round(sum(
                                float(row.get("continuation_rental_revenue", 0)) for row in raw_rows), 2),
                            "continuation_rental_days": sum(
                                int(row.get("continuation_rental_days", 0)) for row in raw_rows),
                        } if self._v47_autonomous_rentals() else {}),
                        **self._offer_snapshot_stats(
                            reports, [offer_id], include_price_regimes=True
                        ),
                    }
                )
            start = max(0, int(cursor))
            page = flat[start : start + limit]
            next_cursor = start + len(page) if start + len(page) < len(flat) else None
            return {
                "granularity": granularity,
                "period": period,
                "offers": page,
                "cursor": start,
                "limit": limit,
                "returned_offer_count": len(page),
                "has_more": next_cursor is not None,
                "next_cursor": next_cursor,
                "total_offers": len(flat),
            }
        return {
            "granularity": granularity,
            "period": period,
            "totals": totals,
        }

    @staticmethod
    def _offer_snapshot_stats(
        reports: list[dict[str, Any]],
        offer_ids: list[str],
        *,
        include_price_regimes: bool,
    ) -> dict[str, Any]:
        identifiers = set(offer_ids)
        by_day: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for report in reports:
            for row in report.get("offer_snapshots", []):
                if row.get("offer_id") in identifiers:
                    by_day[int(report["day"])].append(row)
        available_unit_days = sum(
            int(row.get("available_units", 0))
            for rows in by_day.values()
            for row in rows
            if row.get("service_open", True)
        )
        days_in_stock = sum(
            any(
                bool(row.get("service_open", True))
                and int(row.get("available_units", 0)) > 0
                for row in rows
            )
            for rows in by_day.values()
        )
        prices = [
            float(row["posted_price"])
            for rows in by_day.values()
            for row in rows
            if row.get("posted_price") is not None
        ]
        result: dict[str, Any] = {
            "available_unit_days": available_unit_days,
            "days_in_stock": days_in_stock,
            "average_posted_price": (
                round(sum(prices) / len(prices), 2) if prices else None
            ),
        }
        if include_price_regimes:
            points = [
                (day, float(rows[0]["posted_price"]))
                for day, rows in sorted(by_day.items())
                if rows and rows[0].get("posted_price") is not None
            ]
            regimes: list[dict[str, Any]] = []
            for day, price in points:
                if regimes and regimes[-1]["price"] == price and regimes[-1]["end_day"] == day - 1:
                    regimes[-1]["end_day"] = day
                else:
                    regimes.append({"start_day": day, "end_day": day, "price": price})
            result["price_regimes"] = regimes
        return result

    @staticmethod
    def _history_window(
        state: WorldState,
        horizon_days: int,
        *,
        end_day: int | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        last_completed = int(state.daily_history[-1]["day"]) if state.daily_history else 0
        resolved_end = last_completed if end_day is None else min(int(end_day), last_completed)
        start_day = max(1, resolved_end - horizon_days + 1)
        reports = [
            row
            for row in state.daily_history
            if start_day <= int(row["day"]) <= resolved_end
        ]
        period = {
            "start_day": reports[0]["day"] if reports else None,
            "end_day": reports[-1]["day"] if reports else None,
            "completed_days": len(reports),
        }
        return period, reports

    def view_rental_activity(
        self,
        state: WorldState,
        horizon_days: int,
        *,
        end_day: int | None = None,
        product_types: list[str] | None = None,
        offer_ids: list[str] | None = None,
        asset_ids: list[str] | None = None,
        event_types: list[str] | None = None,
        cursor: str | None = None,
        limit: int = 25,
    ) -> dict[str, Any]:
        if not 1 <= limit <= 50:
            raise ValueError("limit must be in [1, 50]")
        resolved_end = state.day if end_day is None else min(int(end_day), state.day)
        start_day = max(1, resolved_end - horizon_days + 1)
        product_filter = set(product_types or [])
        offer_filter = set(offer_ids or [])
        asset_filter = set(asset_ids or [])
        event_filter = set(event_types or [])
        cursor_index: int | None = None
        if cursor is not None:
            prefix = "rental-event-"
            suffix = cursor[len(prefix) :] if cursor.startswith(prefix) else ""
            if not suffix.isdigit():
                raise ValueError(
                    "cursor must be a rental event ID such as rental-event-000123"
                )
            cursor_index = int(suffix)
            canonical_cursor = f"rental-event-{cursor_index:06d}"
            if (
                cursor != canonical_cursor
                or not 1 <= cursor_index <= len(state.rental_activity)
            ):
                raise ValueError(f"unknown rental activity cursor: {cursor}")
        matches = [
            (index, row)
            for index, row in enumerate(state.rental_activity, start=1)
            if start_day <= int(row["day"]) <= resolved_end
            and (not product_filter or row["product_type"] in product_filter)
            and (not offer_filter or row["offer_id"] in offer_filter)
            and (not asset_filter or row["asset_id"] in asset_filter)
            and (not event_filter or row["event_type"] in event_filter)
        ]
        page_candidates = [
            (index, row)
            for index, row in reversed(matches)
            if cursor_index is None or index < cursor_index
        ]
        selected = page_candidates[:limit]
        events = []
        for index, raw_row in selected:
            row = {
                "event_id": f"rental-event-{index:06d}",
                **dict(raw_row),
            }
            row.pop("product_type", None)
            row["name"] = self.offers[str(row["offer_id"])].name
            events.append(row)
        has_more = len(page_candidates) > len(selected)
        return {
            "period": {"start_day": start_day, "end_day": resolved_end},
            "matching_event_count": len(matches),
            "returned_event_count": len(events),
            "has_more": has_more,
            "next_cursor": events[-1]["event_id"] if has_more and events else None,
            "events": events,
        }

    def _record_rental_activity(
        self,
        state: WorldState,
        event_type: str,
        asset: RentalAsset,
        **details: Any,
    ) -> None:
        if event_type not in RENTAL_ACTIVITY_EVENT_TYPES and not (
            self._v31_warehouse_capacity() and event_type == "asset_disposed"
        ) and not (
            self._v37_supplier_trial()
            and event_type in SUPPLIER_TRIAL_ACTIVITY_EVENT_TYPES
        ):
            raise ValueError(f"unknown rental activity event: {event_type}")
        row: dict[str, Any] = {
            "day": state.day,
            "event_type": event_type,
            "asset_id": asset.asset_id,
            "offer_id": asset.offer_id,
            "product_type": asset.service_line,
            "used_rental_days": asset.used_life_days,
            "completed_rental_cycles": asset.used_rental_cycles,
            "status_after": asset.status,
        }
        row.update(details)
        state.rental_activity.append(row)

    def _capture_opening_rental_holding_basis(self, state: WorldState) -> None:
        if (
            self.config.lifecycle_economics_enabled
            and self.config.lifecycle_holding_enabled
        ):
            held = [asset for asset in state.assets if (
                self._warehouse_asset_held(asset) if self._v31_warehouse_capacity() else asset.status != "sold"
            )]
            owned = [asset for asset in held if asset.ownership_type == "owned"]
            on_site = [asset for asset in held if asset.status != "rented"]
            terms = [
                (
                    asset,
                    self._lifecycle_holding_terms(
                        asset, stored_on_site=asset.status != "rented"
                    ),
                )
                for asset in held
            ]
            state.current_day_ledger["opening_rental_holding_cost_basis"] = round(
                sum(asset.unit_cost for asset in owned), 4
            )
            state.current_day_ledger[
                "opening_rental_idle_holding_cost_basis"
            ] = round(
                sum(
                    asset.unit_cost
                    for asset in on_site
                    if asset.ownership_type == "owned"
                ),
                4,
            )
            state.current_day_ledger[
                "opening_rental_capital_holding_charge"
            ] = round(
                sum(
                    row["capital_charge"]
                    for asset, row in terms
                    if asset.ownership_type == "owned"
                ),
                6,
            )
            state.current_day_ledger[
                "opening_rental_space_holding_charge"
            ] = round(sum(row["space_charge"] for _asset, row in terms), 6)
            state.current_day_ledger[
                "opening_rental_space_burden_units"
            ] = round(
                sum(
                    float(
                        self._lifecycle_projection(asset.offer_id)[
                            "storage_burden_units"
                        ]
                    )
                    for asset in on_site
                ),
                6,
            )
            return
        state.current_day_ledger["opening_rental_holding_cost_basis"] = round(
            sum(
                asset.unit_cost
                for asset in state.assets
                if asset.ownership_type == "owned"
                and asset.status not in {"retired", "sold"}
            ),
            4,
        )
        state.current_day_ledger["opening_rental_idle_holding_cost_basis"] = round(
            sum(
                asset.unit_cost
                for asset in state.assets
                if asset.ownership_type == "owned"
                and asset.status not in {"rented", "retired", "sold"}
            ),
            4,
        )

    def weather_forecast(self, state: WorldState, horizon: int) -> list[dict[str, Any]]:
        adjacency = {
            "sunny": ("cloudy", "rainy"),
            "cloudy": ("sunny", "rainy"),
            "rainy": ("cloudy", "sunny"),
            "snowy": ("cloudy",),
        }
        rows = []
        for target_day in range(
            state.day, min(state.max_days, state.day + horizon - 1) + 1
        ):
            lead = target_day - state.day
            actual = self._weather(target_day)
            source_target = self._source_weather_day(target_day)
            mean, low, high = self._temperatures(target_day)
            sigma = self.rules.forecast_temperature_sigma_by_lead[min(7, lead)]
            temp_rng = self._stable_event_rng(
                _RNG_FORECAST, source_target, identity="temperature_error"
            )
            error = float(temp_rng.normal()) * sigma
            forecast_mean = round(mean + error, 1)
            forecast_low = round(low + error, 1)
            forecast_high = round(high + error, 1)
            accuracy = self.rules.forecast_accuracy_by_lead[min(7, lead)]
            accuracy_rng = self._stable_event_rng(
                _RNG_FORECAST, source_target, identity="condition_accuracy"
            )
            if lead == 0 or accuracy_rng.random() < accuracy:
                predicted = actual
            else:
                miss_rng = self._stable_event_rng(
                    _RNG_FORECAST, source_target, identity="condition_miss"
                )
                choices = adjacency[actual]
                predicted = choices[int(miss_rng.integers(0, len(choices)))]
            if predicted == "snowy" and forecast_mean > 3.0:
                predicted = "rainy"
            elif predicted == "rainy" and forecast_high <= 2.0:
                predicted = "snowy"
            rows.append(
                {
                    "day": target_day,
                    **self._public_calendar(target_day),
                    "condition": predicted,
                    "lead_days": lead,
                    "condition_accuracy": round(accuracy, 2),
                    "temp_mean_c": forecast_mean,
                    "night_low_c": forecast_low,
                    "day_high_c": forecast_high,
                }
            )
        return rows

    def public_offer(self, offer_id: str, *, detailed: bool = False) -> dict[str, Any]:
        offer = self.offers[offer_id]
        hidden_specification_names = {
            "product_type",
            "product_subtype",
            "choice_group",
            "service_key",
            "quality_tier",
            "perceived_quality_tier",
            "decision_signature",
        }
        specifications: list[dict[str, Any]] = []
        seen: set[str] = set()
        for fact in offer.evidence:
            name = str(fact.get("name") or "").strip()
            if (
                not name
                or name in hidden_specification_names
                or fact.get("value") is None
                or name in seen
            ):
                continue
            specifications.append(
                {
                    "name": name,
                    "value": fact.get("value"),
                    "evidence": str(fact.get("evidence") or ""),
                }
            )
            seen.add(name)
        neutral_attributes = {
            **dict(offer.descriptive_attributes),
            **dict(offer.decision_attributes),
            **dict(offer.consumer_value_attributes),
        }
        for name, value in sorted(neutral_attributes.items()):
            if name in hidden_specification_names or name in seen or value is None:
                continue
            evidence = offer.decision_evidence.get(name, {}) or offer.consumer_value_evidence.get(
                name, {}
            )
            evidence_text = "; ".join(
                str(source.get("evidence") or "").strip()
                for source in evidence.get("sources", [])
                if str(source.get("evidence") or "").strip()
            )
            if not evidence_text:
                continue
            specifications.append(
                {"name": name, "value": value, "evidence": evidence_text}
            )
            seen.add(name)
        result = {
            "offer_id": offer.offer_id,
            "kind": (
                "rental_asset" if offer.asset_kind == "durable" else "retail_item"
            ),
            "name": offer.name,
            "brand": offer.brand,
            "source_title": offer.source_title,
            "source_variant_name": offer.source_variant_name,
            "package_price": offer.package_price,
            "package_units": offer.package_units,
            "supplier_unit_cost": offer.unit_cost,
            "consumer_unit": (
                {
                    key: value
                    for key, value in (offer.consumer_unit or {}).items()
                    if key in {"label", "evidence", "quantity", "spec"}
                }
                if offer.consumer_unit is not None
                else None
            ),
            "specifications": specifications,
            "seller_id": offer.seller_id,
            "seller_name": offer.seller_name,
            "estimated_delivery_days": self.supplier_delivery_estimate(
                offer.seller_id
            ),
        }
        if offer.asset_kind == "durable":
            if self._v31_warehouse_capacity():
                result["warehouse_capacity_units_per_asset"] = self._warehouse_weight(offer_id)
            result["normal_return_cleaning"] = self.public_offer_cleaning_rule(
                offer.service_line, offer_id=offer.offer_id
            )
            if self.config.lifecycle_economics_enabled:
                factor = self._lifecycle_factor(offer.offer_id)
                result["lifecycle_profile"] = {
                    "relative_durability_tier": factor["k_lifetime"][
                        "relative_durability_tier"
                    ],
                    "damage_hazard_tier": factor["d_damage"]["hazard_tier"],
                    "damage_severity_tier": factor["d_damage"][
                        "severity_tier"
                    ],
                    "storage_burden_tier": factor["v_storage"]["tier"],
                    "service_life_limit_days": int(
                        self._lifecycle_projection(offer.offer_id)[
                            "service_life_limit_days"
                        ]
                    ),
                    "service_life_limit_cycles": int(
                        self._lifecycle_projection(offer.offer_id)[
                            "service_life_limit_cycles"
                        ]
                    ),
                }
                if detailed and operating_history.enabled(self.dataset.metadata):
                    result["holding_costs"] = operating_history.holding_costs(self, offer_id)
        if detailed:
            result["source_evidence"] = [
                dict(row)
                for row in offer.evidence
                if str(row.get("name") or "") not in hidden_specification_names
            ]
        return result

    def catalog_index(self, asset_kind: str | None = None) -> list[dict[str, Any]]:
        rows = []
        for internal_kind, public_kind in (
            ("durable", "rental_asset"),
            ("consumable", "retail_item"),
        ):
            if asset_kind is not None and internal_kind != asset_kind:
                continue
            offers = [
                offer for offer in self.dataset.offers if offer.asset_kind == internal_kind
            ]
            if not offers:
                continue
            costs = [float(offer.unit_cost) for offer in offers]
            rows.append(
                {
                    "kind": public_kind,
                    "offer_count": len(offers),
                    "minimum_supplier_unit_cost": round(min(costs), 2),
                    "maximum_supplier_unit_cost": round(max(costs), 2),
                }
            )
        return rows

    def search_catalog(
        self,
        query: str,
        product_types: list[str],
        asset_kind: str | None,
        limit: int,
        cursor: int = 0,
    ) -> dict[str, Any]:
        matches: list[
            tuple[tuple[int, int, int, int, int], dict[str, Any]]
        ] = []
        for offer in self.dataset.offers:
            if asset_kind is not None and offer.asset_kind != asset_kind:
                continue
            public = self.public_offer(offer.offer_id)
            score = _catalog_match_score(public, query)
            if score is not None:
                matches.append((score, public))
        matches.sort(
            key=lambda item: (
                tuple(-value for value in item[0]),
                str(item[1]["offer_id"]),
            )
        )
        rows = [public for _score, public in matches]
        page = rows[cursor : cursor + limit]
        next_cursor = (
            cursor + len(page) if cursor + len(page) < len(rows) else None
        )
        return {
            "matched_offer_count": len(rows),
            "returned_offer_count": len(page),
            "truncated": next_cursor is not None,
            "offers": page,
            "cursor": cursor,
            "limit": limit,
            "has_more": next_cursor is not None,
            "next_cursor": next_cursor,
        }
