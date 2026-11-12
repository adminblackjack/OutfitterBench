from __future__ import annotations

from .common import (
    Any,
    CLEANING_COST_RULES,
    CleaningJob,
    Counter,
    Decimal,
    INVENTORY_COST_QUANTUM,
    Iterable,
    MONEY_QUANTUM,
    PurchaseOrder,
    RENTAL_HOLDING_RATE,
    RETAIL_HOLDING_RATE,
    RETAIL_SALVAGE_RATE,
    ROUND_HALF_UP,
    RentalAsset,
    SECONDHAND_GRADE_SCHEDULE,
    SECONDHAND_PARTS_FLOOR_RATE,
    SECONDHAND_SELLABLE_STATUSES,
    TERMINAL_HEALTHY_STATUSES,
    ToolResult,
    WalkInCapacityLimit,
    WorldState,
    _RNG_SUPPLIER_DELIVERY,
    _empty_agent_actions,
    _empty_ledger,
    math,
)
from ..lifecycle_economics import maintenance_terms


class OperationsMixin:
    @staticmethod
    def _order_delivery_timing() -> str:
        return (
            "Deliveries are processed once at day opening. Today's processing "
            "is complete when manager tools are available; an order still in "
            "transit cannot arrive later today. The original estimated delivery "
            "day is an estimate, not a guaranteed deadline."
        )

    def place_purchase_order(
        self,
        state: WorldState,
        seller_id: str,
        items: list[dict[str, Any]],
        procurement_mode: str = "owned",
    ) -> ToolResult:
        self._assert_durable_acquisition_budget(state)
        self._assert_supplier_trials(state)
        if procurement_mode not in {"owned", "supplier_trial"}:
            return ToolResult(False, {
                "error": "procurement_mode must be owned or supplier_trial",
                "cash_and_orders_unchanged": True,
            })
        if procurement_mode == "supplier_trial" and not self._v37_supplier_trial():
            return ToolResult(False, {"error": "supplier_trial is unavailable"})
        if not items:
            return ToolResult(False, {"error": "items must not be empty"})
        normalized: list[dict[str, Any]] = []
        total = 0.0
        for item in items:
            if not isinstance(item, dict):
                return ToolResult(False, {"error": "every order item must be an object"})
            offer_id = str(item.get("offer_id", ""))
            packs = item.get("packs")
            if offer_id not in self.offers:
                return ToolResult(False, {"error": f"unknown offer_id: {offer_id}"})
            offer = self.offers[offer_id]
            if offer.seller_id != str(seller_id):
                return ToolResult(False, {"error": f"offer {offer_id} belongs to another seller"})
            if isinstance(packs, bool) or not isinstance(packs, int) or not 1 <= packs <= 100:
                return ToolResult(False, {"error": "packs must be an integer in [1, 100]"})
            line_cost = round(offer.package_price * packs, 2)
            total += line_cost
            normalized.append(
                {
                    "offer_id": offer_id,
                    "packs": packs,
                    "package_units": offer.package_units,
                    "unit_cost": offer.unit_cost,
                    "line_cost": line_cost,
                }
            )
        total = round(total, 2)
        trial_terms: dict[str, Any] | None = None
        if procurement_mode == "supplier_trial":
            prepared = self._supplier_trial_order_terms(state, normalized)
            if isinstance(prepared, ToolResult):
                return prepared
            trial_terms = prepared
        cash_cost = (
            total
            if procurement_mode == "owned"
            else float(trial_terms["handling_fee"])
        )
        if cash_cost > state.cash + 1e-9:
            return ToolResult(False, {
                "error": "insufficient cash",
                "required": cash_cost,
                "cash": state.cash,
                "cash_and_orders_unchanged": True,
            })
        estimated_delivery_days = self.supplier_delivery_estimate(str(seller_id))
        estimated_delivery_day = state.day + estimated_delivery_days
        delivery_cutoff_day = (
            self.config.max_days
            if self._v24_reservation_operations()
            else state.max_days
        )
        if estimated_delivery_day > delivery_cutoff_day:
            return ToolResult(
                False,
                {
                    "error": (
                        "the estimated delivery date is after the manager decision horizon"
                        if self._v24_reservation_operations()
                        else "the estimated delivery date is after the outdoor season closes"
                    ),
                    "seller_id": str(seller_id),
                    "estimated_delivery_days": estimated_delivery_days,
                    "estimated_delivery_day": estimated_delivery_day,
                    **(
                        {"decision_horizon_day": self.config.max_days}
                        if self._v24_reservation_operations()
                        else {"season_end_day": state.max_days}
                    ),
                },
            )
        if procurement_mode == "owned":
            budget_error = self._durable_acquisition_budget_purchase_check(
                state, normalized
            )
            if budget_error is not None:
                return budget_error
        capacity_error = self._warehouse_purchase_check(state, normalized)
        if capacity_error is not None:
            return capacity_error
        if self._v37_supplier_trial():
            # A supplier-trial request must not shift the delivery tape of
            # otherwise identical owned orders (or vice versa). The global
            # order_id remains an audit identifier, not a random address.
            procurement_sequence = 1 + sum(
                order.procurement_mode == procurement_mode
                for order in state.orders
            )
            delivery_rng = self._stable_event_rng(
                _RNG_SUPPLIER_DELIVERY,
                state.day,
                procurement_sequence,
                identity=f"{procurement_mode}:{seller_id}",
            )
        else:
            delivery_rng = self._stable_event_rng(
                _RNG_SUPPLIER_DELIVERY,
                state.day,
                state.next_order_id,
                identity=str(seller_id),
            )
        actual_delivery_days = int(
            delivery_rng.integers(
                max(
                    self.config.supplier_delivery_min_days,
                    estimated_delivery_days
                    - self.config.supplier_delivery_early_days,
                ),
                min(
                    self.config.supplier_delivery_max_days,
                    estimated_delivery_days
                    + self.config.supplier_delivery_late_days,
                )
                + 1,
            )
        )
        delivery_day = state.day + actual_delivery_days
        order = PurchaseOrder(
            order_id=f"po-{state.next_order_id:05d}",
            seller_id=str(seller_id),
            placed_day=state.day,
            delivery_day=delivery_day,
            total_cost=cash_cost,
            items=normalized,
            estimated_delivery_day=estimated_delivery_day,
            procurement_mode=procurement_mode,
            supplier_trial_id=(
                None if trial_terms is None else str(trial_terms["trial_id"])
            ),
            supplier_trial_handling_fee=(
                0.0 if trial_terms is None else float(trial_terms["handling_fee"])
            ),
        )
        state.next_order_id += 1
        state.orders.append(order)
        trial_record = (
            None
            if trial_terms is None
            else self._accept_supplier_trial(state, order, trial_terms)
        )
        self._assert_durable_acquisition_budget(state)
        self._assert_supplier_trials(state)
        self._warehouse_record_peak(state)
        state.cash = round(state.cash - cash_cost, 2)
        if procurement_mode == "owned":
            state.cumulative_purchase_cost = round(
                state.cumulative_purchase_cost + cash_cost, 2
            )
            self._add(
                state.current_day_ledger["actions"], "purchase_cost", cash_cost
            )
        else:
            self._add(
                state.current_day_ledger["actions"],
                "supplier_trial_handling_fee",
                cash_cost,
            )
        actions = state.current_day_ledger.setdefault(
            "agent_actions", _empty_agent_actions()
        )
        actions.setdefault("purchase_orders", []).append(
            {
                "order_id": order.order_id,
                "seller_id": order.seller_id,
                "estimated_delivery_day": estimated_delivery_day,
                "total_cost": order.total_cost,
                "procurement_mode": procurement_mode,
                **(
                    {
                        "supplier_trial_id": trial_record.trial_id,
                        "catalog_purchase_cost": total,
                        "supplier_trial_handling_fee": cash_cost,
                    }
                    if trial_record is not None
                    else {}
                ),
                "items": [
                    {
                        "offer_id": item["offer_id"],
                        "product_type": self.offers[item["offer_id"]].service_line,
                        "packs": item["packs"],
                        "arrival_units": int(
                            item["packs"] * item["package_units"]
                        ),
                        "line_cost": item["line_cost"],
                    }
                    for item in order.items
                ],
            }
        )
        return ToolResult(True, {
            "order": self._order_view(order), "cash": state.cash,
            **(
                {
                    (
                        "durable_capital"
                        if self._v40_revolving_durable_capital()
                        else "durable_acquisition_budget"
                    ): (
                        self._durable_acquisition_budget_summary(state)
                    )
                }
                if self._durable_capital_control_enabled()
                else {}
            ),
            **({"warehouse_capacity": self._warehouse_summary(state)} if self._v31_warehouse_capacity() else {}),
            **(
                {"supplier_trial": self._supplier_trial_summary(state)}
                if self._v37_supplier_trial()
                else {}
            ),
            **({"delivery_timing": self._order_delivery_timing()}
               if self._v29_reservation_accountability() else {}),
        })

    def _order_view(self, order: PurchaseOrder) -> dict[str, Any]:
        estimated_delivery_day = self.estimated_order_delivery_day(order)
        result = {
            "order_id": order.order_id,
            "seller_id": order.seller_id,
            "placed_day": order.placed_day,
            "estimated_delivery_day": estimated_delivery_day,
            "estimated_delivery_days": estimated_delivery_day - order.placed_day,
            "total_cost": order.total_cost,
            "procurement_mode": order.procurement_mode,
            **(
                {
                    "supplier_trial_id": order.supplier_trial_id,
                    "supplier_trial_handling_fee": order.supplier_trial_handling_fee,
                    "catalog_purchase_cost": round(
                        sum(float(item["line_cost"]) for item in order.items), 2
                    ),
                }
                if order.procurement_mode == "supplier_trial"
                else {}
            ),
            **(
                {
                    (
                        "order_durable_capital_basis"
                        if self._v40_revolving_durable_capital()
                        else "durable_acquisition_budget_amount"
                    ): (
                        self._durable_order_budget_amount(order)
                    )
                }
                if self._durable_capital_control_enabled()
                else {}
            ),
            "items": [
                {
                    "offer_id": item["offer_id"],
                    "packs": item["packs"],
                    "package_units": item["package_units"],
                    "supplier_unit_cost": item["unit_cost"],
                    "line_cost": item["line_cost"],
                    **(
                        {
                            "durable_capital_basis": self._capital_money(
                                item.get("durable_capital_basis_cents")
                            )
                        }
                        if self._v40_revolving_durable_capital()
                        and item.get("durable_capital_basis_cents") is not None
                        else {}
                    ),
                    "name": self.offers[item["offer_id"]].name,
                    "arrival_units": int(item["packs"] * item["package_units"]),
                    **({"received_units": item["received_units"],
                        "remaining_arrival_units": self._ordered_units(item) - item["received_units"],
                        "warehouse_capacity_units_per_asset": self._warehouse_weight(item["offer_id"]),
                        "inbound_reserved_capacity_units": (self._ordered_units(item) - item["received_units"]) * self._warehouse_weight(item["offer_id"])}
                       if self._v31_warehouse_capacity() else {}),
                }
                for item in order.items
            ],
            "status": order.status,
        }
        if order.status == "delivered":
            result["actual_delivery_day"] = order.delivery_day
            result["actual_delivery_days"] = order.delivery_day - order.placed_day
        return result

    def set_prices(self, state: WorldState, values: list[dict[str, Any]]) -> ToolResult:
        if not values:
            return ToolResult(False, {"error": "prices must not be empty"})
        updates: dict[str, float] = {}
        for row in values:
            if not isinstance(row, dict):
                return ToolResult(False, {"error": "every price item must be an object"})
            offer_id = str(row.get("offer_id", ""))
            price = row.get("price")
            if offer_id not in self.offers:
                return ToolResult(False, {"error": f"unknown offer_id: {offer_id}"})
            if (
                isinstance(price, bool)
                or not isinstance(price, (int, float))
                or not math.isfinite(price)
                or price <= 0
            ):
                return ToolResult(False, {"error": "price must be a positive finite number"})
            updates[offer_id] = round(float(price), 2)
        actions = state.current_day_ledger.setdefault(
            "agent_actions", _empty_agent_actions()
        )
        changes = actions.setdefault("price_changes", {})
        for offer_id, new_price in updates.items():
            old_price = state.prices.get(offer_id)
            change = changes.setdefault(
                offer_id,
                {
                    "offer_id": offer_id,
                    "product_type": self.offers[offer_id].service_line,
                    "old_price": old_price,
                    "new_price": new_price,
                },
            )
            change["new_price"] = new_price
        state.prices.update(updates)
        return ToolResult(True, {"updated": updates})

    def set_rental_offer_status(
        self, state: WorldState, values: list[dict[str, Any]]
    ) -> ToolResult:
        if not values:
            return ToolResult(False, {"error": "items must not be empty"})
        updates: dict[str, bool] = {}
        for row in values:
            if not isinstance(row, dict):
                return ToolResult(False, {"error": "every status item must be an object"})
            offer_id = str(row.get("offer_id", ""))
            opened = row.get("open")
            if offer_id not in state.rental_service_open:
                return ToolResult(False, {"error": f"unknown rental offer_id: {offer_id}"})
            if not isinstance(opened, bool):
                return ToolResult(False, {"error": "open must be boolean"})
            updates[offer_id] = opened
        actions = state.current_day_ledger.setdefault(
            "agent_actions", _empty_agent_actions()
        )
        changes = actions.setdefault("rental_product_status_changes", {})
        for offer_id, new_open in updates.items():
            old_open = state.rental_service_open[offer_id]
            change = changes.setdefault(
                offer_id,
                {
                    "offer_id": offer_id,
                    "name": self.offers[offer_id].name,
                    "old_open": old_open,
                    "new_open": new_open,
                },
            )
            change["new_open"] = new_open
        state.rental_service_open.update(updates)
        return ToolResult(
            True,
            {"rental_offer_open": dict(sorted(state.rental_service_open.items()))},
        )

    def _v28_set_walk_in_capacity_limits(
        self, state: WorldState, values: list[dict[str, Any]]
    ) -> ToolResult:
        """Validate the whole batch, then apply order-independent daily upserts."""

        compact_actor_result = self._v41_actor_interface_slimming()

        if not values:
            return ToolResult(
                False,
                {
                    "error": "walk_in_capacity_limit_batch_invalid",
                    "batch_applied": False,
                    "invalid_item_count": 0,
                    "invalid_items": [],
                    "overlap_conflicts": [],
                    "batch_violations": [
                        {
                            "code": "empty_items",
                            "message": "items must not be empty",
                        }
                    ],
                },
            )

        allowed_fields = {
            "action",
            "offer_id",
            "from_day",
            "to_day",
            "max_walk_in_occupied_units",
        }
        violations_by_index: dict[int, list[dict[str, Any]]] = {}
        original_by_index: dict[int, Any] = {}
        normalized: list[dict[str, Any]] = []
        no_op_items: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        def add_violation(
            item_index: int,
            code: str,
            message: str,
            **details: Any,
        ) -> None:
            violations_by_index.setdefault(item_index, []).append(
                {"code": code, "message": message, **details}
            )

        for item_index, raw in enumerate(values):
            original_by_index[item_index] = raw
            if not isinstance(raw, dict):
                add_violation(
                    item_index,
                    "item_not_object",
                    "every capacity item must be an object",
                )
                continue
            unexpected = sorted(set(raw) - allowed_fields)
            if unexpected:
                add_violation(
                    item_index,
                    "unexpected_fields",
                    "capacity item contains unsupported fields",
                    fields=unexpected,
                )
            action = str(raw.get("action", ""))
            offer_id = str(raw.get("offer_id", ""))
            from_day = raw.get("from_day")
            to_day = raw.get("to_day")
            maximum = raw.get("max_walk_in_occupied_units")
            if action not in {"set", "clear"}:
                add_violation(
                    item_index,
                    "invalid_action",
                    "action must be set or clear",
                )
            if (
                offer_id not in self.offers
                or self.offers[offer_id].asset_kind != "durable"
            ):
                add_violation(
                    item_index,
                    "unknown_rental_offer",
                    "offer_id must identify a rental asset offer",
                    offer_id=offer_id,
                )
            dates_are_integers = (
                not isinstance(from_day, bool)
                and isinstance(from_day, int)
                and not isinstance(to_day, bool)
                and isinstance(to_day, int)
            )
            if not dates_are_integers:
                add_violation(
                    item_index,
                    "invalid_date_type",
                    "from_day and to_day must be integers",
                )
            else:
                assert isinstance(from_day, int) and isinstance(to_day, int)
                if from_day < 1:
                    add_violation(
                        item_index,
                        "from_day_before_day_one",
                        "from_day must be at least 1",
                        from_day=from_day,
                    )
                if to_day < from_day:
                    add_violation(
                        item_index,
                        "window_end_before_start",
                        "to_day must be no earlier than from_day",
                        from_day=from_day,
                        to_day=to_day,
                    )
                if to_day > state.max_days:
                    add_violation(
                        item_index,
                        "window_after_runoff",
                        "to_day must not exceed runoff_end_day",
                        to_day=to_day,
                        runoff_end_day=state.max_days,
                    )
                if action == "set" and from_day < state.day:
                    add_violation(
                        item_index,
                        "set_includes_past",
                        "set cannot change a historical capacity day",
                        from_day=from_day,
                        current_day=state.day,
                    )
            if action == "set" and (
                isinstance(maximum, bool)
                or not isinstance(maximum, int)
                or not 0 <= maximum <= 10_000
            ):
                add_violation(
                    item_index,
                    "set_limit_required_or_invalid",
                    "set requires max_walk_in_occupied_units as an integer in [0, 10000]",
                )
            if action == "clear" and "max_walk_in_occupied_units" in raw:
                add_violation(
                    item_index,
                    "clear_forbids_limit",
                    "clear must omit max_walk_in_occupied_units",
                )
            if item_index in violations_by_index:
                continue

            assert isinstance(from_day, int) and isinstance(to_day, int)
            if action == "clear" and to_day < state.day:
                if compact_actor_result:
                    warnings.append(
                        {
                            "warning_code": "historical_clear_range_ignored",
                            "item_index": item_index,
                            "offer_id": offer_id,
                            "ignored_from_day": from_day,
                            "ignored_to_day": to_day,
                            "reason": "no_current_or_future_days",
                        }
                    )
                else:
                    no_op_items.append(
                        {
                            "item_index": item_index,
                            "action": "clear",
                            "offer_id": offer_id,
                            "requested_from_day": from_day,
                            "requested_to_day": to_day,
                            "status": "no_current_or_future_days",
                            "ignored_past_days": list(
                                range(from_day, to_day + 1)
                            ),
                        }
                    )
                continue
            effective_from_day = (
                max(state.day, from_day) if action == "clear" else from_day
            )
            ignored_past_days = (
                list(range(from_day, min(to_day, state.day - 1) + 1))
                if action == "clear" and from_day < state.day
                else []
            )
            if ignored_past_days:
                warnings.append(
                    (
                        {
                            "warning_code": "historical_clear_range_ignored",
                            "item_index": item_index,
                            "offer_id": offer_id,
                            "ignored_from_day": from_day,
                            "ignored_to_day": effective_from_day - 1,
                            "effective_from_day": effective_from_day,
                            "effective_to_day": to_day,
                        }
                        if compact_actor_result
                        else {
                            "warning_code": "historical_clear_days_ignored",
                            "item_index": item_index,
                            "offer_id": offer_id,
                            "ignored_past_days": ignored_past_days,
                            "effective_from_day": effective_from_day,
                            "effective_to_day": to_day,
                        }
                    )
                )
            normalized.append(
                {
                    "item_index": item_index,
                    "action": action,
                    "offer_id": offer_id,
                    "from_day": effective_from_day,
                    "to_day": to_day,
                    **(
                        {"max_walk_in_occupied_units": int(maximum)}
                        if action == "set"
                        else {}
                    ),
                    **(
                        {
                            "requested_from_day": from_day,
                            "requested_to_day": to_day,
                        }
                        if ignored_past_days
                        else {}
                    ),
                }
            )

        overlap_conflicts: list[dict[str, Any]] = []
        by_offer: dict[str, list[dict[str, Any]]] = {}
        for row in normalized:
            by_offer.setdefault(str(row["offer_id"]), []).append(row)
        for offer_id, rows in sorted(by_offer.items()):
            for left_index, left in enumerate(rows):
                left_days = set(
                    range(int(left["from_day"]), int(left["to_day"]) + 1)
                )
                for right in rows[left_index + 1 :]:
                    overlap_days = sorted(
                        left_days.intersection(
                            range(
                                int(right["from_day"]),
                                int(right["to_day"]) + 1,
                            )
                        )
                    )
                    if not overlap_days:
                        continue
                    actions = [str(left["action"]), str(right["action"])]
                    if set(actions) == {"set", "clear"}:
                        guidance = (
                            "remove the clear item and submit only set; set is an "
                            "upsert that directly replaces an existing daily limit"
                        )
                    else:
                        guidance = (
                            "merge these items or submit non-overlapping date windows"
                        )
                    conflict = {
                        "offer_id": offer_id,
                        "item_indexes": sorted(
                            [int(left["item_index"]), int(right["item_index"])]
                        ),
                        "actions": actions,
                        "overlap_days": overlap_days,
                        "guidance": guidance,
                    }
                    overlap_conflicts.append(conflict)
                    for row in (left, right):
                        add_violation(
                            int(row["item_index"]),
                            "overlapping_batch_window",
                            "capacity items for one offer must not overlap",
                            offer_id=offer_id,
                            overlap_days=overlap_days,
                            conflicting_item_indexes=conflict["item_indexes"],
                            guidance=guidance,
                        )

        if violations_by_index:
            return ToolResult(
                False,
                {
                    "error": "walk_in_capacity_limit_batch_invalid",
                    "batch_applied": False,
                    "invalid_item_count": len(violations_by_index),
                    "invalid_items": [
                        {
                            "item_index": item_index,
                            "item": original_by_index[item_index],
                            "violations": violations_by_index[item_index],
                        }
                        for item_index in sorted(violations_by_index)
                    ],
                    "overlap_conflicts": overlap_conflicts,
                    "guidance": (
                        "set is an upsert; to modify a limit, submit set without "
                        "a preceding clear"
                    ),
                },
            )

        limits = {
            (row.offer_id, int(row.day)): int(row.max_walk_in_occupied_units)
            for row in state.walk_in_capacity_limits
        }
        original_limits = dict(limits)
        applied: list[dict[str, Any]] = []
        for row in normalized:
            public_row = dict(row)
            item_index = int(public_row.pop("item_index"))
            public_row["item_index"] = item_index
            for day in range(int(row["from_day"]), int(row["to_day"]) + 1):
                key = (str(row["offer_id"]), day)
                if row["action"] == "clear":
                    limits.pop(key, None)
                    continue
                maximum = int(row["max_walk_in_occupied_units"])
                limits[key] = maximum
                occupied = self._walk_in_occupied_units(
                    state, str(row["offer_id"]), day
                )
                if occupied > maximum:
                    warnings.append(
                        {
                            "warning_code": "existing_occupancy_exceeds_new_limit",
                            "item_index": item_index,
                            "offer_id": str(row["offer_id"]),
                            "day": day,
                            "existing_walk_in_occupied_units": occupied,
                            "max_walk_in_occupied_units": maximum,
                            "existing_contracts_cancelled": False,
                        }
                    )
            applied.append(public_row)
        state.walk_in_capacity_limits = [
            WalkInCapacityLimit(
                offer_id=offer_id,
                day=day,
                max_walk_in_occupied_units=maximum,
            )
            for (offer_id, day), maximum in sorted(limits.items())
        ]
        if applied:
            actions = state.current_day_ledger.setdefault(
                "agent_actions", _empty_agent_actions()
            )
            actions.setdefault("walk_in_capacity_limit_changes", []).extend(
                dict(row) for row in applied
            )
        if compact_actor_result:
            touched_offers = {
                str(row["offer_id"])
                for row in values
                if isinstance(row, dict) and "offer_id" in row
            }
            compact_warnings: list[dict[str, Any]] = []
            occupancy_warnings: dict[
                tuple[int, str, int], list[tuple[int, int]]
            ] = {}
            for warning in warnings:
                if warning.get("warning_code") != "existing_occupancy_exceeds_new_limit":
                    compact_warnings.append(warning)
                    continue
                key = (
                    int(warning["item_index"]),
                    str(warning["offer_id"]),
                    int(warning["max_walk_in_occupied_units"]),
                )
                occupancy_warnings.setdefault(key, []).append(
                    (
                        int(warning["day"]),
                        int(warning["existing_walk_in_occupied_units"]),
                    )
                )
            for (item_index, offer_id, maximum), rows in sorted(
                occupancy_warnings.items()
            ):
                rows.sort()
                start = previous = rows[0][0]
                peak = rows[0][1]
                for day, occupied in rows[1:] + [(rows[-1][0] + 2, -1)]:
                    if day == previous + 1:
                        previous = day
                        peak = max(peak, occupied)
                        continue
                    compact_warnings.append(
                        {
                            "warning_code": "existing_occupancy_exceeds_new_limit",
                            "item_index": item_index,
                            "offer_id": offer_id,
                            "from_day": start,
                            "to_day": previous,
                            "peak_existing_walk_in_occupied_units": peak,
                            "max_walk_in_occupied_units": maximum,
                            "existing_contracts_cancelled": False,
                        }
                    )
                    start = previous = day
                    peak = occupied
            compact_warnings.sort(
                key=lambda row: (
                    int(row.get("item_index", -1)),
                    str(row.get("warning_code", "")),
                    int(row.get("from_day", row.get("ignored_from_day", -1))),
                )
            )
            return ToolResult(
                True,
                {
                    "set_semantics": "upsert",
                    "batch_atomic": True,
                    "state_unchanged": limits == original_limits,
                    "applied": applied,
                    "effective_limit_intervals": (
                        self._compressed_walk_in_capacity_limit_intervals(
                            state,
                            offer_ids=touched_offers,
                            from_day=state.day,
                            to_day=state.max_days,
                        )
                    ),
                    "warnings": compact_warnings,
                    "existing_contracts_cancelled": False,
                },
            )
        return ToolResult(
            True,
            {
                "set_semantics": "upsert",
                "batch_atomic": True,
                "applied": applied,
                "no_op_items": no_op_items,
                "configured_limit_intervals": (
                    self._compressed_walk_in_capacity_limit_intervals(state)
                ),
                "warnings": warnings,
                "existing_contracts_cancelled": False,
            },
        )

    def set_walk_in_capacity_limits(
        self, state: WorldState, values: list[dict[str, Any]]
    ) -> ToolResult:
        """Atomically overwrite or clear exact-offer daily walk-in caps."""

        if not self._v24_reservation_operations():
            return ToolResult(False, {"error": "walk-in capacity limits are unavailable"})
        if self._v28_capacity_summary_audit():
            return self._v28_set_walk_in_capacity_limits(state, values)
        if not values:
            return ToolResult(False, {"error": "items must not be empty"})
        normalized: list[dict[str, Any]] = []
        touched: dict[str, set[int]] = {}
        for row in values:
            if not isinstance(row, dict):
                return ToolResult(False, {"error": "every capacity item must be an object"})
            unexpected = sorted(
                set(row)
                - {
                    "action",
                    "offer_id",
                    "from_day",
                    "to_day",
                    "max_walk_in_occupied_units",
                }
            )
            if unexpected:
                return ToolResult(False, {"error": f"unexpected capacity fields: {unexpected}"})
            action = str(row.get("action", ""))
            offer_id = str(row.get("offer_id", ""))
            from_day = row.get("from_day")
            to_day = row.get("to_day")
            if action not in {"set", "clear"}:
                return ToolResult(False, {"error": "action must be set or clear"})
            if offer_id not in self.offers or self.offers[offer_id].asset_kind != "durable":
                return ToolResult(False, {"error": f"unknown rental offer_id: {offer_id}"})
            if (
                isinstance(from_day, bool)
                or not isinstance(from_day, int)
                or isinstance(to_day, bool)
                or not isinstance(to_day, int)
                or from_day < state.day
                or to_day < from_day
                or to_day > state.max_days
            ):
                return ToolResult(
                    False,
                    {
                        "error": (
                            "capacity window must satisfy current_day <= from_day "
                            "<= to_day <= runoff_end_day"
                        ),
                        "current_day": state.day,
                        "runoff_end_day": state.max_days,
                    },
                )
            maximum = row.get("max_walk_in_occupied_units")
            if action == "set":
                if (
                    isinstance(maximum, bool)
                    or not isinstance(maximum, int)
                    or not 0 <= maximum <= 10_000
                ):
                    return ToolResult(
                        False,
                        {
                            "error": (
                                "set requires max_walk_in_occupied_units as an "
                                "integer in [0, 10000]"
                            )
                        },
                    )
            elif "max_walk_in_occupied_units" in row:
                return ToolResult(
                    False,
                    {"error": "clear must omit max_walk_in_occupied_units"},
                )
            days = set(range(int(from_day), int(to_day) + 1))
            overlap = touched.setdefault(offer_id, set()).intersection(days)
            if overlap:
                return ToolResult(
                    False,
                    {
                        "error": "capacity items for one offer must not overlap",
                        "offer_id": offer_id,
                        "overlap_days": sorted(overlap),
                    },
                )
            touched[offer_id].update(days)
            normalized.append(
                {
                    "action": action,
                    "offer_id": offer_id,
                    "from_day": int(from_day),
                    "to_day": int(to_day),
                    **(
                        {"max_walk_in_occupied_units": int(maximum)}
                        if action == "set"
                        else {}
                    ),
                }
            )

        limits = {
            (row.offer_id, int(row.day)): int(row.max_walk_in_occupied_units)
            for row in state.walk_in_capacity_limits
        }
        warnings: list[dict[str, Any]] = []
        for row in normalized:
            key_days = range(int(row["from_day"]), int(row["to_day"]) + 1)
            for day in key_days:
                key = (str(row["offer_id"]), day)
                if row["action"] == "clear":
                    limits.pop(key, None)
                    continue
                maximum = int(row["max_walk_in_occupied_units"])
                limits[key] = maximum
                occupied = self._walk_in_occupied_units(
                    state, str(row["offer_id"]), day
                )
                if occupied > maximum:
                    warnings.append(
                        {
                            "offer_id": str(row["offer_id"]),
                            "day": day,
                            "existing_walk_in_occupied_units": occupied,
                            "max_walk_in_occupied_units": maximum,
                            "existing_contracts_cancelled": False,
                        }
                    )
        state.walk_in_capacity_limits = [
            WalkInCapacityLimit(
                offer_id=offer_id,
                day=day,
                max_walk_in_occupied_units=maximum,
            )
            for (offer_id, day), maximum in sorted(limits.items())
        ]
        actions = state.current_day_ledger.setdefault(
            "agent_actions", _empty_agent_actions()
        )
        actions.setdefault("walk_in_capacity_limit_changes", []).extend(
            dict(row) for row in normalized
        )
        return ToolResult(
            True,
            {
                "applied": normalized,
                "configured_limit_intervals": (
                    self._compressed_walk_in_capacity_limit_intervals(state)
                ),
                "warnings": warnings,
                "existing_contracts_cancelled": False,
            },
        )

    def repair_or_retire(
        self, state: WorldState, repairs: Iterable[str], retirements: Iterable[str]
    ) -> ToolResult:
        repair_ids = list(dict.fromkeys(map(str, repairs)))
        retire_ids = list(dict.fromkeys(map(str, retirements)))
        if set(repair_ids) & set(retire_ids):
            return ToolResult(False, {"error": "an asset cannot be repaired and retired together"})
        by_id = {asset.asset_id: asset for asset in state.assets}
        targets = repair_ids + retire_ids
        if not targets:
            return ToolResult(False, {"error": "select at least one damaged asset"})
        for asset_id in targets:
            if self._v31_warehouse_capacity() and asset_id in by_id and not self._warehouse_asset_held(by_id[asset_id]):
                return ToolResult(False, {"error": f"asset has already physically left the warehouse: {asset_id}"})
            if asset_id not in by_id or by_id[asset_id].status != "damaged":
                return ToolResult(False, {"error": f"asset is not damaged: {asset_id}"})
            if (
                asset_id in retire_ids
                and by_id[asset_id].ownership_type == "supplier_trial"
            ):
                return ToolResult(False, {
                    "error": "supplier-owned trial assets cannot be retired; repair and return them",
                    "asset_id": asset_id,
                    "state_unchanged": True,
                })
        cost = round(sum(self.repair_cost(by_id[asset_id]) for asset_id in repair_ids), 2)
        if cost > state.cash + 1e-9:
            return ToolResult(False, {"error": "insufficient cash", "required": cost, "cash": state.cash})
        for asset_id in repair_ids:
            asset = by_id[asset_id]
            asset.status = "repairing"
            asset.ready_day = state.day + 2
            self._record_rental_activity(
                state,
                "repair_started",
                asset,
                repair_cost=self.repair_cost(asset),
                ready_day=asset.ready_day,
            )
        for asset_id in retire_ids:
            asset = by_id[asset_id]
            asset.status = "retired"
            asset.ready_day = None
            self._register_durable_capital_exit(
                state, asset, cause="manager_retirement"
            )
            self._record_rental_activity(state, "retired_by_agent", asset)
            self._dispose_warehouse_asset(state, asset, reason="manager_retirement")
        state.cash = round(state.cash - cost, 2)
        state.cumulative_repair_cost = round(state.cumulative_repair_cost + cost, 2)
        self._add(state.current_day_ledger["actions"], "repair_cost", cost)
        actions = state.current_day_ledger.setdefault(
            "agent_actions", _empty_agent_actions()
        )
        actions.setdefault("repairs", []).extend(
            {
                "asset_id": asset_id,
                "offer_id": by_id[asset_id].offer_id,
                "product_type": by_id[asset_id].service_line,
                "repair_cost": self.repair_cost(by_id[asset_id]),
                "ready_day": by_id[asset_id].ready_day,
            }
            for asset_id in repair_ids
        )
        actions.setdefault("retirements", []).extend(
            {
                "asset_id": asset_id,
                "offer_id": by_id[asset_id].offer_id,
                "product_type": by_id[asset_id].service_line,
            }
            for asset_id in retire_ids
        )
        return ToolResult(
            True,
            {"repairing": repair_ids, "retired": retire_ids, "cost": cost, "cash": state.cash,
             **({"durable_capital": self._durable_acquisition_budget_summary(state)}
                if self._v40_revolving_durable_capital() and retire_ids else {}),
             **({"physically_disposed_asset_ids": retire_ids, "warehouse_capacity": self._warehouse_summary(state)}
                if self._v31_warehouse_capacity() else {})},
        )

    def schedule_maintenance(
        self, state: WorldState, asset_ids: Iterable[str]
    ) -> ToolResult:
        if not self.config.preventive_maintenance_enabled:
            return ToolResult(False, {"error": "preventive maintenance is disabled"})
        if self.config.lifecycle_economics_enabled:
            return self._schedule_lifecycle_maintenance(state, asset_ids)
        identifiers = list(dict.fromkeys(map(str, asset_ids)))
        if not identifiers:
            return ToolResult(False, {"error": "asset_ids must not be empty"})
        by_id = {asset.asset_id: asset for asset in state.assets}
        for asset_id in identifiers:
            if asset_id not in by_id:
                return ToolResult(False, {"error": f"unknown asset_id: {asset_id}"})
            asset = by_id[asset_id]
            if asset.status not in {"available", "maintenance_hold"}:
                return ToolResult(
                    False,
                    {"error": f"asset is not available for maintenance: {asset_id}"},
                )
            trial = self._supplier_trial_record_for_asset(state, asset)
            if trial is not None and trial.return_pending:
                return ToolResult(False, {
                    "error": f"supplier-trial asset is pending return: {asset_id}",
                    "state_unchanged": True,
                })
            if self._v31_warehouse_capacity() and not self._warehouse_asset_held(asset):
                return ToolResult(False, {"error": f"asset has physically left ownership: {asset_id}"})
            earliest = (
                1
                if asset.next_inspection_due_day is None
                else asset.next_inspection_due_day
                - int(self.config.maintenance_due_window_days)
            )
            if state.day < earliest:
                return ToolResult(
                    False,
                    {
                        "error": f"maintenance window has not opened: {asset_id}",
                        "window_opens_day": earliest,
                    },
                )
        costs = {
            asset_id: round(
                max(
                    2.0,
                    float(self.config.maintenance_cost_rate)
                    * by_id[asset_id].unit_cost,
                ),
                2,
            )
            for asset_id in identifiers
        }
        total = round(sum(costs.values()), 2)
        if total > state.cash + 1e-9:
            return ToolResult(
                False,
                {"error": "insufficient cash", "required": total, "cash": state.cash},
            )
        ready_day = state.day + int(self.config.maintenance_duration_days)
        for asset_id in identifiers:
            asset = by_id[asset_id]
            asset.status = "maintenance"
            asset.ready_day = ready_day
            asset.maintenance_started_day = state.day
            self._record_rental_activity(
                state,
                "maintenance_started",
                asset,
                maintenance_cost=costs[asset_id],
                ready_day=ready_day,
                inspection_due_day=asset.next_inspection_due_day,
            )
        state.cash = round(state.cash - total, 2)
        state.cumulative_maintenance_cost = round(
            state.cumulative_maintenance_cost + total, 2
        )
        state.cumulative_maintenance_actions += len(identifiers)
        self._add(state.current_day_ledger["actions"], "maintenance_cost", total)
        actions = state.current_day_ledger.setdefault(
            "agent_actions", _empty_agent_actions()
        )
        actions.setdefault("maintenance", []).extend(
            {
                "asset_id": asset_id,
                "offer_id": by_id[asset_id].offer_id,
                "cost": costs[asset_id],
                "ready_day": ready_day,
            }
            for asset_id in identifiers
        )
        return ToolResult(
            True,
            {
                "scheduled_asset_ids": identifiers,
                "ready_day": ready_day,
                "cost": total,
                "cash": state.cash,
            },
        )

    def _schedule_lifecycle_maintenance(
        self, state: WorldState, asset_ids: Iterable[str]
    ) -> ToolResult:
        assert self.lifecycle_scales is not None
        identifiers = list(dict.fromkeys(map(str, asset_ids)))
        if not identifiers:
            return ToolResult(False, {"error": "asset_ids must not be empty"})
        by_id = {asset.asset_id: asset for asset in state.assets}
        statuses: dict[str, dict[str, Any]] = {}
        schedules: dict[str, dict[str, float | int]] = {}
        for asset_id in identifiers:
            asset = by_id.get(asset_id)
            if asset is None:
                return ToolResult(False, {"error": f"unknown asset_id: {asset_id}"})
            if self._v31_warehouse_capacity() and not self._warehouse_asset_held(asset):
                return ToolResult(False, {"error": f"asset has physically left ownership: {asset_id}"})
            if asset.status not in {"available", "maintenance_hold"}:
                return ToolResult(
                    False,
                    {"error": f"asset is not available for maintenance: {asset_id}"},
                )
            trial = self._supplier_trial_record_for_asset(state, asset)
            if trial is not None and trial.return_pending:
                return ToolResult(False, {
                    "error": f"supplier-trial asset is pending return: {asset_id}",
                    "state_unchanged": True,
                })
            status = self._lifecycle_maintenance_status(asset)
            statuses[asset_id] = status
            if not bool(status["window_open"]):
                return ToolResult(
                    False,
                    {
                        "error": f"maintenance window has not opened: {asset_id}",
                        "combined_wear": round(self._combined_wear(asset), 6),
                        "maintenance_due_wear": status["maintenance_due_wear"],
                    },
                )
            severity = str(
                self._lifecycle_factor(asset.offer_id)["d_damage"]["severity_tier"]
            )
            schedules[asset_id] = self._maintenance_service_terms(asset) if self._v48_service_guarantee() else maintenance_terms(
                replacement_cost=asset.unit_cost,
                severity_tier=severity,
                scales=self.lifecycle_scales,
            )
        total = round(
            sum(float(row["cash_cost"]) for row in schedules.values()), 2
        )
        if total > state.cash + 1e-9:
            return ToolResult(
                False, {"error": "insufficient cash", "required": total, "cash": state.cash}
            )
        rows: list[dict[str, Any]] = []
        for asset_id in identifiers:
            asset = by_id[asset_id]
            terms = schedules[asset_id]
            ready_day = state.day + int(terms["duration_days"])
            asset.status = "maintenance"
            asset.ready_day = ready_day
            asset.maintenance_started_day = state.day
            asset.next_inspection_due_day = None
            row = {
                "asset_id": asset_id,
                "offer_id": asset.offer_id,
                "cost": round(float(terms["cash_cost"]), 2),
                "ready_day": ready_day,
                "combined_wear": round(self._combined_wear(asset), 6),
                "wear_since_maintenance": statuses[asset_id][
                    "wear_since_maintenance"
                ],
            }
            rows.append(row)
            self._record_rental_activity(
                state,
                "maintenance_started",
                asset,
                maintenance_cost=row["cost"],
                ready_day=ready_day,
                combined_wear=row["combined_wear"],
                wear_since_maintenance=row["wear_since_maintenance"],
            )
        state.cash = round(state.cash - total, 2)
        state.cumulative_maintenance_cost = round(
            state.cumulative_maintenance_cost + total, 2
        )
        state.cumulative_maintenance_actions += len(identifiers)
        self._add(state.current_day_ledger["actions"], "maintenance_cost", total)
        actions = state.current_day_ledger.setdefault(
            "agent_actions", _empty_agent_actions()
        )
        actions.setdefault("maintenance", []).extend(rows)
        ready_days = sorted({int(row["ready_day"]) for row in rows})
        return ToolResult(
            True,
            {
                "scheduled_asset_ids": identifiers,
                "schedules": rows,
                "ready_day": ready_days[0] if len(ready_days) == 1 else None,
                "cost": total,
                "cash": state.cash,
            },
        )

    def repair_cost(self, asset: RentalAsset) -> float:
        if self.config.lifecycle_economics_enabled:
            if asset.pending_repair_cost is None:
                raise ValueError(
                    f"damaged lifecycle asset has no pending repair cost: {asset.asset_id}"
                )
            return round(float(asset.pending_repair_cost), 2)
        return round(max(10.0, 0.20 * asset.unit_cost), 2)

    @staticmethod
    def damage_deduction(asset: RentalAsset) -> float:
        return round(max(5.0, 0.10 * asset.unit_cost), 2)

    @staticmethod
    def public_cleaning_rule(product_type: str) -> dict[str, float]:
        base_cost, extra_per_day = CLEANING_COST_RULES[product_type]
        return {
            "base_cost": base_cost,
            "extra_per_additional_day": extra_per_day,
        }

    def public_offer_cleaning_rule(
        self, product_type: str, *, offer_id: str
    ) -> dict[str, float]:
        if not self.config.lifecycle_economics_enabled:
            return self.public_cleaning_rule(product_type)
        assert self.lifecycle_scales is not None
        projection = self._lifecycle_projection(offer_id)
        one_day_work = float(projection["one_day_cleaning_work_points"])
        return {
            "one_day_work_points": round(one_day_work, 6),
            "extra_work_fraction_per_additional_day": float(
                self.lifecycle_scales.cleaning_extra_work_fraction_per_additional_day
            ),
            "cash_per_work_point": float(
                self.lifecycle_scales.cleaning_cash_per_work_point
            ),
        }

    def cleaning_cost(self, asset: RentalAsset, duration_days: int) -> float:
        if duration_days < 1:
            raise ValueError("duration_days must be at least 1")
        if self.config.lifecycle_economics_enabled:
            return float(
                self._lifecycle_cleaning_terms(
                    asset, duration_days=duration_days
                )["cash_cost"]
            )
        base_cost, extra_per_day = CLEANING_COST_RULES[asset.service_line]
        return round(base_cost + extra_per_day * (duration_days - 1), 2)

    def cleaning_work_points(
        self,
        product_type: str,
        duration_days: int,
        *,
        offer_id: str | None = None,
    ) -> float:
        if duration_days < 1:
            raise ValueError("duration_days must be at least 1")
        if self.config.lifecycle_economics_enabled:
            if offer_id is None:
                raise ValueError("lifecycle cleaning work requires offer_id")
            projection = self._lifecycle_projection(offer_id)
            assert self.lifecycle_scales is not None
            return float(
                cleaning_terms(
                    one_day_work_points=float(
                        projection["one_day_cleaning_work_points"]
                    ),
                    duration_days=duration_days,
                    scales=self.lifecycle_scales,
                )["work_points"]
            )
        base, extra = self.rules.cleaning_work_points[product_type]
        if offer_id is not None:
            offer = self.offers.get(offer_id)
            if offer is None:
                raise ValueError(f"unknown offer_id: {offer_id}")
            if offer.cleaning_work_points is not None:
                base = float(offer.cleaning_work_points)
        return round(base + extra * (duration_days - 1), 2)

    def _process_cleaning_capacity(self, state: WorldState, messages: list[str]) -> None:
        state.cleaning_queue.sort(key=lambda row: (row.enqueued_day, row.asset_id))
        completed_ids: set[str] = set()
        for job, applied, completes in self._cleaning_capacity_plan(
            state, processing_day=state.day
        ):
            if state.done:
                break
            job.remaining_work_points = round(job.remaining_work_points - applied, 6)
            if not completes:
                continue
            asset = next(row for row in state.assets if row.asset_id == job.asset_id)
            paid = self._charge(
                state,
                job.cleaning_cost,
                reason=f"资产 {job.asset_id} 的清洁费",
            )
            # _charge consumes the remaining cash and records the unpaid
            # shortfall on default. Record the full obligation as well, just
            # as other bankruptcy-adjusted cash-flow categories do.
            state.cumulative_cleaning_cost = round(
                state.cumulative_cleaning_cost + job.cleaning_cost, 2
            )
            self._add(
                state.current_day_ledger["morning"], "cleaning_cost", job.cleaning_cost
            )
            if not paid:
                # Still remove earlier successfully completed jobs below;
                # returning here used to leave them duplicated in the queue.
                break
            asset.status = "available"
            asset.ready_day = None
            self._record_rental_activity(
                state,
                "cleaning_completed",
                asset,
                enqueued_day=job.enqueued_day,
                work_points=(
                    round(
                        job.cleaning_cost
                        / float(self.lifecycle_scales.cleaning_cash_per_work_point),
                        6,
                    )
                    if self.config.lifecycle_economics_enabled
                    and self.lifecycle_scales is not None
                    else self.cleaning_work_points(
                        job.product_type,
                        job.duration_days,
                        offer_id=job.offer_id,
                    )
                ),
                cleaning_cost=job.cleaning_cost,
            )
            completed_ids.add(job.asset_id)
            messages.append(f"资产 {asset.asset_id} 已完成清洁并恢复可租。")
        if completed_ids:
            state.cleaning_queue = [
                job for job in state.cleaning_queue if job.asset_id not in completed_ids
            ]

    def _cleaning_capacity_plan(
        self, state: WorldState, *, processing_day: int
    ) -> list[tuple[CleaningJob, float, bool]]:
        """Plan one morning's FIFO work without mutating jobs or exposing internals."""

        capacity = float(self.rules.cleaning_capacity_points)
        plan: list[tuple[CleaningJob, float, bool]] = []
        for job in sorted(
            state.cleaning_queue, key=lambda row: (row.enqueued_day, row.asset_id)
        ):
            if capacity <= 1e-9:
                break
            if job.enqueued_day >= processing_day:
                continue
            applied = min(capacity, float(job.remaining_work_points))
            remaining = round(float(job.remaining_work_points) - applied, 6)
            capacity = round(capacity - applied, 6)
            plan.append((job, applied, remaining <= 1e-9))
        return plan

    def scheduled_cleaning_charges_due_next_morning(
        self, state: WorldState
    ) -> float:
        return round(
            sum(
                float(job.cleaning_cost)
                for job, _applied, completes in self._cleaning_capacity_plan(
                    state, processing_day=state.day + 1
                )
                if completes
            ),
            2,
        )

    @staticmethod
    def _used_life_fraction(asset: RentalAsset) -> float:
        daily = asset.used_life_days / max(1, asset.service_life_limit_days)
        if asset.service_life_limit_cycles is None:
            return max(0.0, daily)
        turnover = asset.used_rental_cycles / max(
            1, asset.service_life_limit_cycles
        )
        return max(0.0, daily + turnover)

    @classmethod
    def _remaining_life_ratio(cls, asset: RentalAsset) -> float:
        return max(0.0, min(1.0, 1.0 - cls._used_life_fraction(asset)))

    @classmethod
    def _service_life_exhausted(cls, asset: RentalAsset) -> bool:
        return cls._used_life_fraction(asset) >= 1.0 - 1e-12

    def _healthy_secondhand_grade_and_quote(
        self, asset: RentalAsset
    ) -> tuple[str, float]:
        remaining = self._remaining_life_ratio(asset)
        for grade, threshold, cost_rate in SECONDHAND_GRADE_SCHEDULE:
            if remaining >= threshold:
                return grade, round(asset.unit_cost * cost_rate, 2)
        raise AssertionError("secondhand grade schedule must cover zero remaining life")

    def secondhand_quote(
        self, asset: RentalAsset, *, terminal: bool = False
    ) -> dict[str, Any] | None:
        if asset.ownership_type != "owned":
            return None
        if self._v31_warehouse_capacity() and not self._warehouse_asset_held(asset):
            return None
        if terminal:
            if asset.status not in TERMINAL_HEALTHY_STATUSES | {"damaged"}:
                return None
        elif asset.status not in SECONDHAND_SELLABLE_STATUSES:
            return None
        if self.config.lifecycle_economics_enabled:
            effective_condition = (
                asset.repair_target_condition_ratio
                if asset.status == "repairing"
                and asset.repair_target_condition_ratio is not None
                else asset.condition_ratio
            )
            quote = round(
                self._lifecycle_nominal_residual(
                    asset, condition_ratio=effective_condition
                ),
                2,
            )
            return {
                "asset_id": asset.asset_id,
                "offer_id": asset.offer_id,
                "name": self.offers[asset.offer_id].name,
                "status": asset.status,
                "inspection_grade": (
                    "damaged" if asset.status == "damaged" else "condition_wear"
                ),
                "condition_ratio": round(float(effective_condition), 6),
                "combined_wear": round(self._combined_wear(asset), 6),
                "offer_price": quote,
            }
        healthy_grade, healthy_quote = self._healthy_secondhand_grade_and_quote(asset)
        if asset.status == "damaged":
            grade = "damaged"
            quote = round(
                max(
                    SECONDHAND_PARTS_FLOOR_RATE * asset.unit_cost,
                    healthy_quote - self.repair_cost(asset),
                ),
                2,
            )
        else:
            grade = healthy_grade
            quote = healthy_quote
        return {
            "asset_id": asset.asset_id,
            "offer_id": asset.offer_id,
            "name": self.offers[asset.offer_id].name,
            "status": asset.status,
            "inspection_grade": grade,
            "offer_price": quote,
        }

    @staticmethod
    def _money_decimal(value: Decimal) -> Decimal:
        return value.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)

    @classmethod
    def retail_liquidation_unit_quote(cls, average_unit_cost: float) -> float:
        value = Decimal(str(average_unit_cost)) * Decimal(str(RETAIL_SALVAGE_RATE))
        return float(cls._money_decimal(value))

    @classmethod
    def retail_liquidation_proceeds(
        cls, average_unit_cost: float, units: int
    ) -> float:
        unit_quote = Decimal(str(cls.retail_liquidation_unit_quote(average_unit_cost)))
        return float(cls._money_decimal(unit_quote * int(units)))

    def retail_inventory_liquidation_value(self, state: WorldState) -> float:
        nominal = sum(
            self.retail_liquidation_proceeds(
                float(inventory["avg_unit_cost"]), int(inventory["quantity"])
            )
            for inventory in state.retail_inventory.values()
        )
        actual, _ = self._depth_adjusted_pool_proceeds(
            state, "retail_item", nominal
        )
        return actual

    def _depth_integral(self, state: WorldState, pool: str, volume: float) -> float:
        volume = max(0.0, float(volume))
        capacity = max(
            0.01, float(state.liquidation_depth_capacity.get(pool, 0.0))
        )
        end_multiple = float(self.config.liquidation_depth_end_multiple)
        floor_rate = float(self.config.liquidation_depth_floor_rate)
        end = capacity * end_multiple
        if volume <= capacity:
            return volume
        slope = (1.0 - floor_rate) / max(0.01, end - capacity)
        middle_width = min(volume, end) - capacity
        value = capacity + middle_width - 0.5 * slope * middle_width**2
        if volume > end:
            value += floor_rate * (volume - end)
        return value

    def _recent_depth_volume(self, state: WorldState, pool: str) -> float:
        start = state.day - int(self.config.liquidation_depth_window_days) + 1
        return sum(
            float(row.get("nominal_proceeds", 0.0))
            for row in state.liquidation_market_history
            if str(row.get("pool")) == pool and int(row.get("day", 0)) >= start
        )

    def _depth_adjusted_pool_proceeds(
        self, state: WorldState, pool: str, nominal: float
    ) -> tuple[float, dict[str, Any]]:
        nominal = round(max(0.0, float(nominal)), 2)
        if not self.config.liquidation_depth_enabled or nominal <= 0:
            return nominal, {
                "enabled": False,
                "pool": pool,
                "prior_rolling_nominal_volume": 0.0,
                "full_depth_capacity": None,
                "batch_nominal_proceeds": nominal,
                "batch_actual_proceeds": nominal,
                "batch_average_rate": 1.0,
            }
        prior = self._recent_depth_volume(state, pool)
        actual = self._depth_integral(
            state, pool, prior + nominal
        ) - self._depth_integral(state, pool, prior)
        actual = float(self._money_decimal(Decimal(str(max(0.0, actual)))))
        return actual, {
            "enabled": True,
            "pool": pool,
            "rolling_window_days": int(self.config.liquidation_depth_window_days),
            "prior_rolling_nominal_volume": round(prior, 2),
            "full_depth_capacity": round(
                float(state.liquidation_depth_capacity[pool]), 2
            ),
            "batch_nominal_proceeds": nominal,
            "batch_actual_proceeds": actual,
            "batch_average_rate": round(actual / nominal, 6),
        }

    @staticmethod
    def _allocate_depth_proceeds(
        rows: list[tuple[str, float]], target: float
    ) -> dict[str, float]:
        if not rows:
            return {}
        total = sum(max(0.0, float(value)) for _, value in rows)
        if total <= 0:
            return {key: 0.0 for key, _ in rows}
        target_cents = int(round(float(target) * 100))
        raw = [target_cents * max(0.0, float(value)) / total for _, value in rows]
        cents = [math.floor(value) for value in raw]
        remainder = target_cents - sum(cents)
        order = sorted(
            range(len(rows)),
            key=lambda index: (-(raw[index] - cents[index]), rows[index][0]),
        )
        for index in order[:remainder]:
            cents[index] += 1
        return {
            rows[index][0]: cents[index] / 100.0 for index in range(len(rows))
        }

    def _liquidation_request_validation(
        self,
        state: WorldState,
        rental_asset_ids: Iterable[str],
        retail_items: Iterable[dict[str, Any]],
    ) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
        identifiers = list(map(str, rental_asset_ids))
        raw_items = list(retail_items)
        items = [dict(row) if isinstance(row, dict) else {} for row in raw_items]
        errors: list[dict[str, Any]] = [
            {
                "error": "every retail liquidation item must be an object",
                "item_type": "retail_item",
                "item_index": index,
                "status": "invalid_object",
            }
            for index, row in enumerate(raw_items)
            if not isinstance(row, dict)
        ]
        if not identifiers and not items:
            errors.append(
                {
                    "error": "select at least one rental asset or retail inventory item",
                    "item_type": "request",
                    "status": "empty_request",
                }
            )
        if len(identifiers) > 25:
            errors.append(
                {
                    "error": "at most 25 rental assets may be submitted at once",
                    "item_type": "request",
                    "status": "too_many_rental_assets",
                    "submitted_count": len(identifiers),
                }
            )
        if len(items) > 25:
            errors.append(
                {
                    "error": "at most 25 retail inventory lines may be submitted at once",
                    "item_type": "request",
                    "status": "too_many_retail_items",
                    "submitted_count": len(items),
                }
            )
        if len(set(identifiers)) != len(identifiers):
            errors.append(
                {
                    "error": "rental_asset_ids must not contain duplicates",
                    "item_type": "request",
                    "status": "duplicate_rental_asset_id",
                }
            )
        retail_offer_ids = [str(row.get("offer_id", "")) for row in items]
        if len(set(retail_offer_ids)) != len(retail_offer_ids):
            errors.append(
                {
                    "error": "retail_items must not repeat an offer_id",
                    "item_type": "request",
                    "status": "duplicate_retail_offer_id",
                }
            )
        by_id = {asset.asset_id: asset for asset in state.assets}
        for asset_id in identifiers:
            asset = by_id.get(asset_id)
            if asset is None:
                errors.append(
                    {
                        "error": f"unknown asset_id: {asset_id}",
                        "item_type": "rental_asset",
                        "asset_id": asset_id,
                        "status": "unknown_asset",
                    }
                )
            elif self.secondhand_quote(asset) is None:
                errors.append(
                    {
                        "error": f"asset is not sellable in status {asset.status}: {asset_id}",
                        "item_type": "rental_asset",
                        "asset_id": asset_id,
                        "status": asset.status,
                    }
                )
        for index, (raw_item, item, offer_id) in enumerate(
            zip(raw_items, items, retail_offer_ids)
        ):
            if not isinstance(raw_item, dict):
                continue
            offer = self.offers.get(offer_id)
            if offer is None:
                errors.append(
                    {
                        "error": f"unknown offer_id: {offer_id}",
                        "item_type": "retail_item",
                        "item_index": index,
                        "offer_id": offer_id,
                        "status": "unknown_offer",
                        "requested_units": item.get("units"),
                        "on_hand_units": 0,
                    }
                )
                continue
            inventory = state.retail_inventory.get(offer_id)
            on_hand = int(inventory.get("quantity", 0)) if inventory else 0
            if offer.asset_kind != "consumable":
                errors.append(
                    {
                        "error": f"offer is not retail inventory: {offer_id}",
                        "item_type": "retail_item",
                        "item_index": index,
                        "offer_id": offer_id,
                        "status": "not_retail_inventory",
                        "kind": self.public_offer(offer_id)["kind"],
                        "requested_units": item.get("units"),
                        "on_hand_units": on_hand,
                    }
                )
                continue
            units = item.get("units")
            if isinstance(units, bool) or not isinstance(units, int) or units < 1:
                errors.append(
                    {
                        "error": f"retail units must be a positive integer: {offer_id}",
                        "item_type": "retail_item",
                        "item_index": index,
                        "offer_id": offer_id,
                        "status": "invalid_units",
                        "requested_units": units,
                        "on_hand_units": on_hand,
                    }
                )
                continue
            if units > on_hand:
                errors.append(
                    {
                        "error": (
                            "only delivered on-hand retail inventory may be liquidated; "
                            f"requested {units}, available {on_hand}: {offer_id}"
                        ),
                        "item_type": "retail_item",
                        "item_index": index,
                        "offer_id": offer_id,
                        "status": "insufficient_on_hand",
                        "requested_units": units,
                        "on_hand_units": on_hand,
                    }
                )
        return identifiers, items, errors

    def _prepare_liquidation_request(
        self,
        state: WorldState,
        rental_asset_ids: Iterable[str],
        retail_items: Iterable[dict[str, Any]],
        *,
        aggregate_validation_errors: bool = False,
    ) -> ToolResult | dict[str, Any]:
        identifiers, items, errors = self._liquidation_request_validation(
            state, rental_asset_ids, retail_items
        )
        if errors:
            if aggregate_validation_errors:
                return ToolResult(
                    False,
                    {
                        "error": "liquidation quote request contains invalid items",
                        "invalid_item_count": len(errors),
                        "invalid_items": errors,
                    },
                )
            return ToolResult(False, errors[0])

        by_id = {asset.asset_id: asset for asset in state.assets}
        rental_quotes: list[tuple[RentalAsset, dict[str, Any]]] = []
        for asset_id in identifiers:
            asset = by_id[asset_id]
            quote = self.secondhand_quote(asset)
            assert quote is not None
            rental_quotes.append((asset, quote))

        retail_quotes: list[
            tuple[dict[str, float], dict[str, Any], float, float]
        ] = []
        for item in items:
            offer_id = str(item["offer_id"])
            offer = self.offers[offer_id]
            units = int(item["units"])
            inventory = state.retail_inventory.get(offer_id)
            on_hand = int(inventory.get("quantity", 0)) if inventory else 0
            assert inventory is not None
            average_cost = Decimal(str(inventory["avg_unit_cost"]))
            exact_book_cost = (average_cost * units).quantize(
                INVENTORY_COST_QUANTUM, rounding=ROUND_HALF_UP
            )
            unit_quote = Decimal(
                str(self.retail_liquidation_unit_quote(float(average_cost)))
            )
            exact_proceeds = self._money_decimal(unit_quote * units)
            exact_loss = max(Decimal("0"), exact_book_cost - exact_proceeds).quantize(
                INVENTORY_COST_QUANTUM, rounding=ROUND_HALF_UP
            )
            quote = {
                "offer_id": offer_id,
                "name": offer.name,
                "units": units,
                "on_hand_units": on_hand,
                "average_acquisition_unit_cost": round(float(average_cost), 4),
                "dealer_unit_quote": float(unit_quote),
                "book_cost": float(self._money_decimal(exact_book_cost)),
                "proceeds": float(exact_proceeds),
                "liquidation_loss": float(self._money_decimal(exact_loss)),
            }
            retail_quotes.append(
                (inventory, quote, float(exact_book_cost), float(exact_loss))
            )

        rental_nominal = round(
            sum(float(quote["offer_price"]) for _, quote in rental_quotes), 2
        )
        rental_actual, rental_market = self._depth_adjusted_pool_proceeds(
            state, "rental_asset", rental_nominal
        )
        rental_allocations = self._allocate_depth_proceeds(
            [
                (asset.asset_id, float(quote["offer_price"]))
                for asset, quote in rental_quotes
            ],
            rental_actual,
        )
        for asset, quote in rental_quotes:
            nominal = float(quote["offer_price"])
            if self.config.liquidation_depth_enabled:
                quote["nominal_offer_price"] = nominal
            quote["offer_price"] = rental_allocations.get(asset.asset_id, 0.0)
            if self.config.liquidation_depth_enabled:
                quote["depth_average_rate"] = rental_market["batch_average_rate"]

        retail_nominal = round(
            sum(float(quote["proceeds"]) for _, quote, _, _ in retail_quotes), 2
        )
        retail_actual, retail_market = self._depth_adjusted_pool_proceeds(
            state, "retail_item", retail_nominal
        )
        retail_allocations = self._allocate_depth_proceeds(
            [
                (str(quote["offer_id"]), float(quote["proceeds"]))
                for _, quote, _, _ in retail_quotes
            ],
            retail_actual,
        )
        adjusted_retail_quotes = []
        for inventory, quote, book_cost, _loss in retail_quotes:
            nominal = float(quote["proceeds"])
            actual = retail_allocations.get(str(quote["offer_id"]), 0.0)
            if self.config.liquidation_depth_enabled:
                quote["nominal_proceeds"] = nominal
            quote["proceeds"] = actual
            if self.config.liquidation_depth_enabled:
                quote["depth_average_rate"] = retail_market["batch_average_rate"]
            loss = max(0.0, float(book_cost) - actual)
            quote["liquidation_loss"] = round(loss, 2)
            adjusted_retail_quotes.append((inventory, quote, book_cost, loss))
        retail_quotes = adjusted_retail_quotes

        rental_proceeds = rental_actual
        retail_proceeds = retail_actual
        retail_book_cost = round(
            sum(book_cost for _, _, book_cost, _ in retail_quotes), 4
        )
        retail_loss = round(sum(loss for _, _, _, loss in retail_quotes), 4)
        return {
            "rental_quotes": rental_quotes,
            "retail_quotes": retail_quotes,
            "totals": {
                "rental_asset_sale_proceeds": rental_proceeds,
                "retail_liquidation_proceeds": retail_proceeds,
                "retail_liquidation_book_cost": retail_book_cost,
                "retail_liquidation_loss": retail_loss,
                "retail_units_liquidated": sum(
                    int(quote["units"]) for _, quote, _, _ in retail_quotes
                ),
                "total_cash_proceeds": round(rental_proceeds + retail_proceeds, 2),
                **(
                    {
                        "nominal_proceeds": round(
                            rental_nominal + retail_nominal, 2
                        ),
                        "depth_discount": round(
                            rental_nominal
                            + retail_nominal
                            - rental_proceeds
                            - retail_proceeds,
                            2,
                        ),
                    }
                    if self.config.liquidation_depth_enabled
                    else {}
                ),
            },
            "market_depth": {
                "rental_asset": rental_market,
                "retail_item": retail_market,
            },
        }

    def view_liquidation_quotes(
        self,
        state: WorldState,
        rental_asset_ids: Iterable[str],
        retail_items: Iterable[dict[str, Any]],
    ) -> ToolResult:
        unavailable = []
        if self._v51_recovery_integrity():
            identifiers, items, errors = self._liquidation_request_validation(
                state, rental_asset_ids, retail_items)
            # Syntax errors remain errors; holdings eligibility is query data.
            structural = [e for e in errors if e['item_type'] == 'request'
                          or e.get('status') in {'invalid_units', 'invalid_object'}]
            if structural:
                return ToolResult(False, {'error': 'invalid liquidation query',
                    'invalid_items': errors, 'invalid_item_count': len(errors)})
            unavailable = errors
            bad_assets = {e.get('asset_id') for e in errors if e['item_type'] == 'rental_asset'}
            bad_offers = {e.get('offer_id') for e in errors if e['item_type'] == 'retail_item'}
            rental_asset_ids = [a for a in identifiers if a not in bad_assets]
            retail_items = [i for i in items if str(i['offer_id']) not in bad_offers]
            if not rental_asset_ids and not retail_items:
                return ToolResult(True, {'valid_day': state.day, 'rental_assets': [],
                    'retail_items': [], 'unavailable_items': unavailable,
                    'totals_scope': 'eligible_items_only', 'totals': {
                        'rental_asset_sale_proceeds': 0.0, 'retail_liquidation_proceeds': 0.0,
                        'retail_liquidation_book_cost': 0.0, 'retail_liquidation_loss': 0.0,
                        'retail_units_liquidated': 0, 'total_cash_proceeds': 0.0}})
        prepared = self._prepare_liquidation_request(
            state,
            rental_asset_ids,
            retail_items,
            aggregate_validation_errors=True,
        )
        if isinstance(prepared, ToolResult):
            return prepared
        return ToolResult(
            True,
            {
                "valid_day": state.day,
                **({'unavailable_items': unavailable, 'totals_scope': 'eligible_items_only'}
                   if self._v51_recovery_integrity() else {}),
                "rental_assets": [
                    {**quote, "valid_day": state.day}
                    for _, quote in prepared["rental_quotes"]
                ],
                "retail_items": [
                    {**quote, "valid_day": state.day}
                    for _, quote, _, _ in prepared["retail_quotes"]
                ],
                "totals": {
                    key: round(value, 2) if isinstance(value, float) else value
                    for key, value in prepared["totals"].items()
                },
                **(
                    {"market_depth": prepared["market_depth"]}
                    if self.config.liquidation_depth_enabled
                    else {}
                ),
            },
        )

    def sell_to_liquidation_dealer(
        self,
        state: WorldState,
        rental_asset_ids: Iterable[str],
        retail_items: Iterable[dict[str, Any]],
    ) -> ToolResult:
        if state.done:
            return ToolResult(
                False, {"error": "the business is already closed and cannot liquidate holdings"}
            )
        prepared = self._prepare_liquidation_request(
            state, rental_asset_ids, retail_items
        )
        if isinstance(prepared, ToolResult):
            return prepared
        rental_quotes = prepared["rental_quotes"]
        retail_quotes = prepared["retail_quotes"]
        totals = prepared["totals"]
        market_depth = prepared["market_depth"]
        for asset, quote in rental_quotes:
            asset.status = "sold"
            asset.ready_day = None
            asset.sold_day = state.day
            asset.sale_grade = str(quote["inspection_grade"])
            asset.sale_proceeds = float(quote["offer_price"])
            self._register_durable_capital_exit(
                state, asset, cause="dealer_sale"
            )
            self._record_rental_activity(
                state,
                "asset_sold",
                asset,
                inspection_grade=asset.sale_grade,
                sale_proceeds=asset.sale_proceeds,
            )
        for inventory, quote, _, _ in retail_quotes:
            inventory["quantity"] = int(inventory["quantity"]) - int(quote["units"])

        state.cash = round(state.cash + float(totals["total_cash_proceeds"]), 2)
        state.cumulative_asset_sale_proceeds = round(
            state.cumulative_asset_sale_proceeds
            + float(totals["rental_asset_sale_proceeds"]),
            2,
        )
        state.cumulative_assets_sold += len(rental_quotes)
        state.cumulative_retail_liquidation_proceeds = round(
            state.cumulative_retail_liquidation_proceeds
            + float(totals["retail_liquidation_proceeds"]),
            2,
        )
        state.cumulative_retail_liquidation_book_cost = round(
            state.cumulative_retail_liquidation_book_cost
            + float(totals["retail_liquidation_book_cost"]),
            4,
        )
        state.cumulative_retail_liquidation_loss = round(
            state.cumulative_retail_liquidation_loss
            + float(totals["retail_liquidation_loss"]),
            4,
        )
        state.cumulative_retail_units_liquidated += int(
            totals["retail_units_liquidated"]
        )
        self._add(
            state.current_day_ledger["actions"],
            "asset_sale_proceeds",
            float(totals["rental_asset_sale_proceeds"]),
        )
        state.current_day_ledger["actions"]["assets_sold"] += len(rental_quotes)
        self._add(
            state.current_day_ledger["actions"],
            "retail_liquidation_proceeds",
            float(totals["retail_liquidation_proceeds"]),
        )
        self._add(
            state.current_day_ledger["actions"],
            "retail_liquidation_book_cost",
            float(totals["retail_liquidation_book_cost"]),
            digits=4,
        )
        self._add(
            state.current_day_ledger["actions"],
            "retail_liquidation_loss",
            float(totals["retail_liquidation_loss"]),
            digits=4,
        )
        state.current_day_ledger["actions"]["retail_units_liquidated"] += int(
            totals["retail_units_liquidated"]
        )
        if self.config.liquidation_depth_enabled:
            self._add(
                state.current_day_ledger["actions"],
                "liquidation_nominal_proceeds",
                float(totals["nominal_proceeds"]),
            )
            self._add(
                state.current_day_ledger["actions"],
                "liquidation_depth_discount",
                float(totals["depth_discount"]),
            )
            for pool, market in market_depth.items():
                nominal = float(market["batch_nominal_proceeds"])
                if nominal > 0:
                    state.liquidation_market_history.append(
                        {
                            "day": state.day,
                            "pool": pool,
                            "nominal_proceeds": nominal,
                            "actual_proceeds": float(
                                market["batch_actual_proceeds"]
                            ),
                        }
                    )
        actions = state.current_day_ledger.setdefault(
            "agent_actions", _empty_agent_actions()
        )
        actions.setdefault("asset_sales", []).extend(
            {
                "asset_id": asset.asset_id,
                "offer_id": asset.offer_id,
                "product_type": asset.service_line,
                "inspection_grade": quote["inspection_grade"],
                "proceeds": quote["offer_price"],
            }
            for asset, quote in rental_quotes
        )
        actions.setdefault("retail_liquidations", []).extend(
            {
                **quote,
                "product_type": self.offers[str(quote["offer_id"])].service_line,
            }
            for _, quote, _, _ in retail_quotes
        )
        return ToolResult(
            True,
            {
                "rental_assets": [quote for _, quote in rental_quotes],
                **({"warehouse_capacity": self._warehouse_summary(state)} if self._v31_warehouse_capacity() else {}),
                "retail_items": [quote for _, quote, _, _ in retail_quotes],
                "totals": {
                    key: round(value, 2) if isinstance(value, float) else value
                    for key, value in totals.items()
                },
                **(
                    {"market_depth": market_depth}
                    if self.config.liquidation_depth_enabled
                    else {}
                ),
                **(
                    {"durable_capital": self._durable_acquisition_budget_summary(state)}
                    if self._v40_revolving_durable_capital() and rental_quotes
                    else {}
                ),
                "cash": state.cash,
            },
        )

    def close_business_day(
        self, state: WorldState, *, handoff_note: str | None = None
    ) -> ToolResult:
        self._assert_assortment_state(state)
        if state.done:
            return ToolResult(
                False,
                {"error": "the outdoor season has already closed"},
                terminal_day=True,
            )
        state.current_day_ledger["offer_snapshots"] = self._offer_snapshots(state)
        self._auto_confirm_factorized_plans(state)
        self._expire_reservation_requests(state)
        if self._v30_pickup_priority():
            self._finalize_reservation_risks(state)
            if not state.done:
                self._simulate_day(state)
        else:
            # Frozen V29-and-earlier behavior, including replay/resume.
            self._simulate_day(state)
            self._finalize_reservation_risks(state)
        invalid_contracts = [
            contract.contract_id
            for contract in state.contracts
            if contract.return_day > state.max_days
        ]
        if invalid_contracts:
            raise AssertionError(
                "cross-horizon rental invariant violated: "
                + ", ".join(invalid_contracts[:5])
            )

        retail_holding = round(
            float(self.config.retail_holding_rate)
            * sum(
                float(row["quantity"]) * float(row["avg_unit_cost"])
                for row in state.retail_inventory.values()
            ),
            2,
        )
        if (
            self.config.lifecycle_economics_enabled
            and self.config.lifecycle_holding_enabled
        ):
            rental_capital_holding = round(
                float(
                    state.current_day_ledger.get(
                        "opening_rental_capital_holding_charge", 0.0
                    )
                ),
                2,
            )
            rental_space_holding = round(
                float(
                    state.current_day_ledger.get(
                        "opening_rental_space_holding_charge", 0.0
                    )
                ),
                2,
            )
            rental_holding = round(
                rental_capital_holding + rental_space_holding, 2
            )
            rental_idle_holding = 0.0
            state.current_day_ledger[
                "rental_capital_holding_cost"
            ] = rental_capital_holding
            state.current_day_ledger[
                "rental_space_holding_cost"
            ] = rental_space_holding
            state.cumulative_rental_capital_holding_cost = round(
                state.cumulative_rental_capital_holding_cost
                + rental_capital_holding,
                2,
            )
            state.cumulative_rental_space_holding_cost = round(
                state.cumulative_rental_space_holding_cost
                + rental_space_holding,
                2,
            )
        else:
            rental_holding = round(
                float(self.config.rental_holding_rate)
                * float(
                    state.current_day_ledger[
                        "opening_rental_holding_cost_basis"
                    ]
                ),
                2,
            )
            rental_idle_holding = round(
                float(self.config.rental_idle_holding_rate)
                * float(
                    state.current_day_ledger[
                        "opening_rental_idle_holding_cost_basis"
                    ]
                ),
                2,
            )
        state.current_day_ledger["retail_holding_cost"] = retail_holding
        state.current_day_ledger["rental_holding_cost"] = rental_holding
        state.current_day_ledger["rental_idle_holding_cost"] = rental_idle_holding
        state.cumulative_retail_holding_cost = round(
            state.cumulative_retail_holding_cost + retail_holding, 2
        )
        state.cumulative_rental_holding_cost = round(
            state.cumulative_rental_holding_cost
            + rental_holding
            + rental_idle_holding,
            2,
        )
        state.cumulative_rental_idle_holding_cost = round(
            state.cumulative_rental_idle_holding_cost + rental_idle_holding,
            2,
        )
        self._charge(
            state,
            retail_holding + rental_holding + rental_idle_holding,
            reason="库存、租赁资本与闲置资产持有费",
        )
        if not state.done and state.day <= self.config.max_days:
            state.current_day_ledger["operating_fee"] = round(self.config.daily_fee, 2)
            state.cumulative_fees = round(state.cumulative_fees + self.config.daily_fee, 2)
            self._charge(state, self.config.daily_fee, reason="每日运营费")

        state.current_day_ledger["handoff_note"] = handoff_note
        report = self._finalize_daily_report(state)
        state.daily_history.append(report)
        cutoff_day = max(1, int(report["day"]) - 1)
        state.recent_counter_interactions = [
            dict(row)
            for historical in state.daily_history[-2:]
            for row in historical.get("counter_interactions", [])
            if int(row.get("day", 0)) >= cutoff_day
        ]
        if state.bankrupt or state.day >= state.max_days:
            state.done = True
        else:
            state.day += 1
            state.current_day_ledger = _empty_ledger(state.day)
            self._process_morning(state)
            self._capture_opening_rental_holding_basis(state)
            self._prepare_reservation_day(state)
            self._seal_opening_flows(state)
        self._assert_assortment_state(state)
        return ToolResult(
            True,
            {
                "closed_day": self._public_daily_report(report),
                "next_business_day": None if state.done else state.day,
                "season_closed": state.done,
            },
            terminal_day=True,
        )

    def _offer_snapshots(self, state: WorldState) -> list[dict[str, Any]]:
        available_assets = Counter(
            asset.offer_id for asset in state.assets if asset.status == "available"
        )
        rows: list[dict[str, Any]] = []
        for offer in sorted(self.dataset.offers, key=lambda row: row.offer_id):
            if offer.asset_kind == "consumable":
                available_units = int(
                    state.retail_inventory.get(offer.offer_id, {}).get("quantity", 0)
                )
                service_open = True
            else:
                available_units = int(available_assets.get(offer.offer_id, 0))
                service_open = bool(
                    state.rental_service_open.get(offer.offer_id, False)
                )
            rows.append(
                {
                    "offer_id": offer.offer_id,
                    "service_key": self.service_key_for_product_type(
                        offer.service_line
                    ),
                    "product_type": offer.service_line,
                    "posted_price": (
                        None
                        if offer.offer_id not in state.prices
                        else round(float(state.prices[offer.offer_id]), 2)
                    ),
                    "priced_for_sale": offer.offer_id in state.prices,
                    "available_units": available_units,
                    "service_open": service_open,
                }
            )
        return rows
