from __future__ import annotations

from .common import (
    Any,
    CleaningJob,
    Counter,
    DAMAGE_RATES,
    RentalAsset,
    SECONDHAND_GRADE_SCHEDULE,
    SECONDHAND_PARTS_FLOOR_RATE,
    WorldState,
)


class LifecycleMixin:
    def _seal_opening_flows(self, state: WorldState) -> None:
        if self._v29_reservation_accountability():
            state.current_day_ledger["flow_timing_version"] = 1
            state.current_day_ledger["opening_complete"] = True
            state.current_day_ledger.setdefault("intraday", {"refunds": 0.0})

    def _process_maintenance_state(
        self, state: WorldState, messages: list[str]
    ) -> None:
        if not self.config.preventive_maintenance_enabled:
            return
        if self.config.lifecycle_economics_enabled:
            for asset in sorted(state.assets, key=lambda row: row.asset_id):
                if asset.status == "maintenance" and asset.ready_day == state.day:
                    asset.status = "available"
                    asset.ready_day = None
                    asset.maintenance_started_day = None
                    asset.maintenance_wear_anchor = self._combined_wear(asset)
                    status = self._lifecycle_maintenance_status(asset)
                    self._record_rental_activity(
                        state,
                        "maintenance_completed",
                        asset,
                        maintenance_wear_anchor=round(
                            asset.maintenance_wear_anchor, 8
                        ),
                        hazard_protection=status["hazard_protection"],
                    )
                    messages.append(
                        f"资产 {asset.asset_id} 已完成预防维护并恢复可租。"
                    )
                if asset.status == "available":
                    status = self._lifecycle_maintenance_status(asset)
                    if bool(status["safety_hold_required"]):
                        asset.status = "maintenance_hold"
                        asset.ready_day = None
                        self._record_rental_activity(
                            state,
                            "maintenance_hold_started",
                            asset,
                            condition_ratio=round(asset.condition_ratio, 6),
                            combined_wear=round(self._combined_wear(asset), 6),
                        )
                        messages.append(
                            f"资产 {asset.asset_id} 的安全成色不足，暂停出租直至维护。"
                        )
            return
        for asset in sorted(state.assets, key=lambda row: row.asset_id):
            if asset.status == "maintenance" and asset.ready_day == state.day:
                asset.status = "available"
                asset.ready_day = None
                asset.maintenance_started_day = None
                asset.next_inspection_due_day = (
                    state.day + int(self.config.maintenance_interval_days)
                )
                self._record_rental_activity(
                    state,
                    "maintenance_completed",
                    asset,
                    next_inspection_due_day=asset.next_inspection_due_day,
                )
                messages.append(
                    f"资产 {asset.asset_id} 已完成预防维护并恢复可租。"
                )
            if (
                asset.status == "available"
                and asset.next_inspection_due_day is not None
                and state.day
                > asset.next_inspection_due_day
                + int(self.config.maintenance_due_window_days)
            ):
                asset.status = "maintenance_hold"
                asset.ready_day = None
                self._record_rental_activity(
                    state,
                    "maintenance_hold_started",
                    asset,
                    inspection_due_day=asset.next_inspection_due_day,
                )
                messages.append(
                    f"资产 {asset.asset_id} 已超过检查宽限期，暂停出租直至维护。"
                )

    def _service_metric(self, state: WorldState, service_key: str) -> dict:
        """Initialize reporting without creating a demand or fulfillment event."""
        return state.current_day_ledger["service_metrics"].setdefault(service_key, {
            "requested_units": 0, "fulfilled_units": 0, "outside_rejections": 0,
            "stockout_rejections": 0, "service_closed_rejections": 0,
            **({"capacity_policy_rejections": 0} if self._v24_reservation_operations() else {}),
            "horizon_rejections": 0, "unpriced_rejections": 0, "revenue": 0.0,
            "deposits_collected": 0.0, "product_fulfillment_breakdown": {},
            "offer_fulfillment_breakdown": {},
        })

    def _record_attempt(
        self,
        state: WorldState,
        service_key: str,
        outcome: str,
        *,
        product_type: str | None = None,
        offer_id: str | None = None,
        revenue: float = 0.0,
        deposit: float = 0.0,
        rental_days: int = 0,
        retail_cogs: float = 0.0,
        quantity: int = 1,
        outcomes: Counter[str] | None = None,
        channel: str = "walk_in",
    ) -> None:
        quantity = max(1, int(quantity))
        if outcomes is not None:
            outcomes[outcome] += quantity
        demand = state.current_day_ledger["demand"]
        demand["requested_units"] += quantity
        state.cumulative_requested_units += quantity
        row = self._service_metric(state, service_key)
        row["requested_units"] += quantity
        if outcome == "fulfilled":
            from .. import operating_history
            if operating_history.enabled(self.dataset.metadata) and offer_id is not None:
                operating_history.record_delivery(state, offer_id, channel, quantity=quantity,
                    revenue=revenue, rental_days=rental_days, retail_cogs=retail_cogs,
                    rental=self.offers[offer_id].asset_kind == "durable")
            demand["fulfilled_units"] += quantity
            row["fulfilled_units"] += quantity
            row["revenue"] = round(row["revenue"] + revenue, 2)
            row["deposits_collected"] = round(row["deposits_collected"] + deposit, 2)
            if product_type is not None:
                breakdown = row.setdefault("product_fulfillment_breakdown", {})
                product = breakdown.setdefault(
                    product_type,
                    {
                        "fulfilled_units": 0,
                        "revenue": 0.0,
                        "rental_days": 0,
                        "deposits_collected": 0.0,
                        "retail_cogs": 0.0,
                    },
                )
                product["fulfilled_units"] += quantity
                product["revenue"] = round(product["revenue"] + revenue, 2)
                product["rental_days"] += int(rental_days)
                product["deposits_collected"] = round(
                    product["deposits_collected"] + deposit, 2
                )
                product["retail_cogs"] = round(
                    product["retail_cogs"] + retail_cogs, 2
                )
            if offer_id is not None and product_type is not None:
                offer_breakdown = row.setdefault("offer_fulfillment_breakdown", {})
                offer = offer_breakdown.setdefault(
                    offer_id,
                    {
                        "product_type": product_type,
                        "fulfilled_units": 0,
                        "revenue": 0.0,
                        "rental_days": 0,
                        "deposits_collected": 0.0,
                        "retail_cogs": 0.0,
                    },
                )
                offer["fulfilled_units"] += quantity
                offer["revenue"] = round(offer["revenue"] + revenue, 2)
                offer["rental_days"] += int(rental_days)
                offer["deposits_collected"] = round(
                    offer["deposits_collected"] + deposit, 2
                )
                offer["retail_cogs"] = round(offer["retail_cogs"] + retail_cogs, 2)
            state.cumulative_fulfilled_units += quantity
        else:
            key = f"{outcome}_rejections"
            demand[key] = int(demand.get(key, 0)) + quantity
            demand["unfulfilled_units"] = int(demand.get("unfulfilled_units", 0)) + quantity
            row[key] = int(row.get(key, 0)) + quantity
            if outcome == "outside":
                state.cumulative_outside_rejections += quantity
            elif outcome == "stockout":
                state.cumulative_stockout_rejections += quantity
            elif outcome == "service_closed":
                state.cumulative_service_closed_rejections += quantity
                state.cumulative_closed_rejections += quantity
            elif outcome == "capacity_policy":
                state.cumulative_capacity_policy_rejections += quantity
            elif outcome == "horizon":
                state.cumulative_horizon_rejections += quantity
                state.cumulative_closed_rejections += quantity
            elif outcome == "unpriced":
                state.cumulative_unpriced_rejections += quantity
            else:
                raise ValueError(f"unknown request outcome: {outcome}")

    def _process_morning(self, state: WorldState) -> None:
        self._assert_warehouse_snapshot(state)
        self._assert_durable_acquisition_budget(state)
        if self._v29_reservation_accountability():
            state.current_day_ledger["opening_complete"] = False
        messages: list[str] = []
        for order in state.orders:
            if order.status != "in_transit" or order.delivery_day != state.day:
                continue
            if self._v31_warehouse_capacity():
                self._receive_purchase_order(state, order, [self._ordered_units(item) for item in order.items])
                messages.append(f"订单 {order.order_id} 已送达。")
                continue
            for item in order.items:
                offer = self.offers[item["offer_id"]]
                if offer.offer_id not in state.ever_carried_offer_ids:
                    state.ever_carried_offer_ids.append(offer.offer_id)
                    state.ever_carried_offer_ids.sort()
                units = int(item["packs"] * item["package_units"])
                if offer.asset_kind == "durable":
                    for _ in range(units):
                        asset = RentalAsset(
                            asset_id=f"asset-{state.next_asset_id:06d}",
                            offer_id=offer.offer_id,
                            service_line=offer.service_line,
                            unit_cost=offer.unit_cost,
                            used_life_days=0,
                            service_life_limit_days=int(offer.service_life_limit_days),
                            damage_risk_band=str(offer.damage_risk_band),
                            used_rental_cycles=0,
                            service_life_limit_cycles=(
                                None
                                if offer.service_life_limit_cycles is None
                                else int(offer.service_life_limit_cycles)
                            ),
                            acquired_day=state.day,
                            next_inspection_due_day=(
                                state.day
                                + int(self.config.maintenance_interval_days)
                                if self.config.preventive_maintenance_enabled
                                and not self.config.lifecycle_economics_enabled
                                else None
                            ),
                        )
                        if self.config.lifecycle_economics_enabled:
                            projection = self._lifecycle_projection(offer.offer_id)
                            asset.service_life_limit_days = int(
                                projection["service_life_limit_days"]
                            )
                            asset.service_life_limit_cycles = int(
                                projection["service_life_limit_cycles"]
                            )
                        state.assets.append(asset)
                        self._record_rental_activity(
                            state,
                            "asset_acquired",
                            asset,
                            acquisition_cost=asset.unit_cost,
                            purchase_order_id=order.order_id,
                        )
                        state.next_asset_id += 1
                else:
                    inventory = state.retail_inventory.setdefault(
                        offer.offer_id, {"quantity": 0, "avg_unit_cost": offer.unit_cost}
                    )
                    old_qty = int(inventory["quantity"])
                    inventory["avg_unit_cost"] = round(
                        (old_qty * inventory["avg_unit_cost"] + units * offer.unit_cost)
                        / (old_qty + units),
                        4,
                    )
                    inventory["quantity"] = old_qty + units
            order.status = "delivered"
            messages.append(f"订单 {order.order_id} 已送达。")

        self._assert_durable_acquisition_budget(state)

        # Cleaning work returned on an earlier day is processed before today's
        # returns join the queue, preserving a minimum one-business-day outage.
        self._process_cleaning_capacity(state, messages)
        if state.done:
            state.inbox.extend(messages)
            return

        for contract in sorted(state.contracts, key=lambda row: row.contract_id):
            due_today=(self._v47_return_due_today(state,contract) if self._v47_autonomous_rentals()
                       else contract.return_day==state.day)
            if contract.status != "active" or not due_today:
                continue
            asset = next(row for row in state.assets if row.asset_id == contract.asset_id)
            state.deposit_liability = round(max(0.0, state.deposit_liability - contract.deposit), 2)
            deduction = 0.0
            cleaning_cost = 0.0
            damage_details: dict[str, float] = {}
            if self._service_life_exhausted(asset):
                refund = contract.deposit
                asset.status = "retired"
                asset.ready_day = None
                return_event_type = "retired_service_life"
                self._register_durable_capital_exit(
                    state, asset, cause="service_life_retirement"
                )
                self._dispose_warehouse_asset(state, asset, reason="service_life_exhausted_on_return")
                self._mark_supplier_trial_return_pending(
                    state, asset, cause="service_life_exhausted_on_return"
                )
                outcome = "达到正常使用寿命并退役，押金全退"
            elif contract.damaged_on_return:
                if self.config.lifecycle_economics_enabled:
                    damage_details = self._lifecycle_return_damage_terms(
                        asset, contract
                    )
                    deduction = float(damage_details["deposit_deduction"])
                    asset.condition_ratio = float(
                        damage_details["condition_ratio_after_damage"]
                    )
                    asset.pending_repair_cost = float(
                        damage_details["repair_cash_cost"]
                    )
                    asset.repair_target_condition_ratio = float(
                        damage_details["condition_ratio_after_repair"]
                    )
                else:
                    deduction = min(contract.deposit, self.damage_deduction(asset))
                refund = round(contract.deposit - deduction, 2)
                asset.status = "damaged"
                asset.ready_day = None
                state.cumulative_damage_deductions = round(
                    state.cumulative_damage_deductions + deduction, 2
                )
                self._add(state.current_day_ledger["morning"], "damage_deductions", deduction)
                return_event_type = "returned_damaged"
                outcome = f"意外损坏，押金扣除 {deduction:.2f}；实际维修成本可能更高"
            else:
                refund = contract.deposit
                asset.status = "cleaning"
                asset.ready_day = None
                return_event_type = "returned_normal"
                outcome = "正常归还，押金全退并进入清洁队列"
            if not self._refund(state, refund, contract.contract_id):
                return
            contract.status = "returned"
            contract.actual_return_day = state.day
            self._update_reservation_after_contract_return(state, contract)
            if asset.status == "cleaning":
                weather_cleaning_multiplier = (
                    self._return_weather_cleaning_multiplier(
                        contract.start_day, state.day
                    )
                )
                if self.config.lifecycle_economics_enabled:
                    cleaning = self._lifecycle_cleaning_terms(
                        asset,
                        duration_days=contract.duration_days,
                        start_day=contract.start_day,
                        return_day=state.day,
                    )
                    cleaning_cost = float(cleaning["cash_cost"])
                    cleaning_work_points = float(cleaning["work_points"])
                    weather_cleaning_multiplier = float(
                        cleaning["weather_multiplier"]
                    )
                else:
                    cleaning_cost = round(
                        self.cleaning_cost(asset, contract.duration_days)
                        * weather_cleaning_multiplier,
                        2,
                    )
                    cleaning_work_points = round(
                        self.cleaning_work_points(
                            asset.service_line,
                            contract.duration_days,
                            offer_id=asset.offer_id,
                        )
                        * weather_cleaning_multiplier,
                        2,
                    )
                state.cleaning_queue.append(
                    CleaningJob(
                        asset_id=asset.asset_id,
                        offer_id=asset.offer_id,
                        product_type=asset.service_line,
                        enqueued_day=state.day,
                        duration_days=contract.duration_days,
                        remaining_work_points=cleaning_work_points,
                        cleaning_cost=cleaning_cost,
                    )
                )
            self._record_rental_activity(
                state,
                return_event_type,
                asset,
                contract_id=contract.contract_id,
                duration_days=contract.duration_days,
                rent_paid=contract.rent_paid,
                deposit=contract.deposit,
                refund=refund,
                damage_deduction=round(deduction, 2),
                scheduled_cleaning_cost=round(cleaning_cost, 2),
                cleaning_work_points=(
                    cleaning_work_points
                    if asset.status == "cleaning"
                    else 0.0
                ),
                weather_cleaning_multiplier=(
                    round(weather_cleaning_multiplier, 4)
                    if asset.status == "cleaning"
                    else 1.0
                ),
                ready_day=asset.ready_day,
                **(
                    {
                        "condition_loss": damage_details["condition_loss"],
                        "condition_ratio_after_damage": damage_details[
                            "condition_ratio_after_damage"
                        ],
                        "repair_cash_cost": damage_details["repair_cash_cost"],
                        "uncovered_repair_cost": damage_details[
                            "uncovered_repair_cost"
                        ],
                        "repair_target_condition_ratio": damage_details[
                            "condition_ratio_after_repair"
                        ],
                    }
                    if damage_details
                    else {}
                ),
            )
            messages.append(f"租约 {contract.contract_id}: {outcome}。")

        queued_assets = {job.asset_id for job in state.cleaning_queue}
        for asset in state.assets:
            if (
                asset.status in {"cleaning", "repairing"}
                and asset.ready_day == state.day
                and asset.asset_id not in queued_assets
            ):
                completed_status = asset.status
                if (
                    completed_status == "repairing"
                    and self.config.lifecycle_economics_enabled
                ):
                    if asset.repair_target_condition_ratio is None:
                        raise ValueError(
                            f"repairing lifecycle asset has no condition target: {asset.asset_id}"
                        )
                    asset.condition_ratio = float(
                        asset.repair_target_condition_ratio
                    )
                    asset.pending_repair_cost = None
                    asset.repair_target_condition_ratio = None
                asset.status = "available"
                asset.ready_day = None
                self._record_rental_activity(
                    state,
                    (
                        "cleaning_completed"
                        if completed_status == "cleaning"
                        else "repair_completed"
                    ),
                    asset,
                )
                messages.append(f"资产 {asset.asset_id} 已恢复可租。")
        self._process_supplier_trial_morning(state, messages)
        self._process_maintenance_state(state, messages)
        state.inbox.extend(messages)
        self._process_reservation_pickups(state)

    def _refund(self, state: WorldState, amount: float, contract_id: str) -> bool:
        amount = round(amount, 2)
        # All deposit/prepayment/recovery refund paths pass here. A legacy
        # ledger has no timing marker and keeps its historical interpretation.
        phase = "intraday" if state.current_day_ledger.get("opening_complete", False) else "morning"
        self._add(state.current_day_ledger.setdefault(phase, {}), "refunds", amount)
        state.cumulative_refunds = round(state.cumulative_refunds + amount, 2)
        if amount > state.cash + 1e-9 and self.config.emergency_credit_limit > 0:
            self._draw_customer_protection_credit(state, amount)
        if amount > state.cash + 1e-9:
            shortfall = round(amount - state.cash, 2)
            state.bankrupt = True
            state.done = True
            state.bankruptcy_shortfall = round(state.bankruptcy_shortfall + shortfall, 2)
            state.cash = 0.0
            state.inbox.append(
                f"无法为租约 {contract_id} 退款，资金缺口 {shortfall:.2f}，门店破产。"
            )
            return False
        state.cash = round(state.cash - amount, 2)
        return True

    def _charge(self, state: WorldState, amount: float, *, reason: str) -> bool:
        amount = round(amount, 6)
        if amount <= 0:
            return True
        if amount > state.cash + 1e-9:
            state.bankruptcy_shortfall = round(state.bankruptcy_shortfall + amount - state.cash, 2)
            state.cash = 0.0
            state.bankrupt = True
            state.done = True
            state.inbox.append(f"无法支付{reason}，门店破产。")
            return False
        state.cash = round(state.cash - amount, 6)
        return True

    def charge_model_usage(self, state: WorldState, input_tokens: int, output_tokens: int) -> None:
        cost = (
            input_tokens * self.config.input_token_cost_per_million
            + output_tokens * self.config.output_token_cost_per_million
        ) / 1_000_000
        if self.config.charge_model_usage_to_cash:
            self._charge(state, cost, reason="管理工作台服务费")
        state.cumulative_model_cost = round(state.cumulative_model_cost + cost, 6)
        self._add(state.current_day_ledger["actions"], "model_cost", cost, digits=6)

    @staticmethod
    def _build_demand_cube(interactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        dimensions = (
            "day",
            "day_type",
            "holiday_name",
            "weather",
            "visible_party_context",
            "request_text",
            "rental_or_retail",
            "requested_size",
            "service_key",
            "choice_group",
            "customer_segment",
            "size_class",
            "preferred_group_state",
            "disposition",
        )
        values: dict[tuple[Any, ...], dict[str, Any]] = {}
        for interaction in interactions:
            request = interaction["request"]
            response = interaction["response"]
            preferred = interaction["preferred_availability"]
            source = {
                "day": int(interaction["day"]),
                "day_type": interaction["day_type"],
                "holiday_name": interaction.get("holiday_name"),
                "weather": interaction["weather"],
                "visible_party_context": interaction["visible_party_context"],
                "request_text": interaction.get("public_request", {}).get(
                    "request_text", ""
                ),
                "rental_or_retail": interaction.get("public_request", {}).get(
                    "rental_or_retail", ""
                ),
                "requested_size": interaction.get("public_request", {})
                .get("requirements", {})
                .get("requested_size"),
                "service_key": request["service_key"],
                "choice_group": request["asked_group"],
                "customer_segment": request.get("customer_segment", "legacy"),
                "size_class": request["size_class"],
                "preferred_group_state": preferred["state"],
                "disposition": response["disposition"],
            }
            key = tuple(source[name] for name in dimensions)
            row = values.setdefault(
                key,
                {
                    **source,
                    "requested_units": 0,
                    "fulfilled_units": 0,
                    "unfulfilled_units": 0,
                    "interactions": 0,
                    "revenue": 0.0,
                    "retail_cogs": 0.0,
                    "rental_days": 0,
                },
            )
            row["requested_units"] += int(request["requested_units"])
            row["fulfilled_units"] += int(response["accepted_units"])
            row["unfulfilled_units"] += int(response["unfulfilled_units"])
            row["interactions"] += 1
            row["revenue"] = round(
                row["revenue"]
                + sum(float(line.get("revenue", 0.0)) for line in interaction["quote_lines"]),
                2,
            )
            row["retail_cogs"] = round(
                row["retail_cogs"]
                + sum(float(line.get("retail_cogs", 0.0)) for line in interaction["quote_lines"]),
                2,
            )
            row["rental_days"] += sum(
                int(line.get("rental_days", 0)) for line in interaction["quote_lines"]
            )
        return [values[key] for key in sorted(values, key=lambda row: tuple(map(str, row)))]

    def _finalize_daily_report(self, state: WorldState) -> dict[str, Any]:
        ledger = state.current_day_ledger
        worth = self.business_value(state)
        statuses = Counter(asset.status for asset in state.assets)
        financial = {
            "rental_revenue": round(ledger["rental_revenue"], 2),
            "retail_revenue": round(ledger["retail_revenue"], 2),
            "deposits_collected": round(ledger["deposits_collected"], 2),
            **(
                {
                    "reservation_prepayments_collected": round(
                        ledger.get("reservation_prepayments_collected", 0.0), 2
                    ),
                    "reservation_prepayment_refunds": round(
                        ledger.get("reservation_prepayment_refunds", 0.0), 2
                    ),
                    "reservation_prepayment_recognized": round(
                        ledger.get("reservation_prepayment_recognized", 0.0), 2
                    ),
                    "reservation_compensation": round(
                        ledger.get("reservation_compensation", 0.0), 2
                    ),
                    "unearned_revenue_liability_end": round(
                        state.unearned_revenue_liability, 2
                    ),
                }
                if self.config.reservations_enabled
                else {}
            ),
            "refunds": round(
                ledger["morning"]["refunds"] + ledger.get("intraday", {}).get("refunds", 0.0), 2
            ),
            **({
                "refunds_by_phase": {
                    "morning": round(ledger["morning"]["refunds"], 2),
                    "intraday": round(ledger.get("intraday", {}).get("refunds", 0.0), 2),
                },
            } if ledger.get("flow_timing_version") == 1 else {}),
            "damage_deductions": round(ledger["morning"]["damage_deductions"], 2),
            "purchase_cost": round(ledger["actions"]["purchase_cost"], 2),
            **(
                {
                    "supplier_trial_handling_fee": round(
                        ledger["actions"].get(
                            "supplier_trial_handling_fee", 0.0
                        ),
                        2,
                    ),
                    "supplier_trial_usage_fee": round(
                        ledger["actions"].get(
                            "supplier_trial_usage_fee", 0.0
                        ),
                        2,
                    ),
                }
                if self._v37_supplier_trial()
                else {}
            ),
            "cleaning_cost": round(ledger["morning"]["cleaning_cost"], 2),
            "repair_cost": round(
                ledger["actions"]["repair_cost"]
                + ledger["morning"].get("repair_cost", 0.0),
                2,
            ),
            **(
                {
                    "maintenance_cost": round(
                        ledger["actions"].get("maintenance_cost", 0.0), 2
                    )
                }
                if self.config.preventive_maintenance_enabled
                else {}
            ),
            "retail_holding_cost": round(ledger["retail_holding_cost"], 2),
            "rental_holding_cost": round(ledger["rental_holding_cost"], 2),
            **(
                {
                    "rental_capital_holding_cost": round(
                        ledger.get("rental_capital_holding_cost", 0.0), 2
                    ),
                    "rental_space_holding_cost": round(
                        ledger.get("rental_space_holding_cost", 0.0), 2
                    ),
                    "opening_rental_space_burden_units": round(
                        ledger.get("opening_rental_space_burden_units", 0.0), 2
                    ),
                }
                if self.config.lifecycle_economics_enabled
                and self.config.lifecycle_holding_enabled
                else {}
            ),
            "opening_rental_holding_cost_basis": round(
                ledger["opening_rental_holding_cost_basis"], 2
            ),
            **(
                {
                    "rental_idle_holding_cost": round(
                        ledger.get("rental_idle_holding_cost", 0.0), 2
                    ),
                    "opening_rental_idle_holding_cost_basis": round(
                        ledger.get(
                            "opening_rental_idle_holding_cost_basis", 0.0
                        ),
                        2,
                    ),
                }
                if self.config.rental_idle_holding_rate > 0
                else {}
            ),
            "operating_fee": round(ledger["operating_fee"], 2),
            "model_cost": round(ledger["actions"]["model_cost"], 6),
            "asset_sale_proceeds": round(ledger["actions"]["asset_sale_proceeds"], 2),
            "assets_sold": int(ledger["actions"]["assets_sold"]),
            "retail_liquidation_proceeds": round(
                ledger["actions"].get("retail_liquidation_proceeds", 0.0), 2
            ),
            "retail_liquidation_book_cost": round(
                ledger["actions"].get("retail_liquidation_book_cost", 0.0), 2
            ),
            "retail_liquidation_loss": round(
                ledger["actions"].get("retail_liquidation_loss", 0.0), 2
            ),
            "retail_units_liquidated": int(
                ledger["actions"].get("retail_units_liquidated", 0)
            ),
            **(
                {
                    "liquidation_nominal_proceeds": round(
                        ledger["actions"].get(
                            "liquidation_nominal_proceeds", 0.0
                        ),
                        2,
                    ),
                    "liquidation_depth_discount": round(
                        ledger["actions"].get(
                            "liquidation_depth_discount", 0.0
                        ),
                        2,
                    ),
                }
                if self.config.liquidation_depth_enabled
                else {}
            ),
        }
        return {
            "day": state.day,
            "calendar": self._calendar(state.day),
            "weather": self._weather(state.day),
            "arrivals": dict(ledger["arrivals"]),
            "demand": dict(ledger["demand"]),
            "service_metrics": {
                key: dict(value) for key, value in sorted(ledger["service_metrics"].items())
            },
            **({"delivery_metrics": {key: dict(value) for key, value in ledger.get("delivery_metrics", {}).items()}}
               if self.dataset.metadata.get("operating_history_policy") else {}),
            "counter_interactions": [
                dict(row) for row in ledger.get("counter_interactions", [])
            ],
            "internal_channel_outcomes": [
                dict(row) for row in ledger.get("internal_channel_outcomes", [])
            ],
            "internal_basket_intents": [
                dict(row) for row in ledger.get("internal_basket_intents", [])
            ],
            "demand_cube": self._build_demand_cube(
                list(ledger.get("counter_interactions", []))
            ),
            "offer_snapshots": [dict(row) for row in ledger.get("offer_snapshots", [])],
            "financial": financial,
            "agent_actions": self._public_agent_actions(
                ledger.get("agent_actions")
            ),
            "handoff_note": ledger.get("handoff_note"),
            "end": {
                **worth,
                **({"warehouse_capacity": self._warehouse_summary(state)} if self._v31_warehouse_capacity() else {}),
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
                **(
                    {"supplier_trial": self._supplier_trial_summary(state)}
                    if self._v37_supplier_trial()
                    else {}
                ),
                "retail_units": int(
                    sum(row["quantity"] for row in state.retail_inventory.values())
                ),
                "asset_status_counts": dict(sorted(statuses.items())),
                "active_rentals": sum(c.status == "active" for c in state.contracts),
                "cleaning_queue_assets": len(state.cleaning_queue),
                "cleaning_queue_work_points": round(
                    sum(job.remaining_work_points for job in state.cleaning_queue), 2
                ),
                "rental_utilization": round(
                    statuses.get("rented", 0)
                    / max(
                        1,
                        sum(
                            (
                                self._warehouse_asset_held(asset)
                                if self._v31_warehouse_capacity()
                                else asset.status not in {"retired", "sold"}
                            )
                            for asset in state.assets
                        ),
                    ),
                    4,
                ),
            },
        }

    def _in_transit_values(self, state: WorldState) -> tuple[float, float]:
        """Return book and expected delivered resale values for prepaid orders."""
        book_value = 0.0
        going_concern_value = 0.0
        new_durable_resale_rate = SECONDHAND_GRADE_SCHEDULE[0][2]
        for order in state.orders:
            if order.status != "in_transit":
                continue
            if order.procurement_mode == "supplier_trial":
                continue
            if not self._v31_warehouse_capacity():
                book_value += order.total_cost
            for item in order.items:
                offer = self.offers[item["offer_id"]]
                units = int(item["packs"]) * int(item["package_units"])
                line_cost = float(item["line_cost"])
                if self._v31_warehouse_capacity():
                    remaining = units - item["received_units"]
                    line_cost *= remaining / units
                    units = remaining
                    book_value += line_cost
                if offer.asset_kind == "durable":
                    going_concern_value += (
                        line_cost * new_durable_resale_rate
                    )
                else:
                    going_concern_value += self.retail_liquidation_proceeds(
                        offer.unit_cost, units
                    )
        return book_value, going_concern_value

    def _rental_going_concern_values(
        self, state: WorldState
    ) -> dict[str, float]:
        """Value owned equipment assuming active rentals complete normally in expectation."""
        active_by_asset = {
            contract.asset_id: contract
            for contract in state.contracts
            if contract.status == "active"
        }
        fleet_value = 0.0
        rented_recoverable = 0.0
        expected_cleaning = 0.0
        expected_damage_deductions = 0.0
        rented_value = 0.0
        pending_cleaning = {
            job.asset_id: float(job.cleaning_cost) for job in state.cleaning_queue
        }
        for asset in state.assets:
            if asset.ownership_type != "owned":
                continue
            if asset.status in {"retired", "sold"}:
                continue
            if asset.status != "rented":
                quote = self.secondhand_quote(asset, terminal=True)
                if quote is not None:
                    fleet_value += float(quote["offer_price"]) - pending_cleaning.get(
                        asset.asset_id, 0.0
                    )
                continue

            contract = active_by_asset.get(asset.asset_id)
            if contract is None:
                # Preserve the owned asset in diagnostics even if a legacy/corrupt
                # state is missing its active-contract record.
                _, healthy_quote = self._healthy_secondhand_grade_and_quote(asset)
                fleet_value += healthy_quote
                rented_recoverable += healthy_quote
                rented_value += healthy_quote
                continue
            if self._service_life_exhausted(asset):
                # Normal service-life retirement takes precedence at return.
                continue

            if self.config.lifecycle_economics_enabled:
                terms = self._lifecycle_rented_going_concern_terms(
                    asset, contract
                )
                fleet_value += terms["net_value"]
                rented_recoverable += terms["recoverable_value"]
                expected_cleaning += terms["expected_cleaning_cost"]
                expected_damage_deductions += terms[
                    "expected_damage_deduction"
                ]
                rented_value += terms["net_value"]
                continue

            _, healthy_quote = self._healthy_secondhand_grade_and_quote(asset)
            damaged_quote = round(
                max(
                    SECONDHAND_PARTS_FLOOR_RATE * asset.unit_cost,
                    healthy_quote - self.repair_cost(asset),
                ),
                2,
            )
            daily_damage_rate = DAMAGE_RATES[asset.damage_risk_band]
            damage_probability = 1.0 - (
                1.0 - daily_damage_rate
            ) ** contract.duration_days
            recoverable = (
                (1.0 - damage_probability) * healthy_quote
                + damage_probability * damaged_quote
            )
            cleaning = (
                (1.0 - damage_probability)
                * self.cleaning_cost(asset, contract.duration_days)
            )
            damage_deduction = damage_probability * min(
                contract.deposit, self.damage_deduction(asset)
            )
            net_value = recoverable - cleaning + damage_deduction
            fleet_value += net_value
            rented_recoverable += recoverable
            expected_cleaning += cleaning
            expected_damage_deductions += damage_deduction
            rented_value += net_value
        if self.config.liquidation_depth_enabled:
            fleet_value, _ = self._depth_adjusted_pool_proceeds(
                state, "rental_asset", fleet_value
            )
            rented_value, _ = self._depth_adjusted_pool_proceeds(
                state, "rental_asset", rented_value
            )
        return {
            "rental_going_concern_value": fleet_value,
            "rented_asset_recoverable_value": rented_recoverable,
            "rented_expected_cleaning_cost": expected_cleaning,
            "rented_expected_damage_deductions": expected_damage_deductions,
            "rented_asset_going_concern_value": rented_value,
        }

    def business_value(self, state: WorldState) -> dict[str, Any]:
        retail_book = sum(
            float(row["quantity"]) * float(row["avg_unit_cost"])
            for row in state.retail_inventory.values()
        )
        rental_remaining = sum(
            asset.unit_cost * self._remaining_life_ratio(asset)
            for asset in state.assets
            if asset.ownership_type == "owned"
            and asset.status not in {"retired", "sold"}
        )
        retail_book = round(retail_book, 2)
        retail_salvage = self.retail_inventory_liquidation_value(state)
        pending_cleaning_cost = sum(
            float(job.cleaning_cost) for job in state.cleaning_queue
        )
        postseason_cleaning_work_points = sum(
            float(job.remaining_work_points) for job in state.cleaning_queue
        )
        rental_secondhand_nominal = sum(
            float(quote["offer_price"])
            for asset in state.assets
            if (quote := self.secondhand_quote(asset, terminal=True)) is not None
        )
        rental_secondhand, rental_depth = self._depth_adjusted_pool_proceeds(
            state, "rental_asset", rental_secondhand_nominal
        )
        repair_cost_diagnostic = sum(
            self.repair_cost(asset) for asset in state.assets if asset.status == "damaged"
        )
        rental_secondhand = max(0.0, rental_secondhand - pending_cleaning_cost)
        liquidation_value = (
            state.cash
            + retail_salvage
            + rental_secondhand
            - state.deposit_liability
            - state.unearned_revenue_liability
            - state.emergency_credit_liability
        )
        in_transit_book, in_transit_going_concern = self._in_transit_values(state)
        rental_going_concern = self._rental_going_concern_values(state)
        book_net_assets = (
            state.cash
            - state.deposit_liability
            - state.unearned_revenue_liability
            - state.emergency_credit_liability
            + retail_book
            + rental_remaining
            + in_transit_book
        )
        going_concern_value = (
            state.cash
            - state.deposit_liability
            - state.unearned_revenue_liability
            - state.emergency_credit_liability
            + retail_salvage
            + in_transit_going_concern
            + rental_going_concern["rental_going_concern_value"]
        )
        efficiency_adjusted = (
            liquidation_value
            if self.config.charge_model_usage_to_cash
            else liquidation_value - state.cumulative_model_cost
        )
        purchased_durable_cost = sum(
            float(asset.unit_cost)
            for asset in state.assets
            if asset.ownership_type == "owned" and int(asset.acquired_day) > 0
        )
        new_asset_residual_rate = (
            float(self.lifecycle_scales.new_asset_residual_rate)
            if self.lifecycle_scales is not None
            else float(SECONDHAND_GRADE_SCHEDULE[0][2])
        )
        return {
            "cash": round(state.cash, 2),
            "in_transit_value": 0.0,
            "in_transit_book_value": round(in_transit_book, 2),
            "in_transit_going_concern_value": round(
                in_transit_going_concern, 2
            ),
            "retail_inventory_book_value": round(retail_book, 2),
            "retail_inventory_value": round(retail_salvage, 2),
            "rental_remaining_life_book_value": round(rental_remaining, 2),
            "rental_secondhand_value": round(rental_secondhand, 2),
            **(
                {
                    "rental_secondhand_nominal_value": round(
                        rental_secondhand_nominal, 2
                    ),
                    "rental_liquidation_depth_discount": round(
                        rental_secondhand_nominal
                        - float(rental_depth["batch_actual_proceeds"]),
                        2,
                    ),
                }
                if self.config.liquidation_depth_enabled
                else {}
            ),
            **{
                key: round(value, 2)
                for key, value in rental_going_concern.items()
            },
            "deposit_liability": round(state.deposit_liability, 2),
            **(
                {
                    "unearned_revenue_liability": round(
                        state.unearned_revenue_liability, 2
                    )
                }
                if self.config.reservations_enabled
                else {}
            ),
            **(
                {
                    "emergency_credit_principal": round(
                        state.emergency_credit_principal, 2
                    ),
                    "emergency_credit_liability": round(
                        state.emergency_credit_liability, 2
                    ),
                }
                if self.config.emergency_credit_limit > 0
                else {}
            ),
            "damaged_repair_cost_diagnostic": round(repair_cost_diagnostic, 2),
            "pending_cleaning_cost": round(pending_cleaning_cost, 2),
            "postseason_cleaning_cost": round(pending_cleaning_cost, 2),
            "postseason_cleaning_work_points": round(
                postseason_cleaning_work_points, 2
            ),
            "book_net_assets": round(book_net_assets, 2),
            "contract_adjusted_economic_value": round(going_concern_value, 2),
            "going_concern_business_value": round(going_concern_value, 2),
            "official_liquidation_value": round(liquidation_value, 2),
            "liquidation_business_value": round(liquidation_value, 2),
            "standardized_model_cost": round(state.cumulative_model_cost, 6),
            "efficiency_adjusted_value": round(efficiency_adjusted, 2),
            **(
                {
                    "liquidation_policy": "flat_nominal_per_unit_v1",
                    "purchased_durable_cost_diagnostic": round(
                        purchased_durable_cost, 2
                    ),
                    "purchased_durable_new_nominal_residual_diagnostic": round(
                        purchased_durable_cost * new_asset_residual_rate, 2
                    ),
                    "purchased_durable_new_to_used_conversion_loss_diagnostic": round(
                        purchased_durable_cost
                        * (1.0 - new_asset_residual_rate),
                        2,
                    ),
                }
                if self._v29_reservation_accountability()
                else {}
            ),
            # Backward-compatible official season-close score field.
            "business_value": round(liquidation_value, 2),
        }

    def evaluation_score(self, state: WorldState) -> float:
        if state.bankrupt:
            return -round(state.bankruptcy_shortfall, 2)
        return self.business_value(state)["official_liquidation_value"]

    @staticmethod
    def _add(target: dict[str, Any], key: str, amount: float, *, digits: int = 2) -> None:
        target[key] = round(float(target.get(key, 0.0)) + float(amount), digits)
