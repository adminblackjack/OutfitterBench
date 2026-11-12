"""V37 bounded supplier-trial mechanics shared by both assortment tracks."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from ..models import PurchaseOrder, RentalAsset, SupplierTrialRecord, ToolResult, WorldState
from ..supplier_trial import ENVIRONMENT_PROTOCOL, POLICY_ID


class SupplierTrialMixin:
    def _v37_supplier_trial(self) -> bool:
        return bool(
            int(self.dataset.metadata.get("tool_interface_version", 0)) >= 32
            and self.dataset.metadata.get("supplier_trial_policy") == POLICY_ID
        )

    def _validate_supplier_trial_configuration(self) -> None:
        enabled = bool(self.config.supplier_trial_enabled)
        if enabled and not self._v37_supplier_trial():
            raise ValueError("supplier_trial_enabled requires a V37 trial runtime")
        if (
            self.dataset.metadata.get("supplier_trial_policy") == POLICY_ID
            and (not enabled or not self._v37_supplier_trial())
        ):
            raise ValueError("this runtime requires its frozen enabled supplier-trial policy")
        if enabled:
            frozen = dict(self.dataset.metadata.get("supplier_trial") or {})
            expected = {
                "maximum_accepted_trials": self.config.supplier_trial_max_accepted,
                "maximum_active_trials": self.config.supplier_trial_max_active,
                "maximum_contract_starts": self.config.supplier_trial_max_contract_starts,
                "window_days": self.config.supplier_trial_window_days,
                "maximum_unit_cost": self.config.supplier_trial_max_unit_cost,
                "handling_fee": self.config.supplier_trial_handling_fee,
                "usage_fee_multiplier": self.config.supplier_trial_usage_fee_multiplier,
            }
            for key, value in expected.items():
                if frozen.get(key) != value:
                    raise ValueError(f"supplier trial config differs from frozen policy: {key}")

    def _initialize_supplier_trials(self, state: WorldState) -> None:
        if not self._v37_supplier_trial():
            return
        state.supplier_trials = []
        self._assert_supplier_trials(state)

    def _assert_supplier_trials(self, state: WorldState) -> None:
        if not self._v37_supplier_trial():
            if state.supplier_trials:
                raise ValueError("cannot load supplier trials in an older runtime")
            return
        if len(state.supplier_trials) > int(self.config.supplier_trial_max_accepted):
            raise ValueError("supplier trial allowance exceeded")
        identifiers = [row.trial_id for row in state.supplier_trials]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate supplier trial_id")
        offers = [row.offer_id for row in state.supplier_trials]
        if len(offers) != len(set(offers)):
            raise ValueError("an exact SKU may be trialed only once")
        active = [row for row in state.supplier_trials if row.status != "returned"]
        if len(active) > int(self.config.supplier_trial_max_active):
            raise ValueError("supplier trial active-slot limit exceeded")

    def _supplier_trial_summary(self, state: WorldState) -> dict[str, Any]:
        self._assert_supplier_trials(state)
        maximum = int(self.config.supplier_trial_max_accepted)
        active = [row for row in state.supplier_trials if row.status != "returned"]
        return {
            **self._public_supplier_trial_policy(),
            "maximum_accepted_trials": maximum,
            "accepted_trials": len(state.supplier_trials),
            "remaining_trials": maximum - len(state.supplier_trials),
            "maximum_active_trials": int(self.config.supplier_trial_max_active),
            "active_trials": len(active),
            "tried_offer_ids": sorted(row.offer_id for row in state.supplier_trials),
            "trials": [self._public_supplier_trial(row) for row in active],
        }

    def _public_supplier_trial_policy(self) -> dict[str, Any]:
        return {
            "policy": POLICY_ID,
            "maximum_accepted_trials": int(self.config.supplier_trial_max_accepted),
            "maximum_active_or_in_transit_trials": int(
                self.config.supplier_trial_max_active
            ),
            "maximum_contract_starts_per_trial": int(
                self.config.supplier_trial_max_contract_starts
            ),
            "window_days_from_delivery": int(
                self.config.supplier_trial_window_days
            ),
            "maximum_unit_cost": float(self.config.supplier_trial_max_unit_cost),
            "handling_fee": float(self.config.supplier_trial_handling_fee),
            "usage_fee_multiplier": float(
                self.config.supplier_trial_usage_fee_multiplier
            ),
            "usage_fee_formula": (
                "unit_cost * (duration_days / service_life_limit_days + "
                "1 / service_life_limit_cycles); extensions and late days add "
                "only unit_cost * added_days / service_life_limit_days"
            ),
            "eligibility": (
                "one single-unit durable exact SKU never previously owned, formally "
                "ordered, or trialed; unit_cost must not exceed maximum_unit_cost"
            ),
            "return_boundary": (
                "delivery day d is usable; return_deadline d+window_days is an "
                "exclusive new-contract boundary"
            ),
            "admission_horizon_rule": (
                "the latest permitted supplier delivery plus the full trial window "
                "must fit by the runoff end day"
            ),
            "ownership": "supplier_owned",
            **(
                {"consumes_durable_capital": False}
                if self._v40_revolving_durable_capital()
                else {"consumes_durable_acquisition_budget": False}
            ),
            "occupies_warehouse_capacity": True,
            "sale_or_retirement_allowed": False,
            "supplier_cancellation_generated": False,
            "early_return_restores_trial_allowance": False,
            "terminal_value": 0.0,
        }

    def _supplier_trial_order_terms(
        self, state: WorldState, items: list[dict[str, Any]]
    ) -> dict[str, Any] | ToolResult:
        self._assert_supplier_trials(state)
        if len(state.supplier_trials) >= int(self.config.supplier_trial_max_accepted):
            return ToolResult(False, {
                "error": "supplier_trial_allowance_exhausted",
                "supplier_trial": self._supplier_trial_summary(state),
                "cash_and_orders_unchanged": True,
            })
        latest_delivery_day = state.day + int(
            self.config.supplier_delivery_max_days
        )
        if (
            latest_delivery_day + int(self.config.supplier_trial_window_days)
            > state.max_days
        ):
            return ToolResult(False, {
                "error": "supplier_trial_full_window_exceeds_runoff_horizon",
                "latest_delivery_day": latest_delivery_day,
                "required_return_deadline": (
                    latest_delivery_day
                    + int(self.config.supplier_trial_window_days)
                ),
                "runoff_end_day": state.max_days,
                "cash_and_orders_unchanged": True,
            })
        active = [row for row in state.supplier_trials if row.status != "returned"]
        if len(active) >= int(self.config.supplier_trial_max_active):
            return ToolResult(False, {
                "error": "supplier_trial_active_slot_occupied",
                "active_trial": self._public_supplier_trial(active[0]),
                "cash_and_orders_unchanged": True,
            })
        if len(items) != 1 or int(items[0]["packs"]) != 1 or int(items[0]["package_units"]) != 1:
            return ToolResult(False, {
                "error": "supplier_trial_requires_one_single-unit_durable_pack",
                "cash_and_orders_unchanged": True,
            })
        item = items[0]
        offer_id = str(item["offer_id"])
        offer = self.offers[offer_id]
        if offer.asset_kind != "durable":
            return ToolResult(False, {
                "error": "supplier_trial_requires_a_durable_offer",
                "offer_id": offer_id,
                "cash_and_orders_unchanged": True,
            })
        if float(offer.unit_cost) > float(self.config.supplier_trial_max_unit_cost) + 1e-9:
            return ToolResult(False, {
                "error": "supplier_trial_unit_cost_limit_exceeded",
                "offer_id": offer_id,
                "unit_cost": float(offer.unit_cost),
                "maximum_unit_cost": float(self.config.supplier_trial_max_unit_cost),
                "cash_and_orders_unchanged": True,
            })
        if any(row.offer_id == offer_id for row in state.supplier_trials):
            return ToolResult(False, {
                "error": "supplier_trial_exact_sku_already_tried",
                "offer_id": offer_id,
                "cash_and_orders_unchanged": True,
            })
        ever_owned = any(
            asset.offer_id == offer_id and asset.ownership_type == "owned"
            for asset in state.assets
        ) or any(
            order.procurement_mode == "owned"
            and any(str(row["offer_id"]) == offer_id for row in order.items)
            for order in state.orders
        )
        if ever_owned:
            return ToolResult(False, {
                "error": "supplier_trial_exact_sku_already_owned_or_ordered",
                "offer_id": offer_id,
                "cash_and_orders_unchanged": True,
            })
        return {
            "trial_id": f"trial-{state.next_supplier_trial_id:03d}",
            "offer_id": offer_id,
            "handling_fee": round(float(self.config.supplier_trial_handling_fee), 2),
        }

    def _accept_supplier_trial(
        self, state: WorldState, order: PurchaseOrder, terms: dict[str, Any]
    ) -> SupplierTrialRecord:
        record = SupplierTrialRecord(
            trial_id=str(terms["trial_id"]),
            offer_id=str(terms["offer_id"]),
            order_id=order.order_id,
            accepted_day=state.day,
            handling_fee=float(terms["handling_fee"]),
            remaining_contract_starts=int(self.config.supplier_trial_max_contract_starts),
        )
        state.supplier_trials.append(record)
        state.next_supplier_trial_id += 1
        state.cumulative_supplier_trial_handling_fees = round(
            state.cumulative_supplier_trial_handling_fees + record.handling_fee, 2
        )
        self._assert_supplier_trials(state)
        return record

    def _receive_supplier_trial_asset(
        self, state: WorldState, order: PurchaseOrder
    ) -> RentalAsset:
        record = next(
            row for row in state.supplier_trials if row.trial_id == order.supplier_trial_id
        )
        if record.status != "in_transit" or record.asset_id is not None:
            raise ValueError("supplier trial receipt is not pending")
        item = order.items[0]
        offer = self.offers[str(item["offer_id"])]
        asset = RentalAsset(
            # Supplier-owned assets have their own namespace so accepting an
            # unused trial does not renumber later owned assets and perturb
            # their asset-addressed damage tapes.
            asset_id=f"supplier-{record.trial_id}",
            offer_id=offer.offer_id,
            service_line=offer.service_line,
            unit_cost=offer.unit_cost,
            used_life_days=0,
            service_life_limit_days=int(offer.service_life_limit_days),
            damage_risk_band=str(offer.damage_risk_band),
            service_life_limit_cycles=offer.service_life_limit_cycles,
            acquired_day=state.day,
            ownership_type="supplier_trial",
            supplier_trial_id=record.trial_id,
        )
        if self.config.lifecycle_economics_enabled:
            projection = self._lifecycle_projection(offer.offer_id)
            asset.service_life_limit_days = int(projection["service_life_limit_days"])
            asset.service_life_limit_cycles = int(projection["service_life_limit_cycles"])
        state.assets.append(asset)
        record.asset_id = asset.asset_id
        record.delivered_day = state.day
        record.return_deadline = state.day + int(self.config.supplier_trial_window_days)
        record.status = "active"
        self._record_rental_activity(
            state,
            "supplier_trial_delivered",
            asset,
            supplier_trial_id=record.trial_id,
            purchase_order_id=order.order_id,
            handling_fee=record.handling_fee,
            return_deadline=record.return_deadline,
            remaining_contract_starts=record.remaining_contract_starts,
        )
        self._assert_supplier_trials(state)
        return asset

    def _supplier_trial_record_for_asset(
        self, state: WorldState, asset: RentalAsset
    ) -> SupplierTrialRecord | None:
        if asset.ownership_type != "supplier_trial":
            return None
        return next(
            row for row in state.supplier_trials
            if row.trial_id == asset.supplier_trial_id
        )

    def _supplier_trial_asset_can_start_contract(
        self,
        state: WorldState,
        asset: RentalAsset,
        *,
        return_day: int,
    ) -> bool:
        record = self._supplier_trial_record_for_asset(state, asset)
        if record is None:
            return True
        return bool(
            record.status == "active"
            and not record.return_pending
            and record.remaining_contract_starts > 0
            and record.return_deadline is not None
            and int(return_day) <= int(record.return_deadline)
        )

    def _supplier_trial_usage_fee(
        self, asset: RentalAsset, *, duration_days: int, include_cycle: bool
    ) -> float:
        cycles = int(asset.service_life_limit_cycles or 0)
        if asset.service_life_limit_days <= 0 or cycles <= 0:
            raise ValueError("supplier trial asset requires positive frozen life limits")
        fraction = Decimal(str(duration_days)) / Decimal(str(asset.service_life_limit_days))
        if include_cycle:
            fraction += Decimal(1) / Decimal(str(cycles))
        amount = (
            Decimal(str(asset.unit_cost))
            * fraction
            * Decimal(str(self.config.supplier_trial_usage_fee_multiplier))
        )
        return float(amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    def _charge_supplier_trial_usage(
        self,
        state: WorldState,
        asset: RentalAsset,
        *,
        duration_days: int,
        include_cycle: bool,
        reason: str,
    ) -> float | None:
        record = self._supplier_trial_record_for_asset(state, asset)
        if record is None:
            return 0.0
        fee = self._supplier_trial_usage_fee(
            asset, duration_days=duration_days, include_cycle=include_cycle
        )
        if not self._charge(state, fee, reason=reason):
            return None
        state.cumulative_supplier_trial_usage_fees = round(
            state.cumulative_supplier_trial_usage_fees + fee, 2
        )
        self._add(
            state.current_day_ledger.setdefault("actions", {}),
            "supplier_trial_usage_fee",
            fee,
        )
        self._record_rental_activity(
            state,
            "supplier_trial_usage_fee_charged",
            asset,
            supplier_trial_id=record.trial_id,
            fee=fee,
            duration_days=duration_days,
            cycle_component_included=include_cycle,
            reason=reason,
        )
        return fee

    def _start_supplier_trial_contract(
        self, state: WorldState, asset: RentalAsset, *, return_day: int
    ) -> float | None:
        record = self._supplier_trial_record_for_asset(state, asset)
        if record is None:
            return 0.0
        if not self._supplier_trial_asset_can_start_contract(
            state, asset, return_day=return_day
        ):
            raise ValueError("supplier trial asset is not eligible for this return day")
        record.remaining_contract_starts -= 1
        if record.remaining_contract_starts == 0:
            record.return_pending = True
            record.status = "return_pending"
            self._record_rental_activity(
                state,
                "supplier_trial_return_pending",
                asset,
                supplier_trial_id=record.trial_id,
                cause="contract_start_limit_reached",
            )
        return self._charge_supplier_trial_usage(
            state,
            asset,
            duration_days=int(return_day) - state.day,
            include_cycle=True,
            reason="供应商试用资产使用费",
        )

    def _charge_supplier_trial_contract_extension(
        self,
        state: WorldState,
        asset: RentalAsset,
        contract: Any,
        *,
        duration_days: int,
        reason: str,
    ) -> bool:
        """Charge incremental day use and attribute it to one contract."""

        fee = self._charge_supplier_trial_usage(
            state,
            asset,
            duration_days=duration_days,
            include_cycle=False,
            reason=reason,
        )
        if fee is None:
            return False
        if fee > 0:
            contract.supplier_trial_usage_fee = round(
                float(contract.supplier_trial_usage_fee) + fee, 2
            )
        return True

    def _supplier_trial_extension_allowed(
        self, state: WorldState, asset: RentalAsset, *, proposed_return_day: int
    ) -> bool:
        record = self._supplier_trial_record_for_asset(state, asset)
        return bool(
            record is None
            or (
                record.return_deadline is not None
                and int(proposed_return_day) <= int(record.return_deadline)
            )
        )

    def _mark_supplier_trial_return_pending(
        self, state: WorldState, asset: RentalAsset, *, cause: str
    ) -> None:
        record = self._supplier_trial_record_for_asset(state, asset)
        if record is None or record.status == "returned":
            return
        if not record.return_pending:
            record.return_pending = True
            record.status = "return_pending"
            self._record_rental_activity(
                state,
                "supplier_trial_return_pending",
                asset,
                supplier_trial_id=record.trial_id,
                cause=cause,
            )

    def _physically_return_supplier_trial(
        self, state: WorldState, asset: RentalAsset, *, cause: str
    ) -> None:
        record = self._supplier_trial_record_for_asset(state, asset)
        if record is None or record.status == "returned":
            return
        asset.status = "supplier_returned"
        asset.ready_day = None
        self._dispose_warehouse_asset(state, asset, reason="supplier_trial_return")
        record.status = "returned"
        record.return_pending = True
        record.returned_day = state.day
        self._record_rental_activity(
            state,
            "supplier_trial_returned",
            asset,
            supplier_trial_id=record.trial_id,
            cause=cause,
        )
        self._assert_supplier_trials(state)

    def _begin_supplier_trial_auto_repair(
        self, state: WorldState, asset: RentalAsset
    ) -> None:
        cost = round(float(self.repair_cost(asset)), 2)
        if not self._charge(state, cost, reason="供应商试用资产损坏维修费"):
            return
        asset.status = "repairing"
        asset.ready_day = state.day + 2
        state.cumulative_repair_cost = round(state.cumulative_repair_cost + cost, 2)
        phase = (
            "actions"
            if state.current_day_ledger.get("opening_complete", False)
            else "morning"
        )
        self._add(state.current_day_ledger.setdefault(phase, {}), "repair_cost", cost)
        self._record_rental_activity(
            state,
            "repair_started",
            asset,
            repair_cost=cost,
            ready_day=asset.ready_day,
            automatic_supplier_trial_return_repair=True,
        )

    def _process_supplier_trial_morning(
        self, state: WorldState, messages: list[str]
    ) -> None:
        if not self._v37_supplier_trial() or state.done:
            return
        by_id = {asset.asset_id: asset for asset in state.assets}
        for record in state.supplier_trials:
            if record.status == "returned" or record.asset_id is None:
                continue
            asset = by_id[record.asset_id]
            if record.return_deadline is not None and state.day >= record.return_deadline:
                self._mark_supplier_trial_return_pending(
                    state, asset, cause="trial_window_expired"
                )
            if not record.return_pending:
                continue
            if asset.status in {"available", "maintenance_hold", "retired"}:
                self._physically_return_supplier_trial(
                    state, asset, cause="pending_service_complete"
                )
                messages.append(f"供应商试用资产 {asset.asset_id} 已返还。")
            elif asset.status == "damaged":
                self._begin_supplier_trial_auto_repair(state, asset)
                messages.append(
                    f"供应商试用资产 {asset.asset_id} 已自动开始损坏维修，完成后返还。"
                )
        self._assert_supplier_trials(state)

    def return_supplier_trial_assets(
        self, state: WorldState, asset_ids: list[str]
    ) -> ToolResult:
        identifiers = list(map(str, asset_ids))
        if not identifiers or len(identifiers) != len(set(identifiers)):
            return ToolResult(False, {"error": "asset_ids must be nonempty and unique"})
        by_id = {asset.asset_id: asset for asset in state.assets}
        invalid = [
            asset_id for asset_id in identifiers
            if asset_id not in by_id
            or by_id[asset_id].ownership_type != "supplier_trial"
            or self._supplier_trial_record_for_asset(state, by_id[asset_id]).status
            == "returned"
        ]
        if invalid:
            return ToolResult(False, {
                "error": "only active supplier-trial assets may be returned",
                "invalid_asset_ids": invalid,
                "state_unchanged": True,
            })
        immediate = []
        pending = []
        for asset_id in identifiers:
            asset = by_id[asset_id]
            record = self._supplier_trial_record_for_asset(state, asset)
            self._record_rental_activity(
                state,
                "supplier_trial_return_requested",
                asset,
                supplier_trial_id=record.trial_id,
            )
            self._mark_supplier_trial_return_pending(
                state, asset, cause="manager_requested_early_return"
            )
            if asset.status in {"available", "maintenance_hold", "retired"}:
                self._physically_return_supplier_trial(
                    state, asset, cause="manager_requested_early_return"
                )
                immediate.append(asset_id)
            elif asset.status == "damaged":
                self._begin_supplier_trial_auto_repair(state, asset)
                pending.append(asset_id)
            else:
                pending.append(asset_id)
        return ToolResult(True, {
            "returned_asset_ids": immediate,
            "return_pending_asset_ids": pending,
            "trial_allowance_restored": False,
            "supplier_trial": self._supplier_trial_summary(state),
            "warehouse_capacity": self._warehouse_summary(state),
        })

    @staticmethod
    def _public_supplier_trial(row: SupplierTrialRecord) -> dict[str, Any]:
        return {
            "trial_id": row.trial_id,
            "offer_id": row.offer_id,
            "order_id": row.order_id,
            "asset_id": row.asset_id,
            "accepted_day": row.accepted_day,
            "delivered_day": row.delivered_day,
            "return_deadline": row.return_deadline,
            "remaining_contract_starts": row.remaining_contract_starts,
            "return_pending": row.return_pending,
            "returned_day": row.returned_day,
            "status": row.status,
        }
