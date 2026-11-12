"""Mechanical V31 capacity accounting. Never evaluates reservation feasibility."""
from __future__ import annotations

from collections import Counter
from typing import Any

from ..models import PurchaseOrder, RentalAsset, ToolResult, WorldState
from ..warehouse_capacity import WAREHOUSE_POLICY, positive_integer


class WarehouseMixin:
    def _v31_warehouse_capacity(self) -> bool:
        return bool(self._v30_pickup_priority()
                    and int(self.dataset.metadata.get("tool_interface_version", 0)) >= 27
                    and self.dataset.metadata.get("warehouse_capacity_policy") == WAREHOUSE_POLICY)

    def _validate_warehouse_configuration(self) -> None:
        limit = self.config.warehouse_capacity_limit
        if (self.dataset.metadata.get("environment_protocol") == "outfitter-seasonal-v31-return-reserved-warehouse-capacity"
            and not self._v31_warehouse_capacity()):
            raise ValueError("V31 requires its frozen warehouse policy and pickup-priority runtime")
        if limit is not None and not positive_integer(limit):
            raise ValueError("warehouse_capacity_limit must be a positive integer or null")
        if not self._v31_warehouse_capacity():
            if limit is not None or self.dataset.metadata.get("warehouse_capacity_policy"):
                raise ValueError("warehouse capacity requires the V31 runtime")
            return
        expected = {offer.offer_id for offer in self.dataset.offers if offer.asset_kind == "durable"}
        weights = self.dataset.warehouse_capacity_weights
        if set(weights) != expected or any(not positive_integer(w) for w in weights.values()):
            raise ValueError("V31 requires complete positive integer warehouse weights")
        if not self.dataset.warehouse_capacity_weights_sha256:
            raise ValueError("V31 requires frozen warehouse weight hash")
        if any(not positive_integer(offer.package_units) for offer in self.dataset.offers):
            raise ValueError("V31 requires positive integer package_units")

    def _initialize_warehouse(self, state: WorldState) -> None:
        if not self._v31_warehouse_capacity():
            return
        state.warehouse_capacity = {
            "policy": WAREHOUSE_POLICY, "limit_units": self.config.warehouse_capacity_limit,
            "weights_sha256": self.dataset.warehouse_capacity_weights_sha256,
            "peak_occupied_units": 0, "purchase_rejections": 0,
        }
        summary = self._warehouse_summary(state)
        if summary["remaining_units"] is not None and summary["remaining_units"] < 0:
            raise ValueError(f"initial rental holdings exceed warehouse capacity: {summary}")
        state.warehouse_capacity["initial_occupied_units"] = summary["occupied_units"]
        self._warehouse_record_peak(state)

    def _assert_warehouse_snapshot(self, state: WorldState) -> None:
        if not self._v31_warehouse_capacity():
            if state.warehouse_capacity:
                raise ValueError("cannot load a warehouse episode in a legacy runtime")
            return
        frozen = state.warehouse_capacity
        if (frozen.get("policy") != WAREHOUSE_POLICY
            or frozen.get("limit_units") != self.config.warehouse_capacity_limit
            or frozen.get("weights_sha256") != self.dataset.warehouse_capacity_weights_sha256):
            raise ValueError("warehouse limit/weights/policy cannot change during an episode")

    @staticmethod
    def _warehouse_asset_held(asset: RentalAsset) -> bool:
        return asset.sold_day is None and asset.disposed_day is None

    def _warehouse_weight(self, offer_id: str) -> int:
        # Retail is intentionally out of scope, not assigned a zero rental weight.
        return self.dataset.warehouse_capacity_weights.get(offer_id, 0)

    @staticmethod
    def _ordered_units(item: dict[str, Any]) -> int:
        return int(item["packs"] * item["package_units"])

    def _warehouse_rows(self, state: WorldState) -> list[dict[str, Any]]:
        self._assert_warehouse_snapshot(state)
        held = Counter(asset.offer_id for asset in state.assets if self._warehouse_asset_held(asset))
        if held.keys() - self.dataset.warehouse_capacity_weights.keys():
            raise ValueError("held asset is missing a frozen warehouse weight")
        inbound: Counter = Counter()
        for order in state.orders:
            for item in order.items:
                received = item.get("received_units")
                if type(received) is not int or not 0 <= received <= self._ordered_units(item):
                    raise ValueError("V31 purchase lines require valid cumulative received_units")
                inbound[item["offer_id"]] += self._ordered_units(item) - received
        return [{"offer_id": offer_id, "name": self.offers[offer_id].name,
                 "warehouse_capacity_units_per_asset": weight,
                 "held_asset_count": held[offer_id], "inbound_reserved_asset_count": inbound[offer_id],
                 "held_capacity_units": held[offer_id] * weight,
                 "inbound_reserved_capacity_units": inbound[offer_id] * weight,
                 "occupied_units": (held[offer_id] + inbound[offer_id]) * weight}
                for offer_id, weight in sorted(self.dataset.warehouse_capacity_weights.items())]

    def _warehouse_summary(self, state: WorldState) -> dict[str, Any]:
        rows = self._warehouse_rows(state)
        held = sum(row["held_capacity_units"] for row in rows)
        inbound = sum(row["inbound_reserved_capacity_units"] for row in rows)
        limit = state.warehouse_capacity["limit_units"]
        return {"policy": WAREHOUSE_POLICY, "limit_units": limit,
                "limit_mode": "fixed_season_capacity" if limit is not None else "uncapped_calibration_control",
                "held_capacity_units": held, "inbound_reserved_capacity_units": inbound,
                "occupied_units": held + inbound,
                "held_asset_count": sum(row["held_asset_count"] for row in rows),
                "rented_capacity_units": sum(self._warehouse_weight(a.offer_id) for a in state.assets
                                             if a.status == "rented" and self._warehouse_asset_held(a)),
                "remaining_units": None if limit is None else limit - held - inbound}

    def _warehouse_record_peak(self, state: WorldState) -> None:
        if self._v31_warehouse_capacity():
            state.warehouse_capacity["peak_occupied_units"] = max(
                state.warehouse_capacity["peak_occupied_units"], self._warehouse_summary(state)["occupied_units"])

    def warehouse_diagnostics(self, state: WorldState) -> dict[str, Any]:
        summary = {**self._warehouse_summary(state), **state.warehouse_capacity}
        days = [report["end"]["warehouse_capacity"] for report in state.daily_history
                if "warehouse_capacity" in report.get("end", {})]
        summary.update({"completed_day_samples": len(days),
            "days_at_limit": sum(day["remaining_units"] == 0 for day in days),
            "days_at_or_above_90pct_limit": sum(day["limit_units"] is not None and
                day["occupied_units"] >= .9 * day["limit_units"] for day in days),
            "mean_occupied_units": sum(day["occupied_units"] for day in days) / len(days) if days else None,
            "capacity_weighted_utilization": (sum(day["rented_capacity_units"] for day in days) /
                sum(day["held_capacity_units"] for day in days)) if sum(day["held_capacity_units"] for day in days) else None,
            "sampling_basis": "completed-day closing snapshots; utilization is rented weight / held weight, not a feasibility measure"})
        return summary

    def _warehouse_purchase_check(self, state: WorldState, items: list[dict]) -> ToolResult | None:
        if not self._v31_warehouse_capacity():
            return None
        summary = self._warehouse_summary(state)
        requested = sum(self._ordered_units(item) * self._warehouse_weight(item["offer_id"]) for item in items)
        free = summary["remaining_units"]
        if free is not None and requested > free:
            # Diagnostic telemetry only: never consumes cash, IDs, RNG or quota.
            state.warehouse_capacity["purchase_rejections"] += 1
            return ToolResult(False, {"error": "warehouse_capacity_exceeded",
                "warehouse_capacity": summary, "requested_capacity_units": requested,
                "excess_capacity_units": requested - free,
                "cash_and_orders_unchanged": True})
        for item in items:
            item["received_units"] = 0
        return None

    def view_warehouse_capacity(self, state: WorldState, *, offer_ids: list[str] | None = None,
                                cursor: int = 0, limit: int = 50) -> dict[str, Any]:
        if not self._v31_warehouse_capacity():
            raise ValueError("view_warehouse_capacity requires V31")
        if type(cursor) is not int or cursor < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("cursor must be nonnegative integer; limit must be in [1, 100]")
        selected = set(offer_ids or [])
        if selected - self.dataset.warehouse_capacity_weights.keys():
            raise ValueError("offer_ids must identify rental offers")
        rows = [row for row in self._warehouse_rows(state) if not selected or row["offer_id"] in selected]
        page = rows[cursor:cursor + limit]
        next_cursor = cursor + len(page) if cursor + len(page) < len(rows) else None
        excluded_scope = (
            "Retail and customer reservations do not add occupancy."
            if self.config.retail_enabled
            else "Customer reservations do not add occupancy."
        )
        return {"as_of_day": state.day, **self._warehouse_summary(state),
                "totals_scope": "all_rental_offers_regardless_of_filter",
                "offers": page, "total_count": len(rows), "returned_count": len(page),
                "cursor": cursor, "limit": limit, "has_more": next_cursor is not None, "next_cursor": next_cursor,
                "semantics": "Weights are SPACE CONSUMED per asset, not how many assets fit: larger weight uses more of N. Two assets of weight 6 use 12 units. Includes return space for all owned rental assets and confirmed inbound units. "
                + excluded_scope
                + " Status changes do not release capacity; actual sale/disposal does. No additional warehouse fee."}

    def _dispose_warehouse_asset(self, state: WorldState, asset: RentalAsset, *, reason: str) -> None:
        if not self._v31_warehouse_capacity() or not self._warehouse_asset_held(asset):
            return
        asset.disposed_day = state.day
        self._record_rental_activity(state, "asset_disposed", asset,
            disposal_reason=reason, disposed_day=state.day,
            released_warehouse_capacity_units=self._warehouse_weight(asset.offer_id))

    def _receive_purchase_order(self, state: WorldState, order: PurchaseOrder,
                                received_units: list[int]) -> None:
        """Internal cumulative receipts; validation is batch atomic, repeats idempotent.

        Normal simulation submits the full order at its existing morning time.
        No new random splitting or public delivery action is introduced.
        """
        self._assert_warehouse_snapshot(state)
        if not self._v31_warehouse_capacity() or not any(row is order for row in state.orders):
            raise ValueError("receipt requires an existing V31 order")
        if order.status not in {"in_transit", "delivered"} or len(received_units) != len(order.items):
            raise ValueError("receipt must specify every line of an active order")
        for item, target in zip(order.items, received_units):
            if type(target) is not int or not item["received_units"] <= target <= self._ordered_units(item):
                raise ValueError("cumulative received quantity must be monotonic and not exceed ordered units")
        if order.procurement_mode == "supplier_trial":
            if len(order.items) != 1 or self._ordered_units(order.items[0]) != 1:
                raise ValueError("supplier trial receipt must contain one asset")
            item = order.items[0]
            target = received_units[0]
            if target > item["received_units"]:
                offer = self.offers[item["offer_id"]]
                if offer.offer_id not in state.ever_carried_offer_ids:
                    state.ever_carried_offer_ids.append(offer.offer_id)
                    state.ever_carried_offer_ids.sort()
                self._receive_supplier_trial_asset(state, order)
                item["received_units"] = target
            if item["received_units"] == 1:
                order.status = "delivered"
            self._warehouse_record_peak(state)
            self._assert_durable_acquisition_budget(state)
            self._assert_supplier_trials(state)
            return
        for item_index, (item, target) in enumerate(zip(order.items, received_units)):
            units = target - item["received_units"]
            if not units:
                continue
            offer = self.offers[item["offer_id"]]
            if offer.offer_id not in state.ever_carried_offer_ids:
                state.ever_carried_offer_ids.append(offer.offer_id)
                state.ever_carried_offer_ids.sort()
            if offer.asset_kind == "durable":
                first_unit_index = int(item["received_units"])
                for offset in range(units):
                    asset = RentalAsset(asset_id=f"asset-{state.next_asset_id:06d}",
                        offer_id=offer.offer_id, service_line=offer.service_line, unit_cost=offer.unit_cost,
                        used_life_days=0, service_life_limit_days=int(offer.service_life_limit_days),
                        damage_risk_band=str(offer.damage_risk_band),
                        service_life_limit_cycles=offer.service_life_limit_cycles, acquired_day=state.day,
                        next_inspection_due_day=(state.day + int(self.config.maintenance_interval_days)
                            if self.config.preventive_maintenance_enabled and not self.config.lifecycle_economics_enabled else None),
                        **self._received_durable_capital_basis(
                            order,
                            item,
                            item_index=item_index,
                            unit_index=first_unit_index + offset,
                        ))
                    if self.config.lifecycle_economics_enabled:
                        projection = self._lifecycle_projection(offer.offer_id)
                        asset.service_life_limit_days = int(projection["service_life_limit_days"])
                        asset.service_life_limit_cycles = int(projection["service_life_limit_cycles"])
                    state.assets.append(asset)
                    self._record_rental_activity(state, "asset_acquired", asset,
                        acquisition_cost=asset.unit_cost, purchase_order_id=order.order_id)
                    state.next_asset_id += 1
            else:
                inventory = state.retail_inventory.setdefault(offer.offer_id, {"quantity": 0, "avg_unit_cost": offer.unit_cost})
                old_qty = int(inventory["quantity"])
                inventory["avg_unit_cost"] = round((old_qty * inventory["avg_unit_cost"] + units * offer.unit_cost) / (old_qty + units), 4)
                inventory["quantity"] = old_qty + units
            item["received_units"] = target
        if all(item["received_units"] == self._ordered_units(item) for item in order.items):
            order.status = "delivered"
        self._warehouse_record_peak(state)
        self._assert_durable_acquisition_budget(state)
