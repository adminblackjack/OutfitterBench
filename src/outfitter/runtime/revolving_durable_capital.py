"""V40 fixed gross deployed durable-capital envelope."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from ..models import PurchaseOrder, RentalAsset, ToolResult, WorldState
from ..actor_interface_slimming import (
    ENVIRONMENT_PROTOCOL as V41_ENVIRONMENT_PROTOCOL,
)
from ..revolving_durable_capital import POLICY_ID


_CENT = Decimal("0.01")
_EXIT_CAUSES = {
    "service_life_retirement",
    "manager_retirement",
    "dealer_sale",
}


class RevolvingDurableCapitalMixin:
    def _v40_revolving_durable_capital(self) -> bool:
        return bool(
            int(self.dataset.metadata.get("tool_interface_version", 0)) >= 35
            and self.dataset.metadata.get("durable_acquisition_budget_policy")
            == POLICY_ID
        )

    def _durable_capital_control_enabled(self) -> bool:
        return self._v40_revolving_durable_capital() or super()._durable_capital_control_enabled()

    @staticmethod
    def _capital_cents(value: float | int | Decimal) -> int:
        decimal = Decimal(str(value)).quantize(_CENT, rounding=ROUND_HALF_UP)
        return int(decimal * 100)

    @staticmethod
    def _capital_money(cents: int | None) -> float | None:
        return None if cents is None else float(Decimal(cents) / Decimal(100))

    def _validate_durable_acquisition_budget_configuration(self) -> None:
        if not self._v40_revolving_durable_capital():
            super()._validate_durable_acquisition_budget_configuration()
            return
        if self.dataset.metadata.get("durable_acquisition_budget_policy") != POLICY_ID:
            raise ValueError("V40 requires its frozen revolving durable-capital policy")
        policy = dict(self.dataset.metadata.get("durable_capital_policy") or {})
        if policy.get("format") != POLICY_ID or policy.get("release_delay_days") != 1:
            raise ValueError("V40 durable-capital policy metadata is invalid")
        if not self._v39_contingency_actor_clarity():
            raise ValueError("V40 requires the inherited V39 actor-interface policy")
        if not self._v31_warehouse_capacity():
            raise ValueError("V40 requires cumulative receipt and warehouse accounting")
        if (
            self.dataset.metadata.get("environment_protocol")
            == V41_ENVIRONMENT_PROTOCOL
            and not self._v41_actor_interface_slimming()
        ):
            raise ValueError("V41 requires its frozen actor-interface slimming policy")

    def _initialize_durable_acquisition_budget(self, state: WorldState) -> None:
        if not self._v40_revolving_durable_capital():
            super()._initialize_durable_acquisition_budget(state)
            return
        initial_cents = 0
        for asset in state.assets:
            if asset.ownership_type != "owned":
                asset.durable_capital_basis_cents = None
                continue
            basis = self._capital_cents(asset.unit_cost)
            if basis < 0:
                raise ValueError("initial durable capital basis must be nonnegative")
            asset.durable_capital_basis_cents = basis
            initial_cents += basis
        additional = (
            None
            if self.config.durable_acquisition_budget is None
            else self._capital_cents(self.config.durable_acquisition_budget)
        )
        state.durable_acquisition_budget = {
            "policy": POLICY_ID,
            "initial_owned_durable_basis_cents": initial_cents,
            "additional_durable_capital_headroom_cents": additional,
            "durable_capital_limit_cents": (
                None if additional is None else initial_cents + additional
            ),
        }
        self._assert_durable_acquisition_budget(state)

    def _freeze_durable_capital_allocations(
        self, items: list[dict[str, Any]]
    ) -> None:
        if not self._v40_revolving_durable_capital():
            return
        for item in items:
            offer = self.offers[str(item["offer_id"])]
            if offer.asset_kind != "durable":
                item.pop("durable_capital_unit_basis_cents", None)
                item.pop("durable_capital_basis_cents", None)
                continue
            units = self._ordered_units(item)
            total = self._capital_cents(item["line_cost"])
            quotient, remainder = divmod(total, units)
            allocations = [
                quotient + (1 if index < remainder else 0)
                for index in range(units)
            ]
            item["durable_capital_unit_basis_cents"] = allocations
            item["durable_capital_basis_cents"] = total

    def _validate_order_capital_allocations(self, order: PurchaseOrder) -> None:
        if order.procurement_mode == "supplier_trial":
            return
        for item in order.items:
            offer = self.offers[str(item["offer_id"])]
            allocations = item.get("durable_capital_unit_basis_cents")
            total = item.get("durable_capital_basis_cents")
            if offer.asset_kind != "durable":
                if allocations is not None or total is not None:
                    raise ValueError("retail order line cannot carry durable capital")
                continue
            if (
                not isinstance(allocations, list)
                or len(allocations) != self._ordered_units(item)
                or any(type(value) is not int or value < 0 for value in allocations)
                or type(total) is not int
                or total != sum(allocations)
                or total != self._capital_cents(item["line_cost"])
            ):
                raise ValueError("owned durable order has invalid frozen capital allocation")

    def _durable_items_budget_amount(self, items: list[dict[str, Any]]) -> float:
        if not self._v40_revolving_durable_capital():
            return super()._durable_items_budget_amount(items)
        self._freeze_durable_capital_allocations(items)
        return self._capital_money(
            sum(
                int(item.get("durable_capital_basis_cents", 0))
                for item in items
            )
        ) or 0.0

    def _durable_order_budget_amount(self, order: PurchaseOrder) -> float:
        if not self._v40_revolving_durable_capital():
            return super()._durable_order_budget_amount(order)
        if order.procurement_mode == "supplier_trial":
            return 0.0
        self._validate_order_capital_allocations(order)
        return self._capital_money(
            sum(
                int(item.get("durable_capital_basis_cents", 0))
                for item in order.items
            )
        ) or 0.0

    def _capital_asset_status(self, state: WorldState, asset: RentalAsset) -> str:
        if asset.ownership_type != "owned":
            return "exempt_supplier_owned"
        effective = asset.capital_release_effective_day
        if effective is None:
            return "deployed"
        if state.day < effective:
            return "pending_release_next_day"
        return "released"

    def _durable_capital_summary_cents(self, state: WorldState) -> dict[str, int | None]:
        snapshot = state.durable_acquisition_budget
        deployed = sum(
            int(asset.durable_capital_basis_cents or 0)
            for asset in state.assets
            if asset.ownership_type == "owned"
            and (
                asset.capital_release_effective_day is None
                or state.day < asset.capital_release_effective_day
            )
        )
        pending = sum(
            int(asset.durable_capital_basis_cents or 0)
            for asset in state.assets
            if asset.ownership_type == "owned"
            and asset.capital_release_effective_day == state.day + 1
        )
        committed = 0
        for order in state.orders:
            if order.procurement_mode != "owned":
                continue
            self._validate_order_capital_allocations(order)
            for item in order.items:
                allocations = item.get("durable_capital_unit_basis_cents")
                if allocations is None:
                    continue
                received = int(item.get("received_units", 0))
                committed += sum(map(int, allocations[received:]))
        limit = snapshot.get("durable_capital_limit_cents")
        return {
            "initial_owned_durable_basis_cents": int(
                snapshot["initial_owned_durable_basis_cents"]
            ),
            "additional_durable_capital_headroom_cents": snapshot.get(
                "additional_durable_capital_headroom_cents"
            ),
            "durable_capital_limit_cents": limit,
            "deployed_owned_durable_basis_cents": deployed,
            "committed_inbound_durable_basis_cents": committed,
            "pending_release_next_day_cents": pending,
            "remaining_durable_capital_headroom_cents": (
                None if limit is None else int(limit) - deployed - committed
            ),
        }

    def _durable_acquisition_budget_summary(self, state: WorldState) -> dict[str, Any]:
        if not self._v40_revolving_durable_capital():
            return super()._durable_acquisition_budget_summary(state)
        values = self._durable_capital_summary_cents(state)
        return {
            "capital_policy": POLICY_ID,
            "initial_owned_durable_basis": self._capital_money(
                values["initial_owned_durable_basis_cents"]
            ),
            "additional_durable_capital_headroom": self._capital_money(
                values["additional_durable_capital_headroom_cents"]
            ),
            "durable_capital_limit": self._capital_money(
                values["durable_capital_limit_cents"]
            ),
            "deployed_owned_durable_basis": self._capital_money(
                values["deployed_owned_durable_basis_cents"]
            ),
            "committed_inbound_durable_basis": self._capital_money(
                values["committed_inbound_durable_basis_cents"]
            ),
            "pending_release_next_day": self._capital_money(
                values["pending_release_next_day_cents"]
            ),
            "remaining_durable_capital_headroom": self._capital_money(
                values["remaining_durable_capital_headroom_cents"]
            ),
        }

    def _assert_durable_acquisition_budget(self, state: WorldState) -> None:
        if not self._v40_revolving_durable_capital():
            super()._assert_durable_acquisition_budget(state)
            return
        snapshot = state.durable_acquisition_budget
        expected_additional = (
            None
            if self.config.durable_acquisition_budget is None
            else self._capital_cents(self.config.durable_acquisition_budget)
        )
        initial = snapshot.get("initial_owned_durable_basis_cents")
        if (
            snapshot.get("policy") != POLICY_ID
            or type(initial) is not int
            or initial < 0
            or snapshot.get("additional_durable_capital_headroom_cents")
            != expected_additional
            or snapshot.get("durable_capital_limit_cents")
            != (None if expected_additional is None else initial + expected_additional)
        ):
            raise ValueError("durable capital policy or fixed limit changed during episode")
        for asset in state.assets:
            if asset.ownership_type != "owned":
                if asset.durable_capital_basis_cents is not None:
                    raise ValueError("supplier-owned asset cannot consume durable capital")
                continue
            if (
                type(asset.durable_capital_basis_cents) is not int
                or asset.durable_capital_basis_cents < 0
            ):
                raise ValueError("owned asset is missing an integer-cent capital basis")
            exit_fields = (
                asset.capital_exit_day,
                asset.capital_exit_cause,
                asset.capital_release_effective_day,
            )
            if any(value is not None for value in exit_fields):
                if (
                    asset.capital_exit_day is None
                    or asset.capital_exit_cause not in _EXIT_CAUSES
                    or asset.capital_release_effective_day
                    != asset.capital_exit_day + 1
                ):
                    raise ValueError("owned asset has inconsistent durable-capital exit state")
            source_fields = (
                asset.durable_capital_source_order_id,
                asset.durable_capital_source_item_index,
                asset.durable_capital_source_unit_index,
            )
            if any(value is not None for value in source_fields) and any(
                value is None for value in source_fields
            ):
                raise ValueError("owned asset has incomplete durable-capital provenance")
        initial_asset_basis = sum(
            int(asset.durable_capital_basis_cents or 0)
            for asset in state.assets
            if asset.ownership_type == "owned"
            and asset.durable_capital_source_order_id is None
        )
        if initial_asset_basis != initial:
            raise ValueError("initial owned durable basis no longer reconciles")
        for order in state.orders:
            if order.procurement_mode == "owned":
                self._validate_order_capital_allocations(order)
                for item_index, item in enumerate(order.items):
                    allocations = item.get("durable_capital_unit_basis_cents")
                    if allocations is None:
                        continue
                    received = int(item.get("received_units", 0))
                    for unit_index, expected_basis in enumerate(
                        allocations[:received]
                    ):
                        matches = [
                            asset
                            for asset in state.assets
                            if asset.durable_capital_source_order_id == order.order_id
                            and asset.durable_capital_source_item_index == item_index
                            and asset.durable_capital_source_unit_index == unit_index
                        ]
                        if (
                            len(matches) != 1
                            or matches[0].durable_capital_basis_cents
                            != expected_basis
                        ):
                            raise ValueError(
                                "received durable unit does not reconcile to one asset basis"
                            )
        values = self._durable_capital_summary_cents(state)
        remaining = values["remaining_durable_capital_headroom_cents"]
        if remaining is not None and remaining < 0:
            raise ValueError("durable capital envelope is overcommitted")

    def _durable_acquisition_budget_purchase_check(
        self, state: WorldState, items: list[dict[str, Any]]
    ) -> ToolResult | None:
        if not self._v40_revolving_durable_capital():
            return super()._durable_acquisition_budget_purchase_check(state, items)
        self._assert_durable_acquisition_budget(state)
        required = self._durable_items_budget_amount(items)
        summary = self._durable_acquisition_budget_summary(state)
        remaining = summary["remaining_durable_capital_headroom"]
        if remaining is not None and required > float(remaining) + 1e-9:
            return ToolResult(
                False,
                {
                    "error": "durable_capital_limit_exceeded",
                    "required_order_durable_basis": required,
                    "remaining_durable_capital_headroom": remaining,
                    "durable_capital_limit": summary["durable_capital_limit"],
                    "deployed_owned_durable_basis": summary[
                        "deployed_owned_durable_basis"
                    ],
                    "committed_inbound_durable_basis": summary[
                        "committed_inbound_durable_basis"
                    ],
                    "pending_release_next_day": summary[
                        "pending_release_next_day"
                    ],
                    "cash_and_orders_unchanged": True,
                    "durable_capital": summary,
                },
            )
        return None

    def _received_durable_capital_basis(
        self,
        order: PurchaseOrder,
        item: dict[str, Any],
        *,
        item_index: int,
        unit_index: int,
    ) -> dict[str, Any]:
        if not self._v40_revolving_durable_capital():
            return {}
        allocations = item.get("durable_capital_unit_basis_cents")
        if order.procurement_mode != "owned" or allocations is None:
            return {}
        return {
            "durable_capital_basis_cents": int(allocations[unit_index]),
            "durable_capital_source_order_id": order.order_id,
            "durable_capital_source_item_index": int(item_index),
            "durable_capital_source_unit_index": int(unit_index),
        }

    def _register_durable_capital_exit(
        self, state: WorldState, asset: RentalAsset, *, cause: str
    ) -> None:
        if not self._v40_revolving_durable_capital() or asset.ownership_type != "owned":
            return
        if cause not in _EXIT_CAUSES:
            raise ValueError(f"invalid durable-capital exit cause: {cause}")
        if any(
            value is not None
            for value in (
                asset.capital_exit_day,
                asset.capital_exit_cause,
                asset.capital_release_effective_day,
            )
        ):
            raise ValueError(f"durable capital was already released: {asset.asset_id}")
        asset.capital_exit_day = state.day
        asset.capital_exit_cause = cause
        asset.capital_release_effective_day = state.day + 1

    def _public_asset_durable_capital(
        self, state: WorldState, asset: RentalAsset
    ) -> dict[str, Any]:
        if not self._v40_revolving_durable_capital():
            return {}
        status = self._capital_asset_status(state, asset)
        return {
            "durable_capital_basis": self._capital_money(
                asset.durable_capital_basis_cents
            ),
            "durable_capital_status": status,
            **(
                {
                    "capital_exit_day": asset.capital_exit_day,
                    "capital_exit_cause": asset.capital_exit_cause,
                    "capital_release_effective_day": (
                        asset.capital_release_effective_day
                    ),
                }
                if asset.capital_exit_day is not None
                else {}
            ),
        }

    def _record_rental_activity(
        self,
        state: WorldState,
        event_type: str,
        asset: RentalAsset,
        **details: Any,
    ) -> None:
        super()._record_rental_activity(
            state, event_type, asset, **details
        )
        if not self._v40_revolving_durable_capital():
            return
        if event_type not in {
            "asset_acquired",
            "asset_sold",
            "retired_by_agent",
            "retired_service_life",
        }:
            return
        row = state.rental_activity[-1]
        row.update(
            durable_capital_basis=self._capital_money(
                asset.durable_capital_basis_cents
            ),
            durable_capital_status=self._capital_asset_status(state, asset),
        )
        if asset.capital_exit_day is not None:
            row.update(
                capital_exit_day=asset.capital_exit_day,
                capital_exit_cause=asset.capital_exit_cause,
                capital_release_effective_day=asset.capital_release_effective_day,
            )
