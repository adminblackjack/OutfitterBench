"""Non-replenishing durable acquisition budget for the V36 shared core."""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from ..acquisition_budget import ENVIRONMENT_PROTOCOL, POLICY_ID
from ..models import PurchaseOrder, ToolResult, WorldState


class AcquisitionBudgetMixin:
    def _v36_durable_acquisition_budget(self) -> bool:
        return bool(
            int(self.dataset.metadata.get("tool_interface_version", 0)) >= 31
            and self.dataset.metadata.get("durable_acquisition_budget_policy")
            == POLICY_ID
        )

    def _durable_capital_control_enabled(self) -> bool:
        """Return whether either versioned durable-capital policy is active."""
        return self._v36_durable_acquisition_budget()

    def _validate_durable_acquisition_budget_configuration(self) -> None:
        configured = self.config.durable_acquisition_budget
        if configured is not None and not self._v36_durable_acquisition_budget():
            raise ValueError(
                "durable_acquisition_budget requires a V36 budget runtime"
            )
        if (
            self.dataset.metadata.get("environment_protocol")
            == ENVIRONMENT_PROTOCOL
            and not self._v36_durable_acquisition_budget()
        ):
            raise ValueError("V36 requires its frozen durable acquisition budget policy")

    def _initialize_durable_acquisition_budget(self, state: WorldState) -> None:
        if not self._v36_durable_acquisition_budget():
            return
        state.durable_acquisition_budget = {
            "policy": POLICY_ID,
            "initial_durable_acquisition_budget": (
                None
                if self.config.durable_acquisition_budget is None
                else round(float(self.config.durable_acquisition_budget), 2)
            ),
        }
        self._assert_durable_acquisition_budget(state)

    def _durable_order_budget_amount(self, order: PurchaseOrder) -> float:
        if order.procurement_mode == "supplier_trial":
            return 0.0
        amount = sum(
            (
                Decimal(str(item["line_cost"]))
                for item in order.items
                if self.offers[str(item["offer_id"])].asset_kind == "durable"
            ),
            Decimal("0"),
        )
        return float(amount.quantize(Decimal("0.01")))

    def _durable_items_budget_amount(self, items: list[dict[str, Any]]) -> float:
        amount = sum(
            (
                Decimal(str(item["line_cost"]))
                for item in items
                if self.offers[str(item["offer_id"])].asset_kind == "durable"
            ),
            Decimal("0"),
        )
        return float(amount.quantize(Decimal("0.01")))

    def _durable_acquisition_budget_summary(
        self, state: WorldState
    ) -> dict[str, Any]:
        if not self._v36_durable_acquisition_budget():
            raise ValueError("durable acquisition budget is unavailable")
        used = Decimal("0")
        committed = Decimal("0")
        for order in state.orders:
            amount = Decimal(str(self._durable_order_budget_amount(order)))
            if order.status == "delivered":
                used += amount
            elif order.status == "in_transit":
                committed += amount
            else:
                raise ValueError(
                    f"unsupported purchase order status for budget accounting: {order.status}"
                )
        used_value = float(used.quantize(Decimal("0.01")))
        committed_value = float(committed.quantize(Decimal("0.01")))
        initial = state.durable_acquisition_budget.get(
            "initial_durable_acquisition_budget"
        )
        remaining = (
            None
            if initial is None
            else round(float(initial) - used_value - committed_value, 2)
        )
        return {
            "initial_durable_acquisition_budget": initial,
            "used_durable_acquisition_budget": used_value,
            "committed_durable_acquisition_budget": committed_value,
            "remaining_durable_acquisition_budget": remaining,
        }

    def _assert_durable_acquisition_budget(self, state: WorldState) -> None:
        if not self._v36_durable_acquisition_budget():
            if state.durable_acquisition_budget:
                raise ValueError("cannot load a durable-budget episode in an older runtime")
            return
        snapshot = state.durable_acquisition_budget
        expected = (
            None
            if self.config.durable_acquisition_budget is None
            else round(float(self.config.durable_acquisition_budget), 2)
        )
        if (
            snapshot.get("policy") != POLICY_ID
            or snapshot.get("initial_durable_acquisition_budget") != expected
        ):
            raise ValueError(
                "durable acquisition budget/policy cannot change during an episode"
            )
        summary = self._durable_acquisition_budget_summary(state)
        remaining = summary["remaining_durable_acquisition_budget"]
        if remaining is not None and remaining < -1e-9:
            raise ValueError("durable acquisition budget is overcommitted")

    def _durable_acquisition_budget_purchase_check(
        self, state: WorldState, items: list[dict[str, Any]]
    ) -> ToolResult | None:
        if not self._v36_durable_acquisition_budget():
            return None
        self._assert_durable_acquisition_budget(state)
        required = self._durable_items_budget_amount(items)
        summary = self._durable_acquisition_budget_summary(state)
        remaining = summary["remaining_durable_acquisition_budget"]
        if remaining is not None and required > float(remaining) + 1e-9:
            return ToolResult(
                False,
                {
                    "error": "durable_acquisition_budget_exceeded",
                    "required_durable_acquisition_budget": required,
                    "remaining_durable_acquisition_budget": remaining,
                    "cash_and_orders_unchanged": True,
                    "durable_acquisition_budget": summary,
                },
            )
        return None
