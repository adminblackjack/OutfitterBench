"""Public-only deterministic supplier-trial overlays for V37 calibration."""
from __future__ import annotations

from typing import Any, Callable


POLICIES = ("no_trial", "immediate_trial", "delayed_trial", "trial_then_buy")


class SupplierTrialOverlay:
    """A small exploration overlay; it never reads ``WorldState`` directly."""

    def __init__(
        self,
        policy: str,
        rpc: Callable[[str, dict[str, Any]], dict[str, Any]],
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"supplier trial policy must be one of {POLICIES}")
        self.policy = policy
        self.rpc = rpc
        self.target_offer_id: str | None = None
        self.formal_purchase_attempted = False

    def _call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.rpc(name, args or {})

    def _current_assets(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        cursor = 0
        while True:
            result = self._call(
                "view_rental_assets",
                {"view": "asset", "cursor": cursor, "limit": 50},
            )
            if not result["ok"]:
                raise RuntimeError(result["data"])
            rows.extend(result["data"]["assets"])
            if not result["data"].get("has_more"):
                return rows
            cursor = int(result["data"]["next_cursor"])

    def _in_transit_owned_offer_ids(self, day: int) -> set[str]:
        result = self._call(
            "purchase_order_status",
            {
                "status": "in_transit",
                "placed_day_from": 1,
                "placed_day_to": day,
                "limit": 50,
            },
        )
        if not result["ok"]:
            raise RuntimeError(result["data"])
        if result["data"]["summary"]["matching_order_count"] > 50:
            raise RuntimeError("supplier-trial overlay refuses truncated order history")
        return {
            str(item["offer_id"])
            for order in result["data"]["orders"]
            if order.get("procurement_mode", "owned") == "owned"
            for item in order["items"]
        }

    def _candidate(self, merchant: Any, day: int) -> dict[str, Any] | None:
        owned = {
            str(asset["offer_id"])
            for asset in self._current_assets()
            if asset.get("ownership_type", "owned") == "owned"
        }
        unavailable = owned | self._in_transit_owned_offer_ids(day)
        values = [
            offer
            for offer in merchant.offers.values()
            if offer["kind"] == "rental_asset"
            and int(offer["package_units"]) == 1
            and float(offer["supplier_unit_cost"]) <= 1500.0
            and str(offer["offer_id"]) not in unavailable
        ]
        # Calibration needs to exercise the use-fee path, not repeatedly pick
        # the globally cheapest niche SKU. This public, frozen category order
        # selects a low-cost general shelter first and remains independent of
        # realized demand or evaluator outcomes.
        def public_category_priority(row: dict[str, Any]) -> int:
            fields = {
                str(item.get("name")) for item in row.get("specifications", [])
            }
            if "capacity_persons" in fields:
                return 0
            if "comfort_temp_c" in fields:
                return 1
            if "pad_type" in fields or "sleep_slots" in fields:
                return 2
            if "runtime_hours" in fields:
                return 3
            if "chair_style" in fields:
                return 4
            if "functions" in fields:
                return 5
            return 6
        return min(
            values,
            key=lambda row: (
                public_category_priority(row),
                float(row["supplier_unit_cost"]),
                str(row["offer_id"]),
            ),
            default=None,
        )

    def _trial(self, merchant: Any, day: int) -> None:
        candidate = self._candidate(merchant, day)
        if candidate is None:
            merchant.log(day, "supplier_trial", action="no_eligible_candidate")
            return
        result = self._call(
            "place_purchase_order",
            {
                "seller_id": candidate["seller_id"],
                "items": [{"offer_id": candidate["offer_id"], "packs": 1}],
                "procurement_mode": "supplier_trial",
            },
        )
        merchant.log(
            day,
            "supplier_trial",
            action="request_trial",
            offer_id=candidate["offer_id"],
            ok=result["ok"],
            result=result["data"],
        )
        if result["ok"]:
            self.target_offer_id = str(candidate["offer_id"])

    def day(self, merchant: Any, day: int) -> None:
        if self.policy == "no_trial":
            return
        if self.target_offer_id is not None and self.policy != "trial_then_buy":
            return
        if self.formal_purchase_attempted:
            return
        status_result = self._call("business_status")
        if not status_result["ok"]:
            raise RuntimeError(status_result["data"])
        trial = status_result["data"]["supplier_trial"]
        trigger = 1 if self.policy in {"immediate_trial", "trial_then_buy"} else 29
        if int(trial["accepted_trials"]) == 0 and day >= trigger:
            self._trial(merchant, day)
            return
        if (
            self.policy == "trial_then_buy"
            and self.target_offer_id is not None
            and int(trial["accepted_trials"]) > 0
            and int(trial["active_trials"]) == 0
            and not self.formal_purchase_attempted
        ):
            self.formal_purchase_attempted = True
            offer = merchant.offers[self.target_offer_id]
            assets = self._current_assets()
            if any(
                str(asset["offer_id"]) == self.target_offer_id
                and asset.get("ownership_type", "owned") == "owned"
                for asset in assets
            ):
                merchant.log(
                    day,
                    "supplier_trial",
                    action="formal_deployment_already_owned",
                    offer_id=self.target_offer_id,
                )
                return
            price = float(offer["package_price"])
            capital = status_result["data"].get("durable_capital")
            budget = status_result["data"].get("durable_acquisition_budget")
            remaining_budget = (
                capital["remaining_durable_capital_headroom"]
                if capital is not None
                else budget["remaining_durable_acquisition_budget"]
            )
            if (
                float(status_result["data"]["cash"]) + 1e-9 < price
                or float(remaining_budget) + 1e-9 < price
            ):
                merchant.log(
                    day,
                    "supplier_trial",
                    action="formal_purchase_not_affordable",
                    offer_id=self.target_offer_id,
                    package_price=price,
                    cash=status_result["data"]["cash"],
                    remaining_durable_acquisition_budget=remaining_budget,
                )
                return
            result = self._call(
                "place_purchase_order",
                {
                    "seller_id": offer["seller_id"],
                    "items": [{"offer_id": self.target_offer_id, "packs": 1}],
                    "procurement_mode": "owned",
                },
            )
            merchant.log(
                day,
                "supplier_trial",
                action="formal_purchase_after_trial",
                offer_id=self.target_offer_id,
                ok=result["ok"],
                result=result["data"],
            )
