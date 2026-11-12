"""V43 opt-in public documentation boundary; inherited physics are untouched."""
from __future__ import annotations

from ..actor_manual import POLICY_ID, build_business_manual, public_terms
from ..actor_records import public_reservation_events
from ..reservation_difficulty import ASSET_PRICING_BASIS, LINE_PRICING_BASIS


class ActorManualMixin:
    def _v43_actor_manual(self) -> bool:
        return (
            int(self.dataset.metadata.get("tool_interface_version", 0)) >= 38
            and self.dataset.metadata.get("business_manual_policy") == POLICY_ID
        )

    def public_operating_rules(self):
        if self._v43_actor_manual():
            return public_terms(self)
        return super().public_operating_rules()

    def business_manual(self):
        if not self._v43_actor_manual():
            raise ValueError("business rule queries are unavailable")
        return build_business_manual(self)

    def _actor_reservation_events(self, state):
        return public_reservation_events(state.reservation_events, include_guarantee=self._v48_service_guarantee())

    def actor_assortment_projection(self, value):
        projected = super().actor_assortment_projection(value)
        if not self._v43_actor_manual():
            return projected
        # These identify implementation variants or disabled mechanisms, not
        # terms the manager can select. Never remove commitment_version/IDs.
        internal_keys = {
            "policy", "capital_policy", "reservation_contract_mode", "contract_mode",
            "promise_profile_schema_version", "profile_schema_version", "profile_deposit_policy_id",
            "supplier_cancellation_generated", "service_reputation_snapshot",
            "reservation_service_reputation", "reservation_inbox_count",
            "reservation_inbox_count_scope", "settlement_formula",
        }
        trial_summary_keys = {
            "maximum_accepted_trials", "accepted_trials", "remaining_trials",
            "maximum_active_trials", "active_trials", "tried_offer_ids", "trials",
        }

        def project(item):
            if isinstance(item, list):
                return [project(row) for row in item]
            if isinstance(item, dict):
                result = {}
                for key, child in item.items():
                    if key in internal_keys:
                        continue
                    # Shorten actor reports, not the underlying daily ledger.
                    if key == "supplier_trial" and isinstance(child, dict) and "accepted_trials" in child:
                        child = {k: v for k, v in child.items() if k in trial_summary_keys}
                    result[key] = project(child)
                if self._v45_reservation_difficulty():
                    # Label legacy-named monetary fields without renaming tool
                    # arguments or modifying the stored contract amounts.
                    if any(key in result for key in (
                        "line_daily_rent", "line_daily_rents", "locked_line_daily_rent",
                    )) or result.get("pricing_basis") == "line_day":
                        result["pricing_basis"] = LINE_PRICING_BASIS
                    elif any(result.get(key) is not None for key in (
                        "daily_unit_price", "locked_daily_price", "quoted_daily_price",
                    )) and "pricing_basis" not in result:
                        result["pricing_basis"] = ASSET_PRICING_BASIS
                return result
            return item

        return project(projected)
