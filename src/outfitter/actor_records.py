"""Explicit V43 public reservation-event projection, before filtering/paging."""
from __future__ import annotations

from typing import Any

PUBLIC_EVENT_TYPES = frozenset({
    "continuation_rent_collected",
    "extension_intent_notice", "return_warning_notice", "return_eta_notice",
    "customer_protection_credit_drawn", "inquiry_created", "initial_date_options_frozen",
    "fixed_reward_offer_created", "fixed_reward_offer_accept", "fixed_reward_offer_decline",
    "fixed_reward_offer_expired", "inquiry_expired_unquoted", "inquiry_declined_by_manager",
    "customer_declined_quote", "customer_declined_factorized_plan_set",
    "customer_authorized_factorized_plan_set", "reservation_confirmed_after_quote",
    "reservation_confirmed_from_customer_plan_set", "reschedule_requested",
    "reschedule_accepted", "reschedule_declined", "reschedule_expired", "extension_requested",
    "extension_accepted", "extension_declined", "extension_expired", "late_return_estimated",
    "customer_cancelled_weather", "customer_cancelled_non_weather",
    "customer_cancelled_reschedule_declined", "customer_cancelled_reschedule_unanswered",
    "pickup_risk_exposed", "fallback_inquired", "authorized_fallback_selected",
    "preauthorized_recovery_plan_selected", "cancelled_by_manager", "pickup_failed",
    "merchant_induced_customer_cancellation", "pickup_fulfilled", "reservation_completed",
    "reservation_contingency_requested", "reservation_contingency_reported",
    "reservation_contingency_applied", "reservation_contingency_stale",
    "reservation_contingency_expired",
    "reservation_line_staged", "staged_offer_unavailable_authorized_fallback_used",
})

# These are operational fields, not serialized evaluator objects. Nested
# line-specific maps use public line IDs; unknown detail fields fail closed.
PUBLIC_DETAIL_KEYS = frozenset({
    "may_request_extension", "possible_return_delay",
    "request_id", "commitment_id", "contract_id", "asset_id", "line_id", "offer_id",
    "change_id", "inquiry_id", "report_id", "plan_id", "proposal_id", "selection_id",
    "day", "pickup_day", "return_day", "scheduled_return_day", "promised_return_day",
    "approved_return_day", "estimated_return_day", "actual_return_day", "created_day",
    "response_day", "expires_day", "decision_day", "previous_pickup_day", "previous_return_day",
    "proposed_pickup_day", "proposed_return_day", "previous_duration_days", "proposed_duration_days",
    "pickup_shift_days", "return_shift_days", "duration_delta_days", "duration_days",
    "default_pickup_day", "pickup_date_options", "forecast_issued_day", "forecast_condition",
    "forecast_accuracy", "forecast_temp_mean_c", "forecast_night_low_c", "forecast_day_high_c",
    "booking_lead_days", "original_dates_still_possible", "on_decline", "on_expiry",
    "trip_forecast_coverage", "issued_day", "covered_days", "unknown_days",
    "minimum_covered_forecast_night_low_c",
    "change_type", "status", "outcome", "reason", "late_return", "recovery_deadline",
    "prepayment_collected", "prepayment_refund", "prepayment_refunded", "refund",
    "prepayment_due", "prepayment_rate", "total_rent", "total_locked_rent",
    "previous_total_locked_rent", "projected_total_locked_rent", "current_locked_rent",
    "additional_locked_rent", "additional_rent_collected", "projected_additional_locked_rent",
    "continuation_rent_accrued",
    "projected_additional_rent_collected", "line_rent_deltas", "projected_line_rent_deltas",
    "collection_timing", "extension_days", "rent_collected", "rent_collected_at_pickup",
    "rent_due_at_pickup", "outstanding_rent", "total_cash_outflow", "compensation",
    "compensation_paid", "notice_days", "notice_band", "required_service_unit_days",
    "rent_component", "service_unit_day_floor_component", "cancellation_origin",
    "principal", "fee", "cash", "deposit_collected", "deposit_total", "total_security_deposit",
    "optional_credit", "optional_line_credits", "optional_lines_removed", "optional_missing",
    "optional_missing_line_ids", "removed_optional_line_ids", "line_count", "required_lines",
    "optional_lines", "lines", "locked_lines", "assignments", "quantity", "asset_quantity",
    "customer_units", "service_units", "capacity_per_asset", "required", "service_key",
    "locked_rent", "locked_daily_price", "daily_unit_price", "locked_security_deposit",
    "locked_line_daily_rent", "line_daily_rent", "line_daily_rents", "pricing_basis",
    "selected_offer_id", "authorized_offer_ids", "promise_profile_id", "profile_family_id",
    "selected_recovery_plan_id", "recovery_path", "recovery_price_credit", "recovery_deposit_credit",
    "manager_confirmed_plan_id", "customer_preferred_plan_id", "confirmation_origin",
    "customer_authorized_plans", "authorized_plan_count", "reported_decline_feedback",
    "reported_concerns", "customer_mentioned_line_id", "feedback_available",
    "fulfillment_modes", "fulfillment_mode", "accepted_proposal_ids", "rejected_proposal_ids",
    "counteroffer_available", "counteroffer", "proposals", "line_changes", "settlement",
    "commitment_version", "evaluated_commitment_version", "selected_plan_id",
    "fallback_offer_ids", "fallback_offer_ids_by_line", "preauthorized_recovery_plans",
    "cause", "action", "previous_status", "from_offer_id", "to_offer_id",
    "previous_staged_offer_id", "staged_offer_id", "remaining_rent_collected",
    "deposits_collected", "optional_line_credit", "selected_offers",
    "returned_authorized_plan_count", "confirmation_deadline", "complete_plan_count",
    "proposal_count", "previous_terms", "new_pickup_day", "new_return_day",
    "customer_payable_total_rent", "customer_security_deposit_total", "submitted_line_count",
})


def public_reservation_events(events: list[Any], *, include_guarantee: bool = False) -> list[dict[str, Any]]:
    allowed = PUBLIC_DETAIL_KEYS | ({"compensation_amount_due", "compensation_amount_paid",
        "compensation_amount_unpaid", "guarantee_basis"} if include_guarantee else set())
    def project(value: Any) -> Any:
        if isinstance(value, list):
            return [project(row) for row in value]
        if isinstance(value, dict):
            return {
                key: project(child) for key, child in value.items()
                if key in allowed or str(key).startswith("line-")
            }
        return value

    rows = []
    for event in events:
        if event.event_type not in PUBLIC_EVENT_TYPES:
            continue
        rows.append({
            "event_id": f"reservation-event-{len(rows) + 1:06d}",
            "day": event.day,
            "event_type": ("date_options_provided" if event.event_type == "initial_date_options_frozen"
                           else event.event_type),
            "request_id": event.request_id,
            "commitment_id": event.commitment_id,
            "details": project(event.details),
        })
    return rows
