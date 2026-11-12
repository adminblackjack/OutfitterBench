"""V44 forecast information boundary and factual close settlement receipts."""
from __future__ import annotations

import copy
from ..trip_weather import POLICY_ID, FORECAST_DATES, trip_context


class TripWeatherMixin:
    def public_operating_rules(self):
        result = super().public_operating_rules()
        if self._v44_trip_weather():
            result["reservation_close_outcomes_enabled"] = True
        if self._v45_reservation_difficulty():
            result["reservation_difficulty_v45"] = True
        return result

    def _v44_trip_weather(self):
        return (int(self.dataset.metadata.get("tool_interface_version", 0)) >= 39
                and self.dataset.metadata.get("trip_weather_policy") == POLICY_ID)

    def _trip_weather_context(self, state, *, pickup_day, duration_days):
        return trip_context(issued_day=state.day, pickup_day=int(pickup_day),
                            duration_days=int(duration_days),
                            forecast_rows=self.weather_forecast(state, FORECAST_DATES))

    def _customer_night_low(self, state, duration_days, context):
        if not self._v44_trip_weather():
            return self._binding_night_low(state, duration_days)
        if not context:
            raise ValueError("V44 customer evaluation requires a frozen trip forecast")
        return float(context["minimum_covered_forecast_night_low_c"])

    def close_business_day(self, state, *, handoff_note=None):
        if self._v44_trip_weather() and not state.done:
            state.current_day_ledger["_close_event_start"] = len(state.reservation_events)
            state.current_day_ledger["_close_payments"] = {}
        return super().close_business_day(state, handoff_note=handoff_note)

    def _refund(self, state, amount, contract_id):
        paid = super()._refund(state, amount, contract_id)
        if "_close_payments" in state.current_day_ledger:
            state.current_day_ledger["_close_payments"].setdefault(contract_id, {}).update(
                refund_paid=bool(paid), refund_due=float(amount))
        return paid

    def _fail_reservation_commitment(self, state, commitment, **kwargs):
        result = super()._fail_reservation_commitment(state, commitment, **kwargs)
        if "_close_payments" in state.current_day_ledger:
            state.current_day_ledger["_close_payments"][commitment.commitment_id] = dict(result)
        return result

    def _finalize_daily_report(self, state):
        if not self._v44_trip_weather():
            return super()._finalize_daily_report(state)
        ledger = state.current_day_ledger
        start = ledger.pop("_close_event_start", len(state.reservation_events))
        payments = ledger.pop("_close_payments", {})
        outcomes = []
        for event in state.reservation_events[start:]:
            kind = event.event_type
            if kind not in {"pickup_fulfilled", "pickup_failed"} and not kind.startswith("customer_cancelled_"):
                continue
            details = event.details
            payment = payments.get(event.commitment_id, {})
            refund = float(details.get("prepayment_refund", details.get(
                "refund", payment.get("refund_due", 0.0))))
            compensation = float(details.get("compensation", 0.0))
            refund_paid = bool(payment.get("refund_paid", refund == 0))
            compensation_paid = bool(payment.get("compensation_paid", compensation == 0))
            amount_paid = compensation if compensation_paid else 0.0
            if int(self.dataset.metadata.get("tool_interface_version", 0)) >= 44:
                compensation = float(payment.get("compensation", details.get(
                    "compensation", details.get("partial_fulfillment_compensation", 0.0))))
                # A fulfilled legacy partial pickup is emitted only after the
                # compensation charge succeeds. Missing zero-valued fields alone
                # are not evidence of a successful positive payment.
                legacy_partial_paid = (kind == "pickup_fulfilled"
                    and "partial_fulfillment_compensation" in details)
                compensation_paid = bool(payment.get("compensation_paid", details.get(
                    "compensation_paid", legacy_partial_paid or compensation == 0)))
                amount_paid = float(payment.get("compensation_amount_paid", details.get(
                    "compensation_amount_paid", compensation if compensation_paid else 0.0)))
            outcomes.append({
                "day": event.day, "request_id": event.request_id,
                "commitment_id": event.commitment_id, "outcome": kind,
                "refund_due": refund, "refund_paid": refund_paid,
                "refund_amount_paid": refund if refund_paid else 0.0,
                "compensation_due": compensation, "compensation_paid": compensation_paid,
                "compensation_amount_paid": amount_paid,
                "optional_line_credit": float(details.get("optional_line_credit", 0.0)),
                "remaining_rent_collected": float(details.get("remaining_rent_collected", 0.0)),
            })
        report = super()._finalize_daily_report(state)
        report["reservation_close_outcomes"] = outcomes
        return report

    def _public_daily_report(self, report):
        result = super()._public_daily_report(report)
        if self._v44_trip_weather():
            result["reservation_close_outcomes"] = copy.deepcopy(
                report.get("reservation_close_outcomes", []))
        return result

    def business_manual(self):
        manual = super().business_manual()
        if self._v44_trip_weather():
            manual["reservations"]["rules"].extend([
                "Trip forecasts cover at most today and fourteen future dates. "
                "Dates beyond that window are unknown. A reported minimum covered "
                "forecast night temperature covers only the listed dates, not necessarily the whole trip.",
                "Selecting recovery terms does not complete pickup. reservation_close_outcomes "
                "in the closing receipt and next morning's yesterday report lists actual close-time "
                "pickups, failures and cancellations with refund/compensation payment status. "
                "A handoff note is the manager's statement, not a settlement confirmation."])
        if self._v45_reservation_difficulty():
            manual["reservations"]["rules"].append(
                "A selected optional reservation line is a binding promise. If it is unavailable "
                "at pickup, the reservation becomes at risk. A manager may inquire once about "
                "a specific reduced pickup with full missing-line rent credit and disclosed "
                "compensation. The customer may reject; acceptance requires a separate selection "
                "today. Unresolved risk fails at close. Other authorized substitutions remain available."
            )
        return manual
