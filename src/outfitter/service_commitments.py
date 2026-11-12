"""Frozen V32 design constants and exact money helpers (not empirical prices)."""
from decimal import Decimal, ROUND_HALF_UP
from collections import Counter

ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v32-explicit-service-commitments"
POLICY_ID = "outfitter-service-commitments-v1"
TOOL_INTERFACE_VERSION = 28
RNG_NAMESPACE = 3201
POLICY = {
    "format": POLICY_ID,
    "service_offer_attempt_probability": 1 / 3,
    "date_flexibility_probability": 1 / 3,
    "original_dates_unavailable_probability": 0.5,
    "date_offsets": [-3, -2, -1, 1, 2, 3],
    "market_factors": [0.8, 1.0, 1.2],
    "purchase_price_fraction_per_day": 0.1,
    "one_day_cleaning_fraction": 1 / 3,
    "maximum_complete_plans": 3,
    "market_reference_is_empirical": False,
}


def cents(value):
    return int((Decimal(str(value)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def split_cents(amount, count):
    if type(count) is not int or count < 1:
        raise ValueError("positive integer asset count required")
    whole, remainder = divmod(cents(amount), count)
    return [(whole + (i < remainder)) / 100 for i in range(count)]


def service_commitment_diagnostics(state):
    """Factual typed counts and cash, not a causal allocation of shared costs."""
    by_kind = {}
    commitments = {c.request_id: c for c in state.reservation_commitments}
    for kind, fixed in (("price_inquiry", False), ("fixed_reward_offer", True)):
        requests = [r for r in state.reservation_requests if bool(r.service_offer_terms) == fixed]
        request_ids = {r.request_id for r in requests}
        contracts = [commitments[r.request_id] for r in requests if r.request_id in commitments]
        by_kind[kind] = {
            "requests": len(requests), "quotes": sum(r.quote_action == "quote" for r in requests),
            "confirmed": len(contracts), "status_counts": dict(sorted(Counter(r.status for r in requests).items())),
            "requested_service_unit_days": sum(sum(int(l.service_units or l.customer_units) for l in r.lines)
                * (r.pickup_date_options[0]["duration_days"] if r.pickup_date_options else r.return_day-r.pickup_day)
                for r in requests),
            "two_date_requests": sum(len(r.pickup_date_options) == 2 for r in requests),
            "non_original_initial_date_choices": sum(c.initial_agreed_pickup_day is not None
                and c.initial_agreed_pickup_day != c.original_pickup_day for c in contracts),
            "net_rent_cash_collected": sum(cents(c.prepayment_collected)-cents(c.prepayment_refunded)
                +cents(c.rent_collected_at_pickup)+cents(c.post_pickup_rent_collected) for c in contracts)/100,
            "compensation_paid": sum(cents(c.compensation_paid) for c in contracts)/100,
            "pickups": sum(e.event_type == "pickup_fulfilled" and e.request_id in request_ids
                           for e in state.reservation_events),
        }
    return {"format": POLICY_ID, "by_kind": by_kind,
        "service_offer_generation_attempts": sum(bool(r.service_offer_audit.get("attempted")) for r in state.reservation_requests),
        "service_offer_generation_successes": by_kind["fixed_reward_offer"]["requests"],
        "reschedule_original_date_viability_counts": dict(sorted(Counter(
            "still_possible" if c.original_dates_still_possible else "unavailable"
            for c in state.reservation_change_requests if c.original_dates_still_possible is not None).items())),
        "reschedule_decline_customer_cancellations": sum(e.event_type == "customer_cancelled_reschedule_declined" for e in state.reservation_events),
        "reschedule_expiry_customer_cancellations": sum(e.event_type == "customer_cancelled_reschedule_unanswered" for e in state.reservation_events),
        "cash_is_not_net_profit": True, "shared_asset_costs_are_not_arbitrarily_allocated": True}
