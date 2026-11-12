"""Regression: physical continuation must remain queryable and fully reported."""
import copy
from collections import Counter

import pytest

from outfitter.models import RentalAsset, WorldState
from outfitter.runtime.ledger import empty_ledger
from outfitter.tool_executor import ToolExecutor
from test_v47_autonomous_rentals import environment, pickup


def ordinary_pickup(track):
    env = environment(assortment_mode=track)
    state = env.reset()
    request = next(r for r in env.view_reservation_quote_workspace(
        state, mode="pending_page", offer_projection="references_only")["requests"]
        if r.get("request_kind") == "price_inquiry")
    lines = []
    for line, price in zip(request["lines"], [3.4, 2.1, 2.4]):
        term = line["profile_terms"][0]
        lines.append({"line_id": line["line_id"], "options": [{
            "disposition": "quote", "offer_id": term["frozen_member_offer_ids"][0],
            "promise_profile_id": term["promise_profile_id"], "daily_unit_price": price}]})
    env.quote_reservation_inquiries(state, [{"request_id": request["request_id"],
                                           "action": "quote", "lines": lines}])
    quoted = next(r for r in state.reservation_requests if r.request_id == request["request_id"])
    plan = next(p for p in quoted.customer_authorized_plans if p["total_rent"] == 12.1)
    env.confirm_reservation_plans(state, [{"request_id": quoted.request_id,
        "plan_id": plan["plan_id"], "pickup_day": request["pickup_day"]}])
    commitment = state.reservation_commitments[-1]
    state.day = commitment.pickup_day
    state.current_day_ledger = empty_ledger(state.day)
    for asset in state.assets:
        asset.status = "maintenance_hold"
    for line in commitment.lines:
        offer = env.offers[line.offer_id]
        for i in range(line.quantity):
            asset = RentalAsset(asset_id=f"ordinary-{line.line_id}-{i}", offer_id=line.offer_id,
                service_line=offer.service_line, unit_cost=offer.unit_cost, used_life_days=0,
                service_life_limit_days=int(offer.service_life_limit_days),
                damage_risk_band=offer.damage_risk_band,
                service_life_limit_cycles=offer.service_life_limit_cycles)
            env._initialize_lifecycle_asset_state(asset)
            asset.durable_capital_basis_cents = env._capital_cents(asset.unit_cost)
            state.assets.append(asset)
    env._initialize_durable_acquisition_budget(state)
    env._process_reservation_pickups(state)
    assert commitment.status == "active"
    return env, state, commitment, state.rental_episodes[f"trip-{commitment.commitment_id}"]


def record_day(state):
    """Keep only the real ledger fields used by historical service queries."""
    # History queries count completed days; keep the synthetic fixture contiguous.
    next_day = state.daily_history[-1]["day"] + 1 if state.daily_history else 1
    for day in range(next_day, state.day):
        state.daily_history.append({"day": day, "financial": {"rental_revenue": 0.0},
                                   "service_metrics": {}})
    state.daily_history.append({"day": state.day,
        "financial": {"rental_revenue": state.current_day_ledger["rental_revenue"]},
        "service_metrics": copy.deepcopy(state.current_day_ledger["service_metrics"])})


@pytest.mark.parametrize("track", ["rental_only", "rental_and_retail"])
@pytest.mark.parametrize("kind", ["ordinary", "fixed"])
@pytest.mark.parametrize("extra_days", [0, 1, 3])
def test_continuation_preserves_terms_and_public_queries(monkeypatch, track, kind, extra_days):
    env, state, commitment, ep = (ordinary_pickup(track) if kind == "ordinary"
        else pickup(assortment_mode=track))
    state.daily_history = []
    record_day(state)
    locked = commitment.total_locked_rent
    line_terms = [(line.locked_rent, line.duration_days) for line in commitment.lines]
    due, version = commitment.return_day, commitment.commitment_version
    before_revenue = state.cumulative_rental_revenue
    counts = (state.cumulative_customers, state.cumulative_transactions,
              state.cumulative_requested_units, state.cumulative_fulfilled_units)
    daily_rate = round(sum(ep["daily_rates"].values()), 2)
    monkeypatch.setattr(env, "_v47_stays", lambda ep, day, condition: day < due + extra_days)
    for day in range(due, due + extra_days + 1):
        state.day = day
        state.current_day_ledger = empty_ledger(day)
        env._process_morning(state)
        if day < due + extra_days:
            assert state.cumulative_rental_revenue == before_revenue
            assert commitment.post_pickup_rent_collected == 0
        snapshot = copy.deepcopy(state.to_dict())
        env._v47_prepare_returns(state)
        assert state.to_dict() == snapshot
        record_day(state)
    state = WorldState.from_dict(state.to_dict())
    commitment = next(c for c in state.reservation_commitments
                      if c.commitment_id == commitment.commitment_id)
    tools = ToolExecutor(env)
    # This exact query previously crashed after any positive reservation continuation.
    assert tools.execute(state, "view_reservation", {"view": "index", "limit": 1}).ok
    result = tools.execute(state, "view_reservation", {"view": "detail",
        "identifiers": [commitment.commitment_id], "sections": ["current_terms", "fulfillment"]})
    assert result.ok
    terms = result.data["records"][0]["current_terms"]
    assert terms["total_locked_rent"] == locked
    assert terms["continuation_rent_collected"] == pytest.approx(daily_rate * extra_days)
    assert terms["continuation_rent_accrued"] == pytest.approx(daily_rate * extra_days)
    assert terms["extra_occupied_days"] == extra_days
    assert commitment.return_day == due and commitment.commitment_version == version
    assert [(l.locked_rent, l.duration_days) for l in commitment.lines] == line_terms
    assert (state.cumulative_customers, state.cumulative_transactions,
            state.cumulative_requested_units, state.cumulative_fulfilled_units) == counts
    env._assert_v26_reservation_financial_invariants(state)
    if kind == "ordinary" and extra_days == 1:
        assert locked == 12.1
        assert commitment.post_pickup_rent_collected == 6.05
        assert env._v27_reservation_cash_history(commitment)["net_rent_cash_collected"] == 18.15
    state.day += 1
    for granularity in ("totals", "daily", "offer"):
        result = tools.execute(state, "service_history", {"period": "season_to_date",
            "granularity": granularity, **({"limit": 50,
                "offer_ids": sorted({line.offer_id for line in commitment.lines})}
                if granularity == "offer" else {})})
        assert result.ok
        rows = result.data["offers"] if granularity == "offer" else [result.data["totals"]]
        assert sum(r["revenue"] for r in rows) == pytest.approx(locked + daily_rate * extra_days)
        assert sum(r["continuation_rental_revenue"] for r in rows) == pytest.approx(daily_rate * extra_days)
        assert sum(r["continuation_rental_days"] for r in rows) == len(ep["contract_ids"]) * extra_days
    snapshot = copy.deepcopy(state.to_dict())
    env._v47_prepare_returns(state)
    assert state.to_dict() == snapshot


@pytest.mark.parametrize("track", ["rental_only", "rental_and_retail"])
def test_walkin_continuation_posts_days_and_cash_without_new_sales(monkeypatch, track):
    env = environment(assortment_mode=track)
    state = env.reset()
    state.prices["offer-009"] = 4.01
    session = env._make_choice_session(state, service_key="rental:camp_chair", units_requested=2,
        duration_days=2, adults=2, children=0, size_class="n/a", archetype="picnic", weather="sunny",
        request_index=0, event_address=(state.day, 1, 999), intent_role="required")
    env._rent_from_offer(state, session, "offer-009", 2, Counter())
    ep = next(iter(state.rental_episodes.values()))
    due = ep["contractual_return_day"]
    monkeypatch.setattr(env, "_v47_stays", lambda ep, day, condition: day == due)
    counts = state.cumulative_requested_units, state.cumulative_fulfilled_units
    for day in (due, due + 1):
        state.day = day
        state.current_day_ledger = empty_ledger(day)
        env._process_morning(state)
        metric = state.current_day_ledger["service_metrics"]["rental:camp_chair"]
        offer = metric["offer_fulfillment_breakdown"]["offer-009"]
        assert metric["requested_units"] == metric["fulfilled_units"] == 0
        assert offer["fulfilled_units"] == 0
        assert offer["continuation_rental_days"] == (2 if day == due else 0)
        assert offer["rental_days"] == (2 if day == due else 0)
        assert offer["revenue"] == (0 if day == due else 4.02)
        assert offer["continuation_rental_revenue"] == offer["revenue"]
        assert metric["revenue"] == state.current_day_ledger["rental_revenue"]
    assert (state.cumulative_requested_units, state.cumulative_fulfilled_units) == counts
