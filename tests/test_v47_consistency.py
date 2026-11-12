"""Business state, settlement receipts and inventory denominators must agree."""
import copy
from dataclasses import replace

import pytest

from outfitter.models import RentalAsset, WorldState
from outfitter.runtime.ledger import empty_ledger
from outfitter.store import RunStore
from outfitter.tool_executor import ToolExecutor
from test_v47_autonomous_rentals import environment
from test_v45_reservation_difficulty import pending_fixed


def revision_environment(track="rental_only", **overrides):
    env = environment(assortment_mode=track, **overrides)
    assert env.dataset.metadata["tool_interface_version"] == 44
    assert env.dataset.metadata["core_release_id"] == "outfitter-v47-autonomous-rental-core-r3"
    return env


def recovery_fixture(track="rental_only"):
    env = revision_environment(track)
    state = env.reset()
    inquiry = next(r for r in env.view_reservation_quote_workspace(
        state, mode="pending_page", offer_projection="references_only")["requests"]
        if r.get("request_kind") == "price_inquiry")
    lines = []
    for line, price in zip(inquiry["lines"], [3.4, 2.1, 2.4]):
        term = line["profile_terms"][0]
        lines.append({"line_id": line["line_id"], "options": [{
            "disposition": "quote", "offer_id": term["frozen_member_offer_ids"][0],
            "promise_profile_id": term["promise_profile_id"], "daily_unit_price": price}]})
    env.quote_reservation_inquiries(state, [{"request_id": inquiry["request_id"],
                                           "action": "quote", "lines": lines}])
    request = next(r for r in state.reservation_requests if r.request_id == inquiry["request_id"])
    plan = next(p for p in request.customer_authorized_plans if p["total_rent"] == 12.1)
    env.confirm_reservation_plans(state, [{"request_id": request.request_id,
        "plan_id": plan["plan_id"], "pickup_day": request.pickup_day}])
    commitment = state.reservation_commitments[-1]
    state.day = commitment.pickup_day
    state.current_day_ledger = empty_ledger(state.day)
    for asset in state.assets:
        asset.status = "maintenance_hold"
    env._process_reservation_pickups(state)
    assert commitment.status == "at_risk"
    assert commitment.preauthorized_recovery_plans
    return env, state, commitment


@pytest.mark.parametrize("track", ["rental_only", "rental_and_retail"])
@pytest.mark.parametrize("failure", ["mixed_path", "other_commitment"])
def test_recovery_failure_is_atomic_and_persisted(tmp_path, track, failure):
    env, state, c = recovery_fixture(track)
    env.resolve_reservation_risks(state, [{"commitment_id": c.commitment_id,
                                         "action": "inquire_fallback"}])
    first = {"commitment_id": c.commitment_id, "action": "select_preauthorized_plan",
             "plan_id": c.preauthorized_recovery_plans[0]["plan_id"]}
    invalid = ({"commitment_id": c.commitment_id, "action": "select_authorized_fallback",
                "line_id": "line-02", "offer_id": "offer-051"}
               if failure == "mixed_path" else
               {"commitment_id": "missing-reservation", "action": "inquire_fallback"})
    before = copy.deepcopy(state.to_dict())
    tools = ToolExecutor(env)
    store = RunStore.create(tmp_path / "run", {}, state)
    saved, result, cached = store.execute_tool("bad-batch", "resolve_reservation_risks",
        {"actions": [first, invalid]}, tools.execute)
    assert not result.ok and not cached
    assert saved.to_dict() == before
    assert store.load_state().to_dict() == before
    assert result.data["state_unchanged"] and result.data["batch_atomic"]
    _, repeated, cached = store.execute_tool("bad-batch", "resolve_reservation_risks",
        {"actions": [first, invalid]}, tools.execute)
    assert cached and repeated.data == result.data
    assert store.load_state().to_dict() == before


def test_failed_inquiry_does_not_consume_right():
    env, state, c = recovery_fixture()
    before = copy.deepcopy(state.to_dict())
    with pytest.raises(ValueError):
        env.resolve_reservation_risks(state, [
            {"commitment_id": c.commitment_id, "action": "inquire_fallback"},
            {"commitment_id": "missing", "action": "inquire_fallback"}])
    assert state.to_dict() == before


def test_multiple_line_changes_commit_one_version():
    env, state, c = recovery_fixture()
    c.fallback_inquiry_day = state.day
    actions = []
    for line in c.lines[:2]:
        alternate = next(offer for offer in env.offers
                         if env.offers[offer].service_line == line.service_line
                         and offer != line.offer_id)
        c.fallback_revealed_offer_ids_by_line[line.line_id] = [alternate]
        actions.append({"commitment_id": c.commitment_id, "action": "select_authorized_fallback",
                        "line_id": line.line_id, "offer_id": alternate})
    version = c.commitment_version
    env.resolve_reservation_risks(state, actions)
    assert c.commitment_version == version + 1
    assert [line.selected_offer_id for line in c.lines[:2]] == [a["offer_id"] for a in actions]


def test_unexpected_error_rolls_back_and_propagates(monkeypatch):
    env, state, c = recovery_fixture()
    env.resolve_reservation_risks(state, [{"commitment_id": c.commitment_id,
                                         "action": "inquire_fallback"}])
    before = copy.deepcopy(state.to_dict())
    def broken_version(*args, **kwargs):
        raise AssertionError("version invariant failed")
    monkeypatch.setattr(env, "_increment_commitment_version", broken_version)
    with pytest.raises(AssertionError, match="version invariant"):
        ToolExecutor(env).execute(state, "resolve_reservation_risks", {"actions": [{
            "commitment_id": c.commitment_id, "action": "select_preauthorized_plan",
            "plan_id": c.preauthorized_recovery_plans[0]["plan_id"]}]})
    assert state.to_dict() == before


def test_recovery_bankruptcy_commits_and_skips_later_invalid_action():
    env, state, c = recovery_fixture()
    env.config = replace(env.config, emergency_credit_limit=0)
    env.resolve_reservation_risks(state, [{"commitment_id": c.commitment_id,
                                         "action": "inquire_fallback"}])
    state.cash = 0
    result = env.resolve_reservation_risks(state, [
        {"commitment_id": c.commitment_id, "action": "select_preauthorized_plan",
         "plan_id": c.preauthorized_recovery_plans[0]["plan_id"]},
        {"commitment_id": "missing", "action": "select_preauthorized_plan", "plan_id": "bad"}])
    assert state.done and state.bankrupt and state.bankruptcy_shortfall > 0
    assert c.selected_recovery_plan_id is not None
    assert c.commitment_version == 2
    assert result["actions"][0]["refund_status"] == "failed"
    assert result["actions"][1]["status"] == "not_executed"
    assert result["actions"][1]["reason"] == "business_terminated"


def partial_fixture(track, kind="fixed"):
    env = revision_environment(track)
    state = env.reset()
    if kind == "fixed":
        request = pending_fixed(env, state, optional=True)
        plan = request.customer_authorized_plans[0]
        env.respond_reservation_service_offers(state, [{"request_id": request.request_id,
            "action": "accept", "plan_id": plan["plan_id"],
            "pickup_day": request.pickup_date_options[0]["pickup_day"]}])
    else:
        request = next(r for r in state.reservation_requests
                       if r.status == "pending_quote" and any(not l.required for l in r.lines))
        lines = []
        for line in request.lines:
            profile = line.eligible_profile_ids[0]
            offer = request.inquiry_profile_member_ids_by_line[line.line_id][profile][0]
            lines.append({"line_id": line.line_id, "options": [{"disposition": "quote",
                "offer_id": offer, "promise_profile_id": profile, "daily_unit_price": 2.0}]})
        env.quote_reservation_inquiries(state, [{"request_id": request.request_id,
                                                "action": "quote", "lines": lines}])
        optional_ids = {line.line_id for line in request.lines if not line.required}
        plan = next(p for p in request.customer_authorized_plans
                    if any(row["line_id"] in optional_ids and row["disposition"] != "omit"
                           for row in p["lines"]))
        env.confirm_reservation_plans(state, [{"request_id": request.request_id,
            "plan_id": plan["plan_id"], "pickup_day": request.pickup_day}])
    c = state.reservation_commitments[-1]
    state.day = c.pickup_day
    state.current_day_ledger = empty_ledger(state.day)
    for asset in state.assets:
        asset.status = "maintenance_hold"
    for line in c.lines:
        if not line.required:
            continue
        offer = env.offers[line.offer_id]
        for i in range(line.quantity):
            asset = RentalAsset(asset_id=f"partial-{line.line_id}-{i}", offer_id=line.offer_id,
                service_line=offer.service_line, unit_cost=offer.unit_cost, used_life_days=0,
                service_life_limit_days=int(offer.service_life_limit_days),
                damage_risk_band=offer.damage_risk_band,
                service_life_limit_cycles=offer.service_life_limit_cycles)
            env._initialize_lifecycle_asset_state(asset)
            asset.durable_capital_basis_cents = env._capital_cents(asset.unit_cost)
            state.assets.append(asset)
    env._initialize_durable_acquisition_budget(state)
    env._process_reservation_pickups(state)
    assert c.partial_fulfillment_missing_line_ids
    return env, state, c


@pytest.mark.parametrize("track", ["rental_only", "rental_and_retail"])
@pytest.mark.parametrize("compensation", [18.0, 21.0, 32.64])
@pytest.mark.parametrize("kind", ["ordinary", "fixed"])
def test_partial_compensation_in_close_receipt_and_next_morning(monkeypatch, track, compensation, kind):
    env, state, c = partial_fixture(track, kind)
    # Isolate factual settlement, not the already tested customer acceptance draw.
    c.partial_fulfillment_inquiry_day = state.day
    c.partial_fulfillment_accepted = True
    c.partial_fulfillment_compensation = compensation
    env.resolve_reservation_risks(state, [{"commitment_id": c.commitment_id,
                                         "action": "select_partial_fulfillment"}])
    monkeypatch.setattr(env, "_simulate_day", lambda state: None)
    cash_before = state.cash
    result = env.close_business_day(state)
    receipt = next(r for r in result.data["closed_day"]["reservation_close_outcomes"]
                   if r["commitment_id"] == c.commitment_id)
    assert receipt["outcome"] == "pickup_fulfilled"
    assert receipt["compensation_due"] == compensation
    assert receipt["compensation_amount_paid"] == compensation
    assert receipt["compensation_paid"] is True
    assert c.compensation_paid == compensation
    financial = state.daily_history[-1]["financial"]
    assert financial["reservation_compensation"] == compensation
    # No physical returns occur in the next morning for these fixtures. Reconcile
    # cash at close, before that next morning can incur any independent flows.
    assert result.data["closed_day"]["end_cash"] == pytest.approx(
        cash_before + receipt["remaining_rent_collected"] + financial["deposits_collected"]
        - compensation - financial["operating_fee"] - financial["rental_holding_cost"]
        - financial["retail_holding_cost"] - financial["refunds"])
    observation = env.observation(state, consume_inbox=False)
    assert receipt in observation["yesterday"]["reservation_close_outcomes"]
    before = copy.deepcopy(state.to_dict())
    env.observation(state, consume_inbox=False)
    assert state.to_dict() == before


def test_legacy_partial_receipt_uses_event_and_does_not_double_count(monkeypatch):
    env, state, c = partial_fixture("rental_only")
    c.partial_fulfillment_inquiry_day = state.day
    c.partial_fulfillment_accepted = True
    c.partial_fulfillment_compensation = 18.0
    c.partial_fulfillment_selected = True
    start = len(state.reservation_events)
    env._process_reservation_pickups(state, finalize_risks=True)
    event = next(e for e in state.reservation_events[start:] if e.event_type == "pickup_fulfilled")
    for key in ("compensation", "compensation_paid", "compensation_amount_paid"):
        event.details.pop(key, None)
    state.current_day_ledger["_close_event_start"] = start
    state.current_day_ledger["_close_payments"] = {}
    report = env._finalize_daily_report(state)
    receipt = report["reservation_close_outcomes"][0]
    assert receipt["compensation_due"] == receipt["compensation_amount_paid"] == 18.0
    assert receipt["compensation_paid"] is True


def test_rejected_partial_offer_cannot_be_selected(monkeypatch):
    env, state, c = partial_fixture("rental_only")
    original = env._stable_event_rng
    class Reject:
        def random(self):
            return 1.0
    monkeypatch.setattr(env, "_stable_event_rng", lambda *args, **kwargs:
        Reject() if str(kwargs.get("identity", "")).startswith("partial:")
        else original(*args, **kwargs))
    result = env.resolve_reservation_risks(state, [{"commitment_id": c.commitment_id,
                                                  "action": "inquire_partial_fulfillment"}])
    assert result["actions"][0]["customer_accepted"] is False
    before = copy.deepcopy(state.to_dict())
    rejected = ToolExecutor(env).execute(state, "resolve_reservation_risks", {"actions": [{
        "commitment_id": c.commitment_id, "action": "select_partial_fulfillment"}]})
    assert not rejected.ok and rejected.data["state_unchanged"]
    assert state.to_dict() == before


@pytest.mark.parametrize("select", [False, True])
def test_partial_pickup_does_not_report_unpaid_compensation_as_success(monkeypatch, select):
    env, state, c = partial_fixture("rental_only")
    c.partial_fulfillment_inquiry_day = state.day
    c.partial_fulfillment_accepted = True
    c.partial_fulfillment_compensation = 1_000_000.0
    if select:
        env.resolve_reservation_risks(state, [{"commitment_id": c.commitment_id,
                                             "action": "select_partial_fulfillment"}])
    monkeypatch.setattr(env, "_simulate_day", lambda state: None)
    result = env.close_business_day(state)
    outcomes = result.data["closed_day"]["reservation_close_outcomes"]
    assert not any(r["outcome"] == "pickup_fulfilled" and r["commitment_id"] == c.commitment_id
                   for r in outcomes)
    if select:
        assert state.bankrupt and state.done
        assert c.compensation_paid == 0
    else:
        assert c.status == "pickup_failed"
        assert c.compensation_paid != 1_000_000.0


@pytest.mark.parametrize("track", ["rental_only", "rental_and_retail"])
def test_summary_filter_selects_skus_not_count_denominators(track):
    env = revision_environment(track)
    state = env.reset()
    original = state.assets[0]
    statuses = ["available", "rented", "cleaning", "sold", "retired", "supplier_returned"]
    state.assets = [replace(original, asset_id=f"summary-{i}", status=status,
                           sold_day=1 if status == "sold" else None,
                           disposed_day=1 if status in {"retired", "supplier_returned"} else None)
                    for i, status in enumerate(statuses)]
    env._initialize_durable_acquisition_budget(state)
    tools = ToolExecutor(env)
    rows = []
    for status in (None, "available", "cleaning"):
        args = {"view": "summary", "offer_id": original.offer_id}
        if status:
            args["status"] = status
        result = tools.execute(state, "view_rental_assets", args)
        assert result.ok
        row = result.data["asset_groups"][0]
        assert row["matched_assets"] == (6 if status is None else 1)
        assert row["total_assets"] == 6
        assert row["held_assets"] == row["owned_assets"] == 3
        assert row["supplier_trial_assets"] == 0
        assert all(row[s] == 1 for s in statuses)
        rows.append({k:v for k,v in row.items() if k != "matched_assets"})
    assert rows[0] == rows[1] == rows[2]
    assert tools.execute(state, "view_rental_assets", {
        "view":"summary", "status":"repairing"}).data["asset_groups"] == []
