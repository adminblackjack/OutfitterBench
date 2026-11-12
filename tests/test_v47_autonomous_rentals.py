from __future__ import annotations

from collections import Counter
import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from outfitter.autonomous_rentals import continuation_probability,truncated_budget
from outfitter.config import OutfitterConfig
from outfitter.data import OutfitterDataset
from outfitter.environment import OutfitterEnvironment
from outfitter.models import RentalAsset,WorldState
from outfitter.runtime.ledger import empty_ledger

ROOT=Path(__file__).resolve().parents[1]
from release_fixtures import v47_runtime
RUNTIME=v47_runtime()


def environment(**kwargs):
    config=replace(OutfitterConfig.from_yaml(ROOT/"tests/fixtures/outfitter_v50_legacy_main.yaml"),
                   data_dir=str(RUNTIME),**kwargs)
    return OutfitterEnvironment(config,OutfitterDataset.load(RUNTIME))


def pickup(**kwargs):
    env=environment(**kwargs); state=env.reset()
    for _ in range(15):
        request=next((r for r in state.reservation_requests if r.status=="pending_service_offer"
                     and r.return_day-r.pickup_day>=2),None)
        if request:
            break
        env.close_business_day(state)
    assert request
    plan=request.customer_authorized_plans[0]
    env.respond_reservation_service_offers(state,[{"request_id":request.request_id,"action":"accept",
        "plan_id":plan["plan_id"],"pickup_day":request.pickup_date_options[0]["pickup_day"]}])
    c=state.reservation_commitments[-1]
    state.day=c.pickup_day; state.current_day_ledger=empty_ledger(state.day)
    for a in state.assets:
        a.status="maintenance_hold"
    for line in c.lines:
        offer=env.offers[line.offer_id]
        for i in range(line.quantity):
            a=RentalAsset(asset_id=f"v47-{line.line_id}-{i}",offer_id=line.offer_id,service_line=offer.service_line,
                unit_cost=offer.unit_cost,used_life_days=0,service_life_limit_days=int(offer.service_life_limit_days),
                damage_risk_band=offer.damage_risk_band,service_life_limit_cycles=offer.service_life_limit_cycles)
            env._initialize_lifecycle_asset_state(a)
            a.durable_capital_basis_cents=env._capital_cents(a.unit_cost)
            state.assets.append(a)
    env._initialize_durable_acquisition_budget(state)
    env._process_reservation_pickups(state)
    assert c.status=="active"
    return env,state,c,state.rental_episodes[f"trip-{c.commitment_id}"]


def test_probability_price_weather_and_cap():
    day={"is_weekend":False,"day_type":"weekday"}
    values=dict(archetype="overnight",flexibility=.55,condition="sunny",calendar=day,next_calendar=day,
                daily_rent=5.,cash_scale=30.,extra_days=0)
    p=continuation_probability(**values)
    assert 0<p<1
    assert continuation_probability(**{**values,"daily_rent":20.})<p
    assert continuation_probability(**{**values,"extra_days":4})==0
    assert continuation_probability(**{**values,"extra_days":1})<p
    samples=[truncated_budget(float(x)) for x in np.linspace(0,1,1000)]
    assert min(samples)>.6 and max(samples)<1.6
    assert samples==sorted(samples)


def test_silent_holding_settles_once_without_approval(monkeypatch):
    env,state,c,ep=pickup()
    due=c.return_day; version=c.commitment_version
    ep["private"]["notice_mode"]="silent";ep["notices"]=[];ep["reported_return_day"]=None
    for r in state.contracts:
        r.estimated_return_day=None
    monkeypatch.setattr(env,"_v47_stays",lambda ep,day,condition:day<due+2)
    original_rents={r.contract_id:r.rent_paid for r in state.contracts}
    original_days={r.contract_id:r.duration_days for r in state.contracts}
    before=state.cumulative_rental_revenue
    for day in (due,due+1):
        state.day=day;state.current_day_ledger=empty_ledger(day)
        env._process_morning(state)
        assert c.status=="active" and ep["reported_return_day"] is None
        assert state.cumulative_rental_revenue==before
        assert all(r.status=="active" for r in state.contracts)
        accrued=ep["accrued_rent"]
        env._v47_prepare_returns(state)
        assert ep["accrued_rent"]==accrued
    state.day=due+2;state.current_day_ledger=empty_ledger(state.day)
    env._process_morning(state)
    assert c.status=="completed" and c.actual_return_day==due+2
    assert c.return_day==due and c.commitment_version==version
    assert ep["extra_days"]==2 and ep["status"]=="returned"
    assert state.cumulative_rental_revenue-before==pytest.approx(2*sum(ep["daily_rates"].values()))
    assert all(r.duration_days==original_days[r.contract_id]+2 for r in state.contracts)
    assert all(r.rent_paid==pytest.approx(original_rents[r.contract_id]+2*ep["daily_rates"][r.contract_id]) for r in state.contracts)
    assert not [r for r in state.reservation_change_requests if r.change_type=="extension"]
    assert not [e for e in state.reservation_events if e.event_type=="extension_accepted"]
    env._assert_v26_reservation_financial_invariants(state)
    assert WorldState.from_dict(state.to_dict()).rental_episodes==state.rental_episodes


def test_public_episode_has_no_internal_decision_or_risk_labels():
    env,state,c,ep=pickup()
    public=env._v47_public_episode(ep,state.day)
    text=json.dumps(public)
    for forbidden in ("private","notice_mode","return_today","flexibility","cash_scale","itinerary_type","return_logistics"):
        assert forbidden not in text
    request=next(r for r in state.reservation_requests if r.request_id==c.request_id)
    assert not request.trip_context
    assert "trip_context" not in env._public_reservation_request(request)


def test_manual_preserves_tariff_and_reschedule_without_extension_approval():
    text=json.dumps(environment().business_manual())
    assert 'round_half_up_to_cents(0.5 * P)' in text
    assert 'original_dates_still_possible' in text
    assert 'on_decline/on_expiry' in text
    assert 'Accepting an extension immediately' not in text
    assert 'settled on physical return' in text


def test_v47_daily_opportunity_sampler_does_not_change_actual_close():
    from outfitter.reservation_value import _capture_daily_walk_in_opportunity
    env,state,_,_=pickup();baseline=copy.deepcopy(state)
    with _capture_daily_walk_in_opportunity(env,state) as result:
        actual=env.close_business_day(state)
    expected=env.close_business_day(baseline)
    assert actual==expected and state.to_dict()==baseline.to_dict()
    assert result['actual_walk_in_contracts']>=0


def test_reports_use_current_forecasts_not_future_weather(monkeypatch):
    env,state,c,ep=pickup()
    forecasts=env.weather_forecast(state,14)
    monkeypatch.setattr(env,"weather_forecast",lambda state,horizon:copy.deepcopy(forecasts[:horizon]))
    def forbidden(*args,**kwargs):
        raise AssertionError("report read weather truth")
    monkeypatch.setattr(env,"_weather",forbidden)
    monkeypatch.setattr(env,"_temperatures",forbidden)
    env._v47_report(state,ep)
    assert ep["reported_return_day"] is not None
    assert ep["notices"][-1]["is_guarantee"] is False


def test_normalized_demand_conserves_expected_season_mass():
    env=environment()
    for kind in ("overnight","picnic","lightweight"):
        old=sum(env._v47_base_arrival_rate(d,kind) for d in range(1,201))
        new=sum(env._v47_arrival_rate(d,kind) for d in range(1,201))
        assert new==pytest.approx(old)


def test_trip_shape_and_routing_are_independent_of_store_state():
    env=environment(); state=env.reset()
    before=[env._reservation_route(20,1,i) for i in range(100)]
    state.assets=[];state.cash=-10000
    assert before==[env._reservation_route(20,1,i) for i in range(100)]
    assert {lead for selected,lead in before}==set(range(1,15))
    assert any(env._v47_customer_shape(20,"overnight",i)["basecamp"] for i in range(100))


def test_walkin_assets_share_one_episode_and_no_separate_return_draws(monkeypatch):
    env=environment();state=env.reset();state.prices["offer-009"]=4.
    session=env._make_choice_session(state,service_key="rental:camp_chair",units_requested=2,
        duration_days=2,adults=2,children=0,size_class="n/a",archetype="picnic",
        weather="sunny",request_index=0,event_address=(state.day,1,999),intent_role="required")
    env._rent_from_offer(state,session,"offer-009",2,Counter())
    assert len(state.rental_episodes)==1
    ep=next(iter(state.rental_episodes.values()))
    assert len(ep["contract_ids"])==2 and ep["source"]=="walk_in"
    due=ep["contractual_return_day"]
    monkeypatch.setattr(env,"_v47_stays",lambda ep,day,condition:day==due)
    before=state.cumulative_rental_revenue
    for day in (due,due+1):
        state.day=day;state.current_day_ledger=empty_ledger(day);env._process_morning(state)
    assert ep["actual_return_day"]==due+1 and ep["collected_rent"]==4.
    assert state.cumulative_rental_revenue-before==4.
    assert all(r.actual_return_day==due+1 for r in state.contracts)


def test_reports_do_not_force_physical_return(monkeypatch):
    env,state,c,ep=pickup();due=ep["contractual_return_day"]
    ep["reported_return_day"]=due
    ep["private"]["notice_mode"]="silent"
    monkeypatch.setattr(env,"_v47_stays",lambda ep,day,condition:True)
    state.day=due;state.current_day_ledger=empty_ledger(due)
    env._process_morning(state)
    assert ep["status"]=="active" and c.actual_return_day is None
    assert ep["notices"][-1]["type"]=="not_returned"


def test_v47_and_v46_are_distinct_and_extra_cash_is_versioned():
    env=environment();state=env.reset()
    assert env._v47_autonomous_rentals() and not env._v46_return_dynamics()
    assert env.channel_demand_model.rental_budget_tail_coefficient==1.
    assert state.max_days==214
    env._v47_assert_state_version(state)
    legacy=state.to_dict()
    legacy.pop('rental_episode_protocol',None)
    legacy.pop('rental_episodes',None)
    with pytest.raises(ValueError,match="older rental states"):
        env._v47_assert_state_version(WorldState.from_dict(legacy))


def test_main_and_secondary_share_all_rental_parameters():
    main=OutfitterConfig.from_yaml(ROOT/"tests/fixtures/outfitter_v50_legacy_main.yaml").to_dict()
    secondary=OutfitterConfig.from_yaml(ROOT/"tests/fixtures/outfitter_v50_legacy_secondary.yaml").to_dict()
    assert main.pop("assortment_mode")=="rental_only"
    assert secondary.pop("assortment_mode")=="rental_and_retail"
    assert main==secondary


def test_nonlinear_repricing_matches_production_for_high_prices():
    import hashlib
    from outfitter.known_model import KnowledgeProvider,public_request,option_key
    from outfitter.simple_policy_stress import PRICES
    from outfitter.reservation_choice import CompleteContractOption,score_complete_store_contract
    from outfitter.reservation_difficulty import rental_amount
    env=environment();state=env.reset()
    req=next(row for row in env.view_reservation_quote_workspace(state,mode="pending_page",offer_projection="references_only")["requests"]
             if row["request_kind"]=="price_inquiry")
    candidates=[]
    for line in req["lines"]:
        rows=[{"line_id":line["line_id"],"offer_id":oid,"promise_profile_id":term["promise_profile_id"],
               "asset_quantity":term["asset_quantity"],"security_deposit":term["total_security_deposit"],
               "daily_unit_price":1.,"disposition":"quote"}
                  for term in line["profile_terms"] for oid in term["frozen_member_offer_ids"]]
        assert rows
        candidates.append(rows[:1])
    provider=KnowledgeProvider(env.config,samples=4)
    bank=provider.choice_bank(req,candidates);visible=public_request(req)
    nonce=int(hashlib.sha256(json.dumps(visible,sort_keys=True).encode()).hexdigest()[:7],16)
    previous=np.full(4,np.inf)
    for price in (1.,3.75,10.,50.,100.):
        selected=[{**rows[0],"daily_unit_price":price} for rows in candidates]
        actual=[]
        for date in visible["pickup_date_options"]:
            s,r=provider._sample(visible,date,0,nonce)
            enriched={line["line_id"]:[{"option_id":"|".join(option_key(line)),"offer_id":line["offer_id"],
                "promise_profile_id":line["promise_profile_id"],"asset_quantity":line["asset_quantity"],
                "daily_unit_price":price,"total_rent":rental_amount(price,bank.duration,line["asset_quantity"]),
                "locked_security_deposit":line["security_deposit"]}] for line in selected}
            value=provider._env._reservation_bundle_choice_input(s,r,enriched)
            contract=CompleteContractOption("test",tuple(line.store_options[0] for line in value.lines))
            actual.append(score_complete_store_contract(value,contract))
        assert bank.scores(selected)[0]==pytest.approx(min(actual),abs=1e-8)
        assert np.all(bank.scores(selected)<=previous+1e-8)
        previous=bank.scores(selected)
