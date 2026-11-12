import copy
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import pytest

from outfitter.reservation_overlap import (PRIVATE_CONTEXT,TYPES,SHARES,MARKET_FACTORS,
                                          normalized_block,policy_document)
from outfitter.reservation_difficulty import rental_amount


def configuration():
    from outfitter.config import OutfitterConfig
    return OutfitterConfig.from_yaml(Path(__file__).resolve().parents[1]/'configs/outfitter_v54_main.example.yaml')


def environment():
    from outfitter.data import OutfitterDataset
    from outfitter.environment import OutfitterEnvironment
    c=configuration();return OutfitterEnvironment(c,OutfitterDataset.load(c.data_dir))


@pytest.mark.parametrize('n',[1,4,14])
def test_block_mass_and_type_shares(n):
    days=list(range(1,n+1));rates={d:1+d*.13 for d in days}
    cal=lambda d:{'is_weekend':d%7 in (0,6),'day_type':'weekday','future_weather':'not_read'}
    result=normalized_block(days,rates,cal)
    assert sum(sum(v) for v in result.values())==pytest.approx(sum(rates.values()))
    for i,share in enumerate(SHARES):assert sum(v[i] for v in result.values())==pytest.approx(share*sum(rates.values()))


def test_school_holiday_alone_not_a_peak_and_zero_pool():
    rates={d:1. for d in range(1,15)}
    cal=lambda d:{'is_weekend':False,'day_type':'weekday','is_school_holiday':True}
    result=normalized_block(list(rates),rates,cal)
    assert all(tuple(v)==pytest.approx(SHARES) for v in result.values())
    assert normalized_block([1],{1:0.},cal)=={1:(0.,0.,0.)}


def test_full_native_mass_conservation_and_other_streams_unchanged():
    from outfitter.data import OutfitterDataset
    from outfitter.environment import OutfitterEnvironment
    from release_fixtures import v50_runtime
    c=configuration();parent=v50_runtime()
    old=OutfitterEnvironment(replace(c,data_dir=str(parent),daily_fee=15.),OutfitterDataset.load(parent))
    new=environment()
    for first in range(1,201,14):
        days=range(first,min(first+14,201))
        assert sum(new._v47_arrival_rate(d,'picnic') for d in days)==pytest.approx(sum(old._v47_arrival_rate(d,'picnic') for d in days))
    for kind in ('overnight','lightweight'):
        assert [new._v47_arrival_rate(d,kind) for d in range(1,201)]==[old._v47_arrival_rate(d,kind) for d in range(1,201)]


def selected_profiles(env,day=60):
    found={}
    for i in range(200):
        profile=env._overlap_profile((day,2,i));found.setdefault(profile['type'],(i,profile))
    assert set(found)==set(TYPES)
    return found


def test_frozen_identity_route_shape_and_unrelated_calls():
    e=environment()
    for kind,(index,profile) in selected_profiles(e).items():
        shape=e._v47_customer_shape(60,'picnic',index);route=e._reservation_route(60,2,index)
        e._v47_customer_shape(120,'overnight',5);e._reservation_route(85,3,8)
        assert profile==e._overlap_profile((60,2,index))
        assert shape==e._v47_customer_shape(60,'picnic',index) and route==e._reservation_route(60,2,index)
        assert 1<=route[1]<=14
        if kind=='multi_day_camp':assert shape['duration'] in (3,4,5)
        if kind=='short_activity':assert shape['duration'] in (1,2)


def test_customer_attributes_shared_across_lines_channels_and_context_restore():
    e=environment();s=e.reset()
    for kind,(index,profile) in selected_profiles(e).items():
        s.day=50
        primary=e._reservation_primary_session(s,pickup_day=60,archetype='picnic',customer_index=index,booking_lead_days=10)
        assert primary is not None
        sessions=e._reservation_bundle_sessions(s,primary,pickup_day=60,booking_lead_days=10)
        for session,required in sessions:
            frozen=e._frozen_quote_context(session)
            assert frozen[PRIVATE_CONTEXT]==profile
            assert session.request_budget_multiplier==primary.request_budget_multiplier
            assert session.occasion_importance==pytest.approx((1.1 if required else .75)*profile['importance_factor'])
            restored=e._restore_quote_context(frozen)
            assert restored==session
            changed=replace(restored,pickup_day=63,booking_lead_days=13)
            assert e._frozen_quote_context(changed)[PRIVATE_CONTEXT]==profile
        s.day=60
        walk=e._make_choice_session(s,service_key='rental:camp_table',units_requested=1,
            duration_days=primary.duration_days,adults=primary.adults,children=primary.children,size_class='n/a',
            archetype='picnic',weather='sunny',request_index=0,event_address=(60,2,index),intent_role='required')
        assert walk.request_budget_multiplier==primary.request_budget_multiplier
        assert e._frozen_quote_context(walk)[PRIVATE_CONTEXT]==profile


def test_serialization_and_old_state_rejected():
    from outfitter.models import WorldState
    e=environment();s=e.reset();restored=WorldState.from_dict(s.to_dict())
    assert restored.to_dict()==s.to_dict()
    restored.calendar_demand_protocol='summer-rental-arrival-pressure-d2-v1'
    before=copy.deepcopy(restored.to_dict())
    with pytest.raises(ValueError,match='version mismatch'):e.close_business_day(restored)
    assert restored.to_dict()==before


def test_customer_generation_no_unpublished_weather_or_inventory_leak(monkeypatch):
    e=environment();s=e.reset()
    forecasts=copy.deepcopy(e.weather_forecast(s,31))
    original=copy.deepcopy([r.to_dict() if hasattr(r,'to_dict') else __import__('dataclasses').asdict(r) for r in s.reservation_requests])
    from outfitter.models import WorldState
    altered=WorldState.from_dict(s.to_dict());altered.reservation_requests=[];altered.reservation_events=[]
    altered.cumulative_reservation_requests=0;altered.assets=[]
    monkeypatch.setattr(e,'weather_forecast',lambda state,horizon:copy.deepcopy(forecasts[:horizon]))
    actual=e._weather;temperatures=e._temperatures
    monkeypatch.setattr(e,'_weather',lambda day:'snowy' if day>1 else actual(day))
    monkeypatch.setattr(e,'_temperatures',lambda day:(-20.,-25.,-15.) if day>1 else temperatures(day))
    e._overlap_rate_blocks.clear();e._overlap_customer_cache.clear()
    e._generate_reservation_requests(altered)
    regenerated=[__import__('dataclasses').asdict(r) for r in altered.reservation_requests]
    assert regenerated==original


def test_actor_surfaces_do_not_expose_private_types():
    from outfitter.tool_executor import ToolExecutor
    e=environment();s=e.reset();tools=ToolExecutor(e)
    for day in range(1,7):
        for name,args in [('business_status',{}),('reservation_inbox',{}),
            ('view_reservation_quote_workspace',{'mode':'pending_page','offer_projection':'references_only'}),
            ('weather_forecast',{'horizon_days':14})]:
            reply=tools.execute(s,name,args);assert reply.ok,(name,reply.data)
            encoded=json.dumps(reply.data)
            assert PRIVATE_CONTEXT not in encoded
            assert not any(kind in encoded for kind in TYPES)
        e.close_business_day(s)


def test_fixed_factor_from_frozen_context_and_budget_not_reapplied():
    from types import SimpleNamespace
    e=environment();profile=selected_profiles(e)['short_activity'][1]
    r=SimpleNamespace(event_address=[60,2,99,0],quote_context_by_line={'a':{PRIVATE_CONTEXT:profile}})
    draws=[float(e._service_rng(r,'market-factor').choice([.8,1.,1.2])) for _ in range(3)]
    assert len(set(draws))==1 and draws[0] in MARKET_FACTORS['short_activity']


def test_contingency_context_keeps_budget_type_and_importance():
    from outfitter.trip_weather import contextualize
    e=environment();s=e.reset();index,profile=selected_profiles(e)['multi_day_camp'];s.day=50
    session=e._reservation_primary_session(s,pickup_day=60,archetype='picnic',customer_index=index,booking_lead_days=10)
    old=e._frozen_quote_context(session)
    new=contextualize(old,e._trip_weather_context(s,pickup_day=62,duration_days=session.duration_days))
    assert new[PRIVATE_CONTEXT]==old[PRIVATE_CONTEXT]
    assert new['request_budget_multiplier']==old['request_budget_multiplier']
    assert new['occasion_importance']==old['occasion_importance']
    assert e._restore_quote_context(new).pickup_day==62


@pytest.mark.parametrize('track',['rental_only','rental_and_retail'])
def test_both_tracks_share_identical_picnic_latents(track):
    from outfitter.environment import OutfitterEnvironment
    from outfitter.data import OutfitterDataset
    c=configuration();a=environment()
    b=OutfitterEnvironment(replace(c,assortment_mode=track),OutfitterDataset.load(c.data_dir))
    for day in (1,50,92,150,200):
        assert a._v47_arrival_rate(day,'picnic')==b._v47_arrival_rate(day,'picnic')
        for index in range(10):
            assert a._v47_customer_shape(day,'picnic',index)==b._v47_customer_shape(day,'picnic',index)
            assert a._reservation_route(day,2,index)==b._reservation_route(day,2,index)


def test_actual_customer_quote_path_and_public_commitment_privacy():
    from outfitter.tool_executor import ToolExecutor
    e=environment();s=e.reset();tools=ToolExecutor(e);checked=False
    for _ in range(20):
        req=next((r for r in s.reservation_requests if r.created_day==s.day and r.status=='pending_quote'
                  and any(PRIVATE_CONTEXT in v for v in r.quote_context_by_line.values())),None)
        if req:
            workspace=tools.execute(s,'view_reservation_quote_workspace',{'mode':'requests',
                'request_ids':[req.request_id],'offer_projection':'references_only'})
            assert workspace.ok
            public=workspace.data['requests'][0];lines=[]
            for line in public['lines']:
                if not line['required']:
                    options=[{'disposition':'omit'}]
                else:
                    profile=line['profile_terms'][0]
                    options=[{'disposition':'quote','offer_id':profile['frozen_member_offer_ids'][0],
                              'promise_profile_id':profile['promise_profile_id'],'daily_unit_price':.10}]
                lines.append({'line_id':line['line_id'],'options':options})
            frozen=copy.deepcopy(req.quote_context_by_line)
            reply=tools.execute(s,'quote_reservation_inquiries',{'decisions':[{'request_id':req.request_id,
                'action':'quote','lines':lines}]})
            assert reply.ok and req.quote_context_by_line==frozen
            encoded=json.dumps(reply.data)
            assert PRIVATE_CONTEXT not in encoded and not any(t in encoded for t in TYPES)
            checked=True;break
        e.close_business_day(s)
    assert checked
