"""V54 current-day eligibility is factual; consent remains version-scoped."""
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from outfitter.config import OutfitterConfig
from outfitter.data import OutfitterDataset
from outfitter.environment import OutfitterEnvironment
from outfitter.models import RentalAsset, WorldState
from outfitter.runtime.ledger import empty_ledger
from outfitter.tool_executor import ToolExecutor

PACKAGE = Path(__import__('outfitter').__file__).resolve().parent


def environment(track='rental_only'):
    root = PACKAGE.parent.parent
    config_path = (PACKAGE/'design/release/v53_research_inputs/outfitter_v53_main.yaml'
                   if (PACKAGE/'design').exists() else root/'configs/outfitter_v54_main.example.yaml')
    data = (PACKAGE/'data_validation/artifacts/v54-recovery-state-runtime-20261002'
            if (PACKAGE/'design').exists() else PACKAGE/'runtime_data')
    cfg = replace(OutfitterConfig.from_yaml(config_path), data_dir=str(data), assortment_mode=track)
    return OutfitterEnvironment(cfg, OutfitterDataset.load(data))


def fixture(track='rental_only'):
    env=environment(track);s=env.reset()
    for _ in range(14):
        r=next((r for r in s.reservation_requests if r.status=='pending_service_offer'
                and any(not l.required for l in r.lines)),None)
        if r:break
        env.close_business_day(s)
    assert r
    env.respond_reservation_service_offers(s,[{'request_id':r.request_id,'action':'accept',
        'plan_id':r.customer_authorized_plans[0]['plan_id'],'pickup_day':r.pickup_date_options[0]['pickup_day']}])
    c=s.reservation_commitments[-1]
    s.day=c.pickup_day;s.current_day_ledger=empty_ledger(s.day)
    for a in s.assets:a.status='maintenance_hold'
    for line in c.lines:
        if not line.required:continue
        offer=env.offers[line.offer_id]
        for i in range(line.quantity):
            a=RentalAsset(asset_id=f'v54-{line.line_id}-{i}',offer_id=line.offer_id,
                service_line=offer.service_line,unit_cost=offer.unit_cost,used_life_days=0,
                service_life_limit_days=offer.service_life_limit_days,
                service_life_limit_cycles=offer.service_life_limit_cycles,damage_risk_band=offer.damage_risk_band)
            env._initialize_lifecycle_asset_state(a)
            a.durable_capital_basis_cents=env._capital_cents(a.unit_cost);s.assets.append(a)
    env._initialize_durable_acquisition_budget(s)
    env._process_reservation_pickups(s)
    assert c.status=='at_risk' and c.partial_fulfillment_missing_line_ids
    env._increment_commitment_version(s,c,rebuild_recovery=False)
    assert not c.partial_fulfillment_missing_line_ids
    return env,s,c,ToolExecutor(env)


def action(tools,s,c,name):
    return tools.execute(s,'resolve_reservation_risks',{'actions':[{'commitment_id':c.commitment_id,'action':name}]})


@pytest.mark.parametrize('track',['rental_only','rental_and_retail'])
@pytest.mark.parametrize('accept',[True,False])
def test_fresh_fact_after_version_change_preserves_consent(monkeypatch,track,accept):
    env,s,c,tools=fixture(track)
    before=copy.deepcopy(s.to_dict())
    auth=env.recovery_authorization(s,c)
    assert 'inquire_partial_fulfillment' in auth['available_actions']
    inbox=tools.execute(s,'reservation_inbox',{});assert inbox.ok
    row=next(r for r in inbox.data['risk_alerts'] if r['commitment_id']==c.commitment_id)
    assert row['missing_optional_line_ids'] and row['inquired_missing_optional_line_ids']==[]
    detail=tools.execute(s,'view_reservation',{'view':'detail','identifiers':[c.commitment_id],'sections':['risk']})
    assert detail.ok
    risk=detail.data['records'][0]['risk']
    assert risk['missing_optional_line_ids']==row['missing_optional_line_ids']
    assert 'partial_fulfillment_missing_line_ids' not in risk
    assert s.to_dict()==before
    original=env._stable_event_rng
    monkeypatch.setattr(env,'_stable_event_rng',lambda *a,**kw:
        SimpleNamespace(random=lambda:0. if accept else 1.)
        if str(kw.get('identity','')).startswith('partial:') else original(*a,**kw))
    reply=action(tools,s,c,'inquire_partial_fulfillment');assert reply.ok
    assert reply.data['actions'][0]['customer_accepted'] is accept
    after=copy.deepcopy(s.to_dict());assert not action(tools,s,c,'inquire_partial_fulfillment').ok
    assert s.to_dict()==after
    assert c.compensation_paid==0
    restored=WorldState.from_dict(s.to_dict());assert restored.to_dict()==s.to_dict()
    selected=action(tools,s,c,'select_partial_fulfillment');assert selected.ok is accept
    assert c.compensation_paid==0
    env._process_reservation_pickups(s,finalize_risks=True)
    c=next(x for x in s.reservation_commitments if x.commitment_id==c.commitment_id)
    assert c.status==('active' if accept else 'pickup_failed')
    assert c.compensation_paid>0


def test_required_shortage_does_not_offer_reduction():
    env,s,c,tools=fixture()
    for a in s.assets:a.status='maintenance_hold'
    before=copy.deepcopy(s.to_dict())
    assert 'inquire_partial_fulfillment' not in env.recovery_authorization(s,c)['available_actions']
    assert not action(tools,s,c,'inquire_partial_fulfillment').ok
    assert s.to_dict()==before


def test_inventory_change_does_not_reopen_inquiry_or_rewrite_consent(monkeypatch):
    env,s,c,tools=fixture()
    original=env._stable_event_rng
    monkeypatch.setattr(env,'_stable_event_rng',lambda *a,**kw:
        SimpleNamespace(random=lambda:0.) if str(kw.get('identity','')).startswith('partial:') else original(*a,**kw))
    assert action(tools,s,c,'inquire_partial_fulfillment').ok
    frozen=list(c.partial_fulfillment_missing_line_ids)
    required=next(l for l in c.lines if l.required)
    for a in s.assets:
        if a.offer_id==required.offer_id:a.status='maintenance_hold'
    snapshot=copy.deepcopy(s.to_dict())
    assert not action(tools,s,c,'select_partial_fulfillment').ok
    assert not action(tools,s,c,'inquire_partial_fulfillment').ok
    assert s.to_dict()==snapshot
    env._process_reservation_pickups(s,finalize_risks=True)
    assert c.status=='pickup_failed'
    assert c.partial_fulfillment_missing_line_ids==frozen


def test_same_asset_cannot_cover_two_required_lines_and_trial_filter(monkeypatch):
    env,s,c,tools=fixture()
    line=next(l for l in c.lines if l.required)
    duplicate=copy.deepcopy(line);duplicate.line_id='duplicate-line'
    test_commitment=copy.deepcopy(c);test_commitment.lines=[line,duplicate]
    _,ready,_=env._pickup_asset_selection(s,test_commitment)
    assert not ready
    monkeypatch.setattr(env,'_supplier_trial_asset_can_start_contract',lambda *a,**kw:False)
    _,ready,_=env._pickup_asset_selection(s,c)
    assert not ready


def test_old_calendar_state_is_rejected_without_mutation():
    env=environment();s=env.reset();s.calendar_demand_protocol='reservation-overlap-v51-recovery-r2-resale-disclosure-v1-operating-history-v1'
    before=copy.deepcopy(s.to_dict())
    result=ToolExecutor(env).execute(s,'close_business_day',{})
    assert not result.ok and 'version mismatch' in result.data['error']
    assert s.to_dict()==before


@pytest.mark.parametrize('args',[
    {'view':'detail','identifiers':'["reservation-001"]'},
    {'view':'index','limit':'5'}, {'view':'index','limit':True},
    {'view':'detail','identifiers':[]}, {'view':'timeline','identifiers':['a','b']},
    {'view':'index','identifiers':['a']}, {'view':'quote_history','from_day':None},
    {'view':'detail','identifiers':['a','a']},
])
def test_strict_query_modes(args):
    env=environment();s=env.reset();tools=ToolExecutor(env);before=copy.deepcopy(s.to_dict())
    r=tools.execute(s,'view_reservation',args);assert not r.ok
    assert s.to_dict()==before


def test_flat_schema_and_both_modes_are_read_only():
    env=environment();s=env.reset();tools=ToolExecutor(env)
    for name in ('view_reservation','view_reservation_quote_workspace'):
        definition=next(x['function'] for x in tools.definitions() if x['function']['name']==name)
        assert 'oneOf' not in definition['parameters']
        assert definition['parameters']['properties']
    before=copy.deepcopy(s.to_dict())
    assert tools.execute(s,'view_reservation',{'view':'index'}).ok
    ws=tools.execute(s,'view_reservation_quote_workspace',{'mode':'pending_page'})
    assert ws.ok
    rid=ws.data['requests'][0]['request_id']
    assert tools.execute(s,'view_reservation_quote_workspace',{'mode':'requests','request_ids':[rid]}).ok
    assert not tools.execute(s,'view_reservation_quote_workspace',{'mode':'requests','request_ids':f'["{rid}"]'}).ok
    assert s.to_dict()==before
