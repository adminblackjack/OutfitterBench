import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from outfitter.config import OutfitterConfig
from outfitter.data import OutfitterDataset
from outfitter.environment import OutfitterEnvironment
from outfitter.models import WorldState, RentalAsset
from outfitter.runtime.ledger import empty_ledger
from outfitter.service_guarantee import POLICY_ID, allocate_targets, cents, line_basis
from outfitter.tool_executor import ToolExecutor

ROOT=Path(__file__).resolve().parents[1]


def environment(track='rental_only',**kwargs):
    cfg=replace(OutfitterConfig.from_yaml(ROOT/'tests/fixtures/outfitter_v50_legacy_main.yaml'),
                assortment_mode=track,data_dir=str(__import__('release_fixtures').v48_runtime()),**kwargs)
    return OutfitterEnvironment(cfg,OutfitterDataset.load(cfg.data_dir))


def fixed(env,state,optional=False):
    for _ in range(14):
        r=next((r for r in state.reservation_requests if r.status=='pending_service_offer'
                and (not optional or any(not l.required for l in r.lines))),None)
        if r:break
        env.close_business_day(state)
    assert r
    env.respond_reservation_service_offers(state,[{'request_id':r.request_id,'action':'accept',
        'plan_id':r.customer_authorized_plans[0]['plan_id'],'pickup_day':r.pickup_date_options[0]['pickup_day']}])
    return state.reservation_commitments[-1]


def ordinary(env,state):
    r=next(r for r in env.view_reservation_quote_workspace(state,mode='pending_page',
        offer_projection='references_only')['requests'] if r['request_kind']=='price_inquiry')
    lines=[]
    for line,price in zip(r['lines'],[3.4,2.1,2.4]):
        term=line['profile_terms'][0]
        lines.append({'line_id':line['line_id'],'options':[{'disposition':'quote',
            'offer_id':term['frozen_member_offer_ids'][0],'promise_profile_id':term['promise_profile_id'],
            'daily_unit_price':price}]})
    env.quote_reservation_inquiries(state,[{'request_id':r['request_id'],'action':'quote','lines':lines}])
    raw=next(x for x in state.reservation_requests if x.request_id==r['request_id'])
    assert raw.customer_authorized_plans
    env.confirm_reservation_plans(state,[{'request_id':raw.request_id,
        'plan_id':raw.customer_authorized_plans[0]['plan_id'],'pickup_day':raw.pickup_day}])
    return state.reservation_commitments[-1]


def at_risk(env,state,c):
    state.day=c.pickup_day;state.current_day_ledger=empty_ledger(state.day)
    for a in state.assets:a.status='maintenance_hold'
    env._process_reservation_pickups(state)
    assert c.status=='at_risk'


@pytest.mark.parametrize('track',['rental_only','rental_and_retail'])
@pytest.mark.parametrize('kind',['fixed','ordinary'])
def test_initial_basis_is_locked_and_public(track,kind):
    env=environment(track);state=env.reset();c=(fixed if kind=='fixed' else ordinary)(env,state)
    g=c.service_guarantee
    assert g['policy_id']==POLICY_ID
    assert g['basis_by_line_cents']=={l.line_id:line_basis(l) for l in c.lines}
    if kind=='fixed':assert g['total_basis_cents']==sum(cents(l.locked_line_daily_rent) for l in c.lines)
    original=copy.deepcopy(g)
    for l in c.lines:l.locked_daily_price=.01;l.locked_rent=.01
    c.total_locked_rent=round(.01*len(c.lines),2)
    env._increment_commitment_version(state,c)
    state.prices={k:.01 for k in state.prices}
    assert c.service_guarantee==original
    assert env._reservation_failure_terms(c,status='pickup_failed',notice_days=0)['compensation']==g['total_basis_cents']/100
    restored=WorldState.from_dict(state.to_dict());env._assert_v48_state(restored)
    out=env.view_reservation(restored,[c.commitment_id],view='detail',sections=['current_terms'])
    assert out['records'][0]['current_terms']['service_guarantee']['original_first_day_basis']==g['total_basis_cents']/100


def test_terms_notice_symmetry_rounding_and_missing_state():
    env=environment();s=env.reset();c=fixed(env,s)
    values=[env._reservation_failure_terms(c,status='cancelled_by_manager',notice_days=n)['compensation'] for n in range(16)]
    assert all(a>=b for a,b in zip(values,values[1:]))
    assert len(set(values[7:]))==1
    assert values[0]==env._reservation_failure_terms(c,status='pickup_failed',notice_days=0)['compensation']
    assert sum(allocate_targets({'a':1,'b':1,'c':1},7).values())==1
    c.service_guarantee=None
    with pytest.raises(ValueError,match='lacks locked guarantee'):
        ToolExecutor(env).execute(s,'business_status',{})


def test_cancel_payment_preview_and_repeat_do_not_double_pay():
    env=environment();s=env.reset();c=fixed(env,s);tools=ToolExecutor(env)
    terms=env.reservation_cancellation_terms(s,c);cash=s.cash
    result=tools.execute(s,'cancel_reservation_commitment',{'commitment_id':c.commitment_id})
    assert result.ok
    assert cents(cash-s.cash)==cents(terms['total_cash_outflow'])
    assert c.compensation_paid==terms['compensation']
    assert c.service_guarantee['settlements'][0]['amount_paid_cents']==cents(c.compensation_paid)
    event=next(e for e in env._actor_reservation_events(s) if e['event_type']=='cancelled_by_manager')
    assert event['details']['compensation_amount_paid']==c.compensation_paid
    from outfitter.actor_records import public_reservation_events
    legacy=next(e for e in public_reservation_events(s.reservation_events) if e['event_type']=='cancelled_by_manager')
    assert 'compensation_amount_paid' not in legacy['details']
    snapshot=copy.deepcopy(s.to_dict())
    assert not tools.execute(s,'cancel_reservation_commitment',{'commitment_id':c.commitment_id}).ok
    assert s.to_dict()==snapshot


@pytest.mark.parametrize('refund_fails',[False,True])
def test_bankruptcy_records_actual_guarantee_cash(refund_fails):
    env=environment(emergency_credit_limit=0);s=env.reset();c=fixed(env,s)
    refund=c.unearned_revenue
    s.cash=round(refund/2,2) if refund_fails else round(refund+1,2)
    result=env.cancel_reservation_commitment(s,commitment_id=c.commitment_id)
    assert s.bankrupt and s.done
    payment=c.service_guarantee['settlements'][0]
    assert payment['amount_paid_cents']==(0 if refund_fails else 100)
    assert payment['amount_due_cents']>payment['amount_paid_cents']
    assert c.compensation_paid==(0 if refund_fails else 1)


def test_partial_assessment_then_whole_failure_only_pays_increment():
    env=environment();s=env.reset();c=fixed(env,s,optional=True)
    ids=[l.line_id for l in c.lines if not l.required]
    a=env._guarantee_assessment(c,ids,0)
    env._record_guarantee_payment(s,c,kind='partial_pickup',line_ids=ids,notice_days=0,paid=a['amount_cents']/100)
    env._increment_commitment_version(s,c)
    assert env._guarantee_assessment(c,ids,0)['amount_cents']==0
    total=c.service_guarantee['total_basis_cents']
    assert cents(env._reservation_failure_terms(c,status='pickup_failed',notice_days=0)['compensation'])==total-a['amount_cents']


def maintenance_asset(env,state):
    a=state.assets[0];a.used_life_days=int(a.service_life_limit_days*.6)
    assert env._lifecycle_maintenance_status(a)['window_open']
    return a


@pytest.mark.parametrize('status',['rented','cleaning','repairing','damaged'])
def test_window_open_does_not_grant_operation(status):
    env=environment();s=env.reset();a=maintenance_asset(env,s);a.status=status
    before=copy.deepcopy(s.to_dict())
    q=env.asset_service_terms(s,a,'maintenance')
    assert not q['eligible_now'] and 'asset_not_available_for_maintenance' in q['ineligible_reasons']
    result=ToolExecutor(env).execute(s,'schedule_maintenance',{'asset_ids':[a.asset_id]})
    assert not result.ok and result.data['asset_id']==a.asset_id and result.data['current_status']==status
    assert s.to_dict()==before


def test_maintenance_quote_batch_fee_and_effect_match():
    env=environment();s=env.reset();assets=s.assets[:2]
    for a in assets:a.used_life_days=int(a.service_life_limit_days*.6)
    quotes=[env.asset_service_terms(s,a,'maintenance') for a in assets]
    assert all(q['eligible_now'] for q in quotes)
    cost=sum(cents(q['cost']) for q in quotes)/100;cash=s.cash
    original=[(a.condition_ratio,a.used_life_days,a.used_rental_cycles) for a in assets]
    result=env.schedule_maintenance(s,[a.asset_id for a in assets]);assert result.ok
    assert result.data['cost']==cost and cents(cash-s.cash)==cents(cost)
    for a,q in zip(assets,quotes):assert a.ready_day==q['completion_day_if_started_today']
    for day in sorted({a.ready_day for a in assets}):
        s.day=day;env._process_maintenance_state(s,[])
    assert [(a.condition_ratio,a.used_life_days,a.used_rental_cycles) for a in assets]==original


def test_individual_cash_eligibility_is_not_batch_affordability():
    env=environment();s=env.reset();assets=s.assets[:2]
    for a in assets:a.used_life_days=int(a.service_life_limit_days*.6)
    s.cash=max(env.asset_service_terms(s,a,'maintenance')['cost'] for a in assets)
    assert all(env.asset_service_terms(s,a,'maintenance')['eligible_now'] for a in assets)
    before=s.to_dict();assert not env.schedule_maintenance(s,[a.asset_id for a in assets]).ok
    assert s.to_dict()==before


def test_repair_quote_matches_cost_completion_and_condition():
    env=environment();s=env.reset();a=s.assets[0]
    a.status='damaged';a.condition_ratio=.7;a.pending_repair_cost=7.23;a.repair_target_condition_ratio=.94
    q=env.asset_service_terms(s,a,'repair');cash=s.cash
    assert q['eligible_now'] and q['effect']['condition_ratio']==.94
    assert env.repair_or_retire(s,[a.asset_id],[]).ok
    assert cents(cash-s.cash)==723 and a.ready_day==q['completion_day_if_started_today']
    s.day=a.ready_day;env._process_morning(s)
    assert a.condition_ratio==.94 and a.status=='available'


def test_recovery_stages_and_atomic_ordered_inquiry_selection():
    env=environment();s=env.reset();c=ordinary(env,s);at_risk(env,s,c);tools=ToolExecutor(env)
    before=copy.deepcopy(s.to_dict());plan=c.preauthorized_recovery_plans[0]['plan_id']
    choose={'commitment_id':c.commitment_id,'action':'select_preauthorized_plan','plan_id':plan}
    auth=env.recovery_authorization(s,c)
    assert auth['stage']=='not_inquired' and 'select_preauthorized_plan' not in auth['available_actions']
    rejected=tools.execute(s,'resolve_reservation_risks',{'actions':[choose]})
    assert not rejected.ok and rejected.data['state_unchanged'] and rejected.data['recovery_authorizations']
    assert s.to_dict()==before
    g=copy.deepcopy(c.service_guarantee)
    result=tools.execute(s,'resolve_reservation_risks',{'actions':[
        {'commitment_id':c.commitment_id,'action':'inquire_fallback'},choose]})
    assert result.ok,result.data
    assert result.data['actions'][-1]['recovery_authorization']['stage']=='selected_pending_pickup'
    assert c.service_guarantee==g
    assert env._reservation_failure_terms(c,status='pickup_failed',notice_days=0)['compensation']==g['total_basis_cents']/100


def test_new_manual_and_asset_queries_are_read_only():
    env=environment();s=env.reset();tools=ToolExecutor(env);before=copy.deepcopy(s.to_dict())
    out=tools.execute(s,'view_rental_assets',{'view':'asset','limit':50});assert out.ok
    assert all('repair' in a and 'maintenance' in a for a in out.data['assets'])
    assert s.to_dict()==before
    rules=' '.join(env.business_manual()['reservations']['rules'])
    assert '0.75' in rules and 'service-unit-days' not in rules and 'max(15' not in rules


@pytest.mark.parametrize('accept',[True,False])
def test_actual_partial_pickup_consent_and_payment(monkeypatch,accept):
    env=environment();s=env.reset();c=fixed(env,s,optional=True)
    s.day=c.pickup_day;s.current_day_ledger=empty_ledger(s.day)
    for asset in s.assets:asset.status='maintenance_hold'
    for line in c.lines:
        if not line.required:continue
        offer=env.offers[line.offer_id]
        for i in range(line.quantity):
            a=RentalAsset(asset_id=f'v48-{line.line_id}-{i}',offer_id=line.offer_id,
                service_line=offer.service_line,unit_cost=offer.unit_cost,used_life_days=0,
                service_life_limit_days=offer.service_life_limit_days,damage_risk_band=offer.damage_risk_band,
                service_life_limit_cycles=offer.service_life_limit_cycles)
            env._initialize_lifecycle_asset_state(a);a.durable_capital_basis_cents=env._capital_cents(a.unit_cost)
            s.assets.append(a)
    env._initialize_durable_acquisition_budget(s)
    env._process_reservation_pickups(s)
    assert c.status=='at_risk' and c.partial_fulfillment_missing_line_ids
    tools=ToolExecutor(env);basis=copy.deepcopy(c.service_guarantee)
    original_rng=env._stable_event_rng
    with monkeypatch.context() as m:
        m.setattr(env,'_stable_event_rng',lambda *a,**kw:
            SimpleNamespace(random=lambda:0. if accept else 1.) if str(kw.get('identity','')).startswith('partial:')
            else original_rng(*a,**kw))
        reply=tools.execute(s,'resolve_reservation_risks',{'actions':[{
            'commitment_id':c.commitment_id,'action':'inquire_partial_fulfillment'}]})
    assert reply.ok
    quote=reply.data['actions'][0]
    assert quote['customer_accepted'] is accept and c.compensation_paid==0
    assert c.service_guarantee==basis
    selected=tools.execute(s,'resolve_reservation_risks',{'actions':[{
        'commitment_id':c.commitment_id,'action':'select_partial_fulfillment'}]})
    assert selected.ok is accept
    assert c.compensation_paid==0
    env._process_reservation_pickups(s,finalize_risks=True)
    if accept:
        assert c.status=='active'
        assert c.compensation_paid==quote['compensation']
        assert c.optional_line_credits==quote['rent_credit']
        assert s.cumulative_reservation_compensation==quote['compensation']
        assert c.service_guarantee['settlements'][0]['kind']=='partial_pickup'
    else:
        # Failed atomic selection may replace state objects; resolve from state.
        c=next(x for x in s.reservation_commitments if x.commitment_id==c.commitment_id)
        assert c.status=='pickup_failed'
        assert c.compensation_paid==basis['total_basis_cents']/100


def test_empty_recovery_candidates_and_contract_change_refresh():
    env=environment();s=env.reset();c=ordinary(env,s);at_risk(env,s,c)
    c.preauthorized_recovery_plans=[];c.version_recovery_offer_ids_by_line={}
    r=next(r for r in s.reservation_requests if r.request_id==c.request_id)
    r.recovery_offer_ids_by_line={};r.fallback_offer_ids_by_line={}
    result=env.resolve_reservation_risks(s,[{'commitment_id':c.commitment_id,'action':'inquire_fallback'}])
    auth=result['actions'][0]['recovery_authorization']
    assert auth['stage']=='inquired_without_candidates'
    assert 'select_preauthorized_plan' not in auth['available_actions']
    g=copy.deepcopy(c.service_guarantee)
    c.status='confirmed';c.pickup_day=s.day+3;c.return_day=c.pickup_day+1
    env._increment_commitment_version(s,c)
    assert c.fallback_inquiry_day is None and c.service_guarantee==g


def test_invalid_mixed_recovery_batch_restores_guarantee():
    env=environment();s=env.reset();c=ordinary(env,s);at_risk(env,s,c)
    before=copy.deepcopy(s.to_dict())
    result=ToolExecutor(env).execute(s,'resolve_reservation_risks',{'actions':[
        {'commitment_id':c.commitment_id,'action':'inquire_fallback'},
        {'commitment_id':c.commitment_id,'action':'select_preauthorized_plan','plan_id':c.preauthorized_recovery_plans[0]['plan_id']},
        {'commitment_id':'not-a-contract','action':'inquire_fallback'}]})
    assert not result.ok and result.data['state_unchanged']
    assert s.to_dict()==before


def test_builder_preserves_world_payload_and_versions_metadata(tmp_path):
    import hashlib,json
    from outfitter.build_service_guarantee_runtime import build
    from release_fixtures import v47_runtime
    old=v47_runtime()
    new=tmp_path/'v48';build(old,new)
    for p in old.iterdir():
        if p.is_file() and p.name not in {'metadata.json','result.json'}:
            assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256((new/p.name).read_bytes()).digest()
    metadata=json.loads((new/'metadata.json').read_text())
    assert metadata['tool_interface_version']==45
    assert metadata['reservation_failure_settlement']['format']==POLICY_ID
    assert OutfitterDataset.load(new).metadata['service_guarantee_policy']==POLICY_ID
    with pytest.raises(ValueError):build(old,new)


def test_supplier_return_pending_blocks_maintenance_but_not_damaged_repair(monkeypatch):
    env=environment();s=env.reset();a=maintenance_asset(env,s)
    monkeypatch.setattr(env,'_supplier_trial_record_for_asset',lambda *args:SimpleNamespace(return_pending=True))
    assert 'supplier_trial_return_pending' in env.asset_service_terms(s,a,'maintenance')['ineligible_reasons']
    a.status='damaged';a.pending_repair_cost=2.;a.repair_target_condition_ratio=.95
    assert env.asset_service_terms(s,a,'repair')['eligible_now']


def test_service_guarantee_is_in_manifest_source_identity_and_prompt(monkeypatch):
    from outfitter.runner import _implementation_identity
    from outfitter.actor_manual import core_system_prompt
    env=environment();terms=env.public_operating_rules()
    prompt=core_system_prompt(terms,tool_call_protocol='read_only_batch_v1',read_only_tools=[],max_read_only_batch_size=5)
    assert '0.75' in prompt and 'Recovery discounts' in prompt
    identity=_implementation_identity()
    read_bytes=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda p:read_bytes(p)+b'\n# digest probe\n'
        if p.name=='service_guarantee.py' and p.parent.name=='outfitter' else read_bytes(p))
    changed=_implementation_identity()
    assert changed['source_sha256']!=identity['source_sha256']
    assert changed['source_file_count']==identity['source_file_count']


def test_real_bankruptcy_cash_flow_and_guarantee_summary_reconcile(tmp_path):
    from outfitter.runner import OutfitterRunner
    from outfitter.run_scripted_merchant import NoModel
    cfg=replace(OutfitterConfig.from_yaml(ROOT/'configs/outfitter_v54_main.example.yaml'),
                initial_cash=1.,emergency_credit_limit=0.,runs_dir=str(tmp_path))
    runner=OutfitterRunner.create(config=cfg,provider=NoModel(),run_id='bankruptcy-payment')
    s=runner.store.load_state();c=fixed(runner.environment,s)
    result=runner.tools.execute(s,'cancel_reservation_commitment',{'commitment_id':c.commitment_id})
    assert result.ok and s.bankrupt and s.done
    summary=runner.summary(s,'bankrupt')
    assert summary['cash_flow']['reconciliation_difference']==0
    assert summary['service_guarantee']['compensation_paid']==1.
    assert summary['service_guarantee']['compensation_unpaid']==s.bankruptcy_shortfall
