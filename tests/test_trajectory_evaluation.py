from __future__ import annotations

from collections import Counter
from types import SimpleNamespace as NS
import json
import sqlite3

import pytest

from outfitter.cli import build_parser
from outfitter.evaluation.basic import basic_metrics
from outfitter.evaluation.common import feasible, line_units, loads, ratio
from outfitter.evaluation.ledger import nominal_calendar
from outfitter.evaluation.projection import public_problem, supply_function, normal_walk_contribution
from outfitter.evaluation.extended import relaxation_safe
from outfitter.evaluation.opportunities import search_menu


def environment():
    return NS(rules=NS(cleaning_capacity_points=30))


def asset(aid='a',oid='tent',status='available',**kw):
    return NS(asset_id=aid,offer_id=oid,status=status,ready_day=None,ownership_type='owned',**kw)


def world(**kw):
    values=dict(day=10,max_days=214,assets=[asset()],contracts=[],cleaning_queue=[],
                supplier_trials=[],orders=[],reservation_commitments=[])
    return NS(**(values|kw))


def test_optional_and_physical_counts():
    assert line_units([{'offer_id':'pan','asset_quantity':2,'required':False}])=={'pan':2}
    assert line_units([{'disposition':'omit','required':False}])=={}
    with pytest.raises(ValueError):line_units([{'offer_id':'pan','asset_quantity':None}])
    assert ratio(0,0)['value'] is None


def test_removed_optional_line_is_not_an_unfunded_promise():
    from outfitter.evaluation.common import OMITTED_LINE_STATUSES
    for status in OMITTED_LINE_STATUSES:
        assert line_units([dict(status=status,quantity=0,offer_id='pan')])=={}


def test_turnaround_and_alternative():
    old=dict(start=25,end=27,units={'tent':1})
    new=dict(start=27,end=28,units={'tent':1})
    supply=lambda oid,t:1
    assert feasible(new,loads({'old':old},2,0),supply,2,0)
    assert not feasible(new,loads({'old':old},2,1),supply,2,1)
    assert feasible(dict(new,units={'other':1}),loads({'old':old},2,1),supply,2,1)


def test_projection_overdue_not_future_truth():
    contract=NS(asset_id='a',status='active',estimated_return_day=None,approved_return_day=9,
                promised_return_day=9,actual_return_day=20,damaged_on_return=True)
    s=world(assets=[asset(status='rented')],contracts=[contract])
    before=public_problem(environment(),s)
    assert before['tokens']==[]
    contract.actual_return_day=100;contract.damaged_on_return=False
    assert public_problem(environment(),s)==before
    contract.estimated_return_day=12
    assert supply_function(public_problem(environment(),s))('tent',12)==0
    assert supply_function(public_problem(environment(),s))('tent',13)==1


def test_current_fifo_and_delayed_orders():
    s=world(assets=[asset('a','tent','cleaning'),asset('b','tent','cleaning')],
            cleaning_queue=[NS(asset_id='a',enqueued_day=9,remaining_work_points=30),
                            NS(asset_id='b',enqueued_day=9,remaining_work_points=1)],
            orders=[NS(status='in_transit',procurement_mode='owned',estimated_delivery_day=9,
                       delivery_day=11,order_id='o',items=[dict(offer_id='tent',packs=10,package_units=1)])])
    p=public_problem(environment(),s);supply=supply_function(p)
    assert supply('tent',11)==1 and supply('tent',12)==2
    s.orders[0].delivery_day=40
    assert public_problem(environment(),s)==p


def test_trial_boundary_and_used_allowance():
    a=asset();a.ownership_type='supplier_trial'
    trial=NS(asset_id='a',return_pending=False,remaining_contract_starts=1,return_deadline=12)
    s=world(assets=[a],supplier_trials=[trial]);supply=supply_function(public_problem(environment(),s))
    assert supply('tent',11)==1 and supply('tent',12)==0
    trial.remaining_contract_starts=0
    assert not public_problem(environment(),s)['tokens']


def test_normal_contribution_does_not_read_future_outcome():
    a=asset(unit_cost=100,service_life_limit_days=100,service_life_limit_cycles=50)
    c=NS(asset_id='a',rent_paid=20,duration_days=2,damaged_on_return=True)
    env=NS(_lifecycle_cleaning_terms=lambda a,**kw:dict(cash_cost=3))
    value=normal_walk_contribution(env,world(assets=[a]),[c])
    assert value['contribution']==13
    c.damaged_on_return=False
    assert normal_walk_contribution(env,world(assets=[a]),[c])==value


def test_relaxation_checks_increment_and_keeps_baseline():
    p=dict(day=10,book={'old':dict(start=12,end=13,units={'tent':1})})
    baseline=[];candidate=[NS(offer_id='tent',return_day=12)]
    assert relaxation_safe(p,Counter(tent=1),baseline,candidate,1)
    assert not relaxation_safe(p,Counter(tent=2),baseline,candidate,1)
    assert not relaxation_safe(p,Counter(tent=1),candidate,candidate,1)


def request():
    line=NS(line_id='l',required=True,eligible_profile_ids=['p'],service_units=2,customer_units=2)
    return NS(lines=[line],inquiry_profile_member_ids_by_line={'l':{'p':['pad']}},
              pickup_date_options=[dict(pickup_day=11,scheduled_return_day=12),dict(pickup_day=14,scheduled_return_day=15)])


def test_candidate_second_date_and_multislot():
    p=dict(day=10,gap=1,tokens=[('pad',13,99,'a')],book={})
    cat=NS(by_id={'p':NS(capacity_per_asset=3)})
    r=search_menu(request(),cat,p)
    assert r['status']=='feasible' and r['witness']['pickup_day']==14
    assert r['witness']['lines'][0]['asset_quantity']==1


def test_candidate_limit_is_unknown_not_infeasible():
    p=dict(day=10,gap=1,tokens=[('pad',30,99,'a')],book={})
    cat=NS(by_id={'p':NS(capacity_per_asset=1)})
    assert search_menu(request(),cat,p,limit=0)['status']=='undetermined'


def minimal_state():
    h=dict(day=1,financial={'rental_revenue':0,'operating_fee':15},end={'cash':2985,'contract_adjusted_economic_value':3000})
    return dict(day=2,max_days=214,daily_history=[h],assets=[],orders=[],contracts=[],reservation_requests=[],
                reservation_commitments=[],reservation_events=[],rental_activity=[],supplier_trials=[])


def test_empty_basic_and_next_morning_isolation():
    s=minimal_state();s['reservation_events']=[dict(day=2,event_type='pickup_failed',commitment_id='future',details={})]
    r=basic_metrics(s,[],1,{},{} )
    assert r['service']['reliability']['value'] is None
    assert r['business']['evaluation_score'] is None
    assert r['service']['merchant_failures']==0


def test_merchant_induced_cancellation_is_failure():
    s=minimal_state();s['reservation_events']=[dict(day=1,event_type='merchant_induced_customer_cancellation',commitment_id='x',details={})]
    r=basic_metrics(s,[],1,{},{} )
    assert r['service']['merchant_failures']==1 and r['service']['reliability']['value']==0


def test_cli_extended_default_off():
    p=build_parser()
    assert not p.parse_args(['evaluate','run']).extended
    assert p.parse_args(['evaluate','run','--extended']).extended


def test_aggregate_does_not_pool_versions_or_duplicate_seed_repeats():
    from outfitter.evaluation.report import aggregate_reports
    def report(group,seed,value):
        b=basic_metrics(minimal_state(),[],1,{},{} )
        b['service']['reliability']=ratio(value,100)
        return dict(comparison_group=group,run_id=f'{group}-{seed}-{value}',
                    run_metadata=dict(model='m',seed=seed),basic=b,extended={})
    result=aggregate_reports([report('v1',42,10),report('v1',42,30),report('v1',73,80),report('v2',42,100)])
    assert len(result)==2
    assert result[0]['metrics']['service_reliability']['mean']==50
    assert result[0]['metrics']['service_reliability']['contributing_seeds']==2


def test_api_basic_never_spawns_replay(tmp_path,monkeypatch):
    from outfitter.evaluation.api import evaluate_run
    run=tmp_path/'run';run.mkdir();(run/'data').mkdir()
    (run/'manifest.json').write_text(json.dumps(dict(run_id='test',config={'max_days':200})))
    with sqlite3.connect(run/'run.sqlite3') as c:
        c.execute('create table state (id integer, value text)');c.execute('insert into state values (1,?)',(json.dumps(minimal_state()),))
        c.execute('create table tool_calls (day,call_id,name,arguments,result)')
    monkeypatch.setattr('subprocess.run',lambda *a,**kw:pytest.fail('basic evaluation launched replay'))
    r=evaluate_run(run,plots=False)
    assert r['extended_status']=='not_enabled'
    with pytest.raises(ValueError):evaluate_run(run,through_day=2,plots=False)


def test_basic_depreciation_counts_only_elapsed_continuation():
    s=minimal_state()
    s['assets']=[dict(asset_id='a',offer_id='bag',acquired_day=0,unit_cost=100,
                      service_life_limit_days=100,service_life_limit_cycles=50)]
    s['contracts']=[dict(contract_id='r',asset_id='a')]
    s['rental_activity']=[dict(event_type='rental_started',day=1,asset_id='a',duration_days=1,rent_paid=5)]
    s['rental_episodes']={'e':dict(contract_ids=['r'],notices=[dict(type='not_returned',day=1),dict(type='not_returned',day=2)])}
    out=basic_metrics(s,[],1,{},{} )
    assert out['capital_productivity']['use_depreciation']==4


def test_actual_menu_omission_cannot_omit_required():
    p=dict(day=10,gap=1,tokens=[('pad',10,99,'a')],book={})
    cat=NS(by_id={'p':NS(capacity_per_asset=3)})
    r=search_menu(request(),cat,p,submitted=[dict(line_id='l',options=[dict(disposition='omit')])])
    assert r['status']=='infeasible'


def test_object_exposure_does_not_parse_notes():
    from outfitter.evaluation.extended import object_refs
    assert object_refs(dict(content='offer-001',records=[{'offer_id':'offer-002'}]))=={'offer-002'}


def test_failure_then_pickup_is_not_silently_deduplicated():
    s=minimal_state()
    s['reservation_events']=[dict(day=1,event_type=k,commitment_id='c',details={}) for k in ['pickup_fulfilled','pickup_failed']]
    with pytest.raises(ValueError):basic_metrics(s,[],1,{},{} )


def test_recurrence_survives_contract_version():
    from outfitter.evaluation.extended import Recorder,summarize
    rec=Recorder(None,'.',[])
    rec.rows['persistent_gaps']=[dict(day=d,commitment_id='c',request_id='r',version=v,
        pickup_day=10,lead_days=10-d,gap=g,offers=['tent']) for d,v,g in [(1,1,True),(2,2,False),(3,3,True)]]
    state=NS(reservation_events=[],orders=[],rental_activity=[])
    r=summarize(rec,state,3)
    assert len(r['persistent_gaps']['episodes'])==2
    assert r['persistent_gaps']['episodes'][-1]['recurrence']
    assert r['persistent_gaps']['episodes'][-1]['ending']=='right_censored'


def test_current_runtime_basic_analyze_and_extended(tmp_path,monkeypatch):
    from dataclasses import replace
    from test_v54_recovery_state import environment as actual_environment
    from outfitter.runner import OutfitterRunner
    from outfitter.llm import ScriptedProvider
    from outfitter.analyze import analyze_run
    from outfitter.evaluation.api import evaluate_run
    env=actual_environment()
    config=replace(env.config,runs_dir=str(tmp_path))
    messages=[dict(role='assistant',content='',tool_calls=[dict(id=f'close-{d}',type='function',
        function=dict(name='close_business_day',arguments='{}'))]) for d in (1,2)]
    runner=OutfitterRunner.create(config=config,provider=ScriptedProvider(messages),run_id='evaluation-test')
    runner.run(stop_after_day=2)
    run=runner.store.run_dir
    state_before=(run/'run.sqlite3').read_bytes()
    subprocess_run=__import__('subprocess').run
    monkeypatch.setattr('subprocess.run',lambda *a,**k:pytest.fail('analyze started replay'))
    analysis=analyze_run(run,plots=False)
    report=json.loads((run/'analysis/report.json').read_text())
    assert report['trajectory_evaluation']['extended_status']=='not_enabled'
    monkeypatch.setattr('subprocess.run',subprocess_run)
    result=evaluate_run(run,extended=True,plots=False,sample_days=[1,2])
    assert result['metrics']['extended']['replay_mismatches']==0
    assert (run/'run.sqlite3').read_bytes()==state_before
    again=evaluate_run(run,extended=True,plots=False,sample_days=[1,2])
    assert result['metrics']['extended']==again['metrics']['extended']
    assert evaluate_run(run,plots=False)['output_dir'].endswith('/basic')
