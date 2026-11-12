from dataclasses import dataclass,field,replace
import json
from pathlib import Path

import pytest

from outfitter.evaluation.state_reuse import QueryStateReuse,AUDITED_CORE_SOURCES,ordered_state_digest


@dataclass
class State:
    values: dict=field(default_factory=lambda:{'z':1,'a':2})
    inbox: list=field(default_factory=lambda:['hello'])


def reload(state):
    return State(json.loads(json.dumps(state.values,sort_keys=True)),list(state.inbox))


def guard(mode='reuse',source=None,effects=None):
    return QueryStateReuse(mode,{'source_sha256':source or sorted(AUDITED_CORE_SOURCES)[0]},
        effects or {'memory_read':'read_only','memory_write':'mutation','new_query':'read_only'},reload=reload)


def run_query(g,s,name='memory_read',ok=True):
    g.before_tool(s,name,{})
    g.after_tool(s,name,{},ok)


def test_reuse_pure_query_then_rebuild_after_mutation():
    g=guard();s=g.before_call(State());assert list(s.values)==['a','z']
    run_query(g,s);assert g.before_call(s) is s
    g.before_tool(s,'memory_write',{});s.values['b']=3;g.after_tool(s,'memory_write',{},True)
    other=g.before_call(s)
    assert other is not s and list(other.values)==['a','b','z']
    assert g.stats()['reused_boundaries']==1 and g.stats()['reconstructed_boundaries']==2


@pytest.mark.parametrize('case',['failed','unknown','morning'])
def test_dirty_state_cannot_be_cleaned_by_a_successful_query(case):
    g=guard();s=g.before_call(State())
    if case=='failed':run_query(g,s,ok=False)
    elif case=='unknown':run_query(g,s,name='new_query')
    else:g.invalidate('morning_observation');s.inbox.clear()
    run_query(g,s)
    assert g.before_call(s) is not s


@pytest.mark.parametrize('mode,source',[('reference',None),('reuse','unreviewed')])
def test_reference_and_unknown_source_never_reuse(mode,source):
    g=guard(mode,source);s=g.before_call(State());run_query(g,s)
    assert g.before_call(s) is not s and g.stats()['reused_boundaries']==0


@pytest.mark.parametrize('mode',['reuse','audit'])
def test_purity_guard_detects_order_changes_not_only_value_changes(mode):
    g=guard(mode);s=g.before_call(State());g.before_tool(s,'memory_read',{})
    value=s.values.pop('a');s.values['a']=value
    with pytest.raises(RuntimeError,match='whitelist violation'):g.after_tool(s,'memory_read',{},True)


def test_audit_rechecks_previously_certified_queries_and_canonical_boundary():
    g=guard('audit');s=g.before_call(State());run_query(g,s)
    s.values={'z':1,'a':2}
    with pytest.raises(RuntimeError,match='canonical reference'):g.before_call(s)
    g=guard('audit');s=g.before_call(State());run_query(g,s);g.before_tool(s,'memory_read',{})
    s.values['extra']=3
    with pytest.raises(RuntimeError,match='whitelist violation'):g.after_tool(s,'memory_read',{},True)


def test_non_idempotent_canonicalization_is_rejected():
    def bad(s):return State({'counter':s.values.get('counter',0)+1})
    g=QueryStateReuse('reuse',{'source_sha256':sorted(AUDITED_CORE_SOURCES)[0]}, {},reload=bad)
    with pytest.raises(RuntimeError,match='not idempotent'):g.before_call(State())


def test_new_guard_after_checkpoint_always_reconstructs():
    g=guard();s=g.before_call(State());run_query(g,s)
    assert guard().before_call(s) is not s


def make_query_run(tmp_path):
    from test_v54_recovery_state import environment
    from outfitter.runner import OutfitterRunner
    from outfitter.llm import ScriptedProvider
    env=environment();initial=env.reset();oid=initial.assets[0].offer_id
    messages=[]
    def add(name,args):
        messages.append(dict(role='assistant',content='',tool_calls=[dict(id=f'q-{len(messages)}',type='function',
                         function=dict(name=name,arguments=json.dumps(args)))]))
    for day in (1,2,3):
        add('read_business_rules',{})
        add('view_rental_assets',{'view':'summary'})
        add('business_status',{})
        add('memory_write',dict(name='test',description='reuse test',content=str(day)))
        add('memory_read',dict(name='test'))
        add('memory_list',{})
        add('memory_search',dict(query='test'))
        add('memory_read',dict(name='missing'))
        add('weather_forecast',dict(horizon_days=3))
        add('view_rental_capacity',dict(view='timeline',offer_ids=[oid],from_day=day,to_day=day+2))
        add('reservation_inbox',{})
        add('view_reservation',dict(view='index'))
        add('view_reservation_quote_workspace',dict(mode='pending_page'))
        add('view_active_rentals',{})
        add('counter_log',{})
        add('close_business_day',{})
    runner=OutfitterRunner.create(config=replace(env.config,runs_dir=str(tmp_path)),
        provider=ScriptedProvider(messages),run_id='query-state-reuse')
    runner.run(stop_after_day=3)
    return runner.store.run_dir


def test_reference_reuse_audit_exact_outputs_and_separate_caches(tmp_path):
    from outfitter.evaluation.api import evaluate_run_cutoffs,evaluate_run
    run=make_query_run(tmp_path);outputs={};caches={}
    for mode in ('reference','reuse','audit'):
        rows=evaluate_run_cutoffs(run,through_days=[1,3],extended=True,plots=False,
                                 sample_days=[1,2,3],cpu_budget=4,state_reload=mode)
        outputs[mode]=[row['metrics']['extended'] for row in rows]
        caches[mode]=Path(rows[0]['metrics']['execution']['replay_cache'])
    assert outputs['reference']==outputs['reuse']==outputs['audit']
    assert len(set(caches.values()))==3
    timing={k:json.loads((v/'timing.json').read_text())['state_reuse'] for k,v in caches.items()}
    assert timing['reference']['reused_boundaries']==0
    assert timing['reuse']['reused_boundaries']>0
    assert timing['reuse']['reused_boundaries']==timing['audit']['reused_boundaries']
    assert timing['audit']['purity_checks']>timing['reuse']['purity_checks']
    state_values=[json.loads(__import__('gzip').decompress((p/'checkpoint.json.gz').read_bytes()))['state'] for p in caches.values()]
    assert json.dumps(state_values[0])==json.dumps(state_values[1])==json.dumps(state_values[2])
    # Resume from a close checkpoint without relying on an in-memory certificate.
    (caches['reuse']/'extended-day-3.json').unlink()
    resumed=evaluate_run(run,through_day=3,extended=True,plots=False,sample_days=[1,2,3],
                         cpu_budget=1,state_reload='reuse')
    assert resumed['metrics']['extended']==outputs['reuse'][1]


def test_cli_reference_switch_and_default():
    from outfitter.cli import build_parser
    p=build_parser()
    assert p.parse_args(['evaluate','run']).state_reload=='reuse'
    assert p.parse_args(['evaluate','run','--state-reload','reference']).state_reload=='reference'
