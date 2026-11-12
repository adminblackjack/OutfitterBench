from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import copy
import json
from pathlib import Path
import threading
import time

import pytest

from outfitter.evaluation.execution import CpuLease, execution_budget, exclusive_lock, read_state, write_state
from outfitter.evaluation.projection import supply_function


def test_state_codec_preserves_nested_order_and_ignores_temporary(tmp_path):
    value={'z':{'b':1,'a':2},'a':[{'q':3,'c':4}]}
    path=tmp_path/'checkpoint.json.gz'
    write_state(path,value)
    path.with_suffix('.gz.tmp').write_bytes(b'truncated')
    restored=read_state(path)
    assert list(restored)==['z','a'] and list(restored['z'])==['b','a']
    assert restored==value


def test_exclusive_workspace_and_budget_validation(tmp_path):
    with exclusive_lock(tmp_path/'lock'):
        with pytest.raises(RuntimeError,match='already in use'):
            with exclusive_lock(tmp_path/'lock'):pass
    with exclusive_lock(tmp_path/'lock'):pass
    with pytest.raises(ValueError):
        with execution_budget(0):pass
    with execution_budget(2) as (root,size):
        with pytest.raises(ValueError):
            with execution_budget(3,root):pass


def test_global_cpu_budget_and_wait_release():
    active=0;peak=0;guard=threading.Lock()
    with execution_budget(2) as (root,size):
        def task(_):
            nonlocal active,peak
            with CpuLease(root,size):
                with guard:active+=1;peak=max(peak,active)
                time.sleep(.03)
                with guard:active-=1
        with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(task,range(16)))
        assert peak==2
    with execution_budget(1) as (root,size), CpuLease(root,size) as main:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(task_using_lease,root,size)
            with main.suspended():assert future.result(timeout=5)


def task_using_lease(root,size):
    with CpuLease(root,size):return True


def test_supply_index_matches_scan_at_all_boundaries():
    tokens=[('a',2,7,'x'),('a',3,3,'y'),('a',9,4,'z'),('a',5,10,'w'),('b',0,20,'q')]
    supply=supply_function(dict(tokens=tokens))
    for oid in ['a','b','missing']:
        for day in range(-1,23):
            assert supply(oid,day)==sum(o==oid and start<=day<end for o,start,end,_ in tokens)


def test_cli_multi_cutoff_and_budget():
    from outfitter.cli import build_parser
    args=build_parser().parse_args(['evaluate','run','--through-days','1','3','--cpu-budget','24'])
    assert args.through_days==[1,3] and args.cpu_budget==24 and not args.extended
    with pytest.raises(SystemExit):
        build_parser().parse_args(['evaluate','run','--through-day','1','--through-days','2'])


def make_run(tmp_path):
    from test_v54_recovery_state import environment
    from outfitter.runner import OutfitterRunner
    from outfitter.llm import ScriptedProvider
    env=environment(); oid=env.reset().assets[0].offer_id
    def call(name,args,ident):
        return dict(role='assistant',content='',tool_calls=[dict(id=ident,type='function',
                    function=dict(name=name,arguments=json.dumps(args)))])
    messages=[call('set_walk_in_capacity_limits',dict(items=[dict(action='set',offer_id=oid,
        from_day=1,to_day=8,max_walk_in_occupied_units=0)]),'cap')]
    messages.extend(call('close_business_day',{},f'close-{d}') for d in (1,2,3))
    runner=OutfitterRunner.create(config=replace(env.config,runs_dir=str(tmp_path)),
        provider=ScriptedProvider(messages),run_id='parallel-evaluation')
    runner.run(stop_after_day=3)
    return runner.store.run_dir


def semantic(result):
    value=copy.deepcopy(result['metrics']['extended']);value.pop('cache_key',None)
    return value


def test_multicutoff_parallel_serial_restart_and_original_unchanged(tmp_path):
    from outfitter.evaluation.api import evaluate_run,evaluate_run_cutoffs
    run=make_run(tmp_path); before=(run/'run.sqlite3').read_bytes()
    multi=evaluate_run_cutoffs(run,through_days=[1,3],extended=True,plots=False,
        sample_days=[1,2,3],cpu_budget=4)
    # Different sample-list identities force independent serial recomputation.
    single=evaluate_run(run,through_day=1,extended=True,plots=False,sample_days=[1],cpu_budget=1)
    assert semantic(multi[0])==semantic(single)
    assert multi[1]['metrics']['extended']['capacity_relaxation']['records'][0]['candidate_count']>0
    again=evaluate_run_cutoffs(run,through_days=[1,3],extended=True,plots=False,
        sample_days=[1,2,3],cpu_budget=1)
    assert [semantic(r) for r in again]==[semantic(r) for r in multi]
    assert (run/'run.sqlite3').read_bytes()==before
    cache=next(p for p in (run/'analysis/evaluation/outfitter-trajectory-metrics-v1/replay-cache').iterdir()
               if (p/'extended-day-3.json').exists())
    (cache/'extended-day-3.json').unlink()  # Test only: recover output from last daily checkpoint.
    resumed=evaluate_run_cutoffs(run,through_days=[1,3],extended=True,plots=False,
        sample_days=[1,2,3],cpu_budget=2)
    assert [semantic(r) for r in resumed]==[semantic(r) for r in multi]


def test_extension_reuses_earlier_replay_and_does_not_change_early_report(tmp_path):
    from outfitter.evaluation.api import evaluate_run,evaluate_run_cutoffs
    run=make_run(tmp_path)
    first=evaluate_run(run,through_day=1,extended=True,plots=False,cpu_budget=1)
    later=evaluate_run_cutoffs(run,through_days=[1,3],extended=True,plots=False,cpu_budget=3)
    assert semantic(first)==semantic(later[0])
    cache=next((run/'analysis/evaluation/outfitter-trajectory-metrics-v1/replay-cache').iterdir())
    assert json.loads((cache/'timing.json').read_text())['resumed_from_day']==1


def test_branch_failure_propagates_without_scoring_zero(tmp_path):
    from outfitter.evaluation.parallel import BranchDispatcher
    with execution_budget(2) as (root,size),CpuLease(root,size) as lease:
        dispatcher=BranchDispatcher(tmp_path,tmp_path,root,size,None,lease)
        def fail(_):raise RuntimeError('deliberate branch failure')
        dispatcher._run=fail
        dispatcher.enqueue(1)
        with pytest.raises(RuntimeError,match='deliberate branch failure'):dispatcher.collect(1)
        dispatcher.close()


def test_legacy_checkpoint_rejects_missing_sample_and_imports_safe_boundary(tmp_path):
    from outfitter.evaluation.migration import import_checkpoint,LEGACY_CODE
    old=tmp_path/'old';cache=tmp_path/'new';cache.mkdir()
    (cache/'legacy-import.json').write_text(json.dumps(dict(legacy_dir=str(old),code=LEGACY_CODE)))
    state=dict(state=dict(daily_history=[{'day':1}]),next_call=1,seen=[1],rows=dict(capacity_relaxation=[]))
    write_state(old/'checkpoint.json.gz',state)
    calls=[(1,'close_business_day',{}, {'ok':True})]
    with pytest.raises(ValueError,match='sample coverage'):import_checkpoint(old,cache,calls,[1],[1])
    import_checkpoint(old,cache,calls,[],[1])
    assert read_state(cache/'checkpoint.json.gz')==state
