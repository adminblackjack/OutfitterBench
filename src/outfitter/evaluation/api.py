"""Public evaluation entrypoint. Expensive replay is explicitly opt-in."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from .execution import execution_budget, exclusive_lock, worker_environment, CpuLease

from . import FORMAT
from .basic import basic_metrics
from .common import PARAMETERS, digest, implementation_digest, write_json
from .report import save_report


@contextmanager
def snapshot(run_dir):
    source = Path(run_dir).resolve()
    with tempfile.TemporaryDirectory(prefix='outfitter-metrics-') as directory:
        target = Path(directory)
        manifest = json.loads((source/'manifest.json').read_text())
        for name in ('manifest.json','summary.json','system_prompt.md','business_rules.json'):
            if (source/name).is_file():
                shutil.copy2(source/name, target/name)
        data = source/manifest.get('runtime_data','data')
        (target/'data').symlink_to(data.resolve(), target_is_directory=True)
        manifest['runtime_data'] = 'data'
        write_json(target/'manifest.json', manifest)
        with sqlite3.connect((source/'run.sqlite3').as_uri()+'?mode=ro', uri=True) as incoming:
            with sqlite3.connect(target/'run.sqlite3') as outgoing:
                incoming.backup(outgoing)
        yield target, manifest


def read_inputs(path):
    with sqlite3.connect((path/'run.sqlite3').as_uri()+'?mode=ro', uri=True) as conn:
        state = json.loads(conn.execute('select value from state where id=1').fetchone()[0])
        calls = [dict(rowid=r, day=d, call_id=i, name=n, arguments=json.loads(a), result=json.loads(v))
                 for r,d,i,n,a,v in conn.execute('select rowid,day,call_id,name,arguments,result from tool_calls order by rowid')]
        if conn.execute("select 1 from sqlite_master where type='table' and name='model_calls'").fetchone():
            state['_evaluation_usage_by_day']=[dict(day=d,model_calls=n,input_tokens=it,output_tokens=ot)
                for d,n,it,ot in conn.execute('select day,count(*),sum(input_tokens),sum(output_tokens) from model_calls group by day')]
    pp = path/'data/reservation_promise_profiles.json'
    if (path/'summary.json').is_file():
        summary=json.loads((path/'summary.json').read_text())
        state['_evaluation_initial_value']=summary.get('value_change',{}).get('initial_contract_adjusted_economic_value')
    profiles = {p['profile_id']:p for p in json.loads(pp.read_text())['profiles']} if pp.is_file() else {}
    return state, calls, profiles


def _report(manifest,state,calls,profiles,horizon,identity,selected_days,custom):
    return dict(format=FORMAT, run_id=manifest['run_id'], through_day=horizon,
                      run_metadata={k:manifest['config'].get(k) for k in
                                    ['seed','model','reasoning_effort','assortment_mode','max_days','runoff_days']},
                      comparison_group=digest(dict(environment=manifest.get('controller',{}).get('implementation_identity'),
                          runtime=manifest.get('controller',{}).get('runtime_data_sha256'),
                          settings={k:v for k,v in manifest['config'].items() if k not in
                            {'seed','model','reasoning_effort','thinking_type','temperature','provider_extra_body',
                             'data_dir','runs_dir','base_url','api_key_env','request_timeout_seconds',
                             'max_retries','request_rate_limit_per_minute','rate_limit_retry_seconds'}},
                          cutoff=horizon,protocol=PARAMETERS,code=implementation_digest(),samples=selected_days)),
                      input_identity=identity, cache_key=digest(identity), protocol=PARAMETERS,
                      sample_protocol='custom_diagnostic' if custom else 'default',
                      environment_identity=manifest.get('controller',{}).get('implementation_identity'),
                      basic=basic_metrics(state,calls,horizon,manifest,profiles),
                      extended=dict(status='not_enabled'), warnings=[])


def _archive(output):
    current=output/'metrics.json'
    if not current.exists():return
    previous=json.loads(current.read_text())
    archive=output/'history'/previous.get('cache_key',digest(previous))
    archive.mkdir(parents=True,exist_ok=True)
    for item in output.iterdir():
        if item.is_file() and not (archive/item.name).exists():shutil.copy2(item,archive/item.name)


def evaluate_run(run_dir, *, through_day=None, **kwargs):
    """Backward-compatible single-cutoff entrypoint."""
    return evaluate_run_cutoffs(run_dir,through_days=None if through_day is None else [through_day],**kwargs)[0]


def evaluate_run_cutoffs(run_dir, *, through_days=None, extended=False, source_root=None,
                         output_dir=None, plots=True, sample_days=None, cpu_budget=None,
                         budget_dir=None, legacy_cache=None,state_reload='reuse'):
    from .state_reuse import MODES
    if state_reload not in MODES:raise ValueError(f'unknown state reload mode: {state_reload}')
    original=Path(run_dir).expanduser().resolve()
    base=original/'analysis/evaluation'/FORMAT
    with exclusive_lock(base/'.evaluate.lock'), execution_budget(cpu_budget,budget_dir) as (directory,size):
        with snapshot(original) as (path,manifest):
            with CpuLease(directory,size):
                state,calls,profiles=read_inputs(path)
                last=max((h['day'] for h in state['daily_history']),default=0)
                horizons=sorted(set(int(d) for d in through_days)) if through_days is not None else [last]
                if not horizons or any(not 1<=d<=last for d in horizons):
                    raise ValueError(f'through_day must be in [1, {last}], not {horizons}')
                end=int(state.get('decision_horizon_day') or manifest['config']['max_days'])
                selected=(list(range(5,min(max(horizons),end)+1,5)) if sample_days is None
                          else sorted(set(int(d) for d in sample_days)))
                if any(d<1 or d>min(max(horizons),end) for d in selected):
                    raise ValueError('sample days must be completed decision days within cutoff')
                identity=dict(state=digest(state),calls=digest(calls),manifest=digest(manifest),
                    data=digest({p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest()
                        for p in sorted((path/'data').iterdir()) if p.is_file()}),
                    evaluation_code=implementation_digest(),parameters=PARAMETERS)
                # Cutoffs are presentation boundaries, not replay identities.
                replay_key=digest(dict(identity,sampling='default' if sample_days is None else selected,
                                       state_reload=state_reload))
                cache=base/'replay-cache'/replay_key
                if legacy_cache:
                    if not extended:raise ValueError('legacy cache import requires extended evaluation')
                    from .migration import validate_legacy, LEGACY_CODE
                    validate_legacy(legacy_cache,identity)
                    write_json(cache/'legacy-import.json',dict(legacy_dir=str(Path(legacy_cache).resolve()),code=LEGACY_CODE))
                reports={};outputs={}
                for horizon in horizons:
                    days=[d for d in selected if d<=horizon]
                    full=dict(identity,through_day=horizon,sample_days=days)
                    output=(Path(output_dir).expanduser().resolve() if output_dir else base/f'day-{horizon}')
                    if output_dir and len(horizons)>1:output=output/f'day-{horizon}'
                    if not extended and (output/'metrics.json').exists():
                        old=json.loads((output/'metrics.json').read_text())
                        if old.get('extended',{}).get('status')=='complete':output=output/'basic'
                    if extended:_archive(output)
                    report=_report(manifest,state,calls,profiles,horizon,full,days,sample_days is not None)
                    report['execution']=dict(state_reload=state_reload,cpu_budget=size,replay_cache=str(cache))
                    save_report(report,output,plots=False)
                    reports[horizon]=report;outputs[horizon]=output
            error=None
            if extended:
                request=dict(run_dir=str(path),through_day=max(horizons),through_days=horizons,
                    sample_days=selected,cache_dir=str(cache),cache_key=replay_key,
                    cutoff_keys={str(d):r['cache_key'] for d,r in reports.items()},
                    budget_dir=directory,cpu_budget=size,migration_dir=str(Path(legacy_cache).resolve()) if legacy_cache else None,
                    state_reload=state_reload)
                write_json(cache/'request.json',request)
                cmd=[sys.executable,str(Path(__file__).with_name('worker.py')),
                     '--evaluation-root',str(Path(__file__).parent),'--request',str(cache/'request.json')]
                if source_root is not None:cmd+=['--source-root',str(Path(source_root).expanduser().resolve())]
                result=subprocess.run(cmd,capture_output=True,text=True,env=worker_environment())
                if result.returncode:error=result.stderr[-12000:]
            with CpuLease(directory,size):
                rows=[]
                for horizon,report in reports.items():
                    if extended:
                        complete=cache/f'extended-day-{horizon}.json'
                        report['extended']=(json.loads(complete.read_text()) if complete.exists()
                            else dict(status='error',reason=error or 'missing cutoff output'))
                    save_report(report,outputs[horizon],plots=plots)
                    rows.append(dict(run_id=report['run_id'],through_day=horizon,output_dir=str(outputs[horizon]),
                                extended_status=report['extended']['status'],metrics=report))
            if error:raise RuntimeError(f'extended evaluation failed; results retained at {base}:\n{error[-3000:]}')
            return rows
