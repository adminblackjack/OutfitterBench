"""Independent sampled-day branches, launched under one shared CPU budget."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import hashlib
import subprocess
import sys
import time

from .common import write_json
from .execution import CpuLease, read_state, write_state, worker_environment


class BranchDispatcher:
    def __init__(self, run_dir, cache, budget_dir, cpu_budget, source_root, lease):
        self.run_dir=str(run_dir); self.cache=Path(cache)
        self.budget_dir=budget_dir; self.cpu_budget=cpu_budget
        self.source_root=source_root; self.lease=lease
        self.pool=ThreadPoolExecutor(max_workers=cpu_budget)
        self.futures={}

    def submit(self, day, state, expected):
        target=self.cache/'branches'/f'day-{day}'
        target.mkdir(parents=True,exist_ok=True)
        snapshot=target/'input.json.gz'
        write_state(snapshot,dict(state=state.to_dict(),expected=expected))
        self.enqueue(day)

    def enqueue(self, day):
        if day not in self.futures:
            self.futures[day]=self.pool.submit(self._run,day)

    def _run(self, day):
        target=self.cache/'branches'/f'day-{day}'
        result=target/'result.json'
        fingerprint=hashlib.sha256((target/'input.json.gz').read_bytes()).hexdigest()
        receipt=target/'result-input.json'
        if result.is_file() and receipt.is_file() and json.loads(receipt.read_text())['sha256']==fingerprint:
            return json.loads(result.read_text())
        request=target/'request.json'
        write_json(request,dict(task='relax',run_dir=self.run_dir,cache_dir=str(self.cache),
                               snapshot=str(target/'input.json.gz'),output=str(result)))
        cmd=[sys.executable,str(Path(__file__).with_name('worker.py')),
             '--evaluation-root',str(Path(__file__).parent),'--request',str(request)]
        if self.source_root:cmd+=['--source-root',str(self.source_root)]
        with CpuLease(self.budget_dir,self.cpu_budget):
            start=time.monotonic()
            with (target/'worker.log').open('w') as log:
                child=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=worker_environment())
            if child.returncode:
                raise RuntimeError(f'branch day {day} failed: {(target/"worker.log").read_text()[-4000:]}')
            write_json(target/'timing.json',dict(seconds=time.monotonic()-start))
            write_json(receipt,dict(sha256=fingerprint))
        return json.loads(result.read_text())

    def collect(self, through_day):
        # Free the main slot while waiting: budget=1 and five-main/five-slot cases
        # must make progress even if all main replayers reach a cutoff together.
        with self.lease.suspended():
            return [self.futures[d].result() for d in sorted(self.futures) if d<=through_day]

    def stats(self):
        return dict(submitted=len(self.futures),completed=sum(f.done() for f in self.futures.values()),
                    pending=sum(not f.done() for f in self.futures.values()))

    def close(self):
        with self.lease.suspended():
            self.pool.shutdown(wait=True,cancel_futures=True)


def run_branch(run_dir,cache_dir,snapshot,output):
    from ..models import WorldState
    from ..runner import _implementation_identity
    from ..config import OutfitterConfig
    from ..data import OutfitterDataset
    from ..environment import OutfitterEnvironment
    from .extended import Recorder, new_rentals, demand_receipt
    from .projection import public_problem
    import copy
    path=Path(run_dir)
    manifest=json.loads((path/'manifest.json').read_text())
    if manifest.get('format')!='outfitter-run-v1':raise ValueError('invalid branch run manifest')
    config=OutfitterConfig.from_dict(dict(manifest['config'],
        data_dir=str(path/manifest.get('runtime_data','data'))))
    env=OutfitterEnvironment(config,OutfitterDataset.load(config.data_dir))
    if manifest.get('controller',{}).get('implementation_identity')!=_implementation_identity():
        raise RuntimeError('branch frozen source identity mismatch')
    data=read_state(snapshot)
    baseline=WorldState.from_dict(data['state'])
    old_ids={c.contract_id for c in baseline.contracts}
    actual=copy.deepcopy(baseline)
    env._simulate_day(actual)
    contracts=new_rentals(actual,old_ids,baseline.day)
    if demand_receipt(actual,contracts)!=data['expected']:
        raise RuntimeError(f'persisted no-op branch differs at day {baseline.day}')
    recorder=Recorder(env,cache_dir,[])
    recorder.relax(baseline,actual,contracts,old_ids,env._simulate_day,public_problem(env,baseline))
    write_json(output,recorder.rows['capacity_relaxation'][0])
