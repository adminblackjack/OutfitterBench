from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from outfitter.config import OutfitterConfig
from outfitter.data import OutfitterDataset
from outfitter.environment import OutfitterEnvironment
from outfitter.models import CleaningJob
from outfitter.runner import OutfitterRunner
from outfitter.runtime.ledger import empty_ledger

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('cash,costs,expected_completed',[(.45,[.5],0),(1.2,[.7,.6],1),(1.3,[.7,.6],2)])
def test_cleaning_default_reconciles_and_removes_finished_jobs(cash,costs,expected_completed):
    config=replace(OutfitterConfig.from_yaml(ROOT/'configs/outfitter_v54_main.example.yaml'),initial_cash=cash)
    env=OutfitterEnvironment(config,OutfitterDataset.load(config.data_dir));state=env.reset()
    state.day=2;state.current_day_ledger=empty_ledger(2)
    assets=sorted(state.assets,key=lambda a:a.asset_id)[:len(costs)]
    for asset,cost in zip(assets,costs):
        asset.status='cleaning';asset.ready_day=None
        state.cleaning_queue.append(CleaningJob(asset_id=asset.asset_id,offer_id=asset.offer_id,
            product_type=asset.service_line,enqueued_day=1,duration_days=1,remaining_work_points=1.,cleaning_cost=cost))
    env._process_cleaning_capacity(state,[])
    assert state.cumulative_cleaning_cost==pytest.approx(sum(costs))
    assert state.current_day_ledger['morning']['cleaning_cost']==pytest.approx(sum(costs))
    assert [a.asset_id for a in assets if a.status=='available']==[a.asset_id for a in assets[:expected_completed]]
    assert [j.asset_id for j in state.cleaning_queue]==[a.asset_id for a in assets[expected_completed:]]
    assert state.bankrupt==(cash<sum(costs))
    assert state.bankruptcy_shortfall==pytest.approx(max(0.,sum(costs)-cash))
    facade=SimpleNamespace(config=config,environment=env)
    flow=OutfitterRunner._v28_cash_flow_summary(facade,state)
    assert flow['reconciliation_difference']==0
    if state.done:
        before=(state.cash,state.cumulative_cleaning_cost,state.bankruptcy_shortfall)
        env._process_cleaning_capacity(state,[])
        assert before==(state.cash,state.cumulative_cleaning_cost,state.bankruptcy_shortfall)
