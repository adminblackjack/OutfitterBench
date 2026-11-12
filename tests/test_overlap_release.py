"""Release identities, initial capital, immutable V50 payloads and resume boundaries."""
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from outfitter.config import OutfitterConfig
from outfitter.data import OutfitterDataset
from outfitter.environment import OutfitterEnvironment
from outfitter.models import WorldState
from outfitter.reservation_overlap import ENVIRONMENT_PROTOCOL, POLICY_ID
from release_fixtures import v50_runtime

ROOT=Path(__file__).resolve().parents[1]
V51=ENVIRONMENT_PROTOCOL=='outfitter-seasonal-v51-reservation-overlap'
BRANCH='v54'


def setup(track='main'):
    c=OutfitterConfig.from_yaml(ROOT/f'configs/outfitter_{BRANCH}_{track}.example.yaml')
    e=OutfitterEnvironment(c,OutfitterDataset.load(c.data_dir))
    return c,e,e.reset()


def test_defaults_assets_capital_and_sparse_id():
    c,e,s=setup()
    assert (c.initial_cash,c.durable_acquisition_budget,c.max_iterations_per_day,c.max_model_calls)==(3000,1000,200,40000)
    assert c.daily_fee==(15 if V51 else 10)
    assert len(s.assets)==(27 if V51 else 18)
    assert set(Counter(a.offer_id for a in s.assets).values())=={3 if V51 else 2}
    assert s.next_asset_id==max(int(a.asset_id.removeprefix('asset-')) for a in s.assets)+1
    cap=e.observation(s,consume_inbox=False)['durable_capital']
    assert cap['durable_capital_limit']==pytest.approx(2141.83 if V51 else 1761.22)
    initial=Path(c.data_dir)/'initial_assets.jsonl'
    original=ROOT/'tests/fixtures/v50_initial_assets.jsonl'
    old={r['asset_id']:r for r in map(json.loads,original.read_text().splitlines())}
    for row in map(json.loads,initial.read_text().splitlines()):assert old[row['asset_id']]==row
    if V51:assert initial.read_bytes()==original.read_bytes()


def test_state_identity_and_old_state_rejected_without_mutation():
    c,e,s=setup()
    assert s.calendar_demand_protocol==POLICY_ID+'-recovery-r2-resale-disclosure-v1-operating-history-v1-current-pickup-v1'
    assert e.dataset.metadata['tool_interface_version']==52
    old=OutfitterEnvironment(replace(c,data_dir=str(v50_runtime()),daily_fee=15.),OutfitterDataset.load(v50_runtime())).reset()
    before=old.to_dict()
    with pytest.raises(ValueError,match='version mismatch'):e.close_business_day(old)
    assert old.to_dict()==before
    if V51:
        previous=WorldState.from_dict(s.to_dict());previous.calendar_demand_protocol=POLICY_ID
        before=previous.to_dict()
        with pytest.raises(ValueError,match='version mismatch'):e.close_business_day(previous)
        assert previous.to_dict()==before
        old=WorldState.from_dict(s.to_dict());old.calendar_demand_protocol='reservation-overlap-candidate-r1'
        before=old.to_dict()
        with pytest.raises(ValueError,match='version mismatch'):e.close_business_day(old)
        assert old.to_dict()==before


def test_same_branch_main_secondary_and_payload_integrity():
    c,e,s=setup();other,ee,ss=setup('secondary')
    assert {k for k,v in c.to_dict().items() if v!=other.to_dict()[k]}=={'assortment_mode'}
    for name,digest in e.dataset.metadata['content_sha256'].items():
        assert hashlib.sha256((Path(c.data_dir)/name).read_bytes()).hexdigest()==digest
    assert [(a.asset_id,a.offer_id) for a in s.assets]==[(a.asset_id,a.offer_id) for a in ss.assets]


def test_daily_fee_exact_and_asset_ids_unique():
    c,e,s=setup();e.close_business_day(s)
    assert len({a.asset_id for a in s.assets})==len(s.assets)
    assert s.daily_history[0]['financial']['operating_fee']==c.daily_fee
