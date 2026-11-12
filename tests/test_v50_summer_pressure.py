"""V50 must implement the experimental D2 world, not a second approximation."""
import copy
from dataclasses import replace
from datetime import date
import hashlib
import json
from pathlib import Path

import pytest

from outfitter.build_summer_pressure_runtime import build
from outfitter.config import OutfitterConfig
from outfitter.data import OutfitterDataset
from outfitter.environment import OutfitterEnvironment
from outfitter.known_model import KnowledgeProvider
from outfitter.models import WorldState
from outfitter.runtime.common import _ARCHETYPE_INDEX
from outfitter.runtime.ledger import empty_ledger
from outfitter.school_holiday import calendar_with_school
from outfitter.summer_pressure import POLICY_ID, arrival_multiplier
from outfitter.tool_executor import ToolExecutor
from outfitter.v5 import calendar_classification
from release_fixtures import v49_runtime, v48_runtime, v50_runtime

ROOT=Path(__file__).resolve().parents[1]
MAIN=ROOT/'tests/fixtures/outfitter_v50_legacy_main.yaml'


def env(version=50, **kwargs):
    c=OutfitterConfig.from_yaml(MAIN)
    c=replace(c,data_dir=str(v49_runtime() if version==49 else v50_runtime()))
    c=replace(c,**kwargs)
    return OutfitterEnvironment(c,OutfitterDataset.load(c.data_dir))


def experimental_d2(calendar,kind):
    # Independent expression of the preregistered experimental override.
    if kind=='supply' or not calendar.get('is_school_holiday'):return 1.
    if calendar['is_weekend'] or calendar['day_type']=='public_holiday':return 1.3
    return 2. if calendar['weekday']=='Friday' else 3.


@pytest.mark.parametrize('text,value',[('2026-06-30',1.),('2026-07-01',3.),
    ('2026-07-03',2.),('2026-07-04',1.3),('2026-08-30',1.3),
    ('2026-08-31',3.),('2026-09-01',1.),('2026-10-01',1.)])
def test_factor_boundaries(text,value):
    calendar=calendar_with_school(calendar_classification(date.fromisoformat(text)))
    for kind in ('overnight','picnic','lightweight'):
        assert arrival_multiplier(calendar,kind)==value
    assert arrival_multiplier(calendar,'supply')==1.


def test_public_holiday_overrides_weekday_within_school():
    calendar={'is_school_holiday':True,'is_weekend':False,'day_type':'public_holiday','weekday':'Monday'}
    assert arrival_multiplier(calendar,'overnight')==1.3


@pytest.mark.parametrize('seed',[42,73,101])
def test_full_season_rates_and_pool_counts_match_experiment(seed,monkeypatch):
    old,new=env(49,seed=seed),env(seed=seed)
    base=old._v47_arrival_rate
    monkeypatch.setattr(old,'_v47_arrival_rate',lambda day,kind:base(day,kind)*experimental_d2(old._calendar(day),kind))
    for day in range(1,201):
        for kind in ('overnight','picnic','lightweight'):
            assert new._v47_arrival_rate(day,kind)==old._v47_arrival_rate(day,kind)
            assert new._potential_arrival_count(day,kind)==old._potential_arrival_count(day,kind)


@pytest.mark.parametrize('day',[78,90,110,140,153])
def test_actual_requests_match_experimental_override(day,monkeypatch):
    old,new=env(49),env()
    base=old._v47_arrival_rate
    monkeypatch.setattr(old,'_v47_arrival_rate',lambda d,k:base(d,k)*experimental_d2(old._calendar(d),k))
    outputs=[]
    for e in (old,new):
        s=e.reset();s.day=day;s.current_day_ledger=empty_ledger(day)
        s.reservation_requests=[];s.reservation_events=[];s.inbox=[]
        e._generate_reservation_requests(s)
        outputs.append(s.to_dict()['reservation_requests'])
    assert outputs[0] and outputs[0]==outputs[1]


def test_main_secondary_same_rentals_and_retail_supply_unchanged():
    a,b=env(),env(assortment_mode='rental_and_retail')
    old=env(49,assortment_mode='rental_and_retail')
    for day in (1,91,92,94,120,153,154,200):
        assert b._potential_arrival_count(day,'supply')==old._potential_arrival_count(day,'supply')
        for kind in ('overnight','picnic','lightweight'):
            n=a._potential_arrival_count(day,kind)
            assert n==b._potential_arrival_count(day,kind)
            for i in range(n):
                assert a._reservation_route(day,_ARCHETYPE_INDEX[kind],i)==b._reservation_route(day,_ARCHETYPE_INDEX[kind],i)
                assert a._v47_customer_shape(day,kind,i)==b._v47_customer_shape(day,kind,i)


def test_policy_state_roundtrip_and_cross_version_rejection():
    old,new=env(49),env();a,b=old.reset(),new.reset()
    assert b.calendar_demand_protocol==POLICY_ID
    assert WorldState.from_dict(b.to_dict()).to_dict()==b.to_dict()
    with pytest.raises(ValueError,match='Calendar demand version'):new.close_business_day(a)
    with pytest.raises(ValueError,match='Calendar demand version'):old.close_business_day(b)
    assert not ToolExecutor(new).execute(a,'business_status',{}).ok


def test_metadata_tamper_and_legacy_masquerade_rejected():
    e=env();data=copy.deepcopy(e.source_dataset)
    data.metadata['summer_pressure_policy']['multipliers']['school_mon_thu']=2.
    with pytest.raises(ValueError,match='frozen D2'):OutfitterEnvironment(e.config,data)
    data=copy.deepcopy(e.source_dataset)
    data.metadata['environment_protocol']=env(49).dataset.metadata['environment_protocol']
    with pytest.raises(ValueError,match='requires the V50'):OutfitterEnvironment(e.config,data)


def test_no_future_truth_in_request_generation(monkeypatch):
    e=env();base=e.reset();base.day=90;base.current_day_ledger=empty_ledger(90)
    base.reservation_requests=[];base.reservation_events=[]
    forecasts=e.weather_forecast(base,15)
    monkeypatch.setattr(OutfitterEnvironment,'weather_forecast',lambda self,s,h:
        [copy.deepcopy(r) for r in forecasts if s.day<=r['day']<s.day+h])
    results=[]
    for cold in (True,False):
        data=copy.deepcopy(e.source_dataset)
        for day,row in data.weather_calendar.items():
            if day>e._source_weather_day(base.day):
                row.update(night_low_c=-5 if cold else 25,temp_mean_c=0 if cold else 30,day_high_c=5 if cold else 35)
                data.weather[day]='snowy' if cold else 'sunny'
        changed=OutfitterEnvironment(e.config,data);s=copy.deepcopy(base)
        changed._generate_reservation_requests(s)
        results.append(s.to_dict()['reservation_requests'])
    assert results[0] and results[0]==results[1]


def test_predictor_uses_native_d2_without_weather_truth():
    e=env();k=KnowledgeProvider(e.config,samples=1)
    for day in (1,78,92,95,120,153,154,200):
        assert k._env._v47_arrival_rate(day,'overnight')==e._v47_arrival_rate(day,'overnight')
    with pytest.raises(AssertionError):k._env._weather(120)


def test_builder_preserves_payloads(tmp_path):
    source=v49_runtime();before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
    build(source,tmp_path/'runtime')
    data=OutfitterDataset.load(tmp_path/'runtime')
    assert data.metadata['summer_pressure_policy']['profile']=='D2'
    changed={p.name for p in source.iterdir() if p.is_file() and p.read_bytes()!=(tmp_path/'runtime'/p.name).read_bytes()}
    assert changed=={'metadata.json'}
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
    for name,sha in data.metadata['content_sha256'].items():
        assert hashlib.sha256((tmp_path/'runtime'/name).read_bytes()).hexdigest()==sha
    with pytest.raises(ValueError):build(source,tmp_path/'runtime')
    with pytest.raises(ValueError):build(v48_runtime(),tmp_path/'bad-source')


def test_defaults_and_optional_1500_do_not_add_cash():
    c=OutfitterConfig.from_yaml(MAIN)
    assert (c.initial_cash,c.durable_acquisition_budget,c.max_iterations_per_day,c.max_model_calls)==(3000,1000,200,40000)
    secondary=OutfitterConfig.from_yaml(ROOT/'tests/fixtures/outfitter_v50_legacy_secondary.yaml').to_dict()
    assert {k for k,v in c.to_dict().items() if v!=secondary[k]}=={'assortment_mode'}
    a,b=env(),env(durable_acquisition_budget=1500)
    sa,sb=a.reset(),b.reset()
    assert sa.cash==sb.cash==3000
    ca=a.observation(sa,consume_inbox=False)['durable_capital']
    cb=b.observation(sb,consume_inbox=False)['durable_capital']
    assert cb['durable_capital_limit']-ca['durable_capital_limit']==500


@pytest.mark.parametrize('track',['rental_only','rental_and_retail'])
def test_summer_closing(track):
    e=env(assortment_mode=track);s=e.reset();s.day=94;s.current_day_ledger=empty_ledger(94)
    s.reservation_requests=[];s.reservation_events=[];s.inbox=[];s.cumulative_reservation_requests=0
    e._capture_opening_rental_holding_basis(s);e._prepare_reservation_day(s);e._seal_opening_flows(s)
    for _ in range(3):
        result=ToolExecutor(e).execute(s,'close_business_day',{})
        assert result.ok and not s.bankrupt
    assert [r['day'] for r in s.daily_history]==[94,95,96]
