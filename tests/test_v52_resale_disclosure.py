"""Information-only release: disclosure agrees with existing valuation physics."""
import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from outfitter.actor_manual import public_terms
from outfitter.build_resale_disclosure_runtime import build
from outfitter.config import OutfitterConfig
from outfitter.data import OutfitterDataset
from outfitter.environment import OutfitterEnvironment
from outfitter.llm import ScriptedProvider
from outfitter.prompts import system_prompt
from outfitter.resale_disclosure import ENVIRONMENT_PROTOCOL, CORE_RELEASE_ID, POLICY_ID, public_rule
from outfitter.runner import OutfitterRunner
from outfitter.scripted_merchant import PublicMerchant
from outfitter.tool_executor import ToolExecutor
from release_fixtures import v51_runtime, v52_runtime

ROOT=Path(__file__).resolve().parents[1]


def environment(track='main', old=False):
    cfg=OutfitterConfig.from_yaml(ROOT/f'configs/outfitter_v54_{track}.example.yaml')
    cfg=replace(cfg,data_dir=str(v51_runtime() if old else v52_runtime()))
    return cfg,OutfitterEnvironment(cfg,OutfitterDataset.load(cfg.data_dir))


@pytest.mark.parametrize('track',['main','secondary'])
def test_formula_in_daily_prompt_and_relevant_manual_chapters(track):
    cfg,env=environment(track);tools=ToolExecutor(env);terms=public_terms(env)
    rule=terms['durable_resale_formula']
    assert '55%' in rule and 'P * 0.55 * c * max(0, 1 - w) ** 0.65' in rule
    assert 'condition_ratio' in rule and 'combined_wear' in rule
    for forbidden in ('service_life_limit_cycles','used_rental_cycles','used_life_days /','service_life_limit_days'):
        assert forbidden not in rule
    for chapter in ('assets','procurement','scoring'):
        assert rule in env.business_manual()[chapter]['rules']
    prompt=system_prompt(terms,tool_call_protocol='read_only_batch_v1',
        read_only_tools=tools.read_only_names(),max_read_only_batch_size=5)
    assert rule in prompt
    _,old=environment(track,old=True)
    assert 'durable_resale_formula' not in public_terms(old)


@pytest.mark.parametrize('condition,wear',[(1,0),(.8,0),(1,.5),(.7,.7),(1,1),(1,1.2),(0,.1)])
def test_disclosed_formula_matches_existing_dealer_valuation(condition,wear):
    _,env=environment();state=env.reset();asset=copy.deepcopy(state.assets[0])
    asset.unit_cost=100.;asset.condition_ratio=condition;asset.status='available'
    asset.service_life_limit_days=100;asset.service_life_limit_cycles=100
    asset.used_life_days=round(wear*100);asset.used_rental_cycles=0
    expected=round(round(100*.55*condition*max(0.,1-wear)**.65,6),2)
    assert env.secondhand_quote(asset)['offer_price']==expected
    if condition==1 and wear==0:assert expected==55.


def test_disclosure_uses_loaded_scales_not_duplicate_constants():
    _,env=environment()
    env.lifecycle_scales=replace(env.lifecycle_scales,new_asset_residual_rate=.6,residual_wear_exponent=.7)
    rule=public_terms(env)['durable_resale_formula']
    assert '60%' in rule and 'P * 0.6 * c * max(0, 1 - w) ** 0.7' in rule


def test_product_and_dealer_interfaces_unchanged_and_life_fields_retained():
    _,new=environment();_,old=environment(old=True)
    ns,os=new.reset(),old.reset();nt,ot=ToolExecutor(new),ToolExecutor(old)
    assert nt.definitions()==ot.definitions()
    a=nt.execute(ns,'inspect_offer',{'offer_id':'offer-059'})
    b=ot.execute(os,'inspect_offer',{'offer_id':'offer-059'})
    assert a.data==b.data
    assert a.data['offer']['lifecycle_profile']['service_life_limit_cycles']==100
    assert a.data['offer']['lifecycle_profile']['service_life_limit_days']==500
    before=ns.to_dict()
    bad=nt.execute(ns,'view_liquidation_quotes',{'offer_ids':['offer-059']})
    assert not bad.ok and ns.to_dict()==before
    args={'rental_asset_ids':[ns.assets[0].asset_id]}
    assert nt.execute(ns,'view_liquidation_quotes',args).data==ot.execute(os,'view_liquidation_quotes',args).data
    assert new.config.liquidation_depth_enabled is False


def test_runtime_builder_only_changes_metadata_and_rejects_overwrite(tmp_path):
    target=tmp_path/'v52';build(v51_runtime(),target)
    for old in v51_runtime().iterdir():
        if old.is_file() and old.name!='metadata.json':assert old.read_bytes()==(target/old.name).read_bytes()
    meta=json.loads((target/'metadata.json').read_text())
    assert meta['core_release_id']==CORE_RELEASE_ID and meta['environment_protocol']==ENVIRONMENT_PROTOCOL
    assert meta['resale_formula_disclosure_policy']==POLICY_ID
    with pytest.raises(ValueError,match='new independent'):build(v51_runtime(),target)


def test_old_state_rejected_and_new_run_resume_preserves_handbook(tmp_path):
    cfg,new=environment();_,old=environment(old=True);oldstate=old.reset();before=oldstate.to_dict()
    with pytest.raises(ValueError,match='version mismatch'):new.close_business_day(oldstate)
    assert oldstate.to_dict()==before
    cfg=replace(cfg,runs_dir=str(tmp_path))
    close={'role':'assistant','content':'','tool_calls':[{'id':'close','type':'function',
        'function':{'name':'close_business_day','arguments':'{}'}}]}
    p=ScriptedProvider([copy.deepcopy(close),copy.deepcopy(close)])
    runner=OutfitterRunner.create(config=cfg,provider=p,run_id='v52-resume')
    assert runner.run(stop_after_day=1)['day']==2
    resumed=OutfitterRunner.resume(run_dir=runner.store.run_dir,provider=p)
    assert resumed.run(stop_after_day=2)['day']==3
    assert '55%' in (runner.store.run_dir/'system_prompt.md').read_text()
    assert json.loads((runner.store.run_dir/'business_rules.json').read_text())==new.business_manual()


@pytest.mark.parametrize('track',['main','secondary'])
def test_seven_day_paired_public_policy_leaves_economics_and_random_world_unchanged(track):
    cfg,new=environment(track);_,old=environment(track,old=True)
    ns,os=new.reset(),old.reset();nt,ot=ToolExecutor(new),ToolExecutor(old)
    calls=0
    def rpc(name,args):
        nonlocal calls
        calls+=1
        left=ot.execute(os,name,copy.deepcopy(args));right=nt.execute(ns,name,copy.deepcopy(args))
        assert left.ok and right.ok,(name,left.data,right.data)
        assert left.data==right.data,name
        return {'ok':right.ok,'data':copy.deepcopy(right.data),'terminal_day':right.terminal_day}
    merchant=PublicMerchant('calendar',rpc,retail_enabled=cfg.retail_enabled)
    merchant.start()
    for day in range(1,8):
        assert ns.day==os.day==day
        merchant.day();rpc('close_business_day',{})
        left,right=os.to_dict(),ns.to_dict()
        left.pop('calendar_demand_protocol');right.pop('calendar_demand_protocol')
        assert left==right
        assert new.business_value(ns)==old.business_value(os)
    assert ns.day==8 and not ns.done and calls>100
    assert cfg.max_days==200 and cfg.runoff_days==14
