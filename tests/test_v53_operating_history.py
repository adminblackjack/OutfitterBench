"""Engineering acceptance only: no LLM calls or profitability calibration."""
import copy
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from outfitter import operating_history as history
from outfitter.build_operating_history_runtime import build, PRICES
from outfitter.config import OutfitterConfig
from outfitter.data import OutfitterDataset
from outfitter.environment import OutfitterEnvironment
from outfitter.llm import ScriptedProvider
from outfitter.models import WorldState
from outfitter.prompts import system_prompt
from outfitter.runner import OutfitterRunner
from outfitter.scripted_merchant import PublicMerchant
from outfitter.tool_executor import ToolExecutor
from outfitter.visibility import assert_agent_visible

from release_fixtures import v52_runtime

ROOT = Path(__file__).resolve().parents[1]
SOURCE = v52_runtime()


def environment(track='main', old=False):
    config = OutfitterConfig.from_yaml(ROOT / f'configs/outfitter_v54_{track}.example.yaml')
    if old:
        config = replace(config, data_dir=str(SOURCE))
    return config, OutfitterEnvironment(config, OutfitterDataset.load(config.data_dir))


def query(env, state, tool, args):
    result = ToolExecutor(env).execute(state, tool, args)
    assert result.ok, (tool, args, result.data)
    return result.data


def frozen_interaction(env, state, *, identifier='walk-1', offer='offer-064', quantity=2,
                       accepted=False, duration=4, day=4, price=8.0, rental=True):
    state.prices[offer] = price
    if rental:
        state.deposits[offer] = 20.0
    session = SimpleNamespace(duration_days=duration if rental else None)
    raw = {'offer_id': offer, 'asset_units': quantity if rental else 0,
           'offered_units': quantity, 'quoted_unit_price': price,
           'revenue': (price + (duration - 1) * price / 2) * quantity if accepted and rental
                      else price * quantity if accepted else 0.0,
           'contract_ids': ['rental-000001'] if accepted and rental else []}
    line = history.freeze_quote(env, state, session, raw)
    return {'day': day, 'interaction_id': identifier, 'day_type': 'weekday', 'weather': 'sunny',
            'visible_party_context': 'two adults', 'public_request': {'requirements': {'party_adults': 2}},
            'request': {'service_key': env.service_key_for_product_type(env.offers[offer].service_line),
                        'customer_segment': 'MUST_NOT_LEAK'},
            'quote_lines': [line], 'response': {'accepted_units': quantity if accepted else 0,
                                              'unfulfilled_units': 0 if accepted else quantity,
                                              'disposition': 'fulfilled' if accepted else 'declined_after_quote'}}


def with_history(env):
    state = env.reset()
    state.day = 60
    state.daily_history = [{'day': day, 'counter_interactions': []} for day in range(1, 60)]
    return state


def test_build_price_facts_and_unchanged_source(tmp_path):
    cfg, _ = environment(old=True)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.iterdir() if p.is_file()}
    output = tmp_path / 'runtime'
    build(SOURCE, output, cfg)
    old, new = OutfitterDataset.load(SOURCE), OutfitterDataset.load(output)
    prior = {o.offer_id: o for o in old.offers}
    for offer in new.offers:
        if offer.offer_id in PRICES:
            assert offer.unit_cost == offer.consumer_online_offer['unit_price_cny'] == PRICES[offer.offer_id]
            assert offer.package_price == offer.consumer_online_offer['package_price_cny'] == offer.unit_cost * offer.package_units
            assert offer.consumer_online_offer['provenance'] == 'benchmark_product_fact_revision_v53'
            assert 'verified_at' not in offer.consumer_online_offer
            assert offer.reference_daily_rate == round(offer.unit_cost / max(1., offer.service_life_limit_days * .35) * 1.35, 4)
        else:
            assert offer == prior[offer.offer_id]
    assert old.offer_rng_identities == new.offer_rng_identities
    assert old.initial_assets == new.initial_assets
    assert old.warehouse_capacity_weights == new.warehouse_capacity_weights
    for profile in new.reservation_promise_catalog.profiles:
        original = old.reservation_promise_catalog.by_id[profile.profile_id]
        assert profile.member_offer_ids == original.member_offer_ids
        if not set(profile.member_offer_ids) & PRICES.keys():
            assert profile == original
    changed = {'catalog.jsonl', 'reservation_promise_profiles.json', 'warehouse_capacity_weights.json',
               'metadata.json', 'result.json', 'agent_visible_product_data.csv', 'sales_model_product_data.csv'}
    for path in SOURCE.iterdir():
        if path.is_file():
            assert hashes[path.name] == hashlib.sha256(path.read_bytes()).hexdigest()
            if path.name not in changed:
                assert path.read_bytes() == (output / path.name).read_bytes()
    with (output / 'agent_visible_product_data.csv').open() as stream:
        rows = {row['offer_id']: row for row in csv.DictReader(stream)}
    for key, price in PRICES.items():
        assert float(rows[key]['supplier_unit_cost']) == price
        assert json.loads(rows[key]['holding_costs_json'])['cost_basis_cny'] == price
    with pytest.raises(ValueError, match='new independent'):
        build(SOURCE, output, cfg)


@pytest.mark.parametrize('track', ['main', 'secondary'])
def test_holding_disclosure_and_manual(track):
    _, env = environment(track)
    state = env.reset()
    asset = state.assets[0]
    actual = env._lifecycle_holding_terms(asset, stored_on_site=True)
    holding = history.holding_costs(env, asset.offer_id, asset)
    assert holding['capital_charge'] == actual['capital_charge']
    assert holding['on_site_storage_charge'] == actual['space_charge']
    assert holding['total_when_on_site'] == actual['total_charge']
    disclosed_capital = sum(history.holding_costs(env, a.offer_id, a)['capital_charge'] for a in state.assets)
    assert round(disclosed_capital, 6) == state.current_day_ledger['opening_rental_capital_holding_charge']
    owned = query(env, state, 'view_rental_assets', {'view': 'asset', 'offer_id': asset.offer_id})['assets'][0]
    assert owned['holding_costs'] == holding
    assert 'not next-rental' in owned['wear_description']
    trial = copy.deepcopy(asset)
    trial.ownership_type = 'supplier_trial'
    assert history.holding_costs(env, trial.offer_id, trial)['capital_charge'] == 0
    assert history.holding_costs(env, trial.offer_id, trial)['on_site_storage_charge'] == actual['space_charge']
    offer = query(env, state, 'inspect_offer', {'offer_id': 'offer-021'})['offer']
    assert offer['holding_costs']['cost_basis_cny'] == 500
    assert 'service_life_limit_cycles' in offer['lifecycle_profile']
    assert 'resale_quote' not in offer and 'new_asset_resale_value' not in offer
    manual = env.business_manual()
    assert history.HOLDING_RULE in manual['finance']['rules']
    assert history.WEAR_RULE in manual['assets']['rules']
    assert history.HISTORY_RULE in manual['workspace']['rules']
    assert not any('two most recently completed' in rule for rule in manual['workspace']['rules'])
    assert '55%' in json.dumps(manual)
    bad = ToolExecutor(env).execute(state, 'view_liquidation_quotes', {'offer_ids': ['offer-021']})
    assert not bad.ok


def test_entire_season_frozen_quotes_and_privacy():
    _, env = environment()
    state = with_history(env)
    raw = frozen_interaction(env, state)
    state.daily_history[3]['counter_interactions'] = [raw]
    state.recent_counter_interactions = []
    state.prices['offer-064'] = 99
    state.deposits['offer-064'] = 999
    before = copy.deepcopy(state.to_dict())
    result = query(env, state, 'counter_log', {'day': 4})
    quote = result['interactions'][0]['quotes'][0]
    assert quote['quoted_unit_price'] == 8
    assert quote['physical_quantity'] == 2
    assert quote['quoted_total_rent'] == 40
    assert quote['quoted_total_deposit'] == 40
    assert quote['accepted_physical_quantity'] == 0
    assert 'MUST_NOT_LEAK' not in json.dumps(result)
    assert_agent_visible(result)
    assert state.to_dict() == before
    restored = WorldState.from_dict(state.to_dict())
    assert query(env, restored, 'counter_log', {'day': 4}) == result
    assert query(env, state, 'counter_log', {})['period'] == {'from_day': 53, 'to_day': 59, 'last_completed_day': 59}
    assert query(env, state, 'counter_log', {'day': 5})['available'] is True
    assert query(env, state, 'counter_log', {'day': 5})['record_count'] == 0
    assert query(env, state, 'counter_log', {'day': 60})['available'] is False


def test_first_day_and_invalid_channel_are_unambiguous():
    _, env = environment()
    state = env.reset()
    before = copy.deepcopy(state.to_dict())
    result = query(env, state, 'counter_log', {})
    assert not result['available'] and result['unavailable_reason'] == 'no_completed_day_available'
    assert result['period']['from_day'] is result['period']['to_day'] is None
    assert not ToolExecutor(env).execute(state, 'service_history', {'channel': 'unknown'}).ok
    assert state.to_dict() == before


def test_price_summary_filtering_pagination_and_denominators():
    _, env = environment()
    state = with_history(env)
    rows = [frozen_interaction(env, state, identifier=f'walk-{i}', price=8 if i < 2 else 10, accepted=i == 0)
            for i in range(3)]
    noquote = copy.deepcopy(rows[0])
    noquote.update(interaction_id='noquote', quote_lines=[])
    state.daily_history[3]['counter_interactions'] = rows + [noquote]
    args = {'from_day': 1, 'to_day': 59, 'offer_ids': ['offer-064']}
    summary = query(env, state, 'counter_log', {'view': 'price_summary', **args, 'limit': 1})
    assert summary['record_count'] == 2
    assert summary['totals']['quote_count'] == 3
    assert summary['totals']['accepted_quote_count'] == 1
    assert summary['totals']['interaction_count'] == 3
    assert summary['totals']['quoted_physical_units'] == 6
    assert summary['totals']['initial_rental_revenue'] == 40
    second = query(env, state, 'counter_log', {'view': 'price_summary', **args, 'cursor': summary['next_cursor'], 'limit': 1})
    assert second['totals'] == summary['totals']
    assert second['price_groups'] != summary['price_groups']
    assert second['has_more'] is False
    all_records = query(env, state, 'counter_log', {'from_day': 1, 'to_day': 59, 'service_keys': ['rental:tent']})
    assert all_records['interaction_count'] == 4
    assert all_records['totals']['quote_count'] == 3
    assert sum(q['initial_rental_revenue'] for r in all_records['interactions'] for q in r['quotes']) == 40
    unseen = query(env, state, 'counter_log', {**args, 'offer_ids': ['offer-021'], 'view': 'price_summary'})
    assert unseen['price_groups'] == []


@pytest.mark.parametrize('args', [
    {'day': 4, 'from_day': 2}, {'day': 4, 'to_day': 5}, {'from_day': 9, 'to_day': 3},
    {'view': 'optimal_price'}, {'from_day': 0}, {'offer_ids': ['missing']},
    {'service_keys': ['hidden_segment']}, {'limit': 51}, {'cursor': -1},
])
def test_counter_rejects_invalid_queries_without_mutation(args):
    _, env = environment()
    state = with_history(env)
    before = copy.deepcopy(state.to_dict())
    assert not ToolExecutor(env).execute(state, 'counter_log', args).ok
    assert state.to_dict() == before


def test_retail_summary_has_unit_prices_and_no_rent():
    _, env = environment('secondary')
    state = with_history(env)
    offer_id = next(o.offer_id for o in env.offers.values() if o.asset_kind != 'durable')
    state.daily_history[3]['counter_interactions'] = [frozen_interaction(env, state, offer=offer_id, rental=False, accepted=True)]
    group = query(env, state, 'counter_log', {'day': 4, 'view': 'price_summary'})['price_groups'][0]
    assert group['duration_days'] is None
    assert group['pricing_basis'] == 'retail_per_unit'
    assert group['initial_rental_revenue'] == 0
    assert group['retail_revenue'] == 16


def test_quote_quantity_is_not_party_size_and_same_quote_is_not_counted_twice():
    _, env = environment()
    state = with_history(env)
    row = frozen_interaction(env, state, quantity=1, accepted=True)
    row['public_request']['requirements']['party_adults'] = 6
    row['quote_lines'].append(copy.deepcopy(row['quote_lines'][0]))
    state.daily_history[3]['counter_interactions'] = [row]
    result = query(env, state, 'counter_log', {'day': 4, 'view': 'price_summary'})
    assert result['totals']['interaction_count'] == result['totals']['quote_count'] == 1
    assert result['totals']['quoted_physical_units'] == 2
    assert result['totals']['accepted_physical_units'] == 2
    assert result['totals']['initial_rental_revenue'] == 40


def test_offer_filter_keeps_complete_interaction_but_summarizes_matching_offer():
    _, env = environment()
    state = with_history(env)
    row = frozen_interaction(env, state, offer='offer-064')
    other = frozen_interaction(env, state, offer='offer-021', accepted=True)
    row['quote_lines'].extend(other['quote_lines'])
    state.daily_history[3]['counter_interactions'] = [row]
    result = query(env, state, 'counter_log', {'day': 4, 'offer_ids': ['offer-064']})
    assert len(result['interactions'][0]['quotes']) == 2
    assert result['totals']['quote_count'] == 1
    assert result['totals']['accepted_quote_count'] == 0


def test_delivery_channels_continuations_and_cash_unchanged():
    _, env = environment()
    state = env.reset()
    before_cash = state.cash
    for channel, quantity, revenue in [('walk_in', 2, 30), ('reservation', 3, 50)]:
        env._record_attempt(state, 'rental:tent', 'fulfilled', channel=channel,
            product_type='tent', offer_id='offer-064', revenue=revenue, quantity=quantity, rental_days=quantity * 2)
    # Reporting must not collect money a second time.
    assert state.cash == before_cash
    contract = SimpleNamespace(service_line='tent', offer_id='offer-064', reservation_id='reservation-001')
    env._v47_record_continuation_service(state, contract, revenue=7, rental_days=1)
    state.daily_history = [{'day': 1, 'delivery_metrics': copy.deepcopy(state.current_day_ledger['delivery_metrics'])}]
    state.day = 2
    totals = {channel: query(env, state, 'service_history', {'channel': channel})['totals']
              for channel in ('all', 'walk_in', 'reservation')}
    for key in history.SERVICE_FIELDS:
        assert totals['all'].get(key, 0) == pytest.approx(totals['walk_in'].get(key, 0) + totals['reservation'].get(key, 0))
    assert totals['all']['revenue'] == 87
    assert totals['all']['initial_rental_revenue'] == 80
    assert totals['reservation']['continuation_rental_revenue'] == 7
    assert totals['all']['fulfilled_units'] == 5
    offer = query(env, state, 'service_history', {'granularity': 'offer', 'offer_ids': ['offer-064']})
    assert offer['totals'] == totals['all']
    assert offer['offers'][0]['revenue'] == 87
    assert all(key not in offer['offers'][0] for key in ('quote_exposures', 'average_posted_price', 'price_regimes'))


def test_v52_remains_legacy_and_cross_version_state_rejected():
    _, old = environment(old=True)
    _, new = environment()
    state = old.reset()
    before = copy.deepcopy(state.to_dict())
    with pytest.raises(ValueError, match='version mismatch'):
        new.close_business_day(state)
    assert state.to_dict() == before
    assert 'holding_costs' not in old.public_offer('offer-021', detailed=True)
    definitions = {r['function']['name']: r['function'] for r in ToolExecutor(old).definitions()}
    assert 'view' not in definitions['counter_log']['parameters']['properties']
    assert 'channel' not in definitions['service_history']['parameters']['properties']
    newdefs = {r['function']['name']: r['function'] for r in ToolExecutor(new).definitions()}
    assert 'view' in newdefs['counter_log']['parameters']['properties']
    assert 'channel' in newdefs['service_history']['parameters']['properties']


@pytest.mark.parametrize('track', ['main', 'secondary'])
def test_short_public_tool_integration_and_reporting_reconcile(track):
    cfg, env = environment(track)
    state = env.reset()
    tools = ToolExecutor(env)
    def rpc(name, args):
        result = tools.execute(state, name, args)
        assert result.ok, (name, result.data)
        return {'ok': result.ok, 'data': result.data, 'terminal_day': result.terminal_day}
    merchant = PublicMerchant('calendar', rpc, retail_enabled=cfg.retail_enabled)
    merchant.start()
    for _ in range(7):
        merchant.day()
        rpc('close_business_day', {})
    assert state.day == 8
    for report in state.daily_history:
        recorded = sum(row['revenue'] for row in report.get('delivery_metrics', {}).values())
        original = sum(row['revenue'] for row in report['service_metrics'].values())
        assert recorded == pytest.approx(original, abs=.000001)
        for interaction in report['counter_interactions']:
            for line in interaction['quote_lines']:
                if line['pricing_basis'] == 'first_day_per_asset' and line['accepted_physical_quantity']:
                    assert len(line['contract_ids']) == line['accepted_physical_quantity']
    rows = rpc('counter_log', {'view': 'price_summary'})['data']
    expected = sum(q.get('revenue', 0) for report in state.daily_history for r in report['counter_interactions'] for q in r['quote_lines'])
    assert rows['totals']['initial_rental_revenue'] + rows['totals'].get('retail_revenue', 0) == pytest.approx(expected)
    totals = {channel: rpc('service_history', {'channel': channel})['data']['totals'] for channel in ('all', 'walk_in', 'reservation')}
    assert totals['walk_in']['initial_rental_revenue'] > 0
    assert totals['reservation']['initial_rental_revenue'] > 0
    for key in history.SERVICE_FIELDS:
        assert totals['all'].get(key, 0) == pytest.approx(totals['walk_in'].get(key, 0) + totals['reservation'].get(key, 0))


def test_runner_resume_without_api_and_stable_prompt(tmp_path):
    cfg, env = environment()
    cfg = replace(cfg, runs_dir=str(tmp_path))
    close = {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'close', 'type': 'function',
        'function': {'name': 'close_business_day', 'arguments': '{}'}}]}
    provider = ScriptedProvider([copy.deepcopy(close), copy.deepcopy(close)])
    runner = OutfitterRunner.create(config=cfg, provider=provider, run_id='v53-engineering')
    assert runner.run(stop_after_day=1)['day'] == 2
    resumed = OutfitterRunner.resume(run_dir=runner.store.run_dir, provider=provider)
    assert resumed.run(stop_after_day=2)['day'] == 3
    prompt = (runner.store.run_dir / 'system_prompt.md').read_text()
    assert '55% nominal recovery baseline' in prompt
    for full_rule in (history.HOLDING_RULE, history.WEAR_RULE, history.HISTORY_RULE):
        assert full_rule not in prompt
    assert '55%' in prompt
    assert json.loads((runner.store.run_dir / 'business_rules.json').read_text()) == env.business_manual()


@pytest.mark.parametrize('track', ['main', 'secondary'])
def test_daily_prompt_is_layered_and_details_remain_queryable(track):
    cfg, env = environment(track)
    tools = ToolExecutor(env)
    terms = env.public_operating_rules()
    kwargs = dict(tool_call_protocol='read_only_batch_v1', read_only_tools=tools.read_only_names(),
                  max_read_only_batch_size=cfg.max_read_only_batch_size)
    prompt = system_prompt(terms, **kwargs)
    assert len(prompt) <= 5000
    assert 'Every business day starts a fresh model session' in prompt
    assert 'A quotation may create a binding commitment immediately or at close' in prompt
    assert 'normal_return_cleaning' not in prompt
    for full_rule in (history.HOLDING_RULE, history.WEAR_RULE, history.HISTORY_RULE):
        assert full_rule not in prompt
    manual = env.business_manual()
    assert history.HOLDING_RULE in manual['finance']['rules']
    assert history.WEAR_RULE in manual['assets']['rules']
    assert history.HISTORY_RULE in manual['workspace']['rules']
    assert '55%' in prompt
