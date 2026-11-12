"""Information placement, preserved contracts and version isolation."""
import copy
from dataclasses import replace
import json

import pytest

from outfitter.actor_contract_v53 import PARTIAL_PICKUP_RULE, resale_rule
from outfitter.actor_manual import public_terms
from outfitter.data import OutfitterDataset
from outfitter.environment import OutfitterEnvironment
from outfitter.prompts import system_prompt
from outfitter.school_holiday import PUBLIC_RULES as SCHOOL_RULES
from outfitter.service_guarantee import RULE as GUARANTEE_RULE
from outfitter.tool_executor import ToolExecutor
from test_v53_operating_history import environment


def prompt(env, batch=True):
    tools = ToolExecutor(env)
    return system_prompt(env.public_operating_rules(),
        tool_call_protocol='read_only_batch_v1' if batch else 'first_only',
        read_only_tools=tools.read_only_names(), max_read_only_batch_size=env.config.max_read_only_batch_size)


@pytest.mark.parametrize('track', ['main', 'secondary'])
def test_six_core_groups_and_no_workflow_tutorials(track):
    _, env = environment(track)
    text = prompt(env)
    assert len(text) <= 5000
    for required in ('maximize its official final value', 'Day 200', 'Day 214',
                     '55% nominal recovery baseline', 'liabilities, not profit',
                     'not additional cash', 'before walk-ins', 'without approval',
                     'cannot be rented again on its return day', 'immediately or at close',
                     'including a selected optional line', 'without a posted price',
                     'fresh model session', 'not automatically loaded', 'only on the next day',
                     '200 model responses', 'including the next response', 'automatic close',
                     'at most 5', 'Invalid batches execute nothing', 'read_business_rules'):
        assert required in text, required
    for removed in ('Batchable tools:', 'inquire_partial_fulfillment',
                    'request_reservation_contingency', 'max(0, 1 - w)', '0.75',
                    'is_school_holiday', 'school holidays', 'storage_burden_units',
                    'period=season_to_date', 'remaining_combined_wear'):
        assert removed not in text, removed


@pytest.mark.parametrize('batch', [True, False])
def test_configuration_is_rendered_not_hardcoded(batch):
    cfg, _ = environment()
    cfg = replace(cfg, max_days=70, runoff_days=10, max_iterations_per_day=71,
                  max_read_only_batch_size=3, parallel_tool_calls=batch, charge_model_usage_to_cash=True)
    env = OutfitterEnvironment(cfg, OutfitterDataset.load(cfg.data_dir))
    env.lifecycle_scales = replace(env.lifecycle_scales, new_asset_residual_rate=.6, residual_wear_exponent=.7)
    text = prompt(env, batch=batch)
    for value in ('Day 70', 'Day 80', '71 model responses', '60% nominal recovery baseline',
                  'Model usage is charged to shop cash'):
        assert value in text
    assert ('at most 3 independent' in text) == batch
    assert ('only the first executes' in text) == (not batch)
    assert 'P * 0.6 * c * max(0, 1 - w) ** 0.7' in json.dumps(env.business_manual())


@pytest.mark.parametrize('track', ['main', 'secondary'])
def test_manual_keeps_contracts_but_not_generation_hints(track):
    _, env = environment(track)
    state = env.reset()
    before = copy.deepcopy(state.to_dict())
    tools = ToolExecutor(env)
    result = tools.execute(state, 'read_business_rules', {'topics': list(env.business_manual())})
    assert result.ok
    manual = env.business_manual()
    full_text = json.dumps(manual)
    formula = resale_rule(env.lifecycle_scales.new_asset_residual_rate, env.lifecycle_scales.residual_wear_exponent)
    assert manual['scoring']['rules'].count(formula) == 1
    assert sum(formula in rule for chapter in manual.values() for rule in chapter['rules']) == 1
    assert 'See scoring rules' in json.dumps(manual['assets'])
    assert 'See scoring rules' in json.dumps(manual['procurement'])
    assert GUARANTEE_RULE in manual['reservations']['rules']
    assert PARTIAL_PICKUP_RULE in manual['reservations']['rules']
    for required in ('inquire_fallback', 'separate', 'once per commitment version',
                     'once', 'immediately', 'original date and customer-preferred plan',
                     'Rejection, expiry or a new commitment version does not restore',
                     "credits the missing lines' full rent", 'nominal', 'fixed', 'refund'):
        assert required in full_text, required
    assert 'Offer rental_asset_days includes' in full_text
    assert 'Offer rental_days includes' not in full_text
    assert 'complete-plan recovery' in full_text.lower()
    for forbidden in ('School holidays create', 'is_school_holiday is an independent',
                      'school-holiday weekends', 'neither holidays nor forecasts',
                      'Different price groups need not', 'liquidation-depth', 'Liquidation-depth',
                      'The historical field names', 'A selected optional reservation line is a binding promise'):
        assert forbidden not in full_text, forbidden
    # Removing explanations must not remove or reclassify the observable calendar.
    overlap = [env._public_calendar(day) for day in range(1,201)
               if env._calendar(day).get('is_school_holiday') and env._calendar(day)['is_weekend']]
    assert overlap and all(row['is_school_holiday'] and row['is_weekend'] for row in overlap)
    assert state.to_dict() == before


def test_unsupported_depth_stays_rejected_and_wear_fields_retained():
    cfg, env = environment()
    with pytest.raises(ValueError, match='fixes flat nominal liquidation'):
        OutfitterEnvironment(replace(cfg, liquidation_depth_enabled=True), OutfitterDataset.load(cfg.data_dir))
    text = json.dumps(env.business_manual())
    assert 'liquidation-depth' not in text.lower()
    assert 'not the probability of damage or retirement' in text
    assert 'service_life_limit_cycles' in env.public_offer('offer-021', detailed=True)['lifecycle_profile']


@pytest.mark.parametrize('track', ['main', 'secondary'])
def test_every_effective_tool_is_classified_after_overrides(track):
    _, env = environment(track)
    tools = ToolExecutor(env)
    definitions = {row['function']['name']: row['function'] for row in tools.definitions()}
    for name, row in definitions.items():
        text = row['description']
        if tools.is_read_only(name):
            assert text.endswith(' Read-only.'), name
            assert text.count('Read-only') == 1, name
        else:
            assert text.endswith(' State-changing; submit this tool alone in a response.'), name
    assert 'never both' in definitions['service_history']['description']
    assert 'bind at close' in definitions['quote_reservation_inquiries']['description']
    assert 'may induce cancellation and compensation' in definitions['request_reservation_contingency']['description']
    assert 'do not apply automatically' in definitions['apply_reservation_contingency']['description']
    assert 'reservation_close_outcomes' in definitions['close_business_day']['description']
    assert 'before walk-ins' in definitions['close_business_day']['description']


def test_v52_documentation_is_not_rewritten():
    _, old = environment(old=True)
    manual = old.business_manual()
    rule = public_terms(old)['durable_resale_formula']
    for topic in ('assets', 'procurement', 'scoring'):
        assert rule in manual[topic]['rules']
    assert SCHOOL_RULES in manual['reservations']['rules']
    text = prompt(old)
    assert 'Batchable tools:' in text and rule in text and SCHOOL_RULES in text
    assert len(text) == 11583
