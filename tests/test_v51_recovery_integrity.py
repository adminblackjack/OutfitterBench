"""Recovery consent, irreversible information and read-only query contracts."""
import copy
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

from outfitter.config import OutfitterConfig
from outfitter.data import OutfitterDataset
from outfitter.environment import OutfitterEnvironment
from outfitter.tool_executor import ToolExecutor
import test_v48_service_guarantee as legacy

ROOT = Path(__file__).resolve().parents[1]


def environment(track='rental_only', **overrides):
    cfg = replace(OutfitterConfig.from_yaml(ROOT/'configs/outfitter_v54_main.example.yaml'),
                  assortment_mode=track, **overrides)
    return OutfitterEnvironment(cfg, OutfitterDataset.load(cfg.data_dir))


def fixture():
    env = environment()
    state = env.reset()
    commitment = legacy.fixed(env, state, optional=True)
    legacy.at_risk(env, state, commitment)
    return env, state, commitment, ToolExecutor(env)


def omitted_plan(commitment):
    omitted = next(line.line_id for line in commitment.lines if not line.required)
    return {'plan_id': 'old-omit-plan', 'lines': [
        {'line_id': line.line_id, 'disposition': 'omit' if line.line_id == omitted else 'quote'}
        for line in commitment.lines]}


def test_historical_omit_cannot_bypass_refusal_or_guarantee():
    env, state, commitment, tools = fixture()
    commitment.preauthorized_recovery_plans = [omitted_plan(commitment)]
    request = next(r for r in state.reservation_requests if r.request_id == commitment.request_id)
    request.recovery_offer_ids_by_line = {}
    request.fallback_offer_ids_by_line = {}
    commitment.version_recovery_offer_ids_by_line = {}
    commitment.fallback_inquiry_day = state.day
    commitment.fallback_revealed_offer_ids_by_line = {}
    commitment.partial_fulfillment_inquiry_day = state.day
    commitment.partial_fulfillment_accepted = False
    before = copy.deepcopy(state.to_dict())
    result = tools.execute(state, 'resolve_reservation_risks', {'actions': [{
        'action': 'select_preauthorized_plan', 'commitment_id': commitment.commitment_id,
        'plan_id': 'old-omit-plan'}]})
    assert not result.ok and 'omits committed lines' in result.data['error']
    assert state.to_dict() == before
    auth = env.recovery_authorization(state, commitment)
    assert 'select_preauthorized_plan' not in auth['available_actions']
    assert auth['stage'] == 'inquired_without_candidates'
    response = tools.execute(state, 'resolve_reservation_risks', {'actions': [{
        'action': 'inquire_fallback', 'commitment_id': commitment.commitment_id}]})
    assert response.ok
    assert response.data['actions'][0]['preauthorized_recovery_plans'] == []
    detail = tools.execute(state, 'view_reservation', {'view': 'detail',
        'identifiers': [commitment.commitment_id], 'sections': ['risk']})
    assert detail.ok
    assert detail.data['records'][0]['risk']['preauthorized_recovery_plan_count'] == 0


@pytest.mark.parametrize('action', ['inquire_partial_fulfillment', 'inquire_fallback'])
def test_mixed_inquiry_batch_rejected_before_customer_contact(monkeypatch, action):
    env, state, commitment, tools = fixture()
    before = copy.deepcopy(state.to_dict())
    def forbidden(*args, **kwargs):
        raise AssertionError('must validate before querying customer')
    monkeypatch.setattr(env, '_resolve_reservation_risks_with_versions', forbidden)
    result = tools.execute(state, 'resolve_reservation_risks', {'actions': [
        {'action': action, 'commitment_id': commitment.commitment_id},
        {'action': 'select_partial_fulfillment', 'commitment_id': commitment.commitment_id}]})
    assert not result.ok and 'only action' in result.data['error']
    assert result.data['state_unchanged'] and state.to_dict() == before


@pytest.mark.parametrize('accept', [False, True])
def test_partial_consent_and_actual_payment_still_work(monkeypatch, accept):
    # Exercise the existing real pickup/guarantee assertion against V51, not
    # the V48 fixture. Includes no payment on inquiry or selection.
    monkeypatch.setattr(legacy, 'environment', environment)
    legacy.test_actual_partial_pickup_consent_and_payment(monkeypatch, accept)


def test_refused_inquiry_is_persisted_and_cannot_be_reasked(monkeypatch):
    env, state, commitment, tools = fixture()
    # Use a genuine partial-risk state, retaining only required inventory.
    from outfitter.models import RentalAsset
    from types import SimpleNamespace
    for line in commitment.lines:
        if not line.required:
            continue
        offer = env.offers[line.offer_id]
        for i in range(line.quantity):
            asset = RentalAsset(asset_id=f'consent-{line.line_id}-{i}', offer_id=line.offer_id,
                service_line=offer.service_line, unit_cost=offer.unit_cost, used_life_days=0,
                service_life_limit_days=offer.service_life_limit_days,
                service_life_limit_cycles=offer.service_life_limit_cycles,
                damage_risk_band=offer.damage_risk_band)
            env._initialize_lifecycle_asset_state(asset)
            asset.durable_capital_basis_cents=env._capital_cents(asset.unit_cost)
            state.assets.append(asset)
    env._initialize_durable_acquisition_budget(state)
    env._process_reservation_pickups(state)
    original = env._stable_event_rng
    monkeypatch.setattr(env, '_stable_event_rng', lambda *a, **kw:
        SimpleNamespace(random=lambda: 1.) if str(kw.get('identity', '')).startswith('partial:')
        else original(*a, **kw))
    args={'actions': [{'action': 'inquire_partial_fulfillment', 'commitment_id': commitment.commitment_id}]}
    reply=tools.execute(state, 'resolve_reservation_risks', args)
    assert reply.ok and reply.data['actions'][0]['customer_accepted'] is False
    snapshot=copy.deepcopy(state.to_dict())
    rejected=tools.execute(state, 'resolve_reservation_risks', args)
    assert not rejected.ok and 'already had' in rejected.data['error']
    assert state.to_dict()==snapshot
    assert sum(e.event_type=='partial_fulfillment_inquired' for e in state.reservation_events)==1


def test_ordinary_recovery_still_selectable_after_separate_inquiry():
    env, state, commitment, tools = fixture()
    # Force a complete plan made from current terms; no omissions.
    request = next(r for r in state.reservation_requests if r.request_id == commitment.request_id)
    chosen = next(p for p in request.customer_authorized_plans
                  if p['plan_id'] == commitment.original_confirmed_plan_id)
    plan = copy.deepcopy(chosen)
    plan['plan_id'] = 'retain-current'
    commitment.preauthorized_recovery_plans = [plan]
    inquiry = tools.execute(state, 'resolve_reservation_risks', {'actions': [{
        'action': 'inquire_fallback', 'commitment_id': commitment.commitment_id}]})
    assert inquiry.ok
    assert 'retain-current' in [p['plan_id'] for p in inquiry.data['actions'][0]['preauthorized_recovery_plans']]
    result = tools.execute(state, 'resolve_reservation_risks', {'actions': [{
        'action': 'select_preauthorized_plan', 'commitment_id': commitment.commitment_id,
        'plan_id': 'retain-current'}]})
    assert result.ok


@pytest.mark.parametrize('track', ['rental_only', 'rental_and_retail'])
def test_quote_eligible_subset_and_strict_atomic_sale(track):
    env = environment(track)
    state = env.reset()
    tools = ToolExecutor(env)
    good, bad = state.assets[:2]
    good.status = 'available'
    bad.status = 'rented'
    before = copy.deepcopy(state.to_dict())
    args = {'rental_asset_ids': [good.asset_id, bad.asset_id, 'missing-asset']}
    result = tools.execute(state, 'view_liquidation_quotes', args)
    assert result.ok and state.to_dict() == before
    assert [r['asset_id'] for r in result.data['rental_assets']] == [good.asset_id]
    assert {r['asset_id'] for r in result.data['unavailable_items']} == {bad.asset_id, 'missing-asset'}
    assert result.data['totals']['total_cash_proceeds'] == result.data['rental_assets'][0]['offer_price']
    sale = tools.execute(state, 'sell_to_liquidation_dealer', args)
    assert not sale.ok and state.to_dict() == before
    empty = tools.execute(state, 'view_liquidation_quotes', {'rental_asset_ids': [bad.asset_id]})
    assert empty.ok and empty.data['totals']['total_cash_proceeds'] == 0
    duplicate = tools.execute(state, 'view_liquidation_quotes', {'rental_asset_ids': [good.asset_id]*2})
    assert not duplicate.ok and state.to_dict() == before


def test_retail_query_unavailable_subset():
    env = environment('rental_and_retail')
    state = env.reset()
    tools = ToolExecutor(env)
    oid, inventory = next((k, v) for k, v in state.retail_inventory.items() if v['quantity'] > 0)
    before = copy.deepcopy(state.to_dict())
    result = tools.execute(state, 'view_liquidation_quotes', {'retail_items': [
        {'offer_id': oid, 'units': 1}, {'offer_id': 'unknown-retail', 'units': 1}]})
    assert result.ok and state.to_dict() == before
    assert len(result.data['retail_items']) == 1
    assert result.data['unavailable_items'][0]['status'] == 'unknown_offer'


def test_history_schema_expresses_mutual_exclusion_and_descriptions():
    env = environment()
    definitions = ToolExecutor(env).definitions()
    for row in definitions:
        f = row['function']
        props = f['parameters'].get('properties', {})
        if 'period' not in props or 'horizon_days' not in props:
            continue
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({'period': 'last_7_days', 'horizon_days': 7}, f['parameters'])
        jsonschema.validate({'period': 'last_7_days'}, f['parameters'])
        jsonschema.validate({'horizon_days': 7}, f['parameters'])
        assert 'never both' in f['description']
    desc = next(r['function']['description'] for r in definitions
                if r['function']['name'] == 'resolve_reservation_risks')
    assert 'only action' in desc
    assert 'ordered inquiry then selection in a batch is allowed' not in desc
    schema = next(r['function']['parameters'] for r in definitions
                  if r['function']['name'] == 'resolve_reservation_risks')
    inquiry={'action': 'inquire_fallback', 'commitment_id': 'reservation-001'}
    selection={'action': 'select_preauthorized_plan', 'commitment_id': 'reservation-001', 'plan_id': 'plan-01'}
    jsonschema.validate({'actions': [inquiry]}, schema)
    jsonschema.validate({'actions': [selection, selection]}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'actions': [inquiry, selection]}, schema)
    manual = str(env.business_manual())
    assert 'Inquiry then selection may be submitted in order in one atomic batch.' not in manual
