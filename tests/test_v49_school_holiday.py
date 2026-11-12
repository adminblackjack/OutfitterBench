import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from outfitter.autonomous_rentals import continuation_probability
from outfitter.build_school_holiday_runtime import build
from outfitter.config import OutfitterConfig
from outfitter.data import OutfitterDataset
from outfitter.environment import OutfitterEnvironment
from outfitter.known_model import KnowledgeProvider
from outfitter.models import WorldState
from outfitter.reservation_contingency import date_utility_delta
from outfitter.runtime.common import _ARCHETYPE_INDEX, _RNG_CUSTOMER_BASKET
from outfitter.runtime.ledger import empty_ledger
from outfitter.school_holiday import (POLICY_ID, MULTIPLIERS, arrival_base, calendar_with_school,
    child_probability, rental_duration)
from outfitter.tool_executor import ToolExecutor
from outfitter.v5 import V5MarketRules, calendar_classification, seasonal_index
from datetime import date

ROOT = Path(__file__).resolve().parents[1]
from release_fixtures import v48_runtime, v49_runtime
RUNTIME = v49_runtime()
OLD = v48_runtime()


def env(version=49, **kwargs):
    name = 'school_holiday_r2' if version == 49 else 'service_guarantee'
    c = replace(OutfitterConfig.from_yaml(ROOT / 'tests/fixtures/outfitter_v50_legacy_main.yaml'), data_dir=str(RUNTIME if version==49 else OLD), **kwargs)
    return OutfitterEnvironment(c, OutfitterDataset.load(c.data_dir))


@pytest.mark.parametrize('text,school,weekend', [
    ('2026-06-30', False, False), ('2026-07-01', True, False), ('2026-07-04', True, True),
    ('2026-08-30', True, True), ('2026-08-31', True, False), ('2026-09-01', False, False)])
def test_calendar_flags_are_orthogonal(text, school, weekend):
    old = calendar_classification(date.fromisoformat(text)); c = calendar_with_school(old)
    assert c['is_school_holiday'] == school and c['is_weekend'] == weekend
    assert c['day_type'] == old['day_type'] and 'is_school_holiday' not in old


@pytest.mark.parametrize('kind', list(MULTIPLIERS))
@pytest.mark.parametrize('day_type,weekend,holiday', [('weekday', False, None),
    ('weekend', True, None), ('public_holiday', True, 'national_day')])
def test_additive_increment_not_peak_multiplication(kind, day_type, weekend, holiday):
    r = V5MarketRules(); cal = {'day_type': day_type, 'is_weekend': weekend, 'holiday_id': holiday}
    before = arrival_base(kind, cal, r)
    after = arrival_base(kind, {**cal, 'is_school_holiday': True}, r)
    assert after >= before
    assert after-before == pytest.approx(r.archetype_base_lambdas[kind][0]*(MULTIPLIERS[kind]-1))
    if weekend and MULTIPLIERS[kind] > 1:
        assert after < before*MULTIPLIERS[kind]


def test_overlap_child_probability_and_duration_do_not_stack():
    cal = {'day_type': 'weekend', 'is_weekend': True, 'is_school_holiday': True}
    assert child_probability(cal) == .45
    assert child_probability({**cal, 'day_type': 'public_holiday'}) == .45
    a = [rental_duration('overnight', cal, np.random.default_rng(i)) for i in range(100)]
    b = [rental_duration('overnight', {**cal, 'day_type': 'school_holiday', 'is_weekend': False},
                         np.random.default_rng(i)) for i in range(100)]
    assert a == b and min(a) == 2 and max(a) == 5


def test_date_preference_and_continuation_overlap():
    args = dict(customer_type='family', current_day_type='school_holiday', candidate_day_type='weekend',
        current_condition='sunny', candidate_condition='sunny', shift_days=0,
        schedule_flexibility=.5, forecast_trust=.5, occasion_importance=1.)
    assert date_utility_delta(**args) == pytest.approx(-.2)
    assert date_utility_delta(**args, candidate_is_school_holiday=True) == 0
    assert date_utility_delta(**{**args, 'candidate_day_type': 'public_holiday'},
                             candidate_is_school_holiday=True) == pytest.approx(.1)
    c = {'day_type': 'weekend', 'is_weekend': True}
    params = dict(archetype='overnight', flexibility=.55, condition='sunny', calendar=c, next_calendar=c,
                  daily_rent=3., cash_scale=40., extra_days=0)
    assert continuation_probability(**params) == continuation_probability(
        **{**params, 'calendar': {**c, 'is_school_holiday': True}, 'next_calendar': {**c, 'is_school_holiday': True}})


def test_public_forecast_snapshot_manual_and_version_roundtrip():
    e = env(); s = e.reset(); s.day = 90
    rows = e.weather_forecast(s, 15)
    assert next(r for r in rows if r['day'] == 95)['is_school_holiday']
    snap = e._contingency_forecast_snapshot(s, through_day=103)
    assert snap['95']['is_school_holiday']
    assert 'is_school_holiday' in json.dumps(e.business_manual())
    assert 'school_holiday_calendar' in e.public_operating_rules()
    assert s.calendar_demand_protocol == POLICY_ID
    assert WorldState.from_dict(s.to_dict()).to_dict() == s.to_dict()
    old = env(48); state_old = old.reset()
    assert 'calendar_demand_protocol' not in state_old.to_dict()
    assert 'is_school_holiday' not in old._public_calendar(95)
    with pytest.raises(ValueError, match='Calendar demand version'):
        e.close_business_day(state_old)
    with pytest.raises(ValueError, match='Calendar demand version'):
        old.close_business_day(s)
    assert not ToolExecutor(e).execute(state_old, 'business_status', {}).ok


@pytest.mark.parametrize('kind', ['overnight', 'picnic', 'lightweight'])
def test_real_channel_entry_points_share_departure_shape(monkeypatch, kind):
    e = env(); s = e.reset()
    class Capture(Exception):
        pass
    saved = []
    def capture(*args, **kwargs):
        saved.append(kwargs)
        raise Capture()
    monkeypatch.setattr(e, '_make_choice_session', capture)
    for index in range(5):
        expected = e._v47_customer_shape(95, kind, index)
        s.day = 86  # Before the July departure; issue-time date must not set shape.
        with pytest.raises(Capture):
            e._reservation_primary_session(s, pickup_day=95, archetype=kind,
                                           customer_index=index, booking_lead_days=9)
        s.day = 95; s.current_day_ledger = empty_ledger(95)
        address = (95, _ARCHETYPE_INDEX[kind], index)
        with pytest.raises(Capture):
            e._serve_customer_v5(s, e._event_rng(_RNG_CUSTOMER_BASKET, *address), kind,
                                 e._weather(95), e._demand_weather_regime(95), event_address=address)
        for captured in saved[-2:]:
            assert (captured['adults'], captured['children'], captured['duration_days']) == (
                expected['adults'], expected['children'], expected['duration'])


def test_summer_future_truth_invariance_and_query_order(monkeypatch):
    e = env(); base = e.reset(); base.day = 90; base.current_day_ledger = empty_ledger(90)
    base.reservation_requests = []; base.reservation_events = []
    rows = e.weather_forecast(base, 15)
    def fixed(self, state, horizon):
        return [copy.deepcopy(r) for r in rows if state.day <= r['day'] < state.day+horizon]
    monkeypatch.setattr(OutfitterEnvironment, 'weather_forecast', fixed)
    results = []
    for low in (-5., 25.):
        data = copy.deepcopy(e.source_dataset)
        for day, cal in data.weather_calendar.items():
            if day > e._source_weather_day(base.day):
                cal.update(night_low_c=low, temp_mean_c=low+5, day_high_c=low+10)
                data.weather[day] = 'snowy' if low < 0 else 'sunny'
        changed = OutfitterEnvironment(e.config, data); state = copy.deepcopy(base)
        # Irrelevant addressed draws and public queries cannot perturb demand.
        for _ in range(3):
            changed._event_rng(98765, 1).random(); changed.weather_forecast(state, 15)
        changed._generate_reservation_requests(state)
        results.append(state.to_dict()['reservation_requests'])
    assert results[0] and results[0] == results[1]


def test_main_secondary_shared_rental_kernel_and_no_duplicate_route():
    a, b = env(), env(assortment_mode='rental_and_retail')
    for day in (1, 91, 92, 95, 153, 154, 200):
        for kind in ('overnight', 'picnic', 'lightweight'):
            count = a._potential_arrival_count(day, kind)
            assert count == b._potential_arrival_count(day, kind)
            for i in range(count):
                assert a._v47_customer_shape(day, kind, i) == b._v47_customer_shape(day, kind, i)
                assert a._reservation_route(day, _ARCHETYPE_INDEX[kind], i) == b._reservation_route(day, _ARCHETYPE_INDEX[kind], i)


def test_builder_and_config_scope(tmp_path):
    build(OLD, tmp_path / 'data')
    new = OutfitterDataset.load(tmp_path / 'data')
    changed = [p.name for p in OLD.iterdir() if p.is_file() and p.read_bytes() != (tmp_path/'data'/p.name).read_bytes()]
    assert set(changed) == {'metadata.json', 'market_rules.json'}
    assert seasonal_index(196, new.market_rules.seasonal_anchors) == pytest.approx(1.22)
    assert seasonal_index(227, new.market_rules.seasonal_anchors) == pytest.approx(1.19)
    for name, sha in new.metadata['content_sha256'].items():
        assert hashlib.sha256((tmp_path/'data'/name).read_bytes()).hexdigest() == sha
    with pytest.raises(ValueError):
        build(OLD, tmp_path / 'data')
    c = env().config.to_dict(); old = env(48).config.to_dict()
    assert {k for k in c if c[k] != old[k]} == {'data_dir'}
    main = OutfitterConfig.from_yaml(ROOT/'tests/fixtures/outfitter_v50_legacy_main.yaml').to_dict()
    secondary = OutfitterConfig.from_yaml(ROOT/'tests/fixtures/outfitter_v50_legacy_secondary.yaml').to_dict()
    assert {k for k in main if main[k] != secondary[k]} == {'assortment_mode'}


def test_known_population_provider_retains_calendar_without_weather_truth():
    e = env(); k = KnowledgeProvider(e.config)
    assert k._env._calendar(95)['is_school_holiday']
    assert k._env._v47_arrival_rate(95, 'overnight') == e._v47_arrival_rate(95, 'overnight')
    with pytest.raises(AssertionError):
        k._env._weather(95)


@pytest.mark.parametrize('track', ['rental_only', 'rental_and_retail'])
def test_real_summer_closing_and_actor_calendar_on_both_tracks(track):
    e = env(assortment_mode=track); s = e.reset()
    # Synthetic calendar fixture, not a season continuation or profitability run.
    s.day = 94; s.current_day_ledger = empty_ledger(94)
    s.reservation_requests = []; s.reservation_events = []; s.inbox = []
    s.cumulative_reservation_requests = 0
    e._capture_opening_rental_holding_basis(s); e._prepare_reservation_day(s); e._seal_opening_flows(s)
    tools = ToolExecutor(e)
    for _ in range(3):
        forecast = tools.execute(s, 'weather_forecast', {'horizon_days': 3})
        assert forecast.ok and 'is_school_holiday' in json.dumps(forecast.data)
        result = tools.execute(s, 'close_business_day', {})
        assert result.ok and not s.bankrupt
        assert result.data['closed_day']['calendar']['is_school_holiday']
    assert [r['day'] for r in s.daily_history] == [94, 95, 96]


def test_contingency_delta_uses_frozen_school_flags(monkeypatch):
    e = env(); captured = {}
    import outfitter.runtime.reservation_contingency as module
    def capture(**kwargs):
        captured.update(kwargs); return .25
    monkeypatch.setattr(module, 'date_utility_delta', capture)
    monkeypatch.setattr(e, '_contingency_customer_context', lambda _: dict(
        customer_type='family', children=1, archetype='picnic', schedule_flexibility=.6,
        forecast_trust=.7, occasion_importance=1.))
    # Both labels are weekends, but only the first belongs to school holidays.
    snap = {str(d): {'day_type': 'weekend', 'is_school_holiday': flag, 'condition': 'sunny',
        'day_high_c': 28.} for d, flag in ((95, True), (158, False))}
    assert e._contingency_date_delta(commitment=SimpleNamespace(pickup_day=95), request=None,
                                     candidate_pickup_day=158, snapshot=snap) == .25
    assert captured['current_is_school_holiday'] and not captured['candidate_is_school_holiday']
