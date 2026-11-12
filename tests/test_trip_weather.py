from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from outfitter.config import OutfitterConfig
from outfitter.data import OutfitterDataset
from outfitter.environment import OutfitterEnvironment
from outfitter.models import WorldState, RentalAsset
from outfitter.llm import ScriptedProvider
from outfitter.runner import OutfitterRunner
from outfitter.runtime.ledger import empty_ledger
from outfitter.trip_weather import trip_context, contextualize, ENVIRONMENT_PROTOCOL

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/outfitter_v54_main.example.yaml"


def env(**kwargs):
    c = replace(OutfitterConfig.from_yaml(MAIN), **kwargs)
    return OutfitterEnvironment(c, OutfitterDataset.load(c.data_dir))


def confirm(e, s):
    r = None
    for _ in range(14):
        r = next((row for row in s.reservation_requests
                  if row.status == "pending_service_offer"
                  and any(line.service_key == "rental:sleeping_bag" for line in row.lines)), None)
        if r is not None:
            break
        e.close_business_day(s)
    assert r is not None, "fixture requires a fixed offer with a sleeping bag"
    plan = r.customer_authorized_plans[0]
    e.respond_reservation_service_offers(s, [{"request_id": r.request_id,
        "action": "accept", "plan_id": plan["plan_id"],
        "pickup_day": max(x["pickup_day"] for x in r.pickup_date_options)}])
    return r, s.reservation_commitments[-1]


def due_fixture(*, assets=False):
    e = env(); s = e.reset(); r, cm = confirm(e, s)
    s.day = cm.pickup_day; s.current_day_ledger = empty_ledger(s.day)
    for asset in s.assets:
        asset.status = "maintenance_hold"
    if assets:
        for line in cm.lines:
            o = e.offers[line.offer_id]
            for n in range(line.quantity):
                a = RentalAsset(asset_id=f"v44-{line.line_id}-{n}", offer_id=o.offer_id,
                    service_line=o.service_line, unit_cost=o.unit_cost, used_life_days=0,
                    service_life_limit_days=int(o.service_life_limit_days),
                    damage_risk_band=str(o.damage_risk_band),
                    service_life_limit_cycles=o.service_life_limit_cycles)
                e._initialize_lifecycle_asset_state(a)
                a.durable_capital_basis_cents = e._capital_cents(a.unit_cost)
                s.assets.append(a)
        # This fixture supplies a larger initial fleet, with reconciled basis.
        e._initialize_durable_acquisition_budget(s)
    return e, s, r, cm


def test_future_sky_and_temperature_do_not_change_customer_requests(monkeypatch):
    e = env(); s = e.reset(); rows = e.weather_forecast(s, 15)
    def fixed(self, state, horizon):
        return [copy.deepcopy(r) for r in rows if state.day <= r['day'] < state.day + horizon]
    monkeypatch.setattr(OutfitterEnvironment, 'weather_forecast', fixed)
    states = []
    for low in (-5.0, 25.0):
        data = copy.deepcopy(e.source_dataset)
        for day, calendar in data.weather_calendar.items():
            if day > e.config.weather_start_day:
                calendar.update(night_low_c=low, temp_mean_c=low+5, day_high_c=low+10)
                data.weather[day] = 'snowy' if low < 0 else 'sunny'
        changed = OutfitterEnvironment(e.config, data)
        states.append(changed.reset().to_dict()['reservation_requests'])
    assert states[0] == states[1]


def test_quote_fit_uses_trip_snapshot_not_evaluation_day_or_true_weather():
    e = env(); s = e.reset()
    r = next(r for r in s.reservation_requests if any(l.service_key == 'rental:sleeping_bag' for l in r.lines))
    line = next(l for l in r.lines if l.service_key == 'rental:sleeping_bag')
    session = e._restore_quote_context(r.quote_context_by_line[line.line_id])
    original = e._functional_attribute_fit(s, session, 'offer-073')
    s.day = 60
    e._temperatures = lambda *_: (_ for _ in ()).throw(AssertionError('customer read true temperature'))
    assert e._functional_attribute_fit(s, session, 'offer-073') == original
    with pytest.raises(ValueError, match='frozen trip forecast'):
        e._functional_attribute_fit(s, replace(session, trip_weather_context=None), 'offer-073')
    quote = {'option_id': 'test', 'offer_id': line.offer_id,
             'promise_profile_id': line.eligible_profile_ids[0],
             'asset_quantity': max(1, line.service_units), 'daily_unit_price': 1.,
             'total_rent': float(line.duration_days), 'locked_security_deposit': 20.}
    e._bundle_store_option(s, r, line, quote)


def test_partial_forecast_does_not_extrapolate_or_extend_public_window():
    e = env(); s = e.reset()
    context = e._trip_weather_context(s, pickup_day=15, duration_days=5)
    assert context['covered_days'] == [15]
    assert context['unknown_days'] == [16,17,18,19]
    assert len(context['forecasts']) == 1
    from outfitter.tool_executor import ToolExecutor
    assert not ToolExecutor(e).execute(s, 'weather_forecast', {'horizon_days': 16}).ok


def test_each_advertised_date_has_its_own_temperature_context():
    e = env(); s = e.reset()
    for r in s.reservation_requests:
        for opt in r.pickup_date_options:
            for ctx in r.quote_context_by_date[str(opt['pickup_day'])].values():
                weather = ctx['trip_weather_context']
                assert weather['pickup_day'] == opt['pickup_day']
                assert weather['return_day'] == opt['scheduled_return_day']
                assert weather['issued_day'] == r.created_day
                if ctx['service_key'] == 'rental:sleeping_bag':
                    assert (ctx['preferred_group'] == 'three_season_bag') == (
                        weather['minimum_covered_forecast_night_low_c'] <= 15)
    assert WorldState.from_dict(s.to_dict()).to_dict() == s.to_dict()


def test_delayed_equipment_evaluation_uses_submission_snapshot(monkeypatch):
    e = env(); s = e.reset(); r, cm = confirm(e, s)
    line = next(l for l in cm.lines if l.service_key == 'rental:sleeping_bag')
    other = next(o for o in r.inquiry_profile_member_ids_by_line[line.line_id][line.promise_profile_id]
                 if o != line.offer_id)
    snapshot = e._contingency_forecast_snapshot(s, through_day=cm.pickup_day)
    changes = [{'line_id':line.line_id, 'offer_id':other, 'promise_profile_id':line.promise_profile_id}]
    kwargs = dict(pickup_day=cm.pickup_day, forecast_snapshot=snapshot, evaluation_day=s.day)
    before = e._contingency_equipment_raw_delta(s, r, cm, changes, **kwargs)
    s.day += 1
    def forbidden(*args, **kwargs):
        raise AssertionError('delayed customer evaluation read new weather')
    monkeypatch.setattr(e, '_temperatures', forbidden)
    monkeypatch.setattr(e, 'weather_forecast', forbidden)
    assert e._contingency_equipment_raw_delta(s, r, cm, changes, **kwargs) == before


def test_expired_reschedule_cancellation_is_in_close_receipt():
    e,s,r,cm = due_fixture()
    s.day = cm.pickup_day - 1
    s.current_day_ledger = empty_ledger(s.day)
    e._create_change_request(s, cm, change_type='reschedule',
                            proposed_pickup_day=cm.pickup_day+2,
                            proposed_return_day=cm.return_day+2)
    s.reservation_change_requests[-1].original_dates_still_possible = False
    result = e.close_business_day(s)
    receipt = result.data['closed_day']['reservation_close_outcomes'][0]
    assert receipt['outcome'] == 'customer_cancelled_reschedule_unanswered'
    assert receipt['refund_paid'] and receipt['refund_due'] > 0
    assert receipt['compensation_due'] == 0


def test_close_failure_receipt_survives_next_morning_and_roundtrip():
    e, s, r, cm = due_fixture()
    e._process_reservation_pickups(s)
    assert cm.status == 'at_risk'
    result = e.close_business_day(s, handoff_note='Mistakenly think it was solved')
    outcome = next(x for x in result.data['closed_day']['reservation_close_outcomes']
                   if x['commitment_id'] == cm.commitment_id)
    assert outcome['outcome'] == 'pickup_failed'
    assert outcome['refund_paid'] and outcome['compensation_paid']
    assert outcome['refund_amount_paid'] > 0 and outcome['compensation_amount_paid'] > 0
    restored = WorldState.from_dict(s.to_dict())
    opening = e.observation(restored, consume_inbox=False)
    assert opening['yesterday']['reservation_close_outcomes'] == result.data['closed_day']['reservation_close_outcomes']
    assert e.observation(restored, consume_inbox=False) == opening


@pytest.mark.parametrize('payment_stage', ['refund', 'compensation'])
def test_unpaid_settlement_is_never_reported_as_paid(payment_stage):
    e,s,r,cm = due_fixture()
    e.config.emergency_credit_limit = 0
    s.cash = 0 if payment_stage == 'refund' else cm.unearned_revenue + 0.01
    result = e.close_business_day(s)
    receipt = result.data['closed_day']['reservation_close_outcomes'][0]
    assert receipt['outcome'] == 'pickup_failed'
    # V48 distinguishes incomplete settlement from zero cash actually paid.
    assert not receipt['compensation_paid']
    assert receipt['compensation_amount_paid'] == (0.01 if payment_stage == 'compensation' else 0)
    assert receipt['refund_paid'] == (payment_stage == 'compensation')
    assert s.bankrupt


def test_successful_close_reports_real_full_pickup():
    e,s,r,cm = due_fixture(assets=True)
    result = e.close_business_day(s)
    receipt = result.data['closed_day']['reservation_close_outcomes'][0]
    assert receipt['outcome'] == 'pickup_fulfilled'
    assert receipt['optional_line_credit'] == 0
    assert receipt['compensation_due'] == 0


def test_runner_pauses_after_one_day_without_shortening_world(tmp_path):
    c = replace(OutfitterConfig.from_yaml(MAIN), runs_dir=str(tmp_path))
    close = {'role':'assistant','content':'','tool_calls':[{'id':'close','type':'function',
        'function':{'name':'close_business_day','arguments':'{}'}}]}
    p = ScriptedProvider([copy.deepcopy(close), copy.deepcopy(close)])
    runner = OutfitterRunner.create(config=c, provider=p, run_id='v44-pause')
    summary = runner.run(stop_after_day=1)
    assert summary['stop_reason'] == 'decision_day_limit'
    assert summary['day'] == 2 and not summary['done'] and summary['evaluation_score'] is None
    assert len(p.requests) == 1
    assert not runner.store.has_day_start(2)
    resumed = OutfitterRunner.resume(run_dir=runner.store.run_dir, provider=p)
    assert resumed.run(stop_after_day=2)['day'] == 3
    assert len(p.requests) == 2


def test_final_day_preserves_close_receipt_without_next_morning():
    e,s,r,cm = due_fixture()
    e.config.max_days = s.day
    s.max_days = s.day
    result = e.close_business_day(s)
    assert s.done and result.data['season_closed']
    assert result.data['closed_day']['reservation_close_outcomes'][0]['outcome'] == 'pickup_failed'


def test_tracks_share_customer_weather_and_close_policy():
    main = OutfitterConfig.from_yaml(MAIN).to_dict()
    other = OutfitterConfig.from_yaml(ROOT/'configs/outfitter_v54_secondary.example.yaml').to_dict()
    assert main.pop('assortment_mode') == 'rental_only'
    assert other.pop('assortment_mode') == 'rental_and_retail'
    assert main == other


@pytest.mark.parametrize('mode', ['rental_only', 'rental_and_retail'])
def test_seven_day_public_merchant_integration(mode):
    from outfitter.scripted_merchant import PublicMerchant
    from outfitter.tool_executor import ToolExecutor
    e = env(assortment_mode=mode); s = e.reset(); tools = ToolExecutor(e)
    errors = []
    def rpc(name, args):
        result = tools.execute(s, name, args)
        if not result.ok:
            errors.append((name, result.data))
        return {'ok':result.ok, 'data':result.data, 'terminal_day':result.terminal_day}
    merchant = PublicMerchant('calendar', rpc, retail_enabled=e.config.retail_enabled)
    merchant.start()
    for _ in range(7):
        merchant.day()
        receipt = e.close_business_day(s)
        assert 'reservation_close_outcomes' in receipt.data['closed_day']
        e._assert_v26_reservation_financial_invariants(s)
        e._assert_durable_acquisition_budget(s)
    assert s.day == 8 and not s.done and not errors
