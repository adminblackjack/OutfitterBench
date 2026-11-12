from __future__ import annotations

from collections import Counter, defaultdict

from .common import CONFIRM, FAILURES, PARAMETERS, ratio, unavailable
from .ledger import nominal_calendar


def basic_metrics(state, calls, horizon, manifest, profiles):
    history = [h for h in state['daily_history'] if h['day'] <= horizon]
    if [h['day'] for h in history] != list(range(1, horizon+1)):
        raise ValueError('completed daily history is not contiguous')
    events = [e for e in state.get('reservation_events', []) if e['day'] <= horizon]
    cs = {c['commitment_id']: c for c in state.get('reservation_commitments', [])}
    confirmations = {e['commitment_id']: e for e in events if e['event_type'] == CONFIRM}
    good = {e['commitment_id']: e for e in events if e['event_type'] == 'pickup_fulfilled'}
    bad = {e['commitment_id']: e for e in events if e['event_type'] in FAILURES}
    if good.keys() & bad.keys():
        raise ValueError('contract has both pickup and terminal merchant failure')
    external = {e['commitment_id'] for e in events if 'customer_cancelled' in e['event_type']}
    external -= bad.keys()
    pending = confirmations.keys()-good.keys()-bad.keys()-external
    partial = {e['commitment_id'] for e in events if e['event_type'] == 'partial_fulfillment_selected'}
    full = sum(not e['details'].get('optional_line_credit', 0) and cid not in partial
               for cid, e in good.items())
    settled = len(good)+len(bad)
    service = dict(reliability=ratio(len(good), settled), full_without_omission=ratio(full, settled),
                   pickups=len(good), merchant_failures=len(bad), customer_cancellations=len(external),
                   pending=len(pending), confirmed=len(confirmations),
                   request_delivery=ratio(len(good), sum(q['created_day'] <= horizon for q in state.get('reservation_requests', []))),
                   pending_ids=sorted(pending))
    finance = Counter()
    for h in history:
        for key, value in h['financial'].items():
            if isinstance(value, (int, float)):
                finance[key] += value
    finance = {k: round(v, 2) for k, v in finance.items()}
    assets = {a['asset_id']: a for a in state.get('assets', [])}
    activity = [e for e in state.get('rental_activity', []) if e['day'] <= horizon]
    depreciation = defaultdict(float)
    starts = [e for e in activity if e['event_type'] == 'rental_started']
    for e in starts:
        a = assets[e['asset_id']]
        if a.get('ownership_type', 'owned') != 'owned':
            continue
        wear = e['duration_days']/max(1, a['service_life_limit_days'])
        if a.get('service_life_limit_cycles'):
            wear += 1/a['service_life_limit_cycles']
        depreciation[e['day']] += a['unit_cost']*wear
    contracts = {c['contract_id']:c for c in state.get('contracts', [])}
    reservation_rent=sum(e['rent_paid'] for e in starts if contracts.get(e.get('contract_id'),{}).get('reservation_id'))
    reservation_rent+=sum(e['details'].get('additional_rent_collected',0) for e in events
                          if e['event_type']=='continuation_rent_collected')
    for episode in state.get('rental_episodes', {}).values():
        for notice in episode.get('notices', []):
            if notice.get('type') != 'not_returned' or notice['day'] > horizon:
                continue
            for cid in episode['contract_ids']:
                a = assets[contracts[cid]['asset_id']]
                if a.get('ownership_type', 'owned') == 'owned':
                    depreciation[notice['day']] += a['unit_cost']/max(1,a['service_life_limit_days'])
    use_cost = sum(h['financial'].get('rental_use_depreciation', round(depreciation[h['day']], 2)) for h in history)
    costs = ['cleaning_cost', 'repair_cost', 'maintenance_cost', 'rental_holding_cost',
             'rental_idle_holding_cost', 'reservation_compensation', 'supplier_trial_handling_fee', 'supplier_trial_usage_fee']
    cash_service = finance.get('rental_revenue', 0)+finance.get('damage_deductions', 0)-sum(finance.get(k, 0) for k in costs)
    capital_days = sum(h['end'].get('durable_capital', {}).get('deployed_owned_durable_basis', 0)+
                       h['end'].get('durable_capital', {}).get('committed_inbound_durable_basis', 0) for h in history)
    capital = dict(value=100*(cash_service-use_cost)/capital_days if capital_days else None,
                   unit='CNY per 100 capital-CNY-days', status='available' if capital_days else 'missing_data',
                   numerator=round(cash_service-use_cost, 2), denominator=round(capital_days, 2),
                   cash_service_contribution=round(cash_service, 2), use_depreciation=round(use_cost, 2),
                   cash_operating_contribution=round(cash_service-finance.get('operating_fee', 0), 2))
    quick = []
    mature = {cid for cid, e in confirmations.items() if e['day']+2 <= horizon}
    adverse = {'returned_damaged', 'asset_disposed', 'retired_service_life'}
    for cid, e in bad.items():
        if e['event_type'] != 'cancelled_by_manager' or cid not in confirmations:
            continue
        begin = confirmations[cid]['day']
        if e['day']-begin > 2:
            continue
        q = next(q for q in state['reservation_requests'] if q['request_id'] == cs[cid]['request_id'])
        plan = next((p for p in q.get('customer_authorized_plans', [])
                     if p['plan_id'] == confirmations[cid]['details'].get('manager_confirmed_plan_id')), {})
        oids = {l['offer_id'] for l in plan.get('lines', []) if l.get('disposition') != 'omit'}
        external_events = [dict(day=x['day'], event_type=x['event_type'], asset_id=x['asset_id'])
                           for x in activity if begin < x['day'] <= e['day']
                           and x.get('offer_id') in oids and x['event_type'] in adverse]
        # Final expected-return/order fields may have changed after the cutoff.
        # Late-return and supplier updates are classified in the replay layer only.
        contacts = [x['event_id'] for x in events if x.get('commitment_id') == cid
                    and begin <= x['day'] <= e['day'] and x['event_type'] in
                    {'reservation_contingency_requested', 'reservation_contingency_reported', 'fallback_inquired', 'partial_fulfillment_inquired'}]
        quick.append(dict(commitment_id=cid, request_id=q['request_id'], confirmed_day=begin,
                          cancel_day=e['day'], age=e['day']-begin, mature_window=cid in mature,
                          compensation=e['details'].get('compensation_amount_due', e['details'].get('compensation', 0)),
                          observed_external_events=external_events, contact_events=contacts,
                          external_classification='partial_evidence_requires_replay'))
    sales = [e for e in activity if e['event_type'] == 'asset_sold']
    new_assets = []
    for a in assets.values():
        if not 0 < a['acquired_day'] <= horizon or a.get('ownership_type', 'owned') != 'owned':
            continue
        end = a['acquired_day']+29
        ee = [e for e in starts if e['asset_id'] == a['asset_id'] and e['day'] <= min(horizon, end)]
        new_assets.append(dict(asset_id=a['asset_id'], offer_id=a['offer_id'], acquired_day=a['acquired_day'],
                               window_complete=end <= horizon, starts=len(ee),
                               initial_rent=round(sum(e['rent_paid'] for e in ee), 2)))
    errors = [dict(day=t['day'], tool=t['name'], rowid=t['rowid'], data=t['result'].get('data'))
              for t in calls if t['day'] <= horizon and not t['result']['ok']]
    effects=manifest.get('controller',{}).get('tool_effects',{})
    for e in errors:e['effect']=effects.get(e['tool'],'unknown')
    usage=Counter()
    for r in state.get('_evaluation_usage_by_day',[]):
        if r['day']<=horizon:
            for k in ['model_calls','input_tokens','output_tokens']:usage[k]+=r[k]
    try:
        nominal = nominal_calendar(state, horizon, profiles)
    except (KeyError, ValueError, StopIteration) as exc:
        nominal = dict(status='missing_data', value=None, reason=str(exc) or type(exc).__name__)
    end = history[-1]['end']
    closed_complete = horizon == state.get('max_days') and bool(state.get('done'))
    score = dict(cutoff_day=horizon, snapshot_day=state['day'],
                 evaluation_score=end.get('official_liquidation_value', end.get('business_value')) if closed_complete else None,
                 final_score_available=closed_complete,
                 contract_adjusted_value=end.get('contract_adjusted_economic_value', end.get('going_concern_business_value')),
                 cash=end['cash'], financial=finance,
                 initial_value=state.get('_evaluation_initial_value'),
                 reservation_rent=round(reservation_rent,2),
                 walk_in_rent=round(finance.get('rental_revenue',0)-reservation_rent,2),
                 run_bankrupt_at_snapshot=state.get('bankrupt',False))
    score['value_change']=(round(score['contract_adjusted_value']-score['initial_value'],2)
                           if score['initial_value'] is not None and score['contract_adjusted_value'] is not None else None)
    return dict(business=score, service=service, nominal_calendar=nominal, capital_productivity=capital,
                quick_cancellations=dict(rate=ratio(sum(r['mature_window'] for r in quick), len(mature)), records=quick),
                assets=dict(sales=sales, new_asset_30d=new_assets),
                execution=dict(tool_calls=sum(t['day'] <= horizon for t in calls), errors=errors,
                               errors_by_effect=dict(Counter(e['effect'] for e in errors)),usage=dict(usage),
                               forced_day_ends=sum(t['day']<=horizon and str(t['call_id']).startswith('forced-close-day-') for t in calls),
                               note='Tool effects use the recorded manifest, not a current-version name heuristic.'),
                extended=unavailable())
