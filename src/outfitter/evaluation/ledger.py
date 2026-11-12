"""Historical, reservation-only nominal calendar. Never read final mutable terms."""
from __future__ import annotations

from collections import Counter, defaultdict
import math

from .common import CONFIRM, deficits, line_units, loads, occupied, ratio


def nominal_calendar(state, horizon, profiles):
    qs = {r['request_id']: r for r in state.get('reservation_requests', [])}
    cs = {r['commitment_id']: r for r in state.get('reservation_commitments', [])}
    assets = {a['asset_id']: a for a in state.get('assets', [])}
    trials = state.get('supplier_trials', [])
    trial_orders = {t['order_id'] for t in trials}
    initial = Counter(a['offer_id'] for a in assets.values() if a['acquired_day'] == 0
                      and a.get('ownership_type', 'owned') == 'owned')
    events = defaultdict(list)
    for e in state.get('reservation_events', []):
        if e['day'] <= horizon:
            events[e['day']].append(e)
    book, rows, exclusions = {}, [], []
    for day in range(1, horizon+1):
        capacity = initial.copy()
        for order in state.get('orders', []):
            if order['placed_day'] > day or order['order_id'] in trial_orders:
                continue
            # Accepted orders count immediately in this deliberately optimistic layer.
            # Cancellation timing cannot be inferred from a final cancelled status.
            if order['status'] not in {'delivered', 'in_transit', 'pending'}:
                raise ValueError('nominal calendar needs order-cancellation history')
            for item in order['items']:
                capacity[item['offer_id']] += item['packs']*item['package_units']
        exited = {e['asset_id'] for e in state.get('rental_activity', [])
                  if e['day'] <= day and e['event_type'] in
                  {'asset_sold', 'asset_disposed', 'retired_service_life', 'retired_by_agent'}}
        for aid in exited:
            a = assets[aid]
            if a.get('ownership_type', 'owned') == 'owned':
                capacity[a['offer_id']] -= 1
        active_trials = [t for t in trials if t.get('delivered_day') is not None
                         and t['delivered_day'] <= day
                         and (t.get('returned_day') is None or t['returned_day'] > day)]

        def supply(oid, date):
            return capacity[oid]+sum(t['offer_id'] == oid and date < t['return_deadline']
                                     for t in active_trials)

        new = []
        for e in events[day]:
            cid, typ, d = e.get('commitment_id'), e['event_type'], e['details']
            if typ == CONFIRM:
                co = cs[cid]
                q = qs[co['request_id']]
                plan = next(p for p in q['customer_authorized_plans']
                            if p['plan_id'] == d['manager_confirmed_plan_id'])
                lines = {l['line_id']: (l['offer_id'], l['asset_quantity'])
                         for l in plan['lines'] if l.get('disposition') != 'omit'}
                book[cid] = dict(id=cid, request_id=q['request_id'], born=day,
                                 start=co['initial_agreed_pickup_day'],
                                 end=co['initial_agreed_return_day'], units=line_units(plan['lines']),
                                 lines=lines, picked=False)
                new.append(cid)
            elif cid in book:
                b = book[cid]
                if typ == 'reschedule_accepted':
                    b.update(start=d['pickup_day'], end=d['return_day'])
                elif typ == 'reservation_contingency_applied':
                    b.update(start=d['new_pickup_day'], end=d['new_return_day'])
                    r = next(r for r in state['reservation_contingency_reports'] if r['report_id'] == d['report_id'])
                    options = r['proposals']+([r['counteroffer']] if r.get('counteroffer') else [])
                    chosen = next(p for p in options if p['proposal_id'] == d['selection_id'])
                    requested = {l['line_id']: l for l in qs[b['request_id']]['lines']}
                    for change in chosen.get('line_changes', []):
                        l = requested[change['line_id']]
                        n = l.get('service_units') or l.get('customer_units') or l['quantity']
                        n = math.ceil(n/profiles[change['promise_profile_id']]['capacity_per_asset'])
                        b['lines'][change['line_id']] = (change['offer_id'], n)
                elif typ == 'preauthorized_recovery_plan_selected':
                    options = qs[b['request_id']]['customer_authorized_plans']+cs[cid].get('preauthorized_recovery_plans', [])
                    p = next(p for p in options if p['plan_id'] == d['plan_id'])
                    b['lines'] = {l['line_id']: (l['offer_id'], l['asset_quantity']) for l in p['lines']
                                  if l.get('disposition') != 'omit'}
                elif typ == 'authorized_fallback_selected':
                    # A legacy event lacks line_id. Ambiguous repeated-SKU lines must not be guessed.
                    ids = [lid for lid, (oid, _) in b['lines'].items() if oid == d['from_offer_id']]
                    if len(ids) != 1:
                        raise ValueError('ambiguous historical fallback line')
                    lid = ids[0]
                    b['lines'][lid] = (d['to_offer_id'], b['lines'][lid][1])
                elif typ == 'partial_fulfillment_selected':
                    for lid in d['missing_line_ids']:
                        b['lines'].pop(lid, None)
                elif typ == 'pickup_fulfilled':
                    b['picked'] = True
                    b['units'] = dict(Counter(assets[a]['offer_id'] for a in d['assignments']))
                    continue
                elif typ == 'reservation_completed':
                    b['end'] = day
                elif typ == 'pickup_failed' or 'cancelled' in typ or typ == 'merchant_induced_customer_cancellation':
                    del book[cid]
                    continue
                if not b['picked']:
                    counts = Counter()
                    for oid, n in b['lines'].values():
                        counts[oid] += n
                    b['units'] = dict(counts)
        for gap in (0, 1, 2):
            total = loads(book, day, gap)
            for cid in new:
                if cid not in book:
                    if gap == 0:
                        exclusions.append(dict(day=day, commitment_id=cid, reason='terminated_same_day'))
                    continue
                b = book[cid]
                own = occupied(b, day, gap)
                other = total.copy()
                other.subtract(own)
                prior = [p for p in book.values() if p['born'] < day and not p['picked']
                         and set(occupied(p, day, gap)) & set(own)]
                affected = [p['id'] for p in prior if not deficits(p, other, supply, day, gap)
                            and deficits(p, total, supply, day, gap)]
                missing = deficits(b, total, supply, day, gap)
                rows.append(dict(day=day, gap=gap, commitment_id=cid, request_id=b['request_id'],
                                 covered=not missing, shortages=missing,
                                 overlaps_prior=bool(prior), affected_prior=affected))
    metrics = {}
    for gap in (0, 1, 2):
        selected = [r for r in rows if r['gap'] == gap]
        prior = [r for r in selected if r['overlaps_prior']]
        metrics[str(gap)] = dict(
            coverage=ratio(sum(r['covered'] for r in selected), len(selected)),
            protection=ratio(sum(not r['affected_prior'] for r in prior), len(prior)))
    return dict(status='available', method='optimistic-reservation-calendar-v1',
                assumptions=['accepted orders count immediately', 'no walk-ins, queues, damage or future life exhaustion'],
                metrics=metrics, records=rows, exclusions=exclusions)
