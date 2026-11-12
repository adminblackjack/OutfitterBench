"""Whitelisted public facts and deterministic nominal readiness, never future truth."""
from __future__ import annotations

from collections import Counter, defaultdict
from bisect import bisect_right
import math

from .common import OMITTED_LINE_STATUSES, deficits, line_units, loads


def public_problem(environment, state, gap=1):
    day = state.day
    # Only currently visible queue work; future weather and damage draws are not read.
    work = 0.0
    queue_ready = {}
    for job in sorted(state.cleaning_queue, key=lambda x:(x.enqueued_day,x.asset_id)):
        work += job.remaining_work_points
        queue_ready[job.asset_id] = day+max(1,math.ceil(max(0,work-1e-9)/environment.rules.cleaning_capacity_points))
    active = {r.asset_id:r for r in state.contracts if r.status == 'active'}
    trials = {t.asset_id:t for t in state.supplier_trials if t.asset_id}
    tokens = []
    facts = []
    for a in state.assets:
        ready = None
        expiry = state.max_days+gap+1
        public_return = None
        if a.status == 'available':
            ready = day
        elif a.status == 'cleaning':
            ready = queue_ready.get(a.asset_id)
        elif a.status in {'repairing','maintenance'}:
            ready = a.ready_day if a.ready_day is not None and a.ready_day >= day else None
        elif a.status == 'rented' and a.asset_id in active:
            r = active[a.asset_id]
            public_return = r.estimated_return_day or r.approved_return_day or r.promised_return_day
            # Silence after the public due date is not a promise of next-day return.
            if public_return is not None and public_return > day:
                ready = public_return+gap
        if a.ownership_type == 'supplier_trial':
            trial = trials.get(a.asset_id)
            if trial is None or trial.return_pending or trial.remaining_contract_starts < 1:
                ready = None
            else:
                expiry = trial.return_deadline
        facts.append(dict(asset_id=a.asset_id,offer_id=a.offer_id,status=a.status,
                          nominal_ready_day=ready,public_return_day=public_return,expires=expiry))
        if ready is not None:
            tokens.append((a.offer_id,int(ready),int(expiry),a.asset_id))
    for order in state.orders:
        if order.status != 'in_transit' or order.procurement_mode != 'owned':
            continue
        eta = order.estimated_delivery_day
        if eta is None or eta <= day:
            continue
        for index,item in enumerate(order.items):
            for n in range(item['packs']*item['package_units']):
                tokens.append((item['offer_id'],int(eta),state.max_days+gap+1,f"{order.order_id}:{index}:{n}"))
    book = {}
    for c in state.reservation_commitments:
        if c.status not in {'confirmed','at_risk'}:
            continue
        lines = [dict(line_id=l.line_id, offer_id=l.staged_offer_id or l.selected_offer_id or l.offer_id,
                      asset_quantity=l.asset_quantity or l.quantity, status=l.status)
                 for l in c.lines if l.status not in OMITTED_LINE_STATUSES
                 and not (c.partial_fulfillment_selected and l.line_id in c.partial_fulfillment_missing_line_ids)]
        book[c.commitment_id] = dict(id=c.commitment_id,request_id=c.request_id,born=c.confirmed_day,
                                     start=c.pickup_day,end=c.return_day,units=line_units(lines),
                                     version=c.commitment_version)
    return dict(day=day,gap=gap,tokens=tokens,book=book,asset_facts=facts,
                method='public-readiness-calendar-v1',
                assumptions=['current FIFO projected; future returns use public estimates',
                             'fixed future turnaround, no simulated future cleaning contention or random damage',
                             'overdue supply without a future public estimate excluded'])


def supply_function(problem):
    by_offer = defaultdict(list)
    for oid,ready,end,aid in problem['tokens']:
        by_offer[oid].append((ready,end))
    starts = {oid: sorted(begin for begin,end in rows if begin < end) for oid,rows in by_offer.items()}
    ends = {oid: sorted(end for begin,end in rows if begin < end) for oid,rows in by_offer.items()}
    return lambda oid,t: bisect_right(starts.get(oid,()),t)-bisect_right(ends.get(oid,()),t)


def check_book(problem):
    total = loads(problem['book'],problem['day'],problem['gap'])
    supply = supply_function(problem)
    return {cid: deficits(c,total,supply,problem['day'],problem['gap'])
            for cid,c in problem['book'].items()}


def normal_walk_contribution(environment,state,contracts):
    """Public normal-service estimate. Never inspect damaged_on_return or weather tape."""
    assets = {a.asset_id:a for a in state.assets}
    rent = clean = wear = 0.0
    for c in contracts:
        a = assets[c.asset_id]
        rent += c.rent_paid
        clean += environment._lifecycle_cleaning_terms(a,duration_days=c.duration_days)['cash_cost']
        if a.ownership_type == 'supplier_trial':
            wear += c.supplier_trial_usage_fee
        else:
            wear += a.unit_cost*(c.duration_days/max(1,a.service_life_limit_days)+
                                 (1/a.service_life_limit_cycles if a.service_life_limit_cycles else 0))
    return dict(rent=round(rent,6),normal_cleaning=round(clean,6),use_cost=round(wear,6),
                contribution=round(rent-clean-wear,6))
