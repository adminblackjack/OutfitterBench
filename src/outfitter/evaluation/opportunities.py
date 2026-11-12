from __future__ import annotations

from collections import Counter
import itertools
import math

from .common import PARAMETERS, feasible, line_units, loads
from .projection import supply_function


def dates(request):
    return [(int(r['pickup_day']),int(r['scheduled_return_day'])) for r in request.pickup_date_options]


def plan_record(plan,start,end):
    return dict(start=start,end=end,units=line_units(plan['lines']))


def check_plans(plans,date_options,problem,exclude_id=None):
    book = {k:v for k,v in problem['book'].items() if k != exclude_id}
    total = loads(book,problem['day'],problem['gap'])
    supply = supply_function(problem)
    good=[]
    for start,end in date_options:
        for plan in plans:
            if feasible(plan_record(plan,start,end),total,supply,problem['day'],problem['gap']):
                good.append(dict(plan_id=plan['plan_id'],pickup_day=start))
    return good


def line_choices(request,catalog,problem,submitted=None):
    available = {oid for oid,_,_,_ in problem['tokens']}
    inputs = {r['line_id']:r for r in submitted or []}
    choices=[]
    for line in sorted(request.lines,key=lambda l:l.line_id):
        members=request.inquiry_profile_member_ids_by_line.get(line.line_id,{})
        if submitted is None:
            raw=[dict(disposition='quote',offer_id=oid,promise_profile_id=pid)
                 for pid in sorted(line.eligible_profile_ids) for oid in sorted(members.get(pid,[]))
                 if oid in available]
        else:
            raw=inputs.get(line.line_id,{}).get('options',[])
        options=[]
        # Optional omission is implicitly available under the V53/V54 quote protocol.
        if not line.required:
            options.append(dict(line_id=line.line_id,disposition='omit'))
        for row in raw:
            if row.get('disposition') != 'quote':continue
            pid=row['promise_profile_id'];oid=row['offer_id']
            if pid not in line.eligible_profile_ids or oid not in members.get(pid,[]):continue
            n=math.ceil(int(line.service_units or line.customer_units)/catalog.by_id[pid].capacity_per_asset)
            options.append(dict(line_id=line.line_id,disposition='quote',offer_id=oid,
                                promise_profile_id=pid,asset_quantity=n))
        # Technical duplicate offer/quantity pairs need not consume the search budget twice.
        unique={tuple(sorted(x.items())):x for x in options}
        choices.append([unique[k] for k in sorted(unique)])
    return choices


def search_menu(request,catalog,problem,submitted=None,limit=None):
    limit=PARAMETERS['candidate_limit_per_date'] if limit is None else limit
    choices=line_choices(request,catalog,problem,submitted)
    if any(not c for c in choices):return dict(status='infeasible',witness=None,tested=0,truncated=False)
    total=loads(problem['book'],problem['day'],problem['gap']);supply=supply_function(problem)
    tested=0;truncated=False
    for start,end in dates(request):
        combinations=itertools.product(*choices)
        for count,parts in enumerate(combinations):
            if count>=limit:
                truncated=True;break
            tested+=1
            plan=dict(plan_id='technical-candidate',lines=list(parts))
            if feasible(plan_record(plan,start,end),total,supply,problem['day'],problem['gap']):
                return dict(status='feasible',witness=dict(pickup_day=start,return_day=end,lines=list(parts)),
                            tested=tested,truncated=truncated)
    return dict(status='undetermined' if truncated else 'infeasible',witness=None,tested=tested,truncated=truncated)


def inspect_request(request,decision,catalog,problem):
    if request.service_offer_terms:
        # Actor-visible fixed complete plans, not private recovery authorization.
        plans=request.service_offer_terms.get('plans',request.customer_authorized_plans)
        if not plans:
            return dict(request_id=request.request_id,day=problem['day'],status='missing_data')
        good=check_plans(plans,dates(request),problem)
        selected=decision.get('action')=='accept' and any(p['plan_id']==decision.get('plan_id') and
                 p['pickup_day']==decision.get('pickup_day',request.pickup_day) for p in good)
        return dict(request_id=request.request_id,day=problem['day'],status='evaluated',kind='fixed',
                    opportunity=bool(good),covered=bool(selected),candidate_status='feasible' if good else 'infeasible',
                    alternatives=good)
    candidate=search_menu(request,catalog,problem)
    actual=(search_menu(request,catalog,problem,decision.get('lines',[]))
            if decision.get('action')=='quote' else dict(status='not_submitted'))
    return dict(request_id=request.request_id,day=problem['day'],status='evaluated',kind='inquiry',
                opportunity=candidate['status']=='feasible',covered=actual['status']=='feasible',
                candidate_status=candidate['status'],candidate=candidate,submitted=actual,
                quote_action=decision.get('action'))
