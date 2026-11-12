"""Optional exact replay and local demand branches. Never called by basic analysis."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import time
from .execution import read_state, write_state, occupied_cpu_slots

from ..models import WorldState
from ..runner import _implementation_identity
from ..tool_executor import ToolExecutor
from ..reservation_value import (_load_environment, _normalized_replay_result,
                                 _opening_observation_days, _recorded_controller_no_op, _tool_calls)
from .common import FAILURES, PARAMETERS, digest, line_units, loads, occupied, ratio, write_json
from .opportunities import check_plans, dates, inspect_request
from .projection import check_book, normal_walk_contribution, public_problem, supply_function

PROJECTION_ACTIONS = frozenset({'quote_reservation_inquiries','respond_reservation_service_offers',
    'confirm_reservation_plans','respond_reservation_changes','apply_reservation_contingency',
    'sell_to_liquidation_dealer','schedule_maintenance','repair_or_retire_assets'})


def demand_receipt(state, contracts):
    return dict(contracts=[asdict(c) for c in contracts],
        financial={k:getattr(state,k) for k in
                   ['cash','deposit_liability','unearned_revenue_liability','cumulative_rental_revenue']},
        counter_interactions=copy.deepcopy(state.current_day_ledger.get('counter_interactions')))


def new_rentals(state, old_ids, day):
    return [r for r in state.contracts if r.contract_id not in old_ids
            and r.reservation_id is None and r.start_day == day]


def occupied_new(contracts, day, gap):
    out=Counter()
    for c in contracts:
        for t in range(day,c.return_day+gap):out[c.offer_id,t]+=1
    return out


def relaxation_safe(problem, available, actual, candidate, gap):
    """Only increased occupancy cells, with independent currently available supply."""
    before=occupied_new(actual,problem['day'],gap)
    after=occupied_new(candidate,problem['day'],gap)
    promises=loads(problem['book'],problem['day'],gap)
    violations=[]
    for (oid,day),n in after.items():
        if n>before[oid,day] and n+promises[oid,day]>available[oid]:
            violations.append(dict(offer_id=oid,day=day,demand=n+promises[oid,day],supply=available[oid]))
    return violations


class Recorder:
    def __init__(self, environment, cache, sample_days, restored=None, dispatcher=None):
        self.env=environment;self.cache=Path(cache);self.sample_days=set(sample_days)
        self.rows=restored or {k:[] for k in ['walk_in_intrusion','persistent_gaps','opportunities',
            'authorization_selection','action_protection','capacity_relaxation','public_changes','evidence_contact']}
        self.day_queries=[]
        self.day_refs=set()
        self.dispatcher=dispatcher

    def mutation_before(self,state,name,args):
        if name not in PROJECTION_ACTIONS:
            return {}  # Still records evidence contact, without unused projections.
        problem=public_problem(self.env,state)
        by_request={r.request_id:r for r in state.reservation_requests}
        if name in {'quote_reservation_inquiries','respond_reservation_service_offers'}:
            for decision in args.get('decisions',[]):
                q=by_request.get(decision.get('request_id'))
                if q is not None:
                    record=inspect_request(q,decision,self.env.reservation_promise_catalog,problem)
                    record['tool']=name
                    record['batch_shared_prestate']=len(args.get('decisions',[]))>1
                    self.rows['opportunities'].append(record)
        if name=='confirm_reservation_plans':
            for selection in args.get('selections',[]):
                q=by_request.get(selection.get('request_id'))
                if q is not None:
                    self.authorization(q,selection,problem,origin='explicit')
        if name=='respond_reservation_service_offers':
            for decision in args.get('decisions',[]):
                q=by_request.get(decision.get('request_id'))
                if q is not None and decision.get('action')=='accept':
                    self.authorization(q,decision,problem,origin='fixed_offer')
        return problem

    def authorization(self,q,selection,problem,origin):
        good=check_plans(q.customer_authorized_plans,dates(q),problem)
        self.rows['authorization_selection'].append(dict(day=problem['day'],request_id=q.request_id,
            origin=origin,opportunity=bool(good),alternatives=good,
            selected_feasible=any(p['plan_id']==selection.get('plan_id') and
                p['pickup_day']==selection.get('pickup_day',q.pickup_day) for p in good)))

    def mutation_after(self,state,name,args,before,recorded_ok):
        if not recorded_ok or name not in PROJECTION_ACTIONS:return
        after=public_problem(self.env,state)
        if name=='quote_reservation_inquiries':
            before_ids=set(before['book'])
            qs={q.request_id:q for q in state.reservation_requests}
            for cid,co in after['book'].items():
                if cid not in before_ids:
                    q=qs[co['request_id']]
                    self.authorization(q,dict(plan_id=q.manager_confirmed_plan_id or q.customer_preferred_plan_id,
                                              pickup_day=co['start']),before,origin='single_authorized_auto')
        if name in {'quote_reservation_inquiries','respond_reservation_service_offers','confirm_reservation_plans',
                    'respond_reservation_changes','apply_reservation_contingency','sell_to_liquidation_dealer',
                    'schedule_maintenance','repair_or_retire_assets'}:
            old=check_book(before);new=check_book(after)
            affected=[cid for cid in old if cid in new and not old[cid] and new[cid]]
            changed_oids=set()
            bt=Counter((x[0],x[1],x[2]) for x in before['tokens'])
            at=Counter((x[0],x[1],x[2]) for x in after['tokens'])
            changed_oids.update(x[0] for x in set(bt)|set(at) if bt[x]!=at[x])
            for cid in set(before['book'])|set(after['book']):
                b=before['book'].get(cid);a=after['book'].get(cid)
                if b!=a:
                    changed_oids.update((b or {}).get('units',{}));changed_oids.update((a or {}).get('units',{}))
            relevant=[cid for cid,b in before['book'].items() if changed_oids & set(b['units'])]
            self.rows['action_protection'].append(dict(day=before['day'],tool=name,relevant_prior=relevant,
                affected_prior=affected,removed_prior=sorted(set(old)-set(new)),changed_offers=sorted(changed_oids),
                shortages={cid:new[cid] for cid in affected}))

    def observe_world(self, before, after, state):
        bf={a['asset_id']:a for a in before['asset_facts']};af={a['asset_id']:a for a in after['asset_facts']}
        for aid,a in af.items():
            b=bf.get(aid,{})
            if a['status']=='damaged' and b.get('status')!='damaged':
                self.rows['public_changes'].append(dict(day=state.day,offer_id=a['offer_id'],kind='damage',asset_id=aid))
            if a['status']=='rented' and (a['public_return_day'] or 0)<=state.day:
                self.rows['public_changes'].append(dict(day=state.day,offer_id=a['offer_id'],kind='overdue',asset_id=aid))
            elif (a['public_return_day'] is not None and b.get('public_return_day') is not None
                  and a['public_return_day']>b['public_return_day']):
                self.rows['public_changes'].append(dict(day=state.day,offer_id=a['offer_id'],kind='later_public_return',asset_id=aid))
        for o in state.orders:
            if o.status=='in_transit' and o.estimated_delivery_day is not None and o.estimated_delivery_day<=state.day:
                for it in o.items:self.rows['public_changes'].append(dict(day=state.day,offer_id=it['offer_id'],kind='delivery_delay',order_id=o.order_id))

    def demand(self,state,original):
        day=state.day
        baseline=copy.deepcopy(state) if day in self.sample_days else None
        problem=public_problem(self.env,state)
        old_ids={c.contract_id for c in state.contracts}
        original(state)
        actual=new_rentals(state,old_ids,day)
        batches=defaultdict(list)
        for c in actual:batches[c.rental_episode_id or c.contract_id].append(c)
        base_load=loads(problem['book'],day,1);supply=supply_function(problem)
        old_bad={cid for cid,ds in check_book(problem).items() if ds}
        from .common import deficits
        for batch,contracts in batches.items():
            base_load.update(occupied_new(contracts,day,1))
            bad={cid for cid,co in problem['book'].items() if deficits(co,base_load,supply,day,1)}
            self.rows['walk_in_intrusion'].append(dict(day=day,batch_id=batch,contracts=[c.contract_id for c in contracts],
                affected_prior=sorted(bad-old_bad),overlaps_prior=any(c.offer_id in co['units'] for c in contracts for co in problem['book'].values())))
            old_bad=bad
        end_problem=public_problem(self.env,state)
        missing=check_book(end_problem)
        for cid,co in end_problem['book'].items():
            if co['start']<=day:continue
            self.rows['persistent_gaps'].append(dict(day=day,commitment_id=cid,request_id=co['request_id'],
                version=co['version'],pickup_day=co['start'],lead_days=co['start']-day,
                gap=bool(missing[cid]),shortages=missing[cid],offers=sorted(co['units']),method=end_problem['method']))
        if baseline is not None:
            if self.dispatcher is None:
                self.relax(baseline,state,actual,old_ids,original,problem)
            else:
                self.dispatcher.submit(day, baseline, demand_receipt(state,actual))

    def relax(self,baseline,actual_state,actual,old_ids,original,problem):
        day=baseline.day
        prefix=self.cache/'candidates'/f'day-{day}'
        prefix.mkdir(parents=True,exist_ok=True)
        current=normal_walk_contribution(self.env,actual_state,actual)
        available=Counter(a.offer_id for a in baseline.assets if a.status=='available')
        expiry=min(baseline.max_days,day+PARAMETERS['relaxation_window_days'])
        grouped=defaultdict(list)
        for row in baseline.walk_in_capacity_limits:
            if day<=row.day<=expiry:grouped[row.offer_id].append(row)
        allrows=[]
        # No-op verifies exact customer sequence, inventory and rentals at this phase.
        no_op=copy.deepcopy(baseline);original(no_op)
        no_rent=new_rentals(no_op,old_ids,day)
        if ([asdict(c) for c in no_rent] != [asdict(c) for c in actual]
            or any(getattr(no_op,k)!=getattr(actual_state,k) for k in
                   ['cash','deposit_liability','unearned_revenue_liability','cumulative_rental_revenue'])
            or no_op.current_day_ledger.get('counter_interactions')!=actual_state.current_day_ledger.get('counter_interactions')):
            raise RuntimeError(f'no-op demand replay mismatch at day {day}')
        for oid,limits in sorted(grouped.items()):
            cached=prefix/f'{oid}.json'
            if cached.exists():
                record=json.loads(cached.read_text());allrows.append(record);continue
            shadow=copy.deepcopy(baseline)
            items=[dict(action='set',offer_id=oid,from_day=r.day,to_day=r.day,
                        max_walk_in_occupied_units=r.max_walk_in_occupied_units+1) for r in sorted(limits,key=lambda r:r.day)]
            result=self.env.set_walk_in_capacity_limits(shadow,items)
            if not result.ok:raise RuntimeError(f'candidate limit action rejected: {result.data}')
            original(shadow)
            candidates=new_rentals(shadow,old_ids,day)
            value=normal_walk_contribution(self.env,shadow,candidates)
            checks={str(g):relaxation_safe(problem,available,actual,candidates,g) for g in (1,3,7)}
            record=dict(day=day,offer_id=oid,action=items,baseline=current,candidate=value,
                difference=round(value['contribution']-current['contribution'],6),risk_checks=checks,
                candidate_contracts=[dict(offer_id=c.offer_id,start_day=c.start_day,return_day=c.return_day,
                                         rent_paid=c.rent_paid) for c in candidates])
            write_json(cached,record);allrows.append(record)
        self.rows['capacity_relaxation'].append(dict(day=day,baseline=current,candidate_count=len(allrows),
            improvements={str(g):max([0.0]+[r['difference'] for r in allrows if not r['risk_checks'][str(g)]]) for g in (1,3,7)},
            passed_candidates={str(g):sum(not r['risk_checks'][str(g)] for r in allrows) for g in (1,3,7)},candidates=allrows))


def summarize(recorder,state,horizon):
    rows=recorder.rows
    walk=rows['walk_in_intrusion'];risk=rows['persistent_gaps'];opp=rows['opportunities'];auth=rows['authorization_selection']
    groups=defaultdict(list)
    for r in risk:groups[r['commitment_id']].append(r)
    episodes=[]
    events=[asdict(e) for e in state.reservation_events if e.day<=horizon]
    terminals={e['commitment_id']:e['event_type'] for e in events if e['event_type'] in FAILURES | {'pickup_fulfilled'}}
    for record in walk:
        record['associated_outcomes']={cid:terminals.get(cid,'pending_or_external_cancellation') for cid in record['affected_prior']}
    for cid,rr in groups.items():
        start=None;previous=False;count=0
        for r in rr:
            if r['gap'] and not previous:start=r['day'];count+=1
            if not r['gap'] and previous:
                nearby=[e for e in events if e['day']==r['day'] and
                        (e.get('commitment_id')==cid or 'cancel' in e['event_type'])]
                source_labels=set()
                if any('cancel' in e['event_type'] for e in nearby):source_labels.add('cancellation_in_related_calendar')
                if any(e['event_type'] in {'reschedule_accepted','reservation_contingency_applied','preauthorized_recovery_plan_selected','authorized_fallback_selected'} for e in nearby):source_labels.add('authorized_change')
                oids=set(r['offers'])
                if any(o.placed_day==r['day'] and any(it['offer_id'] in oids for it in o.items) for o in state.orders):
                    source_labels.add('purchase_accepted_not_physical_delivery')
                for e in state.rental_activity:
                    if e['day']==r['day'] and e.get('offer_id') in oids and e['event_type'] in {
                        'asset_acquired','cleaning_completed','repair_completed','maintenance_completed','returned_normal','returned_damaged'}:
                        source_labels.add(e['event_type'])
                episodes.append(dict(commitment_id=cid,start_day=start,end_day=r['day'],duration=r['day']-start,
                    ending='prediction_cleared',related_events=[e['event_type'] for e in nearby],
                    candidate_clearance_sources=sorted(source_labels),resolution_attribution='not_unique',recurrence=count>1))
            previous=r['gap']
        if previous:
            later=next((e for e in events if e.get('commitment_id')==cid and e['day']>rr[-1]['day'] and
                        (e['event_type'] in {'pickup_fulfilled','pickup_failed'} or 'cancel' in e['event_type'])),None)
            end=later['day'] if later else horizon+1
            episodes.append(dict(commitment_id=cid,start_day=start,end_day=end,duration=end-start,
                ending=later['event_type'] if later else 'right_censored',recurrence=count>1))
    opportunities=[r for r in opp if r.get('opportunity')]
    authorized=[r for r in auth if r['opportunity']]
    protection={}
    for tool in sorted({r['tool'] for r in rows['action_protection']}):
        applicable=[r for r in rows['action_protection'] if r['tool']==tool and r['relevant_prior']]
        protection[tool]=ratio(sum(not r['affected_prior'] for r in applicable),len(applicable))
    relax_summary={}
    for g in ('1','3','7'):
        dr=rows['capacity_relaxation']
        relax_summary[g]=dict(mean_daily_improvement=sum(r['improvements'][g] for r in dr)/len(dr) if dr else None,
            positive_days=sum(r['improvements'][g]>=PARAMETERS['positive_improvement_cny'] for r in dr),
            evaluated_days=len(dr),no_candidate_days=sum(r['candidate_count']==0 for r in dr),
            passed_candidates=sum(r['passed_candidates'][g] for r in dr))
    result=dict(status='complete',method='public-readiness-calendar-v1',
        walk_in_intrusion=dict(metric=ratio(sum(bool(r['affected_prior']) for r in walk),len(walk)),records=walk),
        persistent_gaps=dict(metric=ratio(sum(r['gap'] for r in risk),len(risk)),records=risk,episodes=episodes,
            recurring_contracts=len({e['commitment_id'] for e in episodes if e['recurrence']}),
            by_lead={label:ratio(sum(r['gap'] for r in risk if lo<=r['lead_days']<=hi),
                                sum(lo<=r['lead_days']<=hi for r in risk))
                     for label,lo,hi in [('1-2',1,2),('3-7',3,7),('8+',8,10000)]}),
        opportunities=dict(metric=ratio(sum(r['covered'] for r in opportunities),len(opportunities)),records=opp,
            undetermined=sum(r.get('candidate_status')=='undetermined' for r in opp)),
        authorization_selection=dict(metric=ratio(sum(r['selected_feasible'] for r in authorized),len(authorized)),records=auth),
        action_protection=dict(by_action=protection,records=rows['action_protection']),
        capacity_relaxation=dict(by_gap=relax_summary,records=rows['capacity_relaxation']),
        evidence_contact=dict(records=rows['evidence_contact'],interpretation='Structured object exposure only, not comprehension.'),
        public_changes=[r for r in rows['public_changes'] if r['day']<=horizon])
    return result


def object_refs(value):
    """Exact structured identifier values only; no natural-language note parsing."""
    refs=set()
    if isinstance(value,dict):
        for key,child in value.items():
            if key not in {'content','description','reasoning_content'}:refs.update(object_refs(child))
    elif isinstance(value,list):
        for child in value:refs.update(object_refs(child))
    elif isinstance(value,str) and ' ' not in value and '\n' not in value and value.startswith(('offer-','asset-','reservation-','rr-')):
        refs.add(value)
    return refs


def replay(run_dir,through_day,sample_days,cache_dir,cache_key,through_days=None,
           cutoff_keys=None,budget_dir=None,cpu_budget=1,source_root=None,migration_dir=None,
           state_reload='reuse'):
    from .execution import CpuLease, execution_budget
    from .parallel import BranchDispatcher
    with execution_budget(cpu_budget,budget_dir) as (directory,size):
        with CpuLease(directory,size) as lease:
            dispatcher=BranchDispatcher(run_dir,cache_dir,directory,size,source_root,lease)
            try:
                return _replay(run_dir,through_day,sample_days,cache_dir,cache_key,
                               through_days or [through_day],cutoff_keys or {},dispatcher,migration_dir,state_reload)
            finally:
                dispatcher.close()


def _replay(run_dir,through_day,sample_days,cache_dir,cache_key,through_days,cutoff_keys,dispatcher,migration_dir,state_reload):
    path=Path(run_dir);cache=Path(cache_dir)
    manifest,env,persisted=_load_environment(path)
    if manifest.get('controller',{}).get('implementation_identity') != _implementation_identity():
        raise RuntimeError('frozen source identity mismatch; provide matching --source-root')
    cutoffs=sorted(set(through_days))
    def completed(d):
        target=cache/f'extended-day-{d}.json'
        return target.is_file() and json.loads(target.read_text()).get('cache_key')==cutoff_keys.get(str(d),cache_key)
    if all(completed(d) for d in cutoffs):return
    calls=_tool_calls(path,through_day);opening=_opening_observation_days(path,through_day)
    checkpoint=cache/'checkpoint.json.gz'
    if migration_dir and not checkpoint.exists():
        from .migration import import_checkpoint
        import_checkpoint(migration_dir,cache,calls,sample_days,cutoffs)
    if checkpoint.exists():
        restored=read_state(checkpoint)
        closed=len(restored['state']['daily_history'])
        # A newly requested earlier cutoff cannot be reconstructed from a future state.
        if any(d<closed and not completed(d) for d in cutoffs):restored=None
    else:restored=None
    if restored is not None:
        state=WorldState.from_dict(restored['state']);index=restored['next_call'];seen=set(restored['seen'])
        recorder=Recorder(env,cache,sample_days,restored['rows'],dispatcher)
        existing={r['day'] for r in recorder.rows['capacity_relaxation']}
        for d in sample_days:
            if d<=len(state.daily_history) and d not in existing:
                if not (cache/'branches'/f'day-{d}'/'input.json.gz').exists():
                    raise RuntimeError(f'missing branch checkpoint for day {d}')
                dispatcher.enqueue(d)
    else:
        state=env.reset();index=0;seen=set();recorder=Recorder(env,cache,sample_days,dispatcher=dispatcher)
    tools=ToolExecutor(env);effects=tools.effect_map();start=time.monotonic()
    from .state_reuse import QueryStateReuse
    reuse=QueryStateReuse(state_reload,_implementation_identity(),effects)
    timings=defaultdict(float)
    def publish(day, count):
        branches={r['day']:r for r in recorder.rows['capacity_relaxation']}
        branches.update({r['day']:r for r in dispatcher.collect(day)})
        recorder.rows['capacity_relaxation']=[branches[d] for d in sorted(branches)]
        expected=[d for d in sample_days if d<=day]
        if sorted(d for d in branches if d<=day)!=expected:
            raise RuntimeError(f'incomplete branch coverage at cutoff {day}')
        # summarize annotates records: never let future summaries mutate old reports.
        frozen=Recorder(env,cache,[],copy.deepcopy(recorder.rows))
        result=_finish_result(frozen,state,day,manifest,env,count,cutoff_keys.get(str(day),cache_key))
        write_json(cache/f'extended-day-{day}.json',result)
    closed=len(state.daily_history)
    if closed in cutoffs and not completed(closed):publish(closed,index)
    for i in range(index,len(calls)):
        day,name,args,recorded=calls[i]
        # RunStore serializes sorted dictionaries and reloads before every tool.
        # Reproduce that boundary rather than ignoring unordered-looking receipts:
        # mapping iteration can also affect later environment transitions.
        tick=time.monotonic()
        state=reuse.before_call(state)
        timings['state_reload_seconds']+=time.monotonic()-tick
        tick=time.monotonic()
        if day not in seen:
            recorder.day_queries=[];recorder.day_refs=set()
            if day in opening:
                observation=env.observation(state,consume_inbox=True)
                recorder.day_refs.update(object_refs(observation))
                reuse.invalidate('morning_observation')
            seen.add(day)
        if _recorded_controller_no_op(recorded):continue
        row_lengths={k:len(v) for k,v in recorder.rows.items()}
        before=None
        # Rejected calls can contain arbitrary malformed arguments. Diagnostic
        # projection assumes an executable action; its rows were always rolled
        # back on rejection below. Do not parse that rejected payload first.
        # Still execute EVERY call and compare its receipt, so a false recorded
        # rejection cannot bypass replay integrity checks.
        if (recorded.get('ok') is True and effects.get(name)!='read_only'
                and name not in {'memory_write','close_business_day'}):
            before=recorder.mutation_before(state,name,args)
        reuse.before_tool(state,name,args)
        if name=='close_business_day':
            problem=public_problem(env,state)
            for q in state.reservation_requests:
                if q.status in {'pending_quote','pending_service_offer'} and q.expires_day<=day:
                    row=inspect_request(q,dict(action='decline'),env.reservation_promise_catalog,problem)
                    row['origin']='unhandled_at_expiry'
                    recorder.rows['opportunities'].append(row)
                if q.status=='awaiting_manager_confirmation':
                    recorder.authorization(q,dict(plan_id=q.customer_preferred_plan_id,
                        pickup_day=q.original_inquiry_pickup_day or q.pickup_day),problem,'automatic_default')
            original=env._simulate_day
            def capture(s):return recorder.demand(s,original)
            env._simulate_day=capture
            try:actual=tools.execute(state,name,args)
            finally:del env.__dict__['_simulate_day']
            if state.day>day:
                recorder.observe_world(problem,public_problem(env,state),state)
        else:
            actual=tools.execute(state,name,args)
        reuse.after_tool(state,name,args,actual.ok)
        normalized=dict(ok=actual.ok,data=actual.data,terminal_day=actual.terminal_day)
        if _normalized_replay_result(normalized)!=_normalized_replay_result(recorded):
            write_json(cache/'mismatch.json',dict(call_index=i,day=day,tool=name,expected=recorded,actual=normalized))
            raise RuntimeError(f'replay mismatch at day {day}, call {i}, tool {name}')
        if not actual.ok:
            for key,n in row_lengths.items():del recorder.rows[key][n:]
        elif before is not None:
            recorder.mutation_after(state,name,args,before,True)
        if effects.get(name)=='read_only' and actual.ok:
            recorder.day_queries.append(dict(tool=name,arguments=args))
            recorder.day_refs.update(object_refs(actual.data))
        elif before is not None and actual.ok:
            referenced=object_refs(args)
            prior_seen=referenced & recorder.day_refs
            recorder.rows['evidence_contact'].append(dict(day=day,tool=name,
                query_count=len(recorder.day_queries),query_tools=sorted({q['tool'] for q in recorder.day_queries}),
                referenced_objects=sorted(referenced),previously_exposed_objects=sorted(prior_seen),
                status='structured_object_contact_not_evidence_completeness'))
            recorder.day_refs.update(object_refs(actual.data))
        if name=='close_business_day' and actual.ok:
            timings['execution_and_metrics_seconds']+=time.monotonic()-tick
            tick=time.monotonic()
            write_state(checkpoint,dict(state=state.to_dict(),next_call=i+1,seen=sorted(seen),rows=recorder.rows))
            timings['checkpoint_seconds']+=time.monotonic()-tick
            write_json(cache/'progress.json',dict(completed_day=day,through_day=through_day,
                elapsed_seconds=time.monotonic()-start,branches=dispatcher.stats(),timings=dict(timings),
                cpu_budget=dispatcher.cpu_budget,
                active_cpu_slots=occupied_cpu_slots(dispatcher.budget_dir,dispatcher.cpu_budget),
                resumed_from_day=closed,state_reuse=reuse.stats()))
            if day in cutoffs and not completed(day):publish(day,i+1)
        else:
            timings['execution_and_metrics_seconds']+=time.monotonic()-tick
    if len(state.daily_history)!=through_day:raise RuntimeError('replayed closing-day count mismatch')
    write_json(cache/'timing.json',dict(elapsed_seconds=time.monotonic()-start,timings=dict(timings),
                                      resumed_from_day=closed,branches=dispatcher.stats(),state_reuse=reuse.stats()))


def _finish_result(recorder,state,through_day,manifest,env,call_count,cache_key):
    result=summarize(recorder,state,through_day)
    # Quick cancellation labels use the replayed event stream, not future-mutated fields.
    from .basic import basic_metrics
    profiles={p.profile_id:p.to_dict() for p in env.reservation_promise_catalog.profiles}
    quick=basic_metrics(state.to_dict(),[],through_day,manifest,profiles)['quick_cancellations']['records']
    for r in quick:
        co=next(c for c in state.reservation_commitments if c.commitment_id==r['commitment_id'])
        oids={l.selected_offer_id or l.offer_id for l in co.lines}
        r['observed_external_events']=[e for e in recorder.rows['public_changes'] if
            r['confirmed_day']<e['day']<=r['cancel_day'] and e['offer_id'] in oids]
        r['external_classification']='observed_related_adverse' if r['observed_external_events'] else 'none_detected'
        r['own_resource_changes']=[e for e in recorder.rows['action_protection'] if
            r['confirmed_day']<=e['day']<=r['cancel_day'] and oids & set(e['changed_offers'])]
    result['quick_cancellations']=quick
    result['cache_key']=cache_key;result['replay_source_identity']=_implementation_identity()
    result['replay_calls']=call_count;result['replay_mismatches']=0
    return result
