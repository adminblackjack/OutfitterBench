"""Versioned guarantee, truthful asset service quotes and recovery stages."""
from ..service_guarantee import POLICY_ID, cents, line_basis, allocate_targets
from ..lifecycle_economics import maintenance_terms
from ..models import ToolResult
from ..errors import ReservationRecoveryBatchValidationError


class ServiceGuaranteeMixin:
    def _v54_recovery_state(self):
        from ..recovery_state import enabled
        return enabled(self.dataset.metadata)

    def _current_partial_pickup(self, state, c):
        """Current physical facts, not a reservation or a future promise."""
        if state.done or c.status != 'at_risk' or c.pickup_day != state.day:
            return [], 'not_current_pickup_risk'
        _, required_ready, missing = self._pickup_asset_selection(state, c)
        if not required_ready:
            return [], 'required_equipment_unavailable'
        if not missing:
            return [], 'no_missing_optional_equipment'
        ids = [line.line_id for line in missing]
        if c.partial_fulfillment_inquiry_day is not None:
            return ids, 'inquiry_already_used_for_version'
        return ids, None

    def _partial_pickup_public(self, state, c):
        missing, reason = self._current_partial_pickup(state, c)
        return {'missing_optional_line_ids': missing,
                'partial_inquiry_eligible_now': reason is None,
                'partial_inquiry_ineligible_reason': reason,
                'inquired_missing_optional_line_ids': (
                    list(c.partial_fulfillment_missing_line_ids)
                    if c.partial_fulfillment_inquiry_day is not None else []),
                'inventory_is_not_reserved': True}

    def _v51_recovery_integrity(self):
        return int(self.dataset.metadata.get('tool_interface_version', 0)) >= 49

    def _recovery_plan_retains_commitment(self, commitment, plan):
        """An old quotation alternative cannot erase an accepted obligation."""
        if not self._v51_recovery_integrity():
            return True
        retained = {str(row['line_id']) for row in plan.get('lines', [])
                    if row.get('disposition') != 'omit'}
        return {line.line_id for line in commitment.lines}.issubset(retained)

    def _eligible_preauthorized_recovery_plans(self, commitment):
        return [plan for plan in commitment.preauthorized_recovery_plans
                if self._recovery_plan_retains_commitment(commitment, plan)]

    def _current_fallback_query_status(self, commitment):
        if not self._v51_recovery_integrity():
            return super()._fallback_query_status(commitment)
        if commitment.fallback_inquiry_day is None:
            return 'not_inquired'
        has_candidates = bool(self._eligible_preauthorized_recovery_plans(commitment)) or any(
            commitment.fallback_revealed_offer_ids_by_line.values())
        return 'inquired_with_candidates' if has_candidates else 'inquired_without_candidates'

    def _v48_service_guarantee(self):
        return (self.dataset.metadata.get('service_guarantee_policy')==POLICY_ID
                and int(self.dataset.metadata.get('tool_interface_version',0))>=45)

    def _guarantee(self,c):
        g=c.service_guarantee
        if not isinstance(g,dict) or g.get('policy_id')!=POLICY_ID:
            raise ValueError(f'V48 commitment lacks locked guarantee: {c.commitment_id}')
        bases=g.get('basis_by_line_cents')
        if not isinstance(bases,dict) or any(type(v) is not int or v<0 for v in bases.values()):
            raise ValueError(f'invalid guarantee basis: {c.commitment_id}')
        if sum(bases.values())!=g.get('total_basis_cents'):
            raise ValueError(f'guarantee basis sum mismatch: {c.commitment_id}')
        if not isinstance(g.get('settlements'),list):
            raise ValueError(f'missing guarantee settlement ledger: {c.commitment_id}')
        for s in g['settlements']:
            if (sum(s['due_by_line_cents'].values())!=s['amount_due_cents']
                    or not 0<=s['amount_paid_cents']<=s['amount_due_cents']):
                raise ValueError(f'invalid guarantee settlement: {c.commitment_id}')
        return g

    def _assert_v48_state(self,state):
        if self._v48_service_guarantee():
            for c in state.reservation_commitments:self._guarantee(c)

    def _confirm_factorized_plan(self,state,request,*,plan_id,origin):
        result=super()._confirm_factorized_plan(state,request,plan_id=plan_id,origin=origin)
        if self._v48_service_guarantee():
            c=next(c for c in state.reservation_commitments if c.commitment_id==request.commitment_id)
            bases={l.line_id:line_basis(l) for l in c.lines}
            c.service_guarantee={'policy_id':POLICY_ID,'locked_day':state.day,
                'basis_by_line_cents':bases,'total_basis_cents':sum(bases.values()),'settlements':[]}
            result['service_guarantee']=self._guarantee_public(c)
        return result

    def _guarantee_assessment(self,c,line_ids,notice_days):
        g=self._guarantee(c)
        bases={k:g['basis_by_line_cents'].get(k,0) for k in sorted(set(line_ids))}
        targets=allocate_targets(bases,notice_days)
        previous={k:sum(s['due_by_line_cents'].get(k,0) for s in g['settlements']) for k in bases}
        due={k:max(0,targets[k]-previous[k]) for k in bases}
        return {'basis_cents':sum(bases.values()),'due_by_line_cents':due,'amount_cents':sum(due.values())}

    def _guarantee_public(self,c):
        g=self._guarantee(c)
        due=sum(s['amount_due_cents'] for s in g['settlements'])
        paid=sum(s['amount_paid_cents'] for s in g['settlements'])
        return {'terms':'original-first-day-guarantee','locked_day':g['locked_day'],
            'original_first_day_basis':g['total_basis_cents']/100,
            'original_line_bases':{k:v/100 for k,v in g['basis_by_line_cents'].items()},
            'compensation_assessed':due/100,'compensation_paid':paid/100,
            'compensation_unpaid':(due-paid)/100}

    def service_guarantee_summary(self,state):
        self._assert_v48_state(state)
        records=[s for c in state.reservation_commitments for s in self._guarantee(c)['settlements']]
        due=sum(s['amount_due_cents'] for s in records);paid=sum(s['amount_paid_cents'] for s in records)
        return {'terms':'original-first-day-guarantee','commitment_count':len(state.reservation_commitments),
                'compensation_assessed':due/100,'compensation_paid':paid/100,'compensation_unpaid':(due-paid)/100}

    def _reservation_failure_terms(self,c,*,status,notice_days):
        if not self._v48_service_guarantee():
            return super()._reservation_failure_terms(c,status=status,notice_days=notice_days)
        assessment=self._guarantee_assessment(c,self._guarantee(c)['basis_by_line_cents'],notice_days)
        amount=assessment['amount_cents']/100;refund=cents(c.unearned_revenue)/100
        return {'notice_days':max(0,int(notice_days)),'notice_band':self._reputation_notice_band(max(0,int(notice_days))),
            'current_locked_rent':c.total_locked_rent,'required_service_unit_days':self._required_service_unit_days(c),
            'prepayment_refund':refund,'rent_component':amount,'service_unit_day_floor_component':0.,
            'compensation':amount,'total_cash_outflow':(cents(refund)+assessment['amount_cents'])/100,
            'formula':POLICY_ID,'guarantee_basis':assessment['basis_cents']/100}

    def _partial_fulfillment_terms(self,c,missing):
        if not self._v48_service_guarantee():return super()._partial_fulfillment_terms(c,missing)
        ids=[l.line_id for l in missing];a=self._guarantee_assessment(c,ids,0)
        return {'missing_line_ids':ids,'rent_credit':sum(cents(l.locked_rent) for l in missing)/100,
                'compensation':a['amount_cents']/100,'guarantee_basis':a['basis_cents']/100}

    def _record_guarantee_payment(self,state,c,*,kind,line_ids,notice_days,paid):
        a=self._guarantee_assessment(c,line_ids,notice_days)
        paid_cents=cents(paid)
        if not 0<=paid_cents<=a['amount_cents']:raise AssertionError('guarantee payment exceeds assessment')
        g=self._guarantee(c)
        if a['amount_cents'] or paid_cents:
            g['settlements'].append({'day':state.day,'kind':kind,'line_ids':sorted(set(line_ids)),
                'notice_days':int(notice_days),'due_by_line_cents':a['due_by_line_cents'],
                'amount_due_cents':a['amount_cents'],'amount_paid_cents':paid_cents})
        c.compensation_paid=sum(s['amount_paid_cents'] for s in g['settlements'])/100
        return {'compensation_amount_due':a['amount_cents']/100,'compensation_amount_paid':paid_cents/100,
                'compensation_amount_unpaid':(a['amount_cents']-paid_cents)/100}

    def _fail_reservation_commitment(self,state,c,*,status,cause,notice_days):
        if self._v48_service_guarantee() and c.status not in {'confirmed','at_risk'}:
            raise ValueError('only a live pre-pickup commitment can fail')
        return super()._fail_reservation_commitment(state,c,status=status,cause=cause,notice_days=notice_days)

    def _maintenance_service_terms(self,asset):
        severity=str(self._lifecycle_factor(asset.offer_id)['d_damage']['severity_tier'])
        terms=maintenance_terms(replacement_cost=asset.unit_cost,severity_tier=severity,scales=self.lifecycle_scales)
        return {**terms,'cash_cost':cents(terms['cash_cost'])/100}

    def asset_service_terms(self,state,asset,operation):
        held=self._warehouse_asset_held(asset);reasons=[];cost=None;duration=None;effect=None
        if not held:reasons.append('asset_not_physically_held')
        if state.done:reasons.append('business_closed')
        if operation=='repair':
            if asset.status!='damaged':reasons.append('asset_not_damaged')
            if held and asset.status=='damaged':
                cost=self.repair_cost(asset);duration=2
                effect={'condition_ratio':asset.repair_target_condition_ratio if self.config.lifecycle_economics_enabled else asset.condition_ratio,
                        'lifetime_wear_reset':False}
        elif operation=='maintenance':
            if not self.config.preventive_maintenance_enabled:reasons.append('maintenance_disabled')
            if asset.status not in {'available','maintenance_hold'}:reasons.append('asset_not_available_for_maintenance')
            trial=self._supplier_trial_record_for_asset(state,asset)
            if trial is not None and trial.return_pending:reasons.append('supplier_trial_return_pending')
            opened=(self._lifecycle_maintenance_status(asset)['window_open'] if self.config.lifecycle_economics_enabled
                else asset.next_inspection_due_day is None or state.day>=asset.next_inspection_due_day-self.config.maintenance_due_window_days)
            if not opened:reasons.append('maintenance_window_not_open')
            if held:
                terms=(self._maintenance_service_terms(asset) if self.config.lifecycle_economics_enabled else
                       {'cash_cost':round(max(2.,self.config.maintenance_cost_rate*asset.unit_cost),2),
                        'duration_days':self.config.maintenance_duration_days})
                cost=terms['cash_cost'];duration=terms['duration_days']
                effect={'condition_ratio':asset.condition_ratio,'lifetime_wear_reset':False,'renews_maintenance_protection':True}
        else:raise ValueError(operation)
        operational=not reasons
        affordable=cost is not None and cents(cost)<=cents(state.cash)
        if operational and not affordable:reasons.append('insufficient_cash')
        return {'eligible_now':not reasons,'ineligible_reasons':reasons,'cost':cost,'duration_days':duration,
                'completion_day_if_started_today':state.day+duration if operational and duration is not None else None,
                'affordable_now':affordable,'effect':effect}

    def _service_rejection(self,state,ids,operation):
        by_id={a.asset_id:a for a in state.assets}
        for aid in dict.fromkeys(map(str,ids)):
            a=by_id.get(aid)
            if a is None:return ToolResult(False,{'error':'unknown_asset','asset_id':aid,'state_unchanged':True})
            quote=self.asset_service_terms(state,a,operation)
            if not quote['eligible_now']:
                return ToolResult(False,{'error':'asset_service_unavailable','asset_id':aid,'current_status':a.status,
                    'operation':operation,'terms':quote,'state_unchanged':True})
        return None

    def repair_or_retire(self,state,repairs,retirements):
        repairs,retirements=list(repairs),list(retirements)
        if self._v48_service_guarantee():
            rejected=self._service_rejection(state,repairs,'repair')
            if rejected is not None:return rejected
        result=super().repair_or_retire(state,repairs,retirements)
        if self._v48_service_guarantee() and not result.ok:result.data['state_unchanged']=True
        return result

    def schedule_maintenance(self,state,asset_ids):
        ids=list(asset_ids)
        if self._v48_service_guarantee():
            rejected=self._service_rejection(state,ids,'maintenance')
            if rejected is not None:return rejected
        result=super().schedule_maintenance(state,ids)
        if self._v48_service_guarantee() and not result.ok:result.data['state_unchanged']=True
        return result

    def recovery_authorization(self,state,c):
        if state.done or c.status!='at_risk':stage='not_available'
        elif c.recovery_path or c.partial_fulfillment_selected:stage='selected_pending_pickup'
        else:stage=self._current_fallback_query_status(c)
        actions=[]
        if not state.done and c.status=='at_risk':
            actions.append('inquire_fallback')
            if c.fallback_inquiry_day is not None:
                if c.recovery_path is None and self._eligible_preauthorized_recovery_plans(c):actions.append('select_preauthorized_plan')
                if c.recovery_path in {None,'profile_line_fallback'} and any(c.fallback_revealed_offer_ids_by_line.values()):
                    actions.append('select_authorized_fallback')
            missing = c.partial_fulfillment_missing_line_ids
            eligible = bool(missing) and c.partial_fulfillment_inquiry_day is None
            if self._v54_recovery_state():
                missing, reason = self._current_partial_pickup(state, c)
                eligible = reason is None
            if eligible:actions.append('inquire_partial_fulfillment')
            if missing:
                if c.partial_fulfillment_inquiry_day==state.day and c.partial_fulfillment_accepted and not c.partial_fulfillment_selected:
                    if not self._v54_recovery_state() or missing == c.partial_fulfillment_missing_line_ids:
                        actions.append('select_partial_fulfillment')
            actions.append('cancel_reservation_commitment')
        return {'stage':stage,'commitment_version':c.commitment_version,'fallback_inquiry_day':c.fallback_inquiry_day,
                'selected_path':c.recovery_path,'available_actions':actions,
                'historical_authorization_is_not_current_selection':True,'inventory_feasibility_evaluated':False}

    def reservation_inbox(self,state):
        result=super().reservation_inbox(state)
        if self._v48_service_guarantee():
            cs={c.commitment_id:c for c in state.reservation_commitments}
            for row in result.get('risk_alerts',[]):
                auth=self.recovery_authorization(state,cs[row['commitment_id']])
                row.update(recovery_authorization=auth,available_actions=auth['available_actions'])
                if self._v54_recovery_state():
                    row.update(self._partial_pickup_public(state,cs[row['commitment_id']]))
        return result

    def view_reservation(self,state,*args,**kwargs):
        result=super().view_reservation(state,*args,**kwargs)
        if self._v48_service_guarantee():
            cs={c.commitment_id:c for c in state.reservation_commitments}
            for row in result.get('records',[]):
                c=cs.get(row.get('commitment_id'))
                if c is not None:
                    if 'current_terms' in row:row['current_terms']['service_guarantee']=self._guarantee_public(c)
                    if 'risk' in row:
                        row['risk']['recovery_authorization']=self.recovery_authorization(state,c)
                        if self._v54_recovery_state():
                            row['risk'].pop('partial_fulfillment_missing_line_ids', None)
                            row['risk'].update(self._partial_pickup_public(state,c))
                        if self._v51_recovery_integrity():
                            row['risk']['preauthorized_recovery_plan_count'] = len(self._eligible_preauthorized_recovery_plans(c))
                            row['risk']['fallback_query_status'] = self._current_fallback_query_status(c)
        return result

    def resolve_reservation_risks(self,state,actions):
        try:
            result=super().resolve_reservation_risks(state,actions)
        except ReservationRecoveryBatchValidationError as error:
            if self._v48_service_guarantee():
                ids={str(a.get('commitment_id','')) for a in actions}
                error.context={'recovery_authorizations':[
                    {'commitment_id':c.commitment_id,**self.recovery_authorization(state,c)}
                    for c in state.reservation_commitments if c.commitment_id in ids]}
            raise
        if self._v48_service_guarantee():
            cs={c.commitment_id:c for c in state.reservation_commitments}
            for row in result['actions']:
                if row.get('commitment_id') in cs:
                    row['recovery_authorization']=self.recovery_authorization(state,cs[row['commitment_id']])
        return result
