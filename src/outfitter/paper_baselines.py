"""Fixed public-tool C/U2 references for the V54 paper, not an optimizer.

No state, dataset, evaluator, customer-model or future-event access here.
Recreate the merchant every day; inherited fixed parameters are deliberate.
"""
from __future__ import annotations

from collections import Counter
import copy
from itertools import product

from .autonomous_controls import AutonomousControl
from .scripted_merchant import job

PROTOCOL = 'v54-paper-public-calendar-r1'
MODES = ('C', 'U2')


class PaperCalendar(AutonomousControl):
    def __init__(self, mode, rpc):
        if mode not in MODES:
            raise ValueError(mode)
        super().__init__(mode, rpc)

    def quote_options(self, request):
        # A binding menu must be safe for any customer-authorized plan, but
        # the manager can select ONE date, not necessarily every offered date.
        candidates = []
        for date in request['pickup_date_options']:
            local = {**request, 'pickup_date_options': [date]}
            menus = super().quote_options(local)
            if menus is None:
                continue
            margins = [self.assess(job(request['request_id'], date['pickup_day'],
                       date['scheduled_return_day'], list(lines)))['margin']
                       for lines in product(*menus)]
            candidates.append((min(margins), -date['pickup_day'], menus))
        return max(candidates, key=lambda c: c[:2])[2] if candidates else None

    def _current_risk(self, cid):
        self.load_record([cid])
        self.physical()
        return self.records[cid]['risk']

    def _partial(self, cid, risk):
        actions = risk['recovery_authorization']['available_actions']
        if 'inquire_partial_fulfillment' in actions:
            row = self.call('resolve_reservation_risks', {'actions': [{
                'commitment_id': cid, 'action': 'inquire_partial_fulfillment'}]})['actions'][0]
            self.log('partial_inquiry', commitment_id=cid,
                     customer_accepted=row['customer_accepted'])
            risk = self._current_risk(cid)
            actions = risk['recovery_authorization']['available_actions']
        if 'select_partial_fulfillment' in actions:
            self.call('resolve_reservation_risks', {'actions': [{
                'commitment_id': cid, 'action': 'select_partial_fulfillment'}]})
            self._current_risk(cid)
            return True
        return False

    def risk(self, alert):
        cid = alert['commitment_id']
        risk = self._current_risk(cid)
        if self._partial(cid, risk):
            return
        risk = self.records[cid]['risk']
        authorization = risk['recovery_authorization']
        data = {}
        if authorization['fallback_inquiry_day'] is None and 'inquire_fallback' in authorization['available_actions']:
            data = self.call('resolve_reservation_risks', {'actions': [{
                'commitment_id': cid, 'action': 'inquire_fallback'}]})['actions'][0]
            risk = self._current_risk(cid)
        terms = self.records[cid]['current_terms']
        req = {'request_id': self.records[cid]['request_id'], 'pickup_date_options': [{
            'pickup_day': terms['pickup_day'], 'scheduled_return_day': terms['return_day']}]}
        allowed = risk['recovery_authorization']['available_actions']
        choice = (self.choose(req, data.get('preauthorized_recovery_plans', []), exclude=cid)
                  if 'select_preauthorized_plan' in allowed else None)
        if choice and choice['accept']:
            self.call('resolve_reservation_risks', {'actions': [{
                'commitment_id': cid, 'action': 'select_preauthorized_plan', 'plan_id': choice['plan_id']}]})
            risk = self._current_risk(cid)
            self._partial(cid, risk)
            return
        candidate = job(cid, terms['pickup_day'], terms['return_day'], terms['locked_lines'])
        available = Counter(a['offer_id'] for a in self.assets if a['status'] == 'available')
        changes = []
        for line in sorted(candidate['lines'], key=lambda l: (not l['required'], l['line_id'])):
            variants = [line['offer_id']]
            if 'select_authorized_fallback' in allowed:
                variants += [r['offer_id'] for r in data.get('authorized_line_fallbacks', [])
                             if r['line_id'] == line['line_id']]
            selected = max(variants, key=lambda oid: (available[oid], oid))
            if selected != line['offer_id']:
                line['offer_id'] = selected
                changes.append({'commitment_id': cid, 'line_id': line['line_id'],
                                'action': 'select_authorized_fallback', 'offer_id': selected})
            available[selected] -= line['asset_quantity']
        # A required-item substitution may make partial-consent inquiry legal.
        required = copy.deepcopy(candidate)
        required['lines'] = [r for r in required['lines'] if r['required']]
        if self.assess(required, cid)['accept']:
            if changes:
                self.call('resolve_reservation_risks', {'actions': changes})
            risk = self._current_risk(cid)
            if self._partial(cid, risk):
                return
            current = self.records[cid]['current_terms']
            if self.assess(job(cid, current['pickup_day'], current['return_day'],
                               current['locked_lines']), cid)['accept']:
                return
        self.cancel(cid, 'no_feasible_authorized_recovery')
