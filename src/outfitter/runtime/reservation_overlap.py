"""Native candidate mixin; old runtime behavior delegates unchanged."""
from __future__ import annotations

import copy
from dataclasses import replace

from ..reservation_overlap import (ENVIRONMENT_PROTOCOL, POLICY_ID, PRIVATE_CONTEXT, STREAM,
    TYPES, VALUE_FACTORS, LEAD_WEIGHTS, MARKET_FACTORS, normalized_block, policy_document)
from ..autonomous_rentals import LEAD_BANDS
from ..resale_disclosure import ENVIRONMENT_PROTOCOL as V52_ENVIRONMENT_PROTOCOL, enabled as resale_disclosure_enabled
from ..operating_history import ENVIRONMENT_PROTOCOL as V53_ENVIRONMENT_PROTOCOL, enabled as operating_history_enabled
from ..recovery_state import ENVIRONMENT_PROTOCOL as V54_ENVIRONMENT_PROTOCOL, enabled as recovery_state_enabled
from .common import _ARCHETYPE_INDEX, _RNG_RESERVATION_ROUTE


class _MarketFactorDraw:
    """Only substitutes the candidate factor distribution, not customer choice."""
    def __init__(self, rng, factors):
        self.rng, self.factors = rng, factors

    def choice(self, original_factors):
        if tuple(original_factors) != (.8, 1., 1.2):
            raise ValueError('unexpected inherited market-factor distribution')
        return self.rng.choice(self.factors)


class ReservationOverlapMixin:
    def _initialize_overlap_candidate(self, dataset):
        enabled = dataset.metadata.get('environment_protocol') in {ENVIRONMENT_PROTOCOL, V52_ENVIRONMENT_PROTOCOL, V53_ENVIRONMENT_PROTOCOL, V54_ENVIRONMENT_PROTOCOL}
        policy = dataset.metadata.get('reservation_overlap_policy')
        if enabled != (policy is not None):
            raise ValueError('overlap candidate protocol/policy mismatch')
        if enabled and policy != policy_document():
            raise ValueError('overlap candidate parameters differ from frozen r1')
        self._overlap_enabled = enabled
        self._overlap_rate_blocks = {}
        self._overlap_customer_cache = {}

    def _calendar_demand_state_protocol(self):
        if self._overlap_enabled and recovery_state_enabled(self.dataset.metadata):
            return POLICY_ID + '-recovery-r2-resale-disclosure-v1-operating-history-v1-current-pickup-v1'
        if self._overlap_enabled and operating_history_enabled(self.dataset.metadata):
            return POLICY_ID + '-recovery-r2-resale-disclosure-v1-operating-history-v1'
        if self._overlap_enabled and resale_disclosure_enabled(self.dataset.metadata):
            return POLICY_ID + '-recovery-r2-resale-disclosure-v1'
        if self._overlap_enabled and self._v51_recovery_integrity():
            return POLICY_ID + '-recovery-r2'
        return POLICY_ID if self._overlap_enabled else super()._calendar_demand_state_protocol()

    def _overlap_rates(self, day):
        first = ((day-1)//14)*14+1
        if first not in self._overlap_rate_blocks:
            days = list(range(first, min(first+14, self.config.max_days+1)))
            base = {d:super(ReservationOverlapMixin,self)._v47_arrival_rate(d,'picnic') for d in days}
            self._overlap_rate_blocks[first] = normalized_block(days,base,self._calendar)
        return self._overlap_rate_blocks[first][day]

    def _v47_arrival_rate(self, day, kind):
        if not self._overlap_enabled or kind != 'picnic':
            return super()._v47_arrival_rate(day,kind)
        return sum(self._overlap_rates(day))

    def _overlap_profile(self, address):
        key = tuple(map(int,address[:3]))
        if key[1] != _ARCHETYPE_INDEX['picnic']:
            return None
        if key not in self._overlap_customer_cache:
            rates = self._overlap_rates(key[0]);total=sum(rates)
            rng = self._stable_event_rng(STREAM,*key,identity='frozen-household-type')
            index = int(rng.choice(3,p=[x/total for x in rates]))
            kind = TYPES[index]
            duration = None
            drng = self._stable_event_rng(STREAM,*key,identity='frozen-duration')
            if kind == 'multi_day_camp':duration=int(drng.choice([3,4,5],p=[.25,.50,.25]))
            elif kind == 'short_activity':duration=int(drng.choice([1,2],p=[.85,.15]))
            self._overlap_customer_cache[key] = {'format':POLICY_ID,'origin_address':list(key),
                'type':kind,'duration_override':duration,'budget_factor':VALUE_FACTORS[index][0],
                'importance_factor':VALUE_FACTORS[index][1]}
        return copy.deepcopy(self._overlap_customer_cache[key])

    def _v47_customer_shape(self,pickup_day,archetype,index):
        shape=super()._v47_customer_shape(pickup_day,archetype,index)
        if self._overlap_enabled and archetype=='picnic':
            profile=self._overlap_profile((pickup_day,_ARCHETYPE_INDEX[archetype],index))
            if profile['duration_override'] is not None:
                shape={**shape,'duration':profile['duration_override']}
        return shape

    def _reservation_route(self,pickup_day,archetype_index,customer_index):
        if (not self._overlap_enabled or archetype_index!=_ARCHETYPE_INDEX['picnic']
                or not self.config.reservations_enabled):
            return super()._reservation_route(pickup_day,archetype_index,customer_index)
        address=(pickup_day,archetype_index,customer_index)
        kind=self._overlap_profile(address)['type']
        if kind=='ordinary_picnic':return super()._reservation_route(*address)
        weights=LEAD_WEIGHTS[kind]
        pairs=[(day,weights[i]/(hi-lo+1)) for i,(lo,hi) in enumerate(LEAD_BANDS)
               for day in range(lo,hi+1) if self.config.reservation_booking_lead_min_days<=day<=self.config.reservation_booking_lead_max_days]
        if not pairs:raise ValueError('no valid booking lead')
        total=sum(w for _,w in pairs)
        lead=int(self._v47_rng(address,'booking-lead').choice([d for d,_ in pairs],p=[w/total for _,w in pairs]))
        selected=self._event_rng(_RNG_RESERVATION_ROUTE,*address).random()<self.config.reservation_share
        return bool(selected and pickup_day-lead>=1),lead

    def _make_choice_session(self,state,**kwargs):
        session=super()._make_choice_session(state,**kwargs)
        if not self._overlap_enabled or session.archetype!='picnic':return session
        profile=self._overlap_profile(session.event_address)
        return replace(session,request_budget_multiplier=session.request_budget_multiplier*profile['budget_factor'],
                       occasion_importance=session.occasion_importance*profile['importance_factor'])

    def _frozen_quote_context(self,session):
        result=super()._frozen_quote_context(session)
        if self._overlap_enabled and session.archetype=='picnic':
            result[PRIVATE_CONTEXT]=self._overlap_profile(session.event_address)
        return result

    @staticmethod
    def _restore_quote_context(value):
        from .reservations import ReservationMixin
        clean=dict(value);clean.pop(PRIVATE_CONTEXT,None)
        return ReservationMixin._restore_quote_context(clean)

    def _service_rng(self,request,identity):
        rng=super()._service_rng(request,identity)
        if not self._overlap_enabled or identity!='market-factor':return rng
        contexts=[c[PRIVATE_CONTEXT] for c in request.quote_context_by_line.values() if PRIVATE_CONTEXT in c]
        if not contexts:return rng
        if any(c!=contexts[0] for c in contexts):raise ValueError('incoherent frozen customer types')
        return _MarketFactorDraw(rng,MARKET_FACTORS[contexts[0]['type']])

    def _v47_register(self,state,contracts,session,*,commitment=None):
        result=super()._v47_register(state,contracts,session,commitment=commitment)
        if self._overlap_enabled and session.archetype=='picnic' and contracts:
            episode=state.rental_episodes[contracts[0].rental_episode_id]
            if commitment:
                request=next(r for r in state.reservation_requests if r.request_id==commitment.request_id)
                contexts=[c[PRIVATE_CONTEXT] for c in request.quote_context_by_line.values() if PRIVATE_CONTEXT in c]
                profile=contexts[0]
            else:profile=self._overlap_profile(session.event_address)
            episode['private'][PRIVATE_CONTEXT]=copy.deepcopy(profile)
        return result
