"""V49-only school-calendar behavior; legacy worlds delegate unchanged."""
from ..school_holiday import (POLICY_ID, PUBLIC_RULES, arrival_base,
                             calendar_with_school, child_probability, rental_duration)
from .common import _ARCHETYPE_INDEX, _RNG_PARTY_COMPOSITION, _RNG_CUSTOMER_BASKET


class SchoolHolidayMixin:
    def _v49_school_holiday(self):
        return (self.dataset.metadata.get('school_holiday_policy', {}).get('format') == POLICY_ID
                and int(self.dataset.metadata.get('tool_interface_version', 0)) >= 46)

    def reset(self):
        state = super().reset()
        if self._v49_school_holiday():
            state.calendar_demand_protocol = self._calendar_demand_state_protocol()
        return state

    def _calendar_demand_state_protocol(self):
        return POLICY_ID if self._v49_school_holiday() else None

    def _v47_assert_state_version(self, state):
        super()._v47_assert_state_version(state)
        expected = self._calendar_demand_state_protocol()
        if state.calendar_demand_protocol != expected:
            raise ValueError('Calendar demand version mismatch: start a fresh state for this version')

    def close_business_day(self, state, **kwargs):
        if self._v47_autonomous_rentals():
            self._v47_assert_state_version(state)
        return super().close_business_day(state, **kwargs)

    def _calendar(self, day):
        result = super()._calendar(day)
        return calendar_with_school(result) if self._v49_school_holiday() else result

    def _public_calendar(self, day):
        result = super()._public_calendar(day)
        if self._v49_school_holiday():
            result['is_school_holiday'] = self._calendar(day)['is_school_holiday']
        return result

    def _calendar_arrival_lambda(self, archetype, calendar):
        if not self._v49_school_holiday():
            return super()._calendar_arrival_lambda(archetype, calendar)
        return arrival_base(archetype, calendar, self.rules)

    def _v49_party_shape(self, day, archetype, index):
        address = (day, _ARCHETYPE_INDEX[archetype], index)
        rng = self._event_rng(_RNG_PARTY_COMPOSITION, *address)
        adults = int(rng.integers(*{'overnight': (1, 4), 'picnic': (2, 5),
                                   'lightweight': (1, 2), 'supply': (1, 4)}[archetype]))
        chance = child_probability(self._calendar(day))
        children = int(rng.integers(1, 3)) if archetype != 'lightweight' and rng.random() < chance else 0
        return {'adults': adults, 'children': children}

    def _v47_customer_shape(self, pickup_day, archetype, index):
        if not self._v49_school_holiday():
            return super()._v47_customer_shape(pickup_day, archetype, index)
        address = (pickup_day, _ARCHETYPE_INDEX[archetype], index)
        basket = self._event_rng(_RNG_CUSTOMER_BASKET, *address)
        return {**self._v49_party_shape(pickup_day, archetype, index),
                'duration': rental_duration(archetype, self._calendar(pickup_day), basket),
                'basecamp': archetype == 'overnight' and self._v47_rng(address, 'basecamp').random() < .30}

    def business_manual(self):
        manual = super().business_manual()
        if self._v49_school_holiday():
            manual['reservations']['rules'].append(PUBLIC_RULES)
        return manual
