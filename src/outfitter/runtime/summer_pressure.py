"""Native V50 D2 arrivals; old versions retain their original behavior."""
from ..summer_pressure import POLICY_ID, arrival_multiplier


class SummerPressureMixin:
    def _v50_summer_pressure(self):
        return (self.dataset.metadata.get('summer_pressure_policy', {}).get('format') == POLICY_ID
                and int(self.dataset.metadata.get('tool_interface_version', 0)) >= 47)

    def _calendar_demand_state_protocol(self):
        if self._v50_summer_pressure():
            return POLICY_ID
        return super()._calendar_demand_state_protocol()

    def _v47_arrival_rate(self, day, kind):
        rate = super()._v47_arrival_rate(day, kind)
        if self._v50_summer_pressure():
            rate *= arrival_multiplier(self._calendar(day), kind)
        return rate
