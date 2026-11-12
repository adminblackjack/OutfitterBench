"""Frozen candidate: same expected picnic pool, overlapping resource uses."""
from __future__ import annotations

import copy
import math

POLICY_ID = 'reservation-overlap-v51'
ENVIRONMENT_PROTOCOL = 'outfitter-seasonal-v51-reservation-overlap'
CORE_RELEASE_ID = 'outfitter-v51-reservation-overlap-core-r2'
STREAM = 51001
PRIVATE_CONTEXT = 'private_overlap_context'
TYPES = ('ordinary_picnic', 'multi_day_camp', 'short_activity')
SHARES = (.60, .25, .15)
VALUE_FACTORS = ((1., 1.), (.90, 1.), (1.35, 1.20))
LEAD_WEIGHTS = {'multi_day_camp': (.15, .35, .50), 'short_activity': (.55, .30, .15)}
MARKET_FACTORS = {'ordinary_picnic': (.8, 1., 1.2), 'multi_day_camp': (.8, 1., 1.2),
                  'short_activity': (1.2, 1.5, 1.8)}


def is_peak(calendar):
    return bool(calendar['is_weekend'] or calendar['day_type'] == 'public_holiday')


def date_weights(day, calendar):
    peak = is_peak(calendar(day))
    before = not peak and any(is_peak(calendar(day+i)) for i in (1, 2))
    return (2. if peak else 1., 2. if before else 1., 3. if peak else 1.)


def normalized_block(days, base_rates, calendar):
    """No weather, inventory, realized arrivals, or agent state input."""
    total = sum(base_rates[d] for d in days)
    if any(not math.isfinite(base_rates[d]) or base_rates[d] < 0 for d in days):
        raise ValueError('invalid base arrival rate')
    weights = {d:date_weights(d, calendar) for d in days}
    denominators = [sum(base_rates[d]*weights[d][i] for d in days) for i in range(3)]
    return {d:tuple(SHARES[i]*total*base_rates[d]*weights[d][i]/denominators[i]
                    if denominators[i] else 0. for i in range(3)) for d in days}


def policy_document():
    return copy.deepcopy({'format':POLICY_ID, 'block_days':14, 'type_names':list(TYPES),
        'shares':list(SHARES), 'type_value_factors':[list(v) for v in VALUE_FACTORS],
        'lead_weights':{k:list(v) for k,v in LEAD_WEIGHTS.items()},
        'market_factors':{k:list(v) for k,v in MARKET_FACTORS.items()},
        'durations':{'multi_day_camp':{'days':[3,4,5],'p':[.25,.50,.25]},
                     'short_activity':{'days':[1,2],'p':[.85,.15]}},
        'calendar_weights':{'ordinary_peak':2.,'camp_nonpeak_before_peak':2.,'short_peak':3.,'other':1.},
        'scope':'picnic only; both channels; normal party/bundle rules unchanged',
        'normalization':'preserve V50 expected potential picnic count within each 14-day block',
        'customer_signal':'existing dates, durations, quantities and quote money only',
        'effective_budget':'base truncated multiplier times frozen type factor; not clipped to base bounds',
        'returns':'unchanged formula; changed budget/rent/duration propagate coherently',
        'not_empirically_calibrated':True})
