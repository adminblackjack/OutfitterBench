"""V47 frozen customer dynamics. These probabilities are not public guarantees."""
from __future__ import annotations

import math
from statistics import NormalDist

ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v47-autonomous-rentals"
TOOL_INTERFACE_VERSION = 44
CORE_RELEASE_ID = "outfitter-v47-autonomous-rental-core-r3"
POLICY_ID = "autonomous-occupancy-joint-demand-bounded-budget-v1"
STREAM = 47001
MAX_EXTRA_DAYS = 4
BASE_CONTINUATION = {"overnight": .28, "picnic": .12, "lightweight": .20}
LEAD_BANDS = ((1,3),(4,7),(8,14))
LEAD_WEIGHTS = {"overnight":(.15,.35,.50),"picnic_group":(.20,.35,.45),
                "picnic":(.55,.30,.15),"lightweight":(.60,.25,.15)}
NOTICE_MODES = ("early","due","overdue","silent")
NOTICE_WEIGHTS = (.30,.20,.20,.30)


def leisure(calendar):
    return bool(calendar["is_weekend"] or calendar.get('is_school_holiday', False)
                or calendar["day_type"] in {"public_holiday","school_holiday"})


def continuation_probability(*, archetype, flexibility, condition, calendar, next_calendar,
                             daily_rent, cash_scale, extra_days):
    if extra_days >= MAX_EXTRA_DAYS:
        return 0.
    if daily_rent < 0 or cash_scale <= 0 or extra_days < 0:
        raise ValueError("invalid continuation inputs")
    voluntary = (BASE_CONTINUATION[archetype] + .10*(flexibility-.55)
                 + .08*leisure(calendar) + .04*leisure(next_calendar)
                 + (.05 if condition in {"sunny","cloudy"} else -.05 if condition in {"rainy","snowy"} else 0.))
    voluntary = min(.60,max(0.,voluntary))*math.exp(-daily_rent/cash_scale)*.65**extra_days
    involuntary = (.025 + (.075 if condition=="rainy" else .175 if condition=="snowy" else 0.))*.65**extra_days
    return 1.-(1.-voluntary)*(1.-involuntary)


def truncated_budget(uniform, sigma=.25, lower=.6, upper=1.6):
    if not 0 < lower < upper or sigma <= 0:
        raise ValueError("invalid truncated budget parameters")
    normal=NormalDist(mu=-.5*sigma*sigma,sigma=sigma)
    lo,hi=normal.cdf(math.log(lower)),normal.cdf(math.log(upper))
    return math.exp(normal.inv_cdf(lo+(hi-lo)*min(1-1e-12,max(1e-12,uniform))))


def tail_penalty(cash, cash_scale, coefficient=1.):
    return coefficient*max(0.,cash/cash_scale-2.)**2


PUBLIC_RULES = (
    "Customers may continue using rented equipment without requesting approval, and may give no advance notice. "
    "The same process applies to reservations and walk-in rentals. You cannot force a return or veto continued use. "
    "There are no late fines: actual extra days incur the locked ordinary additional-day rent, collected on return. "
    "Deposits remain refundable liabilities until physical return. Rental episodes group equipment from one trip. "
    "contractual_return_day/return_day records the original agreed date; estimated_return_day is an optional "
    "customer report, not a guarantee. A report may change or be missed; silence does not confirm punctual return. "
    "Actual return dates appear only after return. Return reports do not amend other commitments or contract versions. "
    "Use view_active_rentals for current episodes and today's return updates. Equipment still out is unavailable now; "
    "future availability must be judged using current evidence. respond_reservation_changes handles pre-pickup "
    "rescheduling only; there are no active-trip extension approvals."
)
