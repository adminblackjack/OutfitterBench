"""V46 trip-level return process, shared by the rental and retail tracks.

Probability tables are evaluator/population knowledge, not actor-facing terms.
No shop inventory or future weather is an input to these customer draws.
"""
from __future__ import annotations

import math

ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v46-trip-return-dynamics"
TOOL_INTERFACE_VERSION = 41
CORE_RELEASE_ID = "outfitter-v46-trip-return-dynamics-core"
POLICY_ID = "independent-trip-delay-and-priced-extension-v1"
RNG_STREAM = 46001
ITINERARIES = ("fixed", "adjustable", "open")
LOGISTICS = ("booked_transport", "self_drive", "dependent_transport")
TYPE_WEIGHTS = {"overnight": (.4, .4, .2), "lightweight": (.4, .4, .2), "picnic": (.75, .2, .05)}
EXTENSION_BASE = {"overnight": (.06, .35, .60), "lightweight": (.04, .25, .45), "picnic": (.01, .06, .12)}
EXTENSION_ONE_DAY = {"overnight": .65, "lightweight": .85, "picnic": 1.}
DELAY_WEIGHTS = {"booked_transport": (.94, .055, .005), "self_drive": (.88, .105, .015),
                 "dependent_transport": (.72, .21, .07)}
PUBLIC_RULES = (
    "Requests disclose trip_context.itinerary_type (fixed/adjustable/open) and return_logistics "
    "(booked_transport/self_drive/dependent_transport). These are informative customer descriptions, not guarantees. "
    "At pickup a customer gives an uncertain extension-intent signal. A formal priced extension request may "
    "arrive one day before the original trip's approved return date, at most once per trip. "
    "Return-logistics delays can occur whether or not an extension was requested or accepted; refusing an "
    "extension does not cause a new delay draw. An uncertain return warning arrives two days before the "
    "pickup-agreed return date (or at pickup for short trips), followed by a firm ETA one day before that date. "
    "Warnings and ETAs appear in reservation_inbox and reservation detail/history. They do not change "
    "contractual terms or commitment_version. At commitment level return_day/approved_return_day remain "
    "the agreed deadline; estimated_return_day is operational availability information, not authorization. "
    "For a physical rental contract, return_day is its currently expected physical return date. "
    "actual_return_day is filled only upon return. Late days carry no extra customer rent but occupy equipment "
    "and accrue actual usage, cleaning and supplier-trial costs. All returns remain within the runoff boundary."
)


def trip_kind(lines):
    services = {getattr(line, "service_line", "") for line in lines}
    return "overnight" if "tent" in services else "picnic" if "camp_table" in services else "lightweight"


def delay_weights(logistics, scale=1.):
    if not .75 <= float(scale) <= 1.25:
        raise ValueError("delay scale must be within [0.75,1.25]")
    _, one, two = DELAY_WEIGHTS[logistics]
    return (1. - (one + two)*scale, one*scale, two*scale)


def extension_probability(kind, itinerary, additional_rent, cash_scale):
    if additional_rent < 0 or cash_scale <= 0:
        raise ValueError("extension pricing inputs must be nonnegative with positive cash scale")
    return EXTENSION_BASE[kind][ITINERARIES.index(itinerary)] * math.exp(-additional_rent / cash_scale)
