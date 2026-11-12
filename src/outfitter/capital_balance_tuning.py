"""Frozen V34 identity for the small capital/reservation balance candidate."""
from __future__ import annotations


ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v34-capital-reservation-balance"
TOOL_INTERFACE_VERSION = 29

# These four values define the candidate.  The first two are episode
# configuration controls; the latter two are frozen request-generation
# probabilities.  No price, cost, WTP, compensation, or valuation coefficient
# is changed by this module.
WAREHOUSE_CAPACITY_LIMIT = 190
RESERVATION_SHARE = 0.70
SERVICE_OFFER_ATTEMPT_PROBABILITY = 0.40
DATE_FLEXIBILITY_PROBABILITY = 0.50
