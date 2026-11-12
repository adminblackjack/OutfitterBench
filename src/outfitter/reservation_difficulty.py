"""Versioned V45 rental mechanics shared by the main and retail tracks."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v45-reservation-difficulty"
TOOL_INTERFACE_VERSION = 40
CORE_RELEASE_ID = "outfitter-v45-reservation-difficulty-core"
POLICY_ID = "coherent-trip-partial-fulfillment-and-first-day-tariff-v1"
ASSET_PRICING_BASIS = "asset_first_day_half_additional"
LINE_PRICING_BASIS = "line_first_day_half_additional"
TARIFF_RULE = (
    "Posted walk-in rental prices and the quotation field daily_unit_price are first-day "
    "prices per asset. For first-day price P, quantity q and duration d, total rent is "
    "q * (P + (d - 1) * round_half_up_to_cents(0.5 * P)). "
    "For fixed-reward offers, P is instead the whole-line first-day rent and q is 1; "
    "round the additional-day amount per whole line, not per asset. "
    "The historical field names daily_unit_price, locked_daily_price and quoted_daily_price "
    "denote first-day per-asset prices; line_daily_rent, line_daily_rents and "
    "locked_line_daily_rent denote whole-line first-day prices. "
    "Locked rent is the complete term amount. Extensions disclose the additional amount "
    "due under the locked terms. A disclosed recovery credit may reduce the payable total."
)


def rental_amount(first_day_price: float, duration_days: int, quantity: int = 1) -> float:
    """First day at the posted price; each further day at half, cent rounded."""
    if first_day_price <= 0 or duration_days < 1 or quantity < 1:
        raise ValueError("positive price, duration and quantity required")
    first = Decimal(str(first_day_price))
    later = (first * Decimal("0.50")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float((first + later * (duration_days - 1)) * quantity)


def extension_amount(first_day_price: float, old_days: int, new_days: int, quantity: int = 1) -> float:
    if new_days <= old_days:
        raise ValueError("extension must increase duration")
    return round(rental_amount(first_day_price, new_days, quantity)
                 - rental_amount(first_day_price, old_days, quantity), 2)
