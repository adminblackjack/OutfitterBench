"""V48 public contractual guarantee; currency arithmetic has one definition."""
from decimal import Decimal, ROUND_HALF_UP

ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v48-service-guarantee"
CORE_RELEASE_ID = "outfitter-v48-service-guarantee-core-r1"
TOOL_INTERFACE_VERSION = 45
POLICY_ID = "locked-first-day-service-guarantee-v1"
RULE = (
    "At confirmation the original whole-line first-day guarantee bases are locked. "
    "Ordinary quotes multiply first-day per-asset price by promised asset quantity; fixed offers "
    "use their whole-line first-day amount without multiplying quantity again. "
    "Additional compensation = affected original first-day basis * "
    "(1 - 0.75 * min(max(days_until_current_pickup, 0) / 7, 1)), rounded half-up to cents. "
    "Whole-order nonfulfillment covers the entire original guarantee; customer-agreed partial "
    "pickup covers only the missing original lines. Unprovided rent is refunded or credited separately. "
    "Same-day manager cancellation, pickup failure and merchant-induced cancellation have the same "
    "guarantee. Normal customer cancellation has no merchant compensation. "
    "Posted-price changes, rescheduling, replacements, recovery discounts and version increments "
    "never reduce the original guarantee. Free added recovery items add no guarantee basis. "
    "Consent to reduced pickup is required; inquiry/selection does not pay compensation. "
    "Payment occurs on actual reduced delivery. Compensation already assessed for the same "
    "affected obligation is not assessed twice. Successful agreed recovery is not whole-order failure."
)


def cents(value):
    return int((Decimal(str(value))*100).quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def line_basis(line):
    if line.locked_line_daily_rent is not None:
        return cents(line.locked_line_daily_rent)
    return cents(line.locked_daily_price)*int(line.asset_quantity or line.quantity)


def allocate_targets(bases, notice_days):
    """Round combined guarantee once, distributing remainder by stable line ID."""
    factor=Decimal(1)-Decimal('0.75')*Decimal(min(max(int(notice_days),0),7))/7
    exact={k:Decimal(v)*factor for k,v in bases.items()}
    target=int(sum(exact.values(),Decimal(0)).quantize(Decimal(1),rounding=ROUND_HALF_UP))
    result={k:int(v) for k,v in exact.items()}
    for k in sorted(exact,key=lambda k:(-(exact[k]-result[k]),k))[:target-sum(result.values())]:
        result[k]+=1
    return result
