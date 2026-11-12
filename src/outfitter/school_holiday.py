"""V49 orthogonal school-holiday calendar and bounded demand increment."""
from datetime import date

from .v5 import SCHOOL_HOLIDAY_START_2026, SCHOOL_HOLIDAY_END_2026

ENVIRONMENT_PROTOCOL = 'outfitter-seasonal-v49-school-holiday-demand'
CORE_RELEASE_ID = 'outfitter-v49-school-holiday-demand-core-r2'
TOOL_INTERFACE_VERSION = 46
POLICY_ID = 'orthogonal-school-holiday-additive-demand-v2'
MULTIPLIERS = {'overnight': 1.15, 'picnic': 1.18, 'lightweight': 1.0, 'supply': 1.14}
SEASONAL_ANCHORS = ((91,.80),(111,1.15),(135,1.25),(166,1.16),
                    (196,1.22),(227,1.19),(258,1.17),(279,1.09),(288,.81))
PUBLIC_RULES = (
    'is_school_holiday is an independent public calendar attribute and can overlap weekends '
    'or public holidays; day_type remains a display classification. School holidays create '
    'additional potential outing opportunities, especially family trips. Family composition '
    'and holiday rental-duration tendencies also apply on school-holiday weekends. '
    'Forecasts still affect advance requests and observed weather affects walk-in visits; '
    'neither holidays nor forecasts guarantee demand, rental returns, availability or profit.'
)


def policy_document():
    return {'format': POLICY_ID, 'multipliers': dict(MULTIPLIERS),
            'combination': 'date_base_plus_weekday_base_times_school_increment',
            'school_start': SCHOOL_HOLIDAY_START_2026.isoformat(),
            'school_end': SCHOOL_HOLIDAY_END_2026.isoformat(),
            'seasonal_profile': 'mild-summer-autumn-reallocation-r2',
            'seasonal_anchors': [list(row) for row in SEASONAL_ANCHORS],
            'overlap_child_probability': 'max', 'overlap_date_value': 'max'}


def calendar_with_school(calendar):
    result = dict(calendar)
    value = date.fromisoformat(str(calendar['date']))
    result['is_school_holiday'] = SCHOOL_HOLIDAY_START_2026 <= value <= SCHOOL_HOLIDAY_END_2026
    return result


def classes(calendar):
    values = {str(calendar['day_type'])}
    if calendar.get('is_weekend'):
        values.add('weekend')
    if calendar.get('is_school_holiday'):
        values.add('school_holiday')
    return values


def child_probability(calendar):
    probabilities = {'weekday': .08, 'weekend': .15, 'public_holiday': .35, 'school_holiday': .45}
    return max(probabilities[k] for k in classes(calendar))


def arrival_base(kind, calendar, rules):
    weekday, weekend = rules.archetype_base_lambdas[kind]
    if calendar['day_type'] == 'public_holiday':
        base = weekday * rules.holiday_arrival_multipliers[calendar['holiday_id']]
    elif calendar['is_weekend']:
        base = weekend
    else:
        base = weekday
    if calendar.get('is_school_holiday'):
        base += weekday * (MULTIPLIERS[kind] - 1)
    return base


def rental_duration(kind, calendar, rng):
    labels = classes(calendar)
    if kind == 'overnight':
        if labels & {'school_holiday', 'public_holiday'}:
            return int(rng.choice([2, 3, 4, 5], p=[.12, .32, .36, .20]))
        return int(rng.choice([1, 2, 3, 4], p=[.22, .40, .28, .10]))
    if kind == 'picnic':
        return int(rng.choice([1, 2], p=[.82, .18])) if labels & {
            'school_holiday', 'public_holiday', 'weekend'} else 1
    if kind == 'lightweight':
        return int(rng.choice([1, 2, 3], p=[.58, .34, .08]))
    raise ValueError(f'Not a rental archetype: {kind}')
