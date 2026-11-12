"""Frozen V50 D2 rental-arrival profile, applied after V49 normalization."""

ENVIRONMENT_PROTOCOL = 'outfitter-seasonal-v50-summer-scheduling-pressure'
CORE_RELEASE_ID = 'outfitter-v50-summer-scheduling-pressure-core-r1'
TOOL_INTERFACE_VERSION = 47
POLICY_ID = 'summer-rental-arrival-pressure-d2-v1'
RENTAL_ARCHETYPES = ('overnight', 'picnic', 'lightweight')
MULTIPLIERS = {'school_mon_thu': 3.0, 'school_friday': 2.0,
               'school_weekend_or_public_holiday': 1.30}


def arrival_multiplier(calendar, archetype):
    if archetype not in RENTAL_ARCHETYPES or not calendar.get('is_school_holiday', False):
        return 1.0
    if calendar['is_weekend'] or calendar['day_type'] == 'public_holiday':
        return MULTIPLIERS['school_weekend_or_public_holiday']
    if calendar['weekday'] == 'Friday':
        return MULTIPLIERS['school_friday']
    return MULTIPLIERS['school_mon_thu']


def policy_document():
    return {'format': POLICY_ID, 'profile': 'D2', 'multipliers': dict(MULTIPLIERS),
            'archetypes': list(RENTAL_ARCHETYPES), 'date_basis': 'original_target_trip_date',
            'stage': 'after_v49_calendar_normalization_before_channel_and_weather',
            'renormalize_after_multiplier': False, 'retail_supply_unchanged': True,
            'outside_school_holiday_unchanged': True}
