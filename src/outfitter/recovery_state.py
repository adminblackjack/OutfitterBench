"""V54 current-pickup facts and portable query interface contract."""
ENVIRONMENT_PROTOCOL = "outfitter-seasonal-v54-recovery-state"
CORE_RELEASE_ID = "outfitter-v54-recovery-state-core-r1"
TOOL_INTERFACE_VERSION = 52
POLICY_ID = "current-pickup-partial-consent-v1"


def enabled(metadata):
    return (metadata.get("environment_protocol") == ENVIRONMENT_PROTOCOL
            and metadata.get("recovery_state_policy") == POLICY_ID
            and metadata.get("tool_interface_version") == TOOL_INTERFACE_VERSION)


def query_parameters(name, original):
    """Portable presentation; conditional legality remains executor-validated."""
    import copy
    if name not in {'view_reservation', 'view_reservation_quote_workspace'}:
        return original
    properties = {}
    for branch in original['oneOf']:
        for key, value in branch['properties'].items():
            if key not in properties:
                properties[key] = copy.deepcopy(value)
                continue
            merged = properties[key]
            if 'enum' in value:
                merged['enum'] = sorted(set(merged['enum']) | set(value['enum']))
            for bound in ('maxItems', 'maximum'):
                if bound in merged and bound in value:
                    merged[bound] = max(merged[bound], value[bound])
            for bound in ('minItems', 'minimum'):
                if bound not in value:
                    merged.pop(bound, None)
                elif bound in merged:
                    merged[bound] = min(merged[bound], value[bound])
    if name == 'view_reservation':
        properties['identifiers']['description'] = (
            'JSON array, not a string. Required for detail (1-3) and timeline (exactly 1); '
            'optional for quote_history (0-3), forbidden for index. Example: ["reservation-001"].')
        properties['sections']['description'] = 'detail only; JSON array, e.g. ["current_terms", "risk"].'
        required = ['view']
    else:
        properties['request_ids']['description'] = (
            'requests mode only, required array of 1-5 distinct IDs; not a JSON-encoded string. '
            'Example: ["rr-006-02-0004"].')
        properties['limit']['description'] = 'pending_page only; integer 1-5, not a quoted string.'
        properties['cursor']['description'] = 'pending_page only; use the returned request-ID string.'
        required = ['mode']
    return {'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False}
