"""Internal model-message identity, never injected into actor context."""
import json
from collections import defaultdict


def response_message(response):
    """Return one assistant message from Chat Completions or Responses.

    Responses function calls are separate output items. Their ``call_id`` is
    the tool-result correlation ID; ``id`` identifies the output item itself.
    Preserve argument payloads (including malformed JSON) for later auditing.
    """
    choices = response.get('choices') or []
    if choices:
        return choices[0].get('message', {})
    if isinstance(response.get('message'), dict):
        return response['message']
    output = response.get('output')
    if not isinstance(output, list):
        return {}
    text = []
    calls = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get('type') == 'function_call':
            calls.append({
                'id': item.get('call_id'),
                'type': 'function',
                'function': {'name': item.get('name'), 'arguments': item.get('arguments')},
            })
        elif item.get('type') == 'message' and item.get('role') == 'assistant':
            for part in item.get('content') or []:
                if not isinstance(part, dict):
                    continue
                value = (part.get('text') if part.get('type') == 'output_text'
                         else part.get('refusal') if part.get('type') == 'refusal' else None)
                if isinstance(value, str):
                    text.append(value)
    message = {'role': 'assistant', 'content': '\n'.join(text)}
    if calls:
        message['tool_calls'] = calls
    return message


def response_finish_reason(response):
    """Keep explicit legacy reasons; translate native Responses terminal state."""
    choices = response.get('choices') or []
    if choices:
        return choices[0].get('finish_reason')
    if response.get('finish_reason') is not None:
        return response['finish_reason']
    if not isinstance(response.get('output'), list):
        return None
    if (response.get('status') == 'incomplete'
            and (response.get('incomplete_details') or {}).get('reason') == 'max_output_tokens'):
        return 'length'
    if response.get('status') == 'completed':
        return 'tool_calls' if response_message(response).get('tool_calls') else 'stop'
    return None


def call_signature(call):
    function = call.get('function') or {}
    args = function.get('arguments', '{}')
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {'_malformed_json': args}
    if not isinstance(args, dict):
        args = {'_non_object_arguments': args}
    return (function.get('name', ''), json.dumps(args, sort_keys=True, ensure_ascii=False))


def compatible(raw, actual, protocol):
    """Validate controller retention rules, not merely sequence proximity."""
    generated = list(response_message(raw).get('tool_calls') or [])
    retained = list(actual.get('tool_calls') or [])
    finish = response_finish_reason(raw)
    if finish == 'length' and not retained:
        return True
    expected = generated[:1] if protocol == 'first_only' else generated
    return [call_signature(c) for c in expected] == [call_signature(c) for c in retained]


def load_identity(connection, protocol):
    """Recover old records conservatively; new records carry explicit linkage."""
    tables = {r[0] for r in connection.execute("select name from sqlite_master where type='table'")}
    messages = {}
    by_day = defaultdict(list)
    agent_ids = set()
    for seq, day, raw in connection.execute('select seq,day,value from messages order by seq'):
        message = json.loads(raw)
        if message.get('role') != 'assistant':
            continue
        messages[seq] = (day, message)
        by_day[day].append((seq, message))
        agent_ids.update(str(c.get('id')) for c in message.get('tool_calls') or [] if c.get('id'))
    raw_by_day = defaultdict(list)
    for mid, day, response in connection.execute('select id,day,response from model_calls order by id'):
        raw_by_day[day].append((mid, json.loads(response)))
    explicit = dict(connection.execute('select model_call_id,assistant_seq from model_message_links')) if 'model_message_links' in tables else {}
    mapped = {}; statuses = {}
    for day, rows in raw_by_day.items():
        saved = by_day[day]
        # Only infer ordered pairs when the whole day's cardinality and every
        # retained function/argument signature agree with controller rules.
        aligned = len(rows) == len(saved) and all(compatible(raw, msg, protocol)
            for (_, raw), (_, msg) in zip(rows, saved))
        for index, (mid, raw) in enumerate(rows):
            match = messages.get(explicit.get(mid))
            if match and match[0] == day and compatible(raw, match[1], protocol):
                message = match[1]; status = 'explicit'
            elif mid not in explicit and aligned:
                message = saved[index][1]; status = 'validated_legacy_sequence'
            else:
                statuses[mid] = 'unknown'; continue
            mapped[mid] = [str(c.get('id')) for c in message.get('tool_calls') or [] if c.get('id')]
            statuses[mid] = status
    automatic = set()
    if 'tool_call_origins' in tables:
        automatic.update(r[0] for r in connection.execute(
            "select call_id from tool_call_origins where origin='automatic' and controller_kind in ('forced_close','runoff')"))
    else:
        # Old controller's exact ID plus tool/date/empty args and matching
        # counters/phase give evidence; an unmatched provider ID alone does not.
        raw_state = connection.execute('select value from state where id=1').fetchone()
        state = json.loads(raw_state[0]) if raw_state else {}
        horizon = state.get('decision_horizon_day') or state.get('max_days', 0)
        for cid,day,name,args in connection.execute('select call_id,day,name,arguments from tool_calls'):
            if cid in agent_ids or name != 'close_business_day' or json.loads(args) != {}:
                continue
            if (cid == f'automatic-runoff-close-day-{day}' and day > horizon
                    or cid == f'forced-close-day-{day}' and state.get('forced_day_ends',0)>0):
                automatic.add(cid)
    return dict(model_ids=mapped,statuses=statuses,agent_ids=agent_ids,automatic_ids=automatic)
