"""Native Responses parsing must retain auditable model/execution identities."""
from copy import deepcopy
import json

import pytest

from outfitter.analyze import _load_events
from outfitter.call_identity import response_message, response_finish_reason, load_identity
from outfitter.models import ToolResult
from outfitter.store import RunStore
from test_v54_recovery_state import environment


def function_item(call_id='call-1', arguments='{}'):
    return {'type': 'function_call', 'id': 'fc-output-item', 'call_id': call_id,
            'name': 'business_status', 'arguments': arguments, 'status': 'completed'}


def native(items, **extra):
    return {'object': 'response', 'status': 'completed', 'output': items, **extra}


def assistant(ids):
    return {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': ident, 'type': 'function', 'function': {'name': 'business_status', 'arguments': '{}'}}
        for ident in ids]}


def test_native_output_uses_call_id_preserves_arguments_and_order():
    raw = native([
        {'type': 'reasoning', 'id': 'rs-1', 'encrypted_content': 'opaque'},
        {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Checking.'}]},
        function_item('call-a'),
        {'type': 'function_call_output', 'call_id': 'call-unrelated', 'output': 'not a call'},
        function_item('call-b', '{"truncated":'),
        {'type': 'message', 'role': 'assistant', 'content': [{'type': 'refusal', 'refusal': 'No.'}]},
    ])
    before = deepcopy(raw)
    parsed = response_message(raw)
    assert [c['id'] for c in parsed['tool_calls']] == ['call-a', 'call-b']
    assert parsed['tool_calls'][1]['function']['arguments'] == '{"truncated":'
    assert parsed['content'] == 'Checking.\nNo.'
    assert raw == before
    missing = function_item(); missing.pop('call_id')
    assert response_message(native([missing]))['tool_calls'][0]['id'] is None


@pytest.mark.parametrize('wrapped', [False, True])
def test_chat_and_legacy_message_are_unchanged(wrapped):
    message = assistant(['legacy'])
    response = {'choices': [{'message': message, 'finish_reason': 'tool_calls'}]} if wrapped else {'message': message, 'finish_reason': 'tool_calls'}
    response['output'] = [function_item('must-not-override-legacy')]
    assert response_message(response) is message
    assert response_finish_reason(response) == 'tool_calls'


@pytest.mark.parametrize('raw,expected', [
    (native([function_item()]), 'tool_calls'),
    (native([{'type': 'reasoning', 'summary': []}]), 'stop'),
    (native([], status='incomplete', incomplete_details={'reason': 'max_output_tokens'}), 'length'),
    (native([], status='incomplete', incomplete_details={'reason': 'content_filter'}), None),
    (native([], status='failed'), None),
    (native([function_item()], finish_reason='length'), 'length'),
])
def test_native_finish_reason(raw, expected):
    assert response_finish_reason(raw) == expected


@pytest.mark.parametrize('legacy_links', [False, True])
def test_native_identity_and_analysis_with_renamed_duplicate_ids(tmp_path, legacy_links):
    state = environment().reset()
    store = RunStore.create(tmp_path/'run', {}, state)
    raw = native([function_item('same'), function_item('same')])
    store.record_model_response(state, request={'messages': []}, response=raw,
        assistant_message=assistant(['same', 'same__renamed']), input_tokens=123, output_tokens=45,
        charge=lambda *args: None)
    for ident in ('same', 'same__renamed'):
        store.execute_tool(ident, 'business_status', {}, lambda *args: ToolResult(True, {}))
    with store._connect() as conn:
        assert conn.execute('select raw_id from model_tool_links order by position').fetchall() == [('same',), ('same',)]
        # Old Responses trajectories have explicit model-message links, but
        # their pre-fix raw_id columns were NULL. They must work without edits.
        conn.execute('update model_tool_links set raw_id=NULL')
        if legacy_links:
            conn.execute('drop table model_message_links')
            conn.execute('drop table model_tool_links')
    models, tools, integrity = _load_events(store.run_dir, tool_call_protocol='read_only_batch_v1')
    assert integrity == 'ok' and len(tools) == 2
    assert models[0]['raw_tool_call_ids'] == ['same', 'same']
    assert models[0]['tool_call_ids'] == ['same', 'same__renamed']
    assert models[0]['raw_tool_calls'] == 2
    assert models[0]['read_only_tool_batches'] == 1
    assert models[0]['batched_read_only_tool_calls'] == 2
    assert models[0]['identity_status'] == ('validated_legacy_sequence' if legacy_links else 'explicit')
    with store._connect() as conn:
        assert json.loads(conn.execute('select response from model_calls').fetchone()[0]) == raw
        if not legacy_links:
            assert conn.execute('select raw_id from model_tool_links').fetchall() == [(None,), (None,)]


def test_native_signature_mismatch_remains_unknown(tmp_path):
    state = environment().reset()
    store = RunStore.create(tmp_path/'run', {}, state)
    store.record_model_response(state, request={}, response=native([function_item(arguments='{"unexpected":1}')]),
        assistant_message=assistant(['call-1']), input_tokens=0, output_tokens=0, charge=lambda *args: None)
    with store._connect() as conn:
        result = load_identity(conn, 'read_only_batch_v1')
    assert result['statuses'][1] == 'unknown'
    assert 1 not in result['model_ids']


def test_native_truncated_json_is_preserved_and_not_treated_as_executed(tmp_path):
    state = environment().reset()
    store = RunStore.create(tmp_path/'run', {}, state)
    raw = native([function_item(arguments='{"unfinished":')], status='incomplete',
                 incomplete_details={'reason': 'max_output_tokens'})
    store.record_model_response(state, request={}, response=raw, assistant_message=assistant([]),
        input_tokens=0, output_tokens=0, charge=lambda *args: None)
    models, tools, _ = _load_events(store.run_dir, tool_call_protocol='read_only_batch_v1')
    assert models[0]['finish_reason'] == 'length'
    assert models[0]['generation_truncated_tool_call_ids'] == ['call-1']
    assert models[0]['identity_status'] == 'explicit'
    assert models[0]['tool_call_ids'] == [] and tools == []
