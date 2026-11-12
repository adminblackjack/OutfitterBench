import json
import sqlite3
from pathlib import Path

import pytest

from outfitter.call_identity import load_identity
from outfitter.models import ToolResult
from outfitter.store import RunStore
from test_v54_recovery_state import environment


def message(ids):
    return {'role':'assistant','content':None,'tool_calls':[
        {'id':i,'type':'function','function':{'name':'business_status','arguments':'{}'}} for i in ids]}


def record(store,state,raw_ids,executed_ids,finish='tool_calls'):
    response={'choices':[{'message':message(raw_ids),'finish_reason':finish}]}
    store.record_model_response(state,request={},response=response,assistant_message=message(executed_ids),
        input_tokens=0,output_tokens=0,charge=lambda *a:None)
    for cid in executed_ids:
        store.execute_tool(cid,'business_status',{},lambda *a:ToolResult(True,{}))


def test_new_mapping_duplicate_ids_and_atomic_rewrite(tmp_path):
    s=environment().reset();store=RunStore.create(tmp_path/'run',{},s)
    record(store,s,['same','same'],['same','same__one'])
    s.day+=1;record(store,s,['same'],['same__two'])
    assert store.rewrite_last_assistant_tool_call_id('same__two','same__three')
    with store._connect() as c:
        assert c.execute('select raw_id,executed_id from model_tool_links order by model_call_id,position').fetchall()==[
            ('same','same'),('same','same__one'),('same','same__three')]
        ids=load_identity(c,'read_only_batch_v1')
        assert ids['model_ids']=={1:['same','same__one'],2:['same__three']}
        assert set(ids['statuses'].values())=={'explicit'}
        assert json.loads(c.execute('select response from model_calls where id=2').fetchone()[0])['choices'][0]['message']['tool_calls'][0]['id']=='same'


@pytest.mark.parametrize('protocol,raw,kept,finish',[
    ('read_only_batch_v1',['x','x'],['x','x__renamed'],'tool_calls'),
    ('first_only',['x','y'],['x'],'tool_calls'),
    ('read_only_batch_v1',['x'],[],'length'),
    ('read_only_batch_v1',[],[],'stop'),
])
def test_legacy_retention_validation(tmp_path,protocol,raw,kept,finish):
    s=environment().reset();store=RunStore.create(tmp_path/'run',{},s)
    record(store,s,raw,kept,finish)
    with store._connect() as c:
        c.execute('drop table model_message_links');c.execute('drop table model_tool_links')
        ids=load_identity(c,protocol)
        assert ids['model_ids'][1]==kept
        assert ids['statuses'][1]=='validated_legacy_sequence'


def test_ambiguous_history_is_unknown_not_automatic(tmp_path):
    s=environment().reset();store=RunStore.create(tmp_path/'run',{},s)
    record(store,s,['x'],['x__renamed'])
    store.append_message(s.day,'controller',{'role':'assistant','content':'unlinked extra reply'})
    with store._connect() as c:
        c.execute('drop table model_message_links');c.execute('drop table model_tool_links')
        ids=load_identity(c,'read_only_batch_v1')
        assert ids['statuses'][1]=='unknown'
        assert 'x__renamed' in ids['agent_ids']
        assert not ids['automatic_ids']


def test_controller_origin_is_explicit(tmp_path):
    s=environment().reset();store=RunStore.create(tmp_path/'run',{},s)
    store.execute_tool('forced-close-day-1','close_business_day',{},lambda *a:ToolResult(True,{}),controller_kind='forced_close')
    store.execute_tool('unmatched','business_status',{},lambda *a:ToolResult(True,{}))
    with store._connect() as c:
        ids=load_identity(c,'read_only_batch_v1')
        assert ids['automatic_ids']=={'forced-close-day-1'}
        assert 'unmatched' not in ids['agent_ids'] | ids['automatic_ids']
