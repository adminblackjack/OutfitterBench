"""Malformed actor calls must be replayed as rejections, not crash diagnostics."""
from copy import deepcopy
from dataclasses import replace
import json
import sqlite3

import pytest

from outfitter.evaluation.api import evaluate_run
from outfitter.llm import ScriptedProvider
from outfitter.runner import OutfitterRunner


def make_rejection_run(tmp_path, *, invalid):
    from test_v54_recovery_state import environment
    env = environment()
    state = env.reset()
    inquiry = next(q for q in state.reservation_requests if q.status == 'pending_quote')
    fixed = next(q for q in state.reservation_requests if q.status == 'pending_service_offer')
    line = inquiry.lines[0]
    actions = []
    if invalid:
        actions = [
            # GLM seed42 Day103: serialized text in place of an array.
            ('quote_reservation_inquiries', {'decisions': '[{"action":"quote"}]'}),
            ('quote_reservation_inquiries', {'decisions': ['not-an-object']}),
            # GLM seed73 Day7: promise_id instead of promise_profile_id.
            ('quote_reservation_inquiries', {'decisions': [{
                'request_id': inquiry.request_id, 'action': 'quote', 'lines': [{
                    'line_id': line.line_id, 'options': [{
                        'disposition': 'quote', 'offer_id': 'offer-022',
                        'daily_unit_price': 4.0, 'promise_id': line.eligible_profile_ids[0],
                    }],
                }],
            }]}),
            ('confirm_reservation_plans', {'selections': '[]'}),
            # Well-shaped payload with an invalid business selection also must
            # not contribute an opportunity or authorization decision.
            ('respond_reservation_service_offers', {'decisions': [{
                'request_id': fixed.request_id, 'action': 'accept',
                'plan_id': 'not-an-authorized-plan',
                'pickup_day': fixed.pickup_date_options[0]['pickup_day'],
            }]}),
        ]
    actions += [
        ('quote_reservation_inquiries', {'decisions': [{
            'request_id': inquiry.request_id, 'action': 'decline',
        }]}),
        ('close_business_day', {}),
    ]
    messages = [dict(role='assistant', content='', tool_calls=[dict(
        id=f'call-{i}', type='function', function=dict(name=name, arguments=json.dumps(args)))])
        for i, (name, args) in enumerate(actions)]
    runner = OutfitterRunner.create(config=replace(env.config, runs_dir=str(tmp_path)),
        provider=ScriptedProvider(messages), run_id='rejected' if invalid else 'control')
    runner.run(stop_after_day=1)
    return runner.store.run_dir, inquiry.request_id


def semantic(report):
    result = deepcopy(report['metrics']['extended'])
    result.pop('cache_key')
    result.pop('replay_calls')
    return result


def test_rejected_payloads_replay_without_diagnostic_credit(tmp_path):
    bad, request_id = make_rejection_run(tmp_path, invalid=True)
    good, _ = make_rejection_run(tmp_path, invalid=False)
    with sqlite3.connect(bad/'run.sqlite3') as conn:
        receipts = [json.loads(r) for r, in conn.execute('select result from tool_calls order by rowid')]
    assert [r['ok'] for r in receipts] == [False]*5 + [True]*2
    original = (bad/'run.sqlite3').read_bytes()
    rejected = evaluate_run(bad, through_day=1, extended=True, plots=False, cpu_budget=1)
    control = evaluate_run(good, through_day=1, extended=True, plots=False, cpu_budget=1)
    assert rejected['extended_status'] == 'complete'
    assert rejected['metrics']['extended']['replay_mismatches'] == 0
    assert len(rejected['metrics']['basic']['execution']['errors']) == 5
    assert semantic(rejected) == semantic(control)
    rows = rejected['metrics']['extended']['opportunities']['records']
    assert sum(r['request_id'] == request_id for r in rows) == 1
    assert (bad/'run.sqlite3').read_bytes() == original


def test_false_recorded_rejection_still_fails_receipt_comparison(tmp_path):
    run, _ = make_rejection_run(tmp_path, invalid=False)
    # Deliberately corrupt only this temporary test fixture. The actual valid
    # decline still executes, so skipping diagnostics cannot hide its success.
    with sqlite3.connect(run/'run.sqlite3') as conn:
        ident, raw = conn.execute('select call_id,result from tool_calls order by rowid limit 1').fetchone()
        receipt = json.loads(raw)
        receipt['ok'] = False
        conn.execute('update tool_calls set result=? where call_id=?', (json.dumps(receipt), ident))
    with pytest.raises(RuntimeError, match='replay mismatch'):
        evaluate_run(run, through_day=1, extended=True, plots=False, cpu_budget=1)
