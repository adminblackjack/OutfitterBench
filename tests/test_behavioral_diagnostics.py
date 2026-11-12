from copy import deepcopy
import json
import os
from pathlib import Path
import socket

import pytest

from outfitter import behavioral_diagnostics as diag


def source_report(extended=True):
    protocol = {"relaxation_gaps": [1, 3, 7], "positive_improvement_cny": .01,
                "relaxation_delta": 1, "relaxation_window_days": 14}
    identity = {"through_day": 10, "sample_days": [5, 10], "parameters": protocol,
                "manifest": "placeholder", "evaluation_code": "frozen-source"}
    capital = {"value": 1.5, "numerator": 3, "denominator": 200,
               "unit": "CNY per 100 capital-CNY-days", "status": "available"}
    risk = [{"commitment_id": f"booking-{i}", "day": i, "pickup_day": i+lead,
             "lead_days": lead, "version": 1, "gap": gap,
             "shortages": [{"shortage": 1}] if gap else []}
            for i, (lead, gap) in enumerate(((8, True), (7, False), (2, True), (1, False)), 1)]
    actions = [{"day": 1, "tool": "confirm_reservation_plans", "relevant_prior": ["a", "b"],
                "affected_prior": ["a", "b"], "removed_prior": [], "shortages": {}},
               {"day": 2, "tool": "confirm_reservation_plans", "relevant_prior": ["c"],
                "affected_prior": [], "removed_prior": ["c"], "shortages": {}},
               {"day": 3, "tool": "confirm_reservation_plans", "relevant_prior": [],
                "affected_prior": [], "removed_prior": [], "shortages": {}},
               {"day": 3, "tool": "schedule_maintenance", "relevant_prior": [],
                "affected_prior": [], "removed_prior": [], "shortages": {}}]
    records = [{"day": 5, "candidate_count": 0, "improvements": {str(g): 0 for g in (1, 3, 7)}},
               {"day": 10, "candidate_count": 2, "improvements": {str(g): .02 for g in (1, 3, 7)}}]
    ext = {"status": "complete", "method": "public-readiness-calendar-v1",
           "persistent_gaps": {"metric": diag.ratio(2, 4), "records": risk,
                               "by_lead": {"1-2": diag.ratio(1, 2), "3-7": diag.ratio(0, 1), "8+": diag.ratio(1, 1)}},
           "walk_in_intrusion": {"metric": diag.ratio(1, 2), "records": [
               {"day": 1, "batch_id": "trip-a", "affected_prior": ["a", "b"]},
               {"day": 1, "batch_id": "trip-b", "affected_prior": []}]},
           "action_protection": {"records": actions, "by_action": {
               "confirm_reservation_plans": diag.ratio(1, 2), "schedule_maintenance": diag.ratio(0, 0)}},
           "capacity_relaxation": {"records": records, "by_gap": {
               str(g): {"positive_days": 1, "evaluated_days": 2, "mean_daily_improvement": .01,
                        "no_candidate_days": 1, "passed_candidates": 2} for g in (1, 3, 7)}},
           "opportunities": {"metric": diag.ratio(1, 2)},
           "authorization_selection": {"metric": diag.ratio(1, 1)}}
    return {"format": diag.SOURCE_FORMAT, "run_id": "example", "through_day": 10,
            "input_identity": identity, "cache_key": diag.digest(identity), "protocol": protocol,
            "sample_protocol": "default", "comparison_group": "original-group",
            "environment_identity": {"source_sha256": "environment"},
            "run_metadata": {"seed": 42, "model": "example-model", "reasoning_effort": "high", "max_days": 10},
            "basic": {"capital_productivity": capital, "nominal_calendar": {"metrics": {}},
                "service": {"pickups": 3, "merchant_failures": 1, "customer_cancellations": 1,
                            "pending": 0, "reliability": diag.ratio(3, 4)},
                "business": {"final_score_available": True, "evaluation_score": 1100,
                             "initial_value": 1000, "value_change": 100, "run_bankrupt_at_snapshot": False}},
            "extended": ext if extended else {"status": "not_enabled"}, "warnings": []}


def derive(report):
    return diag.derive_report(report, "source-hash", {"derived-module": "test-version"})


def test_four_families_and_source_immutability():
    report = source_report()
    original = deepcopy(report)
    profile = derive(report)
    assert report == original
    assert profile['core']['pending_booking_gap_rate']['value'] == 50
    assert profile['core']['near_pickup_gap_rate']['value'] == 50
    assert profile['gap_by_lead']['3-7']['value'] == 0
    assert profile['gap_by_lead']['8+']['value'] == 100
    assert profile['core']['capital_productivity']['value'] == 1.5
    assert profile['core']['capacity_positive_day_rate_g3']['value'] == 50
    assert profile['core']['capacity_mean_improvement_g3']['value'] == .01
    assert profile['capacity_relaxation']['3']['no_candidate_days'] == 1


def test_action_batches_count_once_and_removals_remain_separate():
    profile = derive(source_report())
    action = profile['action_new_gaps']['by_action']['confirm_reservation_plans']
    assert (action['numerator'], action['denominator']) == (1, 2)
    assert action['affected_unique_commitments'] == 2
    assert action['removed_unique_commitments'] == action['relevant_removal_only_calls'] == 1
    assert profile['action_new_gaps']['by_action']['schedule_maintenance']['status'] == 'not_applicable'
    assert profile['core']['walk_in_induced_gap_rate']['numerator'] == 1
    assert len(profile['action_new_gaps']['events']) == 2


def test_not_enabled_is_not_zero_and_capital_can_be_negative():
    report = source_report(False)
    report['basic']['capital_productivity'].update(numerator=-3, value=-1.5)
    profile = derive(report)
    assert profile['core']['capital_productivity']['value'] == -1.5
    for key, obj in profile['core'].items():
        if key != 'capital_productivity':
            assert obj['value'] is None and obj['status'] == 'not_enabled'
    assert profile['action_new_gaps']['by_action'] == {}


def test_empty_lead_observations_and_no_owned_capital_are_na():
    report = source_report()
    report['extended']['persistent_gaps'].update(metric=diag.ratio(0, 0), records=[],
        by_lead={g: diag.ratio(0, 0) for g in ('1-2', '3-7', '8+')})
    report['basic']['capital_productivity'].update(numerator=5, denominator=0, value=None, status='not_applicable')
    p = derive(report)
    assert p['core']['near_pickup_gap_rate']['status'] == 'not_applicable'
    assert p['core']['capital_productivity']['value'] is None


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, '1.5'])
def test_invalid_metric_values_fail(bad):
    r = source_report()
    r['basic']['capital_productivity']['value'] = bad
    with pytest.raises(diag.DiagnosticError): derive(r)


@pytest.mark.parametrize('change', [
    lambda r: r.update(cache_key='wrong'),
    lambda r: r.update(through_day=11),
    lambda r: r['extended'].update(status='error'),
    lambda r: r['extended']['persistent_gaps']['metric'].update(value=51),
    lambda r: r['extended']['persistent_gaps']['records'][0].update(lead_days=7),
    lambda r: r['extended']['persistent_gaps']['records'].append(deepcopy(r['extended']['persistent_gaps']['records'][0])),
    lambda r: r['extended']['walk_in_intrusion']['records'].append(deepcopy(r['extended']['walk_in_intrusion']['records'][0])),
    lambda r: r['extended']['action_protection']['records'][0].update(removed_prior=['a']),
    lambda r: r['extended']['action_protection']['records'][0].update(relevant_prior=[]),
    lambda r: r['extended']['capacity_relaxation']['by_gap']['3'].update(mean_daily_improvement=.02),
    lambda r: r['extended']['capacity_relaxation']['records'].pop(),
    lambda r: r['extended']['walk_in_intrusion']['records'][0].update(day=11),
    lambda r: r['extended']['action_protection']['records'][0].update(day=11),
    lambda r: r['extended']['capacity_relaxation']['records'][1].update(candidate_count=0),
    lambda r: r['extended']['capacity_relaxation']['records'][0]['improvements'].update({'3': -.01}),
    lambda r: r['basic']['business'].update(value_change=99),
])
def test_bad_records_cannot_be_favorable_defaults(change):
    r = source_report(); change(r)
    with pytest.raises(diag.DiagnosticError): derive(r)


def test_custom_sampling_is_explicit_and_uses_all_zero_days():
    r = source_report()
    r['sample_protocol'] = 'custom_diagnostic'
    r['input_identity']['sample_days'] = [4, 10]
    r['cache_key'] = diag.digest(r['input_identity'])
    r['extended']['capacity_relaxation']['records'][0]['day'] = 4
    p = derive(r)
    assert p['source']['sample_protocol'] == 'custom_diagnostic'
    assert p['core']['capacity_mean_improvement_g3']['value'] == .01
    r['sample_protocol'] = 'default'
    with pytest.raises(diag.DiagnosticError, match='default sampling'): derive(r)


def fixture_run(tmp_path, name='example', seed=42):
    root = tmp_path/name
    report = source_report()
    report['run_id'] = name; report['run_metadata']['seed'] = seed
    manifest = {'run_id': name, 'config': {'seed': seed, 'description': '中文测试'}, 'runtime_data': 'data'}
    report['input_identity']['manifest'] = diag.digest(manifest)
    report['cache_key'] = diag.digest(report['input_identity'])
    location = root/'analysis/evaluation'/diag.SOURCE_FORMAT/'day-10'
    location.mkdir(parents=True)
    (location/'metrics.json').write_text(json.dumps(report, ensure_ascii=False))
    (root/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False))
    return root


def test_read_only_derivation_never_invokes_network_or_replay(tmp_path, monkeypatch):
    root = fixture_run(tmp_path)
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    def forbidden(*args, **kwargs): raise AssertionError('network/replay not allowed')
    monkeypatch.setattr(socket, 'socket', forbidden)
    import outfitter.evaluation as evaluation
    monkeypatch.setattr(evaluation, 'evaluate_run_cutoffs', forbidden)
    result = diag.diagnose_run(root, through_day=10)
    for p, data in before.items(): assert Path(p).read_bytes() == data
    destination = Path(result['output_dir'])
    assert destination.name == 'day-10' and destination.parent.name == diag.FORMAT
    assert (destination/'profile.json').is_file()
    assert 'not credited as successful protection' in (destination/'REPORT.md').read_text()


def test_bad_source_does_not_publish(tmp_path):
    root = fixture_run(tmp_path)
    (root/'manifest.json').write_text('{"run_id":"another","config":{"seed":42}}')
    with pytest.raises(diag.DiagnosticError): diag.diagnose_run(root, through_day=10)
    assert not (root/'analysis/evaluation'/diag.FORMAT).exists()


def test_parallel_matches_serial_and_preserves_environment(tmp_path, monkeypatch):
    roots = [fixture_run(tmp_path, f'r{i}', 42+i) for i in range(4)]
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '3')
    serial = diag.diagnose_many(roots, through_day=10, workers=1, publish=False)
    parallel = diag.diagnose_many(roots, through_day=10, workers=4, publish=False)
    assert [r['profile'] for r in serial] == [r['profile'] for r in parallel]
    assert os.environ['OPENBLAS_NUM_THREADS'] == '3'
    assert [r['run_dir'] for r in serial] == [r['run_dir'] for r in parallel]
    for root in roots: assert not (root/'analysis/evaluation'/diag.FORMAT).exists()


def member(profile, model='a', seed=42):
    return {'profile': profile, 'model_key': model, 'model': model, 'label': model,
            'reasoning_effort': 'high', 'seed': seed}


def test_aggregation_is_seed_equal_and_model_independent():
    p = derive(source_report())
    q = deepcopy(p); q['core']['pending_booking_gap_rate'] = diag.ratio(1, 10)
    rows = [member(p), member(p), member(q, seed=73)]
    values = diag.aggregate_members(rows)[0]['metrics']['pending_booking_gap_rate']
    assert values['value'] == 30 and values['contributing_seeds'] == 2
    extra = diag.aggregate_members(rows+[member(q, model='b')])
    assert next(r for r in extra if r['model']=='a')['metrics']['pending_booking_gap_rate'] == values


def test_comparison_groups_and_sampling_are_not_silently_merged():
    p = derive(source_report()); q = deepcopy(p)
    q['source']['comparison_group'] = 'different'
    rows = [{'profile': p, 'run_dir': '/a'}, {'profile': q, 'run_dir': '/b'}]
    assert len(diag.grouped_profiles(rows)) == 2
    contexts = {path: {'comparison_group': 'explicit', 'source_sha256': 'source-hash'} for path in ('/a','/b')}
    with pytest.raises(diag.DiagnosticError): diag.grouped_profiles(rows, contexts)
    assert len(diag.grouped_profiles(rows, contexts, {'policy':'checked'})) == 1
    q['source']['sample_protocol'] = 'custom_diagnostic'
    assert len(diag.grouped_profiles(rows, contexts, {'policy':'checked'})) == 2


def test_cli_and_default_concurrency(monkeypatch):
    from outfitter.cli import build_parser
    a = build_parser().parse_args(['diagnostics','a','b','--workers','80','--aggregate-dir','result'])
    assert a.workers == 80 and a.through_day == 214
    monkeypatch.setattr(diag, 'cpu_count', lambda: 80)
    assert diag.worker_count(None,16) == 16
    assert diag.worker_count(None,120) == 80
    with pytest.raises(diag.DiagnosticError): diag.worker_count(0,16)


def test_empty_samples_do_not_become_zero_opportunity():
    r = source_report()
    r['through_day'] = r['input_identity']['through_day'] = 4
    r['input_identity']['sample_days'] = []
    r['cache_key'] = diag.digest(r['input_identity'])
    r['basic']['business'].update(final_score_available=False, evaluation_score=None)
    r['extended']['capacity_relaxation'] = {'records': [], 'by_gap': {
        str(g): {'positive_days': 0, 'evaluated_days': 0, 'mean_daily_improvement': None,
                 'no_candidate_days': 0, 'passed_candidates': 0} for g in (1,3,7)}}
    p = derive(r)
    assert p['core']['capacity_positive_day_rate_g3']['status'] == 'not_applicable'
    assert p['core']['capacity_mean_improvement_g3']['value'] is None
    assert p['outcomes']['official_score'] is None and p['outcomes']['net_value'] is None


def test_cohort_publishing_retains_missingness_and_evidence(tmp_path):
    roots = [fixture_run(tmp_path, 'a', 42), fixture_run(tmp_path, 'b', 73)]
    results = [diag.diagnose_run(root, through_day=10, publish=False) for root in roots]
    destination = tmp_path/'cohort'
    manifest = diag.write_cohort(results, destination, plots=False)
    assert json.loads((destination/'COMPLETE.json').read_text())['runs'] == 2
    assert len(manifest['groups']) == 1
    assert 'Seed 42' in (destination/'SUMMARY.md').read_text()
    assert 'Seed 73' in (destination/'SUMMARY.md').read_text()
    assert 'confirm_reservation_plans' in (destination/'action_new_gaps.csv').read_text()
    assert '100.0' in (destination/'gap_by_lead.csv').read_text()
    assert not any((r/'analysis/evaluation'/diag.FORMAT).exists() for r in roots)


def test_failed_rerender_does_not_leave_a_complete_marker(tmp_path, monkeypatch):
    root = fixture_run(tmp_path, 'source')
    results = [diag.diagnose_run(root, through_day=10, publish=False)]
    destination = tmp_path/'cohort'
    diag.write_cohort(results, destination, plots=False)
    assert json.loads((destination/'COMPLETE.json').read_text())['status'] == 'complete'
    def fail(*args): raise diag.DiagnosticError('failed rendering')
    monkeypatch.setattr(diag, 'aggregate_members', fail)
    with pytest.raises(diag.DiagnosticError, match='failed rendering'):
        diag.write_cohort(results, destination, plots=False)
    assert json.loads((destination/'COMPLETE.json').read_text())['status'] != 'complete'


@pytest.mark.parametrize('kind',['operating_profiles','pickup_lead_gaps','capacity_opportunities'])
def test_figures_keep_seeds_separate_and_missing_values_missing(tmp_path, monkeypatch, kind):
    from outfitter.behavioral_diagnostic_plots import render_plot
    from matplotlib.figure import Figure
    captured=[]
    monkeypatch.setattr(Figure, 'savefig', lambda self,*a,**kw: captured.append(self))
    p, absent = derive(source_report()), derive(source_report(False))
    members=[]
    for seed in (42,73):
        for index, profile in enumerate((p, absent)):
            members.append(dict(id=f'{index}-{seed}', seed=seed, model_key=str(index),
                label=f'Model {index}', axis_label=f'M{index}', order=index,
                color=None, marker='o', profile=profile))
    render_plot((kind, members, str(tmp_path)))
    fig=captured[0]
    assert len(fig.axes) == (4 if kind=='capacity_opportunities' else 2)
    for i, axis in enumerate(fig.axes):
        seed=(42,73)[i%2]
        objects=axis.lines if kind=='pickup_lead_gaps' else axis.collections
        labels=[obj.get_label() for obj in objects]
        assert labels == [f'0-{seed}']
    if kind=='capacity_opportunities':
        assert any(t.get_text()=='N/A' for ax in fig.axes for t in ax.texts)
    else:
        assert {t.get_text() for t in fig.legends[0].get_texts()} == {'Model 0'}

