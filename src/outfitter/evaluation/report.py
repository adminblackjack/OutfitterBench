from __future__ import annotations

from .common import write_csv, write_json


def aggregate_reports(reports):
    """Separate environment/budget/protocol groups; seeds are the aggregation unit."""
    from collections import defaultdict
    groups=defaultdict(list)
    for report in reports:
        meta=report.get('run_metadata',{})
        groups[(report['comparison_group'],meta.get('model'),meta.get('reasoning_effort'))].append(report)
    result=[]
    for (group,model,effort),members in sorted(groups.items(),key=lambda x:str(x[0])):
        values=defaultdict(lambda:defaultdict(list))
        for r in members:
            seed=r['run_metadata'].get('seed')
            for metric in metric_rows(r):
                if metric.get('value') is not None:values[metric['metric']][seed].append(metric['value'])
        result.append(dict(comparison_group=group,model=model,reasoning_effort=effort,
            run_ids=[r['run_id'] for r in members],seeds=sorted({r['run_metadata'].get('seed') for r in members},key=str),
            metrics={k:dict(mean=sum(sum(v)/len(v) for v in by_seed.values())/len(by_seed),
                            contributing_seeds=len(by_seed),per_seed={str(s):sum(v)/len(v) for s,v in by_seed.items()})
                     for k,by_seed in values.items()},
            interpretation='Means within seeds, then equal weight per seed; N/A is excluded with support shown.'))
    return result


def metric_rows(report):
    b = report['basic']
    values = [('service_reliability', b['service']['reliability']),
              ('delivered_requests', b['service']['request_delivery']),
              ('capital_productivity', b['capital_productivity']),
              ('quick_cancellation', b['quick_cancellations']['rate'])]
    for gap, group in b['nominal_calendar'].get('metrics', {}).items():
        values.extend([(f'nominal_coverage_g{gap}', group['coverage']),
                       (f'old_promise_protection_g{gap}', group['protection'])])
    ext = report.get('extended', {})
    for key in ['walk_in_intrusion', 'persistent_gaps', 'opportunities', 'authorization_selection']:
        obj = ext.get(key, {})
        values.append((key, obj.get('metric', dict(value=None, status=ext.get('status', 'not_enabled')))))
    for gap in ('1','3','7'):
        obj=ext.get('capacity_relaxation',{}).get('by_gap',{}).get(gap,{})
        value=obj.get('mean_daily_improvement')
        values.append((f'local_relaxation_improvement_g{gap}',dict(value=value,unit='CNY per sampled day',
            denominator=obj.get('evaluated_days'),status='available' if value is not None else ext.get('status','not_enabled'))))
    return [dict(metric=k, **{n:v for n,v in obj.items() if n in
                {'value','numerator','denominator','unit','status'}}) for k,obj in values]


def save_report(report, output, plots=True):
    output.mkdir(parents=True, exist_ok=True)
    rows = metric_rows(report)
    write_json(output/'metrics.json', report)
    write_csv(output/'metrics.csv', rows)
    b = report['basic']
    write_csv(output/'admission_checks.csv', b['nominal_calendar'].get('records', []))
    write_csv(output/'quick_cancellations.csv', report.get('extended', {}).get('quick_cancellations', b['quick_cancellations']['records']))
    write_csv(output/'asset_sales.csv', b['assets']['sales'])
    write_csv(output/'new_assets_30d.csv', b['assets']['new_asset_30d'])
    ext = report.get('extended', {})
    for key in ['walk_in_intrusion','persistent_gaps','opportunities','authorization_selection','action_protection','capacity_relaxation','evidence_contact']:
        write_csv(output/f'{key}.csv', ext.get(key, {}).get('records', []))
    lines = ['# Outfitter trajectory evaluation', '',
             f"- Run: `{report['run_id']}`; closed-day cutoff: {report['through_day']}.",
             f"- Protocol: `{report['format']}`; extended: `{ext.get('status','not_enabled')}`.",
             '- Nominal coverage is a diagnostic, not a feasibility guarantee or decision-correctness score.',
             '- N/A is not zero or a perfect score. Local demand branches are hindsight diagnostics, not additive season gains.',
             '- The official business score and environment reward are unchanged.', '',
             '| Metric | Value | Numerator | Denominator | Status |', '|---|---:|---:|---:|---|']
    for row in rows:
        val = row.get('value')
        lines.append(f"| {row['metric']} | {val:.4f} | {row.get('numerator','')} | {row.get('denominator','')} | {row.get('status','')} |" if val is not None
                     else f"| {row['metric']} | N/A | {row.get('numerator','')} | {row.get('denominator','')} | {row.get('status','')} |")
    if report.get('warnings'):
        lines += ['', '## Warnings', '', *['- '+w for w in report['warnings']]]
    lines += ['', '## Interpretation', '',
              'All records are evaluator-only. Full-window metrics and sampled local replays have separate denominators. '
              'Keep versions, seed, business scale, unresolved cases, and candidate coverage visible when comparing runs.',
              'The JSON records protocol parameters, input/source identities, method assumptions and object-level evidence.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        chosen = [r for r in rows if r.get('value') is not None and r.get('unit') == 'percent']
        if chosen:
            fig, ax = plt.subplots(figsize=(10, max(4, len(chosen)*.36)))
            ax.barh([r['metric'] for r in chosen], [r['value'] for r in chosen])
            ax.set(xlabel='Percent (directions differ; not a composite score)', xlim=(0,100))
            fig.tight_layout(); fig.savefig(output/'diagnostics.png', dpi=140); plt.close(fig)
