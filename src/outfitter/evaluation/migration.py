"""Explicit one-time import of the verified pre-optimization evaluator cache."""
from pathlib import Path
import json
import shutil

from .common import digest, PARAMETERS, write_json
from .execution import read_state, write_state

LEGACY_CODE='1005abe9da79e60f50b71b274539c4c527e993db70d2ccb1853c8a350b5c99cf'


def validate_legacy(legacy_dir, identity):
    old=Path(legacy_dir).resolve()
    report=json.loads((old.parent.parent/'metrics.json').read_text())
    if report.get('cache_key')!=old.name:
        report=json.loads((old.parent.parent/'history'/old.name/'metrics.json').read_text())
    previous=report['input_identity']
    if previous['evaluation_code']!=LEGACY_CODE or previous['parameters']!=PARAMETERS:
        raise ValueError('legacy evaluator/protocol is not allowlisted')
    if old.name!=digest(previous):
        raise ValueError('legacy cache identity mismatch')
    for key in ('state','calls','manifest','data'):
        if previous[key]!=identity[key]:raise ValueError(f'legacy input mismatch: {key}')
    return previous


def import_checkpoint(legacy_dir,cache,calls,sample_days,cutoffs):
    old=Path(legacy_dir); cache=Path(cache)
    # The API has validated full input and legacy implementation identities.
    identity=json.loads((cache/'legacy-import.json').read_text())
    if identity['legacy_dir']!=str(old.resolve()) or identity['code']!=LEGACY_CODE:
        raise ValueError('legacy import was not validated')
    restored=read_state(old/'checkpoint.json.gz')
    closed=len(restored['state']['daily_history']); index=restored['next_call']
    if not 0<index<=len(calls) or calls[index-1][0]!=closed or calls[index-1][1]!='close_business_day':
        raise ValueError('legacy checkpoint tool-prefix boundary mismatch')
    if not calls[index-1][3].get('ok') or any(c[0]<=closed for c in calls[index:]):
        raise ValueError('legacy checkpoint does not terminate the complete closed-day prefix')
    if min(cutoffs)<closed:
        raise ValueError('legacy checkpoint is later than first requested cutoff')
    if sorted(restored['seen'])!=list(range(1,closed+1)):
        raise ValueError('legacy observed-day sequence mismatch')
    expected=[d for d in sample_days if d<=closed]
    if sorted(r['day'] for r in restored['rows']['capacity_relaxation'])!=expected:
        raise ValueError('legacy sample coverage mismatch')
    write_state(cache/'checkpoint.json.gz',restored)
    # Only fully closed days are imported; mid-day candidate files stay in the old cache.
    for d in expected:
        source=old/'candidates'/f'day-{d}'
        if source.exists():shutil.copytree(source,cache/'candidates'/f'day-{d}',dirs_exist_ok=True)
    write_json(cache/'migration.json',dict(legacy_dir=str(old),completed_day=closed,
        next_call=index,validated_input=True,legacy_code=LEGACY_CODE))
