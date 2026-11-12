from dataclasses import replace
from pathlib import Path
import json
import sqlite3

import pytest

from outfitter.calendar_baseline import run_calendar
from outfitter.cli import build_parser, command_resume
from outfitter.config import OutfitterConfig
from outfitter.store import RunStore

ROOT = Path(__file__).resolve().parents[1]


def configuration(tmp_path):
    return replace(OutfitterConfig.from_yaml(ROOT / 'configs/outfitter_v54_main.example.yaml'),
                   runs_dir=str(tmp_path))


def test_calendar_resume_matches_uninterrupted_run(tmp_path):
    config = configuration(tmp_path)
    continuous = run_calendar(config, run_id='continuous', stop_after_day=3)
    paused = run_calendar(config, run_id='paused', stop_after_day=1)
    resumed = run_calendar(resume_dir=tmp_path / 'paused', stop_after_day=3)
    assert paused['day'] == 2 and resumed['day'] == 4
    assert continuous['evaluation_score'] is None
    assert resumed['calendar']['model_calls'] == 0
    assert resumed['calendar']['tool_counts'] == continuous['calendar']['tool_counts']
    assert RunStore.resume(tmp_path / 'continuous').load_state().to_dict() == \
           RunStore.resume(tmp_path / 'paused').load_state().to_dict()
    manifest = json.loads((tmp_path / 'paused/manifest.json').read_text())
    assert manifest['config']['max_days'] == 200
    assert manifest['config']['runoff_days'] == 14
    with sqlite3.connect(tmp_path / 'paused/run.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM model_calls').fetchone()[0] == 0
        assert db.execute('SELECT count(*) FROM day_starts').fetchone()[0] == 3
    args = build_parser().parse_args(['resume', str(tmp_path / 'paused')])
    with pytest.raises(ValueError, match='baseline --resume'):
        command_resume(args)


def test_calendar_rejects_nonpaper_horizon_and_invalid_stop(tmp_path):
    config = configuration(tmp_path)
    with pytest.raises(ValueError, match='paper protocol'):
        run_calendar(replace(config, max_days=3), run_id='wrong-horizon')
    with pytest.raises(ValueError, match='stop-after-day'):
        run_calendar(config, run_id='wrong-stop', stop_after_day=0)
    assert not (tmp_path / 'wrong-horizon').exists()
    assert not (tmp_path / 'wrong-stop').exists()
