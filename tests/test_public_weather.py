from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from outfitter.config import OutfitterConfig
from outfitter.public_weather import (
    DATASET_ID, SOURCE_FILES, aggregate_noaa, build, classify, load_era5,
    local_hour, precipitation, temperature,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data_sources/weather"
RUNTIME = ROOT / "src/outfitter/runtime_data"


def row(timestamp, temp="+0100,1", **fields):
    return {"STATION": "54511099999", "DATE": timestamp, "TMP": temp, **fields}


@pytest.mark.parametrize("value,expected", [
    ("+0123,1", 12.3), ("-0123,9", -12.3), ("+9999,1", None),
    ("+0123,2", None), ("garbage", None), ("+0620,1", None),
])
def test_temperature_quality_and_missing(value, expected):
    assert temperature(value) == expected


@pytest.mark.parametrize("value,expected", [
    ("06,0023,9,1", (6, 2.3)), ("01,0000,2,1", (1, 0.0)),
    ("06,0023,3,1", (6, 2.3)), ("06,9999,3,1", None),
    ("99,0023,9,1", None), ("06,9999,9,1", None), ("06,0023,9,2", None),
    ("06,0023,1,1", None), ("06,0023,I,1", None), ("00,0023,9,1", None),
])
def test_precipitation_quality_period_and_missing(value, expected):
    assert precipitation(value) == expected


def test_local_year_and_equal_hour_weighting():
    assert local_hour("2022-12-31T16:30:00").isoformat() == "2023-01-01T00:00:00+08:00"
    values = aggregate_noaa([
        row("2022-12-31T16:00:00", "+0000,1"),
        row("2022-12-31T16:30:00", "+0040,1"),
        row("2022-12-31T17:00:00", "+0100,1"),
        row("2023-12-31T16:00:00", "+0300,1"),
    ])
    assert list(values) == ["2023-01-01"]
    assert values["2023-01-01"]["hourly_temperatures"] == [2, 10]


def test_shortest_precipitation_report_uses_local_end_day():
    values = aggregate_noaa([row("2023-04-04T17:00:00", AA1="24,0023,9,1", AA2="06,0000,9,1")])
    reports = values["2023-04-05"]["precipitation_reports"]
    assert reports == [{"ending_hour": 1, "hours": 6, "amount_mm": 0.0}]


def test_conflicting_shortest_reports_are_not_arbitrarily_selected():
    values = aggregate_noaa([
        row("2023-04-05T01:00:00", AA1="06,0023,9,1"),
        row("2023-04-05T01:30:00", AA1="06,0000,9,1"),
    ])["2023-04-05"]
    assert not values["precipitation_reports"]
    assert values["precipitation_conflict_hours"] == 1


def test_cloud_boundary_and_snow_priority():
    assert classify(rain=False, cloud_cover_pct=20, snowfall_cm=0, snow_depth_max_m=0) == "sunny"
    assert classify(rain=False, cloud_cover_pct=20.01, snowfall_cm=0, snow_depth_max_m=0) == "cloudy"
    assert classify(rain=True, cloud_cover_pct=0, snowfall_cm=0, snow_depth_max_m=0) == "rainy"
    assert classify(rain=True, cloud_cover_pct=0, snowfall_cm=0, snow_depth_max_m=.01) == "snowy"
    assert classify(rain=False, cloud_cover_pct=0, snowfall_cm=.1, snow_depth_max_m=0) == "snowy"


def test_rebuild_is_read_only_and_complete():
    paths = [RUNTIME / "weather.jsonl", RUNTIME / "metadata.json", SOURCES / "daily_provenance.jsonl"]
    before = {str(p): p.read_bytes() for p in paths}
    first = build(SOURCES, RUNTIME, check=True)
    second = build(SOURCES, RUNTIME, check=True)
    assert first == second
    assert first["days"] == 365 and first["dataset_id"] == DATASET_ID
    assert all(p.read_bytes() == before[str(p)] for p in paths)
    assert not (RUNTIME / "weather_finalinfo.pkl").exists()
    meta = json.loads((RUNTIME / "metadata.json").read_text())
    assert "weather_finalinfo.pkl" not in meta["content_sha256"]
    for name, digest in meta["content_sha256"].items():
        assert hashlib.sha256((RUNTIME / name).read_bytes()).hexdigest() == digest


def test_era5_missing_snow_depth_is_not_silently_zero():
    daily = json.loads((SOURCES / SOURCE_FILES[2]).read_text())
    snow = json.loads((SOURCES / SOURCE_FILES[3]).read_text())
    snow["hourly"]["snow_depth"][0] = None
    with pytest.raises(ValueError, match="snow_depth"):
        load_era5(daily, snow)


def test_sources_are_integrity_checked_before_build(tmp_path):
    import shutil
    for name in (*SOURCE_FILES, "sources.json"):
        shutil.copy2(SOURCES / name, tmp_path / name)
    (tmp_path / SOURCE_FILES[0]).write_bytes(b"not a NOAA input")
    with pytest.raises(ValueError, match="source integrity mismatch"):
        build(tmp_path, RUNTIME, check=True)


def test_coverage_records_fallbacks_and_extremes():
    weather = [json.loads(line) for line in (RUNTIME / "weather.jsonl").read_text().splitlines()]
    provenance = [json.loads(line) for line in (SOURCES / "daily_provenance.jsonl").read_text().splitlines()]
    assert [r["day"] for r in weather] == list(range(1, 366))
    for r, p in zip(weather, provenance, strict=True):
        assert r["night_low_c"] <= r["temp_mean_c"] <= r["day_high_c"]
        assert r["weather_source_date"] == p["source_date"]
        assert 0 <= p["noaa_temperature_hours"] <= 24
        assert 0 <= p["noaa_precipitation_report_hours"] <= 24
        if p["precipitation_source"] == "era5_fallback":
            assert p["noaa_precipitation_report_hours"] == 0


def test_run_records_weather_identity_and_rejects_changed_snapshot(tmp_path):
    from outfitter.calendar_baseline import run_calendar
    from outfitter.runner import OutfitterRunner
    from outfitter.run_scripted_merchant import NoModel
    config = replace(OutfitterConfig.from_yaml(ROOT / "configs/outfitter_v54_main.example.yaml"), runs_dir=str(tmp_path))
    run_calendar(config, run_id="weather", stop_after_day=1)
    run = tmp_path / "weather"
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["controller"]["weather_dataset_id"] == DATASET_ID
    weather = run / "data/weather.jsonl"
    weather.write_text(weather.read_text() + "\n")
    with pytest.raises(ValueError, match="integrity|hash|snapshot|runtime data",):
        OutfitterRunner.resume(run_dir=run, provider=NoModel())
