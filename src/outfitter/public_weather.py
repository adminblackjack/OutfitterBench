"""Rebuild the public weather input without access to the paper's weather data.

Only this offline builder reads the source observations. Normal benchmark runs
read the frozen weather.jsonl and use the unchanged V54 weather mechanisms.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Iterable

from .jsonio import write_json, write_jsonl
from .v5 import calendar_for_source_day

DATASET_ID = "beijing-2023-noaa-era5-v1"
BUILD_POLICY = "local-day-hour-balanced-shortest-report-v1"
YEAR = 2023
STATION = "54511099999"
LOCAL_TZ = timezone(timedelta(hours=8), "Asia/Shanghai")
GOOD_QUALITY = frozenset("01459AIMPRU")
TEMPERATURE_QUALITY = GOOD_QUALITY | {"C"}
BASE_RANGES = {1: 10, 2: 11, 3: 12, 4: 12, 5: 11, 6: 10,
               7: 8, 8: 8, 9: 10, 10: 11, 11: 11, 12: 10}
RANGE_ADJUSTMENTS = {"sunny": 2, "cloudy": -1, "rainy": -3, "snowy": -2}
SOURCE_FILES = (
    "noaa-54511099999-2022.csv.gz",
    "noaa-54511099999-2023.csv.gz",
    "open-meteo-era5-beijing-2023.json",
    "open-meteo-era5-land-snow-beijing-2023.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_hour(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(LOCAL_TZ).replace(minute=0, second=0, microsecond=0)


def temperature(value: str) -> float | None:
    fields = value.split(",")
    if len(fields) != 2 or fields[1] not in TEMPERATURE_QUALITY:
        return None
    try:
        raw = int(fields[0])
    except ValueError:
        return None
    if not -932 <= raw <= 618:
        return None
    return raw / 10.0


def precipitation(value: str) -> tuple[int, float] | None:
    fields = value.split(",")
    if len(fields) != 4 or fields[3] not in GOOD_QUALITY:
        return None
    # A finite, quality-accepted numeric amount can accompany an accumulation
    # marker (3/4). Check its explicit period and missing sentinel below rather
    # than treating the marker itself as a missing value. Exclude inaccurate,
    # missing/deleted-period and incomplete measurements.
    if fields[2] not in {"2", "3", "4", "9", "E"}:
        return None
    try:
        hours, raw = int(fields[0]), int(fields[1])
    except ValueError:
        return None
    if not 1 <= hours <= 98 or not 0 <= raw <= 9998:
        return None
    return hours, raw / 10.0


def aggregate_noaa(records: Iterable[dict]) -> dict[str, dict]:
    temps, rain = defaultdict(list), defaultdict(list)
    for row in records:
        if row["STATION"] != STATION:
            raise ValueError("unexpected NOAA station")
        hour = local_hour(row["DATE"])
        if hour.year != YEAR:
            continue
        value = temperature(row.get("TMP", ""))
        if value is not None:
            temps[hour].append(value)
        for field in ("AA1", "AA2", "AA3", "AA4"):
            value = precipitation(row.get(field, ""))
            if value is not None:
                rain[hour].append(value)

    daily = defaultdict(lambda: {"hourly_temperatures": [], "precipitation_reports": [],
                                 "precipitation_conflict_hours": 0})
    for hour, values in sorted(temps.items()):
        daily[hour.date().isoformat()]["hourly_temperatures"].append(mean(values))
    for hour, values in sorted(rain.items()):
        day = daily[hour.date().isoformat()]
        shortest = min(period for period, _ in values)
        amounts = sorted({amount for period, amount in values if period == shortest})
        if len(amounts) != 1:
            # Do not choose whichever report happens to match another dataset.
            # A conflict excludes this hour; remaining daily reports can be used.
            day["precipitation_conflict_hours"] += 1
            continue
        day["precipitation_reports"].append({"ending_hour": hour.hour,
                                           "hours": shortest, "amount_mm": amounts[0]})
    return dict(daily)


def noaa_records(paths: Iterable[Path]) -> Iterable[dict]:
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"STATION", "DATE", "TMP", "AA1"}.issubset(reader.fieldnames or ()):
                raise ValueError(f"invalid NOAA CSV columns: {path.name}")
            yield from reader


def finite(value, field: str, *, lower: float | None = None,
           upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"missing or invalid {field}")
    result = float(value)
    if not math.isfinite(result) or (lower is not None and result < lower) or (
            upper is not None and result > upper):
        raise ValueError(f"out-of-range {field}")
    return result


def load_era5(daily_payload: dict, snow_payload: dict) -> dict[str, dict]:
    for payload in (daily_payload, snow_payload):
        if payload.get("timezone") != "Asia/Shanghai" or payload.get("utc_offset_seconds") != 28800:
            raise ValueError("Open-Meteo inputs must use Asia/Shanghai")
    daily = daily_payload["daily"]
    expected = [(date(YEAR, 1, 1) + timedelta(days=i)).isoformat() for i in range(365)]
    if daily["time"] != expected:
        raise ValueError("ERA5 daily input must cover all 365 local dates in 2023")
    units = daily_payload["daily_units"]
    for field, unit in {"temperature_2m_mean": "°C", "precipitation_sum": "mm",
                        "snowfall_sum": "cm", "cloud_cover_mean": "%"}.items():
        if units.get(field) != unit or len(daily[field]) != 365:
            raise ValueError(f"invalid ERA5 field or units: {field}")
    hourly = snow_payload["hourly"]
    expected_hours = [(datetime(YEAR, 1, 1) + timedelta(hours=i)).isoformat(timespec="minutes")
                      for i in range(365 * 24)]
    if (hourly["time"] != expected_hours or len(hourly["snow_depth"]) != len(expected_hours)
            or snow_payload["hourly_units"].get("snow_depth") != "m"):
        raise ValueError("ERA5-Land snow input must contain 8760 local hours in meters")
    depths = [finite(v, "ERA5-Land snow_depth", lower=0) for v in hourly["snow_depth"]]
    result = {}
    for i, day in enumerate(expected):
        result[day] = {
            "temp_mean_c": finite(daily["temperature_2m_mean"][i], "ERA5 temperature", lower=-93.2, upper=61.8),
            "precipitation_mm": finite(daily["precipitation_sum"][i], "ERA5 precipitation", lower=0),
            "snowfall_cm": finite(daily["snowfall_sum"][i], "ERA5 snowfall", lower=0),
            "cloud_cover_pct": finite(daily["cloud_cover_mean"][i], "ERA5 cloud cover", lower=0, upper=100),
            "snow_depth_max_m": max(depths[i * 24:(i + 1) * 24]),
        }
    return result


def classify(*, rain: bool, cloud_cover_pct: float, snowfall_cm: float,
             snow_depth_max_m: float) -> str:
    if snowfall_cm > 0 or snow_depth_max_m > 0:
        return "snowy"
    if rain:
        return "rainy"
    return "sunny" if cloud_cover_pct <= 20 else "cloudy"


def build_rows(noaa: dict[str, dict], era5: dict[str, dict]) -> tuple[list, list]:
    rows, provenance = [], []
    for offset in range(365):
        source_date = date(YEAR, 1, 1) + timedelta(days=offset)
        key = source_date.isoformat()
        external = era5[key]
        station = noaa.get(key, {})
        temps = station.get("hourly_temperatures", [])
        reports = station.get("precipitation_reports", [])
        daily_mean = mean(temps) if temps else external["temp_mean_c"]
        raining = any(r["amount_mm"] > 0 for r in reports) if reports else external["precipitation_mm"] > 0
        condition = classify(rain=raining, cloud_cover_pct=external["cloud_cover_pct"],
                             snowfall_cm=external["snowfall_cm"],
                             snow_depth_max_m=external["snow_depth_max_m"])
        span = BASE_RANGES[source_date.month] + RANGE_ADJUSTMENTS[condition]
        rows.append({"day": offset + 1, **calendar_for_source_day(offset + 1),
                     "condition": condition, "temp_mean_c": round(daily_mean, 2),
                     "night_low_c": round(daily_mean - 0.5 * span, 2),
                     "day_high_c": round(daily_mean + 0.5 * span, 2),
                     "weather_source_date": key, "weather_source_condition": condition})
        provenance.append({
            "source_date": key, "condition": condition,
            "temperature_source": "noaa" if temps else "era5_fallback",
            "noaa_temperature_hours": len(temps),
            "precipitation_source": "noaa_shortest_reports" if reports else "era5_fallback",
            "noaa_precipitation_report_hours": len(reports),
            "noaa_precipitation_conflict_hours": station.get("precipitation_conflict_hours", 0),
            "selected_precipitation_reports": reports,
            "era5_cloud_cover_pct": external["cloud_cover_pct"],
            "era5_snowfall_cm": external["snowfall_cm"],
            "era5_land_snow_depth_max_m": external["snow_depth_max_m"],
        })
    return rows, provenance


def load_sources(source_dir: Path) -> tuple[list, list, dict]:
    manifest = json.loads((source_dir / "sources.json").read_text(encoding="utf-8"))
    if manifest["dataset_id"] != DATASET_ID or set(manifest["files"]) != set(SOURCE_FILES):
        raise ValueError("unrecognized public weather input manifest")
    for name in SOURCE_FILES:
        if sha256(source_dir / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"source integrity mismatch: {name}")
    noaa = aggregate_noaa(noaa_records(source_dir / n for n in SOURCE_FILES[:2]))
    era5 = load_era5(*(json.loads((source_dir / n).read_text(encoding="utf-8")) for n in SOURCE_FILES[2:]))
    rows, provenance = build_rows(noaa, era5)
    return rows, provenance, manifest


def metadata_for(metadata: dict, rows: list, manifest: dict) -> dict:
    # Use exactly the same JSONL representation as the runtime writer.
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in rows)
    result = json.loads(json.dumps(metadata))
    result["weather_dataset_id"] = DATASET_ID
    result["weather_source"] = "NOAA ISD Beijing Capital Airport; Open-Meteo ERA5 and ERA5-Land, 2023"
    result["weather_source_year"] = YEAR
    result["weather_days"] = 365
    result["weather_mapping"] = "same_month_day_historical_replay"
    result["weather_build"] = {
        "policy": BUILD_POLICY, "local_timezone": "Asia/Shanghai",
        "source_sha256": {n: manifest["files"][n]["sha256"] for n in SOURCE_FILES},
        "cloud_cover_sunny_threshold_pct": 20,
        "temperature": "mean_of_valid_hourly_means; ERA5 daily fallback only when none exist",
        "precipitation": "shortest_valid_accumulation_per_report_hour; any_positive_on_end_date; ERA5_missing_day_fallback",
        "snow": "positive_ERA5_snowfall_or_positive_ERA5_Land_daily_max_snow_depth",
        "temperature_extremes": "unchanged_V54_synthetic_month_and_condition_span",
    }
    result["content_sha256"].pop("weather_finalinfo.pkl", None)
    result["content_sha256"]["weather.jsonl"] = hashlib.sha256(payload.encode()).hexdigest()
    return result


def build(source_dir: Path, output_dir: Path, *, check: bool = False) -> dict:
    if (output_dir / "weather_finalinfo.pkl").exists():
        raise ValueError("use a clean public runtime directory without the legacy weather pickle")
    rows, provenance, manifest = load_sources(source_dir)
    meta_path = output_dir / "metadata.json"
    original = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata = metadata_for(original, rows, manifest)
    for name, digest in original["content_sha256"].items():
        if name not in {"weather.jsonl", "weather_finalinfo.pkl"}:
            if sha256(output_dir / name) != digest:
                raise ValueError(f"non-weather input integrity mismatch: {name}")
    if check:
        if sha256(output_dir / "weather.jsonl") != metadata["content_sha256"]["weather.jsonl"]:
            raise ValueError("weather.jsonl differs from the reproducible build")
        if original != metadata:
            raise ValueError("weather metadata differs from the reproducible build")
        saved = [json.loads(line) for line in (source_dir / "daily_provenance.jsonl").read_text().splitlines()]
        if saved != provenance:
            raise ValueError("daily weather provenance differs from the reproducible build")
    else:
        write_jsonl(output_dir / "weather.jsonl", rows)
        write_jsonl(source_dir / "daily_provenance.jsonl", provenance)
        write_json(meta_path, metadata)
    return {"dataset_id": DATASET_ID, "days": len(rows), "weather_sha256": metadata["content_sha256"]["weather.jsonl"],
            "condition_counts": dict(Counter(r["condition"] for r in rows)),
            "temperature_fallback_days": sum(r["temperature_source"] != "noaa" for r in provenance),
            "precipitation_fallback_days": sum(r["precipitation_source"] == "era5_fallback" for r in provenance),
            "check_only": check}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("data_sources/weather"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "runtime_data")
    parser.add_argument("--check", action="store_true", help="verify the frozen build without writing")
    args = parser.parse_args()
    print(json.dumps(build(args.sources, args.output_dir, check=args.check), indent=2))


if __name__ == "__main__":
    main()
