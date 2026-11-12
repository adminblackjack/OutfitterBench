"""Download public weather inputs into a new snapshot, without credentials.

The committed snapshot is preferred for exact rebuilding. Upstream revisions or
API metadata can change fresh downloads; this command never overwrites a frozen
manifest. An existing complete manifest is verified rather than refreshed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode

import requests

DATASET_ID = "beijing-2023-noaa-era5-v1"
BASE = "https://archive-api.open-meteo.com/v1/archive"
COMMON = {"latitude": "39.9042", "longitude": "116.4074",
          "start_date": "2023-01-01", "end_date": "2023-12-31", "timezone": "Asia/Shanghai"}
SOURCES = {
    f"noaa-54511099999-{year}.csv.gz": {
        "url": f"https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/54511099999.csv",
        "provider": "NOAA Integrated Surface Database",
        "license_url": "https://registry.opendata.aws/noaa-isd/",
        "stored_as": "gzip of the complete original CSV; mtime=0",
    } for year in (2022, 2023)
}
for name, params in (
    ("open-meteo-era5-beijing-2023.json", {**COMMON, "models": "era5",
     "daily": "temperature_2m_mean,precipitation_sum,snowfall_sum,cloud_cover_mean", "hourly": "snow_depth"}),
    ("open-meteo-era5-land-snow-beijing-2023.json", {**COMMON, "models": "era5_land", "hourly": "snow_depth"}),
):
    SOURCES[name] = {
        "url": BASE + "?" + urlencode(params),
        "provider": "Open-Meteo / Copernicus ERA5" if params["models"] == "era5" else "Open-Meteo / Copernicus ERA5-Land",
        "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution_url": "https://open-meteo.com/", "parameters": params,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("data_sources/weather"))
    p.add_argument("--record-existing", action="store_true", help="record already-downloaded inputs from these exact public URLs")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "sources.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("dataset_id") != DATASET_ID or set(manifest["files"]) != set(SOURCES):
            raise ValueError("unexpected existing manifest")
        for name, info in manifest["files"].items():
            if hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest() != info["sha256"]:
                raise ValueError(f"snapshot integrity mismatch: {name}")
        print("Existing frozen public inputs verified; no download or overwrite.")
        return
    files = {}
    for name, info in SOURCES.items():
        dest = args.output_dir / name
        if args.record_existing:
            data = dest.read_bytes()
        else:
            if dest.exists():
                raise FileExistsError(f"use a new output directory: {dest}")
            r = requests.get(info["url"], timeout=(15, 120))
            r.raise_for_status()
            data = gzip.compress(r.content, mtime=0) if name.endswith(".gz") else r.content
            with dest.open("xb") as stream:
                stream.write(data)
        if name.endswith(".gz"):
            if not gzip.decompress(data).startswith(b'"STATION","DATE"'):
                raise ValueError(f"not a NOAA CSV: {name}")
        else:
            payload = json.loads(data)
            if payload.get("error") or payload.get("timezone") != "Asia/Shanghai":
                raise ValueError(f"not the requested Open-Meteo response: {name}")
        files[name] = {**info, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    manifest = {"dataset_id": DATASET_ID, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "files": files, "note": "2022 NOAA input supplies the UTC/local-year boundary; only local 2023 enters the build."}
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Recorded {len(files)} public source snapshots.")


if __name__ == "__main__":
    main()
