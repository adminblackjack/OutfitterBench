"""Summarize two frozen weather inputs without copying the reference records.

The caller supplies a legally held reference. This utility is not used by the
public data builder and does not modify either input.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean


def comparison(reference: Path, candidate: Path) -> dict:
    def read(path):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if [r["day"] for r in rows] != list(range(1, 366)):
            raise ValueError("expected a complete, ordered 365-day weather input")
        return rows
    old, new = read(reference), read(candidate)
    result = {"reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
              "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(), "windows": {}}
    for label, start, end in (("decision", 91, 290), ("full_season", 91, 304), ("full_year", 1, 365)):
        pairs = list(zip(old[start-1:end], new[start-1:end], strict=True))
        errors = [abs(a["temp_mean_c"]-b["temp_mean_c"]) for a,b in pairs]
        result["windows"][label] = {
            "days": len(pairs), "source_day_from": start, "source_day_through": end,
            "category_matches": sum(a["condition"] == b["condition"] for a,b in pairs),
            "temperature_mean_absolute_error_c": mean(errors),
            "temperature_max_absolute_error_c": max(errors),
            "temperature_within_0_2c_days": sum(e <= .2 + 1e-9 for e in errors),
            "reference_category_counts": dict(Counter(a["condition"] for a,_ in pairs)),
            "candidate_category_counts": dict(Counter(b["condition"] for _,b in pairs)),
            "calendar_matches": all(all(a[k] == b[k] for k in ("date","day_type","weekday","is_weekend","holiday_id")) for a,b in pairs),
        }
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--candidate", default=Path("src/outfitter/runtime_data/weather.jsonl"), type=Path)
    p.add_argument("--output", type=Path)
    args=p.parse_args()
    text=json.dumps(comparison(args.reference,args.candidate),ensure_ascii=False,indent=2,sort_keys=True)+"\n"
    if args.output:
        if args.output.resolve() in {args.reference.resolve(), args.candidate.resolve()}:
            raise ValueError("output must not overwrite a weather input")
        with args.output.open("x",encoding="utf-8") as stream: stream.write(text)
    print(text,end="")


if __name__=="__main__":main()
