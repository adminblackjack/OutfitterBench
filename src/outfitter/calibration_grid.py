"""Legacy bounded-demand profit-scale search.

Exact-offer channel runtimes deliberately reject this profit-targeted grid; their
demand is frozen by mechanism/behavior gates before economic baseline QA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .calibrate import FORMAL_STRATEGIES, run_policy, scripted_economic_gate
from .config import OutfitterConfig


def evaluate_profit_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gate = scripted_economic_gate(rows)
    cells = list(gate["worlds"].values())
    if len(cells) == 1:
        gate.update(cells[0])
    gate["distance_to_role_boundaries"] = 0.0 if gate["passed"] else math.inf
    return gate


def _run_point(task: dict[str, Any]) -> dict[str, Any]:
    rows = [
        run_policy(
            OutfitterConfig(
                data_dir=task["data_dir"],
                max_days=task["days"],
                weather_start_day=task["weather_start_day"],
                seed=seed,
                daily_fee=task["daily_fee"],
                arrival_scale_multiplier=task["arrival_scale_multiplier"],
                initial_cash=task["initial_cash"],
                charge_model_usage_to_cash=False,
            ),
            strategy,
        )
        for strategy in FORMAL_STRATEGIES
        for seed in task["seeds"]
    ]
    daily_customers = statistics.median(
        float(row["customers"]) / max(1, int(row["days"])) for row in rows
    )
    gate = evaluate_profit_gate(rows)
    arrival_pass = 6.0 <= daily_customers <= 12.0
    return {
        "arrival_scale_multiplier": task["arrival_scale_multiplier"],
        "daily_fee": task["daily_fee"],
        "median_daily_customers": round(daily_customers, 6),
        "arrival_volume_pass": arrival_pass,
        "profit_gate": gate,
        "passed": gate["passed"] and arrival_pass,
        "runs": rows,
    }


def _point_key(row: dict[str, Any]) -> tuple[float, float]:
    return (
        round(float(row["arrival_scale_multiplier"]), 8),
        round(float(row["daily_fee"]), 8),
    )


def _read_completed_points(path: Path) -> dict[tuple[float, float], dict[str, Any]]:
    if not path.is_file():
        return {}
    completed: dict[tuple[float, float], dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            completed[_point_key(row)] = row
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid calibration checkpoint line {line_number}: {exc}"
            ) from exc
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description="Search V5 arrival/fee calibration grid")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--days", type=int, default=198)
    parser.add_argument("--weather-start-day", type=int, default=91)
    parser.add_argument("--initial-cash", type=float, default=6_000.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 42, 59, 73])
    parser.add_argument(
        "--arrival-scale-multipliers",
        type=float,
        nargs="+",
        default=[round(0.75 + 0.05 * index, 2) for index in range(11)],
    )
    parser.add_argument(
        "--daily-fees",
        type=float,
        nargs="+",
        default=list(range(60, 301, 20)),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    from .data import OutfitterDataset

    if OutfitterDataset.load(args.data_dir).channel_demand_model is not None:
        raise ValueError(
            "profit-targeted arrival/fee grids are disabled for exact-offer channel "
            "runtimes; freeze demand with Demand Lab behavior gates, then run "
            "outfitter.calibrate as economic QA"
        )
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "outfitter-v5-profit-scale-grid-manifest-v1",
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "days": args.days,
        "weather_start_day": args.weather_start_day,
        "initial_cash": args.initial_cash,
        "seeds": list(args.seeds),
        "arrival_scale_multipliers": list(args.arrival_scale_multipliers),
        "daily_fees": list(args.daily_fees),
    }
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise ValueError(
                "calibration output directory contains a checkpoint for a "
                "different grid manifest"
            )
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    tasks = [
        {
            "data_dir": str(Path(args.data_dir).expanduser().resolve()),
            "days": args.days,
            "weather_start_day": args.weather_start_day,
            "initial_cash": args.initial_cash,
            "seeds": list(args.seeds),
            "arrival_scale_multiplier": multiplier,
            "daily_fee": fee,
        }
        for multiplier in args.arrival_scale_multipliers
        for fee in args.daily_fees
    ]
    checkpoint_path = output / "grid_points.partial.jsonl"
    completed = _read_completed_points(checkpoint_path)
    expected_keys = {_point_key(task) for task in tasks}
    unexpected = set(completed) - expected_keys
    if unexpected:
        raise ValueError("calibration checkpoint contains points outside this grid")
    remaining_tasks = [task for task in tasks if _point_key(task) not in completed]
    if completed:
        print(
            f"[grid resume] completed={len(completed)} "
            f"remaining={len(remaining_tasks)}",
            flush=True,
        )
    with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(_run_point, task) for task in remaining_tasks]
            for index, future in enumerate(as_completed(futures), start=1):
                point = future.result()
                completed[_point_key(point)] = point
                checkpoint.write(
                    json.dumps(point, ensure_ascii=False, sort_keys=True) + "\n"
                )
                checkpoint.flush()
                os.fsync(checkpoint.fileno())
                print(
                    f"[grid {len(completed)}/{len(tasks)}] "
                    f"arrival={point['arrival_scale_multiplier']} "
                    f"fee={point['daily_fee']} passed={point['passed']}",
                    flush=True,
                )
    points = list(completed.values())
    if len(points) != len(tasks):
        raise RuntimeError(
            f"calibration grid incomplete: expected {len(tasks)}, got {len(points)}"
        )
    points.sort(key=lambda row: (row["arrival_scale_multiplier"], row["daily_fee"]))
    passing = [row for row in points if row["passed"]]
    selected = min(
        passing,
        key=lambda row: (
            row["profit_gate"]["distance_to_role_boundaries"],
            abs(row["arrival_scale_multiplier"] - 1.0),
            row["daily_fee"],
        ),
    ) if passing else None
    with (output / "grid_points.jsonl").open("w", encoding="utf-8") as handle:
        for row in points:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "format": "outfitter-v5-profit-scale-grid-v1",
        "points": len(points),
        "passing_points": len(passing),
        "selected": selected,
        "targets": "relative baseline roles only; no absolute profit caps",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
