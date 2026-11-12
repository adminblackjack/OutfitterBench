"""Render operating-contribution curves with a separate capital reconciliation.

This is a presentation layer over the authoritative analyzer.  It never uses
contract-adjusted economic value as the management-performance time series.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
from typing import Any

from .analyze import analyze_run
from .csvio import write_csv


FORMAT = "outfitter-paper-management-comparison-v1"
FIGURE_NAME = "paper_management_comparison.png"
DATA_NAME = "figure_data.csv"
MANIFEST_NAME = "figure_manifest.json"


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return result or "series"


def _managed_curve(report: dict[str, Any]) -> list[dict[str, Any]]:
    initial = float(report["headline"]["initial_official_liquidation_value"])
    cumulative = 0.0
    rows = [{
        "day": 0,
        "managed_operating_contribution": 0.0,
        "official_value_change": 0.0,
    }]
    for point in report["daily"]:
        cumulative += float(point["daily_operating_pnl"]) - float(
            point.get("retail_liquidation_loss", 0.0)
        )
        reported = point.get("cumulative_business_pnl_after_liquidation")
        managed = cumulative if reported is None else float(reported)
        if abs(managed - cumulative) > 0.025:
            raise ValueError(
                "cumulative managed contribution does not reconcile with daily fields"
            )
        rows.append({
            "day": int(point["day"]),
            "managed_operating_contribution": round(managed, 2),
            "official_value_change": round(
                float(point["official_liquidation_value_end"]) - initial, 2
            ),
        })
    if len(rows) == 1:
        raise ValueError("analysis report contains no completed days")
    return rows


def _environment_signature(report: dict[str, Any]) -> str:
    run = report["run"]
    headline = report["headline"]
    payload = {
        "environment_rules": run["environment_rules"],
        "max_days": int(run["max_days"]),
        "decision_horizon_day": int(headline["decision_horizon_day"]),
        "runoff_end_day": int(headline["runoff_end_day"]),
        "initial_official_liquidation_value": float(
            headline["initial_official_liquidation_value"]
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _summer_start(report: dict[str, Any]) -> int | None:
    for row in report["daily"]:
        calendar = row.get("calendar") or {}
        if isinstance(calendar, str):
            try:
                calendar = json.loads(calendar)
            except json.JSONDecodeError:
                calendar = {}
        if calendar.get("holiday_name") == "暑假":
            return int(row["day"])
    return None


def build_comparison(
    entries: list[tuple[str, dict[str, Any]]], *, allow_in_progress: bool = False
) -> dict[str, Any]:
    if not entries:
        raise ValueError("at least one run is required")
    signatures = {_environment_signature(report) for _, report in entries}
    if len(signatures) != 1:
        raise ValueError("runs do not share one comparable environment")

    raw_runs = []
    seeds_by_label: defaultdict[str, set[int]] = defaultdict(set)
    summer_days = set()
    for label, report in entries:
        status = str(report["headline"]["status"])
        if status == "in_progress" and not allow_in_progress:
            raise ValueError("in-progress run requires --allow-in-progress")
        seed = int(report["run"]["seed"])
        if seed in seeds_by_label[label]:
            raise ValueError(f"duplicate seed {seed} for label {label!r}")
        seeds_by_label[label].add(seed)
        curve = _managed_curve(report)
        summer = _summer_start(report)
        if summer is not None:
            summer_days.add(summer)
        raw_runs.append({
            "label": label,
            "seed": seed,
            "run_id": str(report["run"]["run_id"]),
            "model": str(report["run"]["model"]),
            "status": status,
            "completed_day": curve[-1]["day"],
            "bankrupt": status == "bankrupt",
            "curve": curve,
        })
    if len(seeds_by_label) > 1 and len({tuple(sorted(v)) for v in seeds_by_label.values()}) != 1:
        raise ValueError("model labels must use the same seed set")
    if len(summer_days) > 1:
        raise ValueError("runs disagree on summer start")

    decision_day = int(entries[0][1]["headline"]["decision_horizon_day"])
    runoff_day = int(entries[0][1]["headline"]["runoff_end_day"])
    all_complete = all(row["status"] != "in_progress" for row in raw_runs)
    for row in raw_runs:
        if not row["bankrupt"] and row["status"] != "in_progress" and row["completed_day"] != runoff_day:
            raise ValueError("non-bankrupt completed run must reach runoff_end_day")
    common_day = (
        runoff_day
        if all_complete
        else min(
            row["completed_day"]
            for row in raw_runs
            if row["status"] == "in_progress"
        )
    )

    curve_rows = []
    for run in raw_runs:
        included = [point for point in run["curve"] if point["day"] <= common_day]
        if run["bankrupt"] and included[-1]["day"] < common_day:
            terminal = included[-1]
            included.extend(
                {
                    **terminal,
                    "day": day,
                    "absorbing_bankruptcy_extension": True,
                }
                for day in range(int(terminal["day"]) + 1, common_day + 1)
            )
        for point in run["curve"]:
            point.setdefault("absorbing_bankruptcy_extension", False)
        for point in included:
            curve_rows.append({
                "label": run["label"],
                "seed": run["seed"],
                "run_id": run["run_id"],
                **point,
            })

    aggregate_rows = []
    by_label_day: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in curve_rows:
        by_label_day[(row["label"], row["day"])].append(row)
    for (label, day), rows in sorted(by_label_day.items()):
        managed = [float(row["managed_operating_contribution"]) for row in rows]
        official = [float(row["official_value_change"]) for row in rows]
        aggregate_rows.append({
            "label": label,
            "day": day,
            "seeds": len(rows),
            "managed_mean": round(statistics.mean(managed), 6),
            "managed_min": min(managed),
            "managed_max": max(managed),
            "official_mean": round(statistics.mean(official), 6),
            "official_min": min(official),
            "official_max": max(official),
        })

    endpoints = []
    for label in seeds_by_label:
        rows = [
            row for row in aggregate_rows
            if row["label"] == label and row["day"] == common_day
        ]
        if len(rows) != 1:
            raise AssertionError("missing aggregate endpoint")
        row = rows[0]
        endpoints.append({
            "label": label,
            "day": common_day,
            "managed_mean": row["managed_mean"],
            "official_mean": row["official_mean"],
            "capital_reconciliation_mean": round(
                row["official_mean"] - row["managed_mean"], 6
            ),
        })
    return {
        "format": FORMAT,
        "status": "complete" if all_complete else "in_progress",
        "common_completed_day": common_day,
        "decision_horizon_day": decision_day,
        "runoff_end_day": runoff_day,
        "summer_start_day": next(iter(summer_days), None),
        "environment_signature": next(iter(signatures)),
        "seed_sets": {key: sorted(value) for key, value in seeds_by_label.items()},
        "metric": {
            "curve": "cumulative_business_pnl_after_liquidation",
            "curve_label": "Cumulative managed operating contribution",
            "capital_endpoint": "official_liquidation_value_change",
            "contract_adjusted_value_used": False,
            "model_cost_included": False,
        },
        "runs": [{k: v for k, v in row.items() if k != "curve"} for row in raw_runs],
        "curves": curve_rows,
        "aggregates": aggregate_rows,
        "endpoints": endpoints,
    }


def render_figure(comparison: dict[str, Any], path: Path, *, dpi: int = 160) -> None:
    if not 72 <= dpi <= 600:
        raise ValueError("dpi must be in [72, 600]")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    labels = list(comparison["seed_sets"])
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    color_by_label = {label: colors[i % len(colors)] for i, label in enumerate(labels)}
    fig, (curve_ax, bridge_ax) = plt.subplots(
        1, 2, figsize=(15.5, 6.8), gridspec_kw={"width_ratios": [3.2, 1.35]},
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.19, top=0.82, wspace=0.34)
    for label in labels:
        color = color_by_label[label]
        run_rows = [row for row in comparison["runs"] if row["label"] == label]
        for run in run_rows:
            points = [
                row for row in comparison["curves"]
                if row["run_id"] == run["run_id"]
            ]
            if len(run_rows) > 1:
                curve_ax.plot(
                    [row["day"] for row in points],
                    [row["managed_operating_contribution"] for row in points],
                    color=color, linewidth=0.8, alpha=0.22,
                )
        aggregate = [row for row in comparison["aggregates"] if row["label"] == label]
        days = [row["day"] for row in aggregate]
        low = [row["managed_min"] for row in aggregate]
        high = [row["managed_max"] for row in aggregate]
        if len(run_rows) > 1:
            curve_ax.fill_between(days, low, high, color=color, alpha=0.12, linewidth=0)
        curve_ax.plot(
            days, [row["managed_mean"] for row in aggregate],
            color=color, linewidth=2.5, label=label,
        )
        curve_ax.scatter(
            [days[-1]], [aggregate[-1]["managed_mean"]], color=color, s=34, zorder=4
        )

    summer = comparison.get("summer_start_day")
    decision = int(comparison["decision_horizon_day"])
    runoff = int(comparison["runoff_end_day"])
    if summer is not None:
        curve_ax.axvline(summer, color="#7f8c8d", linestyle=":", linewidth=1)
        curve_ax.text(summer, 0.99, "Summer starts", rotation=90, va="top", ha="right",
                      transform=curve_ax.get_xaxis_transform(), fontsize=8, color="#59636b")
    curve_ax.axvline(decision, color="#555555", linestyle="--", linewidth=1)
    curve_ax.axvspan(decision, runoff, color="#adb5bd", alpha=0.18, label="Runoff")
    curve_ax.axhline(0, color="black", linewidth=0.8)
    curve_ax.set_xlim(0, runoff)
    curve_ax.set_title("Operating performance over time")
    curve_ax.set_xlabel("Simulation day")
    curve_ax.set_ylabel("Cumulative managed operating contribution (CNY)")
    curve_ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    curve_ax.grid(alpha=0.2)
    curve_ax.legend(frameon=False, loc="best")

    provisional = comparison["status"] != "complete"
    for index, endpoint in enumerate(reversed(comparison["endpoints"])):
        label = endpoint["label"]
        color = color_by_label[label]
        managed = float(endpoint["managed_mean"])
        official = float(endpoint["official_mean"])
        bridge_ax.plot([managed, official], [index, index], color=color,
                       linestyle="--" if provisional else "-", linewidth=1.8, alpha=0.8)
        bridge_ax.scatter([managed], [index], marker="o", s=70, color=color,
                          label="Operating contribution" if index == 0 else None, zorder=3)
        bridge_ax.scatter([official], [index], marker="D", s=62,
                          facecolors="none" if provisional else color,
                          edgecolors=color, linewidths=1.8,
                          label=("Provisional official value" if provisional else "Final official value")
                          if index == 0 else None, zorder=3)
    bridge_ax.set_yticks(range(len(labels)), list(reversed(labels)))
    bridge_ax.axvline(0, color="black", linewidth=0.8)
    bridge_ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    bridge_ax.set_xlabel("Change from opening value (CNY)")
    bridge_ax.set_title(
        f"{'Provisional' if provisional else 'Final'} accounting reconciliation\n"
        f"Day {comparison['common_completed_day']}"
    )
    bridge_ax.grid(axis="x", alpha=0.2)
    bridge_ax.legend(frameon=False, fontsize=8, loc="best")

    qualifier = (
        "IN PROGRESS — the hollow official-value marker is not a final score."
        if provisional else
        "Completed through runoff; official-value markers are final."
    )
    fig.suptitle(
        "OutfitterBench management-performance comparison", fontsize=16, y=0.96
    )
    fig.text(
        0.5, 0.035,
        "Curve excludes durable purchase principal, asset-sale proceeds, deposits, and model fees.\n"
        "It includes use depreciation, holding, cleaning, repair, maintenance, operating fees, "
        f"reservation compensation, and retail liquidation losses. {qualifier}",
        ha="center", va="bottom", fontsize=8.2, color="#495057", linespacing=1.35,
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate(
    specs: list[tuple[str, Path]], output_dir: Path, *, allow_in_progress: bool, dpi: int
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "analysis_inputs"
    inputs.mkdir()
    entries = []
    sources = []
    for index, (label, run_dir) in enumerate(specs, 1):
        snapshot = inputs / f"{index:02d}-{_slug(label)}"
        result = analyze_run(run_dir, output_dir=snapshot, plots=False, dpi=dpi)
        report_path = snapshot / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        entries.append((label, report))
        sources.append({
            "label": label,
            "run_dir": str(run_dir.expanduser().resolve()),
            "run_id": result["run_id"],
            "snapshot_report": str(report_path),
            "snapshot_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "completed_days": int(report["headline"]["completed_days"]),
            "status": str(report["headline"]["status"]),
        })
    comparison = build_comparison(entries, allow_in_progress=allow_in_progress)
    write_csv(output / DATA_NAME, comparison["curves"])
    manifest = {
        **{k: v for k, v in comparison.items() if k not in {"curves", "aggregates"}},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "figure": FIGURE_NAME,
        "figure_data": DATA_NAME,
    }
    (output / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    render_figure(comparison, output / FIGURE_NAME, dpi=dpi)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", nargs=2, metavar=("LABEL", "RUN_DIR"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-in-progress", action="store_true")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    specs = [(label, Path(path)) for label, path in args.run]
    result = generate(
        specs, args.output_dir, allow_in_progress=args.allow_in_progress, dpi=args.dpi
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "figure": result["figure"],
        "status": result["status"],
        "common_completed_day": result["common_completed_day"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
