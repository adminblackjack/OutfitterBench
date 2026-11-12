"""Plot rental-service contribution separately from shared and capital effects.

The main curve is an accounting contribution measure, not a causal estimate of
the value of enabling rentals.  It isolates recognized rental service economics
from retail merchandising, common operating overhead, and balance-sheet
revaluation.  A companion endpoint decomposition keeps those excluded effects
visible and reconciles exactly to official liquidation-value change.
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


FORMAT = "outfitter-rental-service-comparison-v1"
FIGURE_NAME = "rental_service_contribution.png"
DATA_NAME = "rental_service_figure_data.csv"
MANIFEST_NAME = "rental_service_figure_manifest.json"

RENTAL_COST_FIELDS = (
    "supplier_trial_handling_fee",
    "supplier_trial_usage_fee",
    "cleaning_cost",
    "repair_cost",
    "maintenance_cost",
    "rental_holding_cost",
    "rental_idle_holding_cost",
    "rental_use_depreciation",
    "reservation_compensation",
)


def _number(row: dict[str, Any], key: str) -> float:
    value = row.get(key, 0.0)
    return 0.0 if value in (None, "") else float(value)


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return result or "series"


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


def _rental_curve(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Build daily cumulative contribution and an exact accounting bridge."""
    initial_official = float(
        report["headline"]["initial_official_liquidation_value"]
    )
    cumulative = {
        "rental_service_contribution": 0.0,
        "retail_merchandise_contribution": 0.0,
        "shared_operating_overhead": 0.0,
        "retail_liquidation_effect": 0.0,
        "managed_operating_contribution": 0.0,
    }
    rows = [{
        "day": 0,
        **cumulative,
        "official_value_change": 0.0,
        "capital_obligation_reconciliation": 0.0,
    }]
    for point in report["daily"]:
        rental = (
            _number(point, "rental_revenue")
            + _number(point, "damage_deductions")
            - sum(_number(point, key) for key in RENTAL_COST_FIELDS)
        )
        retail = (
            _number(point, "retail_revenue")
            - _number(point, "retail_cogs")
            - _number(point, "retail_holding_cost")
        )
        shared_overhead = -_number(point, "operating_fee")
        retail_liquidation = -_number(point, "retail_liquidation_loss")
        managed = rental + retail + shared_overhead + retail_liquidation

        reported_operating = _number(point, "daily_operating_pnl")
        reported_managed = reported_operating + retail_liquidation
        if abs(managed - reported_managed) > 0.025:
            raise ValueError(
                "rental-service decomposition does not reconcile with daily "
                f"operating fields on day {point['day']}: "
                f"components={managed:.4f}, reported={reported_managed:.4f}"
            )

        cumulative["rental_service_contribution"] += rental
        cumulative["retail_merchandise_contribution"] += retail
        cumulative["shared_operating_overhead"] += shared_overhead
        cumulative["retail_liquidation_effect"] += retail_liquidation
        cumulative["managed_operating_contribution"] += managed
        official = _number(point, "official_liquidation_value_end") - initial_official
        rows.append({
            "day": int(point["day"]),
            **{key: round(value, 2) for key, value in cumulative.items()},
            "official_value_change": round(official, 2),
            "capital_obligation_reconciliation": round(
                official - cumulative["managed_operating_contribution"], 2
            ),
        })
    if len(rows) == 1:
        raise ValueError("analysis report contains no completed days")
    return rows


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
        curve = _rental_curve(report)
        summer = _summer_start(report)
        if summer is not None:
            summer_days.add(summer)
        raw_runs.append({
            "label": label,
            "seed": seed,
            "run_id": str(report["run"]["run_id"]),
            "model": str(report["run"]["model"]),
            "status": status,
            "completed_day": int(curve[-1]["day"]),
            "bankrupt": status == "bankrupt",
            "curve": curve,
        })
    if (
        len(seeds_by_label) > 1
        and len({tuple(sorted(seeds)) for seeds in seeds_by_label.values()}) != 1
    ):
        raise ValueError("model labels must use the same seed set")
    if len(summer_days) > 1:
        raise ValueError("runs disagree on summer start")

    decision_day = int(entries[0][1]["headline"]["decision_horizon_day"])
    runoff_day = int(entries[0][1]["headline"]["runoff_end_day"])
    all_complete = all(run["status"] != "in_progress" for run in raw_runs)
    for run in raw_runs:
        if (
            not run["bankrupt"]
            and run["status"] != "in_progress"
            and run["completed_day"] != runoff_day
        ):
            raise ValueError("non-bankrupt completed run must reach runoff_end_day")
    common_day = (
        runoff_day
        if all_complete
        else min(
            run["completed_day"]
            for run in raw_runs
            if run["status"] == "in_progress"
        )
    )

    curve_rows = []
    for run in raw_runs:
        included = [row for row in run["curve"] if row["day"] <= common_day]
        if run["bankrupt"] and included[-1]["day"] < common_day:
            terminal = included[-1]
            included.extend(
                {**terminal, "day": day, "absorbing_bankruptcy_extension": True}
                for day in range(int(terminal["day"]) + 1, common_day + 1)
            )
        for row in included:
            curve_rows.append({
                "label": run["label"],
                "seed": run["seed"],
                "run_id": run["run_id"],
                "absorbing_bankruptcy_extension": bool(
                    row.get("absorbing_bankruptcy_extension", False)
                ),
                **{key: value for key, value in row.items()
                   if key != "absorbing_bankruptcy_extension"},
            })

    numeric_fields = (
        "rental_service_contribution",
        "retail_merchandise_contribution",
        "shared_operating_overhead",
        "retail_liquidation_effect",
        "managed_operating_contribution",
        "official_value_change",
        "capital_obligation_reconciliation",
    )
    grouped: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in curve_rows:
        grouped[(row["label"], int(row["day"]))].append(row)
    aggregates = []
    for (label, day), rows in sorted(grouped.items()):
        result: dict[str, Any] = {"label": label, "day": day, "seeds": len(rows)}
        for field in numeric_fields:
            values = [float(row[field]) for row in rows]
            result[f"{field}_mean"] = round(statistics.mean(values), 6)
            result[f"{field}_min"] = min(values)
            result[f"{field}_max"] = max(values)
        aggregates.append(result)

    endpoints = []
    for label in seeds_by_label:
        matches = [
            row for row in aggregates
            if row["label"] == label and row["day"] == common_day
        ]
        if len(matches) != 1:
            raise AssertionError("missing aggregate endpoint")
        endpoints.append(matches[0])
    return {
        "format": FORMAT,
        "status": "complete" if all_complete else "in_progress",
        "common_completed_day": common_day,
        "decision_horizon_day": decision_day,
        "runoff_end_day": runoff_day,
        "summer_start_day": next(iter(summer_days), None),
        "environment_signature": next(iter(signatures)),
        "seed_sets": {label: sorted(seeds) for label, seeds in seeds_by_label.items()},
        "metric": {
            "curve": "cumulative_net_rental_service_contribution",
            "curve_label": (
                "Cumulative net rental-service contribution "
                "(before shared overhead and fleet revaluation)"
            ),
            "recognized_rental_revenue_net_of_optional_credits": True,
            "included_costs": list(RENTAL_COST_FIELDS),
            "excluded_from_curve": [
                "retail merchandising",
                "shared operating fee",
                "durable purchase principal",
                "asset-sale proceeds",
                "deposits and refunds",
                "model fees",
                "balance-sheet revaluation",
            ],
            "causal_effect_claimed": False,
            "contract_adjusted_value_used": False,
        },
        "runs": [{key: value for key, value in run.items() if key != "curve"}
                 for run in raw_runs],
        "curves": curve_rows,
        "aggregates": aggregates,
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
    color_by_label = {label: colors[index % len(colors)] for index, label in enumerate(labels)}
    fig, (curve_ax, bridge_ax) = plt.subplots(
        1, 2, figsize=(16.2, 7.2), gridspec_kw={"width_ratios": [3.25, 1.55]},
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.82, wspace=0.34)

    for label in labels:
        color = color_by_label[label]
        runs = [run for run in comparison["runs"] if run["label"] == label]
        for run in runs:
            points = [
                row for row in comparison["curves"] if row["run_id"] == run["run_id"]
            ]
            if len(runs) > 1:
                curve_ax.plot(
                    [row["day"] for row in points],
                    [row["rental_service_contribution"] for row in points],
                    color=color, linewidth=0.8, alpha=0.22,
                )
        aggregate = [row for row in comparison["aggregates"] if row["label"] == label]
        days = [row["day"] for row in aggregate]
        low = [row["rental_service_contribution_min"] for row in aggregate]
        high = [row["rental_service_contribution_max"] for row in aggregate]
        if len(runs) > 1:
            curve_ax.fill_between(days, low, high, color=color, alpha=0.12, linewidth=0)
        values = [row["rental_service_contribution_mean"] for row in aggregate]
        curve_ax.plot(days, values, color=color, linewidth=2.6, label=label)
        curve_ax.scatter([days[-1]], [values[-1]], color=color, s=36, zorder=4)

    summer = comparison.get("summer_start_day")
    decision = int(comparison["decision_horizon_day"])
    runoff = int(comparison["runoff_end_day"])
    if summer is not None:
        curve_ax.axvline(summer, color="#7f8c8d", linestyle=":", linewidth=1)
        curve_ax.text(
            summer, 0.99, "Summer starts", rotation=90, va="top", ha="right",
            transform=curve_ax.get_xaxis_transform(), fontsize=8, color="#59636b",
        )
    curve_ax.axvline(decision, color="#555555", linestyle="--", linewidth=1)
    curve_ax.axvspan(decision, runoff, color="#adb5bd", alpha=0.18, label="Runoff")
    curve_ax.axhline(0, color="black", linewidth=0.8)
    curve_ax.set_xlim(0, runoff)
    curve_ax.set_title("Rental-service contribution over time")
    curve_ax.set_xlabel("Simulation day")
    curve_ax.set_ylabel("Cumulative net rental-service contribution (CNY)")
    curve_ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    curve_ax.grid(alpha=0.2)
    curve_ax.legend(frameon=False, loc="best")

    components = (
        ("rental_service_contribution_mean", "Rental service", "#0072B2"),
        ("retail_merchandise_contribution_mean", "Retail merchandise", "#E69F00"),
        ("shared_operating_overhead_mean", "Shared overhead", "#999999"),
        ("retail_liquidation_effect_mean", "Retail liquidation", "#CC79A7"),
        ("capital_obligation_reconciliation_mean", "Capital & obligations", "#D55E00"),
    )
    for y, endpoint in enumerate(reversed(comparison["endpoints"])):
        positive = 0.0
        negative = 0.0
        for field, component_label, color in components:
            value = float(endpoint[field])
            left = positive if value >= 0 else negative
            bridge_ax.barh(
                y, value, left=left, color=color, alpha=0.88,
                label=component_label if y == 0 else None,
            )
            if value >= 0:
                positive += value
            else:
                negative += value
        official = float(endpoint["official_value_change_mean"])
        bridge_ax.scatter(
            [official], [y], marker="D", s=72,
            facecolors="none" if comparison["status"] != "complete" else "black",
            edgecolors="black", linewidths=1.7, zorder=5,
            label=("Provisional official value" if comparison["status"] != "complete"
                   else "Final official value") if y == 0 else None,
        )
    bridge_ax.set_yticks(range(len(labels)), list(reversed(labels)))
    bridge_ax.axvline(0, color="black", linewidth=0.8)
    bridge_ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    bridge_ax.set_xlabel("Cumulative contribution / value change (CNY)")
    bridge_ax.set_title(
        f"{'Provisional' if comparison['status'] != 'complete' else 'Final'} "
        f"endpoint decomposition\nDay {comparison['common_completed_day']}"
    )
    bridge_ax.grid(axis="x", alpha=0.2)
    bridge_ax.legend(frameon=False, fontsize=7.8, loc="best")

    qualifier = (
        "IN PROGRESS — endpoint decomposition and hollow official marker are provisional."
        if comparison["status"] != "complete"
        else "Completed through runoff; endpoint decomposition is final."
    )
    fig.suptitle("OutfitterBench rental-management comparison", fontsize=16, y=0.96)
    fig.text(
        0.5, 0.035,
        "Rental-service contribution = recognized rent + damage deductions - cleaning - repair "
        "- maintenance - rental holding - use depreciation - reservation compensation.\n"
        "It is an accounting contribution, not a causal mechanism effect or complete business "
        f"profit. The endpoint bars reconcile excluded effects to official value change. {qualifier}",
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
        **{key: value for key, value in comparison.items()
           if key not in {"curves", "aggregates"}},
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
    result = generate(
        [(label, Path(path)) for label, path in args.run],
        args.output_dir,
        allow_in_progress=args.allow_in_progress,
        dpi=args.dpi,
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
