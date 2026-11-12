"""Render comparable rental and reservation fulfillment diagnostics.

Customer choice and physical service execution are deliberately separated.
The rental operational rate excludes outside choices; the reservation honor
rate excludes customer cancellations and commitments not yet resolved.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
from typing import Any

from .analyze import analyze_run
from .csvio import write_csv


FORMAT = "outfitter-fulfillment-comparison-v1"
FIGURE_NAME = "fulfillment_analysis.png"
DATA_NAME = "fulfillment_figure_data.csv"
MANIFEST_NAME = "fulfillment_figure_manifest.json"
ROLLING_DAYS = 14


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


def _int(row: dict[str, Any], key: str) -> int:
    value = row.get(key, 0)
    return 0 if value in (None, "") else int(value)


def _fulfillment_curve(report: dict[str, Any]) -> list[dict[str, Any]]:
    final_day = max(int(row["day"]) for row in report["daily"])
    service_by_day: defaultdict[int, Counter[str]] = defaultdict(Counter)
    for row in report.get("service_daily", []):
        if not str(row.get("service_key", "")).startswith("rental:"):
            continue
        day = int(row["day"])
        for key in (
            "requested_units",
            "fulfilled_units",
            "outside_rejections",
            "stockout_rejections",
            "service_closed_rejections",
            "capacity_policy_rejections",
            "horizon_rejections",
        ):
            service_by_day[day][key] += _int(row, key)

    events_by_day: defaultdict[int, Counter[str]] = defaultdict(Counter)
    for event in report.get("reservation_events", []):
        events_by_day[int(event["day"])][str(event["event_type"])] += 1

    rolling: deque[Counter[str]] = deque()
    rolling_totals: Counter[str] = Counter()
    cumulative_service: Counter[str] = Counter()
    cumulative_events: Counter[str] = Counter()
    rows = []
    for day in range(1, final_day + 1):
        daily = service_by_day[day]
        rolling.append(daily)
        rolling_totals.update(daily)
        if len(rolling) > ROLLING_DAYS:
            expired = rolling.popleft()
            rolling_totals.subtract(expired)
        cumulative_service.update(daily)
        cumulative_events.update(events_by_day[day])

        rolling_denominator = sum(
            rolling_totals[key]
            for key in (
                "fulfilled_units",
                "stockout_rejections",
                "service_closed_rejections",
                "capacity_policy_rejections",
            )
        )
        cumulative_denominator = sum(
            cumulative_service[key]
            for key in (
                "fulfilled_units",
                "stockout_rejections",
                "service_closed_rejections",
                "capacity_policy_rejections",
            )
        )
        reservation_success = cumulative_events["pickup_fulfilled"]
        manager_cancel = cumulative_events["cancelled_by_manager"]
        pickup_failure = cumulative_events["pickup_failed"]
        reservation_denominator = reservation_success + manager_cancel + pickup_failure
        customer_cancel = sum(
            count for event_type, count in cumulative_events.items()
            if event_type.startswith("customer_cancelled")
        )
        confirmed = (
            cumulative_events["reservation_confirmed_from_customer_plan_set"]
            + cumulative_events["reservation_confirmed"]
        )
        pending = max(
            0,
            confirmed
            - reservation_success
            - manager_cancel
            - pickup_failure
            - customer_cancel,
        )
        eligible = max(
            0,
            cumulative_service["requested_units"]
            - cumulative_service["horizon_rejections"],
        )
        rows.append({
            "day": day,
            "rolling_operational_numerator": rolling_totals["fulfilled_units"],
            "rolling_operational_denominator": rolling_denominator,
            "rolling_operational_fulfillment_rate": (
                rolling_totals["fulfilled_units"] / rolling_denominator
                if rolling_denominator else None
            ),
            "cumulative_operational_numerator": cumulative_service["fulfilled_units"],
            "cumulative_operational_denominator": cumulative_denominator,
            "cumulative_operational_fulfillment_rate": (
                cumulative_service["fulfilled_units"] / cumulative_denominator
                if cumulative_denominator else None
            ),
            "cumulative_eligible_fulfillment_rate": (
                cumulative_service["fulfilled_units"] / eligible if eligible else None
            ),
            "rental_requested_units": cumulative_service["requested_units"],
            "rental_fulfilled_units": cumulative_service["fulfilled_units"],
            "rental_outside_rejections": cumulative_service["outside_rejections"],
            "rental_stockout_rejections": cumulative_service["stockout_rejections"],
            "rental_service_closed_rejections": cumulative_service[
                "service_closed_rejections"
            ],
            "rental_capacity_policy_rejections": cumulative_service[
                "capacity_policy_rejections"
            ],
            "rental_horizon_rejections": cumulative_service["horizon_rejections"],
            "reservation_confirmed": confirmed,
            "reservation_successful_pickups": reservation_success,
            "reservation_customer_cancellations": customer_cancel,
            "reservation_manager_cancellations": manager_cancel,
            "reservation_pickup_failures": pickup_failure,
            "reservation_pending": pending,
            "reservation_resolved_manager_controlled": reservation_denominator,
            "reservation_commitment_honor_rate": (
                reservation_success / reservation_denominator
                if reservation_denominator else None
            ),
        })
    return rows


def build_comparison(
    entries: list[tuple[str, dict[str, Any]]], *, allow_in_progress: bool = False
) -> dict[str, Any]:
    if not entries:
        raise ValueError("at least one run is required")
    signatures = {_environment_signature(report) for _, report in entries}
    if len(signatures) != 1:
        raise ValueError("runs do not share one comparable environment")
    seeds_by_label: defaultdict[str, set[int]] = defaultdict(set)
    runs = []
    summer_days = set()
    for label, report in entries:
        status = str(report["headline"]["status"])
        if status == "in_progress" and not allow_in_progress:
            raise ValueError("in-progress run requires --allow-in-progress")
        seed = int(report["run"]["seed"])
        if seed in seeds_by_label[label]:
            raise ValueError(f"duplicate seed {seed} for label {label!r}")
        seeds_by_label[label].add(seed)
        curve = _fulfillment_curve(report)
        summer = _summer_start(report)
        if summer is not None:
            summer_days.add(summer)
        runs.append({
            "label": label,
            "seed": seed,
            "run_id": str(report["run"]["run_id"]),
            "model": str(report["run"]["model"]),
            "status": status,
            "completed_day": int(curve[-1]["day"]),
            "curve": curve,
        })
    if (
        len(seeds_by_label) > 1
        and len({tuple(sorted(seeds)) for seeds in seeds_by_label.values()}) != 1
    ):
        raise ValueError("model labels must use the same seed set")
    if len(summer_days) > 1:
        raise ValueError("runs disagree on summer start")

    common_day = min(run["completed_day"] for run in runs)
    curve_rows = []
    for run in runs:
        for row in run["curve"]:
            if row["day"] <= common_day:
                curve_rows.append({
                    "label": run["label"],
                    "seed": run["seed"],
                    "run_id": run["run_id"],
                    **row,
                })
    rate_fields = (
        "rolling_operational_fulfillment_rate",
        "cumulative_operational_fulfillment_rate",
        "cumulative_eligible_fulfillment_rate",
        "reservation_commitment_honor_rate",
    )
    grouped: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in curve_rows:
        grouped[(row["label"], int(row["day"]))].append(row)
    aggregates = []
    for (label, day), rows in sorted(grouped.items()):
        result: dict[str, Any] = {"label": label, "day": day, "seeds": len(rows)}
        for field in rate_fields:
            values = [float(row[field]) for row in rows if row[field] is not None]
            result[f"{field}_mean"] = statistics.mean(values) if values else None
            result[f"{field}_min"] = min(values) if values else None
            result[f"{field}_max"] = max(values) if values else None
        aggregates.append(result)
    endpoints = [
        row for row in curve_rows if int(row["day"]) == common_day
    ]
    return {
        "format": FORMAT,
        "status": (
            "complete" if all(run["status"] != "in_progress" for run in runs)
            else "in_progress"
        ),
        "common_completed_day": common_day,
        "decision_horizon_day": int(entries[0][1]["headline"]["decision_horizon_day"]),
        "runoff_end_day": int(entries[0][1]["headline"]["runoff_end_day"]),
        "summer_start_day": next(iter(summer_days), None),
        "environment_signature": next(iter(signatures)),
        "seed_sets": {label: sorted(seeds) for label, seeds in seeds_by_label.items()},
        "definitions": {
            "rolling_operational_fulfillment_rate": (
                "trailing-14-day rental fulfilled units divided by fulfilled + stockout + "
                "service-closed + capacity-policy units; ratio of sums"
            ),
            "reservation_commitment_honor_rate": (
                "successful pickups divided by successful pickups + manager cancellations + "
                "pickup failures; customer cancellations and pending commitments excluded"
            ),
            "causal_effect_claimed": False,
        },
        "runs": [{key: value for key, value in run.items() if key != "curve"}
                 for run in runs],
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
    from matplotlib.ticker import PercentFormatter

    labels = list(comparison["seed_sets"])
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    color_by_label = {label: colors[index % len(colors)] for index, label in enumerate(labels)}
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.0))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.15, top=0.89, hspace=0.34, wspace=0.28)
    rolling_ax, reservation_ax, demand_ax, outcome_ax = axes.flat

    for label in labels:
        color = color_by_label[label]
        rows = [row for row in comparison["aggregates"] if row["label"] == label]
        days = [row["day"] for row in rows]
        for ax, field in (
            (rolling_ax, "rolling_operational_fulfillment_rate"),
            (reservation_ax, "reservation_commitment_honor_rate"),
        ):
            means = [row[f"{field}_mean"] for row in rows]
            ax.plot(days, means, color=color, linewidth=2.3, label=label)
            if len(comparison["seed_sets"][label]) > 1:
                ax.fill_between(
                    days,
                    [row[f"{field}_min"] for row in rows],
                    [row[f"{field}_max"] for row in rows],
                    color=color, alpha=0.12, linewidth=0,
                )

    for ax in (rolling_ax, reservation_ax):
        ax.set_xlim(1, int(comparison["runoff_end_day"]))
        ax.set_ylim(0, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("Simulation day")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, loc="best", fontsize=8)
        summer = comparison.get("summer_start_day")
        if summer is not None:
            ax.axvline(summer, color="#7f8c8d", linestyle=":", linewidth=1)
    rolling_ax.set_title("14-day rolling rental operational fulfillment")
    rolling_ax.set_ylabel("Physical fulfillment rate")
    reservation_ax.set_title("Cumulative confirmed-commitment honor rate")
    reservation_ax.set_ylabel("Successful pickup share of manager-controlled resolutions")

    endpoints_by_label: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comparison["endpoints"]:
        endpoints_by_label[row["label"]].append(row)
    demand_parts = (
        ("rental_fulfilled_units", "Fulfilled", "#009E73"),
        ("rental_outside_rejections", "Outside choice", "#56B4E9"),
        ("rental_stockout_rejections", "Stockout", "#D55E00"),
        ("rental_capacity_policy_rejections", "Capacity policy", "#E69F00"),
        ("rental_service_closed_rejections", "Service closed", "#999999"),
    )
    reservation_parts = (
        ("reservation_successful_pickups", "Successful pickup", "#009E73"),
        ("reservation_customer_cancellations", "Customer cancelled", "#56B4E9"),
        ("reservation_manager_cancellations", "Manager cancelled", "#E69F00"),
        ("reservation_pickup_failures", "Pickup failed", "#D55E00"),
        ("reservation_pending", "Pending / not yet resolved", "#BBBBBB"),
    )

    def stacked_percent(ax: Any, parts: tuple[tuple[str, str, str], ...]) -> None:
        for y, label in enumerate(reversed(labels)):
            rows = endpoints_by_label[label]
            values = [statistics.mean(float(row[field]) for row in rows)
                      for field, _, _ in parts]
            total = sum(values)
            left = 0.0
            for value, (_, part_label, color) in zip(values, parts):
                width = value / total if total else 0.0
                ax.barh(
                    y, width, left=left, color=color, alpha=0.9,
                    label=part_label if y == 0 else None,
                )
                left += width
            ax.text(1.01, y, f"n={total:.0f}", va="center", fontsize=8)
        ax.set_yticks(range(len(labels)), list(reversed(labels)))
        ax.set_xlim(0, 1.09)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="x", alpha=0.2)
        ax.legend(frameon=False, fontsize=7.8, loc="best")

    stacked_percent(demand_ax, demand_parts)
    demand_ax.set_title(f"Cumulative rental demand outcomes through Day {comparison['common_completed_day']}")
    demand_ax.set_xlabel("Share of non-horizon rental service units")
    stacked_percent(outcome_ax, reservation_parts)
    outcome_ax.set_title(f"Confirmed reservation outcomes through Day {comparison['common_completed_day']}")
    outcome_ax.set_xlabel("Share of confirmed commitments")

    qualifier = (
        "IN PROGRESS — all curves and endpoint compositions are provisional."
        if comparison["status"] != "complete"
        else "Completed through runoff."
    )
    fig.suptitle("OutfitterBench fulfillment analysis", fontsize=16)
    fig.text(
        0.5, 0.025,
        "Operational fulfillment excludes outside choices and horizon rejections. "
        "Reservation honor excludes customer cancellations and pending commitments from its denominator.\n"
        "Manager cancellations remain visible because a confirmed promise was not served; proactive cancellation may still be economically rational. "
        + qualifier,
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
