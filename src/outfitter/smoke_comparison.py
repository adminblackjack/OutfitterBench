"""Compare real-agent smoke curves with same-window deterministic baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .csvio import write_csv as _write_csv


FORMAL_BASELINE_STRATEGIES = frozenset(
    {"naive", "heuristic", "informed_public"}
)


def _agent_curve(report: dict[str, Any]) -> list[dict[str, Any]]:
    initial = float(report["headline"]["initial_official_liquidation_value"])
    initial_contract = float(
        report["headline"]["initial_contract_adjusted_economic_value"]
    )
    cumulative_requested = 0
    cumulative_fulfilled = 0
    rows = []
    for point in report["daily"]:
        cumulative_requested += int(point["requested_units"])
        cumulative_fulfilled += int(point["fulfilled_units"])
        official = float(point["official_liquidation_value_end"])
        rows.append(
            {
                "day": int(point["day"]),
                "official_liquidation_value": official,
                "official_value_change": round(official - initial, 2),
                "contract_adjusted_economic_value": float(
                    point["contract_adjusted_economic_value_end"]
                ),
                "contract_adjusted_value_change": round(
                    float(point["contract_adjusted_economic_value_end"])
                    - initial_contract,
                    2,
                ),
                "cumulative_revenue": float(point["cumulative_revenue"]),
                "cumulative_operating_pnl": float(
                    point["cumulative_operating_pnl"]
                ),
                "cash": float(point["cash_end"]),
                "cumulative_requested_units": cumulative_requested,
                "cumulative_fulfilled_units": cumulative_fulfilled,
            }
        )
    return rows


def _baseline_curve(row: dict[str, Any]) -> list[dict[str, Any]]:
    cumulative_requested = 0
    cumulative_fulfilled = 0
    result = []
    for point in row["daily_performance_curve"]:
        cumulative_requested += int(point["requested_units"])
        cumulative_fulfilled += int(point["fulfilled_units"])
        result.append(
            {
                "day": int(point["day"]),
                "official_liquidation_value": float(
                    point["official_liquidation_value"]
                ),
                "official_value_change": float(point["official_value_change"]),
                "contract_adjusted_economic_value": float(
                    point["contract_adjusted_economic_value"]
                ),
                "contract_adjusted_value_change": round(
                    float(
                        point.get(
                            "contract_adjusted_value_change",
                            float(point["contract_adjusted_economic_value"])
                            - float(row["initial_contract_adjusted_value"]),
                        )
                    ), 2
                ),
                "cumulative_revenue": float(point["cumulative_revenue"]),
                "cumulative_operating_pnl": None,
                "cash": float(point["cash"]),
                "cumulative_requested_units": cumulative_requested,
                "cumulative_fulfilled_units": cumulative_fulfilled,
            }
        )
    return result


def _agent_summary(report: dict[str, Any]) -> dict[str, Any]:
    headline = report["headline"]
    financials = report["financials"]
    demand = report["demand"]
    usage = report["agent_usage"]
    return {
        "source": "agent",
        "run_id": report["run"]["run_id"],
        "model": report["run"]["model"],
        "reasoning_effort": report["run"].get("reasoning_effort"),
        "value_change": float(headline["official_liquidation_value_change"]),
        "contract_value_change": float(
            headline["contract_adjusted_economic_value_at_stop"]
            - headline["initial_contract_adjusted_economic_value"]
        ),
        "final_official_value": float(headline["official_liquidation_value_at_stop"]),
        "total_revenue": float(financials["total_revenue"]),
        "purchase_cost": float(financials["purchase_cost"]),
        "minimum_cash": min(
            (float(row["cash_end"]) for row in report["daily"]),
            default=float(financials.get("final_cash", 0.0)),
        ),
        "postseason_cleaning_work_points": float(
            financials.get("postseason_cleaning_work_points", 0.0)
        ),
        "postseason_cleaning_cost": float(
            financials.get("postseason_cleaning_cost", 0.0)
        ),
        "purchased_rental_assets_by_product": json.dumps(
            {
                str(row["product_type"]): int(row["purchased_assets"])
                for row in report.get(
                    "purchased_rental_contribution_by_product", []
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "fulfilled_units": int(demand["fulfilled_units"]),
        "requested_units": int(demand["requested_units"]),
        "eligible_fulfillment_rate": float(demand["eligible_fulfillment_rate"]),
        "model_calls": int(usage["model_calls"]),
        "model_cost": float(financials["model_cost"]),
        "tool_errors": int(usage["tool_errors"]),
        "bankrupt": headline["status"] == "bankrupt",
    }


def _baseline_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "baseline",
        "run_id": str(row["strategy"]),
        "model": None,
        "reasoning_effort": None,
        "value_change": float(row["value_change"]),
        "contract_value_change": float(row["contract_value_change"]),
        "final_official_value": float(row["final_official_liquidation_value"]),
        "total_revenue": float(row["rental_revenue"])
        + float(row["retail_revenue"]),
        "purchase_cost": float(row.get("purchase_cost", 0.0)),
        "minimum_cash": float(
            row.get(
                "minimum_cash",
                min(
                    float(point["cash"])
                    for point in row["daily_performance_curve"]
                ),
            )
        ),
        "postseason_cleaning_work_points": float(
            row.get("postseason_cleaning_work_points", 0.0)
        ),
        "postseason_cleaning_cost": float(
            row.get("postseason_cleaning_cost", 0.0)
        ),
        "purchased_rental_assets_by_product": json.dumps(
            row.get("purchased_rental_assets_by_product", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "fulfilled_units": int(row["fulfilled_units"]),
        "requested_units": int(row["requested_units"]),
        "eligible_fulfillment_rate": int(row["fulfilled_units"])
        / max(1, int(row["requested_units"])),
        "model_calls": 0,
        "model_cost": 0.0,
        "tool_errors": 0,
        "bankrupt": bool(row["bankrupt"]),
    }


def build_smoke_comparison(
    scenarios: list[tuple[str, Path, Path]],
) -> dict[str, Any]:
    summaries = []
    curve_rows = []
    scenario_comparisons = []
    for label, run_dir, baseline_path in scenarios:
        report_path = run_dir / "analysis" / "report.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"run must be analyzed first: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        agent = _agent_summary(report)
        agent["scenario"] = label
        summaries.append(agent)
        for point in _agent_curve(report):
            curve_rows.append(
                {"scenario": label, "series": "agent", **point}
            )
        baseline_summaries = []
        for row in baseline["runs"]:
            summary = _baseline_summary(row)
            summary["scenario"] = label
            summaries.append(summary)
            baseline_summaries.append(summary)
            for point in _baseline_curve(row):
                curve_rows.append(
                    {
                        "scenario": label,
                        "series": str(row["strategy"]),
                        **point,
                    }
                )
        formal_baselines = [
            row
            for row in baseline_summaries
            if row["run_id"] in FORMAL_BASELINE_STRATEGIES
        ]
        if not formal_baselines:
            raise ValueError("baseline comparison lacks a formal public strategy")
        best = max(formal_baselines, key=lambda row: row["value_change"])
        scenario_comparisons.append(
            {
                "scenario": label,
                "agent_value_change": agent["value_change"],
                "best_baseline": best["run_id"],
                "best_baseline_value_change": best["value_change"],
                "agent_minus_best_baseline": round(
                    agent["value_change"] - best["value_change"], 2
                ),
                "agent_total_revenue": agent["total_revenue"],
                "agent_eligible_fulfillment_rate": agent[
                    "eligible_fulfillment_rate"
                ],
            }
        )
    return {
        "format": "outfitter-agent-smoke-comparison-v1",
        "scenarios": scenario_comparisons,
        "summaries": summaries,
        "curves": curve_rows,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# OutfitterBench Agent smoke comparison",
        "",
        "All comparisons use the same scenario window, seed, frozen runtime, and "
        "initial cash. Value changes are absolute official liquidation-value changes.",
        "",
        "| Scenario | Agent Δ value | Best baseline | Baseline Δ value | "
        "Agent advantage | Agent revenue | Eligible fulfillment |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["scenarios"]:
        lines.append(
            f"| {row['scenario']} | {row['agent_value_change']:.2f} | "
            f"{row['best_baseline']} | {row['best_baseline_value_change']:.2f} | "
            f"{row['agent_minus_best_baseline']:.2f} | "
            f"{row['agent_total_revenue']:.2f} | "
            f"{row['agent_eligible_fulfillment_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "`curves.csv` contains daily official value change, cumulative revenue, "
            "contract-adjusted value change, cash, and fulfillment for every "
            "Agent/baseline series. Contract-adjusted value is the preferred "
            "within-season management curve because immediate liquidation assigns "
            "zero to rented assets and in-transit orders; official liquidation value "
            "remains the terminal score.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(path: Path, report: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    scenarios = [row["scenario"] for row in report["scenarios"]]
    figure, axes = plt.subplots(
        len(scenarios),
        3,
        figsize=(18, max(4, 3.6 * len(scenarios))),
        squeeze=False,
    )
    for index, scenario in enumerate(scenarios):
        rows = [row for row in report["curves"] if row["scenario"] == scenario]
        series_names = list(dict.fromkeys(row["series"] for row in rows))
        for series in series_names:
            points = [row for row in rows if row["series"] == series]
            width = 2.6 if series == "agent" else 1.2
            axes[index][0].plot(
                [row["day"] for row in points],
                [row["official_value_change"] for row in points],
                label=series,
                linewidth=width,
            )
            axes[index][1].plot(
                [row["day"] for row in points],
                [row["contract_adjusted_value_change"] for row in points],
                label=series,
                linewidth=width,
            )
            axes[index][2].plot(
                [row["day"] for row in points],
                [row["cumulative_revenue"] for row in points],
                label=series,
                linewidth=width,
            )
        axes[index][0].axhline(0, color="black", linewidth=0.7)
        axes[index][1].axhline(0, color="black", linewidth=0.7)
        axes[index][0].set_title(f"{scenario}: official value change")
        axes[index][1].set_title(f"{scenario}: contract-adjusted change")
        axes[index][2].set_title(f"{scenario}: cumulative revenue")
        for axis in axes[index]:
            axis.set_xlabel("Benchmark day")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Agent smoke curves")
    parser.add_argument(
        "--scenario",
        action="append",
        nargs=3,
        metavar=("LABEL", "RUN_DIR", "BASELINE_JSON"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scenarios = [
        (label, Path(run_dir).expanduser().resolve(), Path(baseline).expanduser().resolve())
        for label, run_dir, baseline in args.scenario
    ]
    report = build_smoke_comparison(scenarios)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "summaries.csv", report["summaries"])
    _write_csv(output / "curves.csv", report["curves"])
    _write_markdown(output / "REPORT.md", report)
    _plot(output / "agent_vs_baselines.png", report)
    print(json.dumps(report["scenarios"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
