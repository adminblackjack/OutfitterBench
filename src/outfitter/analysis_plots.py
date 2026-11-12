"""Plot rendering for materialized OutfitterBench analysis reports."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any


PLOT_FILES = (
    "01_financial_overview.png",
    "02_service_demand.png",
    "03_costs_and_utilization.png",
    "04_terminal_salvage_inventory.png",
    "05_agent_activity.png",
    "06_daily_business_value.png",
    "07_product_contribution.png",
    "08_reservation_governance.png",
)


def _moving_average(values: list[float], window: int = 7) -> list[float]:
    return [
        sum(values[max(0, index - window + 1) : index + 1])
        / len(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]


def _plot_financial(report: dict[str, Any], path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    daily = report["daily"]
    days = [row["day"] for row in daily]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    operating_label = (
        "Contract-adjusted economic value"
        if all(
            row["operating_value_basis"] == "going_concern_business_value"
            for row in daily
        )
        else "Reconstructed book net assets"
    )
    axes[0, 0].plot(
        days,
        [row["operating_business_value_end"] for row in daily],
        label=operating_label,
        color="#087f5b",
        linewidth=2,
    )
    axes[0, 0].plot(
        days,
        [row["liquidation_business_value_end"] for row in daily],
        label="Execution liquidation value",
        color="#94a3b8",
        linewidth=1,
        alpha=0.75,
    )
    axes[0, 0].plot(days, [row["cash_end"] for row in daily], label="Cash", color="#1f77b4")
    axes[0, 0].plot(days, [row["deposit_liability_end"] for row in daily], label="Deposit liability", color="#d62728")
    axes[0, 0].set(title="Daily financial position", xlabel="Day", ylabel="CNY")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.25)
    rental = [row["rental_revenue"] for row in daily]
    retail = [row["retail_revenue"] for row in daily]
    axes[0, 1].bar(days, rental, label="Rental", color="#4c78a8")
    axes[0, 1].bar(days, retail, bottom=rental, label="Retail", color="#f58518")
    axes[0, 1].plot(
        days,
        [row.get("asset_sale_proceeds", 0.0) for row in daily],
        color="#9467bd",
        linestyle=":",
        label="Asset sales",
    )
    axes[0, 1].plot(days, _moving_average([row["revenue_total"] for row in daily]), color="black", label="7-day avg")
    axes[0, 1].set(title="Daily revenue", xlabel="Day", ylabel="CNY"); axes[0, 1].legend(); axes[0, 1].grid(axis="y", alpha=0.25)
    axes[1, 0].plot(days, [row["cumulative_operating_pnl"] for row in daily], label="Cumulative operating P&L", color="#087f5b")
    axes[1, 0].bar(days, [row["daily_operating_pnl"] for row in daily], label="Daily operating P&L", color="#75b798", alpha=0.55)
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set(title="Operating P&L (excludes purchases and model fees)", xlabel="Day", ylabel="CNY"); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.25)
    worth = report["final_business_value_components"]
    labels = ["Cash", "Retail salvage", "Rental resale", "Deposit liability", "Repair liability"]
    values = [
        worth["cash"],
        worth["retail_inventory_value"],
        worth.get("rental_secondhand_value", worth.get("rental_remaining_life_value", 0.0)),
        -worth["deposit_liability"],
        -worth.get("damaged_repair_liability", 0.0),
    ]
    axes[1, 1].bar(labels, values, color=["#4c78a8" if value >= 0 else "#e45756" for value in values])
    axes[1, 1].axhline(0, color="black", linewidth=0.8); axes[1, 1].tick_params(axis="x", rotation=22)
    axes[1, 1].set(title="Final business value composition", ylabel="CNY"); axes[1, 1].grid(axis="y", alpha=0.25)
    fig.suptitle("Financial overview", fontsize=15); fig.savefig(path, dpi=dpi); plt.close(fig)

def _plot_business_value(report: dict[str, Any], path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    daily = report["daily"]
    headline = report["headline"]
    days = [0, *[row["day"] for row in daily]]
    operating = [
        headline["initial_operating_business_value"],
        *[row["operating_business_value_end"] for row in daily],
    ]
    liquidation = [
        headline["initial_business_value"],
        *[row["liquidation_business_value_end"] for row in daily],
    ]
    book = [
        headline["initial_book_net_assets"],
        *[row["book_net_assets_end"] for row in daily],
    ]
    exact_going_concern = all(
        row["operating_value_basis"] == "going_concern_business_value"
        for row in daily
    )
    operating_label = (
        "Contract-adjusted economic value"
        if exact_going_concern
        else "Reconstructed book net assets"
    )
    fig, ax = plt.subplots(figsize=(13.5, 6.8), constrained_layout=True)
    ax.plot(
        days,
        operating,
        color="#75b798",
        linewidth=1.25,
        alpha=0.8,
        label=operating_label,
    )
    ax.plot(
        days,
        _moving_average(operating),
        color="#087f5b",
        linewidth=2.4,
        label="7-day operating-value average",
    )
    ax.plot(
        days,
        liquidation,
        color="#94a3b8",
        linewidth=0.9,
        alpha=0.55,
        label="Execution liquidation value",
    )
    if exact_going_concern:
        ax.plot(
            days,
            book,
            color="#4c78a8",
            linewidth=1,
            linestyle=":",
            alpha=0.8,
            label="Book net assets",
        )
    initial = headline["initial_operating_business_value"]
    ax.axhline(
        initial,
        color="#6c757d",
        linestyle="--",
        linewidth=1.2,
        label=f"Initial contract-adjusted value: {initial:,.2f}",
    )
    ax.scatter(
        [days[-1]],
        [operating[-1]],
        color="#d9480f",
        s=50,
        zorder=5,
        label=f"Final contract-adjusted value: {operating[-1]:,.2f}",
    )
    ax.set_title("Daily business-value diagnostics", fontsize=17, pad=14)
    ax.set_xlabel("Day")
    ax.set_ylabel("CNY")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=True, ncol=2)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _plot_demand(report: dict[str, Any], path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    daily = report["daily"]
    days = [row["day"] for row in daily]
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), constrained_layout=True)
    fields = [
        ("fulfilled_units", "Fulfilled", "#54a24b"),
        ("outside_rejections", "Outside", "#9c755f"),
        ("stockout_rejections", "Stockout", "#f58518"),
        ("service_closed_rejections", "Service closed", "#e45756"),
        ("horizon_rejections", "Horizon", "#b279a2"),
    ]
    bottom = [0] * len(days)
    for field, label, color in fields:
        values = [row[field] for row in daily]
        axes[0].bar(days, values, bottom=bottom, label=label, color=color)
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[0].set(title="Requested units and outcomes", ylabel="Units"); axes[0].legend(ncol=4); axes[0].grid(axis="y", alpha=0.25)
    basket_fields = [("full_baskets", "Full", "#54a24b"), ("partial_baskets", "Partial", "#f2cf5b"), ("unfilled_baskets", "Unfilled", "#e45756")]
    bottom = [0] * len(days)
    for field, label, color in basket_fields:
        values = [row[field] for row in daily]
        axes[1].bar(days, values, bottom=bottom, label=label, color=color)
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[1].set(title="Basket fulfillment", ylabel="Customers"); axes[1].legend(ncol=3); axes[1].grid(axis="y", alpha=0.25)
    axes[1].plot(
        days,
        [row.get("operational_full_baskets", 0) for row in daily],
        color="#1b7837",
        linestyle="--",
        label="Operational full",
    )
    axes[1].plot(
        days,
        [row.get("operational_partial_baskets", 0) for row in daily],
        color="#b8860b",
        linestyle="--",
        label="Operational partial",
    )
    axes[1].plot(
        days,
        [row.get("operational_unfilled_baskets", 0) for row in daily],
        color="#a50f15",
        linestyle="--",
        label="Operational unfilled",
    )
    axes[1].legend(ncol=3, fontsize=8)
    services = report["service_summary"]
    top = sorted(services, key=lambda row: row["requested_units"], reverse=True)[:16]
    names = [row["service_key"] for row in reversed(top)]
    requested = [row["requested_units"] for row in reversed(top)]
    fulfilled = [row["fulfilled_units"] for row in reversed(top)]
    axes[2].barh(names, requested, color="#d9d9d9", label="Requested")
    axes[2].barh(names, fulfilled, color="#4c78a8", label="Fulfilled")
    axes[2].set(title="Service demand over the run", xlabel="Units"); axes[2].legend(); axes[2].grid(axis="x", alpha=0.25)
    fig.suptitle("Sales and lost demand", fontsize=15); fig.savefig(path, dpi=dpi); plt.close(fig)

def _plot_costs(report: dict[str, Any], path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    daily = report["daily"]
    days = [row["day"] for row in daily]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    cost_specs = [("operating_fee", "Operating", "#4c78a8"), ("purchase_cost", "Purchases", "#9467bd"), ("cleaning_cost", "Cleaning", "#72b7b2"), ("repair_cost", "Repair", "#e45756"), ("retail_holding_cost", "Retail holding", "#f58518"), ("rental_holding_cost", "Rental holding", "#54a24b"), ("model_cost", "Model", "#b279a2")]
    bottom = [0.0] * len(days)
    for field, label, color in cost_specs:
        values = [row[field] for row in daily]
        axes[0, 0].bar(days, values, bottom=bottom, label=label, color=color)
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[0, 0].set(title="Daily costs", xlabel="Day", ylabel="CNY"); axes[0, 0].legend(fontsize=8, ncol=2); axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 1].plot(days, [100 * row["rental_utilization_end"] for row in daily], color="#2ca02c", label="Utilization")
    axes[0, 1].set(title="Rental utilization", xlabel="Day", ylabel="Percent"); axes[0, 1].set_ylim(bottom=0); axes[0, 1].grid(alpha=0.25)
    axes[1, 0].bar(days, [row["rental_starts"] for row in daily], color="#54a24b", label="Starts")
    axes[1, 0].bar(days, [-row["rental_returns"] for row in daily], color="#e45756", label="Returns")
    axes[1, 0].plot(days, [row["active_rentals_end"] for row in daily], color="black", label="Active")
    axes[1, 0].axhline(0, color="black", linewidth=0.7); axes[1, 0].set(title="Rental flow", xlabel="Day", ylabel="Assets"); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.25)
    axes[1, 1].bar(days, [row["deposits_collected"] for row in daily], color="#4c78a8", label="Collected")
    axes[1, 1].bar(days, [-row["refunds"] for row in daily], color="#e45756", label="Refunded")
    axes[1, 1].plot(days, [row["deposit_liability_end"] for row in daily], color="black", label="Liability")
    axes[1, 1].axhline(0, color="black", linewidth=0.7); axes[1, 1].set(title="Deposit flows", xlabel="Day", ylabel="CNY"); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.25)
    fig.suptitle("Costs and rental operations", fontsize=15); fig.savefig(path, dpi=dpi); plt.close(fig)

def _plot_salvage(report: dict[str, Any], path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    daily = report["daily"]
    days = [row["day"] for row in daily]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes[0, 0].plot(days, [row["retail_book_value_end"] for row in daily], label="Retail book", color="#f58518")
    axes[0, 0].plot(days, [row["retail_salvage_value_end"] for row in daily], label="Retail salvage", color="#f2cf5b")
    axes[0, 0].plot(days, [row["rental_book_value_end"] for row in daily], label="Rental remaining-life book", color="#4c78a8")
    axes[0, 0].plot(days, [row["rental_salvage_value_end"] for row in daily], label="Rental salvage", color="#72b7b2")
    axes[0, 0].set(title="Inventory valuation through time", xlabel="Day", ylabel="CNY"); axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=0.25)
    worth = report["final_business_value_components"]
    axes[0, 1].bar(["Retail", "Rental"], [worth["retail_inventory_book_value"], worth["rental_remaining_life_book_value"]], color="#d9d9d9", label="Book")
    axes[0, 1].bar(
        ["Retail", "Rental"],
        [
            worth["retail_inventory_value"],
            worth.get("rental_secondhand_value", worth.get("rental_remaining_life_value", 0.0)),
        ],
        color=["#f58518", "#4c78a8"],
        label="Terminal value",
    )
    axes[0, 1].set(title="Terminal salvage haircut", ylabel="CNY"); axes[0, 1].legend(); axes[0, 1].grid(axis="y", alpha=0.25)
    retail = report["final_retail_inventory"]
    axes[1, 0].barh([row["product_type"] for row in retail], [row["units"] for row in retail], color="#f58518")
    axes[1, 0].set(title="Final retail units", xlabel="Units"); axes[1, 0].grid(axis="x", alpha=0.25)
    assets = report["final_rental_assets"]
    lines = sorted({row["product_type"] for row in assets})
    statuses = sorted({row["status"] for row in assets})
    left = [0] * len(lines)
    colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2", "#72b7b2")
    for status, color in zip(statuses, colors):
        values = [sum(row["assets"] for row in assets if row["product_type"] == line and row["status"] == status) for line in lines]
        axes[1, 1].barh(lines, values, left=left, label=status, color=color)
        left = [a + b for a, b in zip(left, values)]
    axes[1, 1].set(title="Final rental fleet", xlabel="Assets"); axes[1, 1].legend(fontsize=8); axes[1, 1].grid(axis="x", alpha=0.25)
    fig.suptitle("Inventory and terminal salvage", fontsize=15); fig.savefig(path, dpi=dpi); plt.close(fig)

def _plot_agent(report: dict[str, Any], path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    daily = report["daily"]
    days = [row["day"] for row in daily]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    axes[0, 0].bar(days, [row["model_calls"] for row in daily], color="#4c78a8"); axes[0, 0].set(title="Model calls", xlabel="Day")
    axes[0, 1].plot(days, [row["input_tokens"] for row in daily], label="Input"); axes[0, 1].plot(days, [row["output_tokens"] for row in daily], label="Output"); axes[0, 1].set(title="Tokens", xlabel="Day"); axes[0, 1].legend()
    axes[1, 0].bar(days, [row["tool_calls"] for row in daily], color="#72b7b2", label="Executed")
    axes[1, 0].plot(days, [row["tool_errors"] for row in daily], color="#e45756", label="Errors")
    axes[1, 0].plot(days, [row.get("discarded_tool_calls", 0) for row in daily], color="#9467bd", label="Discarded")
    axes[1, 0].plot(days, [row.get("rejected_batched_tool_calls", 0) for row in daily], color="#f58518", label="Batch-rejected")
    axes[1, 0].set(title="Tool calls, errors, discarded, and batch-rejected", xlabel="Day"); axes[1, 0].legend()
    tools = list(reversed(report["tool_usage"][:15])); axes[1, 1].barh([row["tool"] for row in tools], [row["calls"] for row in tools], color="#b279a2"); axes[1, 1].set(title="Most-used tools")
    for ax in axes.flat: ax.grid(alpha=0.25)
    fig.suptitle("Agent activity", fontsize=15); fig.savefig(path, dpi=dpi); plt.close(fig)

def _plot_product_contribution(report: dict[str, Any], path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    offer_rows = report["offer_concentration"][:12]
    weather_rows = report["product_weather_contribution"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    labels = [row["offer_id"] for row in reversed(offer_rows)]
    values = [row["gross_contribution"] for row in reversed(offer_rows)]
    axes[0].barh(
        labels,
        values,
        color=["#4c78a8" if value >= 0 else "#e45756" for value in values],
    )
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set(
        title="Top SKU gross contribution",
        xlabel="CNY (revenue − retail COGS)",
    )
    axes[0].grid(axis="x", alpha=0.25)

    by_weather: dict[str, float] = defaultdict(float)
    for row in weather_rows:
        by_weather[str(row["weather"])] += float(row["gross_contribution"])
    weather = sorted(by_weather, key=lambda value: (-by_weather[value], value))
    axes[1].bar(
        weather,
        [by_weather[value] for value in weather],
        color="#72b7b2",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].tick_params(axis="x", rotation=28)
    axes[1].set(
        title="Gross contribution by realized weather",
        ylabel="CNY (revenue − retail COGS)",
    )
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Product concentration and weather contribution", fontsize=15)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_reservation_governance(
    report: dict[str, Any], path: Path, dpi: int
) -> None:
    import matplotlib.pyplot as plt

    contingency = report.get("reservations", {}).get(
        "proactive_contingency", {}
    )
    daily = list(contingency.get("daily", []))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    if not daily:
        for axis in axes.flat:
            axis.axis("off")
        axes[0, 0].text(
            0.5, 0.5, "No completed reservation-governance days",
            ha="center", va="center", fontsize=14,
        )
    else:
        days = [int(row["day"]) for row in daily]

        def cumulative(field: str) -> list[int]:
            total = 0
            values = []
            for row in daily:
                total += int(row.get(field, 0))
                values.append(total)
            return values

        for field, label, color in (
            ("inquiries", "Inquiries", "#4c78a8"),
            ("reports", "Reports", "#72b7b2"),
            ("applications", "Applied", "#54a24b"),
            (
                "merchant_induced_customer_cancellations",
                "Contact-induced cancellations",
                "#e45756",
            ),
        ):
            axes[0, 0].plot(
                days, cumulative(field), label=label, color=color, linewidth=2
            )
        axes[0, 0].set(
            title="Cumulative proactive-contingency lifecycle",
            xlabel="Completed day",
            ylabel="Count",
        )
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(alpha=0.25)

        shifts = [int(row.get("signed_pickup_shift_days", 0)) for row in daily]
        axes[0, 1].bar(
            days,
            shifts,
            color=["#54a24b" if value >= 0 else "#e45756" for value in shifts],
        )
        axes[0, 1].axhline(0, color="black", linewidth=0.8)
        axes[0, 1].set(
            title="Applied pickup-date shifts",
            xlabel="Completed day",
            ylabel="Signed days (later positive)",
        )
        axes[0, 1].grid(axis="y", alpha=0.25)

        outcomes = dict(contingency.get("outcome_counts", {}))
        outcome_names = list(outcomes) or ["none"]
        outcome_values = [int(outcomes[name]) for name in outcomes] or [0]
        axes[1, 0].barh(outcome_names, outcome_values, color="#4c78a8")
        axes[1, 0].set(
            title="Customer response outcomes",
            xlabel="Reports",
        )
        axes[1, 0].grid(axis="x", alpha=0.25)

        applications = dict(contingency.get("application_type_counts", {}))
        application_names = list(applications) or ["none"]
        application_values = [int(applications[name]) for name in applications] or [0]
        axes[1, 1].barh(
            application_names, application_values, color="#54a24b"
        )
        axes[1, 1].set(
            title="Applied change types (diagnostic only)",
            xlabel="Applications",
        )
        axes[1, 1].grid(axis="x", alpha=0.25)
    fig.suptitle(
        "Reservation governance — actions are not independent reward terms",
        fontsize=15,
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

def render_plots(report: dict[str, Any], output: Path, dpi: int) -> None:
    try:
        import matplotlib
    except ImportError as error:
        raise RuntimeError("plotting requires matplotlib; use --no-plots if unavailable") from error
    matplotlib.use("Agg")
    _plot_financial(report, output / PLOT_FILES[0], dpi)
    _plot_demand(report, output / PLOT_FILES[1], dpi)
    _plot_costs(report, output / PLOT_FILES[2], dpi)
    _plot_salvage(report, output / PLOT_FILES[3], dpi)
    _plot_agent(report, output / PLOT_FILES[4], dpi)
    _plot_business_value(report, output / PLOT_FILES[5], dpi)
    _plot_product_contribution(report, output / PLOT_FILES[6], dpi)
    _plot_reservation_governance(report, output / PLOT_FILES[7], dpi)
