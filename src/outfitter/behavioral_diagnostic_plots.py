"""Same-seed plots for derived diagnostics; no simulation or scoring logic."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path


def render_plot(job):
    kind, members, output_dir = job
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False})
    seeds = sorted({r["seed"] for r in members})
    models = {}
    for row in sorted(members, key=lambda r: (str(r["order"]), r["model_key"])):
        models.setdefault(row["model_key"], row)
    def color(row):
        return row["color"] or plt.get_cmap("tab20")(
            int(hashlib.sha256(row["model_key"].encode()).hexdigest()[:8], 16) % 20)
    def values(row, field):
        return row["profile"]["core"][field]["value"]
    def legend(figure, shown):
        handles = [Line2D([], [], color=color(row), marker=row["marker"], linestyle="None",
                          markersize=6, label=row["label"]) for key, row in models.items() if key in shown]
        if handles:
            figure.legend(handles=handles, loc="upper center", ncol=min(4, len(handles)), frameon=False)
    def finish(figure):
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        figure.savefig(directory/f"{kind}.pdf", bbox_inches="tight",
                       metadata={"CreationDate": None, "ModDate": None,
                                 "Creator": "outfitter-behavioral-diagnostics-v1"})
        figure.savefig(directory/f"{kind}.png", bbox_inches="tight", dpi=180,
                       metadata={"Software": "outfitter-behavioral-diagnostics-v1"})
        plt.close(figure)
        return kind

    if kind == "operating_profiles":
        fig, axes = plt.subplots(1, len(seeds), figsize=(3.7*len(seeds), 3.8),
                                 sharex=True, sharey=True, squeeze=False)
        points = [(r, values(r, "pending_booking_gap_rate"), values(r, "capital_productivity"))
                  for r in members]
        valid = [(r, x, y) for r, x, y in points if x is not None and y is not None]
        xmax = max(5, math.ceil(max((x for _, x, _ in valid), default=0)/5)*5)
        ymin = min(0, min((y for _, _, y in valid), default=0)*1.1)
        ymax = max(.2, math.ceil(max((y for _, _, y in valid), default=0)*10)/10+.1)
        shown = set()
        for i, seed in enumerate(seeds):
            ax = axes[0, i]
            for row, x, y in valid:
                if row["seed"] == seed:
                    ax.scatter(x, y, s=60, color=color(row), marker=row["marker"],
                               label=row["id"], edgecolor="white", linewidth=.5, zorder=3, clip_on=False)
                    shown.add(row["model_key"])
            ax.set(title=f"Seed {seed}", xlim=(0, xmax), ylim=(ymin, ymax),
                   xlabel="Pending-booking gap rate (%)")
            if i == 0:
                ax.set_ylabel("Capital productivity\n(CNY / 100 capital-CNY-days)")
            ax.grid(color="#e6e6e6", linewidth=.6)
            ax.set_axisbelow(True)
        legend(fig, shown)
        fig.tight_layout(rect=(0, .02, 1, .84))
        return finish(fig)

    if kind == "pickup_lead_gaps":
        fig, axes = plt.subplots(1, len(seeds), figsize=(3.7*len(seeds), 3.9),
                                 sharey=True, squeeze=False)
        shown = set()
        for i, seed in enumerate(seeds):
            ax = axes[0, i]
            for row in members:
                if row["seed"] != seed:
                    continue
                vals = [row["profile"]["gap_by_lead"][g]["value"] for g in ("8+", "3-7", "1-2")]
                if all(v is None for v in vals):
                    continue
                ax.plot(range(3), [np.nan if v is None else v for v in vals],
                        color=color(row), marker=row["marker"], linewidth=1.2, label=row["id"], clip_on=False)
                shown.add(row["model_key"])
            ax.set(title=f"Seed {seed}", xticks=range(3), xticklabels=["8+ days", "3-7 days", "1-2 days"],
                   xlabel="Time to pickup (different observation sets)", ylim=(0, 100))
            if i == 0:
                ax.set_ylabel("Projected gap rate (%)")
            ax.grid(axis="y", color="#e6e6e6", linewidth=.6)
        legend(fig, shown)
        fig.tight_layout(rect=(0, .02, 1, .84))
        return finish(fig)

    if kind == "capacity_opportunities":
        fig, axes = plt.subplots(2, len(seeds), figsize=(3.8*len(seeds), 5.9),
                                 sharex=True, sharey="row", squeeze=False)
        fields = ("capacity_positive_day_rate_g3", "capacity_mean_improvement_g3")
        for row_index, field in enumerate(fields):
            all_values = [values(row, field) for row in members if values(row, field) is not None]
            for col, seed in enumerate(seeds):
                ax = axes[row_index, col]
                for index, (key, model) in enumerate(models.items()):
                    repeats = [r for r in members if r["seed"] == seed and r["model_key"] == key]
                    plotted = False
                    for j, row in enumerate(repeats):
                        v = values(row, field)
                        if v is not None:
                            ax.scatter(index+(j-(len(repeats)-1)/2)*.08, v, s=45,
                                       color=color(row), marker=row["marker"], label=row["id"], zorder=3, clip_on=False)
                            plotted = True
                    if not plotted:
                        ax.text(index, .05, "N/A", transform=ax.get_xaxis_transform(),
                                ha="center", fontsize=6, color="#777777")
                ax.set_xticks(range(len(models)), [r.get("axis_label", r["label"]).replace(" ", "\n")
                                                   for r in models.values()])
                ax.set_xlim(-.5, len(models)-.5)
                ax.tick_params(axis="x", labelsize=7)
                ax.set_ylim(0, max(5 if row_index == 0 else .1, max(all_values, default=0)*1.16))
                ax.grid(axis="y", color="#e6e6e6", linewidth=.6)
                if row_index == 0:
                    ax.set_title(f"Seed {seed}")
                if col == 0:
                    ax.set_ylabel("Positive sample days (%)" if row_index == 0 else "Estimated improvement\n(CNY / sampled day)")
        fig.suptitle("Local capacity-relaxation opportunities (3-day turnaround check)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .95), h_pad=.8)
        return finish(fig)
    raise ValueError(f"unknown plot: {kind}")
