"""Bounded, resumeless diagnostic runs; no model client is constructed."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback

from jsonschema import Draft202012Validator

from .config import OutfitterConfig
from .runner import OutfitterRunner, _implementation_identity
from .run_scripted_merchant import DailyJournal, NoModel
from .simple_policy_stress import (
    POLICIES, PROTOCOL, READ_TOOLS, WRITE_TOOLS, SimpleMerchant, parameters,
)

EXPECTED_CORE = "e1797c8d40b8f25d4ec8f4261217df655a20268e903c5260794e05b75fb5456f"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_hashes():
    package = Path(__file__).parent
    return {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in (
        "simple_policy_stress.py", "simple_policy_stress_run.py", "scripted_merchant.py", "run_scripted_merchant.py")}


class RoundLimit(RuntimeError):
    pass


class PublicRPC:
    """Count only real batches whose inputs are fixed before their outputs arrive."""
    def __init__(self, journal, state, tools, limit=70, batch_size=5):
        self.journal, self.state = journal, state
        self.limit, self.batch_size = limit, batch_size
        self.rounds = 0
        self.counts = Counter()
        self.errors = []
        self.serial = 0
        self.validators = {
            row["function"]["name"]: Draft202012Validator(json.loads(
                json.dumps(row["function"]["parameters"]), parse_float=Decimal))
            for row in tools.definitions()
        }

    def _begin(self, closing=False):
        # Reserve a round for explicit close, like a manager ending their day.
        if self.rounds >= self.limit - (0 if closing else 1):
            raise RoundLimit("daily public interaction round limit reached")
        self.rounds += 1

    def _execute(self, name, args):
        if name not in READ_TOOLS | WRITE_TOOLS:
            raise ValueError(f"tool not allowlisted: {name}")
        self.validators[name].validate(json.loads(json.dumps(args), parse_float=Decimal))
        self.serial += 1
        result = self.journal.execute(self.state, f"simple-d{self.state.day:03d}-{self.serial:05d}", name, args)
        self.counts[name] += 1
        if not result.ok:
            self.errors.append({"day": self.state.day, "tool": name, "args": args, "result": result.data})
        return json.loads(json.dumps({"ok": result.ok, "data": result.data, "terminal_day": result.terminal_day}))

    def __call__(self, name, args):
        self._begin(closing=name == "close_business_day")
        return self._execute(name, args)

    def batch(self, queries):
        if any(name not in READ_TOOLS for name, _ in queries):
            raise ValueError("only read tools may be batched")
        answers = []
        for offset in range(0, len(queries), self.batch_size):
            self._begin()
            answers.extend(self._execute(name, args) for name, args in queries[offset:offset+self.batch_size])
        return answers


def reference_config(reference, runs_dir, policy):
    reference = Path(reference)
    manifest = json.loads((reference / "manifest.json").read_text())
    if manifest["controller"]["implementation_identity"]["source_sha256"] != EXPECTED_CORE:
        raise ValueError("reference run is not the audited V44 implementation")
    if _implementation_identity()["source_sha256"] != EXPECTED_CORE:
        raise ValueError("current environment core differs from the original LLM world")
    values = dict(manifest["config"])
    values.update(data_dir=str((reference / manifest.get("runtime_data", "data")).resolve()),
                  runs_dir=str(Path(runs_dir).resolve()), model=f"scripted-{policy}",
                  base_url=None, reasoning_effort=None, provider_extra_body={})
    return OutfitterConfig.from_dict(values)


def run_one(reference, runs_dir, policy, *, stop_day=None):
    config = reference_config(reference, runs_dir, policy)
    run_id = f"{policy}-seed{config.seed}"
    runner = OutfitterRunner.create(config=config, provider=NoModel(), run_id=run_id)
    state = runner.store.load_state()
    journal = DailyJournal(runner.store, runner.tools)
    provenance = {**parameters(), "policy": policy, "source_sha256": source_hashes(),
        "core_identity": _implementation_identity(), "reference": str(Path(reference).resolve()),
        "no_llm": True, "world_horizon": [config.max_days, config.runoff_days],
        "stop_after_decision_day": stop_day, "tool_round_budget": config.max_iterations_per_day,
        "readonly_batch_maximum": config.max_read_only_batch_size,
        "parameter_tuning_after_panel": False, "uses_public_json_only": True}
    write_json(runner.store.run_dir / "simple_policy_protocol.json", provenance)
    started = time.perf_counter()
    totals, days, budget_hits = Counter(), [], []
    try:
        with (runner.store.run_dir / "simple_policy_decisions.jsonl").open("x", encoding="utf-8") as log:
            while not state.done:
                if stop_day is not None and state.day > stop_day:
                    break
                day = state.day
                journal.begin()
                if day > config.max_days:
                    result = journal.execute(state, f"simple-runoff-{day}", "close_business_day", {})
                    if not result.ok:
                        raise RuntimeError(result.data)
                    journal.commit(state)
                    continue
                journal.opening(state, {"role": "user", "content": f"Day {day}: fixed no-LLM diagnostic merchant."})
                rpc = PublicRPC(journal, state, runner.tools, config.max_iterations_per_day, config.max_read_only_batch_size)
                merchant = SimpleMerchant(policy, rpc)
                try:
                    merchant.day()
                except RoundLimit:
                    budget_hits.append(day)
                    merchant.decisions.append({"day": day, "category": "budget_exhausted", "rounds": rpc.rounds})
                result = rpc("close_business_day", {})
                if not result["ok"]:
                    raise RuntimeError(result["data"])
                if rpc.errors:
                    raise RuntimeError(f"unhandled public tool errors: {rpc.errors}")
                journal.commit(state)
                totals.update(rpc.counts)
                days.append({"day": day, "rounds": rpc.rounds, "tool_calls": sum(rpc.counts.values())})
                for row in merchant.decisions:
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
                if day % 20 == 0:
                    print(f"{run_id}: day {day}, {time.perf_counter()-started:.1f}s", flush=True)
            summary = runner.summary(state, "completed" if state.done else "decision_day_limit")
            summary["simple_policy_stress"] = {
                "policy": policy, "tool_counts": dict(totals), "daily_rounds": days,
                "budget_exhaustion_days": budget_hits, "model_calls": 0, "tool_errors": [],
                "elapsed_seconds": round(time.perf_counter()-started, 3),
            }
            runner.store.write_summary(summary)
            write_json(runner.store.run_dir / "completed.json", {"stop_reason": summary["stop_reason"], "done": state.done})
            return {"run_dir": str(runner.store.run_dir), "summary": summary}
    except BaseException as error:
        write_json(runner.store.run_dir / "failure.json", {"day": state.day, "error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        journal.close()


def analyze_batch(batch):
    """Use complete-world score and factual event decomposition, never rent-minus-ext pseudo-CF."""
    batch = Path(batch)
    protocol = json.loads((batch / "protocol.json").read_text())
    originals = protocol["llm_references"]
    llm = []
    for name, original in originals.items():
        summary = original["summary"]
        llm.append({"run_id": name, "seed": original["seed"],
                    "net_gain": summary["value_change"]["official_value_change"],
                    "final_value": summary["official_liquidation_value"]})
    rows = []
    for path in sorted((batch / "runs").glob("*/summary.json")):
        summary = json.loads(path.read_text())
        if not summary["done"]:
            raise ValueError(f"partial run cannot enter full-horizon score table: {path}")
        run = path.parent
        with sqlite3.connect(f"file:{run / 'run.sqlite3'}?mode=ro", uri=True) as db:
            raw = json.loads(db.execute("select value from state where id=1").fetchone()[0])
            assert db.execute("select count(*) from model_calls").fetchone()[0] == 0
        decisions = [json.loads(line) for line in (run / "simple_policy_decisions.jsonl").read_text().splitlines()]
        state_events = Counter(e["event_type"] for e in raw["reservation_events"])
        seed = json.loads((run / "manifest.json").read_text())["config"]["seed"]
        references = [r for r in llm if r["seed"] == seed]
        mean = sum(r["net_gain"] for r in references) / len(references)
        gain = summary["value_change"]["official_value_change"]
        requests = {r["request_id"]: r for r in raw["reservation_requests"]}
        by_archetype = defaultdict(Counter)
        for r in requests.values():
            archetype = next(iter(r["quote_context_by_line"].values()))["archetype"]
            by_archetype[archetype]["requests"] += 1
        for c in raw["reservation_commitments"]:
            r = requests[c["request_id"]]
            archetype = next(iter(r["quote_context_by_line"].values()))["archetype"]
            by_archetype[archetype]["confirmed"] += 1
            by_archetype[archetype][c["status"]] += 1
        econ = summary["reservation_rent_cash"]
        row = {"run_id": run.name, "policy": summary["simple_policy_stress"]["policy"], "seed": seed,
            "done": summary["done"], "bankrupt": summary["bankrupt"], "net_gain": gain,
            "final_value": summary["official_liquidation_value"], "llm_mean_net_gain": round(mean, 2),
            "llm_profit_retention": round(gain / mean, 6) if mean > 0 else None,
            "llm_gain_differences": {r["run_id"]: round(gain-r["net_gain"], 2) for r in references},
            "rental_revenue": summary["rental_revenue"], "reservation_revenue": econ["net_rent_cash_collected"],
            "walk_in_revenue": round(summary["rental_revenue"]-econ["net_rent_cash_collected"], 2),
            "extension_revenue": econ["post_pickup_rent_collected"],
            "extension_requests": state_events["extension_requested"], "extensions_accepted": state_events["extension_accepted"],
            "planned_optional_shortfall_commitments": sum(bool(d.get("planned_missing_optional")) for d in decisions if d["category"] == "confirmed"),
            "realized_optional_shortfall_commitments": sum(e["event_type"] == "pickup_fulfilled" and e["details"].get("optional_line_credit", 0)>0 for e in raw["reservation_events"]),
            "displaced_commitments": sum(d.get("reason") == "profitable_extension_displacement" for d in decisions),
            "cash_outflows": summary["cash_flow"]["outflows"], "reservations": summary["reservations"],
            "by_archetype": dict(by_archetype), "durable_capital": summary["durable_capital"],
            "budget_exhaustion_days": summary["simple_policy_stress"]["budget_exhaustion_days"],
            "tool_calls": sum(summary["simple_policy_stress"]["tool_counts"].values()),
            "maximum_rounds": max(d["rounds"] for d in summary["simple_policy_stress"]["daily_rounds"]),
            "cash_reconciliation_difference": summary["cash_flow"]["reconciliation_difference"],
        }
        rows.append(row)
    by_pair = {(r["policy"], r["seed"]): r for r in rows}
    contrasts = []
    for seed in (42, 73):
        for left, right in (("core_calendar", "core_myopic"), ("core_calendar", "core_no_extension"),
                            ("core_optional_credit", "core_calendar"), ("core_displace", "core_calendar"),
                            ("picnic_calendar", "core_calendar")):
            if (left, seed) in by_pair and (right, seed) in by_pair:
                contrasts.append({"seed": seed, "left": left, "right": right,
                    "net_gain_difference": round(by_pair[(left, seed)]["net_gain"]-by_pair[(right, seed)]["net_gain"], 2)})
    review_flags = [policy for policy in POLICIES if all(
        (policy, seed) in by_pair and not by_pair[(policy, seed)]["bankrupt"]
        and by_pair[(policy, seed)]["llm_profit_retention"] is not None
        and by_pair[(policy, seed)]["llm_profit_retention"] >= .8 for seed in (42, 73))]
    report = {"protocol": protocol, "llm": llm, "results": rows,
        "paired_policy_contrasts": contrasts,
        "both_seeds_retain_at_least_80pct_llm_mean_profit": review_flags,
        "interpretation": "Retrospective fixed-policy adversarial probes, not independent generalization tests. "
        "An 80% profit-retention flag is an engineering screening threshold, not statistical significance. "
        "No profit upper bound or absence-of-hacks claim follows from weak scripts."}
    write_json(batch / "report.json", report)
    lines = ["# V44 fixed-policy stress panel", "", "| Policy | Seed | Net gain | LLM mean | Retention | Extensions | Optional shortfalls | Displaced | Budget hits |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        retention = "n/a" if r["llm_profit_retention"] is None else f"{r['llm_profit_retention']:.1%}"
        lines.append(f"| {r['policy']} | {r['seed']} | {r['net_gain']:.2f} | {r['llm_mean_net_gain']:.2f} | {retention} | {r['extensions_accepted']} | {r['realized_optional_shortfall_commitments']} | {r['displaced_commitments']} | {len(r['budget_exhaustion_days'])} |")
    lines += ["", report["interpretation"], "", "Full costs, channel revenue and business coverage are in report.json. "
              "Extension receipts are part of reservation revenue, not an extra additive income source."]
    lines += ["", "## Paired policy differences", "", "| Seed | Left minus right | Net gain difference |",
              "|---|---|---:|"]
    for c in contrasts:
        lines.append(f"| {c['seed']} | {c['left']} − {c['right']} | {c['net_gain_difference']:.2f} |")
    lines += ["", f"Policies meeting the descriptive 80% screen on both seeds: {review_flags}."]
    (batch / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Standard plotting library, independent of the full heavyweight analyzer.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, seed in zip(axes, (42, 73)):
        for row in [r for r in rows if r["seed"] == seed]:
            run = batch / "runs" / row["run_id"]
            with sqlite3.connect(f"file:{run / 'run.sqlite3'}?mode=ro", uri=True) as db:
                raw = json.loads(db.execute("select value from state where id=1").fetchone()[0])
            history = raw["daily_history"]
            # Use the environment's saved official/liquidation value, not operating P&L.
            values = [d["end"].get("official_liquidation_value", d["end"].get("liquidation_business_value")) for d in history]
            if any(v is None for v in values):
                values = [d["end"]["business_value"] for d in history]
            initial = json.loads((run / "summary.json").read_text())["value_change"]["initial_official_liquidation_value"]
            ax.plot([d["day"] for d in history], [v-initial for v in values], label=row["policy"])
        for name, original in originals.items():
            if original["seed"] == seed:
                curve = original["official_curve"]
                ax.plot([r[0] for r in curve], [r[1] for r in curve], "--", label=name)
        ax.axhline(0, color="grey", lw=.8)
        ax.set(title=f"Seed {seed}", xlabel="Day", ylabel="Official liquidation value change (CNY)")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(batch / "official_net_value.png", dpi=160)
    plt.close(fig)
    return report


def panel(batch):
    batch = Path(batch).resolve()
    protocol = json.loads((batch / "protocol.json").read_text())
    if source_hashes() != protocol["source_sha256"]:
        raise ValueError("diagnostic code changed after freezing")
    tasks = [(str(batch / "references" / f"seed{seed}"), str(batch / "runs"), policy)
             for policy in POLICIES for seed in (42, 73)]
    write_json(batch / "status.json", {"stage": "running", "total": len(tasks), "completed": []})
    done = []
    try:
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(run_one, *task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                done.append(result["summary"]["run_id"])
                write_json(batch / "status.json", {"stage": "running", "total": len(tasks), "completed": done})
        analyze_batch(batch)
        write_json(batch / "completed.json", {"runs": done, "llm_calls": 0})
        write_json(batch / "status.json", {"stage": "completed", "completed": done})
    except BaseException as error:
        write_json(batch / "failure.json", {"completed": done, "error": repr(error), "traceback": traceback.format_exc()})
        raise


def launch(batch, references):
    batch, references = Path(batch).resolve(), Path(references).resolve()
    batch.mkdir(parents=True, exist_ok=False)
    if _implementation_identity()["source_sha256"] != EXPECTED_CORE:
        raise ValueError("V44 core identity mismatch")
    package = Path(__file__).parent
    snapshot = batch / "source" / "outfitter"
    snapshot.mkdir(parents=True)
    for file in package.glob("*.py"):
        shutil.copy2(file, snapshot / file.name)
    shutil.copytree(package / "runtime", snapshot / "runtime", ignore=shutil.ignore_patterns("__pycache__"))
    llm = {}
    for p in sorted(references.iterdir()):
        if not (p / "manifest.json").is_file():
            continue
        m = json.loads((p / "manifest.json").read_text())
        if m["config"]["model"].startswith("scripted-"):
            continue
        if m["controller"]["implementation_identity"]["source_sha256"] != EXPECTED_CORE:
            raise ValueError(f"unexpected reference version: {p}")
        s = json.loads((p / "summary.json").read_text())
        r = json.loads((p / "analysis" / "report.json").read_text())
        initial = s["value_change"]["initial_official_liquidation_value"]
        llm[p.name] = {"seed": m["config"]["seed"], "summary": s,
                       "official_curve": [[d["day"], d["official_liquidation_value_end"]-initial] for d in r["daily"]]}
    if len(llm) != 4:
        raise ValueError("expected the four audited LLM reference runs")
    for seed in (42, 73):
        source = references / f"deepseek-flash-v44-seed{seed}"
        destination = batch / "references" / f"seed{seed}"
        destination.mkdir(parents=True)
        shutil.copy2(source / "manifest.json", destination / "manifest.json")
        shutil.copytree(source / "data", destination / "data")
    write_json(batch / "protocol.json", {**parameters(), "source_sha256": source_hashes(),
        "core_identity": _implementation_identity(), "workers": 2,
        "seeds": [42, 73], "horizon": [200, 14], "llm_references": llm})
    env = dict(os.environ, PYTHONPATH=str(batch / "source"), PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    with (batch / "coordinator.log").open("xb") as output:
        process = subprocess.Popen([sys.executable, "-B", "-m", "outfitter.simple_policy_stress_run", "panel", "--batch", str(batch)],
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, env=env,
            start_new_session=True, cwd=str(batch))
    write_json(batch / "launched.json", {"pid": process.pid, "batch": str(batch), "detached": True})
    print(json.dumps({"pid": process.pid, "batch": str(batch)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "smoke", "launch", "panel", "analyze"))
    parser.add_argument("--reference", default="runs/outfitter/deepseek-flash-v44-seed42")
    parser.add_argument("--references", default="runs/outfitter")
    parser.add_argument("--output")
    parser.add_argument("--batch")
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--stop-day", type=int)
    args = parser.parse_args()
    if args.mode == "run":
        print(json.dumps(run_one(args.reference, args.output, args.policy, stop_day=args.stop_day)["summary"], indent=2))
    elif args.mode == "smoke":
        results = []
        for policy in POLICIES:
            r = run_one(args.reference, args.output, policy, stop_day=7)
            results.append({"policy": policy, "run_dir": r["run_dir"], "day": r["summary"]["day"],
                            "diagnostic": r["summary"]["simple_policy_stress"]})
        write_json(Path(args.output) / "smoke_results.json", results)
    elif args.mode == "launch":
        launch(args.batch, args.references)
    elif args.mode == "panel":
        panel(args.batch)
    else:
        analyze_batch(args.batch)


if __name__ == "__main__":
    main()
