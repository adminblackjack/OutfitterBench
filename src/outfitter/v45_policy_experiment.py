"""No-LLM V45 reservation-policy experiments using the public actor interface."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from decimal import Decimal, ROUND_HALF_UP
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .config import OutfitterConfig
from .reservation_difficulty import rental_amount
from .runner import OutfitterRunner, _implementation_identity
from .run_scripted_merchant import DailyJournal, NoModel
from .simple_policy_stress import PRICES, SimpleMerchant
from .simple_policy_stress_run import PublicRPC, RoundLimit, write_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/outfitter_v45_reservation_difficulty_main.yaml"
POLICIES = {
    "A": "picnic_calendar", "B": "core_myopic", "C": "core_calendar",
    "D": "core_no_extension", "E": "core_optional_credit",
    "F": "core_displace", "G": "core_calendar",
    "H": "core_calendar", "I": "core_calendar",
}
RUNTIME_DIRS = {
    "100": "v45-reservation-difficulty-runtime-candidate-20260924",
    "140": "v45-reservation-difficulty-peak140-candidate-20260924",
    "180": "v45-reservation-difficulty-peak180-candidate-20260924",
    "220": "v45-reservation-difficulty-peak220-candidate-20260924",
}


class V45Merchant(SimpleMerchant):
    def __init__(self, policy, rpc, price_scale):
        super().__init__(POLICIES[policy], rpc)
        self.v45_policy = policy
        self.price_scale = float(price_scale)
        self.prices = {offer: float((Decimal(str(price * price_scale)))
                       .quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
                       for offer, price in PRICES.items()}
        self.rent_formula = rental_amount

    def risk(self, alert):
        missing = alert.get("missing_optional_line_ids", [])
        if missing:
            cid = alert["commitment_id"]
            reply = self.call("resolve_reservation_risks", {"actions": [{
                "commitment_id": cid, "action": "inquire_partial_fulfillment",
            }]})["actions"][0]
            self.log("partial_inquiry", commitment_id=cid, customer_accepted=reply["customer_accepted"],
                     rent_credit=reply["rent_credit"], compensation=reply["compensation"])
            if reply["customer_accepted"]:
                self.call("resolve_reservation_risks", {"actions": [{
                    "commitment_id": cid, "action": "select_partial_fulfillment",
                }]})
                self.load_record([cid])
                return
            self.cancel(cid, "partial_fulfillment_declined")
            return
        super().risk(alert)

    def request(self, request):
        if self.v45_policy == "G":
            fixed = request["request_kind"] == "fixed_reward_offer"
            tool = "respond_reservation_service_offers" if fixed else "quote_reservation_inquiries"
            result = self.call(tool, {"decisions": [{"request_id": request["request_id"],
                                                    "action": "decline"}]})["decisions"][0]
            self.log("request", request_id=request["request_id"], action="decline",
                     status=result["status"])
            return
        super().request(request)

    def _priority(self, request):
        if request["request_kind"] == "fixed_reward_offer":
            plans = request["service_offer_terms"]["complete_plans"]
            potential = max((float(plan["total_rent"]) for plan in plans), default=0.0)
        else:
            potential = 0.0
            for line in request["lines"]:
                if not line["required"]:
                    continue
                choices = [
                    rental_amount(self.prices[offer], int(line["duration_days"]),
                                  int(term["asset_quantity"]))
                    for term in line["profile_terms"]
                    for offer in term["frozen_member_offer_ids"]
                    if offer in self.targets
                ]
                potential += min(choices) if choices else 0.0
        earliest = min(int(row["pickup_day"]) for row in request["pickup_date_options"])
        duration = max(1, int(request["return_day"]) - int(request["pickup_day"]))
        return (potential / (1.0 + 0.25 * (duration - 1)), -earliest,
                request["request_id"])

    def assess(self, candidate, exclude=None):
        result = super().assess(candidate, exclude)
        if self.v45_policy != "I" or not result["accept"]:
            return result
        # Recomputed daily from public calendar forecasts and current commitments.
        # The current booking earns its full actual rent; this only prices days
        # whose future peak opportunity could be displaced by that booking.
        shadow = 0.0
        for line in candidate["lines"]:
            offer = line["offer_id"]
            if offer not in self.prices:
                continue
            for day in range(candidate["pickup"] + 1, candidate["return"]):
                calendar = self.calendar_by_day.get(day, {})
                if calendar.get("day_type") == "public_holiday" or calendar.get("is_weekend"):
                    shadow += self.prices[offer] * int(line["asset_quantity"]) * 0.55
        net = round(result["margin"] - shadow, 2)
        return {**result, "accept": net > 0, "margin": net,
                "gross_margin": result["margin"], "opportunity_cost": round(shadow, 2)}

    def day(self):
        self.load()
        self.calendar_by_day = {}
        if self.v45_policy == "I":
            response = self.call("weather_forecast", {"horizon_days": 14})
            rows = response.get("forecast", response.get("days", response))
            if isinstance(rows, dict):
                rows = rows.get("forecast", rows.get("days", []))
            self.calendar_by_day = {int(row["day"]): row for row in rows}
        self.setup_prices()
        for change in list(self.inbox["changes"]):
            if change["commitment_id"] in self.records:
                self.change(change)
        for alert in list(self.inbox["risk_alerts"]):
            if alert["commitment_id"] in self.records:
                self.risk(alert)
        self.upkeep()
        self.replenish()
        raw = list(self.inbox["requests"] + self.inbox.get("service_offers", []))
        requests = []
        for request in raw:
            workspace = self.call("view_reservation_quote_workspace", {
                "mode": "requests", "request_ids": [request["request_id"]],
                "offer_projection": "references_only",
            })
            requests.append(workspace["requests"][0])
        if self.v45_policy == "H":
            requests.sort(key=self._priority, reverse=True)
        else:
            requests.sort(key=lambda row: (row.get("default_pickup_day", row["pickup_day"]),
                                           row["request_id"]))
        for request in requests:
            self.request(request)
        self.protect()


def run_one(config_file, runtime_dir, runs_dir, policy, seed, price_scale, stop_day=None):
    if json.loads((Path(runtime_dir)/"metadata.json").read_text()).get("autonomous_rentals_policy"):
        raise ValueError("V47 requires native autonomous-return policies, not legacy return-date assumptions")
    config = OutfitterConfig.from_yaml(config_file)
    config = replace(config, data_dir=str(Path(runtime_dir).resolve()),
                     runs_dir=str(Path(runs_dir).resolve()), seed=int(seed),
                     model=f"scripted-v45-{policy}", base_url=None)
    tag = Path(runtime_dir).name.replace("v45-reservation-difficulty-", "")
    run_id = f"{tag}-{policy}-seed{seed}-price{str(price_scale).replace('.', 'p')}"
    runner = OutfitterRunner.create(config=config, provider=NoModel(), run_id=run_id)
    state = runner.store.load_state()
    journal = DailyJournal(runner.store, runner.tools)
    started = time.perf_counter()
    counts, days, budget_hits = Counter(), [], []
    try:
        with (runner.store.run_dir / "v45_policy_decisions.jsonl").open("x", encoding="utf-8") as log:
            while not state.done:
                if stop_day is not None and state.day > stop_day:
                    break
                day = state.day
                journal.begin()
                if day > config.max_days:
                    result = journal.execute(state, f"runoff-{day}", "close_business_day", {})
                    if not result.ok:
                        raise RuntimeError(result.data)
                    journal.commit(state)
                    continue
                journal.opening(state, {"role": "user", "content": f"Day {day}: V45 no-LLM policy {policy}."})
                rpc = PublicRPC(journal, state, runner.tools,
                                config.max_iterations_per_day, config.max_read_only_batch_size)
                merchant = V45Merchant(policy, rpc, price_scale)
                try:
                    merchant.day()
                except RoundLimit:
                    budget_hits.append(day)
                    merchant.decisions.append({"day": day, "category": "budget_exhausted"})
                result = rpc("close_business_day", {})
                if not result["ok"]:
                    raise RuntimeError(result["data"])
                if rpc.errors:
                    raise RuntimeError(f"public tool errors: {rpc.errors}")
                journal.commit(state)
                counts.update(rpc.counts)
                days.append({"day": day, "rounds": rpc.rounds})
                for row in merchant.decisions:
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
                if day % 20 == 0:
                    print(f"{run_id}: day {day}, {time.perf_counter()-started:.1f}s", flush=True)
        summary = runner.summary(state, "completed" if state.done else "decision_day_limit")
        summary["v45_policy"] = {"policy": policy, "seed": seed,
                                 "price_scale": price_scale, "tool_counts": dict(counts),
                                 "daily_rounds": days, "budget_exhaustion_days": budget_hits,
                                 "model_calls": 0, "elapsed_seconds": round(time.perf_counter()-started, 3)}
        runner.store.write_summary(summary)
        write_json(runner.store.run_dir / "completed.json", {
            "done": state.done, "stop_reason": summary["stop_reason"]})
        return {"run_dir": str(runner.store.run_dir), "done": state.done,
                "official_gain": summary["value_change"]["official_value_change"]}
    except BaseException as error:
        write_json(runner.store.run_dir / "failure.json", {
            "day": state.day, "error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        journal.close()


def panel(manifest):
    spec = json.loads(Path(manifest).read_text())
    tasks = list(spec["tasks"])
    output = Path(spec["output_dir"])
    write_json(output / "status.json", {"stage": "running", "total": len(tasks),
                                       "completed": [], "core": _implementation_identity()})
    complete = []
    try:
        with ProcessPoolExecutor(max_workers=int(spec.get("workers", 3))) as pool:
            futures = {pool.submit(run_one, spec["config_file"], task["runtime_dir"],
                                   output / "runs", task["policy"], task["seed"],
                                   task["price_scale"], task.get("stop_day")): task
                       for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                complete.append(result)
                write_json(output / "status.json", {"stage": "running",
                    "total": len(tasks), "completed": complete})
        write_json(output / "status.json", {"stage": "completed",
            "total": len(tasks), "completed": complete})
    except BaseException as error:
        write_json(output / "status.json", {"stage": "failed", "total": len(tasks),
            "completed": complete, "error": repr(error), "traceback": traceback.format_exc()})
        raise


def prepare(output_dir, config_file, *, variants, policies, seeds, scales,
            workers=3, stop_day=None):
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    tasks = []
    for variant in variants:
        runtime = ROOT / "src/outfitter/data_validation/artifacts" / RUNTIME_DIRS[variant]
        if not runtime.is_dir():
            raise ValueError(f"missing V45 runtime: {runtime}")
        for scale in scales:
            for policy in policies:
                for seed in seeds:
                    tasks.append({"runtime_dir": str(runtime), "policy": policy,
                                  "seed": seed, "price_scale": scale,
                                  "stop_day": stop_day, "variant": variant})
    spec = {"format": "outfitter-v45-no-llm-panel-v1",
            "output_dir": str(output), "config_file": str(Path(config_file).resolve()),
            "workers": workers, "tasks": tasks,
            "core_identity": _implementation_identity(), "llm_calls": 0}
    write_json(output / "manifest.json", spec)
    print(json.dumps({"manifest": str(output / "manifest.json"),
                      "task_count": len(tasks)}, indent=2))


def launch(manifest):
    manifest = Path(manifest).resolve()
    spec = json.loads(manifest.read_text())
    output = Path(spec["output_dir"])
    if (output / "launched.json").exists():
        raise ValueError("panel was already launched")
    if _implementation_identity() != spec["core_identity"]:
        raise ValueError("environment source changed after panel preparation")
    source = output / "source" / "outfitter"
    source.mkdir(parents=True, exist_ok=False)
    package = Path(__file__).parent
    for path in package.glob("*.py"):
        shutil.copy2(path, source / path.name)
    shutil.copytree(package / "runtime", source / "runtime",
                    ignore=shutil.ignore_patterns("__pycache__"))
    config_copy = output / "config.yaml"
    shutil.copy2(spec["config_file"], config_copy)
    datasets = {}
    frozen_tasks = []
    for task in spec["tasks"]:
        runtime = Path(task["runtime_dir"])
        if str(runtime) not in datasets:
            destination = output / "data" / runtime.name
            shutil.copytree(runtime, destination)
            datasets[str(runtime)] = str(destination)
        frozen_tasks.append({**task, "runtime_dir": datasets[str(runtime)]})
    frozen = {**spec, "config_file": str(config_copy), "tasks": frozen_tasks,
              "source_snapshot": str(source), "data_snapshots": datasets}
    frozen_manifest = output / "frozen_manifest.json"
    write_json(frozen_manifest, frozen)
    environment = dict(os.environ, PYTHONPATH=str(output / "source"),
                       PYTHONDONTWRITEBYTECODE="1")
    with (output / "coordinator.log").open("xb") as stream:
        child = subprocess.Popen(
            [sys.executable, "-B", "-m", "outfitter.v45_policy_experiment",
             "panel", "--manifest", str(frozen_manifest)],
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            cwd=str(output), env=environment, start_new_session=True)
    write_json(output / "launched.json", {"pid": child.pid,
        "manifest": str(frozen_manifest), "detached": True})
    print(json.dumps({"pid": child.pid, "output_dir": str(output)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "panel", "prepare", "launch"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--runtime-dir")
    parser.add_argument("--runs-dir")
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--price-scale", type=float, default=1.0)
    parser.add_argument("--stop-day", type=int)
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--variants", default="100,140,180,220")
    parser.add_argument("--policies", default="C")
    parser.add_argument("--seeds", default="42,73")
    parser.add_argument("--price-scales", default="0.75,1.0,1.25,1.5")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.mode == "run":
        if not (args.runtime_dir and args.runs_dir and args.policy):
            parser.error("run requires --runtime-dir, --runs-dir, and --policy")
        print(json.dumps(run_one(args.config, args.runtime_dir, args.runs_dir,
                                 args.policy, args.seed, args.price_scale,
                                 args.stop_day), indent=2))
    elif args.mode == "panel":
        panel(args.manifest)
    elif args.mode == "prepare":
        if not args.output_dir:
            parser.error("prepare requires --output-dir")
        prepare(args.output_dir, args.config,
                variants=args.variants.split(","), policies=args.policies.split(","),
                seeds=[int(x) for x in args.seeds.split(",")],
                scales=[float(x) for x in args.price_scales.split(",")],
                workers=args.workers, stop_day=args.stop_day)
    else:
        launch(args.manifest)


if __name__ == "__main__":
    main()
