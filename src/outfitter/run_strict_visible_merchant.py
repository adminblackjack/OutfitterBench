"""Run a daily-stateless no-LLM policy through the exact agent tool interface."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time

import jsonschema

from .config import OutfitterConfig
from .prompts import day_message
from .run_scripted_merchant import DailyJournal, NoModel
from .runner import OutfitterRunner
from .strict_visible_merchant import (
    ALLOWED_ACTION_TOOLS, ALLOWED_FACT_TOOLS, POLICIES, PROTOCOL, StrictVisibleMerchant,
)


def run(config, policy, *, run_id):
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    started = time.perf_counter()
    config = replace(config, model=f"scripted-v32-{policy}", base_url=None, reasoning_effort=None,
                     charge_model_usage_to_cash=False)
    runner = OutfitterRunner.create(config=config, provider=NoModel(), run_id=run_id)
    if not runner.environment._v32_service_commitments():
        raise ValueError("strict-visible policies require V32")
    definitions = {row["function"]["name"]: row["function"]["parameters"]
                   for row in runner.tools.definitions()}
    exact_definitions = {name: json.loads(json.dumps(schema), parse_float=Decimal)
                         for name, schema in definitions.items()}
    active_allowed_tools = ALLOWED_FACT_TOOLS | ALLOWED_ACTION_TOOLS
    if not config.retail_enabled:
        active_allowed_tools = active_allowed_tools - {"view_retail_inventory"}
    if active_allowed_tools - definitions.keys():
        raise ValueError("strict-visible allowlist contains a tool absent from the evaluated schema")
    provenance = {
        "protocol": PROTOCOL, "policy": policy, "no_llm": True,
        "facts_enter_policy_only_as_json_tool_results": True,
        "new_policy_instance_each_business_day": True,
        "free_cross_day_python_memory": False, "memory_tools_used": False,
        "tool_arguments_validated_against_evaluated_schema": True,
        "environment_state_passed_to_policy": False,
        "not_a_profit_upper_bound_or_llm_capability_result": True,
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__), Path(__file__).with_name("strict_visible_merchant.py"))},
    }
    with (runner.store.run_dir/"strict_visible_protocol.json").open("x") as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    decisions = (runner.store.run_dir/"strict_visible_decisions.jsonl").open("x")
    state = runner.store.load_state()
    journal = DailyJournal(runner.store, runner.tools)
    seq, counts, errors, policy_instances = 0, Counter(), [], 0

    def rpc(name, args):
        nonlocal seq
        if name not in active_allowed_tools:
            raise RuntimeError(f"policy requested non-allowlisted tool: {name}")
        # Decimal avoids the validator's binary-float false rejection of valid
        # cent amounts such as 4.48 against multipleOf: 0.01.
        exact_args = json.loads(json.dumps(args), parse_float=Decimal)
        jsonschema.validate(instance=exact_args, schema=exact_definitions[name])
        seq += 1
        call_id = f"strict-visible-{seq:07d}"
        result = journal.execute(state, call_id, name, args)
        counts[name] += 1
        if not result.ok:
            errors.append({"day": state.day, "tool": name, "arguments": args, "data": result.data})
        # JSON roundtrip prevents dataclass/object references crossing the firewall.
        return json.loads(json.dumps({"ok": result.ok, "data": result.data,
                                      "terminal_day": result.terminal_day}, ensure_ascii=False))

    try:
        while not state.done:
            journal.begin()
            if state.day > config.max_days:
                result = journal.execute(state, f"automatic-runoff-close-day-{state.day}",
                                         "close_business_day", {})
                if not result.ok:
                    raise RuntimeError(result.data)
                journal.commit(state)
                continue
            observation = runner.environment.observation(state, consume_inbox=True)
            journal.opening(state, {"role": "user", "content": day_message(observation)})
            merchant = StrictVisibleMerchant(policy, rpc)
            policy_instances += 1
            merchant.start()
            merchant.day()
            result = journal.execute(state, f"strict-visible-close-{state.day:03d}",
                                     "close_business_day", {})
            if not result.ok:
                raise RuntimeError(result.data)
            runner.environment._assert_v26_reservation_financial_invariants(state)
            journal.commit(state)
            for row in merchant.decisions:
                decisions.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            decisions.flush()
            if len(state.daily_history) % 10 == 0:
                print(f"{run_id}: completed day {len(state.daily_history)}, elapsed {time.perf_counter()-started:.1f}s",
                      flush=True)
        summary = runner.summary(state, "completed")
        summary["strict_visible_merchant"] = {**provenance, "daily_policy_instances": policy_instances,
            "tool_counts": dict(counts), "tool_error_records": errors, "model_calls": 0,
            "wall_seconds": round(time.perf_counter()-started, 3)}
        runner.store.write_summary(summary)
        return summary
    except BaseException as error:
        with (runner.store.run_dir/"strict_visible_failure.json").open("x") as stream:
            json.dump({"day": state.day, "error": repr(error), "tool_errors": errors},
                      stream, ensure_ascii=False, indent=2)
        raise
    finally:
        journal.close()
        decisions.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/outfitter_v32_service_commitments_main.yaml")
    parser.add_argument("--policy", choices=POLICIES, default="visible_owned_calendar")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs/outfitter")
    parser.add_argument("--days", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = OutfitterConfig.from_yaml(args.config)
    config.runs_dir = args.runs_dir
    if args.days is not None:
        config.max_days = args.days
    if args.seed is not None:
        config.seed = args.seed
    result = run(config, args.policy, run_id=args.run_id)
    print(json.dumps({key: result[key] for key in (
        "run_id", "done", "bankrupt", "value_change", "service_commitments")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
