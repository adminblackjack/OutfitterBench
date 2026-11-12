"""Run an auditable V32 merchant without constructing an LLM client.

Policy gets only public JSON tools. State and hidden outcome evaluation stay in
this driver. Outputs use the ordinary run database/summary and Analyzer format.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

from .config import OutfitterConfig
from .prompts import day_message
from .runner import OutfitterRunner
from .scripted_merchant import PROTOCOL, SUPPORTED_POLICIES, PublicMerchant
from .scripted_supplier_trial import (
    POLICIES as SUPPLIER_TRIAL_POLICIES,
    SupplierTrialOverlay,
)


class NoModel:
    def complete(self, **kwargs):
        raise AssertionError("scripted merchant must never call a model")


class DailyJournal:
    """Scripted-only persistence: one atomic day, full tool/message history.

    Fresh, sequential call IDs need no history scan or retry lookup. The LLM
    RunStore path is unchanged. A crash rolls back the current day, not earlier
    days; scripted episodes still do not support resuming an interrupted policy.
    """
    def __init__(self, store, tools):
        self.store, self.tools = store, tools
        self.db = store._connect()

    def begin(self):
        self.db.execute("BEGIN IMMEDIATE")

    def opening(self, state, message):
        self.db.execute("INSERT INTO day_starts(day) VALUES(?)", (state.day,))
        self.db.execute("INSERT INTO messages(day,scope,value) VALUES(?,?,?)",
                        (state.day,"controller",json.dumps(message,ensure_ascii=False)))

    def execute(self, state, call_id, name, args):
        day = state.day
        result = self.tools.execute(state,name,args)
        encoded = json.dumps({"ok":result.ok,"data":result.data,"terminal_day":result.terminal_day},
                             ensure_ascii=False,sort_keys=True)
        self.db.execute("INSERT INTO tool_calls(call_id,day,name,arguments,result) VALUES(?,?,?,?,?)",
                        (call_id,day,name,json.dumps(args,ensure_ascii=False,sort_keys=True),encoded))
        message = {"role":"tool","tool_call_id":call_id,"name":name,"content":self.tools.encode_result(result)}
        self.db.execute("INSERT INTO messages(day,scope,value) VALUES(?,?,?)",
                        (day,"controller",json.dumps(message,ensure_ascii=False)))
        return result

    def commit(self, state):
        self.store._save(self.db,state)
        self.db.commit()

    def close(self):
        self.db.rollback()  # No-op after a successful day; never commit a partial day.
        self.db.close()


def run(
    config,
    policy,
    *,
    run_id,
    persistence="daily",
    supplier_trial_policy="no_trial",
):
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"policy must be one of {SUPPORTED_POLICIES}")
    if persistence not in {"daily","per_tool"}:
        raise ValueError("persistence must be daily or per_tool (slow equivalence control)")
    if supplier_trial_policy not in SUPPLIER_TRIAL_POLICIES:
        raise ValueError(
            f"supplier_trial_policy must be one of {SUPPLIER_TRIAL_POLICIES}"
        )
    started = time.perf_counter()
    config = replace(config, model=f"scripted-v32-{policy}", base_url=None, reasoning_effort=None,
                     charge_model_usage_to_cash=False)
    runner = OutfitterRunner.create(config=config,provider=NoModel(),run_id=run_id)
    if not runner.environment._v32_service_commitments():
        raise ValueError("use a V32 runtime; legacy calibration remains separate")
    seq, counts, errors = 0, Counter(), []
    state = runner.store.load_state()
    # Save policy provenance separately; do not misrepresent scripted calls as model calls.
    provenance = {"protocol":PROTOCOL,"policy":policy,
        "supplier_trial_policy":supplier_trial_policy,
        "information":"public tool results only",
        "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (
                 Path(__file__),
                 Path(__file__).with_name("scripted_merchant.py"),
                 Path(__file__).with_name("scripted_supplier_trial.py"),
             )},
        "no_llm":True,"no_profit_or_ordering_gate":True,"state_checkpointing":persistence,
        "limits":"No hidden demand law, optimality claim, or validation of LLM ability from script ranking."}
    with (runner.store.run_dir/"scripted_protocol.json").open("x") as stream:
        json.dump(provenance,stream,ensure_ascii=False,indent=2)
    log = (runner.store.run_dir/"scripted_decisions.jsonl").open("x")
    journal = DailyJournal(runner.store,runner.tools) if persistence == "daily" else None

    def rpc(name,args):
        nonlocal state,seq
        seq += 1
        automatic = name == "close_business_day" and state.day > config.max_days
        call_id = f"automatic-runoff-close-day-{state.day}" if automatic else f"scripted-call-{seq:07d}"
        if journal is None:
            state,result,_ = runner.store.execute_tool(call_id,name,args,runner.tools.execute)
            runner.store.record_tool_message(state,tool_call_id=call_id,name=name,content=runner.tools.encode_result(result))
        else:
            result = journal.execute(state,call_id,name,args)
        counts[name] += 1
        if not result.ok:
            errors.append({"day":state.day,"name":name,"arguments":args,"data":result.data})
        return {"ok":result.ok,"data":json.loads(json.dumps(result.data)),"terminal_day":result.terminal_day}

    merchant = PublicMerchant(
        policy, rpc, retail_enabled=config.retail_enabled
    )
    trial_overlay = SupplierTrialOverlay(supplier_trial_policy, rpc)
    logged = 0
    try:
        while not state.done:
            if journal is not None:
                journal.begin()
            if state.day > config.max_days:
                result = rpc("close_business_day",{})
                if not result["ok"]:
                    raise RuntimeError(result["data"])
                if journal is not None:
                    journal.commit(state)
                continue
            observation = runner.environment.observation(state,consume_inbox=True)
            opening = {"role":"user","content":day_message(observation)}
            if journal is None:
                runner.store.record_day_start(state,opening)
            else:
                journal.opening(state,opening)
            if not merchant.offers:
                merchant.start()
            if runner.environment._v37_supplier_trial():
                trial_overlay.day(merchant, state.day)
            merchant.day()
            result = rpc("close_business_day",{})
            if not result["ok"]:
                raise RuntimeError(result["data"])
            runner.environment._assert_v26_reservation_financial_invariants(state)
            if journal is not None:
                journal.commit(state)
            for decision in merchant.decisions[logged:]:
                log.write(json.dumps(decision,ensure_ascii=False,allow_nan=False)+"\n")
            logged = len(merchant.decisions)
            log.flush()
            if len(state.daily_history)%10 == 0:
                print(f"{run_id}: completed day {len(state.daily_history)}, elapsed {time.perf_counter()-started:.1f}s",flush=True)
        summary = runner.summary(state,"completed")
        summary["scripted_merchant"] = {**provenance,"tool_counts":dict(counts),"tool_error_records":errors,
            "decision_records":len(merchant.decisions),"model_calls":0,
            "wall_seconds":round(time.perf_counter()-started,3)}
        runner.store.write_summary(summary)
        return summary
    except BaseException as error:
        with (runner.store.run_dir/"scripted_failure.json").open("x") as stream:
            json.dump({"day":state.day,"error":repr(error),"tool_errors":errors},stream,ensure_ascii=False,indent=2)
        raise
    finally:
        if journal is not None:
            journal.close()
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/outfitter_v32_service_commitments_main.yaml")
    parser.add_argument("--policy",choices=SUPPORTED_POLICIES,default="calendar")
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--runs-dir",default="runs/outfitter")
    parser.add_argument("--days",type=int)
    parser.add_argument("--seed",type=int)
    parser.add_argument(
        "--supplier-trial-policy",
        choices=SUPPLIER_TRIAL_POLICIES,
        default="no_trial",
    )
    args = parser.parse_args()
    config = OutfitterConfig.from_yaml(args.config)
    config.runs_dir = args.runs_dir
    if args.days is not None:
        config.max_days = args.days
    if args.seed is not None:
        config.seed = args.seed
    result = run(
        config,
        args.policy,
        run_id=args.run_id,
        supplier_trial_policy=args.supplier_trial_policy,
    )
    print(json.dumps({k:result[k] for k in ("run_id","done","bankrupt","value_change","service_commitments")},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
