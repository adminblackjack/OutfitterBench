from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .llm import OpenAICompatibleProvider

from .analyze import analyze_run
from .config import OutfitterConfig
from .data import OutfitterDataset
from .environment import OutfitterEnvironment
from .runner import OutfitterRunner
from .store import RunStore


def _provider(config: OutfitterConfig) -> OpenAICompatibleProvider:
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"required API key environment variable is not set: {config.api_key_env}")
    return OpenAICompatibleProvider(
        model=config.model,
        api_key=api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        timeout=config.request_timeout_seconds,
        parallel_tool_calls=config.parallel_tool_calls,
        thinking_type=config.thinking_type,
        reasoning_effort=config.reasoning_effort,
        max_retries=config.max_retries,
        request_rate_limit_per_minute=config.request_rate_limit_per_minute,
        rate_limit_retry_seconds=config.rate_limit_retry_seconds,
        provider_extra_body=config.provider_extra_body,
    )


def _resume_config(run_dir: Path) -> OutfitterConfig:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    values = dict(manifest["config"])
    values["data_dir"] = str(run_dir / manifest.get("runtime_data", "data"))
    return OutfitterConfig.from_dict(values)


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def command_run(args: argparse.Namespace) -> None:
    config = OutfitterConfig.from_yaml(args.config)
    if args.days is not None:
        config.max_days = args.days
    if args.runs_dir is not None:
        config.runs_dir = args.runs_dir
    config.validate()
    runner = OutfitterRunner.create(config=config, provider=_provider(config), run_id=args.run_id)
    _print(runner.run(stop_after_model_calls=args.stop_after_model_calls,
                      stop_after_day=args.stop_after_day))


def command_resume(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).expanduser().resolve()
    if (run_dir / 'calendar_protocol.json').is_file():
        raise ValueError('Use outfitter-bench baseline --resume RUN_DIR for Calendar')
    config = _resume_config(run_dir)
    runner = OutfitterRunner.resume(run_dir=run_dir, provider=_provider(config))
    _print(runner.run(stop_after_model_calls=args.stop_after_model_calls,
                      stop_after_day=args.stop_after_day))


def command_inspect(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).expanduser().resolve()
    store = RunStore.resume(run_dir)
    config = _resume_config(run_dir)
    environment = OutfitterEnvironment(config, OutfitterDataset.load(config.data_dir))
    state = store.load_state()
    value = environment.business_value(state)
    _print(
        {
            "run_dir": str(run_dir),
            "status": environment.observation(state, consume_inbox=False),
            "evaluation_score": (
                environment.evaluation_score(state) if state.done else None
            ),
            "final_business_value": (
                value["business_value"] if state.done and not state.bankrupt else None
            ),
            "official_liquidation_value": (
                value["official_liquidation_value"]
                if state.done and not state.bankrupt
                else None
            ),
            "efficiency_adjusted_value": value["efficiency_adjusted_value"],
            "business_value_if_season_ended_now": (
                value["business_value"] if not state.done else None
            ),
            "going_concern_business_value": value[
                "going_concern_business_value"
            ],
            "contract_adjusted_economic_value": value[
                "contract_adjusted_economic_value"
            ],
            "book_net_assets": value["book_net_assets"],
            "business_value_breakdown": value,
            "model_calls": store.model_call_count(),
        }
    )


def command_analyze(args: argparse.Namespace) -> None:
    if args.output_dir is not None and len(args.run_dirs) != 1:
        raise ValueError("--output-dir can only be used with one run directory")
    rows = [
        analyze_run(
            raw,
            output_dir=args.output_dir,
            plots=not args.no_plots,
            dpi=args.dpi,
        )
        for raw in args.run_dirs
    ]
    _print({"analyses": rows})


def command_baseline(args: argparse.Namespace) -> None:
    from .calendar_baseline import run_calendar
    if args.resume:
        if args.config or args.run_id or args.runs_dir:
            raise ValueError('--resume cannot be combined with configuration or output overrides')
        result = run_calendar(resume_dir=args.resume, stop_after_day=args.stop_after_day)
    else:
        if not args.config:
            raise ValueError('--config is required for a new Calendar run')
        config = OutfitterConfig.from_yaml(args.config)
        if args.runs_dir:
            config.runs_dir = args.runs_dir
        result = run_calendar(config, run_id=args.run_id, stop_after_day=args.stop_after_day)
    _print(result)


def command_evaluate(args: argparse.Namespace) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from .evaluation import evaluate_run_cutoffs
    from .evaluation.execution import execution_budget
    from .evaluation.report import aggregate_reports
    if args.output_dir is not None and len(args.run_dirs) != 1:
        raise ValueError('--output-dir requires exactly one run')
    if args.workers < 1:
        raise ValueError('--workers must be positive')
    def evaluate(path):
        result = evaluate_run_cutoffs(path, through_days=args.through_days or ([args.through_day] if args.through_day is not None else None), extended=args.extended,
                              source_root=args.source_root, output_dir=args.output_dir,
                              plots=False, sample_days=args.sample_days,
                              cpu_budget=size,budget_dir=directory,state_reload=args.state_reload)
        return result
    with execution_budget(args.cpu_budget) as (directory,size):
        with ThreadPoolExecutor(max_workers=min(args.workers,len(args.run_dirs),size)) as pool:
            rows = [row for group in pool.map(evaluate,args.run_dirs) for row in group]
    if not args.no_plots:
        from .evaluation.report import save_report
        from pathlib import Path
        for row in rows:save_report(row['metrics'],Path(row['output_dir']),plots=True)
    _print({'evaluations':[{k:v for k,v in r.items() if k!='metrics'} for r in rows],
            'comparison_groups':aggregate_reports([r['metrics'] for r in rows])})


def command_diagnostics(args: argparse.Namespace) -> None:
    from .behavioral_diagnostics import diagnose_many, write_cohort, worker_count
    rows = diagnose_many(args.run_dirs, through_day=args.through_day, workers=args.workers)
    if args.aggregate_dir:
        write_cohort(rows, args.aggregate_dir, workers=args.workers, plots=not args.no_plots)
    _print({"format": "outfitter-behavioral-diagnostics-v1",
            "workers": worker_count(args.workers, len(rows)),
            "runs": [{"run_id": r["profile"]["run_id"], "output_dir": r["output_dir"],
                      "extended_status": r["profile"]["extended_status"]} for r in rows],
            "aggregate_dir": args.aggregate_dir})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="outfitter-bench")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="start a new benchmark run")
    run.add_argument("--config")
    run.add_argument("--run-id")
    run.add_argument("--days", type=int)
    run.add_argument("--runs-dir")
    run.add_argument("--stop-after-model-calls", type=int)
    run.add_argument("--stop-after-day", type=int, help="pause after closing this absolute business day")
    run.set_defaults(function=command_run)
    resume = commands.add_parser("resume", help="resume a saved run")
    resume.add_argument("run_dir")
    resume.add_argument("--stop-after-model-calls", type=int)
    resume.add_argument("--stop-after-day", type=int, help="pause after closing this absolute business day")
    resume.set_defaults(function=command_resume)
    inspect = commands.add_parser("inspect", help="inspect a saved run")
    inspect.add_argument("run_dir")
    inspect.set_defaults(function=command_inspect)
    analyze = commands.add_parser("analyze", help="materialize run tables, report, and plots")
    analyze.add_argument("run_dirs", nargs="+")
    analyze.add_argument("--output-dir", help="custom output directory (one run only)")
    analyze.add_argument("--no-plots", action="store_true", help="write tables/report without PNG figures")
    analyze.add_argument("--dpi", type=int, default=160, help="PNG resolution, 72-600 (default: 160)")
    analyze.set_defaults(function=command_analyze)
    evaluate = commands.add_parser('evaluate', help='offline trajectory metrics; expensive replay is opt-in')
    evaluate.add_argument('run_dirs', nargs='+')
    cutoff=evaluate.add_mutually_exclusive_group()
    cutoff.add_argument('--through-day', type=int)
    cutoff.add_argument('--through-days', nargs='+',type=int,help='emit multiple cutoffs during one replay')
    evaluate.add_argument('--extended', action='store_true', help='enable frozen-source replay and local candidates (no LLM)')
    evaluate.add_argument('--source-root', help='matching frozen source, required when current source identity differs')
    evaluate.add_argument('--workers', type=int, default=2)
    evaluate.add_argument('--cpu-budget',type=int,help='global CPU task limit, including local branches (default: min(8, available CPUs))')
    evaluate.add_argument('--state-reload',choices=['reuse','reference','audit'],default='reuse',
                         help='reuse verified pure-query state; reference reloads every call; audit verifies every reuse')
    evaluate.add_argument('--output-dir')
    evaluate.add_argument('--no-plots', action='store_true')
    evaluate.add_argument('--sample-days', nargs='+', type=int, help='override local branch dates; marks output as custom diagnostic')
    evaluate.set_defaults(function=command_evaluate)
    diagnostics = commands.add_parser('diagnostics', help='derive focused diagnostics from existing reports; no replay')
    diagnostics.add_argument('run_dirs', nargs='+')
    diagnostics.add_argument('--through-day', type=int, default=214)
    diagnostics.add_argument('--workers', type=int, help='default: all available CPUs, bounded by independent tasks')
    diagnostics.add_argument('--aggregate-dir', help='separate cross-run output directory; raw comparison groups stay separate')
    diagnostics.add_argument('--no-plots', action='store_true')
    diagnostics.set_defaults(function=command_diagnostics)
    baseline = commands.add_parser('baseline', help='run the paper Calendar policy without an LLM')
    baseline.add_argument('--config')
    baseline.add_argument('--run-id')
    baseline.add_argument('--runs-dir')
    baseline.add_argument('--resume', metavar='RUN_DIR')
    baseline.add_argument('--stop-after-day', type=int)
    baseline.set_defaults(function=command_baseline)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
