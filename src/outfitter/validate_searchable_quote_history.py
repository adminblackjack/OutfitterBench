"""Validate the V17 query-only projection against V16 and real trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from .analyze import _load_run
from .calibrate import run_policy
from .config import OutfitterConfig
from .data import OutfitterDataset
from .environment import OutfitterEnvironment
from .tool_executor import ToolExecutor


FORMAT = "outfitter-v17-searchable-quote-history-validation-v1"
DEFAULT_SEEDS = (17, 29, 42, 59, 73)
DEFAULT_WORLDS = ("conservative", "base", "convenience-heavy")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _policy_config(
    config_path: str,
    runtime: str,
    *,
    seed: int,
    world: str,
    days: int,
) -> OutfitterConfig:
    config = OutfitterConfig.from_yaml(config_path)
    config.data_dir = runtime
    config.seed = seed
    config.demand_world_id = world
    config.max_days = days
    config.validate()
    return config


def _differential_cell(
    args: tuple[str, str, str, int, str, int, str]
) -> dict[str, Any]:
    config_path, v16_runtime, v17_runtime, seed, world, days, strategy = args
    v16_config = _policy_config(
        config_path, v16_runtime, seed=seed, world=world, days=days
    )
    v17_config = _policy_config(
        config_path, v17_runtime, seed=seed, world=world, days=days
    )
    v16_environment = OutfitterEnvironment(
        v16_config, OutfitterDataset.load(v16_runtime)
    )
    v17_environment = OutfitterEnvironment(
        v17_config, OutfitterDataset.load(v17_runtime)
    )
    reset_match = (
        v16_environment.reset().to_dict() == v17_environment.reset().to_dict()
    )
    v16_result = run_policy(v16_config, strategy)
    v17_result = run_policy(v17_config, strategy)
    return {
        "seed": seed,
        "world": world,
        "days": days,
        "strategy": strategy,
        "reset_state_exact_match": reset_match,
        "scripted_result_exact_match": v16_result == v17_result,
        "v16_result_sha256": _canonical_sha256(v16_result),
        "v17_result_sha256": _canonical_sha256(v17_result),
        "value_change": v17_result["value_change"],
        "bankrupt": v17_result["bankrupt"],
        "reservation_requests": v17_result["reservations"]["requests"],
    }


def _stored_identifier_attempts(run_path: Path) -> list[list[str]]:
    attempts: list[list[str]] = []
    with sqlite3.connect(run_path / "run.sqlite3", timeout=30) as connection:
        for name, raw_arguments in connection.execute(
            "SELECT name,arguments FROM tool_calls ORDER BY rowid"
        ):
            if name != "view_reservation":
                continue
            arguments = json.loads(raw_arguments)
            if arguments.get("view") != "quote_history":
                continue
            identifiers = arguments.get("identifiers")
            if isinstance(identifiers, list) and identifiers:
                attempts.append([str(value) for value in identifiers])
    return attempts


def _encoded_characters(result: Any) -> int:
    return len(ToolExecutor.encode_result(result))


def _trajectory_replay(
    run_dir: str,
    *,
    v17_runtime: str,
) -> dict[str, Any]:
    path, manifest, _summary, config, _dataset, v16_environment, state = (
        _load_run(run_dir)
    )
    v16_tools = ToolExecutor(v16_environment)
    v17_config = OutfitterConfig.from_dict(config.to_dict())
    v17_config.data_dir = v17_runtime
    v17_dataset = OutfitterDataset.load(v17_runtime)
    v17_environment = OutfitterEnvironment(v17_config, v17_dataset)
    v17_tools = ToolExecutor(v17_environment)

    v16_full = v16_tools.execute(
        state,
        "view_reservation",
        {"view": "quote_history", "cursor": 0, "limit": 50},
    )
    v17_recent = v17_tools.execute(
        state, "view_reservation", {"view": "quote_history"}
    )
    stored_identifier_attempts = _stored_identifier_attempts(path)
    identifier_attempts = list(stored_identifier_attempts)
    if not identifier_attempts:
        completed = [
            request
            for request in state.reservation_requests
            if request.status != "pending_quote"
        ]
        if completed:
            newest = max(
                completed, key=lambda row: (row.created_day, row.request_id)
            )
            identifier_attempts = [[newest.request_id]]

    known_identifiers = {
        request.request_id for request in state.reservation_requests
    } | {
        str(request.commitment_id)
        for request in state.reservation_requests
        if request.commitment_id is not None
    }
    exact_replays = []
    for identifiers in identifier_attempts:
        known_at_final_state = all(
            identifier in known_identifiers for identifier in identifiers
        )
        result = v17_tools.execute(
            state,
            "view_reservation",
            {"view": "quote_history", "identifiers": identifiers},
        )
        exact_replays.append(
            {
                "identifiers": identifiers,
                "known_at_final_state": known_at_final_state,
                "ok": result.ok,
                "correct_behavior": (
                    result.ok
                    if known_at_final_state
                    else (
                        not result.ok
                        and "unknown reservation identifiers"
                        in str(result.data.get("error") or "")
                    )
                ),
                "characters": _encoded_characters(result),
                "total_count": (
                    result.data.get("total_count") if result.ok else None
                ),
                "error": result.data.get("error") if not result.ok else None,
            }
        )

    full_characters = _encoded_characters(v16_full)
    successful_exact = [
        row
        for row in exact_replays
        if row["known_at_final_state"] and row["ok"]
    ]
    reductions = [
        round(1.0 - int(row["characters"]) / full_characters, 6)
        for row in successful_exact
        if full_characters
    ]
    return {
        "run_id": manifest["run_id"],
        "model": manifest.get("controller", {}).get("model"),
        "state_day": state.day,
        "reservation_request_count": len(state.reservation_requests),
        "stored_identifier_attempt_count": len(stored_identifier_attempts),
        "v16_full_page": {
            "ok": v16_full.ok,
            "characters": full_characters,
            "returned_count": v16_full.data.get("returned_count"),
            "total_count": v16_full.data.get("total_count"),
        },
        "v17_default_recent_page": {
            "ok": v17_recent.ok,
            "characters": _encoded_characters(v17_recent),
            "returned_count": v17_recent.data.get("returned_count"),
            "total_count": v17_recent.data.get("total_count"),
            "limit": v17_recent.data.get("limit"),
            "sort_order": v17_recent.data.get("sort_order"),
        },
        "exact_replays": exact_replays,
        "successful_exact_replays": len(successful_exact),
        "correct_exact_replay_behaviors": sum(
            row["correct_behavior"] for row in exact_replays
        ),
        "exact_response_reduction_fraction": reductions,
        "minimum_exact_response_reduction_fraction": min(
            reductions, default=None
        ),
    }


def validate(
    *,
    config_path: str,
    v16_runtime: str,
    v17_runtime: str,
    run_dirs: list[str],
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    worlds: tuple[str, ...] = DEFAULT_WORLDS,
    days: int = 28,
    strategy: str = "informed_public",
    workers: int = 4,
) -> dict[str, Any]:
    cells = [
        (config_path, v16_runtime, v17_runtime, seed, world, days, strategy)
        for world in worlds
        for seed in seeds
    ]
    if workers == 1:
        differential = [_differential_cell(cell) for cell in cells]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            differential = list(executor.map(_differential_cell, cells))
    trajectory_replays = [
        _trajectory_replay(run_dir, v17_runtime=v17_runtime)
        for run_dir in run_dirs
    ]
    exact_replays = [
        row
        for replay in trajectory_replays
        for row in replay["exact_replays"]
    ]
    reductions = [
        value
        for replay in trajectory_replays
        for value in replay["exact_response_reduction_fraction"]
    ]
    recent_sizes = [
        replay["v17_default_recent_page"]["characters"]
        for replay in trajectory_replays
    ]
    eligible_single_identifier_reductions = [
        round(
            1.0
            - int(row["characters"])
            / int(replay["v16_full_page"]["characters"]),
            6,
        )
        for replay in trajectory_replays
        if int(replay["v16_full_page"].get("returned_count") or 0) >= 10
        for row in replay["exact_replays"]
        if row["known_at_final_state"]
        and row["ok"]
        and len(row["identifiers"]) == 1
    ]
    result = {
        "format": FORMAT,
        "v16_runtime": str(Path(v16_runtime).resolve()),
        "v17_runtime": str(Path(v17_runtime).resolve()),
        "differential": {
            "cells": differential,
            "cell_count": len(differential),
            "all_reset_states_exact_match": all(
                row["reset_state_exact_match"] for row in differential
            ),
            "all_scripted_results_exact_match": all(
                row["scripted_result_exact_match"] for row in differential
            ),
        },
        "trajectory_replay": {
            "runs": trajectory_replays,
            "run_count": len(trajectory_replays),
            "exact_replay_count": len(exact_replays),
            "successful_exact_replay_count": sum(
                row["known_at_final_state"] and row["ok"]
                for row in exact_replays
            ),
            "unknown_identifier_replay_count": sum(
                not row["known_at_final_state"] for row in exact_replays
            ),
            "correct_exact_replay_behavior_count": sum(
                row["correct_behavior"] for row in exact_replays
            ),
            "minimum_exact_response_reduction_fraction": min(
                reductions, default=None
            ),
            "maximum_v17_default_recent_page_characters": max(
                recent_sizes, default=0
            ),
            "eligible_single_identifier_reduction_count": len(
                eligible_single_identifier_reductions
            ),
            "minimum_eligible_single_identifier_response_reduction_fraction": min(
                eligible_single_identifier_reductions, default=None
            ),
        },
    }
    result["passed"] = bool(
        result["differential"]["all_reset_states_exact_match"]
        and result["differential"]["all_scripted_results_exact_match"]
        and (
            not exact_replays
            or result["trajectory_replay"][
                "correct_exact_replay_behavior_count"
            ]
            == len(exact_replays)
        )
        and (
            not eligible_single_identifier_reductions
            or min(eligible_single_identifier_reductions) >= 0.85
        )
        and max(recent_sizes, default=0) < 50_000
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--v16-runtime", required=True)
    parser.add_argument("--v17-runtime", required=True)
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--days", type=int, default=28)
    parser.add_argument("--strategy", default="informed_public")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = validate(
        config_path=args.config,
        v16_runtime=args.v16_runtime,
        v17_runtime=args.v17_runtime,
        run_dirs=args.run_dir,
        days=args.days,
        strategy=args.strategy,
        workers=args.workers,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
