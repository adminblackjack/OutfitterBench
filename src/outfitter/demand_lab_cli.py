"""CLI and report rendering for :mod:`outfitter.demand_lab`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .demand_lab import (
    LAB_FORMAT,
    _canonical_hash,
    _prepare_data_dir,
    _read_json,
    _safe_output_dir,
    curve,
    explain,
    invariance,
    replay,
)


def _write_result(
    result: dict[str, Any],
    *,
    output_dir: Path | None,
    data_dir: Path,
    mode: str,
) -> Path:
    target = (
        Path(f"/tmp/outfitter-demand-lab/{mode}-{_canonical_hash(result)[:16]}")
        if output_dir is None
        else output_dir
    )
    target = _safe_output_dir(target, data_dir=data_dir)
    path = target / "result.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target / "report.md").write_text(
        _markdown_result(result, mode=mode), encoding="utf-8"
    )
    return path

def _markdown_result(result: dict[str, Any], *, mode: str) -> str:
    lines = [f"# Outfitter Demand Lab: {mode}", ""]
    lines.append(f"- Runtime: `{result.get('data_dir', 'n/a')}`")
    lines.append(f"- World: `{result.get('world_id', 'paired')}`")
    if mode == "explain":
        decision = dict(result["decision"])
        if result.get("model_kind") == "exact_offer_channel":
            lines.extend(
                [
                    f"- Internal selected channel: `{decision['selected_channel']}`",
                    f"- Selected offer: `{decision.get('selected_offer_id')}`",
                    "",
                    "| Offer | compatible | available | store cash | online cash |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for row in result["offer_views"]:
                lines.append(
                    f"| `{row['offer_id']}` | {row['hard_compatible']} | "
                    f"{row['available_units']} | "
                    f"{row.get('store_cash_price', 'n/a')} | "
                    f"{row['online_cash_price']} |"
                )
        else:
            lines.extend(
                [
                    f"- Bought: `{decision['bought']}`",
                    f"- Entry probability: `{decision['entry_probability']:.6f}`",
                    f"- Selected offer: `{decision.get('selected_offer_id')}`",
                    f"- Reason: `{decision['reason']}`",
                    "",
                    "| Offer | compatible | available | price/reference |",
                    "|---|---:|---:|---:|",
                ]
            )
            for row in result["offer_views"]:
                ratio = row.get("price_ratio")
                lines.append(
                    f"| `{row['offer_id']}` | {row['hard_compatible']} | "
                    f"{row['available_units']} | "
                    f"{'n/a' if ratio is None else f'{ratio:.4f}'} |"
                )
    elif mode == "curve":
        channel = result.get("model_kind") == "exact_offer_channel"
        gross = channel and result.get("gross_contribution_optimum") is not None
        lines.extend(
            [
                "",
                (
                    "| price ratio | conversion | revenue/request | gross contribution/request |"
                    if gross
                    else "| price ratio | conversion | revenue/request |"
                ),
                "|---:|---:|---:|---:|" if gross else "|---:|---:|---:|",
            ]
        )
        for row in result["standard_checkpoints"]:
            line = (
                f"| {row['store_to_online_ratio' if channel else 'price_to_reference_ratio']:.2f} | "
                f"{row['store_conversion' if channel else 'conversion']:.6f} | "
                f"{row['expected_store_revenue_per_request' if channel else 'expected_revenue_per_request']:.6f}"
            )
            if gross:
                line += f" | {row['expected_store_gross_contribution_per_request']:.6f}"
            lines.append(line + " |")
        optimum = result["revenue_optimum"]
        lines.extend(
            [
                "",
                f"Revenue optimum on scanned grid: "
                f"`{optimum['store_to_online_ratio' if channel else 'price_to_reference_ratio']:.4f}x`.",
            ]
        )
        if gross:
            gross_optimum = result["gross_contribution_optimum"]
            lines.append(
                "Gross-contribution optimum on scanned grid: "
                f"`{gross_optimum['store_to_online_ratio']:.4f}x`."
            )
    elif mode == "invariance":
        lines.extend(
            [
                f"- Passed: `{result['passed']}`",
                "",
                "| Check | passed | entry-probability delta |",
                "|---|---:|---:|",
            ]
        )
        for row in result["tests"]:
            lines.append(
                f"| `{row['name']}` | {row['passed']} | "
                f"{row['entry_probability_delta']:.3g} |"
            )
    elif mode == "replay":
        lines.extend(
            [
                f"- Random tape hash: `{result['tape_hash']}`",
                "",
                "| World | conversion | revenue/request |",
                "|---|---:|---:|",
            ]
        )
        for world, row in result["aggregates"].items():
            lines.append(
                f"| `{world}` | {row['conversion']:.6f} | "
                f"{row['expected_revenue_per_request']:.6f} |"
            )
    lines.append("")
    return "\n".join(lines)

def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--world", default="base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-reference", action="store_true")
    parser.add_argument("--output-dir", type=Path)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m outfitter.demand_lab")
    commands = parser.add_subparsers(dest="mode", required=True)
    explain_parser = commands.add_parser("explain")
    _common(explain_parser)
    explain_parser.add_argument("--scenario", type=Path, required=True)
    curve_parser = commands.add_parser("curve")
    _common(curve_parser)
    curve_parser.add_argument("--scenario", type=Path, required=True)
    curve_parser.add_argument("--sessions", type=int, default=4096)
    curve_parser.add_argument("--target-offer")
    curve_parser.add_argument("--grid-step", type=float, default=0.025)
    curve_parser.add_argument("--max-ratio", type=float)
    invariant_parser = commands.add_parser("invariance")
    _common(invariant_parser)
    invariant_parser.add_argument("--scenario", type=Path, required=True)
    replay_parser = commands.add_parser("replay")
    _common(replay_parser)
    replay_parser.add_argument(
        "--worlds",
        nargs="+",
        default=["control", "conservative", "base", "convenience-heavy"],
    )
    replay_parser.add_argument("--sessions", type=int, default=1024)
    replay_parser.add_argument("--tape", type=Path)
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.data_dir.expanduser().resolve()
    data_dir = _prepare_data_dir(
        source, synthetic_reference=bool(args.synthetic_reference)
    )
    if args.mode == "explain":
        result = explain(
            data_dir,
            _read_json(args.scenario),
            world=args.world,
            seed=args.seed,
        )
    elif args.mode == "curve":
        if (
            args.sessions <= 0
            or not 0 < args.grid_step <= 0.5
            or args.max_ratio is not None
            and not 0.5 <= args.max_ratio <= 100.0
        ):
            raise ValueError("curve sessions/grid-step are invalid")
        result = curve(
            data_dir,
            _read_json(args.scenario),
            world=args.world,
            seed=args.seed,
            sessions=args.sessions,
            target_offer=args.target_offer,
            grid_step=args.grid_step,
            max_ratio=args.max_ratio,
        )
    elif args.mode == "invariance":
        result = invariance(
            data_dir,
            _read_json(args.scenario),
            world=args.world,
            seed=args.seed,
        )
    else:
        tape = None
        if args.tape is not None:
            tape = [
                json.loads(line)
                for line in args.tape.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        result = replay(
            data_dir,
            worlds=list(args.worlds),
            seed=args.seed,
            sessions=args.sessions,
            tape_rows=tape,
        )
    path = _write_result(
        result,
        output_dir=args.output_dir,
        data_dir=data_dir,
        mode=args.mode,
    )
    print(
        json.dumps(
            {
                "format": LAB_FORMAT,
                "mode": args.mode,
                "result": str(path),
                "passed": result.get("passed"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if result.get("passed") is False else 0
