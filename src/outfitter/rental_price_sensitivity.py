"""Catalog-wide rental cash-salience and payback sensitivity audit."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .context_surface import build_context_surface
from .data import OutfitterDataset
from .data_validation.v9_lifecycle_factors import STANDARD_DURATION_DAYS
from .jsonio import read_json_object, read_jsonl, write_json


RESULT_FORMAT = "outfitter-rental-price-sensitivity-v2-viability-aware"


def _weighted_median(values: list[tuple[float, float]]) -> float | None:
    rows = sorted((float(value), float(weight)) for value, weight in values if weight > 0)
    if not rows:
        return None
    threshold = sum(weight for _value, weight in rows) / 2.0
    cumulative = 0.0
    for value, weight in rows:
        cumulative += weight
        if cumulative + 1e-12 >= threshold:
            return value
    return rows[-1][0]


def _panel_request(offer: Any, index: int) -> dict[str, Any]:
    service_key = str(offer.fulfills_service_keys[0])
    archetype = (
        "picnic"
        if service_key in {"rental:camp_chair", "rental:camp_table"}
        else "overnight"
    )
    child_only = any(
        token in f"{offer.name} {offer.source_variant_name}"
        for token in ("儿童", "小童")
    ) or (
        offer.category_id == "sleeping_bag"
        and str(offer.decision_attributes.get("size_class", "")) == "child"
    )
    return {
        "adults": 0 if child_only else 1,
        "children": 1 if child_only else 0,
        "archetype": archetype,
        "duration_days": int(STANDARD_DURATION_DAYS[offer.service_line]),
        "event_address": [1, 9, index, 0],
        "preferred_group": str(offer.choice_group),
        "service_key": service_key,
        "size_class": "child" if child_only else "adult",
        "units_requested": 1,
        "use_case": "general",
        "intent_role": "required",
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    viable = [
        row
        for row in rows
        if bool(row.get("positive_contribution_price_exists_in_scan"))
    ]
    payback_within_capacity = [
        row
        for row in viable
        if bool(
            row.get("variable_payback_within_effective_contract_capacity")
        )
    ]
    public_positive = [
        row
        for row in rows
        if float(
            row.get(
                "public_context_expected_variable_contribution_per_request",
                0.0,
            )
        )
        > 1e-12
    ]
    oracle_positive = [
        row
        for row in rows
        if float(
            row.get(
                "full_hidden_oracle_expected_variable_contribution_per_request",
                0.0,
            )
        )
        > 1e-12
    ]
    public_payback_within_capacity = [
        row
        for row in rows
        if bool(
            row.get(
                "public_context_variable_payback_within_effective_contract_capacity"
            )
        )
    ]
    oracle_payback_within_capacity = [
        row
        for row in rows
        if bool(
            row.get(
                "full_hidden_oracle_variable_payback_within_effective_contract_capacity"
            )
        )
    ]
    variable = [
        float(row["variable_contribution_payback_contracts"])
        for row in viable
        if row["variable_contribution_payback_contracts"] is not None
    ]
    gross = [
        float(row["gross_payback_contracts"])
        for row in viable
        if row["gross_payback_contracts"] is not None
    ]
    ratios = [float(row["best_fixed_ratio"]) for row in viable]
    return {
        "offers": len(rows),
        "positive_contribution_offers_in_scan": len(viable),
        "no_positive_contribution_price_in_scan_offers": (
            len(rows) - len(viable)
        ),
        "positive_contribution_offer_share_in_scan": round(
            len(viable) / max(1, len(rows)), 6
        ),
        "variable_payback_within_effective_contract_capacity_offers": len(
            payback_within_capacity
        ),
        "variable_payback_within_capacity_share_of_positive": round(
            len(payback_within_capacity) / max(1, len(viable)), 6
        ),
        "variable_payback_within_capacity_share_of_all": round(
            len(payback_within_capacity) / max(1, len(rows)), 6
        ),
        "public_context_positive_contribution_offers_in_scan": len(
            public_positive
        ),
        "full_hidden_oracle_positive_contribution_offers_in_scan": len(
            oracle_positive
        ),
        "public_context_payback_within_effective_contract_capacity_offers": len(
            public_payback_within_capacity
        ),
        "full_hidden_oracle_payback_within_effective_contract_capacity_offers": len(
            oracle_payback_within_capacity
        ),
        "gross_payback_median": (
            None if not gross else round(statistics.median(gross), 6)
        ),
        "variable_contribution_payback_median": (
            None if not variable else round(statistics.median(variable), 6)
        ),
        "variable_contribution_payback_minimum": (
            None if not variable else round(min(variable), 6)
        ),
        "variable_contribution_payback_maximum": (
            None if not variable else round(max(variable), 6)
        ),
        "share_variable_payback_5_to_12": round(
            sum(5.0 <= value <= 12.0 for value in variable)
            / max(1, len(variable)),
            6,
        ),
        "share_all_offers_with_variable_payback_5_to_12": round(
            sum(5.0 <= value <= 12.0 for value in variable)
            / max(1, len(rows)),
            6,
        ),
        "best_fixed_ratio_median": (
            None if not ratios else round(statistics.median(ratios), 6)
        ),
        "viable_best_ratio_at_lower_grid_boundary": sum(
            bool(row.get("best_ratio_at_lower_grid_boundary"))
            for row in viable
        ),
        "viable_best_ratio_at_upper_grid_boundary": sum(
            bool(row.get("best_ratio_at_upper_grid_boundary"))
            for row in viable
        ),
    }


def build_rental_price_sensitivity(
    *,
    data_dir: Path,
    lifecycle_artifact: Path,
    template_spec: Path,
    saliences: list[float],
    worlds: list[str],
    seeds: list[int],
    sessions_per_cell: int,
    ratios: list[float] | None = None,
) -> dict[str, Any]:
    data_dir = data_dir.expanduser().resolve()
    lifecycle_artifact = lifecycle_artifact.expanduser().resolve()
    dataset = OutfitterDataset.load(data_dir)
    template = read_json_object(template_spec)
    template_panel = dict(template["panels"][0])
    public_contexts = [dict(template_panel["public_contexts"][0])]
    hidden_contexts = [dict(row) for row in template_panel["hidden_contexts"]]
    durable = sorted(
        (offer for offer in dataset.offers if offer.asset_kind == "durable"),
        key=lambda offer: offer.offer_id,
    )
    panels = [
        {
            "panel_id": f"catalog-{offer.offer_id}",
            "target_offer_id": offer.offer_id,
            "base_request": _panel_request(offer, index),
            "public_contexts": [
                {
                    **row,
                    "updates": {
                        **dict(row["updates"]),
                        "duration_days": int(
                            STANDARD_DURATION_DAYS[offer.service_line]
                        ),
                    },
                }
                for row in public_contexts
            ],
            "hidden_contexts": hidden_contexts,
        }
        for index, offer in enumerate(durable)
    ]
    ratio_grid = (
        list(map(float, ratios))
        if ratios is not None
        else list(map(float, template["ratios"]))
    )
    if not ratio_grid or any(value <= 0 for value in ratio_grid):
        raise ValueError("ratios must contain positive values")
    if len(set(ratio_grid)) != len(ratio_grid):
        raise ValueError("ratios must be unique")
    ratio_grid.sort()
    spec = {
        "format": template["format"],
        "sessions_per_cell": int(sessions_per_cell),
        "ratios": ratio_grid,
        "worlds": list(worlds),
        "seeds": list(seeds),
        "panels": panels,
    }
    offer_map = {offer.offer_id: offer for offer in durable}
    lifecycle_by_offer = {
        str(row["offer_id"]): row
        for row in read_jsonl(lifecycle_artifact / "offer_lifecycle_factors.jsonl")
    }
    if set(lifecycle_by_offer) != set(offer_map):
        raise ValueError("lifecycle factor ledger does not cover durable offers")
    rows: list[dict[str, Any]] = []
    for salience in saliences:
        surface = build_context_surface(
            data_dir,
            lifecycle_artifact,
            spec,
            rental_contract_cash_salience=salience,
        )
        for run in surface["runs"]:
            comparison = run["policy_comparison"]
            diagnostic_fixed_ratio = float(comparison["best_fixed"]["ratio"])
            fixed_value = float(comparison["best_fixed"]["value"])
            public_value = float(comparison["public_context_value"])
            oracle_value = float(comparison["full_hidden_oracle_value"])
            viable = fixed_value > 1e-12
            ratio_grid = sorted(
                {float(row["ratio"]) for row in run["cells"][0]["curve"]}
            )
            gross_pairs: list[tuple[float, float]] = []
            if viable:
                for cell in run["cells"]:
                    curve_row = next(
                        row
                        for row in cell["curve"]
                        if float(row["ratio"]) == diagnostic_fixed_ratio
                    )
                    gross_pairs.append(
                        (
                            float(curve_row["gross_payback_contracts"]),
                            float(cell["weight"]),
                        )
                    )
            gross_payback = _weighted_median(gross_pairs)
            offer = offer_map[str(run["target_offer_id"])]
            factor = lifecycle_by_offer[offer.offer_id]
            effective_contract_capacity = float(
                factor["world_projections"][str(run["world_id"])][
                    "effective_standard_contracts"
                ]
            )
            variable_payback = (
                comparison[
                    "variable_contribution_payback_weighted_median"
                ]["fixed"]
                if viable
                else None
            )
            public_payback = (
                comparison["variable_contribution_payback_weighted_median"][
                    "public_context"
                ]
                if public_value > 1e-12
                else None
            )
            oracle_payback = (
                comparison["variable_contribution_payback_weighted_median"][
                    "full_hidden_oracle"
                ]
                if oracle_value > 1e-12
                else None
            )
            lower_boundary = viable and diagnostic_fixed_ratio == ratio_grid[0]
            upper_boundary = viable and diagnostic_fixed_ratio == ratio_grid[-1]
            payback_within_capacity = (
                variable_payback is not None
                and float(variable_payback) <= effective_contract_capacity
            )
            rows.append(
                {
                    "cash_salience": float(salience),
                    "world_id": str(run["world_id"]),
                    "seed": int(run["seed"]),
                    "offer_id": offer.offer_id,
                    "service_line": offer.service_line,
                    "choice_group": offer.choice_group,
                    "standard_contract_duration_days": int(
                        STANDARD_DURATION_DAYS[offer.service_line]
                    ),
                    "supplier_unit_cost": float(offer.unit_cost),
                    "pricing_status": (
                        "no_positive_contribution_price_in_scan"
                        if not viable
                        else (
                            "positive_contribution_price_boundary_censored"
                            if lower_boundary or upper_boundary
                            else "positive_contribution_price_interior"
                        )
                    ),
                    "positive_contribution_price_exists_in_scan": viable,
                    "best_fixed_ratio": (
                        diagnostic_fixed_ratio if viable else None
                    ),
                    "zero_contribution_tie_break_ratio": (
                        None if viable else diagnostic_fixed_ratio
                    ),
                    "scan_ratio_minimum": ratio_grid[0],
                    "scan_ratio_maximum": ratio_grid[-1],
                    "best_ratio_at_lower_grid_boundary": lower_boundary,
                    "best_ratio_at_upper_grid_boundary": upper_boundary,
                    "gross_payback_contracts": (
                        round(float(gross_payback), 6)
                        if viable and gross_payback is not None
                        else None
                    ),
                    "variable_contribution_payback_contracts": (
                        variable_payback
                    ),
                    "public_context_variable_contribution_payback_contracts": (
                        public_payback
                    ),
                    "full_hidden_oracle_variable_contribution_payback_contracts": (
                        oracle_payback
                    ),
                    "effective_standard_contract_capacity": round(
                        effective_contract_capacity, 6
                    ),
                    "variable_payback_within_effective_contract_capacity": (
                        payback_within_capacity
                    ),
                    "public_context_variable_payback_within_effective_contract_capacity": (
                        public_payback is not None
                        and float(public_payback) <= effective_contract_capacity
                    ),
                    "full_hidden_oracle_variable_payback_within_effective_contract_capacity": (
                        oracle_payback is not None
                        and float(oracle_payback) <= effective_contract_capacity
                    ),
                    "effective_contract_capacity_headroom": (
                        round(
                            effective_contract_capacity
                            - float(variable_payback),
                            6,
                        )
                        if variable_payback is not None
                        else None
                    ),
                    "fixed_expected_variable_contribution_per_request": round(
                        fixed_value, 8
                    ),
                    "public_context_expected_variable_contribution_per_request": round(
                        public_value, 8
                    ),
                    "full_hidden_oracle_expected_variable_contribution_per_request": round(
                        oracle_value, 8
                    ),
                    "public_context_gain_over_fixed": comparison[
                        "public_context_gain_over_fixed"
                    ],
                    "public_context_absolute_gain_over_fixed": round(
                        float(
                            comparison[
                                "public_context_absolute_gain_over_fixed"
                            ]
                        ),
                        8,
                    ),
                    "hidden_information_gain_over_public": comparison[
                        "hidden_information_gain_over_public"
                    ],
                    "hidden_information_absolute_gain_over_public": round(
                        float(
                            comparison[
                                "hidden_information_absolute_gain_over_public"
                            ]
                        ),
                        8,
                    ),
                }
            )
    grouped: defaultdict[tuple[float, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["cash_salience"], row["world_id"], row["seed"])].append(row)
    summaries = []
    for key, group in sorted(grouped.items()):
        by_service: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_service[str(row["service_line"])].append(row)
        summaries.append(
            {
                "cash_salience": key[0],
                "world_id": key[1],
                "seed": key[2],
                "all_durable_offers": _summary(group),
                "by_service_line": {
                    service: _summary(service_rows)
                    for service, service_rows in sorted(by_service.items())
                },
            }
        )
    return {
        "format": RESULT_FORMAT,
        "claim_boundary": (
            "benchmark price-sensitivity audit; payback is an outcome diagnostic, "
            "not observed market truth and not an input to customer WTP or pricing; "
            "payback within effective contract capacity is necessary but not "
            "sufficient after utilization, holding, maintenance, and damage"
        ),
        "data_dir": str(data_dir.expanduser().resolve()),
        "lifecycle_artifact": str(lifecycle_artifact.expanduser().resolve()),
        "cash_saliences": list(map(float, saliences)),
        "worlds": list(worlds),
        "seeds": list(map(int, seeds)),
        "sessions_per_cell": int(sessions_per_cell),
        "ratios": ratio_grid,
        "durable_offers": len(durable),
        "rows": rows,
        "summaries": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--lifecycle-artifact", type=Path, required=True)
    parser.add_argument("--template-spec", type=Path, required=True)
    parser.add_argument("--cash-saliences", nargs="+", type=float, required=True)
    parser.add_argument("--worlds", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--sessions-per-cell", type=int, default=32)
    parser.add_argument("--ratios", nargs="+", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_rental_price_sensitivity(
        data_dir=args.data_dir,
        lifecycle_artifact=args.lifecycle_artifact,
        template_spec=args.template_spec,
        saliences=args.cash_saliences,
        worlds=args.worlds,
        seeds=args.seeds,
        sessions_per_cell=args.sessions_per_cell,
        ratios=args.ratios,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(
        json.dumps(
            {
                "format": result["format"],
                "output": str(output),
                "rows": len(result["rows"]),
                "durable_offers": result["durable_offers"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
