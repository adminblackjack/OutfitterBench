"""Batch contextual rental-pricing surface with explicit information boundaries.

The runner initializes one environment per world/seed, then reuses it across all
cells.  Public policies may condition only on public context IDs; a separately
reported full-context oracle may also condition on declared hidden cells.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .data_validation.v9_lifecycle_factors import JOINT_WORLDS
from .demand_lab import (
    _apply_scenario,
    _environment,
    _session_from_request,
)
from .jsonio import (
    read_json_object as _read_json,
    read_jsonl as _read_jsonl,
    write_json as _write_json,
)
from .lifecycle_economics import (
    LifecycleEconomicScales,
    cleaning_terms,
    condition_loss_from_severity_uniform,
    damage_event_probability,
    damage_terms,
    maintenance_terms,
    nominal_residual_value,
)


SPEC_FORMAT = "outfitter-v9-context-surface-spec-v1"
RESULT_FORMAT = "outfitter-v9-context-surface-result-v1"
PUBLIC_CONTEXT_FIELDS = frozenset(
    {
        "duration_days",
        "adults",
        "children",
        "units_requested",
        "archetype",
        "visible_party_context",
        "use_case",
        "weather",
        "booking_lead_days",
        "pickup_day",
        "forecast_adversity",
        "forecast_uncertainty",
        "intent_role",
        "preferred_group",
        "size_class",
    }
)
HIDDEN_CONTEXT_FIELDS = frozenset(
    {
        "customer_segment",
        "future_use_expectation",
        "storage_burden",
        "return_burden_multiplier",
        "occasion_importance",
        "forecast_trust",
    }
)


def _positive(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _contexts(
    values: Any,
    *,
    kind: str,
    allowed_fields: frozenset[str],
) -> list[dict[str, Any]]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{kind}_contexts must be a non-empty list")
    result = []
    identifiers = set()
    for raw in values:
        if not isinstance(raw, dict):
            raise ValueError(f"every {kind} context must be an object")
        identifier = str(raw.get("context_id") or "")
        if not identifier or identifier in identifiers:
            raise ValueError(f"every {kind} context needs one unique context_id")
        updates = raw.get("updates")
        if not isinstance(updates, dict):
            raise ValueError(f"{kind} context updates must be an object")
        unknown = sorted(set(updates) - allowed_fields)
        if unknown:
            raise ValueError(
                f"{kind} context contains forbidden fields: {unknown}"
            )
        identifiers.add(identifier)
        result.append(
            {
                "context_id": identifier,
                "weight": _positive(raw.get("weight", 1.0), name="context weight"),
                "updates": dict(updates),
            }
        )
    total = sum(row["weight"] for row in result)
    for row in result:
        row["weight"] /= total
    return result


def _validated_spec(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("format") != SPEC_FORMAT:
        raise ValueError("unsupported context-surface spec")
    sessions = value.get("sessions_per_cell")
    if isinstance(sessions, bool) or not isinstance(sessions, int) or sessions < 1:
        raise ValueError("sessions_per_cell must be a positive integer")
    raw_ratios = value.get("ratios")
    if not isinstance(raw_ratios, list) or not raw_ratios:
        raise ValueError("ratios must be a non-empty list")
    ratios = sorted({_positive(item, name="ratio") for item in raw_ratios})
    if ratios[-1] > 10:
        raise ValueError("context-surface ratios must not exceed 10")
    worlds = [str(item) for item in value.get("worlds") or ["base"]]
    valid_worlds = {row["world_id"] for row in JOINT_WORLDS}
    if not worlds or len(set(worlds)) != len(worlds) or set(worlds) - valid_worlds:
        raise ValueError("context-surface worlds are invalid")
    seeds = value.get("seeds") or [42]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(item, bool) or not isinstance(item, int) for item in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("context-surface seeds are invalid")
    raw_panels = value.get("panels")
    if not isinstance(raw_panels, list) or not raw_panels:
        raise ValueError("context-surface panels must be a non-empty list")
    panels = []
    panel_ids = set()
    for raw in raw_panels:
        if not isinstance(raw, dict):
            raise ValueError("every panel must be an object")
        panel_id = str(raw.get("panel_id") or "")
        target = str(raw.get("target_offer_id") or "")
        base_request = raw.get("base_request")
        if not panel_id or panel_id in panel_ids or not target:
            raise ValueError("every panel needs unique panel_id and target_offer_id")
        if not isinstance(base_request, dict) or not base_request.get("service_key"):
            raise ValueError("every panel needs base_request.service_key")
        public = _contexts(
            raw.get("public_contexts"),
            kind="public",
            allowed_fields=PUBLIC_CONTEXT_FIELDS,
        )
        hidden = _contexts(
            raw.get("hidden_contexts"),
            kind="hidden",
            allowed_fields=HIDDEN_CONTEXT_FIELDS,
        )
        panel_ids.add(panel_id)
        panels.append(
            {
                "panel_id": panel_id,
                "target_offer_id": target,
                "base_request": dict(base_request),
                "public_contexts": public,
                "hidden_contexts": hidden,
            }
        )
    return {
        "format": SPEC_FORMAT,
        "sessions_per_cell": sessions,
        "ratios": ratios,
        "worlds": worlds,
        "seeds": list(seeds),
        "panels": panels,
    }


def _load_lifecycle_artifact(path: Path) -> tuple[dict[str, Any], dict[str, Any], LifecycleEconomicScales]:
    root = path.expanduser().resolve()
    factors = {
        str(row["offer_id"]): row
        for row in _read_jsonl(root / "offer_lifecycle_factors.jsonl")
    }
    worlds_payload = _read_json(root / "worlds.json")
    worlds = {
        str(row["world_id"]): row for row in worlds_payload["worlds"]
    }
    raw_scales = _read_json(root / "candidate_economic_scales.json")
    if raw_scales.get("status") != "candidate_requires_joint_economic_calibration":
        raise ValueError("context surface expects explicitly unreleased lifecycle scales")
    scales = LifecycleEconomicScales(
        **{
            key: value
            for key, value in raw_scales.items()
            if key != "format"
        }
    )
    return factors, worlds, scales


def _request_for_cell(
    panel: dict[str, Any], public: dict[str, Any], hidden: dict[str, Any]
) -> dict[str, Any]:
    return {
        **panel["base_request"],
        **public["updates"],
        **hidden["updates"],
    }


def _contract_cost_terms(
    environment: Any,
    state: Any,
    session: Any,
    *,
    target_offer_id: str,
    factor: dict[str, Any],
    lifecycle_world: dict[str, Any],
    scales: LifecycleEconomicScales,
) -> dict[str, Any]:
    multipliers = lifecycle_world["lifecycle_multipliers"]
    duration = max(1, int(session.duration_days or 1))
    quantity = int(environment._required_resources(session, target_offer_id))
    replacement = float(factor["replacement_cost_cny"])
    projection = factor["world_projections"][lifecycle_world["world_id"]]
    cleaning = cleaning_terms(
        one_day_work_points=float(factor["a_cleaning"]["one_day_work_points"]),
        duration_days=duration,
        world_work_multiplier=float(multipliers["cleaning_work"]),
        scales=scales,
    )
    damage_probability = damage_event_probability(
        hazard_tier=str(factor["d_damage"]["hazard_tier"]),
        duration_days=duration,
        combined_wear=0.0,
        condition_ratio=1.0,
        wear_since_maintenance=0.0,
        world_hazard_multiplier=float(multipliers["damage_hazard"]),
        scales=scales,
    )
    condition_loss = condition_loss_from_severity_uniform(
        severity_tier=str(factor["d_damage"]["severity_tier"]),
        severity_uniform=0.5,
        world_severity_multiplier=float(multipliers["damage_severity"]),
        scales=scales,
    )
    damage = damage_terms(
        replacement_cost=replacement,
        condition_ratio_before=1.0,
        security_deposit=float(state.deposits[target_offer_id]),
        condition_loss=condition_loss,
        scales=scales,
    )
    wear_increment = duration / float(projection["service_life_limit_days"]) + 1.0 / float(
        projection["service_life_limit_cycles"]
    )
    residual_before = nominal_residual_value(
        replacement_cost=replacement,
        condition_ratio=1.0,
        combined_wear=0.0,
        scales=scales,
    )
    residual_after = nominal_residual_value(
        replacement_cost=replacement,
        condition_ratio=1.0,
        combined_wear=wear_increment,
        scales=scales,
    )
    maintenance = maintenance_terms(
        replacement_cost=replacement,
        severity_tier=str(factor["d_damage"]["severity_tier"]),
        scales=scales,
    )
    per_asset = {
        "cleaning_cash": cleaning["cash_cost"],
        "expected_uncovered_damage_cash": round(
            damage_probability * float(damage["uncovered_repair_cost"]), 6
        ),
        "residual_wear_decrement": round(residual_before - residual_after, 6),
        "exposure_attributable_maintenance_cash": round(
            float(maintenance["cash_cost"])
            * wear_increment
            / scales.maintenance_interval_wear,
            6,
        ),
    }
    total_per_asset = sum(per_asset.values())
    return {
        "quantity_assets": quantity,
        "duration_days": duration,
        "capital_at_risk": round(quantity * replacement, 6),
        "per_asset": per_asset,
        "per_asset_total": round(total_per_asset, 6),
        "contract_total": round(quantity * total_per_asset, 6),
        "holding_excluded": True,
        "holding_exclusion_reason": (
            "calendar holding is trajectory-dependent and is not inserted into a "
            "synthetic standard-contract denominator"
        ),
    }


def _target_curve(
    environment: Any,
    opening: Any,
    request: dict[str, Any],
    *,
    target_offer_id: str,
    ratios: list[float],
    sessions: int,
    cost_terms: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    online = environment.offers[target_offer_id].consumer_online_offer
    if not isinstance(online, dict) or float(online.get("unit_price_cny", 0.0)) <= 0:
        raise ValueError("target offer lacks verified exact online unit price")
    online_unit_price = float(online["unit_price_cny"])
    for ratio in ratios:
        state = copy.deepcopy(opening)
        # The fixed-policy axis must be a genuinely fixed daily unit price.
        # A contract/generalized-price ratio would divide by duration and quietly
        # give the fixed policy contextual duration discounts.
        daily_price = round(
            max(0.01, ratio * online_unit_price),
            6,
        )
        state.prices[target_offer_id] = daily_price
        contract_cash = round(
            daily_price
            * int(cost_terms["quantity_assets"])
            * int(cost_terms["duration_days"]),
            6,
        )
        fulfilled = 0
        revenue = 0.0
        channels: Counter[str] = Counter()
        for index in range(sessions):
            session = _session_from_request(
                environment, state, request, sample_index=index
            )
            trace, _views = environment.explain_channel_demand(state, session)
            channels[trace.selected_channel] += 1
            if (
                trace.selected_channel != "store"
                or trace.selected_offer_id != target_offer_id
            ):
                continue
            fulfilled += 1
            selected = next(
                profile
                for profile in trace.profiles
                if profile["profile_id"] == trace.selected_profile_id
            )
            revenue += float(selected["store_cash_price"] or 0.0)
        conversion = fulfilled / sessions
        expected_revenue = revenue / sessions
        expected_contribution = (
            revenue - fulfilled * float(cost_terms["contract_total"])
        ) / sessions
        capital = float(cost_terms["capital_at_risk"])
        contribution_per_contract = contract_cash - float(
            cost_terms["contract_total"]
        )
        rows.append(
            {
                "ratio": ratio,
                "quoted_daily_unit_price": round(daily_price, 6),
                "quoted_contract_cash": contract_cash,
                "sessions": sessions,
                "store_conversion": conversion,
                "channel_shares": {
                    key: value / sessions for key, value in sorted(channels.items())
                },
                "expected_rental_cash_per_request": round(expected_revenue, 8),
                "expected_variable_contribution_per_request": round(
                    expected_contribution, 8
                ),
                "gross_payback_contracts": round(capital / contract_cash, 8),
                "variable_contribution_payback_contracts": (
                    None
                    if contribution_per_contract <= 0
                    else round(capital / contribution_per_contract, 8)
                ),
            }
        )
    return rows


def _best(rows: Iterable[dict[str, Any]], objective: str) -> dict[str, Any]:
    return max(rows, key=lambda row: (float(row[objective]), -float(row["ratio"])))


def _weighted_median(values: list[tuple[float, float]]) -> float | None:
    values = sorted((value, weight) for value, weight in values if weight > 0)
    if not values:
        return None
    total = sum(weight for _, weight in values)
    cumulative = 0.0
    for value, weight in values:
        cumulative += weight
        if cumulative + 1e-15 >= total / 2:
            return value
    return values[-1][0]


def _policy_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    objective = "expected_variable_contribution_per_request"
    ratios = [row["ratio"] for row in cells[0]["curve"]]
    fixed_values = []
    for ratio in ratios:
        fixed_values.append(
            {
                "ratio": ratio,
                objective: sum(
                    float(cell["weight"])
                    * float(next(row for row in cell["curve"] if row["ratio"] == ratio)[objective])
                    for cell in cells
                ),
            }
        )
    fixed = _best(fixed_values, objective)

    public_ids = sorted({str(cell["public_context_id"]) for cell in cells})
    public_policy = []
    public_value = 0.0
    for public_id in public_ids:
        group = [cell for cell in cells if cell["public_context_id"] == public_id]
        group_weight = sum(float(cell["weight"]) for cell in group)
        candidates = []
        for ratio in ratios:
            candidates.append(
                {
                    "ratio": ratio,
                    objective: sum(
                        float(cell["weight"])
                        * float(
                            next(
                                row for row in cell["curve"] if row["ratio"] == ratio
                            )[objective]
                        )
                        for cell in group
                    )
                    / group_weight,
                }
            )
        selected = _best(candidates, objective)
        quoted_prices = {
            round(
                float(
                    next(
                        row
                        for row in cell["curve"]
                        if row["ratio"] == selected["ratio"]
                    )["quoted_daily_unit_price"]
                ),
                6,
            )
            for cell in group
        }
        if len(quoted_prices) != 1:
            raise AssertionError(
                "a public policy price changed across hidden contexts"
            )
        public_value += group_weight * float(selected[objective])
        public_policy.append(
            {
                "public_context_id": public_id,
                "weight": group_weight,
                "ratio": selected["ratio"],
                "quoted_daily_unit_price": quoted_prices.pop(),
                objective: selected[objective],
            }
        )

    full_oracle_value = 0.0
    oracle_policy = []
    for cell in cells:
        selected = _best(cell["curve"], objective)
        full_oracle_value += float(cell["weight"]) * float(selected[objective])
        oracle_policy.append(
            {
                "public_context_id": cell["public_context_id"],
                "hidden_context_id": cell["hidden_context_id"],
                "ratio": selected["ratio"],
                objective: selected[objective],
            }
        )

    fixed_value = float(fixed[objective])
    tolerance = 1e-10
    if public_value + tolerance < fixed_value or full_oracle_value + tolerance < public_value:
        raise AssertionError("information refinement cannot lower an optimized value")

    public_ratio = {
        row["public_context_id"]: float(row["ratio"]) for row in public_policy
    }
    fixed_paybacks = []
    public_paybacks = []
    oracle_paybacks = []
    for cell in cells:
        weight = float(cell["weight"])
        fixed_row = next(
            row for row in cell["curve"] if row["ratio"] == fixed["ratio"]
        )
        public_row = next(
            row
            for row in cell["curve"]
            if row["ratio"] == public_ratio[cell["public_context_id"]]
        )
        oracle_row = _best(cell["curve"], objective)
        for target, row in (
            (fixed_paybacks, fixed_row),
            (public_paybacks, public_row),
            (oracle_paybacks, oracle_row),
        ):
            payback = row["variable_contribution_payback_contracts"]
            if payback is not None:
                target.append((float(payback), weight))

    def relative_gain(value: float, baseline: float) -> float | None:
        if abs(baseline) <= 1e-12:
            return None
        return value / baseline - 1.0

    return {
        "objective": objective,
        "best_fixed": {**fixed, "value": fixed_value},
        "public_context_policy": public_policy,
        "public_context_value": public_value,
        "full_hidden_oracle_policy": oracle_policy,
        "full_hidden_oracle_value": full_oracle_value,
        "public_context_absolute_gain_over_fixed": public_value - fixed_value,
        "hidden_information_absolute_gain_over_public": (
            full_oracle_value - public_value
        ),
        "public_context_gain_over_fixed": relative_gain(public_value, fixed_value),
        "hidden_information_gain_over_public": relative_gain(
            full_oracle_value, public_value
        ),
        "variable_contribution_payback_weighted_median": {
            "fixed": _weighted_median(fixed_paybacks),
            "public_context": _weighted_median(public_paybacks),
            "full_hidden_oracle": _weighted_median(oracle_paybacks),
        },
        "information_value_ordering_passed": True,
    }


def build_context_surface(
    data_dir: Path,
    lifecycle_artifact: Path,
    spec_value: dict[str, Any],
    *,
    rental_contract_cash_salience: float | None = None,
) -> dict[str, Any]:
    data_dir = data_dir.expanduser().resolve()
    spec = _validated_spec(spec_value)
    factors, lifecycle_worlds, scales = _load_lifecycle_artifact(
        lifecycle_artifact
    )
    runs = []
    environment_initializations = 0
    for world_id in spec["worlds"]:
        for seed in spec["seeds"]:
            environment, base_opening = _environment(
                data_dir, world=world_id, seed=seed
            )
            environment_initializations += 1
            if environment.channel_demand_model is None:
                raise ValueError("context surface requires exact-offer channel demand")
            if rental_contract_cash_salience is not None:
                salience = _positive(
                    rental_contract_cash_salience,
                    name="rental_contract_cash_salience",
                )
                environment.channel_demand_model = replace(
                    environment.channel_demand_model,
                    rental_contract_cash_salience=salience,
                )
            for panel in spec["panels"]:
                target = panel["target_offer_id"]
                if target not in environment.offers or target not in factors:
                    raise ValueError(f"panel target lacks offer/factor: {target}")
                offer = environment.offers[target]
                if offer.asset_kind != "durable":
                    raise ValueError("context surface target must be durable")
                if panel["base_request"]["service_key"] not in offer.fulfills_service_keys:
                    raise ValueError("panel request service does not match target offer")

                service_offer_ids = [
                    row.offer_id
                    for row in environment.by_service[
                        str(panel["base_request"]["service_key"])
                    ]
                ]
                opening = copy.deepcopy(base_opening)
                _apply_scenario(
                    environment,
                    opening,
                    {
                        "format": "outfitter-demand-lab-scenario-v1",
                        "request": panel["base_request"],
                        "available_units": {
                            offer_id: 12 if offer_id == target else 0
                            for offer_id in service_offer_ids
                        },
                        "prices": {},
                        "deposits": {},
                    },
                )
                cells = []
                for public in panel["public_contexts"]:
                    for hidden in panel["hidden_contexts"]:
                        request = _request_for_cell(panel, public, hidden)
                        probe_state = copy.deepcopy(opening)
                        probe = _session_from_request(
                            environment, probe_state, request, sample_index=0
                        )
                        target_view = next(
                            row
                            for row in environment._channel_offer_views(
                                probe_state, probe
                            )
                            if row.offer_id == target
                        )
                        if not target_view.hard_compatible:
                            raise ValueError(
                                f"target is incompatible in context "
                                f"{public['context_id']}/{hidden['context_id']}: "
                                f"{target_view.incompatibility_reasons}"
                            )
                        costs = _contract_cost_terms(
                            environment,
                            probe_state,
                            probe,
                            target_offer_id=target,
                            factor=factors[target],
                            lifecycle_world=lifecycle_worlds[world_id],
                            scales=scales,
                        )
                        cells.append(
                            {
                                "public_context_id": public["context_id"],
                                "hidden_context_id": hidden["context_id"],
                                "public_updates": public["updates"],
                                "hidden_updates": hidden["updates"],
                                "weight": public["weight"] * hidden["weight"],
                                "contract_variable_cost": costs,
                                "curve": _target_curve(
                                    environment,
                                    opening,
                                    request,
                                    target_offer_id=target,
                                    ratios=spec["ratios"],
                                    sessions=spec["sessions_per_cell"],
                                    cost_terms=costs,
                                ),
                            }
                        )
                runs.append(
                    {
                        "world_id": world_id,
                        "seed": seed,
                        "panel_id": panel["panel_id"],
                        "target_offer_id": target,
                        "cells": cells,
                        "policy_comparison": _policy_summary(cells),
                    }
                )
    return {
        "format": RESULT_FORMAT,
        "data_dir": str(data_dir),
        "lifecycle_artifact": str(lifecycle_artifact.expanduser().resolve()),
        "claim_boundary": (
            "candidate benchmark contribution/payback diagnostics; hidden oracle "
            "is an information upper bound and is never an Agent policy"
        ),
        "spec": spec,
        "price_axis": "daily_unit_cash_to_exact_online_purchase_unit_price",
        "fixed_policy_semantics": (
            "one daily unit price per exact offer across every public and hidden "
            "contract context"
        ),
        "methodology_revision": (
            "contract/generalized-price ratio rejected because it changed daily "
            "price mechanically with duration"
        ),
        "environment_initializations": environment_initializations,
        "full_cells": sum(len(row["cells"]) for row in runs),
        "curve_evaluations": sum(
            len(cell["curve"]) for row in runs for cell in row["cells"]
        ),
        "common_random_requests_across_ratios": True,
        "rental_contract_cash_salience": (
            1.0
            if rental_contract_cash_salience is None
            else float(rental_contract_cash_salience)
        ),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--lifecycle-artifact", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--sessions-per-cell", type=int)
    parser.add_argument("--rental-contract-cash-salience", type=float)
    args = parser.parse_args()
    spec_value = _read_json(args.spec.expanduser().resolve())
    if args.worlds is not None:
        spec_value["worlds"] = list(args.worlds)
    if args.seeds is not None:
        spec_value["seeds"] = list(args.seeds)
    if args.sessions_per_cell is not None:
        spec_value["sessions_per_cell"] = int(args.sessions_per_cell)
    result = build_context_surface(
        args.data_dir,
        args.lifecycle_artifact,
        spec_value,
        rental_contract_cash_salience=args.rental_contract_cash_salience,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, result)
    print(
        json.dumps(
            {
                "format": result["format"],
                "output": str(output),
                "runs": len(result["runs"]),
                "full_cells": result["full_cells"],
                "curve_evaluations": result["curve_evaluations"],
                "environment_initializations": result[
                    "environment_initializations"
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
