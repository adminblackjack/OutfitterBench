"""Demand-model laboratory: explain, curve, invariance, and paired replay."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

from .config import OutfitterConfig
from .channel_demand import (
    ChannelEvaluationInput,
    ChannelOfferView,
    evaluate_channel_demand,
    online_rationality_envelope,
)
from .data import OutfitterDataset
from .environment import ChoiceSession, OutfitterEnvironment
from .jsonio import read_json_object as _read_json
from .models import RentalAsset
from .reference_demand import (
    ReferenceEvaluationInput,
    ReferenceOfferView,
    evaluate_reference_demand,
)
from .reference_shadow import (
    SYNTHETIC_REFERENCE_BUILDER_VERSION,
    build_synthetic_shadow_runtime,
)


LAB_FORMAT = "outfitter-demand-lab-v1"
SCENARIO_FORMAT = "outfitter-demand-lab-scenario-v1"
STANDARD_RATIOS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
CHANNEL_RENTAL_RATIOS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
CHANNEL_RETAIL_MAX_RATIO = 6.0
ARTIFACT_ROOT = (
    Path(__file__).resolve().parent / "data_validation" / "artifacts"
).resolve()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _safe_output_dir(path: Path, *, data_dir: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == data_dir or _is_relative_to(resolved, data_dir):
        raise ValueError("demand lab must not write inside the input runtime")
    in_tmp = _is_relative_to(resolved, Path("/tmp"))
    in_shadow_artifact = _is_relative_to(resolved, ARTIFACT_ROOT) and any(
        "shadow" in part.lower() for part in resolved.parts
    )
    if not (in_tmp or in_shadow_artifact):
        raise ValueError("demand lab output must be under /tmp or a shadow artifact")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _prepare_data_dir(source: Path, *, synthetic_reference: bool) -> Path:
    source = source.expanduser().resolve()
    if (source / "channel_demand_model.json").is_file():
        return source
    if (source / "reference_transaction_model.json").is_file():
        return source
    if not synthetic_reference:
        raise ValueError(
            "runtime lacks reference_transaction_model.json; use --synthetic-reference "
            "for mechanism-only diagnostics"
        )
    digest = hashlib.sha256(
        (source / "metadata.json").read_bytes()
        + str(source).encode("utf-8")
        + SYNTHETIC_REFERENCE_BUILDER_VERSION.encode("utf-8")
    ).hexdigest()[:16]
    shadow = Path(f"/tmp/outfitter-demand-lab/synthetic-runtime-{digest}")
    if not (shadow / "reference_transaction_model.json").is_file():
        build_synthetic_shadow_runtime(source, shadow)
    return shadow.resolve()


def _environment(
    data_dir: Path,
    *,
    world: str,
    seed: int,
) -> tuple[OutfitterEnvironment, Any]:
    dataset = OutfitterDataset.load(data_dir)
    config = OutfitterConfig(
        data_dir=str(data_dir),
        max_days=28,
        weather_start_day=91,
        seed=seed,
        initial_cash=6_000,
        demand_world_id=world,
        charge_model_usage_to_cash=False,
        daily_fee=float(dataset.metadata.get("formal_daily_fee") or 60.0),
    )
    environment = OutfitterEnvironment(config, dataset)
    if (
        environment.reference_demand_model is None
        and environment.channel_demand_model is None
    ):
        raise ValueError("demand lab requires a channel or reference demand model")
    return environment, environment.reset()


def _default_request(service_key: str, group: str) -> dict[str, Any]:
    overnight = service_key in {
        "rental:tent",
        "rental:sleeping_bag",
        "rental:sleeping_pad",
        "rental:camp_lantern",
        "rental:camp_cookware",
    }
    picnic = service_key in {"rental:camp_chair", "rental:camp_table"}
    adults = 2 if overnight else 3 if picnic else 1
    return {
        "service_key": service_key,
        "units_requested": (
            adults
            if service_key
            in {
                "rental:sleeping_bag",
                "rental:sleeping_pad",
                "rental:camp_chair",
            }
            else 1
        ),
        "duration_days": 2 if overnight else 1 if picnic else None,
        "adults": adults,
        "children": 0,
        "size_class": "adult" if service_key == "retail:rain_gear" else "adult_regular" if service_key == "rental:sleeping_bag" else "n/a",
        "archetype": "overnight" if overnight else "picnic" if picnic else "supply",
        "preferred_group": group,
        "use_case": "cook_meal" if service_key == "rental:camp_cookware" else "general",
        "event_address": [1, 9, 0, 0],
    }


def _scenario(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("format", SCENARIO_FORMAT) != SCENARIO_FORMAT:
        raise ValueError("unsupported demand-lab scenario format")
    request = value.get("request")
    if not isinstance(request, dict) or not request.get("service_key"):
        raise ValueError("scenario requires request.service_key")
    return {
        "format": SCENARIO_FORMAT,
        "request": dict(request),
        "available_units": {
            str(key): int(item)
            for key, item in dict(value.get("available_units") or {}).items()
        },
        "prices": {
            str(key): float(item)
            for key, item in dict(value.get("prices") or {}).items()
        },
        "deposits": {
            str(key): float(item)
            for key, item in dict(value.get("deposits") or {}).items()
        },
    }


def _session_from_request(
    environment: OutfitterEnvironment,
    state: Any,
    request: dict[str, Any],
    *,
    sample_index: int = 0,
) -> ChoiceSession:
    raw_address = tuple(map(int, request.get("event_address", [1, 9, 0, 0])))
    if len(raw_address) != 4:
        raise ValueError("request.event_address must contain four integers")
    address = (raw_address[0], raw_address[1], raw_address[2] + sample_index)
    service_key = str(request["service_key"])
    archetype = str(request.get("archetype", "supply"))
    inferred_required = (
        archetype == "supply"
        or service_key
        in {
            "rental:tent",
            "rental:sleeping_bag",
            "rental:sleeping_pad",
        }
        and archetype in {"overnight", "lightweight"}
        or service_key in {"rental:camp_table", "rental:camp_chair"}
        and archetype == "picnic"
    )
    session = environment._make_choice_session(
        state,
        service_key=service_key,
        units_requested=max(1, int(request.get("units_requested", 1))),
        duration_days=(
            None
            if request.get("duration_days") is None
            else max(1, int(request["duration_days"]))
        ),
        adults=max(0, int(request.get("adults", 1))),
        children=max(0, int(request.get("children", 0))),
        size_class=str(request.get("size_class", "n/a")),
        archetype=archetype,
        weather=str(request.get("weather") or environment._weather(raw_address[0])),
        request_index=raw_address[3],
        event_address=address,
        intent_role=str(
            request.get("intent_role")
            or ("required" if inferred_required else "optional")
        ),
    )
    updates = {}
    for name in (
        "preferred_group",
        "use_case",
        "customer_segment",
        "visible_party_context",
        "portability_sensitive",
        "planned_purchase",
        "intent_role",
        "need_horizon_days",
        "occasion_importance",
        "booking_lead_days",
        "pickup_day",
        "forecast_adversity",
        "forecast_uncertainty",
        "forecast_trust",
        "future_use_expectation",
        "storage_burden",
        "return_burden_multiplier",
    ):
        if name in request:
            updates[name] = request[name]
    return replace(session, **updates)


def _set_available_units(
    environment: OutfitterEnvironment,
    state: Any,
    values: dict[str, int],
) -> None:
    if not values:
        return
    for offer_id, quantity in values.items():
        if offer_id not in environment.offers or quantity < 0:
            raise ValueError(f"invalid availability override: {offer_id}")
        offer = environment.offers[offer_id]
        if offer.asset_kind == "consumable":
            state.retail_inventory[offer_id] = {
                "quantity": quantity,
                "avg_unit_cost": float(offer.unit_cost),
            }
            continue
        matching = [asset for asset in state.assets if asset.offer_id == offer_id]
        for index, asset in enumerate(matching):
            asset.status = "available" if index < quantity else "retired"
        for index in range(len(matching), quantity):
            state.assets.append(
                RentalAsset(
                    asset_id=f"lab-{offer_id}-{index + 1}",
                    offer_id=offer_id,
                    service_line=offer.service_line,
                    unit_cost=float(offer.unit_cost),
                    used_life_days=0,
                    service_life_limit_days=int(offer.service_life_limit_days or 365),
                    used_rental_cycles=0,
                    service_life_limit_cycles=offer.service_life_limit_cycles,
                    damage_risk_band=str(offer.damage_risk_band or "medium"),
                    status="available",
                )
            )
        state.rental_service_open[offer_id] = quantity > 0


def _apply_scenario(
    environment: OutfitterEnvironment,
    state: Any,
    scenario: dict[str, Any],
) -> None:
    _set_available_units(environment, state, scenario["available_units"])
    for offer_id, price in scenario["prices"].items():
        if offer_id not in environment.offers or price <= 0:
            raise ValueError(f"invalid price override: {offer_id}")
        state.prices[offer_id] = price
    for offer_id, deposit in scenario["deposits"].items():
        if offer_id not in state.deposits or deposit < 0:
            raise ValueError(f"invalid deposit override: {offer_id}")
        state.deposits[offer_id] = deposit


def _offer_ids(environment: OutfitterEnvironment, request: dict[str, Any]) -> list[str]:
    return sorted(
        offer.offer_id
        for offer in environment.by_service[str(request["service_key"])]
    )


def explain(
    data_dir: Path,
    scenario_value: dict[str, Any],
    *,
    world: str,
    seed: int,
) -> dict[str, Any]:
    scenario = _scenario(scenario_value)
    environment, state = _environment(data_dir, world=world, seed=seed)
    _apply_scenario(environment, state, scenario)
    session = _session_from_request(environment, state, scenario["request"])
    ids = _offer_ids(environment, scenario["request"])
    if environment.channel_demand_model is not None and world != "control":
        trace, views = environment.explain_channel_demand(state, session)
        explanation = (
            f"hidden channel={trace.selected_channel}; "
            f"store_profile={trace.best_store_profile_id}; "
            f"online_profile={trace.best_online_profile_id}; "
            f"selected_offer={trace.selected_offer_id}"
        )
        model_kind = "exact_offer_channel"
    else:
        trace, views = environment.explain_reference_demand(state, session, ids)
        explanation = (
            f"{trace.reason}; p(shop)={trace.entry_probability:.6f}; "
            f"best_profile={trace.best_profile_id}; "
            f"selected_offer={trace.selected_offer_id}"
        )
        model_kind = "reference_transaction"
    view_rows = [asdict(row) for row in views]
    return {
        "format": "outfitter-demand-lab-explain-v1",
        "data_dir": str(data_dir),
        "world_id": world,
        "model_kind": model_kind,
        "seed": seed,
        "scenario": scenario,
        "normalized_request": asdict(session),
        "natural_request": {
            "service": session.service_key,
            "text": (
                f"request {session.units_requested} unit(s) for "
                f"{session.duration_days or 1} day(s), party="
                f"{session.adults + session.children}"
            ),
        },
        "offer_views": view_rows,
        "consumer_profiles": list(trace.profiles),
        "decision": trace.to_dict(),
        "explanation": explanation,
        "input_hash": _canonical_hash(
            {"scenario": scenario, "world": world, "seed": seed}
        ),
    }


def _set_uniform_ratio(
    environment: OutfitterEnvironment,
    state: Any,
    session: ChoiceSession,
    offer_ids: list[str],
    ratio: float,
    *,
    target_offer: str | None,
) -> None:
    views = environment._reference_offer_views(state, session, offer_ids)
    for view in views:
        if target_offer is not None and view.offer_id != target_offer:
            continue
        if view.reference_generalized_cost is None:
            continue
        offer = environment.offers[view.offer_id]
        resources = max(1, view.required_units)
        duration = 1 if offer.asset_kind == "consumable" else max(
            1, int(session.duration_days or 1)
        )
        target_generalized = ratio * view.reference_generalized_cost
        target_cash = max(0.01, target_generalized - 0.05 * view.shop_deposit)
        state.prices[view.offer_id] = target_cash / (resources * duration)


def _set_channel_ratio(
    environment: OutfitterEnvironment,
    state: Any,
    session: ChoiceSession,
    offer_ids: list[str],
    ratio: float,
    *,
    target_offer: str | None,
) -> None:
    model = environment.channel_demand_model
    assert model is not None
    views = environment._channel_offer_views(state, session)
    duration = max(1, int(session.duration_days or 1))
    for view in views:
        if view.offer_id not in offer_ids:
            continue
        if target_offer is not None and view.offer_id != target_offer:
            continue
        if session.service_key.startswith("rental:"):
            target_generalized = ratio * view.online_cash_price
            target_cash = max(
                0.01,
                target_generalized
                - model.deposit_friction_rate * view.store_deposit,
            )
            state.prices[view.offer_id] = target_cash / (
                max(1, view.requested_units) * duration
            )
        else:
            state.prices[view.offer_id] = ratio * view.online_unit_price


def _channel_curve(
    environment: OutfitterEnvironment,
    opening: Any,
    scenario: dict[str, Any],
    *,
    world: str,
    seed: int,
    sessions: int,
    target_offer: str | None,
    grid_step: float,
    max_ratio: float | None,
) -> dict[str, Any]:
    ids = _offer_ids(environment, scenario["request"])
    rental = str(scenario["request"]["service_key"]).startswith("rental:")
    start = 0.1 if rental else 0.5
    maximum_ratio = (
        float(max_ratio)
        if max_ratio is not None
        else (3.0 if rental else CHANNEL_RETAIL_MAX_RATIO)
    )
    if maximum_ratio < start:
        raise ValueError("curve max_ratio must not be below its starting ratio")
    checkpoints = CHANNEL_RENTAL_RATIOS if rental else STANDARD_RATIOS
    ratios = sorted(
        {ratio for ratio in checkpoints if ratio <= maximum_ratio}
        | {
            round(start + index * grid_step, 6)
            for index in range(
                math.floor((maximum_ratio - start) / grid_step + 1e-12) + 1
            )
        }
        | {round(maximum_ratio, 6)}
    )
    rows = []
    for ratio in ratios:
        state = copy.deepcopy(opening)
        first_session = _session_from_request(
            environment, state, scenario["request"], sample_index=0
        )
        _set_channel_ratio(
            environment,
            state,
            first_session,
            ids,
            ratio,
            target_offer=target_offer,
        )
        store_sales = 0
        revenue = 0.0
        cogs = 0.0
        channels: Counter[str] = Counter()
        shares: Counter[str] = Counter()
        for index in range(sessions):
            session = _session_from_request(
                environment, state, scenario["request"], sample_index=index
            )
            trace, views = environment.explain_channel_demand(state, session)
            channels[trace.selected_channel] += 1
            if trace.selected_channel != "store" or trace.selected_offer_id is None:
                continue
            store_sales += 1
            shares[trace.selected_offer_id] += 1
            selected_profile = next(
                row
                for row in trace.profiles
                if row["profile_id"] == trace.selected_profile_id
            )
            revenue += float(selected_profile["store_cash_price"] or 0.0)
            if not rental:
                cogs += sum(
                    int(line["resource_units"])
                    * float(environment.offers[str(line["offer_id"])].unit_cost)
                    for line in selected_profile["store_quote_plan"]
                )
        expected_revenue = revenue / sessions
        expected_cogs = None if rental else cogs / sessions
        rows.append(
            {
                "store_to_online_ratio": ratio,
                "sessions": sessions,
                "store_conversion": store_sales / sessions,
                "channel_shares": {
                    key: value / sessions for key, value in sorted(channels.items())
                },
                "expected_store_revenue_per_request": expected_revenue,
                "expected_store_cogs_per_request": expected_cogs,
                "expected_store_gross_contribution_per_request": (
                    None if rental else expected_revenue - float(expected_cogs)
                ),
                "conditional_store_sku_shares": {
                    key: value / max(1, store_sales)
                    for key, value in sorted(shares.items())
                },
            }
        )
    optimum = max(
        rows,
        key=lambda row: (
            row["expected_store_revenue_per_request"],
            -row["store_to_online_ratio"],
        ),
    )
    gross_contribution_optimum = (
        None
        if rental
        else max(
            rows,
            key=lambda row: (
                row["expected_store_gross_contribution_per_request"],
                -row["store_to_online_ratio"],
            ),
        )
    )
    return {
        "format": "outfitter-demand-lab-channel-curve-v2",
        "data_dir": str(environment.dataset.root),
        "model_kind": "exact_offer_channel",
        "world_id": world,
        "seed": seed,
        "scenario": scenario,
        "price_axis": (
            "rental_generalized_contract_to_online_purchase"
            if str(scenario["request"]["service_key"]).startswith("rental:")
            else "store_unit_price_to_exact_online_unit_price"
        ),
        "pricing_policy": "target-offer" if target_offer else "group-uniform",
        "target_offer": target_offer,
        "sessions_per_point": sessions,
        "scan_max_ratio": maximum_ratio,
        "standard_checkpoints": [
            row for row in rows if row["store_to_online_ratio"] in checkpoints
        ],
        "grid": rows,
        "revenue_optimum": optimum,
        "gross_contribution_optimum": gross_contribution_optimum,
        "online_rationality_envelope": online_rationality_envelope(
            environment.channel_demand_model, world
        ),
        "common_random_requests": True,
    }


def curve(
    data_dir: Path,
    scenario_value: dict[str, Any],
    *,
    world: str,
    seed: int,
    sessions: int,
    target_offer: str | None = None,
    grid_step: float = 0.025,
    max_ratio: float | None = None,
) -> dict[str, Any]:
    scenario = _scenario(scenario_value)
    environment, opening = _environment(data_dir, world=world, seed=seed)
    _apply_scenario(environment, opening, scenario)
    if environment.channel_demand_model is not None and world != "control":
        return _channel_curve(
            environment,
            opening,
            scenario,
            world=world,
            seed=seed,
            sessions=sessions,
            target_offer=target_offer,
            grid_step=grid_step,
            max_ratio=max_ratio,
        )
    ids = _offer_ids(environment, scenario["request"])
    maximum_ratio = 3.0 if max_ratio is None else float(max_ratio)
    if maximum_ratio < 0.5:
        raise ValueError("curve max_ratio must be at least 0.5")
    ratios = sorted(
        {ratio for ratio in STANDARD_RATIOS if ratio <= maximum_ratio}
        | {
            round(0.5 + index * grid_step, 6)
            for index in range(
                math.floor((maximum_ratio - 0.5) / grid_step + 1e-12) + 1
            )
        }
        | {round(maximum_ratio, 6)}
    )
    rows = []
    for ratio in ratios:
        state = copy.deepcopy(opening)
        first_session = _session_from_request(
            environment, state, scenario["request"], sample_index=0
        )
        _set_uniform_ratio(
            environment,
            state,
            first_session,
            ids,
            ratio,
            target_offer=target_offer,
        )
        bought = 0
        revenue = 0.0
        shares: Counter[str] = Counter()
        outside = 0
        for index in range(sessions):
            session = _session_from_request(
                environment, state, scenario["request"], sample_index=index
            )
            trace, views = environment.explain_reference_demand(
                state, session, ids
            )
            if not trace.bought or trace.selected_offer_id is None:
                outside += 1
                continue
            bought += 1
            shares[trace.selected_offer_id] += 1
            selected = next(
                row for row in views if row.offer_id == trace.selected_offer_id
            )
            revenue += selected.shop_cash_price
        rows.append(
            {
                "price_to_reference_ratio": ratio,
                "sessions": sessions,
                "conversion": bought / sessions,
                "outside_rate": outside / sessions,
                "expected_revenue_per_request": revenue / sessions,
                "conditional_sku_shares": {
                    key: value / max(1, bought)
                    for key, value in sorted(shares.items())
                },
            }
        )
    optimum = max(
        rows,
        key=lambda row: (
            row["expected_revenue_per_request"],
            -row["price_to_reference_ratio"],
        ),
    )
    checkpoints = [
        row for row in rows if row["price_to_reference_ratio"] in STANDARD_RATIOS
    ]
    return {
        "format": "outfitter-demand-lab-curve-v1",
        "data_dir": str(data_dir),
        "world_id": world,
        "seed": seed,
        "scenario": scenario,
        "pricing_policy": "target-offer" if target_offer else "group-uniform",
        "target_offer": target_offer,
        "sessions_per_point": sessions,
        "scan_max_ratio": maximum_ratio,
        "standard_checkpoints": checkpoints,
        "grid": rows,
        "revenue_optimum": optimum,
        "common_random_requests": True,
    }


def _evaluation_from_trace_input(
    environment: OutfitterEnvironment,
    state: Any,
    session: ChoiceSession,
    views: tuple[ReferenceOfferView, ...],
    *,
    profile_gumbels: dict[str, float] | None = None,
):
    model = environment.reference_demand_model
    assert model is not None
    group_key = f"{session.service_key}|{session.preferred_group}"
    group = model.groups[group_key]
    real_offer_ids = sorted(
        {row.offer_id for row in views if row.offer_id in environment.offers}
    )
    base_trace, _ = environment.explain_reference_demand(
        state, session, real_offer_ids
    )
    return evaluate_reference_demand(
        model,
        ReferenceEvaluationInput(
            service_key=session.service_key,
            preferred_group=session.preferred_group,
            customer_segment=session.customer_segment,
            world_id=environment.config.demand_world_id,
            group_need_adjustment=base_trace.context_terms["need"],
            accessibility_adjustment=base_trace.context_terms["accessibility"],
            context_adjustment=base_trace.context_terms["context"],
            local_shift=base_trace.context_terms["local_shift"],
            quality_salience={"low": 0.5, "medium": 1.0, "high": 1.5}[
                group.quality_salience
            ],
            substitution_penalty={"low": 1.2, "medium": 0.7, "high": 0.35}[
                group.substitution_strength
            ],
            shop_gumbel=base_trace.shop_gumbel,
            outside_gumbel=base_trace.outside_gumbel,
            profile_gumbels=(
                {
                    row.profile_id: environment._choice_shock(
                        session, "offer", row.profile_id
                    )
                    for row in views
                }
                if profile_gumbels is None
                else profile_gumbels
            ),
            offers=views,
        ),
    )


def _channel_evaluation_from_views(
    environment: OutfitterEnvironment,
    session: ChoiceSession,
    views: tuple[ChannelOfferView, ...],
    *,
    profile_gumbels: dict[str, float] | None = None,
):
    model = environment.channel_demand_model
    assert model is not None
    return evaluate_channel_demand(
        model,
        ChannelEvaluationInput(
            service_key=session.service_key,
            preferred_group=session.preferred_group,
            customer_segment=session.customer_segment,
            world_id=environment.config.demand_world_id,
            intent_role=session.intent_role,
            existing_option_available=False,
            channel_affinities={},
            channel_gumbels={},
            profile_gumbels=profile_gumbels or {},
            offers=views,
            need_horizon_days=session.need_horizon_days,
            occasion_importance=session.occasion_importance,
        ),
    )


def _channel_invariance(
    environment: OutfitterEnvironment,
    state: Any,
    scenario: dict[str, Any],
    session: ChoiceSession,
) -> dict[str, Any]:
    views = environment._channel_offer_views(state, session)
    compatible = [row for row in views if row.hard_compatible]
    if not compatible:
        raise ValueError("channel invariance requires one compatible exact offer")
    anchor = compatible[0]
    baseline = _channel_evaluation_from_views(environment, session, views)
    tests = []

    duplicated = tuple(views) + tuple(
        replace(anchor, offer_id=f"zz-duplicate-{index}") for index in range(100)
    )
    duplicate = _channel_evaluation_from_views(
        environment, session, duplicated
    )
    tests.append(
        {
            "name": "duplicate_profile_invariance",
            "passed": duplicate.channel_systematic == baseline.channel_systematic,
            "entry_probability_delta": 0.0,
        }
    )

    deep = tuple(
        replace(
            row,
            available_units=max(row.available_units, row.requested_units, 100),
        )
        for row in views
    )
    deep_trace = _channel_evaluation_from_views(environment, session, deep)
    tests.append(
        {
            "name": "sufficient_inventory_depth_invariance",
            "passed": deep_trace.channel_systematic == baseline.channel_systematic,
            "entry_probability_delta": 0.0,
        }
    )

    incompatible = replace(
        anchor,
        offer_id="zz-incompatible-low-price",
        profile_id="zz-incompatible-low-price-profile",
        hard_compatible=False,
        incompatibility_reasons=("synthetic_physical_mismatch",),
        store_unit_price=0.01,
        store_cash_price=0.01,
        online_package_price=0.01,
        online_unit_price=0.01,
        online_cash_price=0.01,
    )
    incompatible_trace = _channel_evaluation_from_views(
        environment, session, tuple(views) + (incompatible,)
    )
    tests.append(
        {
            "name": "incompatible_low_price_invariance",
            "passed": incompatible_trace.channel_systematic
            == baseline.channel_systematic,
            "entry_probability_delta": 0.0,
        }
    )

    shocked = _channel_evaluation_from_views(
        environment,
        session,
        views,
        profile_gumbels={row.profile_id: 1_000.0 for row in views},
    )
    tests.append(
        {
            "name": "offer_shock_excluded_from_channel_entry",
            "passed": shocked.channel_systematic == baseline.channel_systematic,
            "entry_probability_delta": 0.0,
        }
    )

    ids = _offer_ids(environment, scenario["request"])
    original_costs = {
        offer_id: environment.offers[offer_id].unit_cost for offer_id in ids
    }
    try:
        for index, offer_id in enumerate(ids):
            environment.offers[offer_id].unit_cost = original_costs[offer_id] * (
                0.01 if index % 2 else 100.0
            )
        poisoned_views = environment._channel_offer_views(state, session)
        poisoned = _channel_evaluation_from_views(
            environment, session, poisoned_views
        )
    finally:
        for offer_id, unit_cost in original_costs.items():
            environment.offers[offer_id].unit_cost = unit_cost
    tests.append(
        {
            "name": "supplier_cost_poisoning_invariance",
            "passed": poisoned.channel_systematic == baseline.channel_systematic,
            "entry_probability_delta": 0.0,
        }
    )

    unpriced_views = tuple(
        replace(row, store_unit_price=None, store_cash_price=None) for row in views
    )
    unpriced = _channel_evaluation_from_views(
        environment, session, unpriced_views
    )
    tests.append(
        {
            "name": "unpriced_store_is_ineligible",
            "passed": unpriced.channel_systematic["store"] == float("-inf"),
            "entry_probability_delta": 0.0,
        }
    )

    cheaper_online_views = tuple(
        replace(
            row,
            online_package_price=row.online_package_price * 0.5,
            online_unit_price=row.online_unit_price * 0.5,
            online_cash_price=row.online_cash_price * 0.5,
        )
        if row.hard_compatible
        else row
        for row in views
    )
    cheaper_online = _channel_evaluation_from_views(
        environment, session, cheaper_online_views
    )
    tests.append(
        {
            "name": "online_price_sensitivity",
            "passed": cheaper_online.channel_systematic["online"]
            > baseline.channel_systematic["online"],
            "entry_probability_delta": 0.0,
        }
    )
    return {
        "format": "outfitter-demand-lab-channel-invariance-v1",
        "data_dir": str(environment.dataset.root),
        "model_kind": "exact_offer_channel",
        "world_id": environment.config.demand_world_id,
        "scenario": scenario,
        "tests": tests,
        "passed": all(row["passed"] for row in tests),
        "baseline_trace_hash": _canonical_hash(baseline.to_dict()),
    }


def invariance(
    data_dir: Path,
    scenario_value: dict[str, Any],
    *,
    world: str,
    seed: int,
) -> dict[str, Any]:
    scenario = _scenario(scenario_value)
    environment, state = _environment(data_dir, world=world, seed=seed)
    _apply_scenario(environment, state, scenario)
    session = _session_from_request(environment, state, scenario["request"])
    if environment.channel_demand_model is not None and world != "control":
        return _channel_invariance(environment, state, scenario, session)
    ids = _offer_ids(environment, scenario["request"])
    baseline, views = environment.explain_reference_demand(state, session, ids)
    eligible = [row for row in views if row.hard_compatible and row.price_ratio is not None]
    if not eligible:
        raise ValueError("invariance scenario requires one compatible reference-covered offer")
    anchor = eligible[0]
    tests = []

    duplicated = tuple(views) + tuple(
        replace(anchor, offer_id=f"zz-duplicate-{index}") for index in range(100)
    )
    duplicate_trace = _evaluation_from_trace_input(
        environment, state, session, duplicated
    )
    tests.append(
        {
            "name": "duplicate_profile_invariance",
            "passed": math.isclose(
                baseline.entry_probability,
                duplicate_trace.entry_probability,
                abs_tol=1e-12,
            )
            and baseline.bought == duplicate_trace.bought
            and baseline.selected_profile_id == duplicate_trace.selected_profile_id,
            "entry_probability_delta": abs(
                baseline.entry_probability - duplicate_trace.entry_probability
            ),
        }
    )

    deep_inventory = tuple(
        replace(row, available_units=max(row.required_units, 100)) for row in views
    )
    inventory_trace = _evaluation_from_trace_input(
        environment, state, session, deep_inventory
    )
    tests.append(
        {
            "name": "sufficient_inventory_depth_invariance",
            "passed": math.isclose(
                baseline.entry_probability,
                inventory_trace.entry_probability,
                abs_tol=1e-12,
            ),
            "entry_probability_delta": abs(
                baseline.entry_probability - inventory_trace.entry_probability
            ),
        }
    )

    unavailable = replace(
        anchor,
        offer_id="zz-unavailable-potential-offer",
        profile_id="zz-unavailable-potential-profile",
        available_units=0,
        shop_cash_price=0.01,
        shop_generalized_cost=0.01,
        price_ratio=0.001,
    )
    unavailable_trace = _evaluation_from_trace_input(
        environment, state, session, tuple(views) + (unavailable,)
    )
    tests.append(
        {
            "name": "unavailable_potential_offer_invariance",
            "passed": math.isclose(
                baseline.entry_probability,
                unavailable_trace.entry_probability,
                abs_tol=1e-12,
            )
            and baseline.selected_profile_id
            == unavailable_trace.selected_profile_id,
            "entry_probability_delta": abs(
                baseline.entry_probability
                - unavailable_trace.entry_probability
            ),
        }
    )

    incompatible = replace(
        anchor,
        offer_id="zz-incompatible-low-price",
        profile_id="zz-incompatible-low-price-profile",
        hard_compatible=False,
        incompatibility_reasons=("synthetic_physical_mismatch",),
        shop_cash_price=0.01,
        shop_generalized_cost=0.01,
        price_ratio=0.001,
    )
    incompatible_trace = _evaluation_from_trace_input(
        environment, state, session, tuple(views) + (incompatible,)
    )
    tests.append(
        {
            "name": "incompatible_low_price_invariance",
            "passed": math.isclose(
                baseline.entry_probability,
                incompatible_trace.entry_probability,
                abs_tol=1e-12,
            )
            and baseline.selected_profile_id
            == incompatible_trace.selected_profile_id,
            "entry_probability_delta": abs(
                baseline.entry_probability - incompatible_trace.entry_probability
            ),
        }
    )

    poisoned_shocks = {row.profile_id: 1_000.0 for row in views}
    shock_trace = _evaluation_from_trace_input(
        environment,
        state,
        session,
        views,
        profile_gumbels=poisoned_shocks,
    )
    tests.append(
        {
            "name": "offer_shock_excluded_from_shop_entry",
            "passed": math.isclose(
                baseline.entry_probability,
                shock_trace.entry_probability,
                abs_tol=1e-12,
            )
            and baseline.bought == shock_trace.bought,
            "entry_probability_delta": abs(
                baseline.entry_probability - shock_trace.entry_probability
            ),
        }
    )

    original_costs = {offer_id: environment.offers[offer_id].unit_cost for offer_id in ids}
    try:
        for index, offer_id in enumerate(ids):
            environment.offers[offer_id].unit_cost = (
                original_costs[offer_id] * (0.01 if index % 2 else 100.0)
            )
        poisoned, _poison_views = environment.explain_reference_demand(
            state, session, ids
        )
    finally:
        for offer_id, cost in original_costs.items():
            environment.offers[offer_id].unit_cost = cost
    tests.append(
        {
            "name": "supplier_cost_poisoning_invariance",
            "passed": math.isclose(
                baseline.entry_probability,
                poisoned.entry_probability,
                abs_tol=1e-12,
            )
            and baseline.bought == poisoned.bought
            and baseline.selected_profile_id == poisoned.selected_profile_id,
            "entry_probability_delta": abs(
                baseline.entry_probability - poisoned.entry_probability
            ),
        }
    )
    return {
        "format": "outfitter-demand-lab-invariance-v1",
        "data_dir": str(data_dir),
        "world_id": world,
        "scenario": scenario,
        "tests": tests,
        "passed": all(row["passed"] for row in tests),
        "baseline_trace_hash": _canonical_hash(baseline.to_dict()),
    }


def _generated_tape(environment: OutfitterEnvironment, sessions: int) -> list[dict[str, Any]]:
    rows = []
    groups = sorted(environment.by_group)
    for index in range(sessions):
        service_key, group = groups[index % len(groups)]
        request = _default_request(service_key, group)
        request["event_address"] = [1, 19, index, 0]
        rows.append(request)
    return rows


def replay(
    data_dir: Path,
    *,
    worlds: list[str],
    seed: int,
    sessions: int,
    tape_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    base_environment, _ = _environment(data_dir, world="base", seed=seed)
    tape = tape_rows or _generated_tape(base_environment, sessions)
    results = []
    aggregates = {}
    for world in worlds:
        environment, state = _environment(data_dir, world=world, seed=seed)
        purchased = 0
        revenue = 0.0
        selected: Counter[str] = Counter()
        channel_counts: Counter[str] = Counter()
        world_rows = []
        for request in tape:
            scenario = _scenario(
                {
                    "format": SCENARIO_FORMAT,
                    "request": request,
                    "available_units": {
                        offer.offer_id: max(
                            1,
                            int(request.get("units_requested", 1)),
                        )
                        for offer in environment.by_service[str(request["service_key"])]
                    },
                }
            )
            local_state = copy.deepcopy(state)
            _apply_scenario(environment, local_state, scenario)
            session = _session_from_request(environment, local_state, request)
            ids = _offer_ids(environment, request)
            if environment.channel_demand_model is not None and not scenario["prices"]:
                _set_channel_ratio(
                    environment,
                    local_state,
                    session,
                    ids,
                    0.25
                    if session.service_key.startswith("rental:")
                    else 1.0,
                    target_offer=None,
                )
            if world == "control":
                selection = environment._select_quoted_offer_reservation(
                    local_state, session, ids
                )
                bought = selection[3] > selection[4]
                offer_id = selection[1] if bought else None
                probability = None
                reason = "bought" if bought else "outside_option"
                cash = (
                    0.0
                    if offer_id is None
                    else environment._line_price(
                        local_state, session, offer_id
                    )
                )
                channel = "store" if bought else "unobserved_nonpurchase"
            elif environment.channel_demand_model is not None:
                trace, views = environment.explain_channel_demand(
                    local_state, session
                )
                channel = trace.selected_channel
                bought = channel == "store"
                offer_id = trace.selected_offer_id if bought else None
                probability = None
                reason = channel
                cash = (
                    0.0
                    if offer_id is None
                    else float(
                        next(
                            row["store_cash_price"]
                            for row in trace.profiles
                            if row["profile_id"] == trace.selected_profile_id
                        )
                        or 0.0
                    )
                )
            else:
                trace, views = environment.explain_reference_demand(
                    local_state, session, ids
                )
                bought = trace.bought
                offer_id = trace.selected_offer_id
                probability = trace.entry_probability
                reason = trace.reason
                cash = (
                    0.0
                    if offer_id is None
                    else next(
                        row.shop_cash_price for row in views if row.offer_id == offer_id
                    )
                )
                channel = "store" if bought else "unobserved_nonpurchase"
            channel_counts[channel] += 1
            if bought and offer_id is not None:
                purchased += 1
                selected[offer_id] += 1
                revenue += cash
            world_rows.append(
                {
                    "event_address": request["event_address"],
                    "service_key": request["service_key"],
                    "entry_probability": probability,
                    "bought": bought,
                    "selected_offer_id": offer_id,
                    "cash_revenue": cash,
                    "reason": reason,
                    "internal_channel": channel,
                }
            )
        aggregates[world] = {
            "requests": len(tape),
            "conversion": purchased / max(1, len(tape)),
            "expected_revenue_per_request": revenue / max(1, len(tape)),
            "selected_offer_counts": dict(sorted(selected.items())),
            "internal_channel_counts": dict(sorted(channel_counts.items())),
        }
        results.append({"world_id": world, "requests": world_rows})
    return {
        "format": "outfitter-demand-lab-replay-v1",
        "data_dir": str(data_dir),
        "seed": seed,
        "tape": tape,
        "tape_hash": _canonical_hash(tape),
        "paired_randomness_scope": (
            "reference worlds share requests, segments, event addresses, service "
            "shocks, and profile shocks; control shares the tape but consumes its "
            "model-specific reservation stream"
        ),
        "worlds": results,
        "aggregates": aggregates,
    }









def build_parser() -> argparse.ArgumentParser:
    from .demand_lab_cli import build_parser as build

    return build()


def main(argv: list[str] | None = None) -> int:
    from .demand_lab_cli import main as run

    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
