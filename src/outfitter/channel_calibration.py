"""Deterministic mechanism/behavior calibration for channel demand.

The search contains no business-loop profit, inventory purchasing, or Agent
behavior.  It only checks synthetic reference situations whose intended
direction and units are explicit.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from typing import Any

from .channel_demand import (
    ChannelDemandModel,
    ChannelEvaluationInput,
    ChannelOfferView,
    evaluate_channel_demand,
    online_rationality_envelope,
)


CALIBRATION_FORMAT = "outfitter-channel-behavior-calibration-v2"


def _view(
    *,
    choice_group: str,
    kind: str,
    store_unit: float | None,
    store_cash: float | None,
    online_package: float,
    online_units: int,
    online_cash: float,
) -> ChannelOfferView:
    return ChannelOfferView(
        offer_id="synthetic-exact-offer",
        profile_id="synthetic-profile",
        choice_group=choice_group,
        kind=kind,
        hard_compatible=True,
        incompatibility_reasons=(),
        soft_fit=0.0,
        quality_score=0.0,
        requested_units=1,
        available_units=1,
        store_unit_price=store_unit,
        store_cash_price=store_cash,
        store_deposit=0.0,
        online_package_price=online_package,
        online_package_units=online_units,
        online_unit_price=online_package / online_units,
        online_cash_price=online_cash,
    )


def _trace(
    model: ChannelDemandModel,
    *,
    service_key: str,
    choice_group: str,
    world: str,
    view: ChannelOfferView,
    intent_role: str = "required",
    need_horizon_days: int | None = None,
    occasion_importance: float = 1.0,
):
    return evaluate_channel_demand(
        model,
        ChannelEvaluationInput(
            service_key=service_key,
            preferred_group=choice_group,
            customer_segment="balanced",
            world_id=world,
            intent_role=intent_role,
            existing_option_available=False,
            channel_affinities={},
            channel_gumbels={},
            profile_gumbels={},
            offers=(view,),
            need_horizon_days=need_horizon_days,
            occasion_importance=occasion_importance,
        ),
    )


def _choice_probability(
    trace: Any,
    model: ChannelDemandModel,
    channel: str,
    alternatives: tuple[str, ...],
) -> float:
    finite = {
        key: float(trace.channel_systematic[key])
        for key in alternatives
        if math.isfinite(float(trace.channel_systematic[key]))
    }
    peak = max(finite.values())
    weights = {
        key: math.exp((value - peak) / model.channel_shock_scale)
        for key, value in finite.items()
    }
    return weights.get(channel, 0.0) / sum(weights.values())


def _evaluate_candidate(model: ChannelDemandModel) -> dict[str, Any]:
    retail_policy = next(
        row for row in model.groups.values() if row.service_key.startswith("retail:")
    )
    rental_policy = next(
        row for row in model.groups.values() if row.service_key.startswith("rental:")
    )
    disposable_poncho_policy = next(
        row
        for row in model.groups.values()
        if row.service_key == "retail:rain_gear"
        and row.choice_group == "disposable_poncho"
    )
    tests: list[dict[str, Any]] = []
    wipe_limits = {
        "conservative": 0.03,
        "base": 0.10,
        "convenience-heavy": 0.20,
    }
    water_floors = {
        "conservative": 0.40,
        "base": 0.55,
        "convenience-heavy": 0.65,
    }
    parity_bands = {world: (0.45, 0.55) for world in model.worlds}
    for world in sorted(model.worlds):
        wipe = _trace(
            model,
            service_key=retail_policy.service_key,
            choice_group=retail_policy.choice_group,
            world=world,
            view=_view(
                choice_group=retail_policy.choice_group,
                kind="retail_item",
                store_unit=11.85,
                store_cash=11.85,
                online_package=9.78,
                online_units=3,
                online_cash=9.78,
            ),
        )
        wipe_share = _choice_probability(
            wipe, model, "store", ("store", "online")
        )
        tests.append(
            {
                "name": f"{world}:wipe_unit_fairness",
                "value": wipe_share,
                "passed": wipe_share <= wipe_limits[world],
            }
        )

        water = _trace(
            model,
            service_key=retail_policy.service_key,
            choice_group=retail_policy.choice_group,
            world=world,
            view=_view(
                choice_group=retail_policy.choice_group,
                kind="retail_item",
                store_unit=3.48,
                store_cash=3.48,
                online_package=30.90,
                online_units=24,
                online_cash=30.90,
            ),
            need_horizon_days=0,
            occasion_importance=1.0,
        )
        water_share = _choice_probability(
            water, model, "store", ("store", "online")
        )
        tests.append(
            {
                "name": f"{world}:water_break_pack",
                "value": water_share,
                "passed": water_share >= water_floors[world],
            }
        )

        rain_curves: dict[str, list[dict[str, float]]] = {}
        # The frozen exact reusable-rainwear offer is a one-unit package: its
        # supplier unit cost and exact online unit price are both CNY 4.50.
        rainwear_supplier_unit_cost = 4.5
        for label, horizon, importance in (
            ("urgent", 0, 2.50),
            ("planned", 3, 0.85),
        ):
            rows = []
            for ratio in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0):
                    price = 4.5 * ratio
                    rain = _trace(
                        model,
                        service_key=retail_policy.service_key,
                        choice_group=retail_policy.choice_group,
                        world=world,
                        view=_view(
                            choice_group=retail_policy.choice_group,
                            kind="retail_item",
                            store_unit=price,
                            store_cash=price,
                            online_package=4.5,
                            online_units=1,
                            online_cash=4.5,
                        ),
                        need_horizon_days=horizon,
                        occasion_importance=importance,
                    )
                    store_probability = _choice_probability(
                        rain, model, "store", ("store", "online", "abandon")
                    )
                    rows.append(
                        {
                            "ratio": ratio,
                            "store_probability": store_probability,
                            "gross_contribution": store_probability
                            * (price - rainwear_supplier_unit_cost),
                        }
                    )
            rain_curves[label] = rows
        urgent_best = max(
            rain_curves["urgent"],
            key=lambda row: (row["gross_contribution"], -row["ratio"]),
        )
        planned_best = max(
            rain_curves["planned"],
            key=lambda row: (row["gross_contribution"], -row["ratio"]),
        )
        urgent_at_150 = next(
            row for row in rain_curves["urgent"] if row["ratio"] == 1.5
        )
        planned_at_150 = next(
            row for row in rain_curves["planned"] if row["ratio"] == 1.5
        )
        tests.append(
            {
                "name": f"{world}:deadline_aware_rainwear_profit_curve",
                "value": {
                    "supplier_unit_cost": rainwear_supplier_unit_cost,
                    "online_unit_price": 4.5,
                    "urgent": rain_curves["urgent"],
                    "planned": rain_curves["planned"],
                    "urgent_optimum": urgent_best,
                    "planned_optimum": planned_best,
                },
                "passed": urgent_best["ratio"] > 1.0
                and urgent_best["gross_contribution"] > 0.0
                and urgent_best["gross_contribution"]
                >= 1.20 * max(0.01, planned_best["gross_contribution"])
                and urgent_at_150["store_probability"]
                > planned_at_150["store_probability"]
                and rain_curves["urgent"][0]["gross_contribution"] < 0.0
                and rain_curves["urgent"][-1]["gross_contribution"]
                < urgent_best["gross_contribution"],
            }
        )

        poncho_curves: dict[str, list[dict[str, float]]] = {}
        for label, horizon, importance in (
            ("urgent", 0, 2.50),
            ("planned", 3, 0.85),
        ):
            rows = []
            for ratio in (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0):
                price = ratio
                poncho = _trace(
                    model,
                    service_key=disposable_poncho_policy.service_key,
                    choice_group=disposable_poncho_policy.choice_group,
                    world=world,
                    view=_view(
                        choice_group=disposable_poncho_policy.choice_group,
                        kind="retail_item",
                        store_unit=price,
                        store_cash=price,
                        online_package=1.0,
                        online_units=1,
                        online_cash=1.0,
                    ),
                    need_horizon_days=horizon,
                    occasion_importance=importance,
                )
                probability = _choice_probability(
                    poncho, model, "store", ("store", "online", "abandon")
                )
                rows.append(
                    {
                        "ratio": ratio,
                        "store_price_cny": price,
                        "store_probability": probability,
                        "gross_contribution": probability * (price - 1.0),
                    }
                )
            poncho_curves[label] = rows
        urgent_poncho_best = max(
            poncho_curves["urgent"],
            key=lambda row: (row["gross_contribution"], -row["ratio"]),
        )
        planned_poncho_best = max(
            poncho_curves["planned"],
            key=lambda row: (row["gross_contribution"], -row["ratio"]),
        )
        tests.append(
            {
                "name": f"{world}:deadline_aware_disposable_poncho_profit_curve",
                "value": {
                    "supplier_unit_cost": 1.0,
                    "online_unit_price": 1.0,
                    "urgent": poncho_curves["urgent"],
                    "planned": poncho_curves["planned"],
                    "urgent_optimum": urgent_poncho_best,
                    "planned_optimum": planned_poncho_best,
                },
                "passed": 1.0 < urgent_poncho_best["ratio"] < 20.0
                and urgent_poncho_best["gross_contribution"] > 0.0
                and urgent_poncho_best["gross_contribution"]
                >= 3.0 * max(
                    0.01, planned_poncho_best["gross_contribution"]
                )
                and poncho_curves["urgent"][0]["gross_contribution"] < 0.0
                and poncho_curves["urgent"][-1]["gross_contribution"]
                < urgent_poncho_best["gross_contribution"],
            }
        )

        parity = _trace(
            model,
            service_key=retail_policy.service_key,
            choice_group=retail_policy.choice_group,
            world=world,
            view=_view(
                choice_group=retail_policy.choice_group,
                kind="retail_item",
                store_unit=10.0,
                store_cash=10.0,
                online_package=10.0,
                online_units=1,
                online_cash=10.0,
            ),
        )
        parity_share = _choice_probability(
            parity, model, "store", ("store", "online")
        )
        low, high = parity_bands[world]
        tests.append(
            {
                "name": f"{world}:equal_price_channel_balance",
                "value": parity_share,
                "passed": low <= parity_share <= high,
            }
        )

        markup = _trace(
            model,
            service_key=retail_policy.service_key,
            choice_group=retail_policy.choice_group,
            world=world,
            view=_view(
                choice_group=retail_policy.choice_group,
                kind="retail_item",
                store_unit=15.0,
                store_cash=15.0,
                online_package=10.0,
                online_units=1,
                online_cash=10.0,
            ),
        )
        markup_share = _choice_probability(
            markup, model, "store", ("store", "online")
        )
        tests.append(
            {
                "name": f"{world}:markup_reduces_store_share",
                "value": parity_share - markup_share,
                "passed": markup_share <= 0.70 * parity_share,
            }
        )

        rental_shares = {}
        for ratio in (0.1, 0.5, 1.0, 1.5):
            rental = _trace(
                model,
                service_key=rental_policy.service_key,
                choice_group=rental_policy.choice_group,
                world=world,
                view=_view(
                    choice_group=rental_policy.choice_group,
                    kind="rental_asset",
                    store_unit=100.0 * ratio,
                    store_cash=100.0 * ratio,
                    online_package=100.0,
                    online_units=1,
                    online_cash=100.0,
                ),
            )
            rental_shares[str(ratio)] = _choice_probability(
                rental, model, "store", ("store", "online", "abandon")
            )
        ratio_one_limit = {
            "conservative": 0.10,
            "base": 0.20,
            "convenience-heavy": 0.35,
        }[world]
        ratio_high_limit = {
            "conservative": 0.05,
            "base": 0.10,
            "convenience-heavy": 0.15,
        }[world]
        tests.append(
            {
                "name": f"{world}:rental_ratio_curve",
                "value": rental_shares,
                "passed": rental_shares["0.1"] >= 0.60
                and rental_shares["0.1"] > rental_shares["0.5"]
                > rental_shares["1.0"]
                > rental_shares["1.5"]
                and rental_shares["1.0"] <= ratio_one_limit
                and rental_shares["1.5"] <= ratio_high_limit,
            }
        )

        rationality = online_rationality_envelope(model, world)
        tests.append(
            {
                "name": f"{world}:online_seller_rationality_envelope",
                "value": rationality,
                "passed": rationality["passed"],
            }
        )

    cheap = _trace(
        model,
        service_key=retail_policy.service_key,
        choice_group=retail_policy.choice_group,
        world="base",
        view=_view(
            choice_group=retail_policy.choice_group,
            kind="retail_item",
            store_unit=None,
            store_cash=None,
            online_package=70.0,
            online_units=1,
            online_cash=70.0,
        ),
    )
    expensive = _trace(
        model,
        service_key=retail_policy.service_key,
        choice_group=retail_policy.choice_group,
        world="base",
        view=_view(
            choice_group=retail_policy.choice_group,
            kind="retail_item",
            store_unit=None,
            store_cash=None,
            online_package=709.0,
            online_units=1,
            online_cash=709.0,
        ),
    )
    cheap_purchase = _choice_probability(
        cheap, model, "online", ("online", "abandon")
    )
    expensive_purchase = _choice_probability(
        expensive, model, "online", ("online", "abandon")
    )
    tests.append(
        {
            "name": "absolute_price_not_self_justifying",
            "value": cheap_purchase - expensive_purchase,
            "passed": cheap_purchase - expensive_purchase >= 0.30,
        }
    )
    return {"passed": all(row["passed"] for row in tests), "tests": tests}


def calibrate_channel_model(
    model: ChannelDemandModel,
) -> tuple[ChannelDemandModel, dict[str, Any]]:
    candidates = []
    for required in (6.7, 6.9, 7.1):
        for optional in (4.2, 4.4, 4.6):
            for shock in (0.55, 0.60, 0.65):
                for deadline_scale in (0.8, 1.0, 1.2):
                    candidate = replace(
                        model,
                        required_need_intercept=required,
                        optional_need_intercept=optional,
                        channel_shock_scale=shock,
                        deadline_value_scale=deadline_scale,
                    )
                    result = _evaluate_candidate(candidate)
                    candidates.append(
                        {
                            "parameters": {
                                "required_need_intercept": required,
                                "optional_need_intercept": optional,
                                "channel_shock_scale": shock,
                                "deadline_value_scale": deadline_scale,
                            },
                            "passed": result["passed"],
                            "tests": result["tests"],
                            "distance_to_declared_center": round(
                                ((required - 6.9) / 0.2) ** 2
                                + ((optional - 4.4) / 0.2) ** 2
                                + ((shock - 0.60) / 0.05) ** 2
                                + ((deadline_scale - 1.0) / 0.2) ** 2,
                                8,
                            ),
                        }
                    )
    passing = [row for row in candidates if row["passed"]]
    if not passing:
        raise ValueError(
            "no channel parameter candidate passes all mechanism/behavior gates"
        )
    selected = min(
        passing,
        key=lambda row: (
            row["distance_to_declared_center"],
            json.dumps(row["parameters"], sort_keys=True),
        ),
    )
    parameters = selected["parameters"]
    resolved = replace(model, **parameters)
    canonical = json.dumps(
        candidates, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return resolved, {
        "format": CALIBRATION_FORMAT,
        "objective": "mechanism and behavior gates only; no Agent or business profit",
        "candidate_count": len(candidates),
        "passing_candidate_count": len(passing),
        "selected_parameters": parameters,
        "selected_tests": selected["tests"],
        "candidate_grid_sha256": hashlib.sha256(canonical).hexdigest(),
        "candidates": candidates,
    }
