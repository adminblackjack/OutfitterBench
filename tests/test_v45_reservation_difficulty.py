from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from outfitter.config import OutfitterConfig
from outfitter.data import OutfitterDataset
from outfitter.environment import OutfitterEnvironment
from outfitter.models import RentalAsset
from outfitter.reservation_difficulty import rental_amount, extension_amount
from outfitter.runtime.ledger import empty_ledger


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "tests/fixtures/outfitter_v50_legacy_main.yaml"
SECONDARY = ROOT / "tests/fixtures/outfitter_v50_legacy_secondary.yaml"


def environment(config=MAIN, **overrides):
    from release_fixtures import v47_runtime
    settings = replace(OutfitterConfig.from_yaml(config), data_dir=str(v47_runtime()), **overrides)
    return OutfitterEnvironment(settings, OutfitterDataset.load(settings.data_dir))


def pending_fixed(env, state, *, optional=False):
    for _ in range(14):
        request = next((row for row in state.reservation_requests
                        if row.status == "pending_service_offer"
                        and (not optional or any(not line.required for line in row.lines))), None)
        if request is not None:
            return request
        env.close_business_day(state)
    raise AssertionError("fixture requires a fixed offer within fourteen days")


def optional_shortage():
    env = environment()
    state = env.reset()
    request = pending_fixed(env, state, optional=True)
    plan = request.customer_authorized_plans[0]
    env.respond_reservation_service_offers(state, [{
        "request_id": request.request_id, "action": "accept",
        "plan_id": plan["plan_id"],
        "pickup_day": request.pickup_date_options[0]["pickup_day"],
    }])
    commitment = state.reservation_commitments[-1]
    state.day = commitment.pickup_day
    state.current_day_ledger = empty_ledger(state.day)
    for asset in state.assets:
        asset.status = "maintenance_hold"
    for line in commitment.lines:
        if not line.required:
            continue
        offer = env.offers[line.offer_id]
        for index in range(line.quantity):
            asset = RentalAsset(
                asset_id=f"v45-test-{line.line_id}-{index}",
                offer_id=offer.offer_id,
                service_line=offer.service_line,
                unit_cost=offer.unit_cost,
                used_life_days=0,
                service_life_limit_days=int(offer.service_life_limit_days),
                damage_risk_band=str(offer.damage_risk_band),
                service_life_limit_cycles=offer.service_life_limit_cycles,
            )
            env._initialize_lifecycle_asset_state(asset)
            asset.durable_capital_basis_cents = env._capital_cents(asset.unit_cost)
            state.assets.append(asset)
    env._initialize_durable_acquisition_budget(state)
    env._process_reservation_pickups(state)
    assert commitment.status == "at_risk"
    assert commitment.partial_fulfillment_missing_line_ids
    return env, state, request, commitment


def test_v45_protocol_and_tariff_are_shared_by_both_tracks():
    main = environment()
    secondary = environment(SECONDARY)
    assert main._v45_reservation_difficulty()
    assert secondary._v45_reservation_difficulty()
    left = OutfitterConfig.from_yaml(MAIN).to_dict()
    right = OutfitterConfig.from_yaml(SECONDARY).to_dict()
    left.pop("assortment_mode")
    right.pop("assortment_mode")
    assert left == right
    assert rental_amount(7.25, 1, 2) == 14.50
    assert rental_amount(7.25, 3, 2) == 29.02
    assert extension_amount(7.25, 2, 4, 2) == 14.52


def test_future_true_weather_cannot_change_today_customer_pool(monkeypatch):
    base = environment()
    first = base.reset()
    forecast = base.weather_forecast(first, 15)

    def same_public_forecast(self, state, horizon):
        return [copy.deepcopy(row) for row in forecast
                if state.day <= int(row["day"]) < state.day + horizon]

    monkeypatch.setattr(OutfitterEnvironment, "weather_forecast", same_public_forecast)
    observations = []
    for condition in ("sunny", "snowy"):
        source = copy.deepcopy(base.source_dataset)
        for day, row in source.weather_calendar.items():
            if day > base.config.weather_start_day:
                row.update(night_low_c=-5.0 if condition == "snowy" else 25.0,
                           temp_mean_c=0.0 if condition == "snowy" else 30.0,
                           day_high_c=5.0 if condition == "snowy" else 35.0)
                source.weather[day] = condition
        changed = OutfitterEnvironment(base.config, source)
        observations.append(changed.reset().to_dict()["reservation_requests"])
    assert observations[0] == observations[1]


def test_multiday_quote_and_fixed_offer_extension_use_same_tariff():
    env = environment()
    state = env.reset()
    inquiry = None
    for _ in range(14):
        inquiry = next((row for row in state.reservation_requests
                        if row.status == "pending_quote" and row.return_day - row.pickup_day > 1), None)
        if inquiry is not None:break
        env.close_business_day(state)
    assert inquiry is not None
    line = inquiry.lines[0]
    profile_id = line.eligible_profile_ids[0]
    offer_id = inquiry.inquiry_profile_member_ids_by_line[line.line_id][profile_id][0]
    option = env._enriched_pairwise_quote_options(inquiry, line, quote_options=[{
        "option_id": "option-1", "offer_id": offer_id,
        "promise_profile_id": profile_id, "daily_unit_price": 7.25,
    }])[0]
    assert option["total_rent"] == rental_amount(7.25, line.duration_days,
                                                  option["asset_quantity"])

    fixed = pending_fixed(env, state)
    plan = fixed.customer_authorized_plans[0]
    env.respond_reservation_service_offers(state, [{
        "request_id": fixed.request_id, "action": "accept",
        "plan_id": plan["plan_id"],
        "pickup_day": fixed.pickup_date_options[0]["pickup_day"],
    }])
    commitment = state.reservation_commitments[-1]
    assert commitment.total_locked_rent == fixed.service_offer_terms["total_rent"]
    assert all(line.locked_line_pricing_basis == "line_first_day_half_additional"
               for line in commitment.lines)
    _, deltas, total = env._extension_rent_delta(
        commitment, proposed_return_day=commitment.return_day + 2)
    assert total == round(sum(deltas.values()), 2)
    for line in commitment.lines:
        assert deltas[line.line_id] == extension_amount(
            line.locked_line_daily_rent, line.duration_days, line.duration_days + 2)


def test_optional_shortage_requires_explicit_customer_agreement():
    env, state, request, commitment = optional_shortage()
    assert env.reservation_inbox(state)["risk_alerts"][0]["missing_optional_line_ids"]
    env._process_reservation_pickups(state, finalize_risks=True)
    assert commitment.status == "pickup_failed"
    assert state.cumulative_reservation_pickups == 0


def test_accepted_reduced_pickup_refunds_and_compensates(monkeypatch):
    env, state, request, commitment = optional_shortage()

    class AcceptedDraw:
        @staticmethod
        def random():
            return 0.0

    original_rng = env._stable_event_rng

    def fixed_response(*args, **kwargs):
        if str(kwargs.get("identity", "")).startswith("partial:"):
            return AcceptedDraw()
        return original_rng(*args, **kwargs)

    monkeypatch.setattr(env, "_stable_event_rng", fixed_response)
    answer = env.resolve_reservation_risks(state, [{
        "commitment_id": commitment.commitment_id,
        "action": "inquire_partial_fulfillment",
    }])["actions"][0]
    assert answer["customer_accepted"] is True
    assert answer["rent_credit"] > 0
    assert answer["compensation"] > 0
    with pytest.raises(ValueError, match="already had"):
        env.resolve_reservation_risks(state, [{
            "commitment_id": commitment.commitment_id,
            "action": "inquire_partial_fulfillment",
        }])
    env.resolve_reservation_risks(state, [{
        "commitment_id": commitment.commitment_id,
        "action": "select_partial_fulfillment",
    }])
    env._process_reservation_pickups(state, finalize_risks=True)
    assert commitment.status == "active"
    assert commitment.optional_line_credits == answer["rent_credit"]
    assert commitment.compensation_paid == answer["compensation"]
    assert state.cumulative_reservation_pickups == 1
