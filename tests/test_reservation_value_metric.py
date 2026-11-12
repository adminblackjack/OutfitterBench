from __future__ import annotations

import copy
import json
import sqlite3
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from outfitter.models import RentalAsset, RentalContract
from outfitter.reservation_value import (
    CompensationLedger,
    _capture_daily_walk_in_opportunity,
    _contingency_diagnostics,
    _finalize_daily_rows,
    _daily_no_reservation_capacity_shadow,
    _local_contingency_value_summary,
    _normalized_replay_result,
    _opening_observation_days,
    _recorded_controller_no_op,
    _snapshot_source_root,
    _use_depreciation,
    contract_value,
)


def asset():
    return RentalAsset(
        asset_id="asset-1",
        offer_id="offer-1",
        service_line="tent",
        unit_cost=120.0,
        used_life_days=10,
        service_life_limit_days=120,
        damage_risk_band="medium",
        used_rental_cycles=4,
        service_life_limit_cycles=60,
    )


def contract(*, damaged=False):
    return RentalContract(
        contract_id="rental-1",
        asset_id="asset-1",
        offer_id="offer-1",
        service_line="tent",
        start_day=3,
        return_day=5,
        duration_days=2,
        rent_paid=30.0,
        deposit=20.0,
        damaged_on_return=damaged,
        reservation_id="reservation-1",
    )


def test_use_depreciation_charges_day_and_cycle_wear():
    # 120 * (2/120 + 1/60) = 4.
    assert _use_depreciation(asset(), contract()) == pytest.approx(4.0)


def test_completed_contract_uses_recorded_return_settlement():
    probe = SimpleNamespace(
        repair_cost=lambda _asset: 99.0,
        config=SimpleNamespace(lifecycle_economics_enabled=True),
    )
    state = SimpleNamespace(assets=[asset()])
    value = contract_value(
        probe,
        state,
        contract(damaged=True),
        return_activity={
            "event_type": "returned_damaged",
            "damage_deduction": 8.0,
            "repair_cash_cost": 11.0,
        },
    )
    assert value.rent == 30.0
    assert value.damage_deduction == 8.0
    assert value.damage_economic_cost == 11.0
    assert value.use_depreciation == pytest.approx(4.0)
    assert value.net == pytest.approx(23.0)


def test_rnva_identity_and_normalization():
    rows = {
        1: {
            "day": 1,
            "reservation_rent": 100.0,
            "reservation_damage_deductions": 5.0,
            "reservation_cleaning_cost": 10.0,
            "reservation_damage_economic_cost": 4.0,
            "reservation_use_depreciation": 11.0,
            "reservation_failure_compensation": 8.0,
            "local_walk_in_opportunity_upper_bound": 12.0,
        }
    }
    result = _finalize_daily_rows(rows, initial_official_value=1000.0)[0]
    assert result["reservation_contract_contribution"] == 80.0
    assert result["reservation_direct_net_contribution"] == 72.0
    assert result["rnva_lower_bound"] == 60.0
    assert result[
        "cumulative_rnva_lower_bound_pct_initial_official_value"
    ] == 6.0


def test_replay_normalization_only_removes_model_usage_estimate():
    value = {
        "data": {
            "management_service_usage_estimate": 1.25,
            "management_service_cost": 4.5,
            "nested": [
                {"management_service_usage_estimate": 2.0, "cash": 3.0}
            ],
        }
    }
    assert _normalized_replay_result(value) == {
        "data": {"nested": [{"cash": 3.0}]}
    }


def test_snapshot_source_root_accepts_root_or_src(tmp_path):
    runner = tmp_path / "src" / "outfitter" / "runner.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("", encoding="utf-8")
    assert _snapshot_source_root(tmp_path) == tmp_path / "src"
    assert _snapshot_source_root(tmp_path / "src") == tmp_path / "src"


def test_controller_batch_policy_rejection_is_a_replay_no_op():
    assert _recorded_controller_no_op(
        {
            "ok": False,
            "data": {
                "error": "tool_batch_policy_violation",
                "executed": False,
            },
        }
    )
    assert not _recorded_controller_no_op(
        {"ok": False, "data": {"error": "invalid arguments"}}
    )


def test_v38_contingency_diagnostics_do_not_create_reward_terms():
    inquiry = SimpleNamespace(submitted_day=1)
    report = SimpleNamespace(
        report_id="report-1",
        response_day=2,
        outcome="proposal_accepted",
        status="applied",
        accepted_proposal_ids=["later"],
        counteroffer=None,
        proposals=[
            {
                "proposal_id": "later",
                "pickup_day": 8,
                "return_day": 10,
                "line_changes": [
                    {
                        "line_id": "line-1",
                        "offer_id": "offer-2",
                        "promise_profile_id": "profile-1",
                    }
                ],
            }
        ],
        settlement={},
    )
    application = SimpleNamespace(
        event_type="reservation_contingency_applied",
        day=2,
        details={
            "report_id": "report-1",
            "selection_id": "later",
            "previous_terms": {"pickup_day": 6},
            "new_pickup_day": 8,
        },
    )
    state = SimpleNamespace(
        reservation_contingency_inquiries=[inquiry],
        reservation_contingency_reports=[report],
        reservation_events=[application],
        reservation_commitments=[
            SimpleNamespace(confirmed_day=1, commitment_version=2)
        ],
        cumulative_reservation_contingency_inquiries=1,
    )
    result = _contingency_diagnostics(state, completed_day=2)
    assert result["applications"] == 1
    assert result["application_type_counts"] == {"date_and_equipment": 1}
    assert result["signed_pickup_shift_days"] == 2
    assert result["equipment_line_changes"] == 1
    assert "not reward" in result["scoring_role"]


def test_local_contingency_value_is_separate_actual_minus_keep_original():
    actual = SimpleNamespace(
        reservation_commitments=[
            SimpleNamespace(commitment_id="reservation-1", status="completed")
        ],
        contracts=[],
        assets=[],
        rental_activity=[],
        reservation_events=[],
        day=7, bankrupt=False, cumulative_reservation_compensation=0.0,
    )
    keep_original = SimpleNamespace(
        reservation_commitments=[
            SimpleNamespace(commitment_id="reservation-1", status="pickup_failed")
        ],
        contracts=[],
        assets=[],
        rental_activity=[],
        reservation_events=[
            SimpleNamespace(
                event_type="pickup_failed", day=5,
                commitment_id="reservation-1",
                details={"compensation": 25.0},
            )
        ],
        day=7, bankrupt=False, cumulative_reservation_compensation=25.0,
    )
    result = _local_contingency_value_summary(
        SimpleNamespace(),
        actual,
        [
            {
                "report_id": "report-1",
                "commitment_id": "reservation-1",
                "selection_id": "later",
                "application_day": 3,
                "original_terms": {"pickup_day": 5, "return_day": 7},
                "state": keep_original,
                "compensation_ledger": CompensationLedger(charges=[{
                    "day": 5, "commitment_id": "reservation-1",
                    "kind": "termination", "amount": 25.0, "fully_paid": True,
                }]),
                "continuation_action_divergences": [],
            }
        ],
        actual_compensation_records=[],
        through_day=7,
    )
    assert result["included_in_rnva"] is False
    assert result["mature_estimate_count"] == 1
    assert result["estimates"][0]["actual_applied_contract"][
        "net_local_value"
    ] == 0.0
    assert result["estimates"][0]["keep_original_contract"][
        "net_local_value"
    ] == -25.0
    assert result["estimates"][0]["delta_v_local_cny"] == 25.0


class ChargeEnvironment:
    def _charge(self, state, amount, *, reason):
        if amount > state.cash:
            state.cash = 0.0
            state.bankrupt = True
            return False
        state.cash -= amount
        return True


def compensation_state(*, cash=100.0, day=3):
    return SimpleNamespace(day=day, cash=cash, bankrupt=False,
                           cumulative_reservation_compensation=0.0,
                           reservation_events=[])


def event(kind, *, day=3, amount=18.0, cid="reservation-1", **details):
    return SimpleNamespace(event_type=kind, day=day, commitment_id=cid,
                           details={"compensation": amount, **details})


def charge(ledger, env, state, *, amount=18.0, cid="reservation-1", partial=True):
    state.cumulative_reservation_compensation += amount
    suffix = "减配补偿" if partial else "履约补偿"
    with ledger.capture(env, state):
        return env._charge(state, amount, reason=f"预约 {cid} {suffix}")


@pytest.mark.parametrize("kind", ["partial_fulfillment_inquired", "partial_fulfillment_selected"])
@pytest.mark.parametrize("accepted", [False, True])
def test_compensation_proposals_are_not_costs(kind, accepted):
    state = compensation_state()
    state.reservation_events = [event(kind, customer_accepted=accepted)]
    accounting = CompensationLedger().accounting(state, through_day=3)
    assert accounting["incurred_compensation"] == 0.0


def test_partial_selection_followed_by_full_delivery_has_no_compensation():
    state = compensation_state()
    state.reservation_events = [event("partial_fulfillment_inquired"),
                                event("partial_fulfillment_selected"),
                                event("pickup_fulfilled", amount=0.0)]
    assert CompensationLedger().accounting(state, through_day=3)["incurred_compensation"] == 0.0


@pytest.mark.parametrize("cash", [100.0, 5.0])
def test_partial_settlement_counts_once_even_when_payment_fails(cash):
    state, env, ledger = compensation_state(cash=cash), ChargeEnvironment(), CompensationLedger()
    state.reservation_events = [event("partial_fulfillment_inquired"),
                                event("partial_fulfillment_selected")]
    paid = charge(ledger, env, state)
    if paid:
        state.reservation_events.append(event("pickup_fulfilled", amount=0.0,
                                             partial_fulfillment_compensation=18.0))
    accounting = ledger.accounting(state, through_day=3)
    assert accounting["incurred_compensation"] == 18.0
    assert accounting["partial_fulfillment_compensation"] == 18.0
    assert accounting["counter_reconciliation_difference"] == 0.0
    assert accounting["charge_attempts_not_fully_paid"] == (0.0 if paid else 18.0)
    assert "_charge" not in vars(env)


@pytest.mark.parametrize("kind", ["cancelled_by_manager", "pickup_failed",
                                  "merchant_induced_customer_cancellation"])
@pytest.mark.parametrize("cash", [100.0, 5.0])
def test_terminal_obligation_includes_failed_payment(kind, cash):
    state, env, ledger = compensation_state(cash=cash), ChargeEnvironment(), CompensationLedger()
    state.reservation_events = [event(kind), event("unrelated_compensation_quote", amount=999)]
    charge(ledger, env, state, partial=False)
    accounting = ledger.accounting(state, through_day=3)
    assert accounting["terminal_compensation"] == accounting["incurred_compensation"] == 18.0


def test_refund_bankruptcy_retains_unattempted_terminal_obligation():
    state = compensation_state(cash=0.0)
    state.bankrupt = True
    state.reservation_events = [event("pickup_failed")]
    accounting = CompensationLedger().accounting(state, through_day=3)
    assert accounting["incurred_compensation"] == 18.0
    assert accounting["unattempted_due_to_refund_bankruptcy"] == 18.0
    assert accounting["replay_state_counter"] == 0.0


def test_cutoff_excludes_next_morning_but_reconciles_full_replay_counter():
    state, env, ledger = compensation_state(), ChargeEnvironment(), CompensationLedger()
    charge(ledger, env, state)
    state.day = 4
    charge(ledger, env, state, amount=9.0, cid="reservation-2")
    accounting = ledger.accounting(state, through_day=3)
    assert accounting["incurred_compensation"] == 18.0
    assert accounting["excluded_after_cutoff"] == 9.0
    assert accounting["replay_state_counter"] == 27.0


def test_branch_fork_keeps_prefix_and_separates_later_settlements():
    state, env, ledger = compensation_state(), ChargeEnvironment(), CompensationLedger()
    charge(ledger, env, state, amount=5.0)
    other, fork = copy.deepcopy(state), ledger.fork()
    charge(ledger, env, state, amount=7.0, cid="reservation-2")
    charge(fork, env, other, amount=9.0, cid="reservation-3")
    assert ledger.accounting(state, through_day=3)["incurred_compensation"] == 12.0
    assert fork.accounting(other, through_day=3)["incurred_compensation"] == 14.0


def test_unexplained_counter_or_unattempted_settlement_is_rejected():
    state = compensation_state()
    state.cumulative_reservation_compensation = 18.0
    with pytest.raises(RuntimeError, match="counter mismatch"):
        CompensationLedger().accounting(state, through_day=3)
    state.cumulative_reservation_compensation = 0.0
    state.reservation_events = [event("pickup_failed")]
    with pytest.raises(RuntimeError, match="without a charge attempt"):
        CompensationLedger().accounting(state, through_day=3)


def test_capture_does_not_record_other_states_and_restores_override_after_exception():
    state, other, env, ledger = compensation_state(), compensation_state(), ChargeEnvironment(), CompensationLedger()
    original = env._charge
    env._charge = original
    with pytest.raises(ValueError, match="test interruption"):
        with ledger.capture(env, state):
            assert env._charge(other, 18.0, reason="预约 reservation-1 减配补偿")
            raise ValueError("test interruption")
    assert env._charge is original
    assert not ledger.charges


def test_local_value_uses_settlements_not_negotiation_quotes():
    actual, baseline = compensation_state(), compensation_state()
    env = ChargeEnvironment()
    actual_ledger, baseline_ledger = CompensationLedger(), CompensationLedger()
    for state, status in ((actual, "completed"), (baseline, "pickup_failed")):
        state.reservation_commitments = [SimpleNamespace(
            commitment_id="reservation-1", status=status)]
        state.contracts, state.assets, state.rental_activity = [], [], []
    actual.reservation_events = [event("partial_fulfillment_inquired", amount=999),
                                 event("partial_fulfillment_selected", amount=999)]
    charge(actual_ledger, env, actual, amount=18.0)
    baseline.reservation_events = [event("pickup_failed", amount=25.0)]
    charge(baseline_ledger, env, baseline, amount=25.0, partial=False)
    summary = _local_contingency_value_summary(
        SimpleNamespace(), actual, [{
            "report_id": "report-1", "commitment_id": "reservation-1",
            "selection_id": "later", "application_day": 2, "original_terms": {},
            "state": baseline, "compensation_ledger": baseline_ledger,
            "continuation_action_divergences": [],
        }], actual_compensation_records=actual_ledger.accounting(actual, through_day=3)["records"],
        through_day=3,
    )
    assert summary["estimates"][0]["delta_v_local_cny"] == 7.0
    assert summary["estimates"][0]["actual_applied_contract"]["failure_compensation"] == 18.0


def test_opening_replay_distinguishes_real_reports_script_greetings_and_runoff(tmp_path):
    with sqlite3.connect(tmp_path / "run.sqlite3") as db:
        db.execute("CREATE TABLE day_starts(day INTEGER PRIMARY KEY)")
        db.execute("CREATE TABLE messages(seq INTEGER PRIMARY KEY, day INTEGER, scope TEXT, value TEXT)")
        openings = {
            1: "Good morning. Day 1 (date) Here is the opening business report:\n{}",
            2: "Day 2: fixed no-LLM follow-up policy E.",
            3: "Day 3: V45 no-LLM policy C.",
            4: "Day 4: fixed no-LLM diagnostic merchant.",
        }
        for day, content in openings.items():
            db.execute("INSERT INTO day_starts VALUES(?)", (day,))
            db.execute("INSERT INTO messages VALUES(?,?,?,?)", (
                day, day, "controller", json.dumps({"role": "user", "content": content})))
        db.execute("INSERT INTO messages VALUES(5,5,'controller',?)", (
            json.dumps({"role": "tool", "content": "runoff close"}),))
    assert _opening_observation_days(tmp_path, 5) == {1}
    with sqlite3.connect(tmp_path / "run.sqlite3") as db:
        db.execute("INSERT INTO day_starts VALUES(6)")
        db.execute("INSERT INTO messages VALUES(6,6,'controller',?)", (
            json.dumps({"role": "user", "content": "unrecognized opening"}),))
    with pytest.raises(RuntimeError, match="unrecognized recorded day opening"):
        _opening_observation_days(tmp_path, 6)


@pytest.fixture
def demand_sampler_case(monkeypatch):
    from outfitter.config import OutfitterConfig
    from outfitter.data import OutfitterDataset
    from outfitter.environment import OutfitterEnvironment
    from outfitter.runtime.ledger import empty_ledger

    root = Path(__file__).resolve().parents[1]
    config = OutfitterConfig.from_yaml(root / "configs/outfitter_v54_main.example.yaml")
    env = OutfitterEnvironment(config, OutfitterDataset.load(config.data_dir))
    # This fixture reproduces the pre-episode double-retirement regression.
    # V47's real-trip sampler is covered in test_v47_autonomous_rentals.py.
    monkeypatch.setattr(env, '_v47_autonomous_rentals', lambda: False)
    state = env.reset()
    state.day = 26
    state.current_day_ledger = empty_ledger(state.day)
    state.reservation_requests = []
    state.reservation_commitments = []
    state.reservation_calendar_blocks = []
    for row in state.assets:
        row.status = "maintenance_hold"
    target = next(row for row in state.assets if row.offer_id == "offer-048")
    target.status = "rented"
    target.used_life_days = target.service_life_limit_days
    target.used_rental_cycles = 0
    state.contracts = [RentalContract(
        contract_id="rental-000001", asset_id=target.asset_id,
        offer_id=target.offer_id, service_line=target.service_line,
        start_day=24, return_day=27, duration_days=3,
        rent_paid=15.0, deposit=0.0, damaged_on_return=False,
        reservation_id="reservation-test",
    )]
    state.next_contract_id = 2
    state.prices[target.offer_id] = 10.0
    state.rental_service_open[target.offer_id] = True
    env._capture_opening_rental_holding_basis(state)
    session = SimpleNamespace(service_key="rental:tent", units_requested=1,
                              duration_days=1, booking_lead_days=0,
                              event_address=(26, 1, 999))

    def one_customer(current):
        env._rent_from_offer(current, session, target.offer_id, 1, Counter())

    monkeypatch.setattr(env, "_simulate_day", one_customer)
    return env, state, target


def test_daily_sampler_avoids_duplicate_retirement_without_changing_real_close(demand_sampler_case):
    env, state, target = demand_sampler_case
    old_shadow, _ = _daily_no_reservation_capacity_shadow(state)
    with pytest.raises(ValueError, match="durable capital was already released"):
        env.close_business_day(old_shadow)
    baseline = copy.deepcopy(state)
    original_sampler = env._simulate_day
    with _capture_daily_walk_in_opportunity(env, state) as opportunity:
        result = env.close_business_day(state)
    baseline_result = env.close_business_day(baseline)
    assert result == baseline_result
    assert state.to_dict() == baseline.to_dict()
    assert env._simulate_day is original_sampler
    assert state.day == 27
    assert target.status == "retired"
    assert target.capital_exit_day == 27
    assert opportunity["actual_walk_in_contracts"] == 0
    assert opportunity["shadow_walk_in_contracts"] == 1
    assert opportunity["shadow_released_reservation_assets"] == 1
    assert len(state.contracts) == 1


def test_capacity_relaxation_includes_pickups_completed_at_close(demand_sampler_case, monkeypatch):
    env, state, target = demand_sampler_case
    pending = copy.deepcopy(state.contracts[0])
    state.contracts = []
    target.status = "available"

    def pickup(current):
        current.contracts.append(copy.deepcopy(pending))
        next(a for a in current.assets if a.asset_id == pending.asset_id).status = "rented"

    monkeypatch.setattr(env, "_finalize_reservation_risks", pickup)
    with _capture_daily_walk_in_opportunity(env, state) as opportunity:
        env.close_business_day(state)
    assert opportunity["actual_walk_in_contracts"] == 0
    assert opportunity["shadow_walk_in_contracts"] == 1
    assert opportunity["shadow_released_reservation_assets"] == 1


def test_same_phase_valuation_has_zero_gap_without_reservation_constraints(demand_sampler_case):
    env, state, target = demand_sampler_case
    state.contracts = []
    target.status = "available"
    target.used_life_days = 5
    baseline = copy.deepcopy(state)
    with _capture_daily_walk_in_opportunity(env, state) as opportunity:
        env.close_business_day(state)
    env.close_business_day(baseline)
    assert state.to_dict() == baseline.to_dict()
    assert opportunity["actual_walk_in_contracts"] == opportunity["shadow_walk_in_contracts"] == 1
    assert opportunity["local_walk_in_opportunity_upper_bound"] == 0.0
    assert opportunity["actual_walk_in_contract_contribution"] == opportunity["shadow_walk_in_contract_contribution"]


def test_sampler_hook_restored_if_actual_phase_raises(demand_sampler_case, monkeypatch):
    env, state, _ = demand_sampler_case

    def fail(current):
        raise RuntimeError("deliberate sampler failure")

    monkeypatch.setattr(env, "_simulate_day", fail)
    with pytest.raises(RuntimeError, match="deliberate sampler failure"):
        with _capture_daily_walk_in_opportunity(env, state):
            env._simulate_day(state)
    assert env._simulate_day is fail
