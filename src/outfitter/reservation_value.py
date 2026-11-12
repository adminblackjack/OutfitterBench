"""Offline reservation net-value analysis for completed OutfitterBench days.

The headline measure retains its historical RNVA-LB label::

    RNVA-LB = reservation contract contribution
              - reservation failure compensation
              - local capacity-relaxed walk-in opportunity

Compensation means incurred settlement obligations, including failed payment
attempts, never compensation merely proposed during negotiation.

The opportunity term is sampled at the actual walk-in demand phase, after any
same-day pickups that precede walk-ins. It removes walk-in capacity limits and
releases reservation rental/turnaround capacity, then runs only that day's
demand sampler. Both sides value new rentals at the end of that phase using
their frozen lifecycle outcomes. The shadow never runs a subsequent morning.
Daily alternatives may overlap across dates; this is a local capacity-relaxed
diagnostic, not a jointly feasible alternative trajectory or a proven bound
on every possible no-reservation policy. RNVA-LB is the retained metric label.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import OutfitterConfig
from .data import OutfitterDataset
from .environment import OutfitterEnvironment
from .models import RentalAsset, RentalContract, WorldState
from .runner import _implementation_identity
from .tool_executor import ToolExecutor


FORMAT = "outfitter-reservation-net-value-v4"
WALK_IN_OPPORTUNITY_METHOD = "same_day_demand_phase_capacity_relaxation_v1"
RETURN_EVENTS = {
    "returned_normal",
    "returned_damaged",
    "retired_service_life",
}
RESERVATION_SERVICE_STATUSES = {"cleaning", "damaged", "repairing"}
TERMINAL_COMPENSATION_EVENTS = {
    "cancelled_by_manager", "pickup_failed", "merchant_induced_customer_cancellation",
}
_COMPENSATION_CHARGE = re.compile(r"预约 (\S+) (履约补偿|减配补偿)")


def _compensation_cents(value: Any) -> int:
    amount = float(value)
    if not math.isfinite(amount) or amount < 0 or abs(amount * 100 - round(amount * 100)) > 1e-6:
        raise ValueError(f"invalid compensation amount: {value!r}")
    return round(amount * 100)


def _terminal_compensation_records(state: WorldState) -> list[dict[str, Any]]:
    """Termination settles an obligation even if its refund subsequently fails."""
    records, seen = [], set()
    for index, event in enumerate(state.reservation_events):
        if event.event_type not in TERMINAL_COMPENSATION_EVENTS:
            continue
        event_id = getattr(event, "event_id", f"event-index-{index}")
        if event_id in seen:
            raise RuntimeError(f"duplicate compensation settlement event: {event_id}")
        seen.add(event_id)
        amount = _compensation_cents(event.details.get("compensation", 0.0))
        if amount:
            records.append({
                "day": int(event.day), "commitment_id": event.commitment_id,
                "kind": event.event_type, "amount": amount / 100,
                "source": "terminal_settlement_event", "event_id": event_id,
            })
    return records


@dataclass
class CompensationLedger:
    """Analyzer-only charge observations; never added to environment state.

    Terminal events establish cancellation/failure obligations. Partial pickup
    establishes its obligation when charging begins, even if payment fails and
    no pickup_fulfilled event is subsequently emitted. Quotes establish neither.
    """

    charges: list[dict[str, Any]] = field(default_factory=list)

    def fork(self) -> "CompensationLedger":
        return copy.deepcopy(self)

    @contextmanager
    def capture(self, environment: OutfitterEnvironment, state: WorldState):
        original = environment._charge
        had_override = "_charge" in vars(environment)
        override = vars(environment).get("_charge")

        def observed(charged_state, amount, *, reason):
            match = _COMPENSATION_CHARGE.fullmatch(reason)
            record = None
            if charged_state is state and match:
                cents = _compensation_cents(amount)
                if cents:
                    record = {
                        "day": int(state.day), "commitment_id": match.group(1),
                        "kind": ("partial_fulfillment" if match.group(2) == "减配补偿"
                                 else "termination"),
                        "amount": cents / 100, "fully_paid": None,
                    }
                    self.charges.append(record)
            result = original(charged_state, amount, reason=reason)
            if record is not None:
                record["fully_paid"] = bool(result)
            return result

        environment._charge = observed
        try:
            yield
        finally:
            if had_override:
                environment._charge = override
            else:
                del environment._charge

    def execute(self, tools: ToolExecutor, state: WorldState, name: str, arguments):
        with self.capture(tools.environment, state):
            return tools.execute(state, name, arguments)

    def accounting(self, state: WorldState, *, through_day: int) -> dict[str, Any]:
        records = _terminal_compensation_records(state)
        terminal = Counter((r["commitment_id"], r["day"], _compensation_cents(r["amount"]))
                           for r in records)
        attempted = Counter()
        for charge in self.charges:
            if charge["fully_paid"] is None:
                raise RuntimeError("compensation charge did not return a payment outcome")
            key = (charge["commitment_id"], charge["day"], _compensation_cents(charge["amount"]))
            if charge["kind"] == "termination":
                attempted[key] += 1
            else:
                records.append({**charge, "source": "partial_settlement_charge"})
        if attempted - terminal:
            raise RuntimeError("compensation charge has no matching terminal settlement")
        unattempted = terminal - attempted
        if any(not state.bankrupt or day != state.day for _cid, day, _amount in unattempted):
            raise RuntimeError("unexplained terminal compensation without a charge attempt")
        charged = sum(_compensation_cents(r["amount"]) for r in self.charges)
        counter = _compensation_cents(state.cumulative_reservation_compensation)
        if charged != counter:
            raise RuntimeError(
                f"compensation counter mismatch: replay charges={charged / 100:.2f}, "
                f"environment counter={counter / 100:.2f}"
            )
        included = [r for r in records if 1 <= r["day"] <= through_day]
        return {
            "basis": "incurred_settlement_obligations",
            "through_day": through_day,
            "incurred_compensation": sum(_compensation_cents(r["amount"]) for r in included) / 100,
            "terminal_compensation": sum(_compensation_cents(r["amount"]) for r in included
                                         if r["source"] == "terminal_settlement_event") / 100,
            "partial_fulfillment_compensation": sum(_compensation_cents(r["amount"]) for r in included
                                                    if r["source"] == "partial_settlement_charge") / 100,
            "unattempted_due_to_refund_bankruptcy": sum(amount * count
                for (_cid, day, amount), count in unattempted.items() if day <= through_day) / 100,
            "charge_attempts_not_fully_paid": sum(_compensation_cents(r["amount"]) for r in self.charges
                if r["day"] <= through_day and not r["fully_paid"]) / 100,
            "excluded_after_cutoff": sum(_compensation_cents(r["amount"]) for r in records
                                         if r["day"] > through_day) / 100,
            "replay_state_counter": counter / 100,
            "replay_state_charge_attempts": charged / 100,
            "counter_reconciliation_difference": (charged - counter) / 100,
            "counter_scope": "entire replay state, including next-morning processing beyond cutoff",
            "records": included,
        }


@dataclass(frozen=True)
class ContractValue:
    rent: float
    damage_deduction: float
    cleaning_cost: float
    damage_economic_cost: float
    use_depreciation: float

    @property
    def net(self) -> float:
        return (
            self.rent
            + self.damage_deduction
            - self.cleaning_cost
            - self.damage_economic_cost
            - self.use_depreciation
        )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_environment(
    run_dir: Path,
) -> tuple[dict[str, Any], OutfitterEnvironment, WorldState]:
    manifest = _read_json(run_dir / "manifest.json")
    if manifest.get("format") != "outfitter-run-v1":
        raise ValueError(f"not an OutfitterBench run: {run_dir}")
    values = dict(manifest["config"])
    values["data_dir"] = str(
        run_dir / str(manifest.get("runtime_data", "data"))
    )
    config = OutfitterConfig.from_dict(values)
    environment = OutfitterEnvironment(
        config, OutfitterDataset.load(config.data_dir)
    )
    with sqlite3.connect((run_dir / "run.sqlite3").resolve().as_uri() + "?mode=ro",
                         uri=True, timeout=30) as connection:
        row = connection.execute("SELECT value FROM state WHERE id=1").fetchone()
    if row is None:
        raise RuntimeError(f"run state is missing: {run_dir}")
    return manifest, environment, WorldState.from_dict(json.loads(row[0]))


def _normalized_replay_result(value: Any) -> Any:
    """Remove the sole replay-irrelevant field derived from model calls."""

    if isinstance(value, dict):
        return {
            key: _normalized_replay_result(child)
            for key, child in value.items()
            if key
            not in {
                "management_service_usage_estimate",
                "management_service_cost",
            }
        }
    if isinstance(value, list):
        return [_normalized_replay_result(child) for child in value]
    return value


def _recorded_controller_no_op(recorded: dict[str, Any]) -> bool:
    """Whether the controller rejected a batch before invoking any tool."""

    data = recorded.get("data", {})
    return bool(
        recorded.get("ok") is False
        and isinstance(data, dict)
        and data.get("executed") is False
        and data.get("error") == "tool_batch_policy_violation"
    )


def _tool_calls(
    run_dir: Path, completed_day: int
) -> list[tuple[int, str, dict[str, Any], dict[str, Any]]]:
    with sqlite3.connect((run_dir / "run.sqlite3").resolve().as_uri() + "?mode=ro",
                         uri=True, timeout=30) as connection:
        rows = connection.execute(
            "SELECT day,name,arguments,result FROM tool_calls "
            "WHERE day<=? ORDER BY rowid",
            (completed_day,),
        ).fetchall()
    return [
        (int(day), str(name), json.loads(arguments), json.loads(result))
        for day, name, arguments, result in rows
    ]


def _opening_observation_days(run_dir: Path, completed_day: int) -> set[int]:
    """Recorded openings distinguish LLM reports from script-only greetings.

    Recent fixed-policy runners never call observation at opening. Consuming
    their inbox here would change business_status and invalidate exact replay.
    Runoff has no opening report either.
    """
    with sqlite3.connect((run_dir / "run.sqlite3").resolve().as_uri() + "?mode=ro",
                         uri=True, timeout=30) as connection:
        rows = connection.execute(
            "SELECT m.day,m.value FROM messages m JOIN day_starts d ON m.day=d.day "
            "WHERE m.scope='controller' AND m.day<=? ORDER BY m.seq", (completed_day,),
        ).fetchall()
    first = {}
    for day, encoded in rows:
        first.setdefault(int(day), json.loads(encoded))
    days = set()
    for day, message in first.items():
        content = message.get("content", "")
        if (message.get("role") == "user" and isinstance(content, str)
                and content.startswith(f"Good morning. Day {day} ")
                and "Here is the opening business report:\n" in content):
            days.add(day)
        elif not (message.get("role") == "user" and isinstance(content, str)
                  and re.fullmatch(
                      rf"Day {day}: (?:fixed no-LLM diagnostic merchant|"
                      r"fixed no-LLM follow-up policy [A-Z]|V45 no-LLM policy [A-Z])\.",
                      content)):
            raise RuntimeError(f"unrecognized recorded day opening: day {day}")
    return days


def _use_depreciation(asset: RentalAsset, contract: RentalContract) -> float:
    if asset.ownership_type == "supplier_trial":
        return float(contract.supplier_trial_usage_fee)
    wear = int(contract.duration_days) / float(
        max(1, asset.service_life_limit_days)
    )
    if asset.service_life_limit_cycles is not None:
        wear += 1.0 / float(max(1, asset.service_life_limit_cycles))
    return float(asset.unit_cost) * wear


def _return_activity_by_contract(
    state: WorldState,
) -> dict[str, dict[str, Any]]:
    return {
        str(row["contract_id"]): row
        for row in state.rental_activity
        if row.get("contract_id")
        and row.get("event_type") in RETURN_EVENTS
    }


def contract_value(
    environment: OutfitterEnvironment,
    state: WorldState,
    contract: RentalContract,
    *,
    return_activity: dict[str, Any] | None = None,
) -> ContractValue:
    """Economic contribution of one rental, excluding common overhead.

    A completed contract uses its recorded return settlement.  An unsettled
    contract uses the deterministic lifecycle outcome already frozen on the
    contract, so a short run is not rewarded merely because cleaning or damage
    settlement falls just beyond the current completed-day cutoff.
    """

    assets = {asset.asset_id: asset for asset in state.assets}
    asset = assets[contract.asset_id]
    depreciation = _use_depreciation(asset, contract)
    deduction = 0.0
    cleaning = 0.0
    damage_cost = 0.0

    if return_activity is not None:
        event_type = str(return_activity.get("event_type", ""))
        if event_type == "returned_normal":
            cleaning = float(
                return_activity.get("scheduled_cleaning_cost", 0.0)
            )
        elif event_type == "returned_damaged":
            deduction = float(return_activity.get("damage_deduction", 0.0))
            if "repair_cash_cost" in return_activity:
                damage_cost = float(return_activity["repair_cash_cost"])
            else:
                damage_cost = float(environment.repair_cost(asset))
    elif not environment._service_life_exhausted(asset):
        if contract.damaged_on_return:
            if environment.config.lifecycle_economics_enabled:
                terms = environment._lifecycle_return_damage_terms(
                    asset, contract
                )
                deduction = float(terms["deposit_deduction"])
                damage_cost = float(terms["repair_cash_cost"])
            else:
                deduction = min(
                    float(contract.deposit),
                    float(environment.damage_deduction(asset)),
                )
                damage_cost = float(environment.repair_cost(asset))
        elif environment.config.lifecycle_economics_enabled:
            cleaning = float(
                environment._lifecycle_cleaning_terms(
                    asset,
                    duration_days=contract.duration_days,
                    start_day=contract.start_day,
                    return_day=contract.return_day,
                )["cash_cost"]
            )
        else:
            cleaning = float(
                environment.cleaning_cost(asset, contract.duration_days)
            ) * float(
                environment._return_weather_cleaning_multiplier(
                    contract.start_day, contract.return_day
                )
            )

    return ContractValue(
        rent=float(contract.rent_paid),
        damage_deduction=deduction,
        cleaning_cost=cleaning,
        damage_economic_cost=damage_cost,
        use_depreciation=depreciation,
    )


def _reservation_caused_service_assets(state: WorldState) -> set[str]:
    """Assets currently unavailable because of a reservation lifecycle."""

    reservation_contracts = {
        contract.contract_id: contract
        for contract in state.contracts
        if contract.reservation_id is not None
    }
    affected = {
        contract.asset_id
        for contract in reservation_contracts.values()
        if contract.status == "active"
    }
    latest_return: dict[str, dict[str, Any]] = {}
    for row in state.rental_activity:
        if row.get("event_type") not in RETURN_EVENTS:
            continue
        latest_return[str(row.get("asset_id"))] = row
    assets = {asset.asset_id: asset for asset in state.assets}
    for asset_id, row in latest_return.items():
        contract = reservation_contracts.get(str(row.get("contract_id")))
        asset = assets.get(asset_id)
        if (
            contract is not None
            and asset is not None
            and asset.status in RESERVATION_SERVICE_STATUSES
        ):
            affected.add(asset_id)
    return affected


def _daily_no_reservation_capacity_shadow(
    state: WorldState,
) -> tuple[WorldState, set[str]]:
    """Capacity-relaxed sampler input, never a valid lifecycle continuation.

    Historical contracts and service jobs remain for valuation context. They
    must not be processed again by close_business_day or morning settlement.
    """
    shadow = copy.deepcopy(state)
    shadow.walk_in_capacity_limits = []
    released = _reservation_caused_service_assets(shadow)
    for asset in shadow.assets:
        if asset.asset_id in released:
            asset.status = "available"
            asset.ready_day = None
    return shadow, released


def _new_walk_in_contracts(
    state: WorldState, before_ids: set[str], day: int
) -> list[RentalContract]:
    return [
        contract
        for contract in state.contracts
        if contract.contract_id not in before_ids
        and contract.reservation_id is None
        and int(contract.start_day) == day
    ]


def _contract_set_value(
    environment: OutfitterEnvironment,
    state: WorldState,
    contracts: Iterable[RentalContract],
) -> ContractValue:
    activity = _return_activity_by_contract(state)
    values = [
        contract_value(
            environment,
            state,
            contract,
            return_activity=activity.get(contract.contract_id),
        )
        for contract in contracts
    ]
    return ContractValue(
        rent=sum(row.rent for row in values),
        damage_deduction=sum(row.damage_deduction for row in values),
        cleaning_cost=sum(row.cleaning_cost for row in values),
        damage_economic_cost=sum(row.damage_economic_cost for row in values),
        use_depreciation=sum(row.use_depreciation for row in values),
    )


@contextmanager
def _capture_daily_walk_in_opportunity(environment: OutfitterEnvironment, state: WorldState):
    """Observe real demand and sample its relaxed counterpart at the same phase.

    The actual close still runs normally and is checked against the transcript.
    The shadow calls the original demand method directly, never recursively
    closing the shop, fulfilling pending reservations, or settling returns.
    """
    original = environment._simulate_day
    had_override = "_simulate_day" in vars(environment)
    override = vars(environment).get("_simulate_day")
    result: dict[str, Any] = {}

    def observed(current):
        if current is not state:
            return original(current)
        if result:
            raise RuntimeError("walk-in demand phase was executed more than once")
        day = int(current.day)
        before_ids = {contract.contract_id for contract in current.contracts}
        shadow, released = _daily_no_reservation_capacity_shadow(current)
        actual_return = original(current)
        actual_contracts = _new_walk_in_contracts(current, before_ids, day)
        actual_value = _contract_set_value(environment, current, actual_contracts)
        history_length = len(shadow.daily_history)
        original(shadow)
        if shadow.day != day or len(shadow.daily_history) != history_length:
            raise RuntimeError("daily capacity sampler unexpectedly advanced the lifecycle")
        shadow_contracts = _new_walk_in_contracts(shadow, before_ids, day)
        shadow_value = _contract_set_value(environment, shadow, shadow_contracts)
        result.update(
            actual_walk_in_contracts=len(actual_contracts),
            shadow_walk_in_contracts=len(shadow_contracts),
            shadow_released_reservation_assets=len(released),
            actual_walk_in_contract_contribution=actual_value.net,
            shadow_walk_in_contract_contribution=shadow_value.net,
            local_walk_in_opportunity_upper_bound=max(0.0, shadow_value.net - actual_value.net),
        )
        return actual_return

    environment._simulate_day = observed
    try:
        yield result
    finally:
        if had_override:
            environment._simulate_day = override
        else:
            del environment._simulate_day


_LOCAL_VALUE_TERMINAL_STATUSES = {
    "completed",
    "cancelled_by_customer",
    "cancelled_by_manager",
    "merchant_induced_customer_cancellation",
    "pickup_failed",
}


def _local_commitment_value(
    environment: OutfitterEnvironment,
    state: WorldState,
    commitment_id: str,
    *,
    compensation_records: list[dict[str, Any]],
) -> dict[str, Any]:
    commitment = next(
        (
            row
            for row in state.reservation_commitments
            if row.commitment_id == commitment_id
        ),
        None,
    )
    if commitment is None:
        return {
            "mature": False,
            "status": "missing_commitment",
            "contract_contribution": None,
            "failure_compensation": None,
            "net_local_value": None,
        }
    mature = commitment.status in _LOCAL_VALUE_TERMINAL_STATUSES
    contracts = [
        contract
        for contract in state.contracts
        if contract.reservation_id == commitment_id
    ]
    value = _contract_set_value(environment, state, contracts)
    compensation = sum(
        _compensation_cents(row["amount"])
        for row in compensation_records
        if row["commitment_id"] == commitment_id
    ) / 100
    return {
        "mature": mature,
        "status": commitment.status,
        "contract_count": len(contracts),
        "reservation_rent": round(value.rent, 6),
        "damage_deductions": round(value.damage_deduction, 6),
        "cleaning_cost": round(value.cleaning_cost, 6),
        "damage_economic_cost": round(value.damage_economic_cost, 6),
        "use_depreciation": round(value.use_depreciation, 6),
        "contract_contribution": round(value.net, 6),
        "failure_compensation": round(compensation, 6),
        "net_local_value": round(value.net - compensation, 6),
    }


def _local_contingency_value_summary(
    environment: OutfitterEnvironment,
    actual_state: WorldState,
    shadows: list[dict[str, Any]],
    *,
    actual_compensation_records: list[dict[str, Any]],
    through_day: int,
) -> dict[str, Any]:
    estimates = []
    for shadow in shadows:
        actual = _local_commitment_value(
            environment, actual_state, shadow["commitment_id"],
            compensation_records=actual_compensation_records,
        )
        shadow_accounting = shadow["compensation_ledger"].accounting(
            shadow["state"], through_day=through_day)
        keep_original = _local_commitment_value(
            environment, shadow["state"], shadow["commitment_id"],
            compensation_records=shadow_accounting["records"],
        )
        mature = bool(actual["mature"] and keep_original["mature"])
        delta = (
            round(
                float(actual["net_local_value"])
                - float(keep_original["net_local_value"]),
                6,
            )
            if mature
            else None
        )
        divergences = list(shadow["continuation_action_divergences"])
        estimates.append(
            {
                "report_id": shadow["report_id"],
                "commitment_id": shadow["commitment_id"],
                "selection_id": shadow["selection_id"],
                "application_day": shadow["application_day"],
                "original_terms": shadow["original_terms"],
                "actual_applied_contract": actual,
                "keep_original_contract": keep_original,
                "delta_v_local_cny": delta,
                "mature": mature,
                "continuation_action_divergence_count": len(divergences),
                "continuation_action_divergences": divergences,
                "estimate_quality": (
                    "clean_frozen_continuation"
                    if mature and not divergences
                    else "frozen_continuation_with_action_divergence"
                    if mature
                    else "not_yet_mature"
                ),
            }
        )
    mature = [row for row in estimates if row["mature"]]
    clean = [
        row
        for row in mature
        if row["estimate_quality"] == "clean_frozen_continuation"
    ]
    return {
        "name": "local_contingency_value",
        "formula": (
            "V(actual_applied_contract) - V(keep_pre_application_contract)"
        ),
        "application_count": len(estimates),
        "mature_estimate_count": len(mature),
        "clean_mature_estimate_count": len(clean),
        "sum_delta_v_local_cny_all_mature": round(
            sum(float(row["delta_v_local_cny"]) for row in mature), 6
        ),
        "sum_delta_v_local_cny_clean_mature": round(
            sum(float(row["delta_v_local_cny"]) for row in clean), 6
        ),
        "estimates": estimates,
        "included_in_rnva": False,
        "compensation_basis": "incurred_settlement_obligations",
        "scope": (
            "Per applied contingency only. Both branches replay the same later "
            "recorded state-changing actions and exogenous random tape; the baseline "
            "omits only the selected application and retains its pre-application "
            "contract. Value includes that commitment's realized contract contribution "
            "and failure compensation, but excludes cross-contract and walk-in spillovers."
        ),
        "selection_limit": (
            "This is treatment-on-the-treated: it does not value missed inquiries "
            "or unapplied customer responses."
        ),
    }


def _empty_day(day: int) -> dict[str, Any]:
    return {
        "day": day,
        "reservation_contracts_started": 0,
        "reservation_rent": 0.0,
        "reservation_damage_deductions": 0.0,
        "reservation_cleaning_cost": 0.0,
        "reservation_damage_economic_cost": 0.0,
        "reservation_use_depreciation": 0.0,
        "reservation_failure_compensation": 0.0,
        "actual_walk_in_contracts": 0,
        "shadow_walk_in_contracts": 0,
        "shadow_released_reservation_assets": 0,
        "actual_walk_in_contract_contribution": 0.0,
        "shadow_walk_in_contract_contribution": 0.0,
        "local_walk_in_opportunity_upper_bound": 0.0,
    }


def _finalize_daily_rows(
    rows: dict[int, dict[str, Any]],
    *,
    initial_official_value: float,
) -> list[dict[str, Any]]:
    cumulative = defaultdict(float)
    output: list[dict[str, Any]] = []
    for day in sorted(rows):
        row = dict(rows[day])
        row["reservation_contract_contribution"] = (
            float(row["reservation_rent"])
            + float(row["reservation_damage_deductions"])
            - float(row["reservation_cleaning_cost"])
            - float(row["reservation_damage_economic_cost"])
            - float(row["reservation_use_depreciation"])
        )
        row["reservation_direct_net_contribution"] = (
            row["reservation_contract_contribution"]
            - float(row["reservation_failure_compensation"])
        )
        row["rnva_lower_bound"] = (
            row["reservation_direct_net_contribution"]
            - float(row["local_walk_in_opportunity_upper_bound"])
        )
        for key in (
            "reservation_contract_contribution",
            "reservation_failure_compensation",
            "reservation_direct_net_contribution",
            "local_walk_in_opportunity_upper_bound",
            "rnva_lower_bound",
        ):
            cumulative[key] += float(row[key])
            row[f"cumulative_{key}"] = round(cumulative[key], 6)
        row["cumulative_rnva_lower_bound_pct_initial_official_value"] = round(
            100.0
            * cumulative["rnva_lower_bound"]
            / initial_official_value,
            6,
        )
        for key, value in list(row.items()):
            if isinstance(value, float):
                row[key] = round(value, 6)
        output.append(row)
    return output


def _contingency_diagnostics(
    state: WorldState, *, completed_day: int,
    compensation_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Describe V38 negotiation behavior without turning actions into rewards."""

    inquiries = [
        row
        for row in state.reservation_contingency_inquiries
        if int(row.submitted_day) <= completed_day
    ]
    reports = [
        row
        for row in state.reservation_contingency_reports
        if int(row.response_day) <= completed_day
    ]
    report_by_id = {row.report_id: row for row in reports}
    daily = {
        day: {
            "day": day,
            "inquiries": 0,
            "reports": 0,
            "applications": 0,
            "merchant_induced_customer_cancellations": 0,
            "signed_pickup_shift_days": 0,
            "absolute_pickup_shift_days": 0,
            "equipment_line_changes": 0,
        }
        for day in range(1, completed_day + 1)
    }
    for inquiry in inquiries:
        daily[int(inquiry.submitted_day)]["inquiries"] += 1
    for report in reports:
        daily[int(report.response_day)]["reports"] += 1
        if report.outcome == "merchant_induced_customer_cancellation":
            daily[int(report.response_day)][
                "merchant_induced_customer_cancellations"
            ] += 1

    application_type_counts: defaultdict[str, int] = defaultdict(int)
    applied_pickup_shifts: list[int] = []
    equipment_line_changes = 0
    application_events = [
        event
        for event in state.reservation_events
        if event.event_type == "reservation_contingency_applied"
        and int(event.day) <= completed_day
    ]
    for event in application_events:
        report = report_by_id.get(str(event.details.get("report_id", "")))
        selection_id = str(event.details.get("selection_id", ""))
        selected: dict[str, Any] | None = None
        if report is not None:
            selected = next(
                (
                    proposal
                    for proposal in report.proposals
                    if str(proposal.get("proposal_id")) == selection_id
                ),
                None,
            )
            if (
                selected is None
                and report.counteroffer is not None
                and str(report.counteroffer.get("proposal_id")) == selection_id
            ):
                selected = report.counteroffer
        previous = dict(event.details.get("previous_terms") or {})
        old_pickup = previous.get("pickup_day")
        new_pickup = event.details.get("new_pickup_day")
        shift = (
            int(new_pickup) - int(old_pickup)
            if old_pickup is not None and new_pickup is not None
            else 0
        )
        changes = (
            list(selected.get("line_changes", []))
            if selected is not None
            else []
        )
        changed_date = shift != 0
        changed_equipment = bool(changes)
        application_type = (
            "date_and_equipment"
            if changed_date and changed_equipment
            else "date_only"
            if changed_date
            else "equipment_only"
            if changed_equipment
            else "no_material_change"
        )
        application_type_counts[application_type] += 1
        applied_pickup_shifts.append(shift)
        equipment_line_changes += len(changes)
        row = daily[int(event.day)]
        row["applications"] += 1
        row["signed_pickup_shift_days"] += shift
        row["absolute_pickup_shift_days"] += abs(shift)
        row["equipment_line_changes"] += len(changes)

    outcome_counts: defaultdict[str, int] = defaultdict(int)
    status_counts: defaultdict[str, int] = defaultdict(int)
    for report in reports:
        outcome_counts[str(report.outcome)] += 1
        status_counts[str(report.status)] += 1
    cancellations = [
        report
        for report in reports
        if report.outcome == "merchant_induced_customer_cancellation"
    ]
    confirmed = sum(
        commitment.confirmed_day <= completed_day
        for commitment in state.reservation_commitments
    )
    if compensation_records is None:
        compensation_records = _terminal_compensation_records(state)
    return {
        "enabled": bool(
            inquiries
            or reports
            or getattr(state, "cumulative_reservation_contingency_inquiries", 0)
        ),
        "inquiries": len(inquiries),
        "reports": len(reports),
        "applications": len(application_events),
        "inquiry_rate_per_confirmed_commitment": (
            round(len(inquiries) / confirmed, 6) if confirmed else None
        ),
        "accepted_proposal_count": sum(
            len(report.accepted_proposal_ids) for report in reports
        ),
        "counteroffer_count": sum(
            report.counteroffer is not None for report in reports
        ),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "application_type_counts": dict(sorted(application_type_counts.items())),
        "applied_pickup_shift_days": applied_pickup_shifts,
        "signed_pickup_shift_days": sum(applied_pickup_shifts),
        "absolute_pickup_shift_days": sum(
            abs(value) for value in applied_pickup_shifts
        ),
        "equipment_line_changes": equipment_line_changes,
        "merchant_induced_customer_cancellations": len(cancellations),
        "merchant_induced_prepayment_refunds": round(
            sum(
                float(report.settlement.get("prepayment_refund", 0.0))
                for report in cancellations
            ),
            6,
        ),
        "merchant_induced_compensation": round(
            sum(
                float(row["amount"])
                for row in compensation_records
                if row["kind"] == "merchant_induced_customer_cancellation"
                and 1 <= row["day"] <= completed_day
            ),
            6,
        ),
        "commitment_version_distribution": dict(
            sorted(
                Counter(
                    int(commitment.commitment_version)
                    for commitment in state.reservation_commitments
                ).items()
            )
        ),
        "daily": list(daily.values()),
        "scoring_role": (
            "diagnostic only; inquiry/report/application counts are not reward terms"
        ),
    }


def analyze_run(
    run_dir: str | Path, *, through_day: int | None = None
) -> dict[str, Any]:
    path = Path(run_dir).expanduser().resolve()
    manifest, environment, persisted = _load_environment(path)
    recorded_identity = manifest.get("controller", {}).get(
        "implementation_identity"
    )
    replay_identity = _implementation_identity()
    if recorded_identity is not None and recorded_identity != replay_identity:
        raise RuntimeError(
            "trajectory implementation identity differs from replay source; "
            "run the analyzer with the matching frozen --source-root"
        )
    available_completed_day = len(persisted.daily_history)
    completed_day = (
        available_completed_day
        if through_day is None
        else min(int(through_day), available_completed_day)
    )
    if completed_day < 1:
        raise ValueError(f"run has no completed day: {path}")
    calls = _tool_calls(path, completed_day)
    observation_days = _opening_observation_days(path, completed_day)
    if not calls:
        raise ValueError(f"run has no replayable tool calls: {path}")

    initial = environment.reset()
    initial_official_value = float(
        environment.business_value(initial)[
            "contract_adjusted_economic_value"
        ]
    )
    state = initial
    tools = ToolExecutor(environment)
    tool_effects = tools.effect_map()
    seen_days: set[int] = set()
    shadow_rows: dict[int, dict[str, Any]] = {
        day: _empty_day(day) for day in range(1, completed_day + 1)
    }
    replay_mismatches: list[dict[str, Any]] = []
    local_contingency_shadows: list[dict[str, Any]] = []
    compensation_ledger = CompensationLedger()

    for day, name, arguments, recorded in calls:
        if day not in seen_days:
            if day in observation_days:
                environment.observation(state, consume_inbox=True)
            seen_days.add(day)
        if _recorded_controller_no_op(recorded):
            # Oversized read-only batches are rejected by the runner before
            # ToolExecutor is entered. Re-executing their constituent calls
            # would create observations that never occurred in the run.
            continue
        if tool_effects.get(name) != "read_only":
            for local in local_contingency_shadows:
                local_result = local["compensation_ledger"].execute(
                    tools, local["state"], name, arguments)
                if bool(local_result.ok) != bool(recorded.get("ok")):
                    local["continuation_action_divergences"].append(
                        {
                            "day": day,
                            "tool": name,
                            "actual_ok": bool(recorded.get("ok")),
                            "keep_original_ok": bool(local_result.ok),
                            "keep_original_error": (
                                local_result.data.get("error")
                                if isinstance(local_result.data, dict)
                                else None
                            ),
                        }
                    )
        pre_application_state = (
            copy.deepcopy(state)
            if name == "apply_reservation_contingency"
            and recorded.get("ok") is True
            else None
        )
        pre_application_ledger = (
            compensation_ledger.fork() if pre_application_state is not None else None
        )
        daily_opportunity = None
        if name == "close_business_day" and recorded.get("ok") is True:
            with _capture_daily_walk_in_opportunity(environment, state) as daily_opportunity:
                actual_result = compensation_ledger.execute(tools, state, name, arguments)
            if actual_result.ok and not daily_opportunity:
                raise RuntimeError(
                    f"walk-in phase was not reached on day {day}; "
                    "RNVA opportunity cost is unavailable, not zero"
                )
        else:
            actual_result = compensation_ledger.execute(tools, state, name, arguments)
        replayed = {
            "ok": actual_result.ok,
            "data": actual_result.data,
            "terminal_day": actual_result.terminal_day,
        }
        if _normalized_replay_result(replayed) != _normalized_replay_result(
            recorded
        ):
            replay_mismatches.append({"day": day, "tool": name})
            if len(replay_mismatches) >= 5:
                break

        if (
            name == "apply_reservation_contingency"
            and actual_result.ok
            and pre_application_state is not None
        ):
            decisions = list(arguments.get("decisions", []))
            applications = list(actual_result.data.get("applications", []))
            decision_by_report = {
                str(row.get("report_id")): row for row in decisions
            }
            commitments_before = {
                row.commitment_id: row
                for row in pre_application_state.reservation_commitments
            }
            for application in applications:
                report_id = str(application["report_id"])
                commitment_id = str(application["commitment_id"])
                selected_decision = decision_by_report[report_id]
                baseline = copy.deepcopy(pre_application_state)
                assert pre_application_ledger is not None
                baseline_ledger = pre_application_ledger.fork()
                other_decisions = [
                    copy.deepcopy(row)
                    for row in decisions
                    if str(row.get("report_id")) != report_id
                ]
                divergences: list[dict[str, Any]] = []
                if other_decisions:
                    baseline_result = baseline_ledger.execute(
                        tools,
                        baseline,
                        name,
                        {"decisions": other_decisions},
                    )
                    if not baseline_result.ok:
                        divergences.append(
                            {
                                "day": day,
                                "tool": name,
                                "actual_ok": True,
                                "keep_original_ok": False,
                                "keep_original_error": baseline_result.data.get(
                                    "error"
                                ),
                                "phase": "counterfactual_initialization",
                            }
                        )
                original = commitments_before[commitment_id]
                local_contingency_shadows.append(
                    {
                        "report_id": report_id,
                        "commitment_id": commitment_id,
                        "selection_id": str(selected_decision["selection_id"]),
                        "application_day": day,
                        "original_terms": {
                            "commitment_version": int(
                                original.commitment_version
                            ),
                            "pickup_day": int(original.pickup_day),
                            "return_day": int(original.return_day),
                            "line_offer_ids": {
                                line.line_id: line.selected_offer_id or line.offer_id
                                for line in original.lines
                            },
                        },
                        "state": baseline,
                        "compensation_ledger": baseline_ledger,
                        "continuation_action_divergences": divergences,
                    }
                )

        if daily_opportunity is not None:
            shadow_rows[day].update(daily_opportunity)

    if replay_mismatches:
        raise RuntimeError(
            "recorded trajectory does not replay under the current source: "
            + json.dumps(replay_mismatches, ensure_ascii=False)
        )
    if len(state.daily_history) != completed_day:
        raise RuntimeError(
            f"replay completed {len(state.daily_history)} days, expected "
            f"{completed_day}"
        )

    returns = _return_activity_by_contract(state)
    reservation_contracts = [
        contract
        for contract in state.contracts
        if contract.reservation_id is not None
        and int(contract.start_day) <= completed_day
    ]
    for contract in reservation_contracts:
        value = contract_value(
            environment,
            state,
            contract,
            return_activity=returns.get(contract.contract_id),
        )
        row = shadow_rows[int(contract.start_day)]
        row["reservation_contracts_started"] += 1
        row["reservation_rent"] += value.rent
        row["reservation_damage_deductions"] += value.damage_deduction
        row["reservation_cleaning_cost"] += value.cleaning_cost
        row["reservation_damage_economic_cost"] += value.damage_economic_cost
        row["reservation_use_depreciation"] += value.use_depreciation

    compensation_accounting = compensation_ledger.accounting(state, through_day=completed_day)
    for record in compensation_accounting["records"]:
        shadow_rows[record["day"]]["reservation_failure_compensation"] += record["amount"]

    daily = _finalize_daily_rows(
        shadow_rows, initial_official_value=initial_official_value
    )
    last = daily[-1]
    budget = manifest.get("config", {}).get("durable_acquisition_budget")
    contingency = _contingency_diagnostics(
        state, completed_day=completed_day,
        compensation_records=compensation_accounting["records"],
    )
    local_contingency_value = _local_contingency_value_summary(
        environment, state, local_contingency_shadows,
        actual_compensation_records=compensation_accounting["records"],
        through_day=completed_day,
    )
    return {
        "format": FORMAT,
        "run_id": str(manifest.get("run_id", path.name)),
        "run_dir": str(path),
        "completed_day": completed_day,
        "available_completed_day": available_completed_day,
        "seed": manifest.get("config", {}).get("seed"),
        "durable_acquisition_budget": budget,
        "supplier_trial_exploration_cost": {
            "handling_fees": round(
                float(state.cumulative_supplier_trial_handling_fees), 6
            ),
            "usage_fees_in_contract_use_cost": round(
                sum(
                    float(contract.supplier_trial_usage_fee)
                    for contract in state.contracts
                    if contract.supplier_trial_id is not None
                ),
                6,
            ),
            "handling_fees_in_rnva": False,
        },
        "proactive_contingency": contingency,
        "local_contingency_value": local_contingency_value,
        "compensation_accounting": compensation_accounting,
        "initial_official_value": round(initial_official_value, 6),
        "replay_source_identity": replay_identity,
        "metric": {
            "name": "reservation_net_value_added_lower_bound",
            "abbreviation": "RNVA-LB",
            "compensation_basis": "incurred_settlement_obligations",
            "walk_in_opportunity_method": WALK_IN_OPPORTUNITY_METHOD,
            "cny": last["cumulative_rnva_lower_bound"],
            "pct_initial_official_value": last[
                "cumulative_rnva_lower_bound_pct_initial_official_value"
            ],
            "reservation_contract_contribution": last[
                "cumulative_reservation_contract_contribution"
            ],
            "reservation_failure_compensation": last[
                "cumulative_reservation_failure_compensation"
            ],
            "reservation_direct_net_contribution": last[
                "cumulative_reservation_direct_net_contribution"
            ],
            "local_walk_in_opportunity_upper_bound": last[
                "cumulative_local_walk_in_opportunity_upper_bound"
            ],
        },
        "counts": {
            "reservation_contracts_started": sum(
                int(row["reservation_contracts_started"]) for row in daily
            ),
            "days_with_positive_opportunity": sum(
                float(row["local_walk_in_opportunity_upper_bound"]) > 0
                for row in daily
            ),
            "replay_tool_calls": len(calls),
            "replay_mismatches": 0,
        },
        "daily": daily,
        "interpretation": {
            "costs_included": [
                "rental use depreciation",
                "cleaning",
                "damage economic repair cost net of damage deductions",
                "incurred reservation termination and partial-fulfillment compensation",
                "local capacity-relaxed walk-in opportunity",
            ],
            "costs_excluded": [
                "common daily overhead",
                "model cost",
                "general fleet holding cost",
                "durable acquisition cash outlay beyond use depreciation",
                "compensation quoted in inquiries or selected proposals before settlement",
            ],
            "bound_note": (
                "Daily capacity-relaxed demand samples are not carried forward. "
                "Alternatives may overlap across days. The retained RNVA-LB label "
                "does not establish a bound on all feasible no-reservation policies."
            ),
            "walk_in_opportunity_scope": (
                "Sampled at the actual walk-in phase, after preceding due pickups. "
                "Same posted prices, offer openings, customer routing and random tape; "
                "walk-in limits and reservation rental/turnaround unavailability are relaxed. "
                "Only the demand sampler runs in the shadow; no further reservation pickup, "
                "day close, morning return, service completion or capital release executes. "
                "New actual and shadow rentals are valued before next-morning processing. "
                "V4 results must not be mixed with earlier shadow-method results."
            ),
            "v38_contingency_scope": (
                "Realized proactive-contingency effects enter through the final "
                "contract dates/equipment, reservation rent, failure settlement, "
                "and realized capacity use. Inquiry, acceptance, patch, or version "
                "counts receive no direct reward. RNVA-LB is not a causal estimate "
                "of proactive negotiation versus an otherwise identical no-negotiation policy."
            ),
        },
    }


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    fields: list[str] = list(fieldnames or [])
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _label(result: dict[str, Any]) -> str:
    budget = result.get("durable_acquisition_budget")
    return "B=uncapped" if budget is None else f"B={float(budget):g}"


def render_plot(results: list[dict[str, Any]], path: Path) -> None:
    ordered = sorted(
        results,
        key=lambda row: float("inf")
        if row.get("durable_acquisition_budget") is None
        else float(row["durable_acquisition_budget"]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    colors = plt.get_cmap("tab10")
    for index, result in enumerate(ordered):
        daily = result["daily"]
        axes[0].plot(
            [row["day"] for row in daily],
            [
                row[
                    "cumulative_rnva_lower_bound_pct_initial_official_value"
                ]
                for row in daily
            ],
            label=_label(result),
            linewidth=2.0,
            color=colors(index),
        )
    axes[0].axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axes[0].set_title("Cumulative reservation net value (RNVA-LB)")
    axes[0].set_xlabel("Completed day")
    axes[0].set_ylabel("% of initial official value")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    labels = [_label(result) for result in ordered]
    direct = [
        float(result["metric"]["reservation_direct_net_contribution"])
        for result in ordered
    ]
    opportunity = [
        -float(result["metric"]["local_walk_in_opportunity_upper_bound"])
        for result in ordered
    ]
    x = list(range(len(ordered)))
    axes[1].bar(x, direct, label="Direct reservation contribution", alpha=0.9)
    axes[1].bar(
        x,
        opportunity,
        label="Local capacity-relaxed walk-in opportunity",
        alpha=0.9,
    )
    for index, result in enumerate(ordered):
        value = float(result["metric"]["cny"])
        axes[1].scatter(index, value, color="black", marker="D", zorder=4)
        axes[1].annotate(
            f"RNVA-LB {value:.0f}",
            (index, value),
            xytext=(0, 8 if value >= 0 else -15),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Endpoint decomposition")
    axes[1].set_ylabel("CNY")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Reservation Net Value Added — local capacity diagnostic",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_local_contingency_plot(
    results: list[dict[str, Any]], path: Path
) -> None:
    estimates = [
        {"run_label": _label(result), **estimate}
        for result in sorted(
            results,
            key=lambda row: float("inf")
            if row.get("durable_acquisition_budget") is None
            else float(row["durable_acquisition_budget"]),
        )
        for estimate in result["local_contingency_value"]["estimates"]
        if estimate["mature"]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    if not estimates:
        for axis in axes:
            axis.axis("off")
        axes[0].text(
            0.5,
            0.5,
            "No applied contingency has matured in both branches",
            ha="center",
            va="center",
            fontsize=13,
        )
    else:
        labels = [
            f"{row['run_label']}\n{row['report_id']}" for row in estimates
        ]
        delta = [float(row["delta_v_local_cny"]) for row in estimates]
        colors = [
            "#54a24b"
            if row["estimate_quality"] == "clean_frozen_continuation"
            else "#f2cf5b"
            for row in estimates
        ]
        axes[0].bar(labels, delta, color=colors)
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set(
            title=r"Local contingency value $\Delta V_i^{local}$",
            ylabel="CNY: applied contract minus keep-original",
        )
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].grid(axis="y", alpha=0.25)

        x = list(range(len(estimates)))
        width = 0.36
        actual = [
            float(row["actual_applied_contract"]["net_local_value"])
            for row in estimates
        ]
        original = [
            float(row["keep_original_contract"]["net_local_value"])
            for row in estimates
        ]
        axes[1].bar(
            [value - width / 2 for value in x],
            actual,
            width=width,
            label="Actual applied contract",
            color="#4c78a8",
        )
        axes[1].bar(
            [value + width / 2 for value in x],
            original,
            width=width,
            label="Keep original contract",
            color="#b279a2",
        )
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_xticks(x, labels, rotation=20)
        axes[1].set(title="Local branch values", ylabel="CNY")
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend(fontsize=8)
    fig.suptitle(
        "Applied-contingency local value — separate from RNVA",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_outputs(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(results, key=lambda row: str(row["run_id"]))
    (output_dir / "reservation_value_report.json").write_text(
        json.dumps(
            {"format": FORMAT, "runs": ordered},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_rows = [
        {
            "run_id": row["run_id"],
            "completed_day": row["completed_day"],
            "seed": row["seed"],
            "durable_acquisition_budget": row[
                "durable_acquisition_budget"
            ],
            "initial_official_value": row["initial_official_value"],
            **row["metric"],
            **row["counts"],
            "contingency_inquiries": row["proactive_contingency"]["inquiries"],
            "contingency_applications": row["proactive_contingency"]["applications"],
            "contingency_customer_cancellations": row[
                "proactive_contingency"
            ]["merchant_induced_customer_cancellations"],
            "contingency_compensation": row["proactive_contingency"][
                "merchant_induced_compensation"
            ],
            "local_contingency_application_count": row[
                "local_contingency_value"
            ]["application_count"],
            "local_contingency_mature_estimate_count": row[
                "local_contingency_value"
            ]["mature_estimate_count"],
            "sum_delta_v_local_cny_all_mature": row[
                "local_contingency_value"
            ]["sum_delta_v_local_cny_all_mature"],
            "sum_delta_v_local_cny_clean_mature": row[
                "local_contingency_value"
            ]["sum_delta_v_local_cny_clean_mature"],
        }
        for row in ordered
    ]
    _write_csv(output_dir / "reservation_value_summary.csv", summary_rows)
    daily_rows = [
        {
            "run_id": result["run_id"],
            "durable_acquisition_budget": result[
                "durable_acquisition_budget"
            ],
            **row,
        }
        for result in ordered
        for row in result["daily"]
    ]
    _write_csv(output_dir / "reservation_value_daily.csv", daily_rows)
    local_rows = [
        {
            "run_id": result["run_id"],
            "durable_acquisition_budget": result[
                "durable_acquisition_budget"
            ],
            "report_id": estimate["report_id"],
            "commitment_id": estimate["commitment_id"],
            "selection_id": estimate["selection_id"],
            "application_day": estimate["application_day"],
            "mature": estimate["mature"],
            "estimate_quality": estimate["estimate_quality"],
            "continuation_action_divergence_count": estimate[
                "continuation_action_divergence_count"
            ],
            "actual_status": estimate["actual_applied_contract"]["status"],
            "actual_net_local_value": estimate["actual_applied_contract"][
                "net_local_value"
            ],
            "keep_original_status": estimate["keep_original_contract"]["status"],
            "keep_original_net_local_value": estimate[
                "keep_original_contract"
            ]["net_local_value"],
            "delta_v_local_cny": estimate["delta_v_local_cny"],
        }
        for result in ordered
        for estimate in result["local_contingency_value"]["estimates"]
    ]
    _write_csv(
        output_dir / "reservation_local_contingency_value.csv",
        local_rows,
        fieldnames=[
            "run_id",
            "durable_acquisition_budget",
            "report_id",
            "commitment_id",
            "selection_id",
            "application_day",
            "mature",
            "estimate_quality",
            "continuation_action_divergence_count",
            "actual_status",
            "actual_net_local_value",
            "keep_original_status",
            "keep_original_net_local_value",
            "delta_v_local_cny",
        ],
    )
    render_plot(ordered, output_dir / "reservation_net_value_added.png")
    render_local_contingency_plot(
        ordered, output_dir / "reservation_local_contingency_value.png"
    )
    lines = [
        "# Reservation Net Value Added (RNVA-LB)",
        "",
        "The headline metric retains the historical RNVA-LB name and field names:",
        "",
        "`reservation contract contribution - incurred compensation - local capacity-relaxed walk-in opportunity`",
        "",
        "| run | completed day | direct net (CNY) | local opportunity (CNY) | RNVA-LB (CNY) | RNVA-LB (% initial official value) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        metric = row["metric"]
        lines.append(
            f"| {_label(row)} | {row['completed_day']} | "
            f"{metric['reservation_direct_net_contribution']:.2f} | "
            f"{metric['local_walk_in_opportunity_upper_bound']:.2f} | "
            f"{metric['cny']:.2f} | "
            f"{metric['pct_initial_official_value']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "V4 takes the capacity-relaxed sample at the actual walk-in demand phase, after any preceding due pickups. It removes walk-in limits and releases reservation rental/turnaround capacity. The shadow runs only the same day's demand sampler, never another pickup phase, day close or next-morning lifecycle. Actual and shadow new rentals are valued at the same end-of-demand point using frozen lifecycle outcomes. V4 must not be mixed with older shadow-method results.",
            "",
            "Daily alternatives can overlap across dates. The legacy upper_bound field names describe the intended conservative relaxation, not a proven bound on every feasible no-reservation policy. This is not a complete counterfactual season or the causal effect of enabling reservations.",
            "",
            "V3 compensation accounting counts incurred termination and partial-fulfillment obligations, including payment failures. Inquiry quotes, customer acceptance and manager selection alone are not costs. The legacy reservation_failure_compensation column includes both termination and partial fulfillment. JSON compensation_accounting records the settlement sources, cutoff and reconciliation with the replayed environment counter. Termination obligations whose refund bankrupted the shop before compensation charging are disclosed separately.",
            "",
            "For V38, realized proactive-contingency effects enter through actual contract terms, rent, settlements, and capacity use. Inquiry, acceptance, application, or commitment-version counts are diagnostic only and receive no direct reward. RNVA-LB is not the causal effect of enabling negotiation.",
            "",
            "`local_contingency_value` is reported separately and is never added to RNVA. For each applied proposal, it replays a keep-original branch under the same subsequent recorded state-changing actions and computes `V(actual applied contract) - V(keep original contract)` after both contracts mature. It excludes cross-contract and walk-in spillovers and does not value missed or unapplied opportunities.",
            "",
            "Deposits, prepayment refunds, common overhead, model cost, and acquisition cash outlay are not reservation revenue/cost here. Durable use depreciation is included.",
        ]
    )
    (output_dir / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _snapshot_source_root(value: Path) -> Path:
    path = value.expanduser().resolve()
    if (path / "outfitter" / "runner.py").is_file():
        return path
    if (path / "src" / "outfitter" / "runner.py").is_file():
        return path / "src"
    raise ValueError(
        "source root must contain outfitter/runner.py or src/outfitter/runner.py: "
        f"{path}"
    )


def _analyze_run_in_snapshot(
    run_dir: str, source_root: str, through_day: int | None
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    worker = repository / "scripts" / "reservation_value_snapshot_worker.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(worker),
            "--source-root",
            source_root,
            "--analysis-module",
            str(Path(__file__).resolve()),
            "--run-dir",
            run_dir,
            *(
                ["--through-day", str(through_day)]
                if through_day is not None
                else []
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"snapshot replay failed for {run_dir}:\n{completed.stderr.strip()}"
        )
    return json.loads(completed.stdout)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute and plot reservation net value from existing runs."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent run replays to execute in parallel.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "Frozen source tree used by the trajectories. Accepts either its "
            "root or its src directory."
        ),
    )
    parser.add_argument(
        "--through-day",
        type=int,
        help="Analyze only completed days up to this common cutoff.",
    )
    args = parser.parse_args(argv)
    run_dirs = [str(path.expanduser().resolve()) for path in args.run_dirs]
    if args.source_root is not None:
        source_root = str(_snapshot_source_root(args.source_root))
        if args.workers <= 1 or len(run_dirs) == 1:
            results = [
                _analyze_run_in_snapshot(
                    path, source_root, args.through_day
                )
                for path in run_dirs
            ]
        else:
            results = []
            with ThreadPoolExecutor(
                max_workers=min(args.workers, len(run_dirs))
            ) as executor:
                future_by_path = {
                    executor.submit(
                        _analyze_run_in_snapshot,
                        path,
                        source_root,
                        args.through_day,
                    ): path
                    for path in run_dirs
                }
                for future in as_completed(future_by_path):
                    results.append(future.result())
    elif args.workers <= 1 or len(run_dirs) == 1:
        results = [
            analyze_run(path, through_day=args.through_day)
            for path in run_dirs
        ]
    else:
        results = []
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(run_dirs))
        ) as executor:
            future_by_path = {
                executor.submit(
                    analyze_run, path, through_day=args.through_day
                ): path
                for path in run_dirs
            }
            for future in as_completed(future_by_path):
                results.append(future.result())
    write_outputs(results, args.output_dir.expanduser().resolve())
    for result in sorted(results, key=lambda row: str(row["run_id"])):
        print(
            f"{_label(result)} day={result['completed_day']} "
            f"RNVA-LB={result['metric']['cny']:.2f} CNY "
            f"({result['metric']['pct_initial_official_value']:.3f}%)"
        )
    print(args.output_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
