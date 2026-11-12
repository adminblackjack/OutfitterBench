"""Pure request-level reservation bundle choice for channel-demand V5."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Any

from .channel_demand import ChannelSegment, ChannelWorld, _cash_penalty


@dataclass(frozen=True, slots=True)
class BundleStoreOption:
    line_id: str
    option_id: str
    offer_id: str
    promise_profile_id: str
    required: bool
    non_cash_value: float
    total_rent: float
    total_deposit: float
    # Audit-only decomposition for V20 subjective concern generation. These
    # fields do not enter customer choice; ``non_cash_value`` remains the
    # authoritative score input for every legacy mode.
    equipment_fit_value: float = 0.0
    price_relative_value: float = 0.0
    profile_taste_value: float = 0.0
    # A single customer and cash contract, evaluated at each advertised date.
    # Keep whole-date scores correlated; never sum independently worst lines.
    date_variants: tuple["BundleStoreOption", ...] = ()


@dataclass(frozen=True, slots=True)
class BundleOutsideOption:
    line_id: str
    option_id: str
    channel: str
    non_cash_value_before_delivery: float
    cash_price: float
    online_on_time_probability: float = 1.0


@dataclass(frozen=True, slots=True)
class BundleChoiceLine:
    line_id: str
    required: bool
    store_options: tuple[BundleStoreOption, ...]
    outside_options: tuple[BundleOutsideOption, ...]


@dataclass(frozen=True, slots=True)
class BundleChoiceInput:
    lines: tuple[BundleChoiceLine, ...]
    world: ChannelWorld
    cash_segment: ChannelSegment
    deposit_friction_rate: float
    store_cash_salience: float
    store_service_value: float
    return_burden: float
    online_friction: float
    online_lateness_loss_rate: float
    occasion_importance: float
    channel_taste_scale: float
    channel_tastes: dict[str, float]
    # Public, request-frozen V29 experiment term.  It is subtracted once from
    # every complete store plan and never from outside or abandonment.
    store_reputation_penalty: float = 0.0
    date_variants: tuple["BundleChoiceInput", ...] = ()
    rental_budget_tail_coefficient: float = 0.0


@dataclass(frozen=True, slots=True)
class BundleChoiceTrace:
    selected_channel: str
    selected_store_options: dict[str, str]
    selected_store_offer_ids: dict[str, str]
    selected_outside_options: dict[str, str]
    store_score: float
    outside_score: float
    abandon_score: float
    store_total_rent: float
    store_total_deposit: float
    outside_total_cash: float
    outside_late_probability: float
    store_configuration_count: int
    outside_configuration_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_channel": self.selected_channel,
            "selected_store_options": dict(self.selected_store_options),
            "selected_store_offer_ids": dict(self.selected_store_offer_ids),
            "selected_outside_options": dict(self.selected_outside_options),
            "store_score": self.store_score,
            "outside_score": self.outside_score,
            "abandon_score": self.abandon_score,
            "store_total_rent": self.store_total_rent,
            "store_total_deposit": self.store_total_deposit,
            "outside_total_cash": self.outside_total_cash,
            "outside_late_probability": self.outside_late_probability,
            "store_configuration_count": self.store_configuration_count,
            "outside_configuration_count": self.outside_configuration_count,
        }


@dataclass(frozen=True, slots=True)
class CompleteContractOption:
    """One manager-submitted complete store contract.

    Required lines are present exactly once. Optional lines are either present
    exactly once or explicitly omitted before this pure kernel is called.
    """

    contract_option_id: str
    lines: tuple[BundleStoreOption, ...]
    omitted_optional_line_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CompleteContractChoiceTrace:
    selected_channel: str
    selected_contract_option_id: str | None
    best_store_contract_option_id: str
    selected_store_options: dict[str, str]
    selected_store_offer_ids: dict[str, str]
    contract_scores: dict[str, float]
    contract_total_rents: dict[str, float]
    contract_total_deposits: dict[str, float]
    selected_outside_options: dict[str, str]
    outside_score: float
    abandon_score: float
    outside_total_cash: float
    outside_late_probability: float
    outside_configuration_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_channel": self.selected_channel,
            "selected_contract_option_id": self.selected_contract_option_id,
            "best_store_contract_option_id": self.best_store_contract_option_id,
            "selected_store_options": dict(self.selected_store_options),
            "selected_store_offer_ids": dict(self.selected_store_offer_ids),
            "contract_scores": dict(self.contract_scores),
            "contract_total_rents": dict(self.contract_total_rents),
            "contract_total_deposits": dict(self.contract_total_deposits),
            "selected_outside_options": dict(self.selected_outside_options),
            "outside_score": self.outside_score,
            "abandon_score": self.abandon_score,
            "outside_total_cash": self.outside_total_cash,
            "outside_late_probability": self.outside_late_probability,
            "outside_configuration_count": self.outside_configuration_count,
        }


@dataclass(frozen=True, slots=True)
class CustomerPlanSetTrace:
    """Independent acceptability and customer ranking for factorized plans.

    Every store plan is evaluated on its own against the same outside plan and
    abandonment.  ``authorized_plan_ids`` is deliberately capped and therefore
    is not an exhaustive disclosure of every acceptable internal plan.
    """

    customer_preferred_plan_id: str | None
    authorized_plan_ids: tuple[str, ...]
    acceptable_plan_ids: tuple[str, ...]
    plan_scores: dict[str, float]
    outside_score: float
    abandon_score: float
    internal_plan_count: int
    acceptable_plan_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_preferred_plan_id": self.customer_preferred_plan_id,
            "authorized_plan_ids": list(self.authorized_plan_ids),
            "acceptable_plan_ids": list(self.acceptable_plan_ids),
            "plan_scores": dict(self.plan_scores),
            "outside_score": self.outside_score,
            "abandon_score": self.abandon_score,
            "internal_plan_count": self.internal_plan_count,
            "acceptable_plan_count": self.acceptable_plan_count,
        }


def _taste(value: BundleChoiceInput, channel: str) -> float:
    raw = float(value.channel_tastes.get(channel, 0.0))
    if not math.isfinite(raw):
        raise ValueError("bundle channel tastes must be finite")
    return float(value.channel_taste_scale) * min(1.0, max(-1.0, raw))


def _first_best(
    rows: list[tuple[float, tuple[str, ...], Any]],
) -> tuple[float, tuple[str, ...], Any] | None:
    if not rows:
        return None
    rows.sort(key=lambda row: row[1])
    best = rows[0]
    for row in rows[1:]:
        if row[0] > best[0] + 1e-12:
            best = row
    return best


def evaluate_reservation_bundle_choice(
    value: BundleChoiceInput,
) -> BundleChoiceTrace:
    """Choose one complete store bundle, one complete outside plan, or abandon.

    Required lines must be covered. Optional store lines always include a skip
    alternative, so adding an optional quote cannot reduce the best store score.
    Absolute cash, deposit, delivery risk, store service, and return burden are
    applied once per request.
    """

    if not value.lines:
        raise ValueError("bundle choice requires at least one line")
    if len({line.line_id for line in value.lines}) != len(value.lines):
        raise ValueError("bundle line_ids must be unique")
    if not any(line.required for line in value.lines):
        raise ValueError("bundle choice requires at least one required line")
    if value.store_cash_salience <= 0:
        raise ValueError("store_cash_salience must be positive")

    ordered_lines = tuple(sorted(value.lines, key=lambda line: line.line_id))
    store_dimensions: list[tuple[BundleStoreOption | None, ...]] = []
    for line in ordered_lines:
        options = tuple(
            sorted(
                line.store_options,
                key=lambda option: (option.option_id, option.offer_id),
            )
        )
        if line.required and not options:
            store_dimensions = []
            break
        store_dimensions.append(options if line.required else (None, *options))

    store_rows: list[tuple[float, tuple[str, ...], dict[str, Any]]] = []
    for configuration in (
        itertools.product(*store_dimensions) if store_dimensions else ()
    ):
        selected = [option for option in configuration if option is not None]
        rent = sum(float(option.total_rent) for option in selected)
        deposit = sum(float(option.total_deposit) for option in selected)
        generalized = rent + float(value.deposit_friction_rate) * deposit
        score = (
            sum(float(option.non_cash_value) for option in selected)
            - float(value.store_cash_salience)
            * _cash_penalty(
                generalized,
                world=value.world,
                segment=value.cash_segment,
            )
            + float(value.store_service_value)
            - float(value.return_burden)
            + _taste(value, "store")
            - value.rental_budget_tail_coefficient*max(0.,generalized/value.cash_segment.cash_scale_cny-2.)**2
            - max(0.0, float(value.store_reputation_penalty))
        )
        signature = tuple(
            "skip" if option is None else option.option_id
            for option in configuration
        )
        store_rows.append(
            (
                score,
                signature,
                {
                    "options": selected,
                    "rent": rent,
                    "deposit": deposit,
                },
            )
        )
    best_store = _first_best(store_rows)

    required_lines = tuple(line for line in ordered_lines if line.required)
    outside_dimensions: list[tuple[BundleOutsideOption, ...]] = []
    for line in required_lines:
        options = tuple(
            sorted(
                line.outside_options,
                key=lambda option: (option.channel, option.option_id),
            )
        )
        if not options:
            outside_dimensions = []
            break
        outside_dimensions.append(options)
    outside_rows: list[tuple[float, tuple[str, ...], dict[str, Any]]] = []
    for configuration in (
        itertools.product(*outside_dimensions) if outside_dimensions else ()
    ):
        cash = sum(float(option.cash_price) for option in configuration)
        online = [option for option in configuration if option.channel == "online"]
        on_time = math.prod(
            min(
                1.0,
                max(0.0, float(option.online_on_time_probability)),
            )
            for option in online
        )
        late_probability = 1.0 - on_time if online else 0.0
        delivery_penalty = (
            float(value.online_friction)
            + float(value.online_lateness_loss_rate)
            * late_probability
            * float(value.occasion_importance)
            if online
            else 0.0
        )
        score = (
            sum(
                float(option.non_cash_value_before_delivery)
                for option in configuration
            )
            - _cash_penalty(
                cash,
                world=value.world,
                segment=value.cash_segment,
            )
            - delivery_penalty
            + _taste(value, "outside")
        )
        signature = tuple(option.option_id for option in configuration)
        outside_rows.append(
            (
                score,
                signature,
                {
                    "options": configuration,
                    "cash": cash,
                    "late_probability": late_probability,
                },
            )
        )
    best_outside = _first_best(outside_rows)

    store_score = float("-inf") if best_store is None else best_store[0]
    outside_score = float("-inf") if best_outside is None else best_outside[0]
    abandon_score = _taste(value, "abandon")
    selected_channel = max(
        ("abandon", "outside", "store"),
        key=lambda channel: (
            {
                "abandon": abandon_score,
                "outside": outside_score,
                "store": store_score,
            }[channel],
            channel,
        ),
    )
    store_data = {} if best_store is None else best_store[2]
    outside_data = {} if best_outside is None else best_outside[2]
    selected_store = (
        list(store_data.get("options", [])) if selected_channel == "store" else []
    )
    selected_outside = list(outside_data.get("options", []))
    return BundleChoiceTrace(
        selected_channel=selected_channel,
        selected_store_options={
            option.line_id: option.option_id for option in selected_store
        },
        selected_store_offer_ids={
            option.line_id: option.offer_id for option in selected_store
        },
        selected_outside_options={
            option.line_id: option.option_id for option in selected_outside
        },
        store_score=store_score,
        outside_score=outside_score,
        abandon_score=abandon_score,
        store_total_rent=float(store_data.get("rent", 0.0)),
        store_total_deposit=float(store_data.get("deposit", 0.0)),
        outside_total_cash=float(outside_data.get("cash", 0.0)),
        outside_late_probability=float(
            outside_data.get("late_probability", 0.0)
        ),
        store_configuration_count=len(store_rows),
        outside_configuration_count=len(outside_rows),
    )


def evaluate_complete_contract_choice(
    value: BundleChoiceInput,
    contracts: tuple[CompleteContractOption, ...],
) -> CompleteContractChoiceTrace:
    """Choose among one or two explicit complete store contracts.

    Unlike :func:`evaluate_reservation_bundle_choice`, this function never
    forms a Cartesian product across line menus. Every scored store row is one
    complete contract explicitly submitted by the manager.
    """

    if value.date_variants:
        return _evaluate_common_date_contracts(value, contracts)
    if not value.lines:
        raise ValueError("complete contract choice requires at least one line")
    if not 1 <= len(contracts) <= 2:
        raise ValueError("complete contract choice requires one or two contracts")
    if len({row.contract_option_id for row in contracts}) != len(contracts):
        raise ValueError("complete contract option IDs must be unique")
    ordered_lines = tuple(sorted(value.lines, key=lambda line: line.line_id))
    line_by_id = {line.line_id: line for line in ordered_lines}
    if len(line_by_id) != len(ordered_lines):
        raise ValueError("bundle line_ids must be unique")
    required_ids = {line.line_id for line in ordered_lines if line.required}
    optional_ids = set(line_by_id) - required_ids
    if not required_ids:
        raise ValueError("complete contract choice requires a required line")
    if value.store_cash_salience <= 0:
        raise ValueError("store_cash_salience must be positive")

    contract_rows: list[tuple[float, tuple[str, ...], dict[str, Any]]] = []
    for contract in contracts:
        selected = tuple(sorted(contract.lines, key=lambda row: row.line_id))
        selected_ids = [row.line_id for row in selected]
        omitted = tuple(sorted(contract.omitted_optional_line_ids))
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("a complete contract contains a duplicate line")
        if set(selected_ids) - set(line_by_id):
            raise ValueError("a complete contract contains an unknown line")
        if not required_ids.issubset(selected_ids):
            raise ValueError("a complete contract omits a required line")
        if set(omitted) - optional_ids:
            raise ValueError("only optional lines may be omitted")
        if set(selected_ids) & set(omitted):
            raise ValueError("a contract line cannot be quoted and omitted")
        if set(selected_ids) | set(omitted) != set(line_by_id):
            raise ValueError("a complete contract must resolve every line")
        for option in selected:
            if option.required != line_by_id[option.line_id].required:
                raise ValueError("contract line required flags must match inquiry")
        rent = sum(float(option.total_rent) for option in selected)
        deposit = sum(float(option.total_deposit) for option in selected)
        score = score_complete_store_contract(value, contract)
        contract_rows.append(
            (
                score,
                (contract.contract_option_id,),
                {
                    "contract": contract,
                    "options": selected,
                    "rent": rent,
                    "deposit": deposit,
                },
            )
        )
    best_store = _first_best(contract_rows)
    if best_store is None:
        raise AssertionError("one complete store contract must be available")

    required_lines = tuple(line for line in ordered_lines if line.required)
    outside_dimensions: list[tuple[BundleOutsideOption, ...]] = []
    for line in required_lines:
        options = tuple(
            sorted(
                line.outside_options,
                key=lambda option: (option.channel, option.option_id),
            )
        )
        if not options:
            outside_dimensions = []
            break
        outside_dimensions.append(options)
    outside_rows: list[tuple[float, tuple[str, ...], dict[str, Any]]] = []
    for configuration in (
        itertools.product(*outside_dimensions) if outside_dimensions else ()
    ):
        cash = sum(float(option.cash_price) for option in configuration)
        online = [option for option in configuration if option.channel == "online"]
        on_time = math.prod(
            min(1.0, max(0.0, float(option.online_on_time_probability)))
            for option in online
        )
        late_probability = 1.0 - on_time if online else 0.0
        delivery_penalty = (
            float(value.online_friction)
            + float(value.online_lateness_loss_rate)
            * late_probability
            * float(value.occasion_importance)
            if online
            else 0.0
        )
        score = (
            sum(
                float(option.non_cash_value_before_delivery)
                for option in configuration
            )
            - _cash_penalty(
                cash,
                world=value.world,
                segment=value.cash_segment,
            )
            - delivery_penalty
            + _taste(value, "outside")
        )
        signature = tuple(option.option_id for option in configuration)
        outside_rows.append(
            (
                score,
                signature,
                {
                    "options": configuration,
                    "cash": cash,
                    "late_probability": late_probability,
                },
            )
        )
    best_outside = _first_best(outside_rows)
    outside_score = float("-inf") if best_outside is None else best_outside[0]
    abandon_score = _taste(value, "abandon")
    store_score = float(best_store[0])
    selected_channel = max(
        ("abandon", "outside", "store"),
        key=lambda channel: (
            {
                "abandon": abandon_score,
                "outside": outside_score,
                "store": store_score,
            }[channel],
            channel,
        ),
    )
    best_data = best_store[2]
    best_contract: CompleteContractOption = best_data["contract"]
    selected_options = (
        list(best_data["options"]) if selected_channel == "store" else []
    )
    outside_data = {} if best_outside is None else best_outside[2]
    return CompleteContractChoiceTrace(
        selected_channel=selected_channel,
        selected_contract_option_id=(
            best_contract.contract_option_id
            if selected_channel == "store"
            else None
        ),
        best_store_contract_option_id=best_contract.contract_option_id,
        selected_store_options={
            option.line_id: option.option_id for option in selected_options
        },
        selected_store_offer_ids={
            option.line_id: option.offer_id for option in selected_options
        },
        contract_scores={
            str(row[2]["contract"].contract_option_id): float(row[0])
            for row in contract_rows
        },
        contract_total_rents={
            str(row[2]["contract"].contract_option_id): float(row[2]["rent"])
            for row in contract_rows
        },
        contract_total_deposits={
            str(row[2]["contract"].contract_option_id): float(row[2]["deposit"])
            for row in contract_rows
        },
        selected_outside_options={
            option.line_id: option.option_id
            for option in outside_data.get("options", [])
        },
        outside_score=outside_score,
        abandon_score=abandon_score,
        outside_total_cash=float(outside_data.get("cash", 0.0)),
        outside_late_probability=float(
            outside_data.get("late_probability", 0.0)
        ),
        outside_configuration_count=len(outside_rows),
    )


def score_complete_store_contract(
    value: BundleChoiceInput,
    contract: CompleteContractOption,
) -> float:
    """Score one explicit store contract without evaluating outside plans."""

    if value.date_variants:
        return min(score_complete_store_contract(v, _contract_on_date(contract, i))
                   for i, v in enumerate(value.date_variants))
    rent = sum(float(option.total_rent) for option in contract.lines)
    deposit = sum(float(option.total_deposit) for option in contract.lines)
    generalized = rent + float(value.deposit_friction_rate) * deposit
    return (
        sum(float(option.non_cash_value) for option in contract.lines)
        - float(value.store_cash_salience)
        * _cash_penalty(
            generalized,
            world=value.world,
            segment=value.cash_segment,
        )
        + float(value.store_service_value)
        - float(value.return_burden)
        + _taste(value, "store")
        - value.rental_budget_tail_coefficient*max(0.,generalized/value.cash_segment.cash_scale_cny-2.)**2
        - max(0.0, float(value.store_reputation_penalty))
    )


def _contract_on_date(contract, index):
    return replace(contract, lines=tuple(o.date_variants[index] if o.date_variants else o
                                         for o in contract.lines))


def _evaluate_common_date_contracts(value, contracts):
    """Authorize manager date discretion, not an undisclosed date-specific veto."""
    traces = [evaluate_complete_contract_choice(v, tuple(_contract_on_date(c, i) for c in contracts))
              for i, v in enumerate(value.date_variants)]
    first = traces[0]
    if any(t.contract_total_rents != first.contract_total_rents
           or t.contract_total_deposits != first.contract_total_deposits for t in traces):
        raise ValueError("a common date contract must have identical cash terms on every date")
    scores = {c.contract_option_id: min(t.contract_scores[c.contract_option_id] for t in traces)
              for c in contracts}
    best_id = min(scores, key=lambda cid: (-scores[cid], cid))
    outside = max(traces, key=lambda t: t.outside_score)
    abandon = max(t.abandon_score for t in traces)
    channel = max(("abandon", "outside", "store"),
                  key=lambda ch: ({"abandon": abandon, "outside": outside.outside_score,
                                   "store": scores[best_id]}[ch], ch))
    selected = next(c for c in contracts if c.contract_option_id == best_id)
    return replace(first, selected_channel=channel,
        selected_contract_option_id=best_id if channel == "store" else None,
        best_store_contract_option_id=best_id, contract_scores=scores,
        selected_store_options={o.line_id: o.option_id for o in selected.lines} if channel == "store" else {},
        selected_store_offer_ids={o.line_id: o.offer_id for o in selected.lines} if channel == "store" else {},
        outside_score=outside.outside_score, abandon_score=abandon,
        selected_outside_options=outside.selected_outside_options,
        outside_total_cash=outside.outside_total_cash,
        outside_late_probability=outside.outside_late_probability,
        outside_configuration_count=sum(t.outside_configuration_count for t in traces))


def evaluate_customer_plan_set(
    value: BundleChoiceInput,
    plans: tuple[CompleteContractOption, ...],
    *,
    authorization_limit: int = 3,
    internal_plan_limit: int = 24,
) -> CustomerPlanSetTrace:
    """Return a capped set of plans the customer would transact on.

    A plan is acceptable iff it wins when offered as the only store plan.  This
    leave-one-in evaluation prevents a plan from being called unacceptable only
    because another store plan ranks above it.  Customer ranking uses utility,
    then the canonical internal plan ID for deterministic ties.
    """

    if not 1 <= int(authorization_limit) <= int(internal_plan_limit):
        raise ValueError("authorization_limit must be within the internal plan limit")
    if not 1 <= len(plans) <= int(internal_plan_limit):
        raise ValueError(
            f"factorized plan set requires between 1 and {internal_plan_limit} plans"
        )
    identifiers = [str(plan.contract_option_id) for plan in plans]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("factorized plan IDs must be unique")

    single_plan_traces = {
        plan.contract_option_id: evaluate_complete_contract_choice(value, (plan,))
        for plan in plans
    }
    first = single_plan_traces[plans[0].contract_option_id]
    for trace in single_plan_traces.values():
        if not math.isclose(
            float(trace.outside_score),
            float(first.outside_score),
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(trace.abandon_score),
            float(first.abandon_score),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError("all factorized plans must face one frozen outside choice")

    scores = {
        plan_id: float(trace.contract_scores[plan_id])
        for plan_id, trace in single_plan_traces.items()
    }
    acceptable = sorted(
        (
            plan_id
            for plan_id, trace in single_plan_traces.items()
            if trace.selected_channel == "store"
        ),
        key=lambda plan_id: (-scores[plan_id], plan_id),
    )
    authorized = tuple(acceptable[: int(authorization_limit)])
    return CustomerPlanSetTrace(
        customer_preferred_plan_id=authorized[0] if authorized else None,
        authorized_plan_ids=authorized,
        acceptable_plan_ids=tuple(acceptable),
        plan_scores=scores,
        outside_score=float(first.outside_score),
        abandon_score=float(first.abandon_score),
        internal_plan_count=len(plans),
        acceptable_plan_count=len(acceptable),
    )
