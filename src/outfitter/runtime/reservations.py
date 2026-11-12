from __future__ import annotations

from dataclasses import asdict, replace
from decimal import Decimal, ROUND_HALF_UP
from ..service_commitments import split_cents
import itertools
import json
import math
import statistics

from ..channel_demand import (
    REQUEST_BUNDLE_CHANNEL_DEMAND_FORMAT,
    effective_rental_cash_segment,
)
from ..errors import ReservationQuoteBatchValidationError
from ..reservation_choice import (
    BundleChoiceInput,
    BundleChoiceLine,
    BundleOutsideOption,
    BundleStoreOption,
    CompleteContractOption,
    evaluate_complete_contract_choice,
    evaluate_customer_plan_set,
    evaluate_reservation_bundle_choice,
    score_complete_store_contract,
)
from ..reservation_contingency import (
    ADVERSITY,
    SELECTED_PARAMETERS as CONTINGENCY_PARAMETERS,
    cancellation_probability,
    customer_class,
    date_utility_delta,
    initial_date_is_acceptable,
)
from ..reservation_feedback import (
    ContractAssessmentDiagnostic,
    EQUIPMENT_CONFIGURATION_TAG,
    EQUIPMENT_FIT_CONCERN,
    factorized_concern_tag_probability,
    factorized_concern_uniform,
    OTHER_OR_UNCLEAR_TAG,
    PRICE_VALUE_TAG,
    PUBLIC_ASSESSMENT_CATEGORIES,
    PUBLIC_PLAN_SET_DECLINE_CATEGORIES,
    line_mention_uniform,
    reported_assessment_category,
    reported_customer_mentioned_line_id,
    reported_factorized_concerns,
    reported_plan_set_decline_category,
)
from ..weather_information import (
    POTENTIAL_POOL_MULTIPLIERS,
    consideration_probability,
    forecast_regime,
    natural_cancellation_day_weights,
    natural_cancellation_probability,
    weighted_index,
)
from ..reservation_difficulty import rental_amount, extension_amount

from .common import (
    Any,
    CalendarBlock,
    Counter,
    DAMAGE_RATES,
    RentalContract,
    ReservationAssignment,
    ReservationChangeRequest,
    ReservationCommitment,
    ReservationEvent,
    ReservationLine,
    ReservationRequest,
    RESERVATION_REUSE_CATEGORY_ADJUSTMENT,
    RESERVATION_STORAGE_CATEGORY_ADJUSTMENT,
    WorldState,
    _ARCHETYPE_INDEX,
    _RNG_ARRIVALS,
    _RNG_CUSTOMER_BASKET,
    _RNG_CUSTOMER_WEATHER_CONSIDERATION,
    _RNG_NATURAL_CANCELLATION,
    _RNG_PARTY_COMPOSITION,
    _RNG_REQUEST_DAMAGE,
    _RNG_REQUEST_DAMAGE_SEVERITY,
    _RNG_RESERVATION_LEAD,
    _RNG_RESERVATION_CHANGE,
    _RNG_RESERVATION_FALLBACK,
    _RNG_RESERVATION_LATE_RETURN,
    _RNG_RESERVATION_QUOTE_CANDIDATES,
    _RNG_RESERVATION_QUOTE_RECOVERY,
    _RNG_RESERVATION_BUNDLE_CHOICE,
    _RNG_RESERVATION_FEEDBACK_REPORT,
    _RNG_RESERVATION_FEEDBACK_RESPONSE,
    _RNG_RESERVATION_ROUTE,
    _RNG_RESERVATION_WEATHER_RESPONSE,
    copy,
    ChoiceSession,
    seasonal_index,
)


class ReservationMixin:
    """Advance commitments with delayed exact-offer asset assignment."""

    @staticmethod
    def _reservation_line_rng_identity(request_id: str, line_id: str) -> str:
        """Keep random tapes stable while public line IDs become request-local."""

        if line_id.startswith(f"{request_id}-"):
            return line_id
        return f"{request_id}-{line_id}"

    @staticmethod
    def _reservation_commitment_rng_identity(request_id: str) -> str:
        suffix = request_id.removeprefix("rr-")
        return f"reservation-{suffix}"

    @classmethod
    def _reservation_change_rng_identity(
        cls,
        commitment: ReservationCommitment,
        change: ReservationChangeRequest,
    ) -> str:
        return (
            f"change-{cls._reservation_commitment_rng_identity(commitment.request_id)}-"
            f"{change.change_type}-{change.created_day:03d}"
        )

    def _rental_damage_probability(
        self, daily_rate: float, start_day: int, return_day: int
    ) -> float:
        if not self.config.weather_wear_enabled:
            return 1.0 - (1.0 - daily_rate) ** max(1, return_day - start_day)
        survival = 1.0
        multipliers = {
            "sunny": 1.0,
            "cloudy": 1.0,
            "hot": 1.15,
            "cold": 1.25,
            "rainy": 1.55,
            "snowy": 1.75,
        }
        for day in range(start_day, return_day):
            hazard = min(
                0.95,
                daily_rate
                * multipliers.get(self._demand_weather_regime(day), 1.0),
            )
            survival *= 1.0 - hazard
        return 1.0 - survival

    def _draw_customer_protection_credit(
        self, state: WorldState, required_cash: float
    ) -> float:
        shortfall = max(0.0, float(required_cash) - float(state.cash))
        available = max(
            0.0,
            float(self.config.emergency_credit_limit)
            - state.emergency_credit_principal,
        )
        draw = round(min(shortfall, available), 2)
        if draw <= 0:
            return 0.0
        fee = round(draw * float(self.config.emergency_credit_fee_rate), 2)
        state.cash = round(state.cash + draw, 2)
        state.emergency_credit_principal = round(
            state.emergency_credit_principal + draw, 2
        )
        state.emergency_credit_liability = round(
            state.emergency_credit_liability + draw + fee, 2
        )
        state.cumulative_emergency_credit_fees = round(
            state.cumulative_emergency_credit_fees + fee, 2
        )
        self._reservation_event(
            state,
            "customer_protection_credit_drawn",
            details={"principal": draw, "fee": fee},
        )
        return draw

    def _return_weather_cleaning_multiplier(
        self, start_day: int, return_day: int
    ) -> float:
        if not self.config.weather_wear_enabled:
            return 1.0
        exposure = sum(
            {
                "rainy": 0.18,
                "snowy": 0.28,
                "hot": 0.05,
                "cold": 0.08,
            }.get(self._demand_weather_regime(day), 0.0)
            for day in range(start_day, return_day)
        )
        return min(1.75, 1.0 + exposure)

    def _reservation_event(
        self,
        state: WorldState,
        event_type: str,
        *,
        request_id: str | None = None,
        commitment_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        state.reservation_events.append(
            ReservationEvent(
                event_id=f"reservation-event-{state.next_reservation_event_id:06d}",
                day=state.day,
                event_type=event_type,
                request_id=request_id,
                commitment_id=commitment_id,
                details=dict(details or {}),
            )
        )
        state.next_reservation_event_id += 1

    def _partial_fulfillment_terms(
        self, commitment: ReservationCommitment, missing: list[ReservationLine]
    ) -> dict[str, Any]:
        """Refund promised optional rent and compensate the affected service only."""
        affected_rent = round(sum(line.locked_rent for line in missing), 2)
        affected_units = sum(
            int(line.service_units or line.customer_units) * int(line.duration_days)
            for line in missing
        )
        policy = self.dataset.metadata["reservation_failure_settlement"]["pickup_failure"]
        compensation = self._cent_half_up(max(
            float(policy["locked_rent_rate"]) * affected_rent,
            float(policy["service_base"]) + float(policy["service_unit_day_rate"]) * affected_units,
        ))
        return {
            "missing_line_ids": [line.line_id for line in missing],
            "rent_credit": affected_rent,
            "compensation": compensation,
        }

    def _inquire_partial_fulfillment(
        self, state: WorldState, request: ReservationRequest,
        commitment: ReservationCommitment,
    ) -> dict[str, Any]:
        if not self._v45_reservation_difficulty():
            raise ValueError("partial fulfillment requires V45")
        if state.day != commitment.pickup_day or commitment.status != "at_risk":
            raise ValueError("partial fulfillment is available only for today's pickup risk")
        if commitment.partial_fulfillment_inquiry_day is not None:
            raise ValueError("this commitment version already had a partial fulfillment inquiry")
        missing_ids = set(commitment.partial_fulfillment_missing_line_ids)
        if self._v54_recovery_state():
            current_missing, reason = self._current_partial_pickup(state, commitment)
            if reason is not None:
                raise ValueError(f"partial fulfillment unavailable: {reason}")
            missing_ids = set(current_missing)
        missing = [line for line in commitment.lines if line.line_id in missing_ids]
        if not missing or any(line.required for line in missing):
            raise ValueError("partial fulfillment requires missing optional lines only")
        if self._v54_recovery_state():
            # Freeze the proposed omissions only on a valid customer inquiry.
            commitment.partial_fulfillment_missing_line_ids = [line.line_id for line in missing]
        terms = self._partial_fulfillment_terms(commitment, missing)
        # Re-evaluate the customer's remaining real basket with the same frozen
        # profile/taste context used at quotation, but with today's revised cash
        # terms. Physical inventory is never read into this customer preference.
        dated_request = replace(request, quote_context_by_date={})
        options_by_line = {}
        for line in commitment.lines:
            quantity = int(line.asset_quantity or line.quantity)
            first_day = (
                float(line.locked_line_daily_rent) / quantity
                if line.locked_line_daily_rent is not None
                else float(line.locked_daily_price)
            )
            options_by_line[line.line_id] = [{
                "option_id": f"current:{line.line_id}",
                "offer_id": line.selected_offer_id or line.offer_id,
                "promise_profile_id": line.promise_profile_id,
                "asset_quantity": quantity,
                "daily_unit_price": first_day,
                "total_rent": float(line.locked_rent),
                "locked_security_deposit": float(line.locked_security_deposit),
            }]
        choice = self._reservation_bundle_choice_input(state, dated_request, options_by_line)
        chosen = {line.line_id: line.store_options[0] for line in choice.lines
                  if line.store_options}
        if any(line.line_id not in chosen for line in commitment.lines):
            raise AssertionError("confirmed reservation has no frozen customer option")
        original = CompleteContractOption("original", tuple(chosen.values()))
        remaining = [chosen[line.line_id] for line in commitment.lines
                     if line.line_id not in missing_ids]
        if not remaining:
            raise AssertionError("partial fulfillment must retain required lines")
        # Compensation is a cash transfer in addition to rent credit. Put it
        # into the reduced basket only for utility comparison; actual settlement
        # remains a separately charged cash flow at pickup.
        remaining[0] = replace(
            remaining[0], total_rent=remaining[0].total_rent - terms["compensation"])
        reduced = CompleteContractOption("reduced", tuple(remaining))
        score_change = (score_complete_store_contract(choice, reduced)
                        - score_complete_store_contract(choice, original))
        inconvenience = 0.6 + 0.2 * len(missing)
        probability = max(0.08, min(0.90,
            1.0 / (1.0 + math.exp(-max(-40.0, min(40.0,
                (score_change - inconvenience) / 1.5))))))
        accepted = bool(self._stable_event_rng(
            _RNG_RESERVATION_FALLBACK, *request.event_address,
            identity=f"partial:{commitment.commitment_version}",
        ).random() < probability)
        commitment.partial_fulfillment_inquiry_day = state.day
        commitment.partial_fulfillment_accepted = accepted
        commitment.partial_fulfillment_compensation = terms["compensation"]
        self._reservation_event(
            state, "partial_fulfillment_inquired",
            request_id=request.request_id,
            commitment_id=commitment.commitment_id,
            details={**terms, "customer_accepted": accepted},
        )
        return {**terms, "customer_accepted": accepted,
                "requires_explicit_selection": accepted,
                "expires_day": state.day}

    def _reservation_route(
        self, pickup_day: int, archetype_index: int, customer_index: int
    ) -> tuple[bool, int]:
        if not self.config.reservations_enabled:
            return False, 0
        low = int(self.config.reservation_booking_lead_min_days)
        high = int(self.config.reservation_booking_lead_max_days)
        lead_rng = self._event_rng(
            _RNG_RESERVATION_LEAD,
            pickup_day,
            archetype_index,
            customer_index,
        )
        lead = int(lead_rng.integers(low, high + 1))
        route_rng = self._event_rng(
            _RNG_RESERVATION_ROUTE,
            pickup_day,
            archetype_index,
            customer_index,
        )
        selected = bool(route_rng.random() < float(self.config.reservation_share))
        return selected and pickup_day - lead >= 1, lead

    def _latent_arrival_count(self, pickup_day: int, archetype: str) -> int:
        calendar = self._calendar(pickup_day)
        source_day = self._source_weather_day(pickup_day)
        season = seasonal_index(source_day, self.rules.seasonal_anchors)
        rate = self._calendar_arrival_lambda(archetype, calendar)
        rate *= self.rules.arrival_scale * self.config.arrival_scale_multiplier
        rate *= season
        demand_weather = self._demand_weather_regime(pickup_day)
        rate *= self.rules.weather_multipliers[demand_weather][archetype]
        return int(
            self._event_rng(
                _RNG_ARRIVALS,
                pickup_day,
                _ARCHETYPE_INDEX[archetype],
            ).poisson(rate)
        )

    def _potential_arrival_count(self, pickup_day: int, archetype: str) -> int:
        """V42 weather-independent customer pool for one target day."""

        calendar = self._calendar(pickup_day)
        source_day = self._source_weather_day(pickup_day)
        rate = self._calendar_arrival_lambda(archetype, calendar)
        rate *= self.rules.arrival_scale * self.config.arrival_scale_multiplier
        rate *= seasonal_index(source_day, self.rules.seasonal_anchors)
        rate *= float(POTENTIAL_POOL_MULTIPLIERS[archetype])
        if self._v45_reservation_difficulty():
            peak = float(self.dataset.metadata.get("reservation_calendar_peak_multiplier", 1.0))
            if not 1.0 <= peak <= 2.2:
                raise ValueError("V45 calendar peak multiplier outside supported range")
            calendar_class = str(calendar["day_type"])
            factor = (
                peak if calendar_class == "public_holiday" or bool(calendar["is_weekend"])
                else max(0.35, (7.0 - 2.0 * peak) / 5.0)
            )
            rate *= factor
        return int(
            self._event_rng(
                _RNG_ARRIVALS,
                pickup_day,
                _ARCHETYPE_INDEX[archetype],
            ).poisson(rate)
        )

    def _customer_weather_considers(
        self,
        *,
        archetype: str,
        regime: str,
        event_address: tuple[int, int, int],
        forecast_trust: float | None,
        channel: str,
    ) -> bool:
        probability = consideration_probability(
            archetype=archetype,
            regime=regime,
            forecast_trust=forecast_trust,
        )
        draw = self._stable_event_rng(
            _RNG_CUSTOMER_WEATHER_CONSIDERATION,
            *event_address,
            identity=f"{channel}-consideration-v1",
        ).random()
        return bool(draw < probability)

    def _reset_v42_weather_review(
        self,
        state: WorldState,
        commitment: ReservationCommitment,
        *,
        forecast_snapshot: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if not self._v42_weather_information_coherence():
            return
        row = None
        if forecast_snapshot is not None:
            row = forecast_snapshot.get(str(commitment.pickup_day))
        if row is None:
            row = next(
                item
                for item in self.weather_forecast(
                    state, commitment.pickup_day - state.day + 1
                )
                if int(item["day"]) == commitment.pickup_day
            )
        regime = forecast_regime(row)
        commitment.weather_review_pickup_day = commitment.pickup_day
        commitment.weather_review_baseline_regime = regime
        commitment.weather_review_max_adversity = float(ADVERSITY[regime])
        commitment.weather_reviewed_regimes = [regime]

    def _initialize_v42_commitment_customer_state(
        self,
        state: WorldState,
        request: ReservationRequest,
        commitment: ReservationCommitment,
    ) -> None:
        if not self._v42_weather_information_coherence():
            return
        self._reset_v42_weather_review(state, commitment)
        days = natural_cancellation_day_weights(
            confirmed_day=state.day,
            pickup_day=commitment.pickup_day,
        )
        if not days:
            return
        probability = natural_cancellation_probability(
            booking_lead_days=commitment.pickup_day - state.day,
            schedule_flexibility=request.schedule_flexibility,
        )
        occurrence_rng = self._stable_event_rng(
            _RNG_NATURAL_CANCELLATION,
            *request.event_address,
            identity="non-weather-occurrence-v1",
        )
        if occurrence_rng.random() >= probability:
            return
        timing_rng = self._stable_event_rng(
            _RNG_NATURAL_CANCELLATION,
            *request.event_address,
            identity="non-weather-timing-v1",
        )
        index = weighted_index(
            float(timing_rng.random()), [weight for _day, weight in days]
        )
        commitment.natural_cancellation_day = int(days[index][0])

    def _reservation_primary_session(
        self,
        state: WorldState,
        *,
        pickup_day: int,
        archetype: str,
        customer_index: int,
        booking_lead_days: int,
    ) -> Any | None:
        if archetype == "supply":
            return None
        archetype_index = _ARCHETYPE_INDEX[archetype]
        event_address = (pickup_day, archetype_index, customer_index)
        party_rng = self._event_rng(_RNG_PARTY_COMPOSITION, *event_address)
        basket_rng = self._event_rng(_RNG_CUSTOMER_BASKET, *event_address)
        adult_ranges = {
            "overnight": (1, 4),
            "picnic": (2, 5),
            "lightweight": (1, 2),
        }
        low, high = adult_ranges[archetype]
        adults = int(party_rng.integers(low, high))
        day_type = str(self._calendar(pickup_day)["day_type"])
        child_probability = {
            "weekday": 0.08,
            "weekend": 0.15,
            "public_holiday": 0.35,
            "school_holiday": 0.45,
        }[day_type]
        children = (
            int(party_rng.integers(1, 3))
            if archetype != "lightweight"
            and party_rng.random() < child_probability
            else 0
        )
        if archetype == "overnight":
            if self._v45_reservation_difficulty():
                duration = (
                    int(basket_rng.choice([2, 3, 4, 5], p=[0.12, 0.32, 0.36, 0.20]))
                    if day_type in {"public_holiday", "school_holiday"}
                    else int(basket_rng.choice([1, 2, 3, 4], p=[0.22, 0.40, 0.28, 0.10]))
                )
            else:
                duration = (
                    int(basket_rng.choice([2, 3, 4, 5], p=[0.2, 0.35, 0.30, 0.15]))
                    if day_type == "public_holiday"
                    else int(basket_rng.choice([1, 2, 3], p=[0.45, 0.40, 0.15]))
                )
            service_key = "rental:tent"
            units = 1
            size = "n/a"
        elif archetype == "picnic":
            duration = (
                int(basket_rng.choice([1, 2], p=[0.82, 0.18]))
                if self._v45_reservation_difficulty()
                and day_type in {"weekend", "public_holiday", "school_holiday"}
                else 1
            )
            service_key = "rental:camp_table"
            units = 1
            size = "n/a"
        else:
            duration = (
                int(basket_rng.choice([1, 2, 3], p=[0.58, 0.34, 0.08]))
                if self._v45_reservation_difficulty()
                else int(basket_rng.choice([1, 2], p=[0.7, 0.3]))
            )
            service_key = "rental:sleeping_bag"
            units = 1
            size = "adult_regular"
        if self._v49_school_holiday():
            # Shared with routing and walk-in; the legacy draws above remain
            # unchanged for pre-V49 worlds and are not the V49 source of truth.
            shape = self._v47_customer_shape(pickup_day, archetype, customer_index)
            adults, children, duration = shape['adults'], shape['children'], shape['duration']
        if pickup_day + duration > state.max_days:
            return None
        forecast = next(
            row
            for row in self.weather_forecast(state, booking_lead_days + 1)
            if int(row["day"]) == pickup_day
        )
        future_state = copy.copy(state)
        future_state.day = pickup_day
        session = self._make_choice_session(
            future_state,
            service_key=service_key,
            units_requested=units,
            duration_days=duration,
            adults=adults,
            children=children,
            size_class=size,
            archetype=archetype,
            weather=str(forecast["condition"]),
            request_index=0,
            event_address=event_address,
            intent_role="required",
            **({"trip_weather_context": self._trip_weather_context(
                state, pickup_day=pickup_day, duration_days=duration)}
               if self._v44_trip_weather() else {}),
        )
        adversity = {
            "sunny": 0.0,
            "cloudy": 0.18,
            "hot": 0.35,
            "cold": 0.45,
            "rainy": 0.82,
            "snowy": 1.0,
        }.get(str(forecast["condition"]), 0.2)
        return replace(
            session,
            interaction_id=(
                f"reservation-{pickup_day:03d}-{archetype_index:02d}-"
                f"{customer_index:04d}"
            ),
            booking_lead_days=booking_lead_days,
            pickup_day=pickup_day,
            forecast_adversity=adversity,
            forecast_uncertainty=max(
                0.0, 1.0 - float(forecast["condition_accuracy"])
            ),
        )

    def _reservation_bundle_sessions(
        self,
        state: WorldState,
        primary: Any,
        *,
        pickup_day: int,
        booking_lead_days: int,
    ) -> list[tuple[Any, bool]]:
        if not self.config.reservation_bundles_enabled:
            return [(primary, True)]
        party = primary.adults + primary.children
        duration = int(primary.duration_days or 1)
        specs: list[tuple[str, int, str, bool]]
        if primary.archetype == "overnight":
            specs = [
                ("rental:tent", 1, "n/a", True),
                ("rental:sleeping_bag", max(1, primary.adults), "adult_regular", True),
                ("rental:sleeping_pad", max(1, party), "n/a", True),
                ("rental:camp_lantern", 1, "n/a", False),
            ]
        elif primary.archetype == "picnic":
            specs = [
                ("rental:camp_table", 1, "n/a", True),
                ("rental:camp_chair", max(1, party), "n/a", True),
                ("rental:camp_cookware", 1, "n/a", False),
            ]
        else:
            specs = [
                ("rental:sleeping_bag", 1, "adult_regular", True),
                ("rental:sleeping_pad", 1, "n/a", True),
                ("rental:camp_lantern", 1, "n/a", False),
            ]
        optional_rng = self._stable_event_rng(
            _RNG_RESERVATION_CHANGE,
            *primary.event_address,
            identity="bundle-optional-line",
        )
        include_optional = bool(optional_rng.random() < 0.60)
        forecast = next(
            row
            for row in self.weather_forecast(state, booking_lead_days + 1)
            if int(row["day"]) == pickup_day
        )
        future_state = copy.copy(state)
        future_state.day = pickup_day
        result = []
        for index, (service_key, units, size, required) in enumerate(specs):
            if not required and not include_optional:
                continue
            if service_key == primary.service_key:
                session = primary
            else:
                session = self._make_choice_session(
                    future_state,
                    service_key=service_key,
                    units_requested=units,
                    duration_days=duration,
                    adults=primary.adults,
                    children=primary.children,
                    size_class=size,
                    archetype=primary.archetype,
                    weather=str(forecast["condition"]),
                    request_index=index,
                    event_address=primary.event_address[:3],
                    intent_role="required" if required else "optional",
                    **({"trip_weather_context": self._trip_weather_context(
                        state, pickup_day=pickup_day, duration_days=duration)}
                       if self._v44_trip_weather() else {}),
                )
                session = replace(
                    session,
                    interaction_id=f"{primary.interaction_id}-line-{index + 1:02d}",
                    booking_lead_days=booking_lead_days,
                    pickup_day=pickup_day,
                    forecast_adversity=self._forecast_adversity(
                        str(forecast["condition"])
                    ),
                    forecast_uncertainty=max(
                        0.0, 1.0 - float(forecast["condition_accuracy"])
                    ),
                )
            result.append(
                (
                    self._reservation_customer_coherent_session(primary, session),
                    required,
                )
            )
        return result[:4]

    @staticmethod
    def _reservation_customer_coherent_session(
        primary: ChoiceSession,
        line: ChoiceSession,
    ) -> ChoiceSession:
        """Share customer traits while retaining product-line preferences."""

        storage = min(
            1.0,
            max(
                0.0,
                float(primary.storage_burden)
                + RESERVATION_STORAGE_CATEGORY_ADJUSTMENT.get(
                    line.service_key, 0.0
                ),
            ),
        )
        reuse = min(
            1.0,
            max(
                0.0,
                float(primary.future_use_expectation)
                + RESERVATION_REUSE_CATEGORY_ADJUSTMENT.get(
                    line.service_key, 0.0
                ),
            ),
        )
        return replace(
            line,
            customer_segment=primary.customer_segment,
            price_beta=primary.price_beta,
            forecast_trust=primary.forecast_trust,
            storage_burden=round(storage, 6),
            future_use_expectation=round(reuse, 6),
            return_burden_multiplier=primary.return_burden_multiplier,
            request_budget_multiplier=primary.request_budget_multiplier,
        )

    @staticmethod
    def _reservation_substitution_hint(choice_group: str) -> str:
        return (
            f"除询价时的公开候选或确认后的已授权集合外，该客户对 "
            f"{choice_group} 类可能仍有未披露的同类备选。只有在已授权集合"
            "无法履约时，才可通过"
            " inquire_fallback 询问届时本店可信供应的候选。"
        )

    @staticmethod
    def _frozen_quote_context(session: ChoiceSession) -> dict[str, Any]:
        value = asdict(session)
        if value.get("trip_weather_context") is None:
            value.pop("trip_weather_context", None)
        value["event_address"] = list(session.event_address)
        return value

    @staticmethod
    def _restore_quote_context(value: dict[str, Any]) -> ChoiceSession:
        restored = dict(value)
        restored["event_address"] = tuple(map(int, restored["event_address"]))
        return ChoiceSession(**restored)

    def _reservation_inquiry_candidates(
        self,
        state: WorldState,
        session: ChoiceSession,
        *,
        rng_line_identity: str,
        quantity: int,
        duration_days: int,
    ) -> tuple[list[str], list[str], dict[str, float], dict[str, float]]:
        """Freeze ordered public candidates and the private recovery universe."""

        views = self._channel_offer_views(state, session)
        service_offers = list(self.by_service.get(session.service_key, []))
        if not any(
            bool(state.rental_service_open.get(offer.offer_id, False))
            for offer in service_offers
        ):
            return [], [], {}, {}
        compatible = [
            offer.offer_id
            for offer in sorted(
                service_offers,
                key=lambda row: row.offer_id,
            )
            if str(offer.choice_group) == str(session.preferred_group)
            and self._required_resources(session, offer.offer_id) == quantity
            and self._reference_hard_compatibility(
                state, session, offer.offer_id
            )[0]
        ]
        ceilings: dict[str, float] = {}
        deposits: dict[str, float] = {}
        preference_scores: dict[str, float] = {}
        views_by_id = {view.offer_id: view for view in views}
        for offer_id in compatible:
            deposit = round(quantity * float(state.deposits[offer_id]), 2)
            ceiling = self._reservation_candidate_price_ceiling(
                session,
                views,
                offer_id=offer_id,
                quantity=quantity,
                duration_days=duration_days,
                locked_security_deposit=deposit,
            )
            if ceiling is None:
                continue
            ceilings[offer_id] = ceiling
            deposits[offer_id] = deposit
            generalized_ceiling = (
                (rental_amount(ceiling, duration_days, quantity)
                 if self._v45_reservation_difficulty()
                 else ceiling * quantity * duration_days)
                + float(self.channel_demand_model.deposit_friction_rate)
                * deposit
            )
            preference_scores[offer_id] = generalized_ceiling / max(
                0.01, float(views_by_id[offer_id].online_cash_price)
            )
        ranked = sorted(
            ceilings,
            key=lambda offer_id: (-preference_scores[offer_id], offer_id),
        )
        if not ranked:
            return [], ranked, ceilings, deposits
        rng = self._stable_event_rng(
            _RNG_RESERVATION_QUOTE_CANDIDATES,
            *session.event_address,
            identity=f"inquiry-candidates:{rng_line_identity}",
        )
        maximum = int(self.config.reservation_authorized_alternative_max)
        alternative_count = min(
            max(0, len(ranked) - 1), int(rng.integers(0, maximum + 1))
        )
        tail_indices = (
            list(
                map(
                    int,
                    rng.choice(
                        len(ranked) - 1,
                        size=alternative_count,
                        replace=False,
                    ),
                )
            )
            if alternative_count
            else []
        )
        alternatives = [ranked[index + 1] for index in tail_indices]
        order = {offer_id: index for index, offer_id in enumerate(ranked)}
        selected = [ranked[0], *alternatives]
        selected = [selected[0], *sorted(selected[1:], key=order.__getitem__)]
        return selected, ranked, ceilings, deposits

    def _credible_reservation_offer(
        self, state: WorldState, offer_id: str
    ) -> bool:
        if offer_id not in state.prices or float(state.prices[offer_id]) <= 0:
            return False
        if not bool(state.rental_service_open.get(offer_id, False)):
            return False
        owned = any(
            asset.offer_id == offer_id
            and (
                self._warehouse_asset_held(asset)
                if self._v31_warehouse_capacity()
                else asset.status not in {"retired", "sold"}
            )
            for asset in state.assets
        )
        ordered = any(
            order.status == "in_transit"
            and any(str(item["offer_id"]) == offer_id for item in order.items)
            for order in state.orders
        )
        return owned or ordered

    def _reservation_candidate_selected_at_price(
        self,
        session: Any,
        views: tuple[Any, ...],
        *,
        offer_id: str,
        quantity: int,
        duration_days: int,
        daily_price: float,
        locked_security_deposit: float,
    ) -> bool:
        candidate_views = tuple(
            replace(
                view,
                profile_id=(
                    f"{view.profile_id}:exact:{view.offer_id}"
                    if self._reservation_pairwise_mode()
                    and view.offer_id == offer_id
                    else view.profile_id
                ),
                available_units=quantity if view.offer_id == offer_id else 0,
                store_unit_price=(
                    daily_price
                    if view.offer_id == offer_id
                    else view.store_unit_price
                ),
                store_cash_price=(
                    (rental_amount(daily_price, duration_days, quantity)
                     if self._v45_reservation_difficulty()
                     else quantity * duration_days * daily_price)
                    if view.offer_id == offer_id
                    else view.store_cash_price
                ),
                store_deposit=(
                    locked_security_deposit
                    if view.offer_id == offer_id
                    else view.store_deposit
                ),
            )
            for view in views
        )
        trace = self._evaluate_channel_views(session, candidate_views)
        return (
            trace.selected_channel == "store"
            and trace.selected_offer_id == offer_id
        )

    def _reservation_candidate_price_ceiling(
        self,
        session: Any,
        views: tuple[Any, ...],
        *,
        offer_id: str,
        quantity: int,
        duration_days: int,
        locked_security_deposit: float,
    ) -> float | None:
        """Return the greatest accepted two-decimal daily price.

        The channel evaluator is monotone in this counterfactual: customer
        context and outside options are frozen, the candidate is the only
        store offer with capacity, and only its rental price changes.
        """

        def accepted(price_cents: int) -> bool:
            return self._reservation_candidate_selected_at_price(
                session,
                views,
                offer_id=offer_id,
                quantity=quantity,
                duration_days=duration_days,
                daily_price=price_cents / 100.0,
                locked_security_deposit=locked_security_deposit,
            )

        if not accepted(1):
            return None
        accepted_cents = 1
        rejected_cents = 100
        maximum_cents = 100_000_000
        while accepted(rejected_cents) and rejected_cents < maximum_cents:
            accepted_cents = rejected_cents
            rejected_cents = min(maximum_cents, rejected_cents * 2)
        if rejected_cents == maximum_cents and accepted(rejected_cents):
            return maximum_cents / 100.0
        while accepted_cents + 1 < rejected_cents:
            midpoint = (accepted_cents + rejected_cents) // 2
            if accepted(midpoint):
                accepted_cents = midpoint
            else:
                rejected_cents = midpoint
        return accepted_cents / 100.0

    def _reservation_authorization_for_line(
        self,
        state: WorldState,
        session: Any,
        *,
        line_id: str,
        primary_offer_id: str,
        quantity: int,
        duration_days: int,
        locked_daily_price: float,
        locked_security_deposit: float,
    ) -> tuple[list[str], list[str], dict[str, float], str | None]:
        views = self._channel_offer_views(state, session)
        primary_group = str(self.offers[primary_offer_id].choice_group)
        candidate_ids = [
            offer.offer_id
            for offer in sorted(
                self.by_service.get(session.service_key, []),
                key=lambda row: row.offer_id,
            )
            if offer.offer_id != primary_offer_id
            and str(offer.choice_group) == primary_group
            and self._required_resources(session, offer.offer_id) == quantity
            and self._reference_hard_compatibility(
                state, session, offer.offer_id
            )[0]
        ]
        ceilings: dict[str, float] = {}
        primary_ceiling = self._reservation_candidate_price_ceiling(
            session,
            views,
            offer_id=primary_offer_id,
            quantity=quantity,
            duration_days=duration_days,
            locked_security_deposit=locked_security_deposit,
        )
        if primary_ceiling is not None:
            ceilings[primary_offer_id] = primary_ceiling
        acceptable = []
        for candidate_id in candidate_ids:
            ceiling = self._reservation_candidate_price_ceiling(
                session,
                views,
                offer_id=candidate_id,
                quantity=quantity,
                duration_days=duration_days,
                locked_security_deposit=locked_security_deposit,
            )
            if ceiling is None or ceiling + 1e-9 < locked_daily_price:
                continue
            ceilings[candidate_id] = ceiling
            acceptable.append(candidate_id)
        authorization_rng = self._stable_event_rng(
            _RNG_RESERVATION_FALLBACK,
            *session.event_address,
            identity=f"latent-acceptable-set:{line_id}",
        )
        maximum = int(self.config.reservation_authorized_alternative_max)
        alternative_count = min(
            len(acceptable), int(authorization_rng.integers(0, maximum + 1))
        )
        alternatives = (
            [
                acceptable[index]
                for index in map(
                    int,
                    authorization_rng.choice(
                        len(acceptable),
                        size=alternative_count,
                        replace=False,
                    ),
                )
            ]
            if alternative_count
            else []
        )
        latent = [primary_offer_id, *alternatives]
        mode = self.config.reservation_authorization_mode
        if mode == "all_latent":
            authorized = list(latent)
        elif mode == "risk_only":
            authorized = [primary_offer_id]
        else:
            authorized = [
                primary_offer_id,
                *[
                    offer_id
                    for offer_id in alternatives
                    if self._credible_reservation_offer(state, offer_id)
                ],
            ]
        hidden = [offer_id for offer_id in latent if offer_id not in authorized]
        hint = (
            self._reservation_substitution_hint(primary_group)
            if hidden
            else None
        )
        latent_ceilings = {
            offer_id: ceilings[offer_id]
            for offer_id in latent
            if offer_id in ceilings
        }
        return authorized, latent, latent_ceilings, hint

    def _reservation_profile_mode(self) -> bool:
        return self.config.reservation_contract_mode in {
            "promise_profiles_v1",
            "adaptive_pairwise_exact_quotes_v1",
            "complete_contract_options_v1",
            "factorized_line_options_plan_set_v1",
        }

    def _reservation_pairwise_mode(self) -> bool:
        return (
            self.config.reservation_contract_mode
            == "adaptive_pairwise_exact_quotes_v1"
        )

    def _reservation_complete_contract_mode(self) -> bool:
        return self.config.reservation_contract_mode == "complete_contract_options_v1"

    def _reservation_factorized_plan_mode(self) -> bool:
        return (
            self.config.reservation_contract_mode
            == "factorized_line_options_plan_set_v1"
        )

    def _reservation_request_bundle_choice_mode(self) -> bool:
        return bool(
            (
                self._reservation_pairwise_mode()
                or self._reservation_complete_contract_mode()
                or self._reservation_factorized_plan_mode()
            )
            and self.channel_demand_model is not None
            and self.channel_demand_model.format
            == REQUEST_BUNDLE_CHANNEL_DEMAND_FORMAT
        )

    @staticmethod
    def _reputation_notice_band(notice_days: int) -> str:
        notice = max(0, int(notice_days))
        if notice >= 8:
            return "8_plus"
        if notice >= 4:
            return "4_7"
        if notice >= 2:
            return "2_3"
        if notice == 1:
            return "1"
        return "same_day"

    def _reservation_service_reputation(
        self, state: WorldState
    ) -> dict[str, Any]:
        if not self._v29_reservation_accountability():
            return {"enabled": False, "available": False}
        policy = self._reservation_reputation_policy()
        enabled = bool(self.config.reservation_service_reputation_enabled)
        if not enabled:
            return {
                "enabled": False,
                "available": True,
                "formal_default_enabled": False,
            }
        window_size = int(policy["outcome_window"])
        weights = dict(policy["failure_weights"])
        outcomes: list[dict[str, Any]] = []
        for event in state.reservation_events:
            if event.event_type == "pickup_fulfilled":
                weight = float(weights["pickup_fulfilled"])
                category = "pickup_fulfilled"
            elif event.event_type == "pickup_failed":
                weight = float(weights["pickup_failure"])
                category = "pickup_failure"
            elif event.event_type == "cancelled_by_manager":
                band = self._reputation_notice_band(
                    int(event.details.get("notice_days", 0))
                )
                weight_key = (
                    "manager_cancel_same_day"
                    if band == "same_day"
                    else f"manager_cancel_notice_{band}"
                )
                weight = float(weights[weight_key])
                category = weight_key
            else:
                continue
            outcomes.append(
                {
                    "event_id": event.event_id,
                    "day": int(event.day),
                    "category": category,
                    "failure_weight": weight,
                }
            )
        selected = outcomes[-window_size:]
        resolved = len(selected)
        weighted_failures = sum(
            float(row["failure_weight"]) for row in selected
        )
        pseudo = int(policy["cold_start_pseudo_outcomes"])
        burden = weighted_failures / max(1, pseudo + resolved)
        reliability = 100.0 * max(
            0.0,
            1.0
            - float(policy["reliability_score_burden_multiplier"])
            * burden,
        )
        penalty = min(
            float(policy["choice_penalty_cap"]),
            float(policy["choice_penalty_slope"]) * burden,
        )
        counts = Counter(str(row["category"]) for row in selected)
        return {
            "enabled": True,
            "available": True,
            "outcome_window": window_size,
            "resolved_outcomes_in_window": resolved,
            "cold_start_pseudo_outcomes": pseudo,
            "outcome_counts": dict(sorted(counts.items())),
            "weighted_failure_count": round(weighted_failures, 6),
            "regularized_failure_burden": round(burden, 8),
            "reliability_score": round(reliability, 4),
            "customer_choice_penalty": round(penalty, 6),
            "choice_penalty_cap": float(policy["choice_penalty_cap"]),
        }

    @staticmethod
    def _required_service_unit_days(
        commitment: ReservationCommitment,
    ) -> int:
        return sum(
            int(line.service_units or line.customer_units)
            * int(line.duration_days)
            for line in commitment.lines
            if line.required
        )

    @staticmethod
    def _cent_half_up(value: float) -> float:
        return float(
            Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )

    def _reservation_failure_terms(
        self,
        commitment: ReservationCommitment,
        *,
        status: str,
        notice_days: int,
    ) -> dict[str, Any]:
        notice = max(0, int(notice_days))
        locked_rent = round(float(commitment.total_locked_rent), 2)
        service_unit_days = self._required_service_unit_days(commitment)
        refund = (
            float(commitment.unearned_revenue)
            if self._v26_contract_state()
            else float(commitment.prepayment_collected)
        )
        if self._v29_reservation_accountability():
            policy = dict(
                self.dataset.metadata.get("reservation_failure_settlement") or {}
            )
            if policy.get("format") != (
                "outfitter-reservation-failure-settlement-v1"
            ):
                raise AssertionError("V29 failure settlement policy is missing")
            if status == "pickup_failed":
                row = dict(policy["pickup_failure"])
                rent_component = float(row["locked_rent_rate"]) * locked_rent
                service_component = float(row["service_base"]) + float(
                    row["service_unit_day_rate"]
                ) * service_unit_days
            else:
                row = dict(policy["manager_cancellation"])
                decay_days = float(row["notice_decay_days"])
                rent_notice = max(
                    float(row["rent_notice_floor"]),
                    1.0 - notice / decay_days,
                )
                service_notice = max(
                    float(row["service_notice_floor"]),
                    1.0 - notice / decay_days,
                )
                rent_component = (
                    float(row["locked_rent_rate"])
                    * locked_rent
                    * rent_notice
                )
                service_component = (
                    float(row["service_base"])
                    + float(row["service_unit_day_rate"])
                    * service_unit_days
                ) * service_notice
            compensation = self._cent_half_up(
                max(rent_component, service_component)
            )
            formula = str(policy["format"])
        else:
            compensation_rate = float(
                self.config.reservation_failure_compensation_rate
            ) * max(0.25, 1.0 - notice / 7.0)
            rent_component = locked_rent * compensation_rate
            service_component = 0.0
            compensation = round(rent_component, 2)
            formula = "legacy_locked_rent_fraction"
        return {
            "notice_days": notice,
            "notice_band": self._reputation_notice_band(notice),
            "current_locked_rent": locked_rent,
            "required_service_unit_days": service_unit_days,
            "prepayment_refund": round(refund, 2),
            "rent_component": self._cent_half_up(rent_component),
            "service_unit_day_floor_component": self._cent_half_up(
                service_component
            ),
            "compensation": compensation,
            "total_cash_outflow": round(refund + compensation, 2),
            "formula": formula,
        }

    @staticmethod
    def _reservation_public_profile_envelope(
        session: ChoiceSession,
    ) -> dict[str, Any]:
        service_key = str(session.service_key)
        hard: dict[str, Any] = {}
        if service_key == "rental:tent":
            hard["minimum_capacity_persons"] = int(
                session.adults + session.children
            )
        elif service_key == "rental:sleeping_bag":
            hard["size_class"] = str(session.size_class)
        elif service_key == "rental:camp_chair":
            hard["audience"] = "adult" if session.adults > 0 else "child"
        elif service_key == "rental:camp_cookware":
            hard["required_functions"] = list(
                {
                    "hot_drink": ("boil",),
                    "cook_meal": ("boil", "stew"),
                    "fry_meal": ("fry",),
                }.get(session.use_case, ("boil",))
            )
        elif service_key == "rental:camp_lantern":
            hard["minimum_runtime_hours"] = 8 * max(
                1, int(session.duration_days or 1)
            )

        preferences: list[dict[str, Any]] = []
        group_preferences = {
            "compact_tent": ("tent_style", "compact"),
            "family_tent": ("tent_style", "family_space"),
            "three_season_bag": ("temperature_use", "cool_weather"),
            "warm_weather_bag": ("temperature_use", "warm_weather"),
            "air_mattress": ("pad_type", "air_mattress"),
            "closed_cell_foam_pad": ("pad_type", "closed_cell_foam_pad"),
            "self_inflating_pad": ("pad_type", "self_inflating_pad"),
            "lightweight_chair": ("chair_style", "lightweight"),
            "comfort_chair": ("chair_style", "comfort"),
        }
        if session.preferred_group in group_preferences:
            name, value = group_preferences[session.preferred_group]
            preferences.append({"attribute": name, "value": value})
        if session.portability_sensitive and len(preferences) < 2:
            preferences.append(
                {"attribute": "portability", "value": "preferred"}
            )
        return {
            "service_key": service_key,
            "service_units": int(session.units_requested),
            "duration_days": int(session.duration_days or 1),
            "party": {
                "adults": int(session.adults),
                "children": int(session.children),
            },
            "use_case": str(session.use_case),
            "hard_requirements": hard,
            "stated_preferences": preferences[:2],
        }

    def _reservation_profile_inquiry_line(
        self,
        state: WorldState,
        session: ChoiceSession,
    ) -> dict[str, Any] | None:
        catalog = self.reservation_promise_catalog
        if catalog is None:
            raise AssertionError("reservation promise catalog is unavailable")
        service_offers = list(self.by_service.get(session.service_key, []))
        if not any(
            bool(state.rental_service_open.get(offer.offer_id, False))
            for offer in service_offers
        ):
            return None
        envelope = self._reservation_public_profile_envelope(session)
        eligible = catalog.eligible_profiles(envelope)
        if not eligible:
            return None
        views = self._channel_offer_views(state, session)
        policy_id = str(self.config.reservation_profile_deposit_policy)
        members_by_profile: dict[str, list[str]] = {}
        ceilings_by_profile: dict[str, dict[str, float]] = {}
        deposits_by_profile: dict[str, float] = {}
        if self._reservation_request_bundle_choice_mode():
            viable = False
            for profile_id in eligible:
                profile = catalog.by_id[profile_id]
                quantity = math.ceil(
                    int(envelope["service_units"])
                    / profile.capacity_per_asset
                )
                deposit = round(
                    catalog.unit_deposit(profile_id, policy_id) * quantity,
                    2,
                )
                members = [
                    offer_id
                    for offer_id in profile.member_offer_ids
                    if self._required_resources(session, offer_id) == quantity
                    and self._reference_hard_compatibility(
                        state, session, offer_id
                    )[0]
                ]
                members_by_profile[profile_id] = members
                ceilings_by_profile[profile_id] = {}
                deposits_by_profile[profile_id] = deposit
                viable |= bool(members)
            if not viable:
                return None
            return {
                "envelope": envelope,
                "eligible_profile_ids": [
                    profile_id
                    for profile_id in eligible
                    if members_by_profile[profile_id]
                ],
                "members_by_profile": members_by_profile,
                "ceilings_by_profile": ceilings_by_profile,
                "deposits_by_profile": deposits_by_profile,
            }
        ceiling_cache: dict[tuple[str, int, float], float | None] = {}
        viable = False
        for profile_id in eligible:
            profile = catalog.by_id[profile_id]
            quantity = math.ceil(
                int(envelope["service_units"]) / profile.capacity_per_asset
            )
            deposit = round(
                catalog.unit_deposit(profile_id, policy_id) * quantity, 2
            )
            members = list(profile.member_offer_ids)
            member_set = set(members)
            ceilings: dict[str, float] = {}
            for offer in service_offers:
                if self._required_resources(session, offer.offer_id) != quantity:
                    continue
                if not self._reference_hard_compatibility(
                    state, session, offer.offer_id
                )[0]:
                    continue
                cache_key = (offer.offer_id, quantity, deposit)
                if cache_key not in ceiling_cache:
                    ceiling_cache[cache_key] = (
                        self._reservation_candidate_price_ceiling(
                            session,
                            views,
                            offer_id=offer.offer_id,
                            quantity=quantity,
                            duration_days=int(envelope["duration_days"]),
                            locked_security_deposit=deposit,
                        )
                    )
                ceiling = ceiling_cache[cache_key]
                if ceiling is not None:
                    ceilings[offer.offer_id] = ceiling
            members_by_profile[profile_id] = members
            ceilings_by_profile[profile_id] = ceilings
            deposits_by_profile[profile_id] = deposit
            viable |= any(offer_id in ceilings for offer_id in member_set)
        if not viable:
            return None
        return {
            "envelope": envelope,
            "eligible_profile_ids": eligible,
            "members_by_profile": members_by_profile,
            "ceilings_by_profile": ceilings_by_profile,
            "deposits_by_profile": deposits_by_profile,
        }

    def _generate_reservation_requests(self, state: WorldState) -> None:
        if not self.config.reservations_enabled:
            return
        reputation_snapshot = (
            self._reservation_service_reputation(state)
            if self._v29_reservation_accountability()
            and self.config.reservation_service_reputation_enabled
            else {}
        )
        maximum = int(self.config.reservation_booking_lead_max_days)
        for pickup_day in range(
            state.day + 1,
            min(self.config.max_days, state.day + maximum) + 1,
        ):
            for archetype in ("overnight", "picnic", "lightweight"):
                archetype_index = _ARCHETYPE_INDEX[archetype]
                count = (
                    self._potential_arrival_count(pickup_day, archetype)
                    if self._v42_weather_information_coherence()
                    else self._latent_arrival_count(pickup_day, archetype)
                )
                for customer_index in range(count):
                    selected, lead = self._reservation_route(
                        pickup_day, archetype_index, customer_index
                    )
                    if not selected or pickup_day - lead != state.day:
                        continue
                    session = self._reservation_primary_session(
                        state,
                        pickup_day=pickup_day,
                        archetype=archetype,
                        customer_index=customer_index,
                        booking_lead_days=lead,
                    )
                    if session is None:
                        continue
                    if self._v42_weather_information_coherence():
                        forecast_row = next(
                            row
                            for row in self.weather_forecast(state, lead + 1)
                            if int(row["day"]) == pickup_day
                        )
                        if not self._customer_weather_considers(
                            archetype=archetype,
                            regime=forecast_regime(forecast_row),
                            event_address=(
                                pickup_day,
                                archetype_index,
                                customer_index,
                            ),
                            forecast_trust=float(session.forecast_trust),
                            channel="reservation",
                        ):
                            continue
                    request_id = (
                        f"rr-{pickup_day:03d}-{archetype_index:02d}-"
                        f"{customer_index:04d}"
                    )
                    if any(
                        row.request_id == request_id
                        for row in state.reservation_requests
                    ):
                        continue
                    lines: list[ReservationLine] = []
                    fallback_by_line: dict[str, list[str]] = {}
                    latent_by_line: dict[str, list[str]] = {}
                    latent_ceilings_by_line: dict[str, dict[str, float]] = {}
                    quote_context_by_line: dict[str, dict[str, Any]] = {}
                    candidate_deposit_by_line: dict[str, dict[str, float]] = {}
                    public_envelopes_by_line: dict[str, dict[str, Any]] = {}
                    profile_members_by_line: dict[
                        str, dict[str, list[str]]
                    ] = {}
                    profile_ceilings_by_line: dict[
                        str, dict[str, dict[str, float]]
                    ] = {}
                    profile_deposits_by_line: dict[str, dict[str, float]] = {}
                    bundle_valid = True
                    bundle_sessions = self._reservation_bundle_sessions(
                            state,
                            session,
                            pickup_day=pickup_day,
                            booking_lead_days=lead,
                        )
                    for line_index, (line_session, required) in enumerate(
                        bundle_sessions,
                        start=1,
                    ):
                        duration = int(line_session.duration_days or 1)
                        line_id = f"line-{line_index:02d}"
                        if self._reservation_profile_mode():
                            profile_snapshot = self._reservation_profile_inquiry_line(
                                state, line_session
                            )
                            if profile_snapshot is None:
                                if required:
                                    bundle_valid = False
                                    break
                                continue
                            envelope = dict(profile_snapshot["envelope"])
                            eligible = list(
                                profile_snapshot["eligible_profile_ids"]
                            )
                            members_by_profile = {
                                str(profile_id): list(members)
                                for profile_id, members in dict(
                                    profile_snapshot["members_by_profile"]
                                ).items()
                            }
                            ceilings_by_profile = {
                                str(profile_id): {
                                    str(offer_id): float(ceiling)
                                    for offer_id, ceiling in dict(ceilings).items()
                                }
                                for profile_id, ceilings in dict(
                                    profile_snapshot["ceilings_by_profile"]
                                ).items()
                            }
                            deposits_by_profile = {
                                str(profile_id): float(deposit)
                                for profile_id, deposit in dict(
                                    profile_snapshot["deposits_by_profile"]
                                ).items()
                            }
                            representative = members_by_profile[eligible[0]][0]
                            if self._reservation_request_bundle_choice_mode():
                                latent_union = sorted(
                                    {
                                        offer_id
                                        for members in members_by_profile.values()
                                        for offer_id in members
                                    }
                                )
                                latent_ceiling_union = {}
                            else:
                                latent_union = sorted(
                                    {
                                        offer_id
                                        for ceilings in ceilings_by_profile.values()
                                        for offer_id in ceilings
                                    }
                                )
                                latent_ceiling_union = {
                                    offer_id: max(
                                        ceilings[offer_id]
                                        for ceilings in ceilings_by_profile.values()
                                        if offer_id in ceilings
                                    )
                                    for offer_id in latent_union
                                }
                            public_envelopes_by_line[line_id] = envelope
                            profile_members_by_line[line_id] = members_by_profile
                            profile_ceilings_by_line[line_id] = (
                                ceilings_by_profile
                            )
                            profile_deposits_by_line[line_id] = (
                                deposits_by_profile
                            )
                            latent_by_line[line_id] = latent_union
                            latent_ceilings_by_line[line_id] = (
                                latent_ceiling_union
                            )
                            fallback_by_line[line_id] = []
                            quote_context_by_line[line_id] = (
                                self._frozen_quote_context(line_session)
                            )
                            lines.append(
                                ReservationLine(
                                    line_id=line_id,
                                    offer_id=representative,
                                    service_line=str(
                                        line_session.service_key
                                    ).removeprefix("rental:"),
                                    quantity=0,
                                    customer_units=int(
                                        line_session.units_requested
                                    ),
                                    duration_days=duration,
                                    candidate_offer_ids=[],
                                    authorized_offer_ids=[],
                                    required=required,
                                    status="pending_quote",
                                    service_key=str(line_session.service_key),
                                    service_units=int(
                                        line_session.units_requested
                                    ),
                                    public_hard_requirements=dict(
                                        envelope["hard_requirements"]
                                    ),
                                    public_preferences=[
                                        dict(row)
                                        for row in envelope[
                                            "stated_preferences"
                                        ]
                                    ],
                                    eligible_profile_ids=eligible,
                                    eligible_profile_family_ids=sorted(
                                        {
                                            self.reservation_promise_catalog.by_id[
                                                profile_id
                                            ].family_id
                                            for profile_id in eligible
                                        }
                                    ),
                                )
                            )
                            continue

                        quantity = int(line_session.units_requested)
                        # Exact resource requirements can differ from customer
                        # units (for example multi-person tents).
                        matching = [
                            offer
                            for offer in self.by_service.get(
                                line_session.service_key, []
                            )
                            if str(offer.choice_group)
                            == str(line_session.preferred_group)
                            and self._reference_hard_compatibility(
                                state, line_session, offer.offer_id
                            )[0]
                        ]
                        if matching:
                            quantity = self._required_resources(
                                line_session, matching[0].offer_id
                            )
                        candidates, latent, ceilings, deposits = (
                            self._reservation_inquiry_candidates(
                                state,
                                line_session,
                                rng_line_identity=(
                                    self._reservation_line_rng_identity(
                                        request_id, line_id
                                    )
                                ),
                                quantity=quantity,
                                duration_days=duration,
                            )
                        )
                        if not candidates:
                            if required:
                                bundle_valid = False
                                break
                            continue
                        offer_id = candidates[0]
                        latent_by_line[line_id] = latent
                        latent_ceilings_by_line[line_id] = ceilings
                        fallback_by_line[line_id] = []
                        quote_context_by_line[line_id] = (
                            self._frozen_quote_context(line_session)
                        )
                        candidate_deposit_by_line[line_id] = deposits
                        lines.append(
                            ReservationLine(
                                line_id=line_id,
                                offer_id=offer_id,
                                service_line=self.offers[offer_id].service_line,
                                quantity=quantity,
                                customer_units=int(line_session.units_requested),
                                duration_days=duration,
                                locked_daily_price=0.0,
                                locked_rent=0.0,
                                candidate_offer_ids=candidates,
                                authorized_offer_ids=[],
                                locked_security_deposit=0.0,
                                substitution_hint=(
                                    self._reservation_substitution_hint(
                                        str(self.offers[offer_id].choice_group)
                                    )
                                    if any(
                                        candidate not in candidates
                                        for candidate in latent
                                    )
                                    else None
                                ),
                                required=required,
                                status="pending_quote",
                            )
                        )
                    if not bundle_valid or not lines:
                        continue
                    total_locked_rent = 0.0
                    prepayment = 0.0
                    fallback_offer_ids: list[str] = []
                    change_rng = self._stable_event_rng(
                        _RNG_RESERVATION_CHANGE,
                        *session.event_address,
                        identity="customer-flexibility",
                    )
                    forecast = next(
                        row
                        for row in self.weather_forecast(state, lead + 1)
                        if int(row["day"]) == pickup_day
                    )
                    request = ReservationRequest(
                        request_id=request_id,
                        created_day=state.day,
                        expires_day=state.day,
                        pickup_day=pickup_day,
                        return_day=pickup_day
                        + max(line.duration_days for line in lines),
                        booking_lead_days=lead,
                        lines=lines,
                        visible_party_context=session.visible_party_context,
                        forecast_condition=str(forecast["condition"]),
                        forecast_accuracy=float(forecast["condition_accuracy"]),
                        prepayment_rate=float(
                            self.config.reservation_prepayment_rate
                        ),
                        total_locked_rent=total_locked_rent,
                        prepayment_due=prepayment,
                        event_address=list(session.event_address),
                        forecast_trust=float(session.forecast_trust),
                        schedule_flexibility=float(
                            change_rng.uniform(0.15, 0.95)
                        ),
                        fallback_offer_ids=fallback_offer_ids,
                        fallback_offer_ids_by_line=fallback_by_line,
                        customer_segment=session.customer_segment,
                        shared_price_beta=float(session.price_beta),
                        storage_burden_base=float(session.storage_burden),
                        future_use_expectation_base=float(
                            session.future_use_expectation
                        ),
                        return_burden_multiplier=float(
                            session.return_burden_multiplier
                        ),
                        request_budget_multiplier=float(
                            session.request_budget_multiplier
                        ),
                        inquiry_latent_offer_ids_by_line=latent_by_line,
                        inquiry_latent_price_ceiling_by_line=(
                            latent_ceilings_by_line
                        ),
                        quote_context_by_line=quote_context_by_line,
                        candidate_deposit_by_line=candidate_deposit_by_line,
                        reservation_contract_mode=(
                            str(self.config.reservation_contract_mode)
                        ),
                        catalog_snapshot_sha256=(
                            None
                            if self.reservation_promise_catalog is None
                            else self.reservation_promise_catalog.source_catalog_sha256
                        ),
                        promise_profile_schema_version=(
                            None
                            if self.reservation_promise_catalog is None
                            else self.reservation_promise_catalog.schema_version
                        ),
                        profile_deposit_policy_id=(
                            self.config.reservation_profile_deposit_policy
                        ),
                        inquiry_public_envelope_by_line=(
                            public_envelopes_by_line
                        ),
                        inquiry_profile_member_ids_by_line=(
                            profile_members_by_line
                        ),
                        inquiry_profile_price_ceiling_by_line=(
                            profile_ceilings_by_line
                        ),
                        inquiry_profile_deposit_by_line=(
                            profile_deposits_by_line
                        ),
                        latent_snapshot_provenance=(
                            "immutable_profile_inquiry_no_ceiling_v2"
                            if self._reservation_request_bundle_choice_mode()
                            else "immutable_profile_inquiry_v1"
                            if self._reservation_profile_mode()
                            else "immutable_inquiry_v1"
                        ),
                        original_inquiry_pickup_day=pickup_day,
                        original_inquiry_return_day=(
                            pickup_day
                            + max(line.duration_days for line in lines)
                        ),
                        service_reputation_snapshot=dict(
                            reputation_snapshot
                        ),
                    )
                    state.reservation_requests.append(request)
                    state.cumulative_reservation_requests += 1
                    self._reservation_event(
                        state,
                        "inquiry_created",
                        request_id=request_id,
                        details={
                            "pickup_day": pickup_day,
                            "return_day": request.return_day,
                            "line_count": len(lines),
                            "required_lines": sum(line.required for line in lines),
                            "optional_lines": sum(not line.required for line in lines),
                            "candidate_set_sizes": [
                                len(line.candidate_offer_ids) for line in lines
                            ],
                            "profile_option_counts": [
                                len(line.eligible_profile_ids) for line in lines
                            ],
                            "contract_mode": str(
                                self.config.reservation_contract_mode
                            ),
                            **(
                                {
                                    "service_reputation_snapshot": dict(
                                        reputation_snapshot
                                    )
                                }
                                if reputation_snapshot
                                else {}
                            ),
                        },
                    )
        pending = sum(
            request.status == "pending_quote" and request.created_day == state.day
            for request in state.reservation_requests
        )
        if pending:
            state.inbox.append(
                f"今日有 {pending} 个新预约询价，须在日终前一次性报价或拒绝。"
            )

    @staticmethod
    def _forecast_adversity(condition: str) -> float:
        return {
            "sunny": 0.0,
            "cloudy": 0.18,
            "hot": 0.35,
            "cold": 0.45,
            "rainy": 0.82,
            "snowy": 1.0,
        }.get(condition, 0.2)

    def _v26_reschedule_repeat_decay(self, prior_request_count: int) -> float:
        if prior_request_count <= 0:
            return 1.0
        mechanism = dict(self.dataset.metadata.get("reservation_mechanism") or {})
        if prior_request_count == 1:
            return float(
                mechanism.get(
                    "reschedule_nondeterioration_decay_after_one", 0.30
                )
            )
        return float(
            mechanism.get(
                "reschedule_nondeterioration_decay_after_two", 0.075
            )
        )

    @staticmethod
    def _reschedule_request_count(
        state: WorldState, commitment_id: str
    ) -> int:
        return sum(
            row.commitment_id == commitment_id
            and row.change_type == "reschedule"
            for row in state.reservation_change_requests
        )

    def _assert_v26_reservation_financial_invariants(
        self, state: WorldState
    ) -> None:
        if not self._v26_contract_state():
            return
        terminal_statuses = {
            "completed",
            "cancelled_by_customer",
            "cancelled_by_manager",
            "pickup_failed",
        }
        stale_terminal_balances = [
            commitment.commitment_id
            for commitment in state.reservation_commitments
            if commitment.status in terminal_statuses
            and (
                abs(float(commitment.outstanding_rent)) > 0.0049
                or abs(float(commitment.unearned_revenue)) > 0.0049
            )
        ]
        if stale_terminal_balances:
            raise AssertionError(
                "terminal reservation commitments retain live balances: "
                f"{stale_terminal_balances}"
            )
        expected_liability = round(
            sum(
                float(commitment.unearned_revenue)
                for commitment in state.reservation_commitments
                if commitment.status not in terminal_statuses
            ),
            2,
        )
        if not math.isclose(
            float(state.unearned_revenue_liability),
            expected_liability,
            abs_tol=0.0049,
        ):
            raise AssertionError(
                "reservation unearned-revenue liability mismatch: "
                f"state={state.unearned_revenue_liability:.2f}, "
                f"commitments={expected_liability:.2f}"
            )
        if not self._v27_extension_cash_audit():
            return
        for commitment in state.reservation_commitments:
            components = (
                float(commitment.prepayment_collected),
                float(commitment.prepayment_refunded),
                float(commitment.rent_collected_at_pickup),
                float(commitment.post_pickup_rent_collected),
            )
            if any(value < -0.0049 for value in components):
                raise AssertionError(
                    "reservation cash-history components must be nonnegative: "
                    f"{commitment.commitment_id}"
                )
            gross_cash = round(
                commitment.prepayment_collected
                + commitment.rent_collected_at_pickup
                + commitment.post_pickup_rent_collected,
                2,
            )
            net_cash = round(
                gross_cash - commitment.prepayment_refunded, 2
            )
            if commitment.status in {"confirmed", "at_risk"}:
                expected_net_cash = round(
                    float(commitment.unearned_revenue), 2
                )
                expected_contract_balance = round(
                    float(commitment.outstanding_rent)
                    + float(commitment.unearned_revenue),
                    2,
                )
                if not math.isclose(
                    expected_contract_balance,
                    float(commitment.total_locked_rent),
                    abs_tol=0.0049,
                ):
                    raise AssertionError(
                        "pre-pickup reservation balances do not reconcile to "
                        f"locked rent: {commitment.commitment_id}"
                    )
            elif commitment.status in {"active", "completed"}:
                continuation_cash = 0.0
                if self._v47_autonomous_rentals():
                    continuation_cash = self._v47_continuation_terms(
                        state, commitment.commitment_id)["continuation_rent_collected"]
                    if not math.isclose(float(commitment.post_pickup_rent_collected),
                                        float(continuation_cash), abs_tol=0.0049):
                        raise AssertionError(
                            "reservation continuation cash does not reconcile with episode: "
                            f"{commitment.commitment_id}"
                        )
                expected_net_cash = round(
                    float(commitment.total_locked_rent)
                    - float(commitment.optional_line_credits),
                    2,
                )
                expected_net_cash = round(expected_net_cash + continuation_cash, 2)
            elif commitment.status in terminal_statuses:
                expected_net_cash = 0.0
            else:
                continue
            if not math.isclose(
                net_cash, expected_net_cash, abs_tol=0.0049
            ):
                raise AssertionError(
                    "reservation rent-cash history does not reconcile: "
                    f"{commitment.commitment_id}, net={net_cash:.2f}, "
                    f"expected={expected_net_cash:.2f}"
                )

    @staticmethod
    def _v27_reservation_cash_history(
        commitment: ReservationCommitment,
    ) -> dict[str, float]:
        gross_cash = round(
            float(commitment.prepayment_collected)
            + float(commitment.rent_collected_at_pickup)
            + float(commitment.post_pickup_rent_collected),
            2,
        )
        return {
            "prepayment_collected": float(commitment.prepayment_collected),
            "prepayment_refunded": float(commitment.prepayment_refunded),
            "rent_due_at_pickup": float(
                commitment.rent_due_at_pickup or 0.0
            ),
            "rent_collected_at_pickup": float(
                commitment.rent_collected_at_pickup
            ),
            "post_pickup_rent_collected": float(
                commitment.post_pickup_rent_collected
            ),
            "gross_rent_cash_collected": gross_cash,
            "net_rent_cash_collected": round(
                gross_cash - float(commitment.prepayment_refunded), 2
            ),
            "compensation_paid": float(commitment.compensation_paid),
        }

    def _cancel_customer_reservation(
        self, state: WorldState, commitment: ReservationCommitment,
        *, reason: str = "weather", change_id: str | None = None,
    ) -> None:
        if commitment.status not in {"confirmed", "at_risk"}:
            return
        request = next(
            row
            for row in state.reservation_requests
            if row.request_id == commitment.request_id
        )
        commitment.status = "cancelled_by_customer"
        request.status = "cancelled_by_customer"
        refund = (
            float(commitment.unearned_revenue)
            if self._v26_contract_state()
            else float(commitment.prepayment_collected)
        )
        state.unearned_revenue_liability = round(
            max(
                0.0,
                state.unearned_revenue_liability
                - refund,
            ),
            2,
        )
        state.current_day_ledger["reservation_prepayment_refunds"] = round(
            float(
                state.current_day_ledger.get(
                    "reservation_prepayment_refunds", 0.0
                )
            )
            + refund,
            2,
        )
        state.cumulative_reservation_prepayment_refunds = round(
            state.cumulative_reservation_prepayment_refunds
            + refund,
            2,
        )
        state.cumulative_reservation_customer_cancellations += 1
        for block in state.reservation_calendar_blocks:
            if block.commitment_id == commitment.commitment_id:
                block.status = "cancelled"
        self._reservation_event(
            state,
            f"customer_cancelled_{reason}",
            request_id=request.request_id,
            commitment_id=commitment.commitment_id,
            details={"refund": refund, **({"change_id": change_id} if change_id is not None else {})},
        )
        if self._v26_contract_state():
            commitment.prepayment_refunded = round(
                commitment.prepayment_refunded + refund, 2
            )
            commitment.outstanding_rent = 0.0
            commitment.unearned_revenue = 0.0
        self._assert_v26_reservation_financial_invariants(state)
        self._refund(state, refund, commitment.commitment_id)

    def _create_change_request(
        self,
        state: WorldState,
        commitment: ReservationCommitment,
        *,
        change_type: str,
        proposed_pickup_day: int,
        proposed_return_day: int,
    ) -> None:
        if self._v29_reservation_accountability():
            if proposed_return_day <= proposed_pickup_day:
                raise ValueError("proposed rental duration must be positive")
            if change_type == "reschedule" and (
                proposed_return_day - proposed_pickup_day
                != commitment.return_day - commitment.pickup_day
            ):
                raise ValueError("reschedule must preserve the rental duration")
            if change_type == "extension" and proposed_pickup_day != commitment.pickup_day:
                raise ValueError("extension must preserve the pickup day")
        if (
            change_type == "reschedule"
            and proposed_pickup_day > self.config.max_days
        ):
            raise ValueError(
                "rescheduled pickup must remain within the decision horizon"
            )
        if proposed_return_day > state.max_days:
            raise ValueError("reservation change exceeds the runoff horizon")
        if any(
            row.commitment_id == commitment.commitment_id
            and row.change_type == change_type
            and row.created_day == state.day
            for row in state.reservation_change_requests
        ):
            return
        extension_days = max(
            0, proposed_return_day - int(commitment.return_day)
        )
        if self._v29_reservation_accountability() and change_type == "reschedule":
            extension_days = 0
        projected_line_rent_deltas: dict[str, float] = {}
        projected_additional_locked_rent: float | None = None
        collection_timing: str | None = None
        if change_type == "extension" and self._v27_extension_cash_audit():
            (
                extension_days,
                projected_line_rent_deltas,
                projected_additional_locked_rent,
            ) = self._extension_rent_delta(
                commitment,
                proposed_return_day=proposed_return_day,
                bill_fulfilled_only=True,
            )
            collection_timing = "immediate_on_acceptance"
        if self._v29_reservation_accountability() and change_type == "reschedule":
            projected_additional_locked_rent = 0.0
            projected_line_rent_deltas = {line.line_id: 0.0 for line in commitment.lines}
            collection_timing = "no_immediate_collection"
        change_id = f"change-{len(state.reservation_change_requests) + 1:03d}"
        change = ReservationChangeRequest(
            change_id=change_id,
            commitment_id=commitment.commitment_id,
            change_type=change_type,
            created_day=state.day,
            expires_day=state.day,
            proposed_pickup_day=proposed_pickup_day,
            proposed_return_day=proposed_return_day,
            extension_days=extension_days,
            original_dates_still_possible=self._reschedule_original_dates_possible(
                state, commitment, change_type=change_type,
                proposed_pickup_day=proposed_pickup_day,
            ),
            projected_additional_locked_rent=(
                projected_additional_locked_rent
            ),
            projected_line_rent_deltas=projected_line_rent_deltas,
            collection_timing=collection_timing,
            **({
                "previous_pickup_day": int(commitment.pickup_day),
                "previous_return_day": int(commitment.return_day),
                "previous_total_locked_rent": float(commitment.total_locked_rent),
            } if self._v29_reservation_accountability() else {}),
        )
        state.reservation_change_requests.append(change)
        if change_type == "extension":
            state.cumulative_reservation_extensions_requested += 1
        self._reservation_event(
            state,
            f"{change_type}_requested",
            request_id=commitment.request_id,
            commitment_id=commitment.commitment_id,
            details={
                "change_id": change_id,
                "proposed_pickup_day": proposed_pickup_day,
                "proposed_return_day": proposed_return_day,
                **self._reservation_change_effects(change),
                **(
                    {
                        "projected_additional_locked_rent": (
                            projected_additional_locked_rent
                        ),
                        "projected_line_rent_deltas": [
                            {
                                "line_id": line_id,
                                "additional_locked_rent": amount,
                            }
                            for line_id, amount in sorted(
                                projected_line_rent_deltas.items()
                            )
                        ],
                        "collection_timing": collection_timing,
                    }
                    if change_type == "extension"
                    and self._v27_extension_cash_audit()
                    else {}
                ),
            },
        )
        state.inbox.append(
            f"预约 {commitment.commitment_id} 有新的 {change_type} 请求，须在日终前回应。"
        )

    def _extension_rent_delta(
        self,
        commitment: ReservationCommitment,
        *,
        proposed_return_day: int,
        bill_fulfilled_only: bool = False,
    ) -> tuple[int, dict[str, float], float]:
        extension_days = proposed_return_day - int(commitment.return_day)
        if extension_days <= 0:
            raise ValueError("extension must move the return day later")
        if self._v45_reservation_difficulty():
            line_deltas = {}
            for line in commitment.lines:
                if bill_fulfilled_only and line.status != "fulfilled":
                    line_deltas[line.line_id] = 0.0
                    continue
                old_days = max(1, int(line.duration_days))
                if line.recovery_source_plan_id is not None:
                    # Recovery protects the lower payable amount. Preserve its
                    # discount basis rather than charging an unprotected price.
                    factor = 1.0 + 0.5 * (old_days - 1)
                    later = self._cent_half_up(line.locked_rent / factor * 0.5)
                    line_deltas[line.line_id] = round(later * extension_days, 2)
                elif line.locked_line_daily_rent is not None:
                    line_deltas[line.line_id] = extension_amount(
                        line.locked_line_daily_rent, old_days,
                        old_days + extension_days)
                else:
                    line_deltas[line.line_id] = extension_amount(
                        line.locked_daily_price, old_days,
                        old_days + extension_days, line.quantity)
            return extension_days, line_deltas, round(sum(line_deltas.values()), 2)
        line_deltas = {
            line.line_id: round(
                0.0
                if bill_fulfilled_only and line.status != "fulfilled"
                else (
                    (
                        line.locked_line_daily_rent
                        if line.locked_line_daily_rent is not None
                        else line.locked_rent / max(1, line.duration_days)
                        if line.recovery_source_plan_id is not None
                        else line.locked_daily_price * line.quantity
                    )
                    * extension_days
                ),
                2,
            )
            for line in commitment.lines
        }
        return extension_days, line_deltas, round(sum(line_deltas.values()), 2)

    def _schedule_late_return(
        self,
        state: WorldState,
        commitment: ReservationCommitment,
        *,
        identity: str,
    ) -> bool:
        if commitment.estimated_return_day is None:
            return False
        request = next(
            row
            for row in state.reservation_requests
            if row.request_id == commitment.request_id
        )
        late_rng = self._stable_event_rng(
            _RNG_RESERVATION_LATE_RETURN,
            *request.event_address,
            identity=identity,
        )
        if (
            late_rng.random()
            >= float(
                self.config.late_return_probability_after_extension_decline
            )
            or commitment.estimated_return_day >= state.max_days
        ):
            return False
        previous = int(commitment.estimated_return_day)
        commitment.estimated_return_day = previous + 1
        commitment.return_day = previous + 1
        for contract in state.contracts:
            if (
                contract.reservation_id == commitment.commitment_id
                and contract.status == "active"
            ):
                contract.return_day = previous + 1
                contract.estimated_return_day = previous + 1
                contract.duration_days += 1
                asset = next(
                    row for row in state.assets if row.asset_id == contract.asset_id
                )
                asset.used_life_days += 1
                self._refresh_contract_lifecycle_damage(asset, contract)
                self._charge_supplier_trial_contract_extension(
                    state,
                    asset,
                    contract,
                    duration_days=1,
                    reason="供应商试用资产逾期使用费",
                )
        for line in commitment.lines:
            selected_offer_id = line.selected_offer_id or line.offer_id
            state.reservation_calendar_blocks.append(
                CalendarBlock(
                    block_id=(
                        f"{commitment.commitment_id}-{line.line_id}-"
                        f"late-{previous:03d}"
                    ),
                    commitment_id=commitment.commitment_id,
                    line_id=line.line_id,
                    offer_id=selected_offer_id,
                    day=previous,
                    quantity=line.quantity,
                    authorized_offer_ids=[selected_offer_id],
                    selected_offer_id=selected_offer_id,
                    block_type="estimated_late_return",
                    promise_profile_id=line.promise_profile_id,
                    profile_schema_version=line.profile_schema_version,
                    service_units=line.service_units,
                    capacity_per_asset=line.capacity_per_asset,
                    staged_offer_id=line.staged_offer_id,
                )
            )
        state.cumulative_reservation_late_returns += 1
        self._reservation_event(
            state,
            "late_return_estimated",
            request_id=commitment.request_id,
            commitment_id=commitment.commitment_id,
            details={
                "promised_return_day": commitment.promised_return_day,
                "approved_return_day": commitment.approved_return_day,
                "estimated_return_day": commitment.estimated_return_day,
            },
        )
        return True

    def _apply_reservation_change(
        self,
        state: WorldState,
        change: ReservationChangeRequest,
    ) -> dict[str, Any]:
        commitment = next(
            row
            for row in state.reservation_commitments
            if row.commitment_id == change.commitment_id
        )
        accepted_delta: dict[str, Any] = {}
        if change.change_type == "reschedule":
            for block in state.reservation_calendar_blocks:
                if block.commitment_id == commitment.commitment_id:
                    block.status = "rescheduled"
            commitment.pickup_day = change.proposed_pickup_day
            commitment.return_day = change.proposed_return_day
            commitment.promised_return_day = change.proposed_return_day
            commitment.approved_return_day = change.proposed_return_day
            commitment.estimated_return_day = change.proposed_return_day
            for line in commitment.lines:
                for day in range(commitment.pickup_day, commitment.return_day):
                    state.reservation_calendar_blocks.append(
                        CalendarBlock(
                            block_id=(
                                f"{commitment.commitment_id}-{line.line_id}-"
                                f"rescheduled-{day:03d}"
                            ),
                            commitment_id=commitment.commitment_id,
                            line_id=line.line_id,
                            offer_id=line.offer_id,
                            day=day,
                            quantity=line.quantity,
                            authorized_offer_ids=list(
                                line.authorized_offer_ids or [line.offer_id]
                            ),
                            promise_profile_id=line.promise_profile_id,
                            profile_schema_version=line.profile_schema_version,
                            service_units=line.service_units,
                            capacity_per_asset=line.capacity_per_asset,
                            staged_offer_id=line.staged_offer_id,
                        )
                    )
            state.cumulative_reservation_reschedules += 1
            if self._v26_contract_state():
                reset_forecast = next(
                    row
                    for row in self.weather_forecast(
                        state, commitment.pickup_day - state.day + 1
                    )
                    if int(row["day"]) == commitment.pickup_day
                )
                reset_condition = str(reset_forecast["condition"])
                commitment.reschedule_forecast_baseline_condition = (
                    reset_condition
                )
                commitment.reschedule_last_request_condition = reset_condition
                commitment.reschedule_last_request_day = state.day
        elif change.change_type == "extension":
            previous = int(commitment.return_day)
            extension, computed_line_deltas, computed_total = (
                self._extension_rent_delta(
                    commitment,
                    proposed_return_day=change.proposed_return_day,
                    bill_fulfilled_only=self._v27_extension_cash_audit(),
                )
            )
            if self._v27_extension_cash_audit():
                if change.projected_additional_locked_rent is None:
                    change.projected_additional_locked_rent = computed_total
                    change.projected_line_rent_deltas = dict(
                        computed_line_deltas
                    )
                    change.collection_timing = "immediate_on_acceptance"
                if (
                    int(change.extension_days) != extension
                    or set(change.projected_line_rent_deltas)
                    != set(computed_line_deltas)
                    or any(
                        not math.isclose(
                            float(change.projected_line_rent_deltas[line_id]),
                            amount,
                            abs_tol=0.0049,
                        )
                        for line_id, amount in computed_line_deltas.items()
                    )
                    or not math.isclose(
                        float(change.projected_additional_locked_rent),
                        computed_total,
                        abs_tol=0.0049,
                    )
                    or change.collection_timing
                    != "immediate_on_acceptance"
                ):
                    raise AssertionError(
                        "frozen extension rent delta no longer matches the "
                        f"locked commitment: {change.change_id}"
                    )
                line_extra_rent = dict(
                    change.projected_line_rent_deltas
                )
                extra_rent = float(
                    change.projected_additional_locked_rent
                )
            else:
                line_extra_rent = computed_line_deltas
                extra_rent = computed_total
            state.cash = round(state.cash + extra_rent, 2)
            state.cumulative_rental_revenue = round(
                state.cumulative_rental_revenue + extra_rent, 2
            )
            self._add(state.current_day_ledger, "rental_revenue", extra_rent)
            commitment.total_locked_rent = round(
                commitment.total_locked_rent + extra_rent, 2
            )
            if self._v27_extension_cash_audit():
                commitment.post_pickup_rent_collected = round(
                    commitment.post_pickup_rent_collected + extra_rent, 2
                )
            commitment.return_day = change.proposed_return_day
            commitment.approved_return_day = change.proposed_return_day
            commitment.estimated_return_day = (
                self._v46_effective_return(commitment, change.proposed_return_day)
                if self._v46_return_dynamics() else change.proposed_return_day)
            exact_extension_allocations = (
                self._v32_extension_allocations(state, commitment, line_extra_rent)
                if self._v32_service_commitments() else None
            )
            for contract in state.contracts:
                if (
                    contract.reservation_id == commitment.commitment_id
                    and contract.status == "active"
                ):
                    physical_extra = (commitment.estimated_return_day - contract.return_day
                                      if self._v46_return_dynamics() else extension)
                    contract.return_day = commitment.estimated_return_day
                    contract.approved_return_day = change.proposed_return_day
                    contract.estimated_return_day = commitment.estimated_return_day
                    contract.duration_days += physical_extra
                    contract.rent_paid = round(
                        contract.rent_paid
                        + (exact_extension_allocations[contract.contract_id]
                           if exact_extension_allocations is not None
                           else extra_rent / max(1, len(commitment.assignments))),
                        2,
                    )
                    asset = next(
                        row
                        for row in state.assets
                        if row.asset_id == contract.asset_id
                    )
                    asset.used_life_days += physical_extra
                    self._refresh_contract_lifecycle_damage(asset, contract)
                    self._charge_supplier_trial_contract_extension(
                        state,
                        asset,
                        contract,
                        duration_days=physical_extra,
                        reason="供应商试用资产延租使用费",
                    )
            for line in commitment.lines:
                if (
                    self._v27_extension_cash_audit()
                    and line.status != "fulfilled"
                ):
                    continue
                line.duration_days += extension
                line.locked_rent = round(
                    line.locked_rent
                    + line_extra_rent[line.line_id],
                    2,
                )
                for day in range(previous, change.proposed_return_day):
                    selected_offer_id = line.selected_offer_id or line.offer_id
                    if self._v46_return_dynamics():
                        existing = next((block for block in state.reservation_calendar_blocks
                            if block.commitment_id == commitment.commitment_id
                            and block.line_id == line.line_id and block.day == day
                            and block.status in {"active", "bound"}), None)
                        if existing is not None:
                            existing.block_type = "approved_extension"
                            continue  # The same late day is not a second occupancy.
                    state.reservation_calendar_blocks.append(
                        CalendarBlock(
                            block_id=(
                                f"{commitment.commitment_id}-{line.line_id}-"
                                f"extension-{day:03d}"
                            ),
                            commitment_id=commitment.commitment_id,
                            line_id=line.line_id,
                            offer_id=selected_offer_id,
                            day=day,
                            quantity=line.quantity,
                            authorized_offer_ids=[selected_offer_id],
                            selected_offer_id=selected_offer_id,
                            block_type="approved_extension",
                            promise_profile_id=line.promise_profile_id,
                            profile_schema_version=line.profile_schema_version,
                            service_units=line.service_units,
                            capacity_per_asset=line.capacity_per_asset,
                            staged_offer_id=line.staged_offer_id,
                        )
                    )
            state.cumulative_reservation_extensions_approved += 1
            accepted_delta = {
                "additional_locked_rent": round(extra_rent, 2),
                "additional_rent_collected": round(extra_rent, 2),
                "line_rent_deltas": [
                    {
                        "line_id": line_id,
                        "additional_locked_rent": round(amount, 2),
                    }
                    for line_id, amount in sorted(line_extra_rent.items())
                ],
                "collection_timing": "immediate_on_acceptance",
            }
        else:
            raise ValueError(f"unknown change type: {change.change_type}")
        if change.previous_pickup_day is not None:
            accepted_delta = {
                "additional_locked_rent": 0.0,
                "additional_rent_collected": 0.0,
                **self._reservation_change_effects(change),
                **accepted_delta,
            }
        change.status = "accepted"
        change.decision_day = state.day
        self._reservation_event(
            state,
            f"{change.change_type}_accepted",
            request_id=commitment.request_id,
            commitment_id=commitment.commitment_id,
            details={
                "change_id": change.change_id,
                "pickup_day": commitment.pickup_day,
                "return_day": commitment.return_day,
                **(
                    accepted_delta
                    if change.previous_pickup_day is not None or (
                        self._v27_extension_cash_audit()
                        and change.change_type == "extension"
                    )
                    else {}
                ),
            },
        )
        self._assert_v26_reservation_financial_invariants(state)
        return (
            accepted_delta
            if change.previous_pickup_day is not None or (
                self._v27_extension_cash_audit()
                and change.change_type == "extension"
            )
            else {}
        )

    def _v42_customer_date_context(
        self, request: ReservationRequest
    ) -> tuple[str, ChoiceSession]:
        primary = next(
            (line for line in request.lines if line.required), request.lines[0]
        )
        session = self._restore_quote_context(
            request.quote_context_by_line[primary.line_id]
        )
        return (
            customer_class(
                children=session.children,
                archetype=session.archetype,
            ),
            session,
        )

    def _process_v42_pretrip_customer(
        self,
        state: WorldState,
        request: ReservationRequest,
        commitment: ReservationCommitment,
    ) -> None:
        if commitment.natural_cancellation_day == state.day:
            self._cancel_customer_reservation(
                state, commitment, reason="non_weather"
            )
            return
        if commitment.status != "confirmed" or state.day >= commitment.pickup_day:
            return
        forecast_rows = {
            int(row["day"]): row
            for row in self.weather_forecast(
                state, commitment.pickup_day - state.day + 1
            )
        }
        current = forecast_rows[commitment.pickup_day]
        regime = forecast_regime(current)
        adversity = float(ADVERSITY[regime])
        if commitment.weather_review_pickup_day != commitment.pickup_day:
            self._reset_v42_weather_review(state, commitment)
            return
        previous_max = float(commitment.weather_review_max_adversity or 0.0)
        if regime in commitment.weather_reviewed_regimes:
            return
        commitment.weather_reviewed_regimes.append(regime)
        if adversity <= previous_max:
            self._reservation_event(
                state,
                "weather_review_no_change",
                request_id=request.request_id,
                commitment_id=commitment.commitment_id,
                details={"forecast_regime": regime, "incremental_adversity": 0.0},
            )
            return

        # Record the public information before any stochastic response. Keeping
        # the original booking therefore consumes this information state too.
        commitment.weather_review_max_adversity = max(previous_max, adversity)
        deterioration = max(0.0, adversity - previous_max)
        response_rng = self._stable_event_rng(
            _RNG_RESERVATION_WEATHER_RESPONSE,
            *request.event_address,
            identity=(
                f"v42-pretrip:v{commitment.commitment_version}:"
                f"{commitment.pickup_day}:{regime}"
            ),
        )
        response_probability = min(
            0.80,
            0.65 * deterioration * float(request.forecast_trust),
        )
        if response_rng.random() >= response_probability:
            self._reservation_event(
                state,
                "weather_review_no_change",
                request_id=request.request_id,
                commitment_id=commitment.commitment_id,
                details={"forecast_regime": regime, "incremental_adversity": deterioration},
            )
            return

        duration = commitment.return_day - commitment.pickup_day
        offsets = list(
            self.dataset.metadata["service_commitment_policy"]["date_offsets"]
        )
        candidates = [
            commitment.pickup_day + int(offset)
            for offset in offsets
            if state.day < commitment.pickup_day + int(offset) <= self.config.max_days
            and commitment.pickup_day + int(offset) + duration <= state.max_days
            and commitment.pickup_day + int(offset) - state.day
            <= self.config.reservation_booking_lead_max_days
        ]
        if candidates:
            through = max(candidates)
            forecast_rows.update(
                {
                    int(row["day"]): row
                    for row in self.weather_forecast(
                        state, through - state.day + 1
                    )
                }
            )
        customer_type, session = self._v42_customer_date_context(request)
        scored: list[tuple[int, float]] = []
        all_scores: list[tuple[int, float]] = []
        for day in sorted(set(candidates)):
            candidate = forecast_rows[day]
            delta = date_utility_delta(
                customer_type=customer_type,
                current_day_type=str(self._calendar(commitment.pickup_day)["day_type"]),
                candidate_day_type=str(self._calendar(day)["day_type"]),
                current_is_school_holiday=bool(self._calendar(commitment.pickup_day).get('is_school_holiday', False)),
                candidate_is_school_holiday=bool(self._calendar(day).get('is_school_holiday', False)),
                current_condition=regime,
                candidate_condition=forecast_regime(candidate),
                shift_days=day - commitment.pickup_day,
                schedule_flexibility=request.schedule_flexibility,
                forecast_trust=request.forecast_trust,
                occasion_importance=session.occasion_importance,
            )
            all_scores.append((day, delta))
            if initial_date_is_acceptable(delta, request.schedule_flexibility):
                scored.append((day, delta))
        if scored:
            maximum = max(score for _day, score in scored)
            temperature = float(
                CONTINGENCY_PARAMETERS["initial_sampling_temperature"]
            )
            weights = [
                math.exp(temperature * (score - maximum))
                for _day, score in scored
            ]
            choice_rng = self._stable_event_rng(
                _RNG_RESERVATION_CHANGE,
                *request.event_address,
                identity=(
                    f"v42-customer-date:v{commitment.commitment_version}:"
                    f"{commitment.pickup_day}:{regime}"
                ),
            )
            selected = scored[
                weighted_index(float(choice_rng.random()), weights)
            ][0]
            self._create_change_request(
                state,
                commitment,
                change_type="reschedule",
                proposed_pickup_day=selected,
                proposed_return_day=selected + duration,
            )
            return

        best_day, best_delta = (
            max(all_scores, key=lambda row: (row[1], -abs(row[0] - commitment.pickup_day)))
            if all_scores
            else (commitment.pickup_day, -deterioration)
        )
        cancel_probability = cancellation_probability(
            best_utility_delta=best_delta,
            best_shift_days=best_day - commitment.pickup_day,
            schedule_flexibility=request.schedule_flexibility,
            notice_days=commitment.pickup_day - state.day,
        )
        if response_rng.random() < cancel_probability:
            self._cancel_customer_reservation(state, commitment, reason="weather")
        else:
            self._reservation_event(
                state,
                "weather_review_no_change",
                request_id=request.request_id,
                commitment_id=commitment.commitment_id,
                details={"forecast_regime": regime, "incremental_adversity": deterioration},
            )

    def _process_customer_reservation_changes(self, state: WorldState) -> None:
        if not self.config.reservation_disruptions_enabled:
            return
        requests = {row.request_id: row for row in state.reservation_requests}
        for commitment in sorted(
            state.reservation_commitments, key=lambda row: row.request_id
        ):
            request = requests[commitment.request_id]
            if (
                self._v42_weather_information_coherence()
                and commitment.status == "confirmed"
                and state.day < commitment.pickup_day
            ):
                self._process_v42_pretrip_customer(state, request, commitment)
            if (
                not self._v42_weather_information_coherence()
                and commitment.status == "confirmed"
                and state.day < commitment.pickup_day
            ):
                reviewed = any(
                    event.commitment_id == commitment.commitment_id
                    and event.day == state.day
                    and event.event_type
                    in {
                        "weather_review_no_change",
                        "reschedule_requested",
                        "customer_cancelled_weather",
                    }
                    for event in state.reservation_events
                )
                if not reviewed and commitment.pickup_day - state.day <= 14:
                    forecast = next(
                        row
                        for row in self.weather_forecast(
                            state, commitment.pickup_day - state.day + 1
                        )
                        if int(row["day"]) == commitment.pickup_day
                    )
                    baseline_condition = (
                        commitment.reschedule_forecast_baseline_condition
                        if self._v26_contract_state()
                        and commitment.reschedule_forecast_baseline_condition
                        else request.forecast_condition
                    )
                    reference_adversity = self._forecast_adversity(
                        str(baseline_condition)
                    )
                    if (
                        self._v26_contract_state()
                        and commitment.reschedule_last_request_condition
                    ):
                        reference_adversity = max(
                            reference_adversity,
                            self._forecast_adversity(
                                commitment.reschedule_last_request_condition
                            ),
                        )
                    deterioration = max(
                        0.0,
                        self._forecast_adversity(str(forecast["condition"]))
                        - reference_adversity,
                    )
                    response_rng = self._stable_event_rng(
                        _RNG_RESERVATION_WEATHER_RESPONSE,
                        *request.event_address,
                        identity=f"pretrip:{state.day}",
                    )
                    uncertainty_probability = 0.12 * (
                        1.0 - float(forecast["condition_accuracy"])
                    )
                    if self._v26_contract_state():
                        prior_requests = self._reschedule_request_count(
                            state, commitment.commitment_id
                        )
                        repeat_decay = self._v26_reschedule_repeat_decay(
                            prior_requests
                        )
                        change_probability = min(
                            0.85,
                            repeat_decay
                            * (0.08 + uncertainty_probability)
                            + 0.65
                            * deterioration
                            * request.forecast_trust,
                        )
                    else:
                        change_probability = min(
                            0.85,
                            0.08
                            + 0.65 * deterioration * request.forecast_trust
                            + uncertainty_probability,
                        )
                    if response_rng.random() < change_probability:
                        shift_rng = self._stable_event_rng(
                            _RNG_RESERVATION_CHANGE,
                            *request.event_address,
                            identity=f"pretrip-shift:{state.day}",
                        )
                        shift = int(shift_rng.integers(1, 4))
                        can_reschedule = (
                            commitment.pickup_day + shift
                            <= self.config.max_days
                            and commitment.return_day + shift
                            <= state.max_days
                            and shift_rng.random() < request.schedule_flexibility
                        )
                        if can_reschedule:
                            self._create_change_request(
                                state,
                                commitment,
                                change_type="reschedule",
                                proposed_pickup_day=commitment.pickup_day + shift,
                                proposed_return_day=commitment.return_day + shift,
                            )
                            if self._v26_contract_state():
                                commitment.reschedule_last_request_condition = str(
                                    forecast["condition"]
                                )
                                commitment.reschedule_last_request_day = state.day
                        else:
                            self._cancel_customer_reservation(state, commitment)
                    else:
                        self._reservation_event(
                            state,
                            "weather_review_no_change",
                            request_id=request.request_id,
                            commitment_id=commitment.commitment_id,
                        )
            if (
                commitment.status == "active"
                and not self._v46_return_dynamics()
                and not self._v47_autonomous_rentals()
                and commitment.estimated_return_day is not None
                and state.day == commitment.estimated_return_day - 1
                and not any(
                    row.commitment_id == commitment.commitment_id
                    and row.change_type == "extension"
                    for row in state.reservation_change_requests
                )
            ):
                response_rng = self._stable_event_rng(
                    _RNG_RESERVATION_WEATHER_RESPONSE,
                    *request.event_address,
                    identity="trip-extension",
                )
                adversity = self._forecast_adversity(
                    self._demand_weather_regime(state.day)
                )
                probability = min(
                    0.70,
                    0.14
                    + 0.25 * adversity
                    + 0.22 * request.schedule_flexibility,
                )
                if self._v45_reservation_difficulty():
                    # Trip changes depend on the original activity, elapsed stay
                    # and the public contractual price, not on shop availability.
                    activity_base = (
                        0.19 if any(line.service_line == "tent" for line in commitment.lines)
                        else 0.025 if any(line.service_line == "camp_table" for line in commitment.lines)
                        else 0.10
                    )
                    elapsed_days = max(1, commitment.return_day - commitment.pickup_day)
                    base = activity_base * max(0.55, 1.15 - 0.13 * (elapsed_days - 1))
                    base *= 0.75 + 0.5 * request.schedule_flexibility
                    base *= 1.0 + 0.15 * adversity
                    _, _, one_day_rent = self._extension_rent_delta(
                        commitment,
                        proposed_return_day=commitment.return_day + 1,
                        bill_fulfilled_only=True,
                    )
                    price_ratio = one_day_rent / max(1.0, commitment.total_locked_rent)
                    probability = max(0.0, min(0.35, base * math.exp(-0.9 * price_ratio)))
                if response_rng.random() < probability:
                    extension_rng = self._stable_event_rng(
                        _RNG_RESERVATION_CHANGE,
                        *request.event_address,
                        identity="trip-extension-days",
                    )
                    extension = (
                        int(extension_rng.choice([1, 2], p=[0.86, 0.14]))
                        if self._v45_reservation_difficulty()
                        else int(extension_rng.integers(1, 4))
                    )
                    proposed = min(
                        state.max_days,
                        int(commitment.estimated_return_day) + extension,
                    )
                    if proposed > int(commitment.estimated_return_day):
                        self._create_change_request(
                            state,
                            commitment,
                            change_type="extension",
                            proposed_pickup_day=commitment.pickup_day,
                            proposed_return_day=proposed,
                        )

    def _prepare_reservation_day(self, state: WorldState) -> None:
        if not self.config.reservations_enabled or state.done:
            return
        if state.day > self.config.max_days:
            return
        self._process_customer_reservation_changes(state)
        self._generate_reservation_requests(state)

    def _expire_reservation_requests(self, state: WorldState) -> None:
        if not self.config.reservations_enabled:
            return
        for request in state.reservation_requests:
            if (
                request.status != "pending_quote"
                or request.expires_day != state.day
            ):
                continue
            request.status = "expired_unquoted"
            request.decision_day = state.day
            state.cumulative_reservations_expired += 1
            self._reservation_event(
                state, "inquiry_expired_unquoted", request_id=request.request_id
            )
        for change in state.reservation_change_requests:
            if change.status != "pending" or change.expires_day != state.day:
                continue
            change.status = "expired_unanswered"
            change.decision_day = state.day
            commitment = next(
                row
                for row in state.reservation_commitments
                if row.commitment_id == change.commitment_id
            )
            late = (
                self._schedule_late_return(
                    state,
                    commitment,
                    identity=(
                        "expired:"
                        + self._reservation_change_rng_identity(
                            commitment, change
                        )
                    ),
                )
                if change.change_type == "extension"
                else False
            )
            self._reservation_event(
                state,
                f"{change.change_type}_expired",
                request_id=commitment.request_id,
                commitment_id=commitment.commitment_id,
                details={
                    "change_id": change.change_id, "late_return": late,
                    **self._reservation_change_effects(change),
                    **({"additional_locked_rent": 0.0, "additional_rent_collected": 0.0}
                       if change.previous_pickup_day is not None else {}),
                },
            )
            if self._v32_service_commitments():
                self._resolve_v32_unavailable_dates(state, change, expired=True)

    @staticmethod
    def _public_reservation_line(line: ReservationLine) -> dict[str, Any]:
        row = asdict(line)
        if line.locked_line_daily_rent is None:
            row.pop("locked_line_daily_rent", None)
            row.pop("locked_line_pricing_basis", None)
        if not line.eligible_profile_ids and line.promise_profile_id is None:
            for key in (
                "service_key",
                "service_units",
                "public_hard_requirements",
                "public_preferences",
                "eligible_profile_ids",
                "eligible_profile_family_ids",
                "promise_profile_id",
                "profile_family_id",
                "profile_schema_version",
                "capacity_per_asset",
                "asset_quantity",
                "staged_offer_id",
            ):
                row.pop(key, None)
        return row

    @classmethod
    def _public_reservation_request(
        cls, request: ReservationRequest
    ) -> dict[str, Any]:
        public = {
            key: value
            for key, value in asdict(request).items()
            if key
            not in {
                "event_address",
                "quote_context_by_date",
                "service_offer_audit",
                "forecast_trust",
                "customer_segment",
                "shared_price_beta",
                "storage_burden_base",
                "future_use_expectation_base",
                "return_burden_multiplier",
                "request_budget_multiplier",
                "schedule_flexibility",
                "fallback_offer_ids",
                "fallback_offer_ids_by_line",
                "inquiry_latent_offer_ids_by_line",
                "inquiry_latent_price_ceiling_by_line",
                "authorized_offer_ids_by_line",
                "authorized_price_ceiling_by_line",
                "recovery_offer_ids_by_line",
                "recovery_price_ceiling_by_line",
                "latent_snapshot_provenance",
                "quote_context_by_line",
                "candidate_deposit_by_line",
                "inquiry_public_envelope_by_line",
                "inquiry_profile_member_ids_by_line",
                "inquiry_profile_price_ceiling_by_line",
                "inquiry_profile_deposit_by_line",
                "submitted_contract_options",
                "selected_contract_option_id",
                "reported_contract_assessments",
                "feedback_response_available",
                "contract_feedback_audit",
                "submitted_line_options",
                "customer_authorized_plans",
                "customer_preferred_plan_id",
                "manager_confirmed_plan_id",
                "plan_confirmation_origin",
                "original_inquiry_pickup_day",
                "original_inquiry_return_day",
                "manager_confirmation_day",
                "internal_plan_count",
                "acceptable_plan_count",
                "reported_decline_feedback",
                "plan_set_feedback_audit",
                "reservation_contract_mode",
                "catalog_snapshot_sha256",
                "promise_profile_schema_version",
                "profile_deposit_policy_id",
            }
        }
        for key in ("pickup_date_options", "service_offer_terms"):
            if not public.get(key):
                public.pop(key, None)
        profile_mode = request.reservation_contract_mode in {
            "promise_profiles_v1",
            "adaptive_pairwise_exact_quotes_v1",
            "complete_contract_options_v1",
            "factorized_line_options_plan_set_v1",
        }
        if profile_mode:
            public.update(
                {
                    "reservation_contract_mode": request.reservation_contract_mode,
                    "catalog_snapshot_sha256": request.catalog_snapshot_sha256,
                    "promise_profile_schema_version": (
                        request.promise_profile_schema_version
                    ),
                    "profile_deposit_policy_id": (
                        request.profile_deposit_policy_id
                    ),
                }
            )
        else:
            public["lines"] = [
                cls._public_reservation_line(line) for line in request.lines
            ]
        if request.status in {"pending_quote", "pending_service_offer"}:
            public.pop("total_locked_rent", None)
            public.pop("prepayment_due", None)
            public.pop("commitment_id", None)
            public.pop("quote_action", None)
            public.pop("quote_day", None)
            if profile_mode:
                public["lines"] = [
                    {
                        "line_id": line.line_id,
                        "required": line.required,
                        "service_key": line.service_key,
                        "service_units": line.service_units,
                        "duration_days": line.duration_days,
                        "hard_requirements": dict(
                            line.public_hard_requirements
                        ),
                        "stated_preferences": [
                            dict(row) for row in line.public_preferences
                        ],
                        "promise_profile_family_ids": list(
                            line.eligible_profile_family_ids
                        ),
                    }
                    for line in request.lines
                ]
            else:
                public["lines"] = [
                    {
                        "line_id": line.line_id,
                        "required": line.required,
                        "service_line": line.service_line,
                        "quantity": line.quantity,
                        "customer_units": line.customer_units,
                        "duration_days": line.duration_days,
                        "candidate_offer_ids": list(
                            line.candidate_offer_ids
                        ),
                    }
                    for line in request.lines
                ]
        elif profile_mode:
            public["lines"] = [
                cls._public_reservation_line(line) for line in request.lines
            ]
            if request.reservation_contract_mode == "complete_contract_options_v1":
                public.update(
                    {
                        "submitted_contract_options": [
                            dict(row) for row in request.submitted_contract_options
                        ],
                        "selected_contract_option_id": (
                            request.selected_contract_option_id
                        ),
                        "contract_assessments": [
                            dict(row)
                            for row in request.reported_contract_assessments
                        ],
                    }
                )
            elif request.reservation_contract_mode == (
                "factorized_line_options_plan_set_v1"
            ):
                public.update(
                    {
                        "submitted_line_options": [
                            dict(row) for row in request.submitted_line_options
                        ],
                        "internal_plan_count": int(request.internal_plan_count),
                        "customer_authorized_plans": [
                            dict(row) for row in request.customer_authorized_plans
                        ],
                        "customer_preferred_plan_id": (
                            request.customer_preferred_plan_id
                        ),
                        "manager_confirmed_plan_id": (
                            request.manager_confirmed_plan_id
                        ),
                        "plan_confirmation_origin": (
                            request.plan_confirmation_origin
                        ),
                        "manager_confirmation_day": (
                            request.manager_confirmation_day
                        ),
                        **(
                            {
                                "decline_feedback": dict(
                                    request.reported_decline_feedback
                                )
                            }
                            if request.status == "customer_declined_quote"
                            else {}
                        ),
                    }
                )
        return public

    @classmethod
    def _public_reservation_commitment(
        cls, commitment: ReservationCommitment,
    ) -> dict[str, Any]:
        public = asdict(commitment)
        if (commitment.partial_fulfillment_inquiry_day is None
                and not commitment.partial_fulfillment_missing_line_ids):
            for key in (
                "partial_fulfillment_inquiry_day",
                "partial_fulfillment_missing_line_ids",
                "partial_fulfillment_accepted",
                "partial_fulfillment_selected",
                "partial_fulfillment_compensation",
            ):
                public.pop(key, None)
        for key in ("initial_agreed_pickup_day", "initial_agreed_return_day"):
            if public.get(key) is None:
                public.pop(key, None)
        for hidden_key in (
            "reschedule_forecast_baseline_condition",
            "reschedule_last_request_condition",
            "reschedule_last_request_day",
            "rent_collected_at_pickup",
            "post_pickup_rent_collected",
            "weather_review_pickup_day",
            "weather_review_baseline_regime",
            "weather_review_max_adversity",
            "weather_reviewed_regimes",
            "natural_cancellation_day",
            "service_guarantee",
        ):
            public.pop(hidden_key, None)
        public["lines"] = [
            cls._public_reservation_line(line) for line in commitment.lines
        ]
        public["assignments"] = [
            {
                "line_id": assignment.line_id,
                "asset_id": assignment.asset_id,
                "assigned_day": assignment.assigned_day,
                "status": assignment.status,
            }
            for assignment in commitment.assignments
        ]
        return public

    @staticmethod
    def _reservation_change_effects(change: ReservationChangeRequest) -> dict[str, Any]:
        if change.previous_pickup_day is None or change.previous_return_day is None:
            return {}
        previous_duration = change.previous_return_day - change.previous_pickup_day
        proposed_duration = change.proposed_return_day - change.proposed_pickup_day
        rent_delta = float(change.projected_additional_locked_rent or 0.0)
        return {
            **({
                "original_dates_still_possible": change.original_dates_still_possible,
                "on_decline": "retain_original_contract" if change.original_dates_still_possible else "customer_cancels_with_prepayment_refund",
                "on_expiry": "retain_original_contract" if change.original_dates_still_possible else "customer_cancels_with_prepayment_refund",
            } if change.original_dates_still_possible is not None else {}),
            "previous_pickup_day": change.previous_pickup_day,
            "previous_return_day": change.previous_return_day,
            "proposed_pickup_day": change.proposed_pickup_day,
            "proposed_return_day": change.proposed_return_day,
            "pickup_shift_days": change.proposed_pickup_day - change.previous_pickup_day,
            "return_shift_days": change.proposed_return_day - change.previous_return_day,
            "previous_duration_days": previous_duration,
            "proposed_duration_days": proposed_duration,
            "duration_delta_days": proposed_duration - previous_duration,
            "previous_total_locked_rent": change.previous_total_locked_rent,
            "projected_total_locked_rent": round(float(change.previous_total_locked_rent) + rent_delta, 2),
            "projected_additional_locked_rent": rent_delta,
            "projected_additional_rent_collected": rent_delta if change.change_type == "extension" else 0.0,
            "projected_line_rent_deltas": [
                {"line_id": line_id, "additional_locked_rent": amount}
                for line_id, amount in sorted(change.projected_line_rent_deltas.items())
            ],
            "collection_timing": change.collection_timing,
            **({"extension_days": change.extension_days} if change.change_type == "extension" else {}),
        }

    @classmethod
    def _reservation_change_view(
        cls, change: ReservationChangeRequest, *, extension_cash_audit: bool,
    ) -> dict[str, Any]:
        public = asdict(change)
        public.pop("weather_forecast_snapshot_by_day", None)
        if change.original_dates_still_possible is None:
            public.pop("original_dates_still_possible", None)
        for key in ("previous_pickup_day", "previous_return_day", "previous_total_locked_rent"):
            public.pop(key, None)
        effects = cls._reservation_change_effects(change)
        if effects:
            if change.change_type != "extension":
                public.pop("extension_days", None)
            return {**public, **effects}
        if (
            not extension_cash_audit
            or change.change_type != "extension"
        ):
            for key in (
                "projected_additional_locked_rent",
                "projected_line_rent_deltas",
                "collection_timing",
            ):
                public.pop(key, None)
        elif "projected_line_rent_deltas" in public:
            public["projected_line_rent_deltas"] = [
                {
                    "line_id": line_id,
                    "additional_locked_rent": amount,
                }
                for line_id, amount in sorted(
                    change.projected_line_rent_deltas.items()
                )
            ]
        return public

    def _public_reservation_change(self, change: ReservationChangeRequest) -> dict[str, Any]:
        return self._reservation_change_view(
            change, extension_cash_audit=self._v27_extension_cash_audit()
        )

    @staticmethod
    def _v22_compact_quote_history_record(
        request: ReservationRequest,
        *,
        quote_outcome: str,
    ) -> dict[str, Any]:
        submitted_by_line = {
            str(row["line_id"]): row for row in request.submitted_line_options
        }
        lines = []
        for line in request.lines:
            options = []
            prices = []
            for option in submitted_by_line.get(line.line_id, {}).get(
                "options", []
            ):
                if option.get("disposition") == "omit":
                    options.append(
                        {
                            "line_option_id": str(option.get("option_id", "omit")),
                            "disposition": "omit",
                        }
                    )
                    continue
                price = float(option["daily_unit_price"])
                prices.append(price)
                options.append(
                    {
                        "line_option_id": str(option["option_id"]),
                        "disposition": "quote",
                        "offer_id": str(option["offer_id"]),
                        "promise_profile_id": str(
                            option["promise_profile_id"]
                        ),
                        "daily_unit_price": price,
                    }
                )
            lines.append(
                {
                    "line_id": line.line_id,
                    "required": bool(line.required),
                    "service_key": line.service_key,
                    "service_units": line.service_units,
                    "duration_days": int(line.duration_days),
                    "hard_requirements": dict(line.public_hard_requirements),
                    "stated_preferences": [
                        dict(row) for row in line.public_preferences
                    ],
                    "quote_options": options,
                    "daily_unit_price_range": (
                        None
                        if not prices
                        else {"minimum": min(prices), "maximum": max(prices)}
                    ),
                    "current_line_status": line.status,
                }
            )
        plan_references = []
        for plan in request.customer_authorized_plans:
            plan_references.append(
                {
                    "plan_id": str(plan["plan_id"]),
                    "line_option_ids": {
                        str(line["line_id"]): str(
                            line.get("line_option_id", "omit")
                        )
                        for line in plan.get("lines", [])
                    },
                    "total_rent": float(plan["total_rent"]),
                    "total_security_deposit": float(
                        plan["total_security_deposit"]
                    ),
                }
            )
        return {
            "request_id": request.request_id,
            "commitment_id": request.commitment_id,
            "created_day": int(request.created_day),
            "quote_day": request.quote_day,
            "pickup_day": int(request.pickup_day),
            "return_day": int(request.return_day),
            "booking_lead_days": int(request.booking_lead_days),
            "visible_party_context": request.visible_party_context,
            "forecast_condition": request.forecast_condition,
            "forecast_accuracy": float(request.forecast_accuracy),
            "quote_action": request.quote_action,
            "quote_outcome": quote_outcome,
            "current_request_status": request.status,
            "total_locked_rent": float(request.total_locked_rent),
            "internal_plan_count": int(request.internal_plan_count),
            "lines": lines,
            "customer_authorized_plan_count": len(
                request.customer_authorized_plans
            ),
            "customer_authorized_plan_references": plan_references,
            "customer_preferred_plan_id": request.customer_preferred_plan_id,
            "manager_confirmed_plan_id": request.manager_confirmed_plan_id,
            "plan_confirmation_origin": request.plan_confirmation_origin,
            "decline_feedback": dict(request.reported_decline_feedback),
            **(
                {
                    "service_reputation_snapshot": dict(
                        request.service_reputation_snapshot
                    )
                }
                if request.service_reputation_snapshot
                else {}
            ),
        }

    @classmethod
    def _v22_reservation_detail_record(
        cls,
        request: ReservationRequest,
        commitment: ReservationCommitment | None,
        *,
        changes: list[ReservationChangeRequest],
        matched_identifiers: list[str],
        sections: tuple[str, ...],
        v26_contract_state: bool = False,
        v27_extension_cash_audit: bool = False,
        v29_public_timing: bool = False,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "request_id": request.request_id,
            "commitment_id": request.commitment_id,
            "matched_identifiers": matched_identifiers,
            "current_request_status": request.status,
            "current_commitment_status": (
                None if commitment is None else commitment.status
            ),
        }
        if "inquiry_terms" in sections:
            if not v26_contract_state:
                raise ValueError("inquiry_terms requires tool interface 22")
            record["inquiry_terms"] = {
                "created_day": int(request.created_day),
                "decision_day": request.decision_day,
                "expires_day": int(request.expires_day),
                "original_pickup_day": int(
                    request.original_inquiry_pickup_day
                    if request.original_inquiry_pickup_day is not None
                    else request.pickup_day
                ),
                "original_return_day": int(
                    request.original_inquiry_return_day
                    if request.original_inquiry_return_day is not None
                    else request.return_day
                ),
                "booking_lead_days": int(request.booking_lead_days),
                "visible_party_context": request.visible_party_context,
                "forecast_condition": request.forecast_condition,
                "forecast_accuracy": float(request.forecast_accuracy),
                "reservation_contract_mode": request.reservation_contract_mode,
                "promise_profile_schema_version": (
                    request.promise_profile_schema_version
                ),
                "profile_deposit_policy_id": request.profile_deposit_policy_id,
                **(
                    {
                        "service_reputation_snapshot": dict(
                            request.service_reputation_snapshot
                        )
                    }
                    if request.service_reputation_snapshot
                    else {}
                ),
                "demand_lines": [
                    {
                        "line_id": line.line_id,
                        "required": bool(line.required),
                        "service_key": line.service_key,
                        "service_units": line.service_units,
                        "duration_days": int(line.duration_days),
                        "hard_requirements": dict(
                            line.public_hard_requirements
                        ),
                        "stated_preferences": [
                            dict(row) for row in line.public_preferences
                        ],
                    }
                    for line in request.lines
                ],
            }
        if "current_terms" in sections:
            locked_lines = []
            if commitment is not None:
                locked_lines = [
                    {
                        "line_id": line.line_id,
                        "required": bool(line.required),
                        "service_key": line.service_key,
                        "service_units": line.service_units,
                        "duration_days": int(line.duration_days),
                        "offer_id": line.offer_id,
                        "promise_profile_id": line.promise_profile_id,
                        "profile_family_id": line.profile_family_id,
                        "capacity_per_asset": line.capacity_per_asset,
                        "asset_quantity": line.asset_quantity,
                        "locked_daily_price": float(line.locked_daily_price),
                        "locked_rent": float(line.locked_rent),
                        "locked_security_deposit": float(
                            line.locked_security_deposit
                        ),
                        **(
                            {
                                "authorized_line_rent": float(
                                    line.authorized_line_rent or 0.0
                                ),
                                "authorized_security_deposit": float(
                                    line.authorized_security_deposit or 0.0
                                ),
                                "recovery_price_credit": float(
                                    line.recovery_price_credit
                                ),
                                "recovery_deposit_credit": float(
                                    line.recovery_deposit_credit
                                ),
                                "recovery_source_plan_id": (
                                    line.recovery_source_plan_id
                                ),
                            }
                            if line.recovery_source_plan_id is not None
                            else {}
                        ),
                    }
                    for line in commitment.lines
                ]
            if v26_contract_state:
                current_total = (
                    None
                    if commitment is None
                    else round(
                        sum(float(line.locked_rent) for line in commitment.lines),
                        2,
                    )
                )
                if commitment is not None and not math.isclose(
                    float(commitment.total_locked_rent),
                    float(current_total),
                    abs_tol=0.0049,
                ):
                    raise AssertionError(
                        "commitment total_locked_rent does not equal the "
                        f"locked-line sum: {commitment.commitment_id}"
                    )
                record["current_terms"] = {
                    "contract_formed": commitment is not None,
                    "confirmed_day": (
                        None if commitment is None else int(commitment.confirmed_day)
                    ),
                    "status": (
                        request.status
                        if commitment is None
                        else commitment.status
                    ),
                    "pickup_day": int(
                        request.pickup_day
                        if commitment is None
                        else commitment.pickup_day
                    ),
                    "return_day": int(
                        request.return_day
                        if commitment is None
                        else commitment.return_day
                    ),
                    "locked_lines": locked_lines,
                    "total_locked_rent": current_total,
                    "outstanding_rent": (
                        None
                        if commitment is None
                        else float(commitment.outstanding_rent)
                    ),
                    "unearned_revenue": (
                        None
                        if commitment is None
                        else float(commitment.unearned_revenue)
                    ),
                }
            else:
                record["current_terms"] = {
                    "created_day": int(request.created_day),
                    "decision_day": request.decision_day,
                    "expires_day": int(request.expires_day),
                    "pickup_day": int(request.pickup_day),
                    "return_day": int(request.return_day),
                    "booking_lead_days": int(request.booking_lead_days),
                    "visible_party_context": request.visible_party_context,
                    "forecast_condition": request.forecast_condition,
                    "forecast_accuracy": float(request.forecast_accuracy),
                    "reservation_contract_mode": request.reservation_contract_mode,
                    "promise_profile_schema_version": (
                        request.promise_profile_schema_version
                    ),
                    "profile_deposit_policy_id": request.profile_deposit_policy_id,
                    "demand_lines": [
                        {
                            "line_id": line.line_id,
                            "required": bool(line.required),
                            "service_key": line.service_key,
                            "service_units": line.service_units,
                            "duration_days": int(line.duration_days),
                            "hard_requirements": dict(
                                line.public_hard_requirements
                            ),
                            "stated_preferences": [
                                dict(row) for row in line.public_preferences
                            ],
                        }
                        for line in request.lines
                    ],
                    "locked_lines": locked_lines,
                    "total_locked_rent": float(request.total_locked_rent),
                    "prepayment_due": float(request.prepayment_due),
                }
        if "submitted_quote" in sections:
            record["submitted_quote"] = {
                "quote_action": request.quote_action,
                "quote_day": request.quote_day,
                "internal_plan_count": int(request.internal_plan_count),
                "submitted_line_options": [
                    dict(row) for row in request.submitted_line_options
                ],
            }
        if "customer_authorization" in sections:
            record["customer_authorization"] = {
                "customer_authorized_plans": [
                    dict(row) for row in request.customer_authorized_plans
                ],
                "customer_preferred_plan_id": (
                    request.customer_preferred_plan_id
                ),
                "manager_confirmed_plan_id": request.manager_confirmed_plan_id,
                "plan_confirmation_origin": request.plan_confirmation_origin,
                "manager_confirmation_day": request.manager_confirmation_day,
                "decline_feedback": dict(request.reported_decline_feedback),
                "authorization_is_exhaustive": False,
            }
        if "fulfillment" in sections:
            record["fulfillment"] = (
                None
                if commitment is None
                else {
                    "confirmed_day": int(commitment.confirmed_day),
                    "status": commitment.status,
                    "pickup_day": int(commitment.pickup_day),
                    "return_day": int(commitment.return_day),
                    "original_pickup_day": commitment.original_pickup_day,
                    "original_return_day": commitment.original_return_day,
                    "promised_return_day": commitment.promised_return_day,
                    "approved_return_day": commitment.approved_return_day,
                    "estimated_return_day": commitment.estimated_return_day,
                    "actual_return_day": commitment.actual_return_day,
                    "prepayment_collected": float(
                        commitment.prepayment_collected
                    ),
                    **(
                        {
                            "rent_due_at_pickup": commitment.rent_due_at_pickup,
                            "prepayment_refunded": float(
                                commitment.prepayment_refunded
                            ),
                            "compensation_paid": float(
                                commitment.compensation_paid
                            ),
                        }
                        if v26_contract_state
                        else {}
                    ),
                    **(
                        {
                            key: value
                            for key, value in cls._v27_reservation_cash_history(
                                commitment
                            ).items()
                            if key
                            in {
                                "rent_collected_at_pickup",
                                "post_pickup_rent_collected",
                                "gross_rent_cash_collected",
                                "net_rent_cash_collected",
                            }
                        }
                        if v27_extension_cash_audit
                        else {}
                    ),
                    "outstanding_rent": float(commitment.outstanding_rent),
                    "unearned_revenue": float(commitment.unearned_revenue),
                    "optional_line_credits": float(
                        commitment.optional_line_credits
                    ),
                    "line_statuses": [
                        {
                            "line_id": line.line_id,
                            "status": line.status,
                            "selected_offer_id": line.selected_offer_id,
                            "fulfillment_mode": line.fulfillment_mode,
                            "credit_amount": float(line.credit_amount),
                        }
                        for line in commitment.lines
                    ],
                    "assignments": [
                        {
                            "line_id": assignment.line_id,
                            "asset_id": assignment.asset_id,
                            "assigned_day": assignment.assigned_day,
                            "status": assignment.status,
                        }
                        for assignment in commitment.assignments
                    ],
                }
            )
        if "changes" in sections:
            record["changes"] = [
                cls._reservation_change_view(change, extension_cash_audit=v27_extension_cash_audit)
                for change in changes
            ]
        if "risk" in sections:
            record["risk"] = (
                None
                if commitment is None
                else {
                    "risk_exposed_day": commitment.risk_exposed_day,
                    "fallback_inquiry_day": commitment.fallback_inquiry_day,
                    **({"fallback_query_status": cls._fallback_query_status(commitment)}
                       if v29_public_timing else {}),
                    "fallback_revealed_offer_ids_by_line": dict(
                        commitment.fallback_revealed_offer_ids_by_line
                    ),
                    "failure_cause": commitment.failure_cause,
                    **({
                        "partial_fulfillment_missing_line_ids": list(
                            commitment.partial_fulfillment_missing_line_ids),
                        "partial_fulfillment_inquiry_day": (
                            commitment.partial_fulfillment_inquiry_day),
                        "partial_fulfillment_accepted": (
                            commitment.partial_fulfillment_accepted),
                        "partial_fulfillment_selected": (
                            commitment.partial_fulfillment_selected),
                        "partial_fulfillment_compensation": (
                            commitment.partial_fulfillment_compensation),
                    } if commitment.partial_fulfillment_missing_line_ids else {}),
                    **(
                        {
                            "preauthorized_recovery_plan_count": len(
                                commitment.preauthorized_recovery_plans
                            ),
                            "selected_recovery_plan_id": (
                                commitment.selected_recovery_plan_id
                            ),
                            "recovery_path": commitment.recovery_path,
                        }
                        if commitment is not None
                        and commitment.original_confirmed_plan_id is not None
                        else {}
                    ),
                }
            )
        return record

    @staticmethod
    def _public_reservation_calendar_block(block: CalendarBlock) -> dict[str, Any]:
        row = {key: value for key, value in asdict(block).items() if key != "block_id"}
        if block.promise_profile_id is None:
            for key in (
                "promise_profile_id",
                "profile_schema_version",
                "service_units",
                "capacity_per_asset",
                "staged_offer_id",
            ):
                row.pop(key, None)
        return row

    @staticmethod
    def _fallback_query_status(commitment: ReservationCommitment) -> str:
        if commitment.fallback_inquiry_day is None:
            return "not_inquired"
        has_candidates = bool(commitment.preauthorized_recovery_plans) or any(
            commitment.fallback_revealed_offer_ids_by_line.values()
        )
        return "inquired_with_candidates" if has_candidates else "inquired_without_candidates"

    def reservation_inbox(self, state: WorldState) -> dict[str, Any]:
        requests = [
            self._public_reservation_request(request)
            for request in state.reservation_requests
            if request.status == "pending_quote"
            and request.created_day == state.day
        ]
        plan_confirmations = [
            self._public_reservation_request(request)
            for request in state.reservation_requests
            if request.status == "awaiting_manager_confirmation"
            and request.quote_day == state.day
        ]
        changes = [
            self._public_reservation_change(change)
            for change in state.reservation_change_requests
            if change.status == "pending" and change.created_day == state.day
        ]
        risk_alerts = [
            {
                "commitment_id": commitment.commitment_id,
                "pickup_day": commitment.pickup_day,
                "return_day": commitment.return_day,
                "locked_lines": [
                    self._public_reservation_line(line)
                    for line in commitment.lines
                ],
                "recovery_deadline": "end_of_current_business_day",
                **({
                    "missing_optional_line_ids": list(commitment.partial_fulfillment_missing_line_ids),
                    "partial_fulfillment_inquiry_day": commitment.partial_fulfillment_inquiry_day,
                    "partial_fulfillment_accepted": commitment.partial_fulfillment_accepted,
                    "partial_fulfillment_selected": commitment.partial_fulfillment_selected,
                } if self._v45_reservation_difficulty() else {}),
                **({
                    "fallback_query_status": self._current_fallback_query_status(commitment),
                    "fallback_inquiry_day": commitment.fallback_inquiry_day,
                } if self._v29_reservation_accountability() else {}),
                "available_actions": [
                    "inquire_fallback",
                    *(
                        ["select_preauthorized_plan"]
                        if self._v24_reservation_operations()
                        else []
                    ),
                    "select_authorized_fallback",
                    *(["inquire_partial_fulfillment", "select_partial_fulfillment"]
                      if self._v45_reservation_difficulty()
                      and commitment.partial_fulfillment_missing_line_ids else []),
                    *(
                        ["cancel_reservation_commitment"]
                        if self._v29_reservation_accountability()
                        else ["cancel_early"]
                    ),
                ],
            }
            for commitment in state.reservation_commitments
            if commitment.status == "at_risk"
        ]
        return {
            "day": state.day,
            "requests": requests,
            "request_count": len(requests),
            "plan_confirmations": plan_confirmations,
            "plan_confirmation_count": len(plan_confirmations),
            "changes": changes,
            "change_count": len(changes),
            "risk_alerts": risk_alerts,
            "risk_alert_count": len(risk_alerts),
            "deadline": "end_of_current_business_day",
            "unanswered_default": "expired_unquoted",
            "unanswered_plan_confirmation_default": "customer_preferred_plan",
        }

    def _reservation_profile_representatives(
        self, profile_id: str
    ) -> list[dict[str, Any]]:
        catalog = self.reservation_promise_catalog
        if catalog is None:
            raise AssertionError("reservation promise catalog is unavailable")
        profile = catalog.by_id[profile_id]
        members = [self.offers[offer_id] for offer_id in profile.member_offer_ids]
        if not members:
            return []
        cheapest = min(members, key=lambda row: (float(row.unit_cost), row.offer_id))
        guarded_fast = [
            row
            for row in members
            if float(row.unit_cost) <= 1.5 * float(cheapest.unit_cost) + 1e-9
        ]
        fastest = min(
            guarded_fast,
            key=lambda row: (
                self.supplier_delivery_estimate(row.seller_id),
                float(row.unit_cost),
                row.offer_id,
            ),
        )
        guarded_durable = [
            row
            for row in members
            if float(row.unit_cost) <= 2.0 * float(cheapest.unit_cost) + 1e-9
        ]
        durability_order = {"low": 0, "medium": 1, "high": 2}

        def durability_key(offer: Any) -> tuple[int, float, str]:
            public = self.public_offer(offer.offer_id)
            tier = str(
                dict(public.get("lifecycle_profile") or {}).get(
                    "relative_durability_tier", "low"
                )
            )
            return (
                -durability_order.get(tier, 0),
                float(offer.unit_cost),
                offer.offer_id,
            )

        durable = min(guarded_durable, key=durability_key)
        roles = [
            ("lowest_purchase_cost", cheapest),
            ("fastest_within_1_5x_lowest_cost", fastest),
            ("highest_durability_tier_within_2x_lowest_cost", durable),
        ]
        result = []
        seen: set[str] = set()
        for role, offer in roles:
            if offer.offer_id in seen:
                continue
            seen.add(offer.offer_id)
            public = self.public_offer(offer.offer_id)
            result.append(
                {
                    "role": role,
                    "offer_id": offer.offer_id,
                    "supplier_unit_cost": float(offer.unit_cost),
                    "package_price": float(offer.package_price),
                    "package_units": int(offer.package_units),
                    "estimated_delivery_days": int(
                        public["estimated_delivery_days"]
                    ),
                    "lifecycle_profile": dict(
                        public.get("lifecycle_profile") or {}
                    ),
                }
            )
        return result

    def _reservation_profile_member_row(
        self, state: WorldState, offer_id: str
    ) -> dict[str, Any]:
        statuses = Counter(
            asset.status for asset in state.assets if asset.offer_id == offer_id
        )
        in_transit = []
        for order in state.orders:
            if order.status != "in_transit":
                continue
            for item in order.items:
                if str(item["offer_id"]) != offer_id:
                    continue
                in_transit.append(
                    {
                        "order_id": order.order_id,
                        "units": int(item["packs"]) * int(item["package_units"]) - (
                            item["received_units"] if self._v31_warehouse_capacity() else 0),
                        "estimated_delivery_day": self.estimated_order_delivery_day(
                            order
                        ),
                        "status": order.status,
                    }
                )
        return {
            "offer": self.public_offer(offer_id),
            "asset_status_counts": dict(sorted(statuses.items())),
            "in_transit_orders": in_transit,
        }

    def _reservation_quote_workspace_offer(self, offer_id: str) -> dict[str, Any]:
        public = self.public_offer(offer_id)
        attributes = [
            {"name": str(row["name"]), "value": row.get("value")}
            for row in public.get("specifications", [])
        ]
        attributes.sort(key=lambda row: (row["name"], str(row.get("value"))))
        return {
            "offer_id": offer_id,
            "name": public["name"],
            **({"brand": public["brand"]} if public.get("brand") else {}),
            "public_attributes": attributes,
            "supplier_unit_cost": public["supplier_unit_cost"],
            "package_price": public["package_price"],
            "package_units": public["package_units"],
            "seller_id": public["seller_id"],
            "estimated_delivery_days": public["estimated_delivery_days"],
            **({"warehouse_capacity_units_per_asset": public["warehouse_capacity_units_per_asset"]}
               if self._v31_warehouse_capacity() else {}),
        }

    def view_reservation_quote_workspace(
        self,
        state: WorldState,
        *,
        mode: str,
        request_ids: list[str] | None = None,
        cursor: str | None = None,
        limit: int = 5,
        offer_projection: str = "compact",
    ) -> dict[str, Any]:
        if not self._v25_quote_workspace():
            raise ValueError("request-level quote workspace requires V25")
        if mode not in {"pending_page", "requests"}:
            raise ValueError("mode must be pending_page or requests")
        if offer_projection not in {"compact", "references_only"}:
            raise ValueError(
                "offer_projection must be compact or references_only"
            )
        if not 1 <= limit <= 5:
            raise ValueError("limit must be in [1, 5]")

        todays_requests = sorted(
            (
                request
                for request in state.reservation_requests
                if request.created_day == state.day
            ),
            key=lambda request: request.request_id,
        )
        pending = [
            request for request in todays_requests if request.status in {"pending_quote", "pending_service_offer"}
        ]
        if mode == "pending_page":
            if request_ids is not None:
                raise ValueError(
                    "request_ids are valid only when mode=requests"
                )
            if cursor is not None and cursor not in {
                request.request_id for request in todays_requests
            }:
                raise ValueError(
                    "cursor must be a request_id created on the current day"
                )
            candidates = [
                request
                for request in pending
                if cursor is None or request.request_id > cursor
            ]
            selected = candidates[:limit]
            has_more = len(candidates) > len(selected)
            next_cursor = selected[-1].request_id if selected and has_more else None
        else:
            if cursor is not None:
                raise ValueError("cursor is valid only when mode=pending_page")
            exact_ids = list(request_ids or [])
            if (
                not 1 <= len(exact_ids) <= 5
                or len(set(exact_ids)) != len(exact_ids)
            ):
                raise ValueError(
                    "request_ids must contain 1 to 5 unique values"
                )
            pending_by_id = {request.request_id: request for request in pending}
            invalid = sorted(set(exact_ids) - set(pending_by_id))
            if invalid:
                raise ValueError(
                    "request_ids are not pending quote requests created today: "
                    f"{invalid}"
                )
            selected = [pending_by_id[request_id] for request_id in sorted(exact_ids)]
            has_more = False
            next_cursor = None

        catalog = self.reservation_promise_catalog
        if catalog is None:
            raise AssertionError("reservation promise catalog is unavailable")
        profiles_by_id: dict[str, dict[str, Any]] = {}
        referenced_offer_ids: set[str] = set()
        public_requests = []
        for request in selected:
            lines = []
            for line in sorted(request.lines, key=lambda row: row.line_id):
                profile_terms = []
                for profile_id in sorted(line.eligible_profile_ids):
                    profile = catalog.by_id[profile_id]
                    quantity = math.ceil(
                        int(line.service_units or line.customer_units)
                        / profile.capacity_per_asset
                    )
                    member_ids = sorted(
                        request.inquiry_profile_member_ids_by_line[
                            line.line_id
                        ][profile_id]
                    )
                    referenced_offer_ids.update(member_ids)
                    total_deposit = float(
                        request.inquiry_profile_deposit_by_line[
                            line.line_id
                        ][profile_id]
                    )
                    profile_terms.append(
                        {
                            "promise_profile_id": profile_id,
                            "asset_quantity": quantity,
                            "total_security_deposit": round(total_deposit, 2),
                            "frozen_member_offer_ids": member_ids,
                        }
                    )
                    profiles_by_id[profile_id] = {
                        "promise_profile_id": profile_id,
                        "profile_family_id": profile.family_id,
                        "service_key": profile.service_key,
                        "public_guarantee": dict(profile.public_guarantee),
                        "capacity_per_asset": profile.capacity_per_asset,
                        "unit_security_deposit": round(
                            catalog.unit_deposit(
                                profile_id,
                                str(request.profile_deposit_policy_id),
                            ),
                            2,
                        ),
                    }
                lines.append(
                    {
                        "line_id": line.line_id,
                        "service_key": line.service_key,
                        "service_units": line.service_units,
                        "duration_days": line.duration_days,
                        "required": line.required,
                        "hard_requirements": dict(
                            line.public_hard_requirements
                        ),
                        "stated_preferences": [
                            dict(row) for row in line.public_preferences
                        ],
                        "profile_terms": profile_terms,
                    }
                )
            public_requests.append(
                {
                    "request_id": request.request_id,
                    "status": request.status,
                    "created_day": request.created_day,
                    "expires_day": request.expires_day,
                    "quote_deadline": "end_of_current_business_day",
                    "pickup_day": request.pickup_day,
                    "return_day": request.return_day,
                    "booking_lead_days": request.booking_lead_days,
                    "visible_party_context": request.visible_party_context,
                    "forecast_condition": request.forecast_condition,
                    "forecast_accuracy": request.forecast_accuracy,
                    "prepayment_rate": request.prepayment_rate,
                    **(
                        {
                            "service_reputation_snapshot": dict(
                                request.service_reputation_snapshot
                            )
                        }
                        if request.service_reputation_snapshot
                        else {}
                    ),
                    "lines": lines,
                }
            )

        sorted_offer_ids = sorted(referenced_offer_ids)
        result = {
            "view": "reservation_quote_workspace",
            "mode": mode,
            "day": state.day,
            "offer_projection": offer_projection,
            "promise_profile_schema_version": catalog.schema_version,
            "profile_deposit_policy_id": (
                self.config.reservation_profile_deposit_policy
            ),
            "quote_constraints": {
                "required_line_quote_options": [1, 2],
                "optional_line_quote_options": [0, 2],
                "optional_line_requires_one_omit": True,
                "maximum_internal_plan_count": 24,
                "daily_unit_price": "positive_finite_cent_exact",
            },
            "requests": public_requests,
            "profiles_by_id": {
                profile_id: profiles_by_id[profile_id]
                for profile_id in sorted(profiles_by_id)
            },
            "referenced_offer_ids": sorted_offer_ids,
            "pending_request_count": len(pending),
            "returned_request_count": len(public_requests),
            "profile_definition_count": len(profiles_by_id),
            "offer_definition_count": (
                len(sorted_offer_ids) if offer_projection == "compact" else 0
            ),
            "cursor": cursor,
            "limit": limit if mode == "pending_page" else len(public_requests),
            "has_more": has_more,
            "next_cursor": next_cursor,
            "dynamic_operating_data_included": False,
            "recommendation_or_feasibility_included": False,
        }
        if offer_projection == "compact":
            result["offers_by_id"] = {
                offer_id: self._reservation_quote_workspace_offer(offer_id)
                for offer_id in sorted_offer_ids
            }
        return result

    def view_reservation_options(
        self,
        state: WorldState,
        *,
        request_id: str,
        line_id: str,
        view: str,
        profile_family_id: str | None = None,
        promise_profile_id: str | None = None,
        cursor: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        if not self._reservation_profile_mode():
            raise ValueError("reservation promise options require a profile contract mode")
        request = next(
            (
                row
                for row in state.reservation_requests
                if row.request_id == request_id
            ),
            None,
        )
        if request is None:
            raise ValueError(f"unknown request_id: {request_id}")
        line = next((row for row in request.lines if row.line_id == line_id), None)
        if line is None:
            raise ValueError(f"unknown inquiry line_id: {line_id}")
        if cursor < 0 or not 1 <= limit <= 50:
            raise ValueError("cursor/limit must satisfy cursor >= 0 and 1 <= limit <= 50")
        catalog = self.reservation_promise_catalog
        if catalog is None:
            raise AssertionError("reservation promise catalog is unavailable")

        if view == "profiles":
            profile_ids = [
                profile_id
                for profile_id in line.eligible_profile_ids
                if profile_family_id is None
                or catalog.by_id[profile_id].family_id == profile_family_id
            ]
            unknown_family = (
                profile_family_id is not None
                and profile_family_id not in line.eligible_profile_family_ids
            )
            if unknown_family:
                raise ValueError(
                    f"profile family is not eligible for this line: {profile_family_id}"
                )
            rows = []
            for profile_id in profile_ids:
                profile = catalog.by_id[profile_id]
                quantity = math.ceil(
                    int(line.service_units or line.customer_units)
                    / profile.capacity_per_asset
                )
                aggregate_status = Counter()
                in_transit_units = 0
                for offer_id in request.inquiry_profile_member_ids_by_line[
                    line_id
                ][profile_id]:
                    aggregate_status.update(
                        asset.status
                        for asset in state.assets
                        if asset.offer_id == offer_id
                    )
                    for order in state.orders:
                        if order.status != "in_transit":
                            continue
                        for item in order.items:
                            if str(item["offer_id"]) == offer_id:
                                in_transit_units += int(item["packs"]) * int(
                                    item["package_units"]
                                )
                deposit = float(
                    request.inquiry_profile_deposit_by_line[line_id][profile_id]
                )
                rows.append(
                    {
                        "promise_profile_id": profile_id,
                        "profile_family_id": profile.family_id,
                        "public_guarantee": dict(profile.public_guarantee),
                        "capacity_per_asset": profile.capacity_per_asset,
                        "asset_quantity": quantity,
                        "unit_profile_deposit": round(deposit / quantity, 2),
                        "total_profile_deposit": round(deposit, 2),
                        "member_count": len(profile.member_offer_ids),
                        "asset_status_counts": dict(
                            sorted(aggregate_status.items())
                        ),
                        "in_transit_units": in_transit_units,
                        "procurement_representatives": (
                            self._reservation_profile_representatives(profile_id)
                        ),
                    }
                )
            page = rows[cursor : cursor + limit]
            next_cursor = cursor + len(page)
            if next_cursor >= len(rows):
                next_cursor = None
            return {
                "view": view,
                "request_id": request_id,
                "line_id": line_id,
                "profile_schema_version": request.promise_profile_schema_version,
                "profile_deposit_policy_id": request.profile_deposit_policy_id,
                "profile_families": list(line.eligible_profile_family_ids),
                "profiles": page,
                "total_count": len(rows),
                "returned_count": len(page),
                "cursor": cursor,
                "limit": limit,
                "has_more": next_cursor is not None,
                "next_cursor": next_cursor,
            }

        if view not in {"profile_members", "authorized_members"}:
            raise ValueError(
                "view must be profiles, profile_members, or authorized_members"
            )
        if view == "profile_members":
            if promise_profile_id not in line.eligible_profile_ids:
                raise ValueError("promise_profile_id is not eligible for this line")
            member_ids = list(
                request.inquiry_profile_member_ids_by_line[line_id][
                    str(promise_profile_id)
                ]
            )
        else:
            if request.commitment_id is None:
                raise ValueError("authorized members are visible only after confirmation")
            committed = next(
                row
                for row in state.reservation_commitments
                if row.commitment_id == request.commitment_id
            )
            committed_line = next(
                row for row in committed.lines if row.line_id == line_id
            )
            if (
                promise_profile_id is not None
                and promise_profile_id != committed_line.promise_profile_id
            ):
                raise ValueError("promise_profile_id differs from the locked profile")
            promise_profile_id = committed_line.promise_profile_id
            member_ids = list(committed_line.authorized_offer_ids)
        page_ids = member_ids[cursor : cursor + limit]
        next_cursor = cursor + len(page_ids)
        if next_cursor >= len(member_ids):
            next_cursor = None
        return {
            "view": view,
            "request_id": request_id,
            "commitment_id": request.commitment_id,
            "line_id": line_id,
            "promise_profile_id": promise_profile_id,
            "members": [
                self._reservation_profile_member_row(state, offer_id)
                for offer_id in page_ids
            ],
            "total_count": len(member_ids),
            "returned_count": len(page_ids),
            "cursor": cursor,
            "limit": limit,
            "has_more": next_cursor is not None,
            "next_cursor": next_cursor,
        }

    def _quoted_line_outcome(
        self,
        state: WorldState,
        request: ReservationRequest,
        line: ReservationLine,
        daily_price: float,
    ) -> dict[str, Any]:
        session = self._restore_quote_context(
            request.quote_context_by_line[line.line_id]
        )
        views = self._channel_offer_views(state, session)
        deposits = request.candidate_deposit_by_line[line.line_id]
        first_pass = [
            offer_id
            for offer_id in line.candidate_offer_ids
            if self._reservation_candidate_selected_at_price(
                session,
                views,
                offer_id=offer_id,
                quantity=line.quantity,
                duration_days=line.duration_days,
                daily_price=daily_price,
                locked_security_deposit=float(deposits[offer_id]),
            )
        ]
        if not first_pass:
            return {
                "primary_offer_id": None,
                "authorized_offer_ids": [],
                "fallback_offer_ids": [],
                "locked_security_deposit": 0.0,
                "authorized_ceilings": {},
                "recovery_ceilings": {},
            }
        primary = first_pass[0]
        locked_deposit = float(deposits[primary])
        public_survivors = [
            offer_id
            for offer_id in first_pass
            if self._reservation_candidate_selected_at_price(
                session,
                views,
                offer_id=offer_id,
                quantity=line.quantity,
                duration_days=line.duration_days,
                daily_price=daily_price,
                locked_security_deposit=locked_deposit,
            )
        ]
        if primary not in public_survivors:
            raise AssertionError("confirmed primary must survive its own deposit")
        accepted_all: list[str] = []
        accepted_ceilings: dict[str, float] = {}
        for offer_id in request.inquiry_latent_offer_ids_by_line.get(
            line.line_id, []
        ):
            ceiling = self._reservation_candidate_price_ceiling(
                session,
                views,
                offer_id=offer_id,
                quantity=line.quantity,
                duration_days=line.duration_days,
                locked_security_deposit=locked_deposit,
            )
            if ceiling is None or ceiling + 1e-9 < daily_price:
                continue
            accepted_all.append(offer_id)
            accepted_ceilings[offer_id] = ceiling
        hidden = [
            offer_id
            for offer_id in accepted_all
            if offer_id not in public_survivors
        ]
        rng = self._stable_event_rng(
            _RNG_RESERVATION_QUOTE_RECOVERY,
            *session.event_address,
            identity=(
                "quoted-recovery:"
                + self._reservation_line_rng_identity(
                    request.request_id, line.line_id
                )
            ),
        )
        fallback_count = min(
            len(hidden),
            int(
                rng.integers(
                    0,
                    int(self.config.reservation_authorized_alternative_max) + 1,
                )
            ),
        )
        selected_hidden = (
            [
                hidden[index]
                for index in map(
                    int,
                    rng.choice(
                        len(hidden), size=fallback_count, replace=False
                    ),
                )
            ]
            if fallback_count
            else []
        )
        rank = {
            offer_id: index
            for index, offer_id in enumerate(
                request.inquiry_latent_offer_ids_by_line.get(line.line_id, [])
            )
        }
        selected_hidden.sort(key=rank.__getitem__)
        mode = self.config.reservation_authorization_mode
        if mode == "risk_only":
            authorized = [primary]
            fallback = list(
                dict.fromkeys(
                    [
                        *[
                            offer_id
                            for offer_id in public_survivors
                            if offer_id != primary
                        ],
                        *selected_hidden,
                    ]
                )
            )
        elif mode == "all_latent":
            authorized = [
                primary,
                *[offer_id for offer_id in accepted_all if offer_id != primary],
            ]
            fallback = []
        else:
            authorized = public_survivors
            fallback = selected_hidden
        return {
            "primary_offer_id": primary,
            "authorized_offer_ids": authorized,
            "fallback_offer_ids": fallback,
            "locked_security_deposit": locked_deposit,
            "authorized_ceilings": {
                offer_id: accepted_ceilings[offer_id]
                for offer_id in authorized
                if offer_id in accepted_ceilings
            },
            "recovery_ceilings": {
                offer_id: accepted_ceilings[offer_id]
                for offer_id in fallback
                if offer_id in accepted_ceilings
            },
        }

    def _quoted_profile_line_outcome(
        self,
        request: ReservationRequest,
        line: ReservationLine,
        *,
        profile_id: str,
        daily_price: float,
    ) -> dict[str, Any]:
        catalog = self.reservation_promise_catalog
        if catalog is None:
            raise AssertionError("reservation promise catalog is unavailable")
        if profile_id not in line.eligible_profile_ids:
            raise ValueError(
                f"promise profile is not eligible for {line.line_id}: {profile_id}"
            )
        profile = catalog.by_id[profile_id]
        quantity = math.ceil(
            int(line.service_units or line.customer_units)
            / profile.capacity_per_asset
        )
        deposit = float(
            request.inquiry_profile_deposit_by_line[line.line_id][profile_id]
        )
        ceilings = dict(
            request.inquiry_profile_price_ceiling_by_line[line.line_id][
                profile_id
            ]
        )
        members = set(
            request.inquiry_profile_member_ids_by_line[line.line_id][profile_id]
        )
        authorized = sorted(
            (
                offer_id
                for offer_id in members
                if float(ceilings.get(offer_id, -1.0)) + 1e-9 >= daily_price
            ),
            key=lambda offer_id: (-float(ceilings[offer_id]), offer_id),
        )
        recovery = sorted(
            (
                offer_id
                for offer_id, ceiling in ceilings.items()
                if offer_id not in members and float(ceiling) + 1e-9 >= daily_price
            ),
            key=lambda offer_id: (-float(ceilings[offer_id]), offer_id),
        )
        return {
            "primary_offer_id": authorized[0] if authorized else None,
            "authorized_offer_ids": authorized,
            "fallback_offer_ids": recovery,
            "locked_security_deposit": round(deposit, 2),
            "authorized_ceilings": {
                offer_id: float(ceilings[offer_id]) for offer_id in authorized
            },
            "recovery_ceilings": {
                offer_id: float(ceilings[offer_id]) for offer_id in recovery
            },
            "promise_profile_id": profile_id,
            "profile_family_id": profile.family_id,
            "profile_schema_version": catalog.schema_version,
            "capacity_per_asset": profile.capacity_per_asset,
            "asset_quantity": quantity,
        }

    @staticmethod
    def _explicit_pairwise_quotes(
        request: ReservationRequest, decision: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        action = str(decision.get("action", ""))
        if action == "decline":
            if "lines" in decision:
                raise ValueError("decline must not include lines")
            return {}
        if action != "quote":
            raise ValueError("action must be quote or decline")
        raw_lines = decision.get("lines")
        if not isinstance(raw_lines, list):
            raise ValueError("quote requires complete lines")
        by_line = {line.line_id: line for line in request.lines}
        result: dict[str, dict[str, Any]] = {}
        for raw_line in raw_lines:
            if not isinstance(raw_line, dict):
                raise ValueError("every quoted line must be an object")
            line_id = str(raw_line.get("line_id", ""))
            if line_id not in by_line:
                raise ValueError(f"unknown inquiry line_id: {line_id}")
            if line_id in result:
                raise ValueError(f"duplicate inquiry line_id: {line_id}")
            raw_options = raw_line.get("quote_options")
            if not isinstance(raw_options, list) or not 1 <= len(raw_options) <= 2:
                raise ValueError("quote_options must contain one or two options")
            exact_ids: set[str] = set()
            options = []
            line = by_line[line_id]
            for raw_option in raw_options:
                if not isinstance(raw_option, dict):
                    raise ValueError("every quote option must be an object")
                offer_id = str(raw_option.get("offer_id", ""))
                profile_id = str(raw_option.get("promise_profile_id", ""))
                value = raw_option.get("daily_unit_price")
                if offer_id in exact_ids:
                    raise ValueError(
                        f"duplicate exact offer_id for {line_id}: {offer_id}"
                    )
                exact_ids.add(offer_id)
                if profile_id not in line.eligible_profile_ids:
                    raise ValueError(
                        "promise_profile_id is not eligible for "
                        f"{line_id}: {profile_id}"
                    )
                frozen_members = request.inquiry_profile_member_ids_by_line[
                    line_id
                ][profile_id]
                if offer_id not in frozen_members:
                    raise ValueError(
                        f"offer_id is not a member of {profile_id}: {offer_id}"
                    )
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(
                        "daily_unit_price must be a positive finite number"
                    )
                cents = float(value) * 100.0
                if not math.isclose(
                    cents, round(cents), rel_tol=0.0, abs_tol=1e-7
                ):
                    raise ValueError(
                        "daily_unit_price must use at most two decimal places"
                    )
                options.append(
                    {
                        "offer_id": offer_id,
                        "promise_profile_id": profile_id,
                        "daily_unit_price": round(float(value), 2),
                    }
                )
            options.sort(
                key=lambda row: (
                    str(row["offer_id"]),
                    str(row["promise_profile_id"]),
                    float(row["daily_unit_price"]),
                )
            )
            for index, option in enumerate(options, start=1):
                option["option_id"] = f"option-{index:02d}"
            result[line_id] = {"quote_options": options}
        missing = sorted(set(by_line) - set(result))
        if missing:
            raise ValueError(
                "quote must cover every inquiry line; missing line_ids: "
                + ", ".join(missing)
            )
        return result

    @staticmethod
    def _explicit_complete_contract_options(
        request: ReservationRequest, decision: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate and canonicalize one or two explicit complete contracts."""

        action = str(decision.get("action", ""))
        if action == "decline":
            if "contract_options" in decision:
                raise ValueError("decline must not include contract_options")
            return {"contract_options": []}
        if action != "quote":
            raise ValueError("action must be quote or decline")
        raw_contracts = decision.get("contract_options")
        if not isinstance(raw_contracts, list) or not 1 <= len(raw_contracts) <= 2:
            raise ValueError("quote requires one or two complete contract_options")
        line_by_id = {line.line_id: line for line in request.lines}
        normalized: list[dict[str, Any]] = []
        signatures: set[tuple[Any, ...]] = set()
        for raw_contract in raw_contracts:
            if not isinstance(raw_contract, dict):
                raise ValueError("every contract option must be an object")
            if set(raw_contract) != {"lines"}:
                raise ValueError("a contract option may contain only lines")
            raw_lines = raw_contract.get("lines")
            if not isinstance(raw_lines, list):
                raise ValueError("every contract option requires complete lines")
            resolved: dict[str, dict[str, Any]] = {}
            for raw_line in raw_lines:
                if not isinstance(raw_line, dict):
                    raise ValueError("every contract line must be an object")
                line_id = str(raw_line.get("line_id", ""))
                if line_id not in line_by_id:
                    raise ValueError(f"unknown inquiry line_id: {line_id}")
                if line_id in resolved:
                    raise ValueError(f"duplicate inquiry line_id: {line_id}")
                line = line_by_id[line_id]
                disposition = str(raw_line.get("disposition", ""))
                if disposition == "omit":
                    if line.required:
                        raise ValueError(f"required line cannot be omitted: {line_id}")
                    if set(raw_line) != {"line_id", "disposition"}:
                        raise ValueError(
                            "an omitted optional line must contain only line_id "
                            "and disposition"
                        )
                    resolved[line_id] = {
                        "line_id": line_id,
                        "disposition": "omit",
                    }
                    continue
                if disposition != "quote":
                    raise ValueError("line disposition must be quote or omit")
                if set(raw_line) != {
                    "line_id",
                    "disposition",
                    "offer_id",
                    "promise_profile_id",
                    "daily_unit_price",
                }:
                    raise ValueError(
                        "a quoted contract line requires only line_id, disposition, "
                        "offer_id, promise_profile_id, and daily_unit_price"
                    )
                offer_id = str(raw_line.get("offer_id", ""))
                profile_id = str(raw_line.get("promise_profile_id", ""))
                value = raw_line.get("daily_unit_price")
                if profile_id not in line.eligible_profile_ids:
                    raise ValueError(
                        "promise_profile_id is not eligible for "
                        f"{line_id}: {profile_id}"
                    )
                members = request.inquiry_profile_member_ids_by_line[line_id][
                    profile_id
                ]
                if offer_id not in members:
                    raise ValueError(
                        f"offer_id is not a member of {profile_id}: {offer_id}"
                    )
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(
                        "daily_unit_price must be a positive finite number"
                    )
                cents = float(value) * 100.0
                if not math.isclose(
                    cents, round(cents), rel_tol=0.0, abs_tol=1e-7
                ):
                    raise ValueError(
                        "daily_unit_price must use at most two decimal places"
                    )
                resolved[line_id] = {
                    "line_id": line_id,
                    "disposition": "quote",
                    "offer_id": offer_id,
                    "promise_profile_id": profile_id,
                    "daily_unit_price": round(float(value), 2),
                }
            missing = sorted(set(line_by_id) - set(resolved))
            if missing:
                raise ValueError(
                    "every contract option must resolve every inquiry line; "
                    "missing line_ids: " + ", ".join(missing)
                )
            rows = [resolved[line_id] for line_id in sorted(resolved)]
            signature = tuple(
                (
                    row["line_id"],
                    row["disposition"],
                    row.get("offer_id"),
                    row.get("promise_profile_id"),
                    row.get("daily_unit_price"),
                )
                for row in rows
            )
            if signature in signatures:
                raise ValueError("contract_options must be distinct")
            signatures.add(signature)
            normalized.append({"lines": rows, "_signature": signature})
        normalized.sort(key=lambda row: row["_signature"])
        for index, row in enumerate(normalized, start=1):
            row.pop("_signature", None)
            row["contract_option_id"] = f"contract-option-{index:02d}"
        return {"contract_options": normalized}

    def _factorized_line_option_invalid_items(
        self,
        request: ReservationRequest,
        decision: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Collect deterministic V22 errors without evaluating a customer."""

        request_id = str(decision.get("request_id") or request.request_id)
        invalid: list[dict[str, Any]] = []

        def add(
            *,
            field: str,
            error_code: str,
            error: str,
            line_id: str | None = None,
            line_index: int | None = None,
            option_index: int | None = None,
            submitted_value: Any = None,
            include_value: bool = False,
        ) -> None:
            row: dict[str, Any] = {
                "request_id": request_id,
                "field": field,
                "error_code": error_code,
                "error": error,
            }
            if line_id is not None:
                row["line_id"] = line_id
            if line_index is not None:
                row["line_index"] = line_index
            if option_index is not None:
                row["option_index"] = option_index
            if include_value:
                row["submitted_value"] = submitted_value
            invalid.append(row)

        action = str(decision.get("action", ""))
        if action == "decline":
            extras = sorted(set(decision) - {"request_id", "action"})
            for field in extras:
                add(
                    field=field,
                    error_code="unexpected_decline_field",
                    error="decline must contain only request_id and action",
                )
            return invalid
        if action != "quote":
            add(
                field="action",
                error_code="invalid_action",
                error="action must be quote or decline",
                submitted_value=decision.get("action"),
                include_value=True,
            )
            return invalid
        expected_decision_fields = {"request_id", "action", "lines"}
        for field in ("request_id", "action", "lines"):
            if field not in decision:
                add(
                    field=field,
                    error_code="missing_quote_field",
                    error="quote requires request_id, action, and lines",
                )
        for field in sorted(set(decision) - expected_decision_fields):
            add(
                field=field,
                error_code="unexpected_quote_field",
                error="quote must contain only request_id, action, and lines",
            )
        raw_lines = decision.get("lines")
        if not isinstance(raw_lines, list):
            add(
                field="lines",
                error_code="lines_not_array",
                error="quote requires a lines array",
                submitted_value=raw_lines,
                include_value=isinstance(raw_lines, (str, int, float, bool, type(None))),
            )
            return invalid

        line_by_id = {line.line_id: line for line in request.lines}
        seen_line_ids: set[str] = set()
        valid_line_option_counts: dict[str, int] = {}
        for line_index, raw_line in enumerate(raw_lines):
            if not isinstance(raw_line, dict):
                add(
                    field="lines",
                    error_code="line_not_object",
                    error="every quoted line must be an object",
                    line_index=line_index,
                )
                continue
            line_id = str(raw_line.get("line_id", ""))
            line_context = line_id or None
            expected_line_fields = {"line_id", "options"}
            for field in ("line_id", "options"):
                if field not in raw_line:
                    add(
                        field=field,
                        error_code="missing_line_field",
                        error="every quoted line requires line_id and options",
                        line_id=line_context,
                        line_index=line_index,
                    )
            for field in sorted(set(raw_line) - expected_line_fields):
                add(
                    field=field,
                    error_code="unexpected_line_field",
                    error="every quoted line requires only line_id and options",
                    line_id=line_context,
                    line_index=line_index,
                )
            if line_id not in line_by_id:
                add(
                    field="line_id",
                    error_code="unknown_line_id",
                    error=f"unknown inquiry line_id: {line_id}",
                    line_id=line_context,
                    line_index=line_index,
                    submitted_value=line_id,
                    include_value=True,
                )
                continue
            if line_id in seen_line_ids:
                add(
                    field="line_id",
                    error_code="duplicate_line_id",
                    error=f"duplicate inquiry line_id: {line_id}",
                    line_id=line_id,
                    line_index=line_index,
                    submitted_value=line_id,
                    include_value=True,
                )
                continue
            seen_line_ids.add(line_id)
            raw_options = raw_line.get("options")
            if not isinstance(raw_options, list):
                add(
                    field="options",
                    error_code="options_not_array",
                    error=f"line options must be an array: {line_id}",
                    line_id=line_id,
                    line_index=line_index,
                )
                continue

            line = line_by_id[line_id]
            quoted_count = 0
            omit_count = 0
            exact_profiles: set[tuple[str, str]] = set()
            line_error_start = len(invalid)
            for option_index, raw_option in enumerate(raw_options):
                if not isinstance(raw_option, dict):
                    add(
                        field="options",
                        error_code="option_not_object",
                        error="every line option must be an object",
                        line_id=line_id,
                        line_index=line_index,
                        option_index=option_index,
                    )
                    continue
                disposition = str(raw_option.get("disposition", ""))
                if disposition == "omit":
                    omit_count += 1
                    for field in sorted(set(raw_option) - {"disposition"}):
                        add(
                            field=field,
                            error_code="unexpected_omit_field",
                            error="an omit option must contain only disposition",
                            line_id=line_id,
                            line_index=line_index,
                            option_index=option_index,
                        )
                    continue
                if disposition != "quote":
                    add(
                        field="disposition",
                        error_code="invalid_disposition",
                        error="line option disposition must be quote or omit",
                        line_id=line_id,
                        line_index=line_index,
                        option_index=option_index,
                        submitted_value=raw_option.get("disposition"),
                        include_value=True,
                    )
                    continue
                quoted_count += 1
                expected_option_fields = {
                    "disposition",
                    "offer_id",
                    "promise_profile_id",
                    "daily_unit_price",
                }
                for field in (
                    "offer_id",
                    "promise_profile_id",
                    "daily_unit_price",
                ):
                    if field not in raw_option:
                        add(
                            field=field,
                            error_code="missing_quote_option_field",
                            error=(
                                "a quote option requires disposition, offer_id, "
                                "promise_profile_id, and daily_unit_price"
                            ),
                            line_id=line_id,
                            line_index=line_index,
                            option_index=option_index,
                        )
                for field in sorted(set(raw_option) - expected_option_fields):
                    add(
                        field=field,
                        error_code="unexpected_quote_option_field",
                        error=(
                            "a quote option requires only disposition, offer_id, "
                            "promise_profile_id, and daily_unit_price"
                        ),
                        line_id=line_id,
                        line_index=line_index,
                        option_index=option_index,
                    )
                offer_id = str(raw_option.get("offer_id", ""))
                profile_id = str(raw_option.get("promise_profile_id", ""))
                value = raw_option.get("daily_unit_price")
                profile_valid = profile_id in line.eligible_profile_ids
                if not profile_valid:
                    add(
                        field="promise_profile_id",
                        error_code="ineligible_profile",
                        error=(
                            "promise_profile_id is not eligible for "
                            f"{line_id}: {profile_id}"
                        ),
                        line_id=line_id,
                        line_index=line_index,
                        option_index=option_index,
                        submitted_value=profile_id,
                        include_value=True,
                    )
                elif offer_id not in request.inquiry_profile_member_ids_by_line[
                    line_id
                ][profile_id]:
                    add(
                        field="offer_id",
                        error_code="offer_not_profile_member",
                        error=f"offer_id is not a member of {profile_id}: {offer_id}",
                        line_id=line_id,
                        line_index=line_index,
                        option_index=option_index,
                        submitted_value=offer_id,
                        include_value=True,
                    )
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    add(
                        field="daily_unit_price",
                        error_code="invalid_daily_unit_price",
                        error="daily_unit_price must be a positive finite number",
                        line_id=line_id,
                        line_index=line_index,
                        option_index=option_index,
                        submitted_value=value,
                        include_value=isinstance(value, (str, int, float, bool, type(None))),
                    )
                elif not math.isclose(
                    float(value) * 100.0,
                    round(float(value) * 100.0),
                    rel_tol=0.0,
                    abs_tol=1e-7,
                ):
                    add(
                        field="daily_unit_price",
                        error_code="price_not_cent_exact",
                        error="daily_unit_price must use at most two decimal places",
                        line_id=line_id,
                        line_index=line_index,
                        option_index=option_index,
                        submitted_value=value,
                        include_value=True,
                    )
                signature = (offer_id, profile_id)
                if offer_id and profile_id:
                    if signature in exact_profiles:
                        add(
                            field="options",
                            error_code="duplicate_exact_profile",
                            error=(
                                "one line cannot quote the same exact offer/profile twice"
                            ),
                            line_id=line_id,
                            line_index=line_index,
                            option_index=option_index,
                            submitted_value={
                                "offer_id": offer_id,
                                "promise_profile_id": profile_id,
                            },
                            include_value=True,
                        )
                    exact_profiles.add(signature)
            if line.required:
                if omit_count:
                    add(
                        field="options",
                        error_code="required_line_omit",
                        error=f"required line cannot be omitted: {line_id}",
                        line_id=line_id,
                        line_index=line_index,
                    )
                if not 1 <= quoted_count <= 2:
                    add(
                        field="options",
                        error_code="required_quote_option_count",
                        error="every required line needs one or two quote options",
                        line_id=line_id,
                        line_index=line_index,
                    )
            else:
                valid_omit_counts = {0, 1} if self._v37_supplier_trial() else {1}
                if omit_count not in valid_omit_counts:
                    add(
                        field="options",
                        error_code="optional_omit_count",
                        error=(
                            "an optional line accepts zero or one explicit omit; "
                            "the environment always adds one implicit omit"
                            if self._v37_supplier_trial()
                            else "every optional line needs exactly one explicit omit option"
                        ),
                        line_id=line_id,
                        line_index=line_index,
                    )
                if quoted_count > 2:
                    add(
                        field="options",
                        error_code="optional_quote_option_count",
                        error="an optional line may contain at most two quote options",
                        line_id=line_id,
                        line_index=line_index,
                    )
            if len(invalid) == line_error_start:
                valid_line_option_counts[line_id] = quoted_count + (
                    0 if line.required else 1
                )

        for line in request.lines:
            if line.line_id not in seen_line_ids:
                add(
                    field="lines",
                    error_code="missing_line_id",
                    error=(
                        "quote must resolve every inquiry line; missing line_ids: "
                        f"{line.line_id}"
                    ),
                    line_id=line.line_id,
                )
        if not invalid and len(valid_line_option_counts) == len(line_by_id):
            plan_count = math.prod(valid_line_option_counts.values())
            if not 1 <= plan_count <= 24:
                add(
                    field="lines",
                    error_code="internal_plan_limit",
                    error="factorized quote may form at most 24 complete plans",
                    submitted_value=plan_count,
                    include_value=True,
                )
        return invalid

    def _explicit_factorized_line_options(
        self, request: ReservationRequest, decision: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate V20 line options before forming any complete plans."""

        action = str(decision.get("action", ""))
        if action == "decline":
            if set(decision) != {"request_id", "action"}:
                raise ValueError("decline must contain only request_id and action")
            return {"line_options": [], "internal_plan_count": 0}
        if action != "quote":
            raise ValueError("action must be quote or decline")
        if set(decision) != {"request_id", "action", "lines"}:
            raise ValueError("quote must contain only request_id, action, and lines")
        raw_lines = decision.get("lines")
        if not isinstance(raw_lines, list):
            raise ValueError("quote requires a lines array")
        line_by_id = {line.line_id: line for line in request.lines}
        resolved: dict[str, dict[str, Any]] = {}
        for raw_line in raw_lines:
            if not isinstance(raw_line, dict) or set(raw_line) != {
                "line_id",
                "options",
            }:
                raise ValueError(
                    "every quoted line requires only line_id and options"
                )
            line_id = str(raw_line.get("line_id", ""))
            if line_id not in line_by_id:
                raise ValueError(f"unknown inquiry line_id: {line_id}")
            if line_id in resolved:
                raise ValueError(f"duplicate inquiry line_id: {line_id}")
            raw_options = raw_line.get("options")
            if not isinstance(raw_options, list):
                raise ValueError(f"line options must be an array: {line_id}")
            quoted: list[dict[str, Any]] = []
            omit_count = 0
            exact_profiles: set[tuple[str, str]] = set()
            for raw_option in raw_options:
                if not isinstance(raw_option, dict):
                    raise ValueError("every line option must be an object")
                disposition = str(raw_option.get("disposition", ""))
                if disposition == "omit":
                    if set(raw_option) != {"disposition"}:
                        raise ValueError(
                            "an omit option must contain only disposition"
                        )
                    omit_count += 1
                    continue
                if disposition != "quote":
                    raise ValueError("line option disposition must be quote or omit")
                if set(raw_option) != {
                    "disposition",
                    "offer_id",
                    "promise_profile_id",
                    "daily_unit_price",
                }:
                    raise ValueError(
                        "a quote option requires only disposition, offer_id, "
                        "promise_profile_id, and daily_unit_price"
                    )
                offer_id = str(raw_option.get("offer_id", ""))
                profile_id = str(raw_option.get("promise_profile_id", ""))
                value = raw_option.get("daily_unit_price")
                line = line_by_id[line_id]
                if profile_id not in line.eligible_profile_ids:
                    raise ValueError(
                        "promise_profile_id is not eligible for "
                        f"{line_id}: {profile_id}"
                    )
                members = request.inquiry_profile_member_ids_by_line[line_id][
                    profile_id
                ]
                if offer_id not in members:
                    raise ValueError(
                        f"offer_id is not a member of {profile_id}: {offer_id}"
                    )
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(
                        "daily_unit_price must be a positive finite number"
                    )
                cents = float(value) * 100.0
                if not math.isclose(
                    cents, round(cents), rel_tol=0.0, abs_tol=1e-7
                ):
                    raise ValueError(
                        "daily_unit_price must use at most two decimal places"
                    )
                signature = (offer_id, profile_id)
                if signature in exact_profiles:
                    raise ValueError(
                        "one line cannot quote the same exact offer/profile twice"
                    )
                exact_profiles.add(signature)
                quoted.append(
                    {
                        "disposition": "quote",
                        "offer_id": offer_id,
                        "promise_profile_id": profile_id,
                        "daily_unit_price": round(float(value), 2),
                    }
                )
            line = line_by_id[line_id]
            if line.required:
                if omit_count:
                    raise ValueError(f"required line cannot be omitted: {line_id}")
                if not 1 <= len(quoted) <= 2:
                    raise ValueError(
                        "every required line needs one or two quote options"
                    )
            else:
                valid_omit_counts = {0, 1} if self._v37_supplier_trial() else {1}
                if omit_count not in valid_omit_counts:
                    raise ValueError(
                        "an optional line accepts zero or one explicit omit; "
                        "the environment always adds one implicit omit"
                        if self._v37_supplier_trial()
                        else "every optional line needs exactly one explicit omit option"
                    )
                if len(quoted) > 2:
                    raise ValueError(
                        "an optional line may contain at most two quote options"
                    )
            quoted.sort(
                key=lambda row: (
                    str(row["offer_id"]),
                    str(row["promise_profile_id"]),
                    float(row["daily_unit_price"]),
                )
            )
            for index, option in enumerate(quoted, start=1):
                option["option_id"] = f"option-{index:02d}"
            options = quoted + ([{"disposition": "omit", "option_id": "omit"}] if not line.required else [])
            resolved[line_id] = {"line_id": line_id, "options": options}
        missing = sorted(set(line_by_id) - set(resolved))
        if missing:
            raise ValueError(
                "quote must resolve every inquiry line; missing line_ids: "
                + ", ".join(missing)
            )
        normalized = [resolved[line_id] for line_id in sorted(resolved)]
        plan_count = math.prod(len(row["options"]) for row in normalized)
        if not 1 <= plan_count <= 24:
            raise ValueError("factorized quote may form at most 24 complete plans")
        return {
            "line_options": normalized,
            "internal_plan_count": int(plan_count),
        }

    def _quoted_pairwise_line_outcome(
        self,
        state: WorldState,
        request: ReservationRequest,
        line: ReservationLine,
        *,
        quote_options: list[dict[str, Any]],
    ) -> dict[str, Any]:
        enriched = self._enriched_pairwise_quote_options(
            request,
            line,
            quote_options=quote_options,
        )
        option_by_offer = {
            str(option["offer_id"]): option for option in enriched
        }
        session = self._restore_quote_context(
            request.quote_context_by_line[line.line_id]
        )
        views = self._channel_offer_views(state, session)
        menu_views = tuple(
            replace(
                view,
                profile_id=(
                    f"{view.profile_id}:exact:{view.offer_id}"
                    if view.offer_id in option_by_offer
                    else view.profile_id
                ),
                available_units=(
                    int(option_by_offer[view.offer_id]["asset_quantity"])
                    if view.offer_id in option_by_offer
                    else 0
                ),
                store_unit_price=(
                    float(option_by_offer[view.offer_id]["daily_unit_price"])
                    if view.offer_id in option_by_offer
                    else None
                ),
                store_cash_price=(
                    float(option_by_offer[view.offer_id]["total_rent"])
                    if view.offer_id in option_by_offer
                    else None
                ),
                store_deposit=(
                    float(
                        option_by_offer[view.offer_id][
                            "locked_security_deposit"
                        ]
                    )
                    if view.offer_id in option_by_offer
                    else view.store_deposit
                ),
            )
            for view in views
        )
        trace = self._evaluate_channel_views(session, menu_views)
        selected = (
            option_by_offer.get(str(trace.selected_offer_id))
            if trace.selected_channel == "store"
            else None
        )
        if selected is None:
            return self._empty_pairwise_line_outcome(enriched)
        selected_offer_id = str(selected["offer_id"])
        selected_profile_id = str(selected["promise_profile_id"])
        selected_price = float(selected["daily_unit_price"])
        ceilings = request.inquiry_profile_price_ceiling_by_line[line.line_id][
            selected_profile_id
        ]
        members = request.inquiry_profile_member_ids_by_line[line.line_id][
            selected_profile_id
        ]
        recovery = sorted(
            offer_id
            for offer_id in members
            if offer_id != selected_offer_id
            and offer_id in ceilings
            and float(ceilings[offer_id]) + 1e-9 >= selected_price
        )
        return {
            "submitted_quote_options": enriched,
            "selected_quote_option_id": str(selected["option_id"]),
            "primary_offer_id": selected_offer_id,
            "authorized_offer_ids": [selected_offer_id],
            "fallback_offer_ids": recovery,
            "locked_security_deposit": float(
                selected["locked_security_deposit"]
            ),
            "authorized_ceilings": {
                selected_offer_id: float(
                    ceilings.get(selected_offer_id, selected_price)
                )
            },
            "recovery_ceilings": {
                offer_id: float(ceilings[offer_id]) for offer_id in recovery
            },
            **selected,
        }

    def _enriched_pairwise_quote_options(
        self,
        request: ReservationRequest,
        line: ReservationLine,
        *,
        quote_options: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Attach frozen contract economics without choosing a channel."""

        catalog = self.reservation_promise_catalog
        if catalog is None:
            raise AssertionError("reservation promise catalog is unavailable")
        enriched = []
        for raw in quote_options:
            profile_id = str(raw["promise_profile_id"])
            offer_id = str(raw["offer_id"])
            profile = catalog.by_id[profile_id]
            quantity = math.ceil(
                int(line.service_units or line.customer_units)
                / profile.capacity_per_asset
            )
            deposit = float(
                request.inquiry_profile_deposit_by_line[line.line_id][profile_id]
            )
            price = float(raw["daily_unit_price"])
            enriched.append(
                {
                    "option_id": str(raw["option_id"]),
                    "offer_id": offer_id,
                    "promise_profile_id": profile_id,
                    "profile_family_id": profile.family_id,
                    "profile_schema_version": catalog.schema_version,
                    "capacity_per_asset": int(profile.capacity_per_asset),
                    "asset_quantity": quantity,
                    "daily_unit_price": price,
                    "total_rent": round(
                        (rental_amount(price, line.duration_days, quantity)
                         if self._v45_reservation_difficulty()
                         else price * quantity * line.duration_days), 2
                    ),
                    "locked_security_deposit": round(deposit, 2),
                }
            )
        return enriched

    @staticmethod
    def _empty_pairwise_line_outcome(
        enriched: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "submitted_quote_options": enriched,
            "selected_quote_option_id": None,
            "primary_offer_id": None,
            "authorized_offer_ids": [],
            "fallback_offer_ids": [],
            "locked_security_deposit": 0.0,
            "authorized_ceilings": {},
            "recovery_ceilings": {},
        }

    def _bundle_store_option(
        self,
        state: WorldState,
        request: ReservationRequest,
        line: ReservationLine,
        option: dict[str, Any],
    ) -> BundleStoreOption | None:
        """Evaluate one exact quote while removing request-level terms.

        The ordinary channel evaluator remains the single source of product
        fit, quantity/duration value, and relative rent-versus-buy value.  This
        adapter removes absolute cash, service, and return terms so the pure
        bundle kernel can apply each of them once to the complete contract.
        """

        session = self._restore_quote_context(
            request.quote_context_by_line[line.line_id]
        )
        offer_id = str(option["offer_id"])
        views = self._channel_offer_views(state, session)
        menu_views = tuple(
            replace(
                view,
                available_units=(
                    int(option["asset_quantity"])
                    if view.offer_id == offer_id
                    else 0
                ),
                store_unit_price=(
                    float(option["daily_unit_price"])
                    if view.offer_id == offer_id
                    else None
                ),
                store_cash_price=(
                    float(option["total_rent"])
                    if view.offer_id == offer_id
                    else None
                ),
                store_deposit=(
                    float(option["locked_security_deposit"])
                    if view.offer_id == offer_id
                    else view.store_deposit
                ),
            )
            for view in views
        )
        trace = self._evaluate_channel_views(session, menu_views)
        row = next(
            (
                profile
                for profile in trace.profiles
                if any(
                    str(plan_line.get("offer_id")) == offer_id
                    for plan_line in profile.get("store_quote_plan", [])
                )
            ),
            None,
        )
        if row is None or not math.isfinite(float(row["store_systematic"])):
            return None
        # store_choice_score differs from store_systematic only by the frozen
        # product-profile shock. Keep it inside the line option, while moving
        # the three request-level terms into the bundle evaluator.
        profile_taste = float(row["store_choice_score"]) - float(
            row["store_systematic"]
        )
        non_cash = (
            float(row["store_systematic"])
            + float(row.get("rental_tail_penalty",0.))
            + float(row["store_cash_penalty"])
            * float(row["store_cash_salience"])
            - float(row["store_service_value"])
            + float(row["return_burden"])
            + profile_taste
        )
        return BundleStoreOption(
            line_id=line.line_id,
            option_id=str(option["option_id"]),
            offer_id=offer_id,
            promise_profile_id=str(option["promise_profile_id"]),
            required=bool(line.required),
            non_cash_value=non_cash,
            total_rent=float(option["total_rent"]),
            total_deposit=float(option["locked_security_deposit"]),
            equipment_fit_value=float(row["store_functional_value"]),
            price_relative_value=float(row.get("rental_relative_term") or 0.0),
            profile_taste_value=profile_taste,
        )

    def _bundle_outside_options(
        self,
        state: WorldState,
        request: ReservationRequest,
        line: ReservationLine,
    ) -> tuple[tuple[BundleOutsideOption, ...], Any]:
        """Return complete-line outside alternatives and its frozen trace."""

        model = self.channel_demand_model
        if model is None:
            raise AssertionError("channel demand model is unavailable")
        session = self._restore_quote_context(
            request.quote_context_by_line[line.line_id]
        )
        views = self._channel_offer_views(state, session)
        trace = self._evaluate_channel_views(session, views)
        context = trace.rental_context
        if context is None:
            raise AssertionError("request-level rental trace lacks context")
        online_penalty = float(context["online_penalty"])
        late_probability = float(context["online_late_probability"])
        options: list[BundleOutsideOption] = []
        for index, row in enumerate(
            sorted(trace.profiles, key=lambda item: str(item["profile_id"])),
            start=1,
        ):
            online_systematic = float(row["online_systematic"])
            if not math.isfinite(online_systematic):
                continue
            profile_id = str(row["profile_id"])
            profile_taste = model.profile_shock_scale * self._choice_shock(
                session,
                "offer",
                f"channel-profile:{profile_id}",
            )
            # Cash and delivery are aggregated by the bundle kernel. Adding
            # their old line-level deductions back here leaves only functional
            # value, ownership/reuse terms, and the product-profile taste.
            non_cash = (
                online_systematic
                + float(row["online_cash_penalty"])
                + online_penalty
                + profile_taste
            )
            options.append(
                BundleOutsideOption(
                    line_id=line.line_id,
                    option_id=f"online-{index:02d}",
                    channel="online",
                    non_cash_value_before_delivery=non_cash,
                    cash_price=float(row["online_cash_price"]),
                    online_on_time_probability=1.0 - late_probability,
                )
            )
        existing = float(trace.channel_systematic.get("existing", float("-inf")))
        if math.isfinite(existing):
            options.append(
                BundleOutsideOption(
                    line_id=line.line_id,
                    option_id="existing",
                    channel="existing",
                    non_cash_value_before_delivery=existing,
                    cash_price=0.0,
                )
            )
        return tuple(options), trace

    def _reservation_bundle_choice_input(
        self,
        state: WorldState,
        request: ReservationRequest,
        enriched_by_line: dict[str, list[dict[str, Any]]],
    ) -> BundleChoiceInput:
        model = self.channel_demand_model
        if model is None:
            raise AssertionError("channel demand model is unavailable")
        if model.format != REQUEST_BUNDLE_CHANNEL_DEMAND_FORMAT:
            raise AssertionError("request bundle choice requires channel model v5")

        bundle_lines: list[BundleChoiceLine] = []
        trace_by_line: dict[str, Any] = {}
        for line in request.lines:
            store_options = tuple(
                option
                for option in (
                    self._bundle_store_option(state, request, line, row)
                    for row in enriched_by_line.get(line.line_id, [])
                )
                if option is not None
            )
            outside_options, trace = self._bundle_outside_options(
                state, request, line
            )
            trace_by_line[line.line_id] = trace
            bundle_lines.append(
                BundleChoiceLine(
                    line_id=line.line_id,
                    required=bool(line.required),
                    store_options=store_options,
                    outside_options=outside_options,
                )
            )

        primary_line = next(line for line in request.lines if line.required)
        primary_session = self._restore_quote_context(
            request.quote_context_by_line[primary_line.line_id]
        )
        primary_context = trace_by_line[primary_line.line_id].rental_context
        if primary_context is None:
            raise AssertionError("request-level rental trace lacks context")
        world = model.worlds[self.config.demand_world_id]
        base_segment = model.segments[request.customer_segment]
        party_equivalent = max(
            1.0,
            float(primary_session.adults)
            + 0.6 * float(primary_session.children),
        )
        cash_segment = effective_rental_cash_segment(
            model,
            base_segment,
            duration_days=int(primary_session.duration_days or 1),
            party_equivalent=party_equivalent,
            request_budget_multiplier=float(request.request_budget_multiplier),
        )
        event_address = tuple(map(int, request.event_address))
        channel_tastes = {
            channel: float(
                self._stable_event_rng(
                    _RNG_RESERVATION_BUNDLE_CHOICE,
                    *event_address,
                    identity=f"request-channel:{channel}",
                ).uniform(-1.0, 1.0)
            )
            for channel in ("store", "outside", "abandon")
        }
        return BundleChoiceInput(
            lines=tuple(bundle_lines),
            world=world,
            cash_segment=cash_segment,
            deposit_friction_rate=float(model.deposit_friction_rate),
            store_cash_salience=float(model.rental_request_cash_salience),
            store_service_value=float(primary_context["store_service_value"]),
            return_burden=float(primary_context["return_burden"]),
            online_friction=float(world.online_friction),
            online_lateness_loss_rate=float(
                model.rental_delivery_lateness_loss_rate
            ),
            occasion_importance=float(primary_session.occasion_importance),
            channel_taste_scale=float(model.channel_taste_scale),
            channel_tastes=channel_tastes,
            rental_budget_tail_coefficient=model.rental_budget_tail_coefficient,
            store_reputation_penalty=float(
                request.service_reputation_snapshot.get(
                    "customer_choice_penalty", 0.0
                )
                if self.config.reservation_service_reputation_enabled
                else 0.0
            ),
        )

    def _evaluate_reservation_quote_bundle(
        self,
        state: WorldState,
        request: ReservationRequest,
        enriched_by_line: dict[str, list[dict[str, Any]]],
    ) -> Any:
        return evaluate_reservation_bundle_choice(
            self._reservation_bundle_choice_input(
                state, request, enriched_by_line
            )
        )

    def _request_bundle_recovery_offers(
        self,
        state: WorldState,
        request: ReservationRequest,
        selected_by_line: dict[str, dict[str, Any]],
    ) -> dict[str, list[str]]:
        """Replay full locked contracts for same-Profile exact substitutes."""

        recovery: dict[str, list[str]] = {
            line.line_id: [] for line in request.lines
        }
        line_by_id = {line.line_id: line for line in request.lines}
        for line_id, selected in selected_by_line.items():
            profile_id = str(selected["promise_profile_id"])
            selected_offer_id = str(selected["offer_id"])
            members = request.inquiry_profile_member_ids_by_line[line_id][
                profile_id
            ]
            for offer_id in sorted(members):
                if offer_id == selected_offer_id:
                    continue
                candidate = self._enriched_pairwise_quote_options(
                    request,
                    line_by_id[line_id],
                    quote_options=[
                        {
                            "option_id": f"recovery:{offer_id}",
                            "offer_id": offer_id,
                            "promise_profile_id": profile_id,
                            "daily_unit_price": float(
                                selected["daily_unit_price"]
                            ),
                        }
                    ],
                )[0]
                replay = {
                    other_line_id: [
                        candidate
                        if other_line_id == line_id
                        else other_selected
                    ]
                    for other_line_id, other_selected in selected_by_line.items()
                }
                trace = self._evaluate_reservation_quote_bundle(
                    state, request, replay
                )
                if (
                    trace.selected_channel == "store"
                    and trace.selected_store_offer_ids.get(line_id) == offer_id
                ):
                    recovery[line_id].append(offer_id)
        return recovery

    def _quoted_pairwise_request_outcomes(
        self,
        state: WorldState,
        request: ReservationRequest,
        terms: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], bool]:
        """Choose one complete V18 contract, outside plan, or abandonment."""

        enriched_by_line = {
            line.line_id: self._enriched_pairwise_quote_options(
                request,
                line,
                quote_options=list(terms[line.line_id]["quote_options"]),
            )
            for line in request.lines
        }
        trace = self._evaluate_reservation_quote_bundle(
            state, request, enriched_by_line
        )
        if trace.selected_channel != "store":
            return (
                {
                    line.line_id: self._empty_pairwise_line_outcome(
                        enriched_by_line[line.line_id]
                    )
                    for line in request.lines
                },
                True,
            )

        selected_by_line: dict[str, dict[str, Any]] = {}
        outcomes: dict[str, dict[str, Any]] = {}
        for line in request.lines:
            enriched = enriched_by_line[line.line_id]
            selected_option_id = trace.selected_store_options.get(line.line_id)
            selected = next(
                (
                    row
                    for row in enriched
                    if str(row["option_id"]) == selected_option_id
                ),
                None,
            )
            if selected is None:
                outcomes[line.line_id] = self._empty_pairwise_line_outcome(
                    enriched
                )
                continue
            selected_by_line[line.line_id] = selected
            offer_id = str(selected["offer_id"])
            outcomes[line.line_id] = {
                "submitted_quote_options": enriched,
                "selected_quote_option_id": str(selected["option_id"]),
                "primary_offer_id": offer_id,
                "authorized_offer_ids": [offer_id],
                "fallback_offer_ids": [],
                "locked_security_deposit": float(
                    selected["locked_security_deposit"]
                ),
                "authorized_ceilings": {},
                "recovery_ceilings": {},
                **selected,
            }
        recovery = self._request_bundle_recovery_offers(
            state, request, selected_by_line
        )
        for line_id, offer_ids in recovery.items():
            outcomes[line_id]["fallback_offer_ids"] = list(offer_ids)
        return outcomes, False

    def _quoted_complete_contract_request_outcomes(
        self,
        state: WorldState,
        request: ReservationRequest,
        terms: dict[str, Any],
    ) -> tuple[
        dict[str, dict[str, Any]],
        bool,
        Any,
        dict[str, CompleteContractOption],
    ]:
        """Evaluate only the one or two explicitly submitted contracts."""

        merged_by_line: dict[str, list[dict[str, Any]]] = {
            line.line_id: [] for line in request.lines
        }
        enriched_by_contract: dict[str, dict[str, dict[str, Any]]] = {}
        public_contracts: list[dict[str, Any]] = []
        line_by_id = {line.line_id: line for line in request.lines}
        for contract in list(terms["contract_options"]):
            contract_id = str(contract["contract_option_id"])
            contract_rows: dict[str, dict[str, Any]] = {}
            public_lines: list[dict[str, Any]] = []
            for term in list(contract["lines"]):
                line_id = str(term["line_id"])
                if term["disposition"] == "omit":
                    public_lines.append(
                        {"line_id": line_id, "disposition": "omit"}
                    )
                    continue
                raw = {
                    "option_id": f"{contract_id}:{line_id}",
                    "offer_id": str(term["offer_id"]),
                    "promise_profile_id": str(term["promise_profile_id"]),
                    "daily_unit_price": float(term["daily_unit_price"]),
                }
                enriched = self._enriched_pairwise_quote_options(
                    request,
                    line_by_id[line_id],
                    quote_options=[raw],
                )[0]
                contract_rows[line_id] = enriched
                merged_by_line[line_id].append(enriched)
                public_lines.append(
                    {
                        "line_id": line_id,
                        "disposition": "quote",
                        "offer_id": str(enriched["offer_id"]),
                        "promise_profile_id": str(
                            enriched["promise_profile_id"]
                        ),
                        "profile_family_id": str(
                            enriched["profile_family_id"]
                        ),
                        "service_units": int(
                            line_by_id[line_id].service_units
                            or line_by_id[line_id].customer_units
                        ),
                        "capacity_per_asset": int(
                            enriched["capacity_per_asset"]
                        ),
                        "asset_quantity": int(enriched["asset_quantity"]),
                        "daily_unit_price": float(
                            enriched["daily_unit_price"]
                        ),
                        "line_rent": float(enriched["total_rent"]),
                        "security_deposit": float(
                            enriched["locked_security_deposit"]
                        ),
                    }
                )
            enriched_by_contract[contract_id] = contract_rows
            public_contracts.append(
                {
                    "contract_option_id": contract_id,
                    "lines": public_lines,
                    "total_rent": round(
                        sum(
                            float(row.get("line_rent", 0.0))
                            for row in public_lines
                        ),
                        2,
                    ),
                    "total_security_deposit": round(
                        sum(
                            float(row.get("security_deposit", 0.0))
                            for row in public_lines
                        ),
                        2,
                    ),
                }
            )

        choice_input = self._reservation_bundle_choice_input(
            state, request, merged_by_line
        )
        option_by_key = {
            (line.line_id, option.option_id): option
            for line in choice_input.lines
            for option in line.store_options
        }
        complete_by_id: dict[str, CompleteContractOption] = {}
        for contract in public_contracts:
            contract_id = str(contract["contract_option_id"])
            complete_by_id[contract_id] = CompleteContractOption(
                contract_option_id=contract_id,
                lines=tuple(
                    option_by_key[(line_id, str(row["option_id"]))]
                    for line_id, row in sorted(
                        enriched_by_contract[contract_id].items()
                    )
                ),
                omitted_optional_line_ids=tuple(
                    sorted(
                        str(row["line_id"])
                        for row in contract["lines"]
                        if row["disposition"] == "omit"
                    )
                ),
            )
        trace = evaluate_complete_contract_choice(
            choice_input,
            tuple(complete_by_id[contract_id] for contract_id in sorted(complete_by_id)),
        )
        request.submitted_contract_options = public_contracts
        request.selected_contract_option_id = trace.selected_contract_option_id
        self._complete_contract_assessments(
            state,
            request,
            choice_input=choice_input,
            trace=trace,
            complete_by_id=complete_by_id,
        )
        if trace.selected_channel != "store":
            return (
                {
                    line.line_id: self._empty_pairwise_line_outcome([])
                    for line in request.lines
                },
                True,
                trace,
                complete_by_id,
            )

        selected_id = str(trace.selected_contract_option_id)
        selected_rows = enriched_by_contract[selected_id]
        outcomes: dict[str, dict[str, Any]] = {}
        selected_by_line: dict[str, dict[str, Any]] = {}
        for line in request.lines:
            selected = selected_rows.get(line.line_id)
            if selected is None:
                outcomes[line.line_id] = self._empty_pairwise_line_outcome([])
                continue
            selected_by_line[line.line_id] = selected
            offer_id = str(selected["offer_id"])
            outcomes[line.line_id] = {
                "submitted_quote_options": [dict(selected)],
                "selected_quote_option_id": selected_id,
                "primary_offer_id": offer_id,
                "authorized_offer_ids": [offer_id],
                "fallback_offer_ids": [],
                "locked_security_deposit": float(
                    selected["locked_security_deposit"]
                ),
                "authorized_ceilings": {},
                "recovery_ceilings": {},
                **selected,
            }
        recovery = self._complete_contract_recovery_offers(
            state,
            request,
            choice_input=choice_input,
            selected_contract=complete_by_id[selected_id],
            selected_by_line=selected_by_line,
        )
        for line_id, offer_ids in recovery.items():
            outcomes[line_id]["fallback_offer_ids"] = list(offer_ids)
        return outcomes, False, trace, complete_by_id

    def _quoted_factorized_customer_plan_set(
        self,
        state: WorldState,
        request: ReservationRequest,
        terms: dict[str, Any],
    ) -> tuple[Any, dict[str, CompleteContractOption]]:
        """Expand line options privately and expose only the top authorized plans."""

        line_by_id = {line.line_id: line for line in request.lines}
        normalized_by_line = {
            str(row["line_id"]): row for row in terms["line_options"]
        }
        enriched_by_line: dict[str, list[dict[str, Any]]] = {}
        submitted_lines: list[dict[str, Any]] = []
        dimensions: list[tuple[dict[str, Any] | None, ...]] = []
        for line_id in sorted(line_by_id):
            line = line_by_id[line_id]
            raw_options = list(normalized_by_line[line_id]["options"])
            quoted = self._enriched_pairwise_quote_options(
                request,
                line,
                quote_options=[
                    row for row in raw_options if row["disposition"] == "quote"
                ],
            )
            enriched_by_line[line_id] = quoted
            public_options = [
                {
                    "option_id": str(row["option_id"]),
                    "disposition": "quote",
                    "offer_id": str(row["offer_id"]),
                    "promise_profile_id": str(row["promise_profile_id"]),
                    "profile_family_id": str(row["profile_family_id"]),
                    "service_units": int(line.service_units or line.customer_units),
                    "capacity_per_asset": int(row["capacity_per_asset"]),
                    "asset_quantity": int(row["asset_quantity"]),
                    "daily_unit_price": float(row["daily_unit_price"]),
                    "line_rent": float(row["total_rent"]),
                    "security_deposit": float(row["locked_security_deposit"]),
                }
                for row in quoted
            ]
            if not line.required:
                public_options.append({"option_id": "omit", "disposition": "omit"})
            submitted_lines.append(
                {
                    "line_id": line_id,
                    "required": bool(line.required),
                    "options": public_options,
                }
            )
            dimensions.append(tuple(quoted) if line.required else (None, *quoted))

        choice_input = self._reservation_bundle_choice_input(
            state, request, enriched_by_line
        )
        option_by_key = {
            (line.line_id, option.option_id): option
            for line in choice_input.lines
            for option in line.store_options
        }
        complete_by_id: dict[str, CompleteContractOption] = {}
        public_by_internal_id: dict[str, dict[str, Any]] = {}
        for index, configuration in enumerate(
            itertools.product(*dimensions), start=1
        ):
            internal_id = f"internal-plan-{index:03d}"
            selected = {
                line_id: option
                for line_id, option in zip(sorted(line_by_id), configuration)
                if option is not None
            }
            omitted = tuple(
                line_id
                for line_id, option in zip(sorted(line_by_id), configuration)
                if option is None
            )
            complete_by_id[internal_id] = CompleteContractOption(
                contract_option_id=internal_id,
                lines=tuple(
                    option_by_key[(line_id, str(row["option_id"]))]
                    for line_id, row in sorted(selected.items())
                ),
                omitted_optional_line_ids=omitted,
            )
            plan_lines: list[dict[str, Any]] = []
            for line_id in sorted(line_by_id):
                line = line_by_id[line_id]
                row = selected.get(line_id)
                if row is None:
                    plan_lines.append(
                        {
                            "line_id": line_id,
                            "required": False,
                            "disposition": "omit",
                        }
                    )
                    continue
                plan_lines.append(
                    {
                        "line_id": line_id,
                        "required": bool(line.required),
                        "disposition": "quote",
                        "line_option_id": str(row["option_id"]),
                        "offer_id": str(row["offer_id"]),
                        "promise_profile_id": str(row["promise_profile_id"]),
                        "asset_quantity": int(row["asset_quantity"]),
                        "daily_unit_price": float(row["daily_unit_price"]),
                        "line_rent": float(row["total_rent"]),
                        "security_deposit": float(
                            row["locked_security_deposit"]
                        ),
                    }
                )
            total_rent = round(
                sum(float(row.get("line_rent", 0.0)) for row in plan_lines), 2
            )
            public_by_internal_id[internal_id] = {
                "lines": plan_lines,
                "total_rent": total_rent,
                "prepayment_due": round(
                    total_rent * float(request.prepayment_rate), 2
                ),
                "total_security_deposit": round(
                    sum(
                        float(row.get("security_deposit", 0.0))
                        for row in plan_lines
                    ),
                    2,
                ),
            }

        if len(complete_by_id) != int(terms["internal_plan_count"]):
            raise AssertionError("factorized plan count changed after validation")
        trace = evaluate_customer_plan_set(
            choice_input,
            tuple(complete_by_id[plan_id] for plan_id in sorted(complete_by_id)),
            authorization_limit=3,
            internal_plan_limit=24,
        )
        authorized_plans: list[dict[str, Any]] = []
        returned_internal_ids: dict[str, str] = {}
        public_id_by_internal_id: dict[str, str] = {}
        for index, internal_id in enumerate(
            sorted(trace.authorized_plan_ids), start=1
        ):
            plan_id = f"plan-{index:02d}"
            returned_internal_ids[plan_id] = internal_id
            public_id_by_internal_id[internal_id] = plan_id
            authorized_plans.append(
                {
                    "plan_id": plan_id,
                    **public_by_internal_id[internal_id],
                }
            )
        request.submitted_line_options = submitted_lines
        request.customer_authorized_plans = authorized_plans
        request.customer_preferred_plan_id = (
            public_id_by_internal_id.get(
                str(trace.customer_preferred_plan_id)
            )
            if authorized_plans
            else None
        )
        request.internal_plan_count = int(trace.internal_plan_count)
        request.acceptable_plan_count = int(trace.acceptable_plan_count)
        request.plan_set_feedback_audit = {
            "returned_internal_plan_ids": returned_internal_ids,
            "acceptable_internal_plan_ids": list(trace.acceptable_plan_ids),
            "plan_scores": dict(trace.plan_scores),
            "outside_score": float(trace.outside_score),
            "abandon_score": float(trace.abandon_score),
        }
        return trace, complete_by_id

    def _complete_contract_score(
        self,
        value: BundleChoiceInput,
        contract: CompleteContractOption,
    ) -> float:
        return float(score_complete_store_contract(value, contract))

    def _factorized_price_curve_action_gain(
        self,
        state: WorldState,
        request: ReservationRequest,
        *,
        choice_input: BundleChoiceInput,
        contract: CompleteContractOption,
    ) -> tuple[float, dict[str, float]]:
        """Average full-choice response over the frozen V23 price curve."""

        policy = self.plan_set_feedback_policy
        if policy is None or not policy.factorized_concern_tags_enabled:
            raise AssertionError("V23 price evidence requires its frozen policy")
        baseline = self._complete_contract_score(choice_input, contract)
        line_by_id = {line.line_id: line for line in request.lines}
        gains: dict[str, float] = {}
        for discount in tuple(policy.price_curve_discount_fractions or ()):
            repriced: list[BundleStoreOption] = []
            for selected in contract.lines:
                line = line_by_id[selected.line_id]
                profile = self.reservation_promise_catalog.by_id[
                    selected.promise_profile_id
                ]
                quantity = math.ceil(
                    int(line.service_units or line.customer_units)
                    / int(profile.capacity_per_asset)
                )
                daily = max(
                    0.01,
                    round(
                        float(selected.total_rent)
                        / quantity
                        / int(line.duration_days)
                        * (1.0 - float(discount)),
                        2,
                    ),
                )
                raw = self._enriched_pairwise_quote_options(
                    request,
                    line,
                    quote_options=[
                        {
                            "option_id": f"feedback-price:{selected.line_id}",
                            "offer_id": selected.offer_id,
                            "promise_profile_id": selected.promise_profile_id,
                            "daily_unit_price": daily,
                        }
                    ],
                )[0]
                option = self._bundle_store_option(state, request, line, raw)
                if option is None:
                    raise AssertionError(
                        "a quoted plan line lost its V23 price replay"
                    )
                repriced.append(option)
            changed = CompleteContractOption(
                contract_option_id=contract.contract_option_id,
                lines=tuple(repriced),
                omitted_optional_line_ids=contract.omitted_optional_line_ids,
            )
            gains[str(int(round(float(discount) * 100)))] = max(
                0.0,
                self._complete_contract_score(choice_input, changed) - baseline,
            )
        if not gains:
            raise AssertionError("V23 price curve must contain at least one point")
        return statistics.mean(gains.values()), gains

    def _factorized_equipment_action_evidence_by_line(
        self,
        state: WorldState,
        request: ReservationRequest,
        *,
        choice_input: BundleChoiceInput,
        contract: CompleteContractOption,
    ) -> tuple[float, dict[str, float]]:
        """Robust top-two public line-replacement evidence with cash frozen."""

        policy = self.plan_set_feedback_policy
        if policy is None or not policy.factorized_concern_tags_enabled:
            raise AssertionError("V23 equipment evidence requires its policy")

        def systematic_score(value: CompleteContractOption) -> float:
            return self._complete_contract_score(choice_input, value) - sum(
                float(option.profile_taste_value) for option in value.lines
            )

        baseline = systematic_score(contract)
        line_by_id = {line.line_id: line for line in request.lines}
        selected_by_line = {option.line_id: option for option in contract.lines}
        by_line: dict[str, float] = {}
        top_k = int(policy.equipment_action_top_k or 0)
        for line_id, selected in selected_by_line.items():
            line = line_by_id[line_id]
            candidate_gains = [0.0]
            seen: set[tuple[str, str]] = set()
            candidate_index = 0
            for profile_id in line.eligible_profile_ids:
                profile = self.reservation_promise_catalog.by_id[profile_id]
                quantity = math.ceil(
                    int(line.service_units or line.customer_units)
                    / int(profile.capacity_per_asset)
                )
                daily = max(
                    0.01,
                    round(
                        float(selected.total_rent)
                        / quantity
                        / int(line.duration_days),
                        2,
                    ),
                )
                for offer_id in request.inquiry_profile_member_ids_by_line[
                    line_id
                ][profile_id]:
                    key = (str(profile_id), str(offer_id))
                    if key in seen:
                        continue
                    seen.add(key)
                    if key == (
                        str(selected.promise_profile_id),
                        str(selected.offer_id),
                    ):
                        continue
                    candidate_index += 1
                    raw = self._enriched_pairwise_quote_options(
                        request,
                        line,
                        quote_options=[
                            {
                                "option_id": (
                                    f"feedback-equipment:{line_id}:"
                                    f"{candidate_index:03d}"
                                ),
                                "offer_id": str(offer_id),
                                "promise_profile_id": str(profile_id),
                                "daily_unit_price": daily,
                            }
                        ],
                    )[0]
                    alternative = self._bundle_store_option(
                        state, request, line, raw
                    )
                    if alternative is None:
                        continue
                    # The cent-exact daily price keeps line rent approximately
                    # fixed; freezing the deposit isolates equipment/Profile
                    # content. The real alternative remains undisclosed and
                    # may still lose after its actual deposit and joint terms.
                    alternative = replace(
                        alternative,
                        total_deposit=float(selected.total_deposit),
                    )
                    changed = CompleteContractOption(
                        contract_option_id=contract.contract_option_id,
                        lines=tuple(
                            alternative if row.line_id == line_id else row
                            for row in contract.lines
                        ),
                        omitted_optional_line_ids=(
                            contract.omitted_optional_line_ids
                        ),
                    )
                    candidate_gains.append(
                        max(0.0, systematic_score(changed) - baseline)
                    )
            top = sorted(candidate_gains, reverse=True)[:top_k]
            if len(top) < top_k:
                top.extend([0.0] * (top_k - len(top)))
            by_line[line_id] = statistics.mean(top)
        return max(by_line.values(), default=0.0), by_line

    def _complete_contract_configuration_gain(
        self,
        state: WorldState,
        request: ReservationRequest,
        *,
        choice_input: BundleChoiceInput,
        contract: CompleteContractOption,
    ) -> float:
        """Best one-line fit change with that line's cash/deposit frozen."""

        baseline = self._complete_contract_score(choice_input, contract)
        best = baseline
        line_by_id = {line.line_id: line for line in request.lines}
        selected_by_line = {row.line_id: row for row in contract.lines}
        for line_id, selected in selected_by_line.items():
            line = line_by_id[line_id]
            candidate_index = 0
            seen: set[tuple[str, str]] = set()
            for profile_id in line.eligible_profile_ids:
                for offer_id in request.inquiry_profile_member_ids_by_line[
                    line_id
                ][profile_id]:
                    key = (str(profile_id), str(offer_id))
                    if key in seen:
                        continue
                    seen.add(key)
                    candidate_index += 1
                    profile = self.reservation_promise_catalog.by_id[profile_id]
                    quantity = math.ceil(
                        int(line.service_units or line.customer_units)
                        / profile.capacity_per_asset
                    )
                    price = max(
                        0.01,
                        round(
                            float(selected.total_rent)
                            / (quantity * (
                                1.0 + 0.5 * (line.duration_days - 1)
                                if self._v45_reservation_difficulty()
                                else line.duration_days)),
                            2,
                        ),
                    )
                    enriched = self._enriched_pairwise_quote_options(
                        request,
                        line,
                        quote_options=[
                            {
                                "option_id": (
                                    f"feedback:{line_id}:{candidate_index:03d}"
                                ),
                                "offer_id": str(offer_id),
                                "promise_profile_id": str(profile_id),
                                "daily_unit_price": price,
                            }
                        ],
                    )[0]
                    alternative = self._bundle_store_option(
                        state, request, line, enriched
                    )
                    if alternative is None:
                        continue
                    alternative = replace(
                        alternative,
                        total_rent=float(selected.total_rent),
                        total_deposit=float(selected.total_deposit),
                    )
                    replacement = CompleteContractOption(
                        contract_option_id=contract.contract_option_id,
                        lines=tuple(
                            alternative if row.line_id == line_id else row
                            for row in contract.lines
                        ),
                        omitted_optional_line_ids=(
                            contract.omitted_optional_line_ids
                        ),
                    )
                    best = max(
                        best,
                        self._complete_contract_score(choice_input, replacement),
                    )
        return max(0.0, best - baseline)

    def _complete_contract_assessments(
        self,
        state: WorldState,
        request: ReservationRequest,
        *,
        choice_input: BundleChoiceInput,
        trace: Any,
        complete_by_id: dict[str, CompleteContractOption],
    ) -> None:
        policy = self.contract_feedback_policy
        if policy is None:
            raise AssertionError(
                "complete_contract_options_v1 requires a feedback policy"
            )
        selected_id = trace.selected_contract_option_id
        unselected_ids = [
            contract_id
            for contract_id in sorted(complete_by_id)
            if contract_id != selected_id
        ]
        if not unselected_ids:
            request.feedback_response_available = None
            request.reported_contract_assessments = [
                {
                    "contract_option_id": str(selected_id),
                    "outcome": "selected",
                }
            ]
            request.contract_feedback_audit = {}
            return
        event_address = tuple(map(int, request.event_address))
        response_uniform = float(
            self._stable_event_rng(
                _RNG_RESERVATION_FEEDBACK_RESPONSE,
                *event_address,
                identity="complete-contract-feedback-response",
            ).random()
        )
        available = response_uniform < float(policy.response_probability)
        request.feedback_response_available = available
        external_score = max(float(trace.outside_score), float(trace.abandon_score))
        reference_score = (
            float(trace.contract_scores[selected_id])
            if selected_id is not None
            else external_score
        )
        assessments: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        for contract_id in sorted(complete_by_id):
            if contract_id == selected_id:
                assessments.append(
                    {"contract_option_id": contract_id, "outcome": "selected"}
                )
                continue
            contract = complete_by_id[contract_id]
            score = float(trace.contract_scores[contract_id])
            discounted = CompleteContractOption(
                contract_option_id=contract.contract_option_id,
                lines=tuple(
                    replace(
                        row,
                        total_rent=round(
                            float(row.total_rent)
                            * (1.0 - float(policy.price_discount_fraction)),
                            2,
                        ),
                    )
                    for row in contract.lines
                ),
                omitted_optional_line_ids=contract.omitted_optional_line_ids,
            )
            price_gain = max(
                0.0,
                self._complete_contract_score(choice_input, discounted) - score,
            )
            configuration_gain = self._complete_contract_configuration_gain(
                state,
                request,
                choice_input=choice_input,
                contract=contract,
            )
            diagnostic = ContractAssessmentDiagnostic(
                contract_option_id=contract_id,
                contract_score=score,
                reference_score=reference_score,
                external_score=external_score,
                price_gain=price_gain,
                configuration_gain=configuration_gain,
                another_store_contract_selected=selected_id is not None,
            )
            # Response availability is request-shared.  Assessment content
            # noise is contract-specific so two labels are not coupled to the
            # canonical display order by reuse of one categorical draw.
            report_uniform = float(
                self._stable_event_rng(
                    _RNG_RESERVATION_FEEDBACK_REPORT,
                    *event_address,
                    identity=(
                        "complete-contract-feedback-report:" + contract_id
                    ),
                ).random()
            )
            reported, latent = reported_assessment_category(
                policy,
                diagnostic,
                report_uniform=report_uniform,
            )
            assessment = {
                "contract_option_id": contract_id,
                "outcome": "not_selected",
                "reported_assessment": (
                    {"available": True, "category": reported}
                    if available
                    else {"available": False}
                ),
            }
            assessments.append(assessment)
            audit_rows.append(
                {
                    "contract_option_id": contract_id,
                    "contract_score": score,
                    "reference_score": reference_score,
                    "external_score": external_score,
                    "price_gain": price_gain,
                    "configuration_gain": configuration_gain,
                    "another_store_contract_selected": (
                        selected_id is not None
                    ),
                    "report_uniform": report_uniform,
                    "latent_topic": latent,
                    "reported_category": reported,
                }
            )
        request.reported_contract_assessments = assessments
        request.contract_feedback_audit = {
            "response_uniform": response_uniform,
            "response_available": available,
            "rows": audit_rows,
        }

    def _complete_contract_recovery_offers(
        self,
        state: WorldState,
        request: ReservationRequest,
        *,
        choice_input: BundleChoiceInput,
        selected_contract: CompleteContractOption,
        selected_by_line: dict[str, dict[str, Any]],
    ) -> dict[str, list[str]]:
        """Replay the selected explicit contract with one same-Profile substitute."""

        recovery = {line.line_id: [] for line in request.lines}
        line_by_id = {line.line_id: line for line in request.lines}
        selected_options = {row.line_id: row for row in selected_contract.lines}
        for line_id, selected in selected_by_line.items():
            profile_id = str(selected["promise_profile_id"])
            selected_offer_id = str(selected["offer_id"])
            for offer_id in sorted(
                request.inquiry_profile_member_ids_by_line[line_id][profile_id]
            ):
                if offer_id == selected_offer_id:
                    continue
                enriched = self._enriched_pairwise_quote_options(
                    request,
                    line_by_id[line_id],
                    quote_options=[
                        {
                            "option_id": f"recovery:{line_id}:{offer_id}",
                            "offer_id": offer_id,
                            "promise_profile_id": profile_id,
                            "daily_unit_price": float(
                                selected["daily_unit_price"]
                            ),
                        }
                    ],
                )[0]
                alternative = self._bundle_store_option(
                    state, request, line_by_id[line_id], enriched
                )
                if alternative is None:
                    continue
                replay = CompleteContractOption(
                    contract_option_id=selected_contract.contract_option_id,
                    lines=tuple(
                        alternative if row.line_id == line_id else row
                        for row in selected_options.values()
                    ),
                    omitted_optional_line_ids=(
                        selected_contract.omitted_optional_line_ids
                    ),
                )
                replay_trace = evaluate_complete_contract_choice(
                    choice_input, (replay,)
                )
                if replay_trace.selected_channel == "store":
                    recovery[line_id].append(offer_id)
        return recovery

    def _confirm_factorized_plan(
        self,
        state: WorldState,
        request: ReservationRequest,
        *,
        plan_id: str,
        origin: str,
    ) -> dict[str, Any]:
        """Create cash, commitment, and calendar effects for one authorized plan."""

        plan = next(
            (
                row
                for row in request.customer_authorized_plans
                if str(row["plan_id"]) == str(plan_id)
            ),
            None,
        )
        if plan is None:
            raise ValueError(
                f"plan_id is not customer-authorized for {request.request_id}: {plan_id}"
            )
        if request.status not in {
            "awaiting_manager_confirmation",
            "customer_authorized_single_plan",
        }:
            raise ValueError(
                f"request is not awaiting plan confirmation: {request.request_id}"
            )
        line_by_id = {line.line_id: line for line in request.lines}
        plan_by_line = {str(row["line_id"]): row for row in plan["lines"]}
        selected_by_line: dict[str, dict[str, Any]] = {}
        for line_id, row in plan_by_line.items():
            if row["disposition"] == "omit":
                continue
            selected_by_line[line_id] = self._enriched_pairwise_quote_options(
                request,
                line_by_id[line_id],
                quote_options=[
                    {
                        "option_id": str(row["line_option_id"]),
                        "offer_id": str(row["offer_id"]),
                        "promise_profile_id": str(row["promise_profile_id"]),
                        "daily_unit_price": float(row["daily_unit_price"]),
                    }
                ],
            )[0]
        if request.service_offer_terms:
            # A binding customer offer is not reevaluated at acceptance, even
            # to derive latent fallback. Its recovery snapshot was frozen too.
            recovery = copy.deepcopy(request.service_offer_audit["recovery_by_plan_id"][str(plan_id)])
        else:
            choice_input = self._reservation_bundle_choice_input(
                state, request, {line_id: [row] for line_id, row in selected_by_line.items()},
            )
            option_by_key = {(line.line_id, option.option_id): option
                             for line in choice_input.lines for option in line.store_options}
            selected_contract = CompleteContractOption(
                contract_option_id=str(plan_id),
                lines=tuple(option_by_key[(line_id, str(row["option_id"]))]
                            for line_id, row in sorted(selected_by_line.items())),
                omitted_optional_line_ids=tuple(sorted(line_id for line_id, row in plan_by_line.items()
                                                       if row["disposition"] == "omit")),
            )
            recovery = self._complete_contract_recovery_offers(
                state, request, choice_input=choice_input, selected_contract=selected_contract,
                selected_by_line=selected_by_line,
            )

        commitment_lines: list[ReservationLine] = []
        fallback_by_line: dict[str, list[str]] = {}
        authorized_by_line: dict[str, list[str]] = {}
        removed_optional_lines: list[str] = []
        for line in request.lines:
            selected = selected_by_line.get(line.line_id)
            fallback_by_line[line.line_id] = []
            authorized_by_line[line.line_id] = []
            if selected is None:
                if line.required:
                    raise AssertionError("an authorized plan omitted a required line")
                line.status = "optional_removed_at_confirmation"
                line.locked_daily_price = 0.0
                line.locked_rent = 0.0
                line.locked_security_deposit = 0.0
                removed_optional_lines.append(line.line_id)
                continue
            primary = str(selected["offer_id"])
            line.offer_id = primary
            line.service_line = self.offers[primary].service_line
            line.selected_quote_option_id = str(selected["option_id"])
            line.selected_offer_id = primary
            line.quoted_daily_price = float(selected["daily_unit_price"])
            line.promise_profile_id = str(selected["promise_profile_id"])
            line.profile_family_id = str(selected["profile_family_id"])
            line.profile_schema_version = str(selected["profile_schema_version"])
            line.capacity_per_asset = int(selected["capacity_per_asset"])
            line.asset_quantity = int(selected["asset_quantity"])
            line.quantity = int(selected["asset_quantity"])
            line.locked_daily_price = float(selected["daily_unit_price"])
            line.locked_rent = round(
                (rental_amount(
                    request.service_offer_terms["line_daily_rents"][line.line_id],
                    line.duration_days)
                 if self._v45_reservation_difficulty() and request.service_offer_terms
                 else rental_amount(line.locked_daily_price, line.duration_days, line.quantity)
                 if self._v45_reservation_difficulty()
                 else line.locked_daily_price * line.quantity * line.duration_days), 2
            )
            line.authorized_offer_ids = [primary]
            line.locked_security_deposit = float(
                selected["locked_security_deposit"]
            )
            if self._v24_reservation_operations():
                line.authorized_line_rent = float(line.locked_rent)
                line.authorized_security_deposit = float(
                    line.locked_security_deposit
                )
            authorized_by_line[line.line_id] = [primary]
            fallback_by_line[line.line_id] = list(recovery.get(line.line_id, []))
            line.substitution_hint = (
                "锁定 exact 无法履约时，才可通过 inquire_fallback 询问同价同押金"
                "且不降低 Profile 保证的客户备选。"
                if fallback_by_line[line.line_id]
                else None
            )
            line.status = "committed"
            commitment_lines.append(replace(line))

        request.total_locked_rent = round(
            sum(line.locked_rent for line in commitment_lines), 2
        )
        request.prepayment_due = (
            float(request.service_offer_terms["prepayment_due"])
            if request.service_offer_terms else round(request.total_locked_rent * request.prepayment_rate, 2)
        )
        request.return_day = request.pickup_day + max(
            line.duration_days for line in commitment_lines
        )
        request.fallback_offer_ids_by_line = fallback_by_line
        request.fallback_offer_ids = sorted(
            {
                offer_id
                for values in fallback_by_line.values()
                for offer_id in values
            }
        )
        request.authorized_offer_ids_by_line = authorized_by_line
        request.authorized_price_ceiling_by_line = {
            line_id: {} for line_id in authorized_by_line
        }
        request.recovery_offer_ids_by_line = {
            line_id: list(values) for line_id, values in fallback_by_line.items()
        }
        request.recovery_price_ceiling_by_line = {
            line_id: {} for line_id in fallback_by_line
        }
        request.manager_confirmed_plan_id = str(plan_id)
        request.plan_confirmation_origin = str(origin)
        request.manager_confirmation_day = state.day
        request.status = "accepted"
        if removed_optional_lines:
            state.cumulative_reservation_optional_lines_removed_at_quote += len(
                removed_optional_lines
            )

        commitment_id = f"reservation-{len(state.reservation_commitments) + 1:03d}"
        commitment = ReservationCommitment(
            commitment_id=commitment_id,
            request_id=request.request_id,
            confirmed_day=state.day,
            pickup_day=request.pickup_day,
            return_day=request.return_day,
            lines=commitment_lines,
            total_locked_rent=request.total_locked_rent,
            prepayment_collected=request.prepayment_due,
            outstanding_rent=round(
                request.total_locked_rent - request.prepayment_due, 2
            ),
            unearned_revenue=request.prepayment_due,
            original_pickup_day=(
                request.original_inquiry_pickup_day
                if request.original_inquiry_pickup_day is not None
                else request.pickup_day
            ),
            original_return_day=(
                request.original_inquiry_return_day
                if request.original_inquiry_return_day is not None
                else request.return_day
            ),
            promised_return_day=request.return_day,
            approved_return_day=request.return_day,
            estimated_return_day=request.return_day,
            **(
                {
                    "preauthorized_recovery_plans": [
                        copy.deepcopy(row)
                        for row in request.customer_authorized_plans
                        if str(row["plan_id"]) != str(plan_id)
                    ],
                    "original_confirmed_plan_id": str(plan_id),
                    "original_total_locked_rent": request.total_locked_rent,
                    "original_total_security_deposit": round(
                        sum(
                            line.locked_security_deposit
                            for line in commitment_lines
                        ),
                        2,
                    ),
                }
                if self._v24_reservation_operations()
                else {}
            ),
            **(
                {
                    "rent_due_at_pickup": round(
                        request.total_locked_rent - request.prepayment_due, 2
                    ),
                    "reschedule_forecast_baseline_condition": (
                        request.forecast_condition
                    ),
                }
                if self._v26_contract_state()
                else {}
            ),
        )
        request.commitment_id = commitment_id
        self._initialize_v42_commitment_customer_state(
            state, request, commitment
        )
        state.reservation_commitments.append(commitment)
        for line in commitment.lines:
            for day in range(commitment.pickup_day, commitment.return_day):
                state.reservation_calendar_blocks.append(
                    CalendarBlock(
                        block_id=f"{commitment_id}-{line.line_id}-{day:03d}",
                        commitment_id=commitment_id,
                        line_id=line.line_id,
                        offer_id=line.offer_id,
                        day=day,
                        quantity=line.quantity,
                        authorized_offer_ids=list(line.authorized_offer_ids),
                        selected_offer_id=line.selected_offer_id,
                        promise_profile_id=line.promise_profile_id,
                        profile_schema_version=line.profile_schema_version,
                        service_units=line.service_units,
                        capacity_per_asset=line.capacity_per_asset,
                        staged_offer_id=line.staged_offer_id,
                    )
                )
        state.cash = round(state.cash + request.prepayment_due, 2)
        state.unearned_revenue_liability = round(
            state.unearned_revenue_liability + request.prepayment_due, 2
        )
        state.cumulative_reservation_prepayments = round(
            state.cumulative_reservation_prepayments + request.prepayment_due, 2
        )
        state.current_day_ledger["reservation_prepayments_collected"] = round(
            float(
                state.current_day_ledger.get(
                    "reservation_prepayments_collected", 0.0
                )
            )
            + request.prepayment_due,
            2,
        )
        self._assert_v26_reservation_financial_invariants(state)
        state.cumulative_reservations_confirmed += 1
        self._reservation_event(
            state,
            "reservation_confirmed_from_customer_plan_set",
            request_id=request.request_id,
            commitment_id=commitment_id,
            details={
                "manager_confirmed_plan_id": str(plan_id),
                "customer_preferred_plan_id": request.customer_preferred_plan_id,
                "confirmation_origin": str(origin),
                "prepayment_collected": request.prepayment_due,
                "optional_lines_removed": len(removed_optional_lines),
            },
        )
        return {
            "request_id": request.request_id,
            "status": request.status,
            "commitment_id": commitment_id,
            "manager_confirmed_plan_id": str(plan_id),
            "customer_preferred_plan_id": request.customer_preferred_plan_id,
            "confirmation_origin": str(origin),
            "total_locked_rent": request.total_locked_rent,
            "prepayment_collected": request.prepayment_due,
            "total_security_deposit": round(
                sum(line.locked_security_deposit for line in commitment.lines), 2
            ),
            "lines": [
                {
                    "line_id": line.line_id,
                    "required": line.required,
                    "selected_offer_id": line.offer_id,
                    "promise_profile_id": line.promise_profile_id,
                    "profile_family_id": line.profile_family_id,
                    "service_units": line.service_units,
                    "capacity_per_asset": line.capacity_per_asset,
                    "asset_quantity": line.asset_quantity,
                    "locked_daily_price": line.locked_daily_price,
                    "locked_rent": line.locked_rent,
                    "locked_security_deposit": line.locked_security_deposit,
                }
                for line in commitment.lines
            ],
            "removed_optional_line_ids": removed_optional_lines,
        }

    def confirm_reservation_plans(
        self, state: WorldState, selections: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Atomically validate then confirm customer-authorized V20 plans."""

        if not selections:
            raise ValueError("selections must not be empty")
        request_ids = [str(row.get("request_id") or "") for row in selections]
        if any(not value for value in request_ids) or len(set(request_ids)) != len(
            request_ids
        ):
            raise ValueError("each selection needs one unique request_id")
        by_id = {row.request_id: row for row in state.reservation_requests}
        validated: list[tuple[ReservationRequest, str]] = []
        for selection in selections:
            if not isinstance(selection, dict) or set(selection) != {
                "request_id",
                "plan_id",
            }:
                raise ValueError(
                    "every selection requires only request_id and plan_id"
                )
            request_id = str(selection["request_id"])
            plan_id = str(selection["plan_id"])
            request = by_id.get(request_id)
            if request is None:
                raise ValueError(f"unknown request_id: {request_id}")
            if request.reservation_contract_mode != (
                "factorized_line_options_plan_set_v1"
            ) or request.status != "awaiting_manager_confirmation":
                raise ValueError(
                    f"request is not awaiting manager confirmation: {request_id}"
                )
            if plan_id not in {
                str(row["plan_id"]) for row in request.customer_authorized_plans
            }:
                raise ValueError(
                    f"plan_id is not customer-authorized for {request_id}: {plan_id}"
                )
            validated.append((request, plan_id))
        return {
            "confirmations": [
                self._confirm_factorized_plan(
                    state,
                    request,
                    plan_id=plan_id,
                    origin="manager_selected",
                )
                for request, plan_id in validated
            ]
        }

    def _auto_confirm_factorized_plans(self, state: WorldState) -> None:
        for request in sorted(
            state.reservation_requests, key=lambda row: row.request_id
        ):
            if request.status != "awaiting_manager_confirmation":
                continue
            if request.reservation_contract_mode != (
                "factorized_line_options_plan_set_v1"
            ):
                continue
            if request.customer_preferred_plan_id is None:
                raise AssertionError("awaiting plan set lacks a customer preferred plan")
            self._confirm_factorized_plan(
                state,
                request,
                plan_id=request.customer_preferred_plan_id,
                origin="day_end_customer_preferred_auto",
            )

    def _factorized_decline_feedback(
        self,
        state: WorldState,
        request: ReservationRequest,
        *,
        trace: Any,
        complete_by_id: dict[str, CompleteContractOption],
    ) -> None:
        """Generate versioned noisy feedback only after an empty plan set."""

        if request.customer_authorized_plans:
            raise AssertionError("an authorized plan set must not receive decline feedback")
        policy = self.plan_set_feedback_policy
        if policy is None:
            raise AssertionError("factorized plan feedback requires a frozen policy")
        best_internal_id = min(
            trace.plan_scores,
            key=lambda plan_id: (-float(trace.plan_scores[plan_id]), plan_id),
        )
        contract = complete_by_id[best_internal_id]
        line_by_id = {line.line_id: line for line in request.lines}

        if policy.factorized_concern_tags_enabled:
            choice_input = self._reservation_bundle_choice_input(
                state,
                request,
                {line_id: [] for line_id in line_by_id},
            )
            price_action_gain, price_curve_gains = (
                self._factorized_price_curve_action_gain(
                    state,
                    request,
                    choice_input=choice_input,
                    contract=contract,
                )
            )
            (
                equipment_action_gain,
                equipment_action_gain_by_line,
            ) = self._factorized_equipment_action_evidence_by_line(
                state,
                request,
                choice_input=choice_input,
                contract=contract,
            )
            event_address = tuple(map(int, request.event_address))
            response_uniform = float(
                self._stable_event_rng(
                    _RNG_RESERVATION_FEEDBACK_RESPONSE,
                    *event_address,
                    identity="factorized-plan-set-decline-response",
                ).random()
            )
            available = response_uniform < float(policy.response_probability)
            price_uniform = factorized_concern_uniform(
                seed=int(self.config.seed),
                request_id=request.request_id,
                stream="price-tag",
            )
            equipment_uniform = factorized_concern_uniform(
                seed=int(self.config.seed),
                request_id=request.request_id,
                stream="equipment-tag",
            )
            concerns = reported_factorized_concerns(
                policy,
                price_action_gain=price_action_gain,
                equipment_action_gain=equipment_action_gain,
                price_uniform=price_uniform,
                equipment_uniform=equipment_uniform,
            )
            feedback: dict[str, Any] = (
                {
                    "available": True,
                    "reported_concerns": list(concerns),
                }
                if available
                else {"available": False}
            )
            line_availability_uniform: float | None = None
            line_content_uniform: float | None = None
            mentioned_line_id: str | None = None
            if available and EQUIPMENT_CONFIGURATION_TAG in concerns:
                line_availability_uniform = line_mention_uniform(
                    seed=int(self.config.seed),
                    request_id=request.request_id,
                    stream="availability",
                )
                line_content_uniform = line_mention_uniform(
                    seed=int(self.config.seed),
                    request_id=request.request_id,
                    stream="content",
                )
                mentioned_line_id = reported_customer_mentioned_line_id(
                    policy,
                    equipment_fit_regret_by_line=(
                        equipment_action_gain_by_line
                    ),
                    availability_uniform=line_availability_uniform,
                    content_uniform=line_content_uniform,
                )
                if mentioned_line_id is not None:
                    feedback["customer_mentioned_line_id"] = mentioned_line_id
            request.reported_decline_feedback = feedback
            request.plan_set_feedback_audit.update(
                {
                    "best_rejected_internal_plan_id": best_internal_id,
                    "quoted_score": float(trace.plan_scores[best_internal_id]),
                    "price_curve_gains": price_curve_gains,
                    "price_action_gain": price_action_gain,
                    "equipment_action_gain": equipment_action_gain,
                    "equipment_action_gain_by_line": (
                        equipment_action_gain_by_line
                    ),
                    "response_uniform": response_uniform,
                    "response_available": available,
                    "price_tag_uniform": price_uniform,
                    "equipment_tag_uniform": equipment_uniform,
                    "price_tag_probability": (
                        factorized_concern_tag_probability(
                            policy,
                            tag=PRICE_VALUE_TAG,
                            action_gain=price_action_gain,
                        )
                    ),
                    "equipment_tag_probability": (
                        factorized_concern_tag_probability(
                            policy,
                            tag=EQUIPMENT_CONFIGURATION_TAG,
                            action_gain=equipment_action_gain,
                        )
                    ),
                    "reported_concerns": list(concerns),
                    "line_mention_availability_uniform": (
                        line_availability_uniform
                    ),
                    "line_mention_content_uniform": line_content_uniform,
                    "customer_mentioned_line_id": mentioned_line_id,
                }
            )
            return

        floor_options: list[BundleStoreOption] = []
        equipment_regret = 0.0
        equipment_regret_by_line: dict[str, float] = {}
        for option in contract.lines:
            line = line_by_id[option.line_id]
            floor_raw = self._enriched_pairwise_quote_options(
                request,
                line,
                quote_options=[
                    {
                        "option_id": f"feedback-floor:{option.line_id}",
                        "offer_id": option.offer_id,
                        "promise_profile_id": option.promise_profile_id,
                        "daily_unit_price": 0.01,
                    }
                ],
            )[0]
            floor_option = self._bundle_store_option(
                state, request, line, floor_raw
            )
            if floor_option is None:
                raise AssertionError("a quoted plan line lost its price-floor replay")
            floor_options.append(floor_option)

            best_fit = float(option.equipment_fit_value)
            asset_quantity = max(1, int(floor_raw["asset_quantity"]))
            current_daily_price = round(
                float(option.total_rent)
                / asset_quantity
                / max(1, int(line.duration_days)),
                2,
            )
            for profile_id in line.eligible_profile_ids:
                for offer_id in request.inquiry_profile_member_ids_by_line[
                    line.line_id
                ][profile_id]:
                    candidate_raw = self._enriched_pairwise_quote_options(
                        request,
                        line,
                        quote_options=[
                            {
                                "option_id": (
                                    f"feedback-fit:{line.line_id}:{profile_id}:"
                                    f"{offer_id}"
                                ),
                                "offer_id": offer_id,
                                "promise_profile_id": profile_id,
                                "daily_unit_price": max(0.01, current_daily_price),
                            }
                        ],
                    )[0]
                    candidate = self._bundle_store_option(
                        state, request, line, candidate_raw
                    )
                    if candidate is not None:
                        best_fit = max(
                            best_fit, float(candidate.equipment_fit_value)
                        )
            line_regret = max(
                0.0, best_fit - float(option.equipment_fit_value)
            )
            equipment_regret_by_line[option.line_id] = line_regret
            equipment_regret = max(equipment_regret, line_regret)
        for line_id in contract.omitted_optional_line_ids:
            equipment_regret_by_line[line_id] = 0.0

        choice_input = self._reservation_bundle_choice_input(
            state,
            request,
            {
                line_id: [] for line_id in line_by_id
            },
        )
        floor_contract = CompleteContractOption(
            contract_option_id=contract.contract_option_id,
            lines=tuple(floor_options),
            omitted_optional_line_ids=contract.omitted_optional_line_ids,
        )
        quoted_score = float(trace.plan_scores[best_internal_id])
        price_burden = max(
            0.0,
            self._complete_contract_score(choice_input, floor_contract)
            - quoted_score,
        )
        event_address = tuple(map(int, request.event_address))
        response_uniform = float(
            self._stable_event_rng(
                _RNG_RESERVATION_FEEDBACK_RESPONSE,
                *event_address,
                identity="factorized-plan-set-decline-response",
            ).random()
        )
        report_uniform = float(
            self._stable_event_rng(
                _RNG_RESERVATION_FEEDBACK_REPORT,
                *event_address,
                identity="factorized-plan-set-decline-report",
            ).random()
        )
        reported, latent = reported_plan_set_decline_category(
            policy,
            price_burden=price_burden,
            equipment_fit_regret=equipment_regret,
            report_uniform=report_uniform,
        )
        available = response_uniform < float(policy.response_probability)
        feedback = (
            {"available": True, "category": reported}
            if available
            else {"available": False}
        )
        line_mention_availability_uniform: float | None = None
        line_mention_content_uniform: float | None = None
        customer_mentioned_line_id: str | None = None
        if (
            available
            and reported == EQUIPMENT_FIT_CONCERN
            and policy.noisy_line_mentions_enabled
        ):
            line_mention_availability_uniform = line_mention_uniform(
                seed=int(self.config.seed),
                request_id=request.request_id,
                stream="availability",
            )
            line_mention_content_uniform = line_mention_uniform(
                seed=int(self.config.seed),
                request_id=request.request_id,
                stream="content",
            )
            customer_mentioned_line_id = (
                reported_customer_mentioned_line_id(
                    policy,
                    equipment_fit_regret_by_line=equipment_regret_by_line,
                    availability_uniform=line_mention_availability_uniform,
                    content_uniform=line_mention_content_uniform,
                )
            )
            if customer_mentioned_line_id is not None:
                feedback["customer_mentioned_line_id"] = (
                    customer_mentioned_line_id
                )
        request.reported_decline_feedback = feedback
        request.plan_set_feedback_audit.update(
            {
                "best_rejected_internal_plan_id": best_internal_id,
                "quoted_score": quoted_score,
                "price_floor_score": self._complete_contract_score(
                    choice_input, floor_contract
                ),
                "price_burden": price_burden,
                "equipment_fit_regret": equipment_regret,
                "equipment_fit_regret_by_line": equipment_regret_by_line,
                "response_uniform": response_uniform,
                "report_uniform": report_uniform,
                "response_available": available,
                "latent_concern": latent,
                "reported_category": reported,
                **(
                    {
                        "line_mention_availability_uniform": (
                            line_mention_availability_uniform
                        ),
                        "line_mention_content_uniform": (
                            line_mention_content_uniform
                        ),
                        "customer_mentioned_line_id": (
                            customer_mentioned_line_id
                        ),
                    }
                    if policy.noisy_line_mentions_enabled
                    else {}
                ),
            }
        )

    @staticmethod
    def _explicit_quote_prices(
        request: ReservationRequest, decision: dict[str, Any]
    ) -> dict[str, float]:
        action = str(decision.get("action", ""))
        if action == "decline":
            if "line_prices" in decision:
                raise ValueError("decline must not include line_prices")
            return {}
        if action != "quote":
            raise ValueError("action must be quote or decline")
        if "line_prices" not in decision:
            raise ValueError("quote requires complete line_prices")
        by_line = {line.line_id: line for line in request.lines}
        prices: dict[str, float] = {}
        raw_prices = decision.get("line_prices")
        if not isinstance(raw_prices, list):
            raise ValueError("line_prices must be an array")
        for row in raw_prices:
            if not isinstance(row, dict):
                raise ValueError("every line price must be an object")
            line_id = str(row.get("line_id", ""))
            value = row.get("daily_unit_price")
            if line_id not in by_line:
                raise ValueError(f"unknown inquiry line_id: {line_id}")
            if line_id in prices:
                raise ValueError(f"duplicate inquiry line_id: {line_id}")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    "daily_unit_price must be a positive finite number"
                )
            cents = float(value) * 100.0
            if not math.isclose(
                cents,
                round(cents),
                rel_tol=0.0,
                abs_tol=1e-7,
            ):
                raise ValueError(
                    "daily_unit_price must use at most two decimal places"
                )
            prices[line_id] = round(float(value), 2)
        missing = sorted(set(by_line) - set(prices))
        if missing:
            raise ValueError(
                "quote must price every inquiry line; missing line_ids: "
                + ", ".join(missing)
            )
        return prices

    @staticmethod
    def _explicit_profile_quotes(
        request: ReservationRequest, decision: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        action = str(decision.get("action", ""))
        if action == "decline":
            if "lines" in decision:
                raise ValueError("decline must not include lines")
            return {}
        if action != "quote":
            raise ValueError("action must be quote or decline")
        raw_lines = decision.get("lines")
        if not isinstance(raw_lines, list):
            raise ValueError("quote requires complete lines")
        by_line = {line.line_id: line for line in request.lines}
        result: dict[str, dict[str, Any]] = {}
        for row in raw_lines:
            if not isinstance(row, dict):
                raise ValueError("every quoted line must be an object")
            line_id = str(row.get("line_id", ""))
            profile_id = str(row.get("promise_profile_id", ""))
            value = row.get("daily_unit_price")
            if line_id not in by_line:
                raise ValueError(f"unknown inquiry line_id: {line_id}")
            if line_id in result:
                raise ValueError(f"duplicate inquiry line_id: {line_id}")
            if profile_id not in by_line[line_id].eligible_profile_ids:
                raise ValueError(
                    f"promise_profile_id is not eligible for {line_id}: {profile_id}"
                )
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    "daily_unit_price must be a positive finite number"
                )
            cents = float(value) * 100.0
            if not math.isclose(
                cents, round(cents), rel_tol=0.0, abs_tol=1e-7
            ):
                raise ValueError(
                    "daily_unit_price must use at most two decimal places"
                )
            result[line_id] = {
                "promise_profile_id": profile_id,
                "daily_unit_price": round(float(value), 2),
            }
        missing = sorted(set(by_line) - set(result))
        if missing:
            raise ValueError(
                "quote must price every inquiry line; missing line_ids: "
                + ", ".join(missing)
            )
        return result

    def quote_reservation_inquiries(
        self, state: WorldState, decisions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not decisions:
            raise ValueError("decisions must not be empty")
        v22_validation = bool(
            self._reservation_factorized_plan_mode()
            and int(self.dataset.metadata.get("tool_interface_version", 0)) >= 18
        )
        if v22_validation:
            by_id_for_validation = {
                row.request_id: row for row in state.reservation_requests
            }
            seen_request_ids: set[str] = set()
            invalid_items: list[dict[str, Any]] = []
            for batch_index, decision in enumerate(decisions):
                request_id = str(decision.get("request_id") or "")

                def request_error(
                    *,
                    field: str,
                    error_code: str,
                    error: str,
                    submitted_value: Any = None,
                    include_value: bool = False,
                ) -> None:
                    row: dict[str, Any] = {
                        "batch_index": batch_index,
                        "request_id": request_id,
                        "field": field,
                        "error_code": error_code,
                        "error": error,
                    }
                    if include_value:
                        row["submitted_value"] = submitted_value
                    invalid_items.append(row)

                if not request_id:
                    request_error(
                        field="request_id",
                        error_code="missing_request_id",
                        error="each decision needs one request_id",
                    )
                    continue
                duplicate = request_id in seen_request_ids
                if duplicate:
                    request_error(
                        field="request_id",
                        error_code="duplicate_request_id",
                        error=f"duplicate request_id in batch: {request_id}",
                        submitted_value=request_id,
                        include_value=True,
                    )
                seen_request_ids.add(request_id)
                request = by_id_for_validation.get(request_id)
                if request is None:
                    request_error(
                        field="request_id",
                        error_code="unknown_request_id",
                        error=f"unknown request_id: {request_id}",
                        submitted_value=request_id,
                        include_value=True,
                    )
                    continue
                if (
                    request.status != "pending_quote"
                    or request.created_day != state.day
                ):
                    request_error(
                        field="request_id",
                        error_code="inquiry_not_pending_today",
                        error=f"inquiry is not pending today: {request_id}",
                        submitted_value=request_id,
                        include_value=True,
                    )
                    continue
                for item in self._factorized_line_option_invalid_items(
                    request, decision
                ):
                    invalid_items.append(
                        {"batch_index": batch_index, **item}
                    )
            if invalid_items:
                raise ReservationQuoteBatchValidationError(invalid_items)
        identifiers = [str(row.get("request_id") or "") for row in decisions]
        if any(not value for value in identifiers) or len(set(identifiers)) != len(
            identifiers
        ):
            raise ValueError("each decision needs one unique request_id")
        by_id = {row.request_id: row for row in state.reservation_requests}
        validated: list[tuple[ReservationRequest, str, dict[str, Any]]] = []
        for decision in decisions:
            request_id = str(decision["request_id"])
            if request_id not in by_id:
                raise ValueError(f"unknown request_id: {request_id}")
            request = by_id[request_id]
            if (
                request.status != "pending_quote"
                or request.created_day != state.day
            ):
                raise ValueError(f"inquiry is not pending today: {request_id}")
            action = str(decision.get("action", ""))
            if action not in {"quote", "decline"}:
                raise ValueError("action must be quote or decline")
            if (
                request.reservation_contract_mode
                == "adaptive_pairwise_exact_quotes_v1"
            ):
                terms = self._explicit_pairwise_quotes(request, decision)
            elif request.reservation_contract_mode == "complete_contract_options_v1":
                terms = self._explicit_complete_contract_options(
                    request, decision
                )
            elif request.reservation_contract_mode == (
                "factorized_line_options_plan_set_v1"
            ):
                terms = self._explicit_factorized_line_options(request, decision)
            elif request.reservation_contract_mode == "promise_profiles_v1":
                terms = self._explicit_profile_quotes(request, decision)
            else:
                terms = {
                    line_id: {
                        "daily_unit_price": price,
                        "promise_profile_id": None,
                    }
                    for line_id, price in self._explicit_quote_prices(
                        request, decision
                    ).items()
                }
            validated.append((request, action, terms))

        results: list[dict[str, Any]] = []
        for request, action, terms in validated:
            request.decision_day = state.day
            request.quote_day = state.day
            request.quote_action = action
            if action == "decline":
                request.status = "declined_by_manager"
                state.cumulative_reservations_declined += 1
                state.cumulative_reservation_manager_quote_declines += 1
                self._reservation_event(
                    state, "inquiry_declined_by_manager", request_id=request.request_id
                )
                results.append(
                    {"request_id": request.request_id, "status": request.status}
                )
                continue

            state.cumulative_reservation_quotes += 1
            if request.reservation_contract_mode == (
                "factorized_line_options_plan_set_v1"
            ):
                plan_trace, complete_by_id = (
                    self._quoted_factorized_customer_plan_set(
                        state, request, terms
                    )
                )
                for line in request.lines:
                    line.status = "customer_authorized_pending_confirmation"
                if not request.customer_authorized_plans:
                    request.status = "customer_declined_quote"
                    state.cumulative_reservation_customer_quote_declines += 1
                    for line in request.lines:
                        line.status = "quote_not_accepted"
                    self._factorized_decline_feedback(
                        state,
                        request,
                        trace=plan_trace,
                        complete_by_id=complete_by_id,
                    )
                    self._reservation_event(
                        state,
                        "customer_declined_factorized_plan_set",
                        request_id=request.request_id,
                        details={
                            "submitted_line_count": len(
                                request.submitted_line_options
                            ),
                            "internal_plan_count": request.internal_plan_count,
                            "feedback_available": bool(
                                request.reported_decline_feedback.get("available")
                            ),
                        },
                    )
                    results.append(
                        {
                            "request_id": request.request_id,
                            "status": request.status,
                            "submitted_line_options": [
                                dict(row) for row in request.submitted_line_options
                            ],
                            "internal_plan_count": request.internal_plan_count,
                            "customer_authorized_plans": [],
                            "customer_preferred_plan_id": None,
                            "decline_feedback": dict(
                                request.reported_decline_feedback
                            ),
                        }
                    )
                    continue
                request.reported_decline_feedback = {}
                if len(request.customer_authorized_plans) == 1 and len(request.pickup_date_options) <= 1:
                    request.status = "customer_authorized_single_plan"
                    confirmation = self._confirm_factorized_plan(
                        state,
                        request,
                        plan_id=str(request.customer_preferred_plan_id),
                        origin="single_authorized_auto",
                    )
                    results.append(
                        {
                            **confirmation,
                            "submitted_line_options": [
                                dict(row) for row in request.submitted_line_options
                            ],
                            "internal_plan_count": request.internal_plan_count,
                            "customer_authorized_plans": [
                                dict(row)
                                for row in request.customer_authorized_plans
                            ],
                        }
                    )
                    continue
                request.status = "awaiting_manager_confirmation"
                self._reservation_event(
                    state,
                    "customer_authorized_factorized_plan_set",
                    request_id=request.request_id,
                    details={
                        "internal_plan_count": request.internal_plan_count,
                        "returned_authorized_plan_count": len(
                            request.customer_authorized_plans
                        ),
                        "customer_preferred_plan_id": (
                            request.customer_preferred_plan_id
                        ),
                        "confirmation_deadline": "end_of_current_business_day",
                    },
                )
                results.append(
                    {
                        "request_id": request.request_id,
                        "status": request.status,
                        "submitted_line_options": [
                            dict(row) for row in request.submitted_line_options
                        ],
                        "internal_plan_count": request.internal_plan_count,
                        "customer_authorized_plans": [
                            dict(row) for row in request.customer_authorized_plans
                        ],
                        "customer_preferred_plan_id": (
                            request.customer_preferred_plan_id
                        ),
                        "confirmation_deadline": "end_of_current_business_day",
                        "unanswered_confirmation_default": (
                            "customer_preferred_plan"
                        ),
                    }
                )
                continue
            outcomes: dict[str, dict[str, Any]] = {}
            customer_declined = False
            pairwise_mode = (
                request.reservation_contract_mode
                == "adaptive_pairwise_exact_quotes_v1"
            )
            complete_mode = (
                request.reservation_contract_mode
                == "complete_contract_options_v1"
            )
            complete_trace = None
            if complete_mode:
                (
                    outcomes,
                    customer_declined,
                    complete_trace,
                    _complete_by_id,
                ) = self._quoted_complete_contract_request_outcomes(
                    state, request, terms
                )
            elif pairwise_mode and self._reservation_request_bundle_choice_mode():
                outcomes, customer_declined = (
                    self._quoted_pairwise_request_outcomes(
                        state, request, terms
                    )
                )
            else:
                for line in request.lines:
                    term = terms[line.line_id]
                    if pairwise_mode:
                        outcomes[line.line_id] = (
                            self._quoted_pairwise_line_outcome(
                                state,
                                request,
                                line,
                                quote_options=list(term["quote_options"]),
                            )
                        )
                    elif (
                        request.reservation_contract_mode
                        == "promise_profiles_v1"
                    ):
                        price = float(term["daily_unit_price"])
                        outcomes[line.line_id] = (
                            self._quoted_profile_line_outcome(
                                request,
                                line,
                                profile_id=str(term["promise_profile_id"]),
                                daily_price=price,
                            )
                        )
                    else:
                        price = float(term["daily_unit_price"])
                        outcomes[line.line_id] = self._quoted_line_outcome(
                            state, request, line, price
                        )
                    if (
                        line.required
                        and not outcomes[line.line_id][
                            "authorized_offer_ids"
                        ]
                    ):
                        customer_declined = True
            quoted_total = (
                round(
                    min(
                        float(row["total_rent"])
                        for row in request.submitted_contract_options
                    ),
                    2,
                )
                if complete_mode
                else round(
                    sum(
                        (
                            min(
                                float(option["total_rent"])
                                for option in outcomes[line.line_id][
                                    "submitted_quote_options"
                                ]
                            )
                            if pairwise_mode
                            else (
                                rental_amount(
                                    float(terms[line.line_id]["daily_unit_price"]),
                                    line.duration_days,
                                    int(outcomes[line.line_id].get(
                                        "asset_quantity", line.quantity)))
                                if self._v45_reservation_difficulty()
                                else float(terms[line.line_id]["daily_unit_price"])
                                * int(outcomes[line.line_id].get(
                                    "asset_quantity", line.quantity))
                                * line.duration_days
                            )
                        )
                        for line in request.lines
                    ),
                    2,
                )
            )
            for line in request.lines:
                outcome = outcomes[line.line_id]
                if complete_mode:
                    line.submitted_quote_options = []
                    line.selected_quote_option_id = None
                    line.quoted_daily_price = None
                elif pairwise_mode:
                    line.submitted_quote_options = [
                        dict(option)
                        for option in outcome["submitted_quote_options"]
                    ]
                    line.selected_quote_option_id = None
                    line.quoted_daily_price = None
                else:
                    line.quoted_daily_price = float(
                        terms[line.line_id]["daily_unit_price"]
                    )
                if request.reservation_contract_mode == "promise_profiles_v1":
                    line.promise_profile_id = str(
                        outcome["promise_profile_id"]
                    )
                    line.profile_family_id = str(
                        outcome["profile_family_id"]
                    )
                    line.profile_schema_version = str(
                        outcome["profile_schema_version"]
                    )
                    line.capacity_per_asset = int(
                        outcome["capacity_per_asset"]
                    )
                    line.asset_quantity = int(outcome["asset_quantity"])
                    line.quantity = int(outcome["asset_quantity"])
            if customer_declined:
                request.status = "customer_declined_quote"
                state.cumulative_reservation_customer_quote_declines += 1
                for line in request.lines:
                    line.status = "quote_not_accepted"
                self._reservation_event(
                    state,
                    "customer_declined_quote",
                    request_id=request.request_id,
                    details=(
                        {
                            "submitted_contract_option_count": len(
                                request.submitted_contract_options
                            ),
                            "feedback_response_available": bool(
                                request.feedback_response_available
                            ),
                        }
                        if complete_mode
                        else
                        {
                            "submitted_line_count": len(request.lines),
                            "submitted_option_count": sum(
                                len(line.submitted_quote_options)
                                for line in request.lines
                            ),
                        }
                        if pairwise_mode
                        else {"quoted_total_rent": quoted_total}
                    ),
                )
                result = {
                    "request_id": request.request_id,
                    "status": request.status,
                }
                if complete_mode:
                    result.update(
                        {
                            "submitted_contract_options": [
                                dict(row)
                                for row in request.submitted_contract_options
                            ],
                            "contract_assessments": [
                                dict(row)
                                for row in request.reported_contract_assessments
                            ],
                        }
                    )
                elif pairwise_mode:
                    result.update(
                        {
                            "submitted_line_count": len(request.lines),
                            "submitted_option_count": sum(
                                len(line.submitted_quote_options)
                                for line in request.lines
                            ),
                        }
                    )
                else:
                    result["quoted_total_rent"] = quoted_total
                results.append(result)
                continue

            commitment_lines: list[ReservationLine] = []
            fallback_by_line: dict[str, list[str]] = {}
            authorized_by_line: dict[str, list[str]] = {}
            authorized_ceilings_by_line: dict[str, dict[str, float]] = {}
            recovery_by_line: dict[str, list[str]] = {}
            recovery_ceilings_by_line: dict[str, dict[str, float]] = {}
            removed_optional_lines: list[str] = []
            for line in request.lines:
                outcome = outcomes[line.line_id]
                authorized = list(outcome["authorized_offer_ids"])
                fallback_by_line[line.line_id] = []
                authorized_by_line[line.line_id] = []
                authorized_ceilings_by_line[line.line_id] = {}
                recovery_by_line[line.line_id] = []
                recovery_ceilings_by_line[line.line_id] = {}
                if not authorized:
                    line.status = "optional_removed_at_quote"
                    line.locked_daily_price = 0.0
                    line.locked_rent = 0.0
                    removed_optional_lines.append(line.line_id)
                    continue
                primary = str(outcome["primary_offer_id"])
                line.offer_id = primary
                line.service_line = self.offers[primary].service_line
                if pairwise_mode or complete_mode:
                    line.selected_quote_option_id = str(
                        outcome["selected_quote_option_id"]
                    )
                    line.selected_offer_id = primary
                    line.quoted_daily_price = float(outcome["daily_unit_price"])
                    line.promise_profile_id = str(
                        outcome["promise_profile_id"]
                    )
                    line.profile_family_id = str(outcome["profile_family_id"])
                    line.profile_schema_version = str(
                        outcome["profile_schema_version"]
                    )
                    line.capacity_per_asset = int(
                        outcome["capacity_per_asset"]
                    )
                    line.asset_quantity = int(outcome["asset_quantity"])
                    line.quantity = int(outcome["asset_quantity"])
                    line.locked_daily_price = float(outcome["daily_unit_price"])
                else:
                    line.locked_daily_price = float(
                        terms[line.line_id]["daily_unit_price"]
                    )
                line.locked_rent = round(
                    (rental_amount(
                        request.service_offer_terms["line_daily_rents"][line.line_id],
                        line.duration_days)
                     if self._v45_reservation_difficulty() and request.service_offer_terms
                     else rental_amount(line.locked_daily_price, line.duration_days, line.quantity)
                     if self._v45_reservation_difficulty()
                     else line.locked_daily_price * line.quantity * line.duration_days),
                    2,
                )
                line.authorized_offer_ids = authorized
                authorized_by_line[line.line_id] = list(authorized)
                authorized_ceilings_by_line[line.line_id] = dict(
                    outcome["authorized_ceilings"]
                )
                line.locked_security_deposit = float(
                    outcome["locked_security_deposit"]
                )
                fallback_by_line[line.line_id] = list(
                    outcome["fallback_offer_ids"]
                )
                recovery_by_line[line.line_id] = list(
                    fallback_by_line[line.line_id]
                )
                recovery_ceilings_by_line[line.line_id] = dict(
                    outcome["recovery_ceilings"]
                )
                if fallback_by_line[line.line_id]:
                    line.substitution_hint = (
                        "锁定 exact 无法履约时，才可通过 inquire_fallback "
                        "询问同价同押金且不降低 Profile 保证的客户备选。"
                            if pairwise_mode or complete_mode
                        else (
                            "仅当完整冻结授权集合无法履约时，才可通过 "
                            "inquire_fallback 询问同价同押金的客户授权备选。"
                            if request.reservation_contract_mode
                            == "promise_profiles_v1"
                            else self._reservation_substitution_hint(
                                str(self.offers[primary].choice_group)
                            )
                        )
                    )
                else:
                    line.substitution_hint = None
                line.status = "committed"
                commitment_lines.append(replace(line))

            request.total_locked_rent = round(
                sum(line.locked_rent for line in commitment_lines), 2
            )
            request.prepayment_due = round(
                request.total_locked_rent * request.prepayment_rate, 2
            )
            request.return_day = request.pickup_day + max(
                line.duration_days for line in commitment_lines
            )
            request.fallback_offer_ids_by_line = fallback_by_line
            request.fallback_offer_ids = sorted(
                {
                    offer_id
                    for values in fallback_by_line.values()
                    for offer_id in values
                }
            )
            request.authorized_offer_ids_by_line = authorized_by_line
            request.authorized_price_ceiling_by_line = (
                authorized_ceilings_by_line
            )
            request.recovery_offer_ids_by_line = recovery_by_line
            request.recovery_price_ceiling_by_line = recovery_ceilings_by_line
            request.status = "accepted"
            if removed_optional_lines:
                state.cumulative_reservation_optional_lines_removed_at_quote += len(
                    removed_optional_lines
                )

            commitment_id = f"reservation-{len(state.reservation_commitments) + 1:03d}"
            commitment = ReservationCommitment(
                commitment_id=commitment_id,
                request_id=request.request_id,
                confirmed_day=state.day,
                pickup_day=request.pickup_day,
                return_day=request.return_day,
                lines=commitment_lines,
                total_locked_rent=request.total_locked_rent,
                prepayment_collected=request.prepayment_due,
                outstanding_rent=round(
                    request.total_locked_rent - request.prepayment_due, 2
                ),
                unearned_revenue=request.prepayment_due,
                original_pickup_day=(
                    request.original_inquiry_pickup_day
                    if request.original_inquiry_pickup_day is not None
                    else request.pickup_day
                ),
                original_return_day=(
                    request.original_inquiry_return_day
                    if request.original_inquiry_return_day is not None
                    else request.return_day
                ),
                promised_return_day=request.return_day,
                approved_return_day=request.return_day,
                estimated_return_day=request.return_day,
                **(
                    {
                        "rent_due_at_pickup": round(
                            request.total_locked_rent
                            - request.prepayment_due,
                            2,
                        ),
                        "reschedule_forecast_baseline_condition": (
                            request.forecast_condition
                        ),
                    }
                    if self._v26_contract_state()
                    else {}
                ),
            )
            request.commitment_id = commitment_id
            self._initialize_v42_commitment_customer_state(
                state, request, commitment
            )
            state.reservation_commitments.append(commitment)
            for line in commitment.lines:
                for day in range(commitment.pickup_day, commitment.return_day):
                    state.reservation_calendar_blocks.append(
                        CalendarBlock(
                            block_id=f"{commitment_id}-{line.line_id}-{day:03d}",
                            commitment_id=commitment_id,
                            line_id=line.line_id,
                            offer_id=line.offer_id,
                            day=day,
                            quantity=line.quantity,
                            authorized_offer_ids=list(line.authorized_offer_ids),
                            selected_offer_id=line.selected_offer_id,
                            promise_profile_id=line.promise_profile_id,
                            profile_schema_version=(
                                line.profile_schema_version
                            ),
                            service_units=line.service_units,
                            capacity_per_asset=line.capacity_per_asset,
                            staged_offer_id=line.staged_offer_id,
                        )
                    )
            state.cash = round(state.cash + request.prepayment_due, 2)
            state.unearned_revenue_liability = round(
                state.unearned_revenue_liability + request.prepayment_due, 2
            )
            state.cumulative_reservation_prepayments = round(
                state.cumulative_reservation_prepayments + request.prepayment_due,
                2,
            )
            state.current_day_ledger["reservation_prepayments_collected"] = round(
                float(
                    state.current_day_ledger.get(
                        "reservation_prepayments_collected", 0.0
                    )
                )
                + request.prepayment_due,
                2,
            )
            self._assert_v26_reservation_financial_invariants(state)
            state.cumulative_reservations_confirmed += 1
            self._reservation_event(
                state,
                "reservation_confirmed_after_quote",
                request_id=request.request_id,
                commitment_id=commitment_id,
                details={
                    "prepayment_collected": request.prepayment_due,
                    "optional_lines_removed": len(removed_optional_lines),
                    **(
                        {
                            "submitted_option_count": sum(
                                len(line.submitted_quote_options)
                                for line in request.lines
                            ),
                            "selected_options": {
                                line.line_id: line.selected_quote_option_id
                                for line in commitment.lines
                            },
                        }
                        if pairwise_mode
                        else {}
                        | (
                            {
                                "submitted_contract_option_count": len(
                                    request.submitted_contract_options
                                ),
                                "selected_contract_option_id": (
                                    request.selected_contract_option_id
                                ),
                                "feedback_response_available": bool(
                                    request.feedback_response_available
                                ),
                            }
                            if complete_mode
                            else {}
                        )
                    ),
                },
            )
            if complete_mode:
                result_lines = [
                    {
                        "line_id": line.line_id,
                        "required": line.required,
                        "selected_offer_id": line.offer_id,
                        "promise_profile_id": line.promise_profile_id,
                        "profile_family_id": line.profile_family_id,
                        "service_units": line.service_units,
                        "capacity_per_asset": line.capacity_per_asset,
                        "asset_quantity": line.asset_quantity,
                        "locked_daily_price": line.locked_daily_price,
                        "locked_rent": line.locked_rent,
                        "locked_security_deposit": (
                            line.locked_security_deposit
                        ),
                    }
                    for line in commitment.lines
                ]
            elif pairwise_mode:
                result_lines = [
                    {
                        "line_id": line.line_id,
                        "required": line.required,
                        "selected_quote_option_id": (
                            line.selected_quote_option_id
                        ),
                        "selected_offer_id": line.offer_id,
                        "promise_profile_id": line.promise_profile_id,
                        "profile_family_id": line.profile_family_id,
                        "service_units": line.service_units,
                        "capacity_per_asset": line.capacity_per_asset,
                        "asset_quantity": line.asset_quantity,
                        "locked_daily_price": line.locked_daily_price,
                        "locked_rent": line.locked_rent,
                        "locked_security_deposit": (
                            line.locked_security_deposit
                        ),
                        "submitted_option_count": len(
                            line.submitted_quote_options
                        ),
                    }
                    for line in commitment.lines
                ]
            elif request.reservation_contract_mode == "promise_profiles_v1":
                result_lines = [
                    {
                        "line_id": line.line_id,
                        "required": line.required,
                        "promise_profile_id": line.promise_profile_id,
                        "profile_family_id": line.profile_family_id,
                        "service_units": line.service_units,
                        "capacity_per_asset": line.capacity_per_asset,
                        "asset_quantity": line.asset_quantity,
                        "locked_daily_price": line.locked_daily_price,
                        "locked_rent": line.locked_rent,
                        "locked_security_deposit": (
                            line.locked_security_deposit
                        ),
                        "authorized_offer_count": len(
                            line.authorized_offer_ids
                        ),
                        "authorized_offer_preview": list(
                            line.authorized_offer_ids[:3]
                        ),
                        "authorized_offers_have_more": (
                            len(line.authorized_offer_ids) > 3
                        ),
                    }
                    for line in commitment.lines
                ]
            else:
                result_lines = [asdict(line) for line in commitment.lines]
            result = {
                    "request_id": request.request_id,
                    "commitment_id": commitment_id,
                    "status": "accepted",
                    "lines": result_lines,
                    "total_locked_rent": request.total_locked_rent,
                    "prepayment_collected": request.prepayment_due,
                    "total_locked_security_deposit": round(
                        sum(
                            line.locked_security_deposit
                            for line in commitment.lines
                        ),
                        2,
                    ),
                    "optional_lines_removed": removed_optional_lines,
                    "unearned_revenue_liability": state.unearned_revenue_liability,
                }
            if complete_mode:
                result.update(
                    {
                        "selected_contract_option_id": (
                            request.selected_contract_option_id
                        ),
                        "submitted_contract_options": [
                            dict(row) for row in request.submitted_contract_options
                        ],
                        "contract_assessments": [
                            dict(row)
                            for row in request.reported_contract_assessments
                        ],
                    }
                )
            results.append(result)
        return {"decisions": results, "cash": state.cash}

    def stage_reservation_line(
        self,
        state: WorldState,
        *,
        commitment_id: str,
        line_id: str,
        authorized_offer_id: str,
    ) -> dict[str, Any]:
        if self.config.reservation_contract_mode != "promise_profiles_v1":
            raise ValueError("reservation staging requires promise_profiles_v1")
        commitment = next(
            (
                row
                for row in state.reservation_commitments
                if row.commitment_id == commitment_id
            ),
            None,
        )
        if commitment is None:
            raise ValueError(f"unknown commitment_id: {commitment_id}")
        if commitment.status != "confirmed":
            raise ValueError("only a confirmed reservation can be staged")
        if state.day >= commitment.pickup_day:
            raise ValueError("staging must occur before the pickup business day")
        line = next(
            (row for row in commitment.lines if row.line_id == line_id), None
        )
        if line is None:
            raise ValueError(f"unknown commitment line_id: {line_id}")
        if authorized_offer_id not in line.authorized_offer_ids:
            raise ValueError("authorized_offer_id is not in the frozen authorization set")
        previous = line.staged_offer_id
        line.staged_offer_id = authorized_offer_id
        for block in state.reservation_calendar_blocks:
            if (
                block.commitment_id == commitment_id
                and block.line_id == line_id
                and block.status == "active"
            ):
                block.staged_offer_id = authorized_offer_id
        state.cumulative_reservation_staging_actions += 1
        self._reservation_event(
            state,
            "reservation_line_staged",
            request_id=commitment.request_id,
            commitment_id=commitment_id,
            details={
                "line_id": line_id,
                "previous_staged_offer_id": previous,
                "staged_offer_id": authorized_offer_id,
            },
        )
        return {
            "commitment_id": commitment_id,
            "line_id": line_id,
            "previous_staged_offer_id": previous,
            "staged_offer_id": authorized_offer_id,
            "authorized_offer_count": len(line.authorized_offer_ids),
            "asset_binding": "not_started",
        }

    def respond_reservation_changes(
        self, state: WorldState, decisions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not decisions:
            raise ValueError("decisions must not be empty")
        identifiers = [str(row.get("change_id") or "") for row in decisions]
        if any(not value for value in identifiers) or len(set(identifiers)) != len(
            identifiers
        ):
            raise ValueError("each decision needs one unique change_id")
        by_id = {
            row.change_id: row for row in state.reservation_change_requests
        }
        validated = []
        for decision in decisions:
            change_id = str(decision["change_id"])
            action = str(decision.get("action", ""))
            if action not in {"accept", "decline"}:
                raise ValueError("action must be accept or decline")
            if change_id not in by_id:
                raise ValueError(f"unknown change_id: {change_id}")
            change = by_id[change_id]
            if change.status != "pending" or change.created_day != state.day:
                raise ValueError(f"change is not pending today: {change_id}")
            if action == "accept" and change.change_type == "extension":
                active_contracts = [
                    contract
                    for contract in state.contracts
                    if contract.reservation_id == change.commitment_id
                    and contract.status == "active"
                ]
                assets = {asset.asset_id: asset for asset in state.assets}
                incompatible = [
                    contract.asset_id
                    for contract in active_contracts
                    if not self._supplier_trial_extension_allowed(
                        state,
                        assets[contract.asset_id],
                        proposed_return_day=change.proposed_return_day,
                    )
                ]
                if incompatible:
                    raise ValueError(
                        "extension exceeds a supplier-trial return deadline for "
                        + ", ".join(incompatible)
                    )
            validated.append((change, action))
        results = []
        for change, action in validated:
            commitment = next(
                row
                for row in state.reservation_commitments
                if row.commitment_id == change.commitment_id
            )
            accepted_delta: dict[str, Any] = {}
            declined_delta: dict[str, Any] = {}
            if action == "accept":
                accepted_delta = self._apply_reservation_change(state, change)
                late = False
            else:
                change.status = "declined_explicitly"
                change.decision_day = state.day
                late = (
                    self._schedule_late_return(
                        state,
                        commitment,
                        identity=(
                            "declined:"
                            + self._reservation_change_rng_identity(
                                commitment, change
                            )
                        ),
                    )
                    if change.change_type == "extension"
                    else False
                )
                self._reservation_event(
                    state,
                    f"{change.change_type}_declined",
                    request_id=commitment.request_id,
                    commitment_id=commitment.commitment_id,
                    details={
                        "change_id": change.change_id,
                        "late_return": late,
                        **self._reservation_change_effects(change),
                        **({"additional_locked_rent": 0.0, "additional_rent_collected": 0.0}
                           if change.previous_pickup_day is not None else {}),
                    },
                )
                if self._v32_service_commitments():
                    declined_delta = self._resolve_v32_unavailable_dates(state, change, expired=False)
            results.append(
                {
                    "change_id": change.change_id,
                    "commitment_id": change.commitment_id,
                    "status": change.status,
                    "late_return_estimated": late,
                    **declined_delta,
                    **self._reservation_change_effects(change),
                    **({"additional_locked_rent": 0.0, "additional_rent_collected": 0.0}
                       if change.previous_pickup_day is not None else {}),
                    **(
                        accepted_delta
                        if (change.previous_pickup_day is not None or (
                            self._v27_extension_cash_audit()
                            and change.change_type == "extension"
                        ))
                        and action == "accept"
                        else {}
                    ),
                }
            )
        return {"decisions": results, "cash": state.cash}

    def view_reservation(
        self,
        state: WorldState,
        identifiers: list[str],
        *,
        view: str = "index",
        cursor: int = 0,
        limit: int = 20,
        from_day: int | None = None,
        to_day: int | None = None,
        offer_ids: list[str] | None = None,
        service_keys: list[str] | None = None,
        quote_outcomes: list[str] | None = None,
        menu_sizes: list[int] | None = None,
        contract_option_counts: list[int] | None = None,
        assessment_categories: list[str] | None = None,
        authorized_plan_counts: list[int] | None = None,
        decline_feedback_categories: list[str] | None = None,
        decline_feedback_tags: list[str] | None = None,
        decline_feedback_concern_sets: list[str] | None = None,
        sections: list[str] | None = None,
        sort_order: str | None = None,
        searchable_quote_history: bool = False,
    ) -> dict[str, Any]:
        v22_surface = bool(
            self._reservation_factorized_plan_mode()
            and int(self.dataset.metadata.get("tool_interface_version", 0)) >= 18
        )
        v23_surface = bool(
            self._reservation_factorized_plan_mode()
            and int(self.dataset.metadata.get("tool_interface_version", 0)) >= 19
        )
        v26_surface = self._v26_contract_state()
        v27_surface = self._v27_extension_cash_audit()
        v29_surface = self._v29_reservation_accountability()
        maximum_limit = 10 if v22_surface else 50
        if cursor < 0 or not 1 <= limit <= maximum_limit:
            raise ValueError(
                "cursor must be non-negative and limit must be in "
                f"[1, {maximum_limit}]"
            )
        valid_sections = {
            "current_terms",
            "submitted_quote",
            "customer_authorization",
            "fulfillment",
            "changes",
            "risk",
        }
        if v26_surface:
            valid_sections.add("inquiry_terms")
        if v29_surface:
            valid_sections.add("cancellation_terms")
        normalized_sections = tuple(
            sections or ("current_terms", "fulfillment")
        )
        if sections is not None and (
            not v22_surface
            or view != "detail"
            or len(sections) != len(set(sections))
            or not set(sections).issubset(valid_sections)
        ):
            raise ValueError(
                "sections are unique detail values from "
                + ("inquiry_terms, " if v26_surface else "")
                + "current_terms, submitted_quote, customer_authorization, "
                "fulfillment, changes, risk"
                + (", or cancellation_terms" if v29_surface else "")
            )
        if sections is None and view != "detail":
            normalized_sections = ()
        history_controls_supplied = any(
            value is not None
            for value in (
                from_day,
                to_day,
                offer_ids,
                service_keys,
                quote_outcomes,
                menu_sizes,
                contract_option_counts,
                assessment_categories,
                authorized_plan_counts,
                decline_feedback_categories,
                decline_feedback_tags,
                decline_feedback_concern_sets,
                sort_order,
            )
        )
        if view != "quote_history" and history_controls_supplied:
            raise ValueError(
                "quote-history filters and sort_order are valid only for the "
                "quote_history view"
            )
        if view == "index":
            if identifiers:
                raise ValueError("identifiers are not valid for the index view")
            commitments_by_request = {
                row.request_id: row for row in state.reservation_commitments
            }
            rows = []
            for request in sorted(
                state.reservation_requests,
                key=lambda row: (-row.created_day, row.request_id),
            ):
                commitment = commitments_by_request.get(request.request_id)
                if v26_surface:
                    original_pickup_day = int(
                        commitment.original_pickup_day
                        if commitment is not None
                        and commitment.original_pickup_day is not None
                        else request.original_inquiry_pickup_day
                        if request.original_inquiry_pickup_day is not None
                        else request.pickup_day
                    )
                    original_return_day = int(
                        commitment.original_return_day
                        if commitment is not None
                        and commitment.original_return_day is not None
                        else request.original_inquiry_return_day
                        if request.original_inquiry_return_day is not None
                        else request.return_day
                    )
                    current_pickup_day = int(
                        request.pickup_day
                        if commitment is None
                        else commitment.pickup_day
                    )
                    current_return_day = int(
                        request.return_day
                        if commitment is None
                        else commitment.return_day
                    )
                    current_total = (
                        None
                        if commitment is None
                        else round(
                            sum(
                                float(line.locked_rent)
                                for line in commitment.lines
                            ),
                            2,
                        )
                    )
                    if commitment is not None and not math.isclose(
                        float(commitment.total_locked_rent),
                        float(current_total),
                        abs_tol=0.0049,
                    ):
                        raise AssertionError(
                            "commitment total_locked_rent does not equal the "
                            f"locked-line sum: {commitment.commitment_id}"
                        )
                row = {
                        "request_id": request.request_id,
                        "commitment_id": request.commitment_id,
                        "created_day": request.created_day,
                        "decision_day": request.decision_day,
                        **(
                            {
                                "original_pickup_day": original_pickup_day,
                                "original_return_day": original_return_day,
                                "current_pickup_day": current_pickup_day,
                                "current_return_day": current_return_day,
                            }
                            if v26_surface
                            else {
                                "pickup_day": request.pickup_day,
                                "return_day": request.return_day,
                            }
                        ),
                        "request_status": request.status,
                        "commitment_status": (
                            None if commitment is None else commitment.status
                        ),
                        "line_count": len(request.lines),
                        "required_line_count": sum(
                            line.required for line in request.lines
                        ),
                        "quote_action": request.quote_action,
                        "total_locked_rent": (
                            current_total
                            if v26_surface
                            else request.total_locked_rent
                        ),
                        "risk_exposed_day": (
                            None if commitment is None else commitment.risk_exposed_day
                        ),
                    }
                rows.append(row)
            page = rows[cursor : cursor + limit]
            next_cursor = (
                cursor + len(page)
                if cursor + len(page) < len(rows)
                else None
            )
            return {
                "view": "index",
                "reservations": page,
                "total_count": len(rows),
                "returned_count": len(page),
                "cursor": cursor,
                "limit": limit,
                "has_more": next_cursor is not None,
                "next_cursor": next_cursor,
            }

        if view == "quote_history":
            if not searchable_quote_history and identifiers:
                raise ValueError(
                    "identifiers are not valid for the quote_history view"
                )
            if not searchable_quote_history and history_controls_supplied:
                raise ValueError(
                    "quote-history filters and sort_order require tool interface 13"
                )
            if searchable_quote_history and (
                len(identifiers) > 3 or len(set(identifiers)) != len(identifiers)
            ):
                raise ValueError(
                    "quote_history accepts at most 3 unique identifiers"
                )

            normalized_offer_ids = sorted(set(offer_ids or []))
            normalized_service_keys = sorted(set(service_keys or []))
            normalized_quote_outcomes = sorted(set(quote_outcomes or []))
            normalized_menu_sizes = sorted(set(menu_sizes or []))
            normalized_contract_option_counts = sorted(
                set(contract_option_counts or [])
            )
            normalized_assessment_categories = sorted(
                set(assessment_categories or [])
            )
            normalized_authorized_plan_counts = sorted(
                set(authorized_plan_counts or [])
            )
            normalized_decline_feedback_categories = sorted(
                set(decline_feedback_categories or [])
            )
            normalized_decline_feedback_tags = sorted(
                set(decline_feedback_tags or [])
            )
            normalized_decline_feedback_concern_sets = sorted(
                set(decline_feedback_concern_sets or [])
            )
            if offer_ids is not None and (
                not offer_ids or len(offer_ids) > 20 or len(set(offer_ids)) != len(offer_ids)
            ):
                raise ValueError("offer_ids must contain 1 to 20 unique values")
            if service_keys is not None and (
                not service_keys
                or len(service_keys) > 20
                or len(set(service_keys)) != len(service_keys)
            ):
                raise ValueError("service_keys must contain 1 to 20 unique values")
            maximum_quote_outcomes = (
                8 if self._v32_service_commitments() else 5 if self._reservation_factorized_plan_mode() else 4
            )
            if quote_outcomes is not None and (
                not quote_outcomes
                or len(quote_outcomes) > maximum_quote_outcomes
                or len(set(quote_outcomes)) != len(quote_outcomes)
            ):
                raise ValueError(
                    "quote_outcomes contains an invalid number of unique values"
                )
            valid_menu_sizes = (
                {0, 1, 2}
                if self._reservation_factorized_plan_mode()
                else {1, 2}
            )
            if menu_sizes is not None and (
                not menu_sizes
                or len(menu_sizes) > len(valid_menu_sizes)
                or len(set(menu_sizes)) != len(menu_sizes)
            ):
                raise ValueError("menu_sizes contains invalid unique values")
            if contract_option_counts is not None and (
                not contract_option_counts
                or len(contract_option_counts) > 2
                or len(set(contract_option_counts))
                != len(contract_option_counts)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value not in {1, 2}
                    for value in contract_option_counts
                )
            ):
                raise ValueError(
                    "contract_option_counts may contain only unique values 1 and 2"
                )
            if assessment_categories is not None and (
                not assessment_categories
                or len(assessment_categories) > len(PUBLIC_ASSESSMENT_CATEGORIES)
                or len(set(assessment_categories)) != len(assessment_categories)
            ):
                raise ValueError(
                    "assessment_categories must contain unique public categories"
                )
            unknown_assessment_categories = sorted(
                set(normalized_assessment_categories)
                - set(PUBLIC_ASSESSMENT_CATEGORIES)
            )
            if unknown_assessment_categories:
                raise ValueError(
                    "unknown assessment_categories: "
                    f"{unknown_assessment_categories}"
                )
            if authorized_plan_counts is not None and (
                not authorized_plan_counts
                or len(authorized_plan_counts) > 4
                or len(set(authorized_plan_counts))
                != len(authorized_plan_counts)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value not in {0, 1, 2, 3}
                    for value in authorized_plan_counts
                )
            ):
                raise ValueError(
                    "authorized_plan_counts may contain only unique values 0 to 3"
                )
            if v23_surface and decline_feedback_categories is not None:
                raise ValueError(
                    "V23 quote history uses decline_feedback_tags or "
                    "decline_feedback_concern_sets"
                )
            if not v23_surface and (
                decline_feedback_tags is not None
                or decline_feedback_concern_sets is not None
            ):
                raise ValueError(
                    "V23 decline-feedback filters require tool interface 19"
                )
            if decline_feedback_categories is not None and (
                not decline_feedback_categories
                or len(decline_feedback_categories)
                > len(PUBLIC_PLAN_SET_DECLINE_CATEGORIES)
                or len(set(decline_feedback_categories))
                != len(decline_feedback_categories)
            ):
                raise ValueError(
                    "decline_feedback_categories must contain unique public categories"
                )
            unknown_decline_categories = sorted(
                set(normalized_decline_feedback_categories)
                - set(PUBLIC_PLAN_SET_DECLINE_CATEGORIES)
            )
            if unknown_decline_categories:
                raise ValueError(
                    "unknown decline_feedback_categories: "
                    f"{unknown_decline_categories}"
                )
            valid_decline_tags = {
                PRICE_VALUE_TAG,
                EQUIPMENT_CONFIGURATION_TAG,
                OTHER_OR_UNCLEAR_TAG,
            }
            if decline_feedback_tags is not None and (
                not decline_feedback_tags
                or len(decline_feedback_tags) > len(valid_decline_tags)
                or len(set(decline_feedback_tags)) != len(decline_feedback_tags)
            ):
                raise ValueError(
                    "decline_feedback_tags must contain unique public V23 tags"
                )
            unknown_decline_tags = sorted(
                set(normalized_decline_feedback_tags) - valid_decline_tags
            )
            if unknown_decline_tags:
                raise ValueError(
                    f"unknown decline_feedback_tags: {unknown_decline_tags}"
                )
            valid_concern_sets = {
                PRICE_VALUE_TAG,
                EQUIPMENT_CONFIGURATION_TAG,
                f"{PRICE_VALUE_TAG}+{EQUIPMENT_CONFIGURATION_TAG}",
                OTHER_OR_UNCLEAR_TAG,
            }
            if decline_feedback_concern_sets is not None and (
                not decline_feedback_concern_sets
                or len(decline_feedback_concern_sets) > len(valid_concern_sets)
                or len(set(decline_feedback_concern_sets))
                != len(decline_feedback_concern_sets)
            ):
                raise ValueError(
                    "decline_feedback_concern_sets must contain unique public "
                    "V23 concern sets"
                )
            unknown_concern_sets = sorted(
                set(normalized_decline_feedback_concern_sets)
                - valid_concern_sets
            )
            if unknown_concern_sets:
                raise ValueError(
                    "unknown decline_feedback_concern_sets: "
                    f"{unknown_concern_sets}"
                )

            unknown_offer_ids = sorted(
                set(normalized_offer_ids) - set(self.offers)
            )
            if unknown_offer_ids:
                raise ValueError(f"unknown offer_ids: {unknown_offer_ids}")
            non_rental_offer_ids = sorted(
                offer_id
                for offer_id in normalized_offer_ids
                if self.offers[offer_id].asset_kind != "durable"
            )
            if non_rental_offer_ids:
                raise ValueError(
                    "quote history accepts rental offer_ids only; "
                    f"non-rental offer_ids: {non_rental_offer_ids}"
                )
            all_service_keys = {
                str(row["service_key"]) for row in self._service_registry
            }
            rental_service_keys = {
                str(row["service_key"])
                for row in self._service_registry
                if row["kind"] == "rental"
            }
            unknown_service_keys = sorted(
                set(normalized_service_keys) - all_service_keys
            )
            if unknown_service_keys:
                raise ValueError(
                    f"unknown service_keys: {unknown_service_keys}"
                )
            non_rental_service_keys = sorted(
                set(normalized_service_keys) - rental_service_keys
            )
            if non_rental_service_keys:
                raise ValueError(
                    "quote history accepts rental service_keys only; "
                    f"non-rental service_keys: {non_rental_service_keys}"
                )
            valid_quote_outcomes = {
                "accepted",
                "customer_declined_quote",
                "declined_by_manager",
                "expired_unquoted",
            }
            if self._reservation_factorized_plan_mode():
                valid_quote_outcomes.add("awaiting_manager_confirmation")
            if self._v32_service_commitments():
                valid_quote_outcomes.update({"service_offer_accepted", "service_offer_declined", "service_offer_expired"})
            invalid_quote_outcomes = sorted(
                set(normalized_quote_outcomes) - valid_quote_outcomes
            )
            if invalid_quote_outcomes:
                raise ValueError(
                    f"unknown quote_outcomes: {invalid_quote_outcomes}"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value not in valid_menu_sizes
                for value in normalized_menu_sizes
            ):
                raise ValueError(
                    f"menu_sizes may contain only {sorted(valid_menu_sizes)}"
                )
            effective_sort_order = sort_order or (
                "newest_first" if searchable_quote_history else "oldest_first"
            )
            if effective_sort_order not in {"newest_first", "oldest_first"}:
                raise ValueError(
                    "sort_order must be newest_first or oldest_first"
                )
            effective_from_day = 1 if from_day is None else from_day
            effective_to_day = state.day if to_day is None else to_day
            if (
                isinstance(effective_from_day, bool)
                or isinstance(effective_to_day, bool)
                or not isinstance(effective_from_day, int)
                or not isinstance(effective_to_day, int)
                or effective_from_day < 1
                or effective_to_day < effective_from_day
                or effective_to_day > state.day
            ):
                raise ValueError(
                    "created-day range must satisfy "
                    f"1 <= from_day <= to_day <= {state.day}"
                )

            normalized_identifiers = sorted(set(identifiers))
            if searchable_quote_history and normalized_identifiers:
                known_identifiers = {
                    request.request_id for request in state.reservation_requests
                } | {
                    str(request.commitment_id)
                    for request in state.reservation_requests
                    if request.commitment_id is not None
                }
                unknown_identifiers = sorted(
                    set(normalized_identifiers) - known_identifiers
                )
                if unknown_identifiers:
                    raise ValueError(
                        "unknown reservation identifiers: "
                        f"{unknown_identifiers}"
                    )
            rows = []
            for request in state.reservation_requests:
                if request.status in {"pending_quote", "pending_service_offer"}:
                    continue
                line_rows = []
                for line in request.lines:
                    public_context = {
                        "line_id": line.line_id,
                        "required": line.required,
                        "service_key": line.service_key,
                        "service_line": line.service_line,
                        "service_units": line.service_units,
                        "customer_units": line.customer_units,
                        "duration_days": line.duration_days,
                        "hard_requirements": dict(
                            line.public_hard_requirements
                        ),
                        "stated_preferences": [
                            dict(row) for row in line.public_preferences
                        ],
                    }
                    if (
                        request.reservation_contract_mode
                        == "adaptive_pairwise_exact_quotes_v1"
                    ):
                        quote_was_accepted = request.commitment_id is not None
                        if quote_was_accepted:
                            quote_line_outcome = (
                                "accepted"
                                if line.selected_quote_option_id is not None
                                else "optional_removed_at_quote"
                            )
                        elif request.quote_action == "quote":
                            quote_line_outcome = "customer_declined_quote"
                        else:
                            quote_line_outcome = request.status
                        public_context.update(
                            {
                                "submitted_quote_options": [
                                    dict(option)
                                    for option in line.submitted_quote_options
                                ],
                                "selected_quote_option_id": (
                                    line.selected_quote_option_id
                                    if quote_was_accepted
                                    else None
                                ),
                                "line_outcome": quote_line_outcome,
                                "current_line_status": line.status,
                            }
                        )
                    elif (
                        request.reservation_contract_mode
                        == "complete_contract_options_v1"
                    ):
                        public_context.update(
                            {
                                "line_outcome": line.status,
                                "current_line_status": line.status,
                            }
                        )
                    elif request.reservation_contract_mode == (
                        "factorized_line_options_plan_set_v1"
                    ):
                        submitted = next(
                            (
                                row
                                for row in request.submitted_line_options
                                if str(row["line_id"]) == line.line_id
                            ),
                            {"options": []},
                        )
                        public_context.update(
                            {
                                "submitted_quote_options": [
                                    dict(option)
                                    for option in submitted.get("options", [])
                                    if option.get("disposition") == "quote"
                                ],
                                "optional_omit_submitted": any(
                                    option.get("disposition") == "omit"
                                    for option in submitted.get("options", [])
                                ),
                                "selected_quote_option_id": (
                                    line.selected_quote_option_id
                                    if request.commitment_id is not None
                                    else None
                                ),
                                "line_outcome": line.status,
                                "current_line_status": line.status,
                            }
                        )
                    else:
                        public_context.update(
                            {
                                "quoted_daily_price": line.quoted_daily_price,
                                "line_outcome": (
                                    line.status
                                    if request.quote_action == "quote"
                                    else request.status
                                ),
                            }
                        )
                    line_rows.append(public_context)
                quote_outcome = (
                    "accepted"
                    if request.commitment_id is not None
                    else "awaiting_manager_confirmation"
                    if request.status == "awaiting_manager_confirmation"
                    else (
                        "customer_declined_quote"
                        if request.quote_action == "quote"
                        else request.status
                    )
                )
                row = {
                        "request_id": request.request_id,
                        "created_day": request.created_day,
                        "quote_day": request.quote_day,
                        "pickup_day": request.pickup_day,
                        "return_day": request.return_day,
                        "booking_lead_days": request.booking_lead_days,
                        "visible_party_context": request.visible_party_context,
                        "forecast_condition": request.forecast_condition,
                        "forecast_accuracy": request.forecast_accuracy,
                        "reservation_contract_mode": (
                            request.reservation_contract_mode
                        ),
                        "quote_action": request.quote_action,
                        "quote_outcome": quote_outcome,
                        "request_outcome": quote_outcome,
                        "current_request_status": request.status,
                        "commitment_id": request.commitment_id,
                        "lines": line_rows,
                    }
                if (
                    request.reservation_contract_mode
                    == "complete_contract_options_v1"
                ):
                    row.update(
                        {
                            "submitted_contract_options": [
                                dict(option)
                                for option in request.submitted_contract_options
                            ],
                            "selected_contract_option_id": (
                                request.selected_contract_option_id
                            ),
                            "contract_assessments": [
                                dict(assessment)
                                for assessment in request.reported_contract_assessments
                            ],
                        }
                    )
                elif request.reservation_contract_mode == (
                    "factorized_line_options_plan_set_v1"
                ):
                    row.update(
                        {
                            "submitted_line_options": [
                                dict(option)
                                for option in request.submitted_line_options
                            ],
                            "internal_plan_count": int(
                                request.internal_plan_count
                            ),
                            "customer_authorized_plans": [
                                dict(plan)
                                for plan in request.customer_authorized_plans
                            ],
                            "customer_preferred_plan_id": (
                                request.customer_preferred_plan_id
                            ),
                            "manager_confirmed_plan_id": (
                                request.manager_confirmed_plan_id
                            ),
                            "plan_confirmation_origin": (
                                request.plan_confirmation_origin
                            ),
                            "decline_feedback": dict(
                                request.reported_decline_feedback
                            ),
                        }
                    )
                if v22_surface and request.reservation_contract_mode == (
                    "factorized_line_options_plan_set_v1"
                ):
                    row = self._v22_compact_quote_history_record(
                        request,
                        quote_outcome=quote_outcome,
                    )
                rows.append(row)
            if searchable_quote_history:
                identifier_set = set(normalized_identifiers)
                offer_id_set = set(normalized_offer_ids)
                service_key_set = set(normalized_service_keys)
                quote_outcome_set = set(normalized_quote_outcomes)
                menu_size_set = set(normalized_menu_sizes)
                contract_option_count_set = set(
                    normalized_contract_option_counts
                )
                assessment_category_set = set(
                    normalized_assessment_categories
                )
                authorized_plan_count_set = set(
                    normalized_authorized_plan_counts
                )
                decline_feedback_category_set = set(
                    normalized_decline_feedback_categories
                )
                decline_feedback_tag_set = set(
                    normalized_decline_feedback_tags
                )
                decline_feedback_concern_set = set(
                    normalized_decline_feedback_concern_sets
                )
                rows = [
                    row
                    for row in rows
                    if not (row.get("request_kind") == "fixed_reward_offer" and (
                        menu_size_set or contract_option_count_set or assessment_category_set
                        or decline_feedback_category_set or decline_feedback_tag_set or decline_feedback_concern_set
                    ))
                    and (
                        not identifier_set
                        or row["request_id"] in identifier_set
                        or row["commitment_id"] in identifier_set
                    )
                    and effective_from_day <= row["created_day"] <= effective_to_day
                    and (
                        not service_key_set
                        or any(
                            line["service_key"] in service_key_set
                            for line in row["lines"]
                        )
                    )
                    and (
                        not offer_id_set
                        or any(
                            str(option["offer_id"]) in offer_id_set
                            for line in row["lines"]
                            for option in (
                                line.get("submitted_quote_options")
                                or line.get("quote_options")
                                or []
                            )
                            if option.get("disposition", "quote") == "quote"
                        )
                        or any(
                            str(contract_line.get("offer_id")) in offer_id_set
                            for contract in (row.get("submitted_contract_options") or row.get("offered_complete_plans", []))
                            for contract_line in contract.get("lines", [])
                            if contract_line.get("disposition") == "quote"
                        )
                    )
                    and (
                        not quote_outcome_set
                        or row["quote_outcome"] in quote_outcome_set
                    )
                    and (
                        not menu_size_set
                        or any(
                            (
                                len(line.get("submitted_quote_options", []))
                                if line.get("submitted_quote_options") is not None
                                else len(line.get("quote_options", []))
                                - sum(
                                    option.get("disposition") == "omit"
                                    for option in line.get("quote_options", [])
                                )
                            )
                            in menu_size_set
                            for line in row["lines"]
                        )
                        or len(row.get("submitted_contract_options", []))
                        in menu_size_set
                    )
                    and (
                        not contract_option_count_set
                        or len(row.get("submitted_contract_options", []))
                        in contract_option_count_set
                    )
                    and (
                        not assessment_category_set
                        or any(
                            assessment.get("reported_assessment", {}).get(
                                "category"
                            )
                            in assessment_category_set
                            for assessment in row.get(
                                "contract_assessments", []
                            )
                        )
                    )
                    and (
                        not authorized_plan_count_set
                        or int(
                            row.get(
                                "customer_authorized_plan_count",
                                len(row.get("customer_authorized_plans", [])),
                            )
                        )
                        in authorized_plan_count_set
                    )
                    and (
                        not decline_feedback_category_set
                        or row.get("decline_feedback", {}).get("category")
                        in decline_feedback_category_set
                    )
                    and (
                        not decline_feedback_tag_set
                        or any(
                            tag in decline_feedback_tag_set
                            for tag in row.get("decline_feedback", {}).get(
                                "reported_concerns", []
                            )
                        )
                    )
                    and (
                        not decline_feedback_concern_set
                        or "+".join(
                            str(tag)
                            for tag in row.get("decline_feedback", {}).get(
                                "reported_concerns", []
                            )
                        )
                        in decline_feedback_concern_set
                    )
                ]
            rows.sort(
                key=(
                    (lambda row: (-row["created_day"], row["request_id"]))
                    if effective_sort_order == "newest_first"
                    else (lambda row: (row["created_day"], row["request_id"]))
                )
            )
            page = rows[cursor : cursor + limit]
            next_cursor = (
                cursor + len(page)
                if cursor + len(page) < len(rows)
                else None
            )
            result = {
                "view": "quote_history",
                "quote_records": page,
                "total_count": len(rows),
                "returned_count": len(page),
                "cursor": cursor,
                "limit": limit,
                "has_more": next_cursor is not None,
                "next_cursor": next_cursor,
            }
            if searchable_quote_history:
                query = {
                    "identifiers": normalized_identifiers,
                    "created_day": {
                        "from_day": effective_from_day,
                        "to_day": effective_to_day,
                    },
                    "offer_ids": normalized_offer_ids,
                    "service_keys": normalized_service_keys,
                    "quote_outcomes": normalized_quote_outcomes,
                }
                if self._reservation_complete_contract_mode():
                    query.update(
                        {
                            "contract_option_counts": (
                                normalized_contract_option_counts
                            ),
                            "assessment_categories": (
                                normalized_assessment_categories
                            ),
                        }
                    )
                elif self._reservation_factorized_plan_mode():
                    query.update(
                        {
                            "menu_sizes": normalized_menu_sizes,
                            "authorized_plan_counts": (
                                normalized_authorized_plan_counts
                            ),
                            **(
                                {
                                    "decline_feedback_tags": (
                                        normalized_decline_feedback_tags
                                    ),
                                    "decline_feedback_concern_sets": (
                                        normalized_decline_feedback_concern_sets
                                    ),
                                }
                                if v23_surface
                                else {
                                    "decline_feedback_categories": (
                                        normalized_decline_feedback_categories
                                    )
                                }
                            ),
                        }
                    )
                else:
                    query["menu_sizes"] = normalized_menu_sizes
                result.update(
                    {
                        "sort_order": effective_sort_order,
                        "query": query,
                    }
                )
            return result

        if not identifiers:
            raise ValueError(f"identifiers are required for the {view} view")
        if view == "detail" and len(identifiers) > 3:
            raise ValueError("detail view accepts at most 3 identifiers")
        if view == "timeline" and len(identifiers) != 1:
            raise ValueError("timeline view requires exactly one identifier")

        requested = set(identifiers)
        requested_change_ids = {
            row.change_id
            for row in state.reservation_change_requests
            if row.change_id in requested
        }
        change_commitment_ids = {
            row.commitment_id
            for row in state.reservation_change_requests
            if row.change_id in requested_change_ids
        }
        request_objects = [
            row
            for row in state.reservation_requests
            if row.request_id in requested
            or (row.commitment_id is not None and row.commitment_id in requested)
            or (
                row.commitment_id is not None
                and row.commitment_id in change_commitment_ids
            )
        ]
        commitment_objects = [
            row
            for row in state.reservation_commitments
            if row.commitment_id in requested
            or row.request_id in requested
            or row.commitment_id in change_commitment_ids
        ]
        requests = [
            self._public_reservation_request(row) for row in request_objects
        ]
        commitments = [
            self._public_reservation_commitment(row)
            for row in commitment_objects
        ]
        matched = {
            row.request_id for row in request_objects
            if row.request_id in requested
        } | {
            row.commitment_id for row in commitment_objects
            if row.commitment_id in requested
        } | requested_change_ids
        unknown = sorted(requested - matched)
        if unknown:
            raise ValueError(f"unknown reservation identifiers: {unknown}")
        related_ids = matched | {
            row["request_id"] for row in commitments
        } | {
            row["commitment_id"] for row in commitments
        }
        if view == "timeline":
            events = [
                row for row in (
                    self._actor_reservation_events(state)
                    if self._v43_actor_manual()
                    else [asdict(event) for event in state.reservation_events]
                )
                if row["request_id"] in related_ids or row["commitment_id"] in related_ids
            ]
            events.sort(key=lambda row: (int(row["day"]), str(row["event_id"])))
            page = events[cursor : cursor + limit]
            next_cursor = (
                cursor + len(page)
                if cursor + len(page) < len(events)
                else None
            )
            return {
                "view": "timeline",
                "identifiers": sorted(requested),
                "events": page,
                "total_count": len(events),
                "returned_count": len(page),
                "cursor": cursor,
                "limit": limit,
                "has_more": next_cursor is not None,
                "next_cursor": next_cursor,
            }
        if view != "detail":
            raise ValueError(
                "view must be index, detail, timeline, or quote_history"
            )
        if v22_surface:
            commitment_by_request = {
                row.request_id: row for row in commitment_objects
            }
            changes_by_commitment: dict[str, list[ReservationChangeRequest]] = {}
            for change in state.reservation_change_requests:
                changes_by_commitment.setdefault(
                    change.commitment_id, []
                ).append(change)
            records = []
            for request in sorted(
                request_objects, key=lambda row: row.request_id
            ):
                commitment = commitment_by_request.get(request.request_id)
                related_changes = (
                    []
                    if commitment is None
                    else sorted(
                        changes_by_commitment.get(
                            commitment.commitment_id, []
                        ),
                        key=lambda row: row.change_id,
                    )
                )
                identifiers_for_record = sorted(
                    identifier
                    for identifier in requested
                    if identifier == request.request_id
                    or (
                        commitment is not None
                        and identifier == commitment.commitment_id
                    )
                    or any(
                        identifier == change.change_id
                        for change in related_changes
                    )
                )
                record = self._v22_reservation_detail_record(
                        request,
                        commitment,
                        changes=related_changes,
                        matched_identifiers=identifiers_for_record,
                        sections=normalized_sections,
                        v26_contract_state=v26_surface,
                        v27_extension_cash_audit=v27_surface,
                        v29_public_timing=self._v29_reservation_accountability(),
                    )
                if "cancellation_terms" in normalized_sections:
                    record["cancellation_terms"] = (
                        None
                        if commitment is None
                        else self.reservation_cancellation_terms(
                            state, commitment
                        )
                    )
                records.append(record)
            return {
                "view": "detail",
                "records": records,
                "requested_sections": list(normalized_sections),
                "identifier_count": len(requested),
                "record_count": len(records),
                "has_more": False,
                "next_cursor": None,
            }
        commitment_ids = {row["commitment_id"] for row in commitments}
        changes = [
            asdict(row)
            for row in state.reservation_change_requests
            if row.commitment_id in commitment_ids
        ]
        return {
            "view": "detail",
            "requests": requests,
            "commitments": commitments,
            "changes": changes,
            "identifier_count": len(requested),
            "limit": len(requested),
            "has_more": False,
            "next_cursor": None,
        }

    def view_reservation_calendar(
        self,
        state: WorldState,
        *,
        offer_ids: list[str],
        from_day: int,
        to_day: int,
    ) -> dict[str, Any]:
        selected = sorted(
            set(offer_ids)
            or {
                offer_id
                for commitment in state.reservation_commitments
                for line in commitment.lines
                for offer_id in (
                    line.authorized_offer_ids or [line.offer_id]
                )
            }
        )
        selected_set = set(selected)
        rows = []
        for day in range(from_day, to_day + 1):
            blocks = [
                self._public_reservation_calendar_block(block)
                for block in state.reservation_calendar_blocks
                if selected_set.intersection(
                    block.authorized_offer_ids or [block.offer_id]
                )
                and block.day == day
                and block.status in {"active", "bound"}
            ]
            deliveries = [
                {
                    "order_id": order.order_id,
                    "offer_id": str(item["offer_id"]),
                    "units": int(item["packs"] * item["package_units"]),
                    "actual_delivery_day": order.delivery_day,
                    "estimated_delivery_day": self.estimated_order_delivery_day(order),
                    "status": order.status,
                }
                for order in state.orders
                for item in order.items
                if str(item["offer_id"]) in selected
                and (
                    order.delivery_day == day
                    or self.estimated_order_delivery_day(order) == day
                )
            ]
            rows.append({"day": day, "commitment_blocks": blocks, "deliveries": deliveries})
        assets = [
            {
                "asset_id": asset.asset_id,
                "offer_id": asset.offer_id,
                "status": asset.status,
                "ready_day": asset.ready_day,
                **(
                    {
                        "ownership_type": asset.ownership_type,
                        "supplier_trial": (
                            self._public_supplier_trial(
                                self._supplier_trial_record_for_asset(state, asset)
                            )
                            if asset.ownership_type == "supplier_trial"
                            else None
                        ),
                    }
                    if self._v37_supplier_trial()
                    else {}
                ),
            }
            for asset in state.assets
            if asset.offer_id in selected
            and (
                self._warehouse_asset_held(asset)
                if self._v31_warehouse_capacity()
                else asset.status not in {"retired", "sold"}
            )
        ]
        return {
            "from_day": from_day,
            "to_day": to_day,
            "offers": selected,
            "current_assets": assets,
            "days": rows,
            "interpretation": (
                "raw physical, commitment, and delivery facts only; the manager "
                "must reconcile the calendars and asset states"
            ),
        }

    def _rental_capacity_service_jobs(
        self, state: WorldState
    ) -> list[dict[str, Any]]:
        """Return current service jobs without projecting any future arrivals."""

        capacity = float(self.rules.cleaning_capacity_points)
        cumulative_work = 0.0
        jobs: list[dict[str, Any]] = []
        for position, job in enumerate(
            sorted(
                state.cleaning_queue,
                key=lambda row: (row.enqueued_day, row.asset_id),
            ),
            start=1,
        ):
            work_ahead = cumulative_work
            cumulative_work += float(job.remaining_work_points)
            mornings = math.ceil(
                max(0.0, cumulative_work - 1e-9) / max(capacity, 1e-9)
            )
            jobs.append(
                {
                    "job_type": "cleaning",
                    "global_queue_position": position,
                    "asset_id": job.asset_id,
                    "offer_id": job.offer_id,
                    "enqueued_day": int(job.enqueued_day),
                    "remaining_work_points": round(
                        float(job.remaining_work_points), 6
                    ),
                    "work_points_ahead": round(work_ahead, 6),
                    "cleaning_cost": round(float(job.cleaning_cost), 2),
                    "projected_completion_day": int(state.day + mornings),
                    "cost_payment_timing": "on_completion",
                }
            )

        latest_start: dict[tuple[str, str], dict[str, Any]] = {}
        for event in state.rental_activity:
            event_type = str(event.get("event_type") or "")
            if event_type not in {"maintenance_started", "repair_started"}:
                continue
            latest_start[(str(event["asset_id"]), event_type)] = event
        for asset in sorted(state.assets, key=lambda row: row.asset_id):
            if asset.status not in {"maintenance", "repairing"}:
                continue
            event_type = (
                "maintenance_started"
                if asset.status == "maintenance"
                else "repair_started"
            )
            event = latest_start.get((asset.asset_id, event_type))
            if event is None:
                raise AssertionError(
                    f"current service job lacks a start event: {asset.asset_id}"
                )
            cost_field = (
                "maintenance_cost"
                if asset.status == "maintenance"
                else "repair_cost"
            )
            jobs.append(
                {
                    "job_type": (
                        "maintenance"
                        if asset.status == "maintenance"
                        else "repair"
                    ),
                    "asset_id": asset.asset_id,
                    "offer_id": asset.offer_id,
                    "started_day": int(event["day"]),
                    "ready_day": int(asset.ready_day),
                    "paid_cost": round(float(event[cost_field]), 2),
                    "cost_payment_timing": "paid_at_start",
                }
            )
        type_order = {"cleaning": 0, "maintenance": 1, "repair": 2}
        jobs.sort(
            key=lambda row: (
                type_order[str(row["job_type"])],
                int(row.get("global_queue_position") or 0),
                int(
                    row.get("projected_completion_day")
                    or row.get("ready_day")
                    or 0
                ),
                str(row["asset_id"]),
            )
        )
        return jobs

    def _compressed_capacity_commitment_intervals(
        self,
        state: WorldState,
        blocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        grouped: dict[tuple[Any, ...], list[int]] = {}
        payload_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
        for block in blocks:
            payload = {key: value for key, value in block.items() if key != "day"}
            key = tuple(
                (name, json.dumps(value, ensure_ascii=False, sort_keys=True))
                for name, value in sorted(payload.items())
            )
            grouped.setdefault(key, []).append(int(block["day"]))
            payload_by_key[key] = payload
        intervals: list[dict[str, Any]] = []
        for key in sorted(
            grouped,
            key=lambda value: (
                str(payload_by_key[value].get("commitment_id")),
                str(payload_by_key[value].get("line_id")),
                str(payload_by_key[value].get("offer_id")),
            ),
        ):
            days = sorted(set(grouped[key]))
            start = previous = days[0]
            for day in days[1:] + [days[-1] + 2]:
                if day == previous + 1:
                    previous = day
                    continue
                if self._v24_reservation_operations():
                    commitment_id = str(
                        payload_by_key[key].get("commitment_id") or ""
                    )
                    commitment = next(
                        (
                            row
                            for row in state.reservation_commitments
                            if row.commitment_id == commitment_id
                        ),
                        None,
                    )
                    intervals.append(
                        {
                            **payload_by_key[key],
                            "pickup_day": (
                                start
                                if commitment is None
                                else int(commitment.pickup_day)
                            ),
                            "occupied_from_day": (
                                start
                                if commitment is None
                                else int(commitment.pickup_day)
                            ),
                            "occupied_through_day": (
                                previous
                                if commitment is None
                                else int(commitment.return_day) - 1
                            ),
                            "scheduled_return_day": (
                                previous + 1
                                if commitment is None
                                else int(commitment.return_day)
                            ),
                        }
                    )
                else:
                    intervals.append(
                        {
                            **payload_by_key[key],
                            "start_day": start,
                            "end_day": previous,
                        }
                    )
                start = previous = day
        intervals.sort(
            key=lambda row: (
                int(
                    row.get("occupied_from_day", row.get("start_day", 0))
                ),
                int(
                    row.get("occupied_through_day", row.get("end_day", 0))
                ),
                str(row.get("commitment_id")),
                str(row.get("line_id")),
                str(row.get("offer_id")),
            )
        )
        return intervals

    def view_rental_capacity(
        self,
        state: WorldState,
        *,
        view: str,
        offer_ids: list[str],
        from_day: int | None = None,
        to_day: int | None = None,
        job_types: list[str] | None = None,
        cursor: int = 0,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Expose raw capacity facts without calculating operational feasibility."""

        all_jobs = self._rental_capacity_service_jobs(state)
        if view == "service_jobs":
            selected_offers = set(offer_ids)
            selected_types = set(job_types or {"cleaning", "maintenance", "repair"})
            filtered = [
                row
                for row in all_jobs
                if (not selected_offers or row["offer_id"] in selected_offers)
                and row["job_type"] in selected_types
            ]
            page = filtered[cursor : cursor + limit]
            next_cursor = (
                cursor + len(page)
                if cursor + len(page) < len(filtered)
                else None
            )
            return {
                "view": "service_jobs",
                **({"warehouse_capacity": self._warehouse_summary(state)} if self._v31_warehouse_capacity() else {}),
                "as_of_day": state.day,
                "offers": sorted(selected_offers),
                "job_types": sorted(selected_types),
                "jobs": page,
                "total_count": len(filtered),
                "returned_count": len(page),
                "cursor": cursor,
                "limit": limit,
                "has_more": next_cursor is not None,
                "next_cursor": next_cursor,
                "global_cleaning_queue_jobs": sum(
                    row["job_type"] == "cleaning" for row in all_jobs
                ),
                "projection_basis": {
                    "daily_cleaning_work_points": float(
                        self.rules.cleaning_capacity_points
                    ),
                    "current_jobs_only": True,
                    "future_return_jobs_included": False,
                    "offer_filter_changes_global_fifo": False,
                },
            }
        if view != "timeline":
            raise ValueError("view must be timeline or service_jobs")
        if from_day is None or to_day is None:
            raise ValueError("timeline requires from_day and to_day")

        selected = set(offer_ids)
        if not selected:
            selected.update(
                asset.offer_id
                for asset in state.assets
                if (
                    self._warehouse_asset_held(asset)
                    if self._v31_warehouse_capacity()
                    else asset.status not in {"retired", "sold"}
                )
            )
            selected.update(
                str(item["offer_id"])
                for order in state.orders
                if order.status == "in_transit"
                for item in order.items
                if self.offers[str(item["offer_id"])].asset_kind == "durable"
            )
            selected.update(
                str(row["offer_id"])
                for row in all_jobs
            )
            selected.update(
                offer_id
                for block in state.reservation_calendar_blocks
                if from_day <= block.day <= to_day
                and block.status in {"active", "bound"}
                for offer_id in (
                    block.authorized_offer_ids or [block.offer_id]
                )
            )
            if self._v24_reservation_operations():
                selected.update(
                    row.offer_id
                    for row in state.walk_in_capacity_limits
                    if from_day <= int(row.day) <= to_day
                )
                selected.update(
                    contract.offer_id
                    for contract in state.contracts
                    if contract.status == "active"
                    and contract.reservation_id is None
                    and contract.start_day <= to_day
                    and contract.return_day > from_day
                )
        selected_list = sorted(selected)

        assets_by_offer: list[dict[str, Any]] = []
        status_order = (
            "available",
            "rented",
            "cleaning",
            "damaged",
            "repairing",
            "maintenance",
            "maintenance_hold",
        )
        for offer_id in selected_list:
            assets = [
                asset
                for asset in state.assets
                if asset.offer_id == offer_id
                and (
                    self._warehouse_asset_held(asset)
                    if self._v31_warehouse_capacity()
                    else asset.status not in {"retired", "sold"}
                )
            ]
            assets_by_offer.append(
                {
                    "offer_id": offer_id,
                    "name": self.offers[offer_id].name,
                    "total_current_assets": len(assets),
                    **(
                        {
                            "owned_assets": sum(
                                asset.ownership_type == "owned" for asset in assets
                            ),
                            "supplier_trial_assets": [
                                self._public_supplier_trial(
                                    self._supplier_trial_record_for_asset(state, asset)
                                )
                                for asset in assets
                                if asset.ownership_type == "supplier_trial"
                            ],
                        }
                        if self._v37_supplier_trial()
                        else {}
                    ),
                    "status_counts": {
                        status: sum(asset.status == status for asset in assets)
                        for status in status_order
                        if any(asset.status == status for asset in assets)
                    },
                }
            )

        public_blocks = [
            self._public_reservation_calendar_block(block)
            for block in state.reservation_calendar_blocks
            if from_day <= block.day <= to_day
            and block.status in {"active", "bound"}
            and selected.intersection(
                block.authorized_offer_ids or [block.offer_id]
            )
        ]
        supply_events: list[dict[str, Any]] = []
        for order in state.orders:
            if order.status == "in_transit":
                event_day = self.estimated_order_delivery_day(order)
                event_type = "estimated_delivery"
            elif order.status == "delivered":
                event_day = int(order.delivery_day)
                event_type = "actual_delivery"
            else:
                continue
            if not from_day <= event_day <= to_day:
                continue
            for item in order.items:
                offer_id = str(item["offer_id"])
                if offer_id not in selected or self.offers[offer_id].asset_kind != "durable":
                    continue
                event = {
                    "day": event_day,
                    "event_type": event_type,
                    "order_id": order.order_id,
                    "offer_id": offer_id,
                    "units": int(item["packs"] * item["package_units"]),
                    "order_status": order.status,
                    **(
                        {
                            "procurement_mode": order.procurement_mode,
                            "supplier_trial_id": order.supplier_trial_id,
                        }
                        if self._v37_supplier_trial()
                        else {}
                    ),
                    "estimated_delivery_day": self.estimated_order_delivery_day(
                        order
                    ),
                }
                if self._v31_warehouse_capacity() and order.status == "in_transit":
                    event["units"] -= item["received_units"]
                if order.status == "delivered":
                    event["actual_delivery_day"] = int(order.delivery_day)
                supply_events.append(event)
        for job in all_jobs:
            offer_id = str(job["offer_id"])
            completion_day = int(
                job.get("projected_completion_day")
                or job.get("ready_day")
            )
            if offer_id not in selected or not from_day <= completion_day <= to_day:
                continue
            supply_events.append(
                {
                    "day": completion_day,
                    "event_type": "service_completion",
                    "job_type": job["job_type"],
                    "asset_id": job["asset_id"],
                    "offer_id": offer_id,
                    "projection": (
                        "current_queue_mechanical"
                        if job["job_type"] == "cleaning"
                        else "fixed_ready_day"
                    ),
                }
            )
        supply_events.sort(
            key=lambda row: (
                int(row["day"]),
                str(row["event_type"]),
                str(row.get("offer_id")),
                str(row.get("order_id") or row.get("asset_id")),
            )
        )
        walk_in_occupancy_intervals: list[dict[str, Any]] = []
        if self._v24_reservation_operations():
            grouped_walk_ins: Counter[tuple[str, int, int]] = Counter()
            for contract in state.contracts:
                if (
                    contract.status != "active"
                    or contract.reservation_id is not None
                    or contract.offer_id not in selected
                    or contract.start_day > to_day
                    or contract.return_day <= from_day
                ):
                    continue
                grouped_walk_ins[
                    (
                        contract.offer_id,
                        int(contract.start_day),
                        int(contract.return_day),
                    )
                ] += 1
            walk_in_occupancy_intervals = [
                {
                    "offer_id": offer_id,
                    "pickup_day": pickup_day,
                    "occupied_from_day": max(pickup_day, from_day),
                    "occupied_through_day": min(return_day - 1, to_day),
                    "scheduled_return_day": return_day,
                    "occupied_units": quantity,
                }
                for (offer_id, pickup_day, return_day), quantity in sorted(
                    grouped_walk_ins.items()
                )
            ]
        return {
            "view": "timeline",
            **({"warehouse_capacity": self._warehouse_summary(state)} if self._v31_warehouse_capacity() else {}),
            "from_day": from_day,
            "to_day": to_day,
            "offers": selected_list,
            "current_assets": assets_by_offer,
            "commitment_intervals": self._compressed_capacity_commitment_intervals(
                state, public_blocks
            ),
            **(
                {
                    "walk_in_capacity_limit_intervals": (
                        self._compressed_walk_in_capacity_limit_intervals(
                            state,
                            offer_ids=selected,
                            from_day=from_day,
                            to_day=to_day,
                        )
                    ),
                    "walk_in_occupancy_intervals": walk_in_occupancy_intervals,
                }
                if self._v24_reservation_operations()
                else {}
            ),
            "supply_events": supply_events,
            "interpretation": (
                "raw current assets, binding intervals, public delivery timing, "
                + (
                    "configured walk-in limits, confirmed walk-in occupancy, "
                    if self._v24_reservation_operations()
                    else ""
                )
                + "and current service-job completions only; no net availability, "
                "shortage, coverage, feasibility, or recommendation is computed"
            ),
        }

    def _v24_preauthorized_recovery_plan_view(
        self,
        commitment: ReservationCommitment,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        original_by_line = {line.line_id: line for line in commitment.lines}
        protected_lines: list[dict[str, Any]] = []
        for row in plan.get("lines", []):
            line_id = str(row["line_id"])
            if row.get("disposition") == "omit":
                protected_lines.append(
                    {
                        "line_id": line_id,
                        "required": False,
                        "disposition": "omit",
                        "customer_payable_rent": 0.0,
                        "customer_security_deposit": 0.0,
                    }
                )
                continue
            original = original_by_line.get(line_id)
            authorized_rent = round(float(row["line_rent"]), 2)
            authorized_deposit = round(float(row["security_deposit"]), 2)
            original_rent = 0.0 if original is None else float(original.locked_rent)
            original_deposit = (
                0.0
                if original is None
                else float(original.locked_security_deposit)
            )
            payable_rent = round(min(original_rent, authorized_rent), 2)
            payable_deposit = round(
                min(original_deposit, authorized_deposit), 2
            )
            protected_lines.append(
                {
                    **dict(row),
                    "authorized_line_rent": authorized_rent,
                    "authorized_security_deposit": authorized_deposit,
                    "customer_payable_rent": payable_rent,
                    "customer_security_deposit": payable_deposit,
                    "recovery_price_credit": round(
                        authorized_rent - payable_rent, 2
                    ),
                    "recovery_deposit_credit": round(
                        authorized_deposit - payable_deposit, 2
                    ),
                }
            )
        total_rent = round(
            sum(float(row["customer_payable_rent"]) for row in protected_lines),
            2,
        )
        total_deposit = round(
            sum(
                float(row["customer_security_deposit"])
                for row in protected_lines
            ),
            2,
        )
        return {
            "plan_id": str(plan["plan_id"]),
            "lines": protected_lines,
            "authorized_total_rent": round(float(plan["total_rent"]), 2),
            "authorized_total_security_deposit": round(
                float(plan["total_security_deposit"]), 2
            ),
            "customer_payable_total_rent": total_rent,
            "customer_security_deposit_total": total_deposit,
            "rebased_prepayment_due": (
                self._cent_half_up(total_rent * float(self.config.reservation_prepayment_rate))
                if any(l.locked_line_daily_rent is not None for l in commitment.lines)
                else round(total_rent * float(self.config.reservation_prepayment_rate), 2)
            ),
            "reserves_inventory_before_selection": False,
            "feasibility_evaluated": False,
        }

    def _v24_select_preauthorized_recovery_plan(
        self,
        state: WorldState,
        request: ReservationRequest,
        commitment: ReservationCommitment,
        *,
        plan_id: str,
    ) -> dict[str, Any]:
        if commitment.recovery_path is not None:
            raise ValueError(
                "a recovery path has already been selected for this commitment"
            )
        plan = next(
            (
                row
                for row in commitment.preauthorized_recovery_plans
                if str(row["plan_id"]) == str(plan_id)
            ),
            None,
        )
        if plan is None:
            raise ValueError(
                f"plan_id is not a preauthorized recovery plan: {plan_id}"
            )
        if not self._recovery_plan_retains_commitment(commitment, plan):
            raise ValueError(
                "recovery plan omits committed lines; use inquire_partial_fulfillment "
                "and, only after customer acceptance, select_partial_fulfillment"
            )
        protected = self._v24_preauthorized_recovery_plan_view(
            commitment, plan
        )
        request_lines = {line.line_id: line for line in request.lines}
        protected_by_line = {
            str(row["line_id"]): row for row in protected["lines"]
        }
        replacement_lines: list[ReservationLine] = []
        for plan_row in plan.get("lines", []):
            line_id = str(plan_row["line_id"])
            if plan_row.get("disposition") == "omit":
                continue
            request_line = request_lines[line_id]
            enriched = self._enriched_pairwise_quote_options(
                request,
                request_line,
                quote_options=[
                    {
                        "option_id": str(plan_row["line_option_id"]),
                        "offer_id": str(plan_row["offer_id"]),
                        "promise_profile_id": str(
                            plan_row["promise_profile_id"]
                        ),
                        "daily_unit_price": float(
                            plan_row["daily_unit_price"]
                        ),
                    }
                ],
            )[0]
            protected_row = protected_by_line[line_id]
            primary = str(enriched["offer_id"])
            line = replace(request_line)
            line.offer_id = primary
            line.service_line = self.offers[primary].service_line
            line.selected_quote_option_id = str(enriched["option_id"])
            line.selected_offer_id = primary
            line.quoted_daily_price = float(enriched["daily_unit_price"])
            line.promise_profile_id = str(enriched["promise_profile_id"])
            line.profile_family_id = str(enriched["profile_family_id"])
            line.profile_schema_version = str(
                enriched["profile_schema_version"]
            )
            line.capacity_per_asset = int(enriched["capacity_per_asset"])
            line.asset_quantity = int(enriched["asset_quantity"])
            line.quantity = int(enriched["asset_quantity"])
            line.locked_daily_price = float(enriched["daily_unit_price"])
            line.authorized_line_rent = float(
                protected_row["authorized_line_rent"]
            )
            line.authorized_security_deposit = float(
                protected_row["authorized_security_deposit"]
            )
            line.locked_rent = float(
                protected_row["customer_payable_rent"]
            )
            line.locked_security_deposit = float(
                protected_row["customer_security_deposit"]
            )
            line.recovery_price_credit = float(
                protected_row["recovery_price_credit"]
            )
            line.recovery_deposit_credit = float(
                protected_row["recovery_deposit_credit"]
            )
            line.recovery_source_plan_id = str(plan_id)
            line.authorized_offer_ids = [primary]
            line.fulfillment_mode = "preauthorized_plan_selected"
            line.status = "committed"
            replacement_lines.append(line)

        new_total_rent = round(
            sum(line.locked_rent for line in replacement_lines), 2
        )
        new_prepayment = round(
            new_total_rent * float(request.prepayment_rate), 2
        )
        current_prepayment = (
            float(commitment.unearned_revenue)
            if self._v26_contract_state()
            else float(commitment.prepayment_collected)
        )
        prepayment_refund = round(
            max(0.0, current_prepayment - new_prepayment), 2
        )
        refund_succeeded = True
        if prepayment_refund > 0:
            state.unearned_revenue_liability = round(
                max(
                    0.0,
                    state.unearned_revenue_liability - prepayment_refund,
                ),
                2,
            )
            state.current_day_ledger["reservation_prepayment_refunds"] = round(
                float(
                    state.current_day_ledger.get(
                        "reservation_prepayment_refunds", 0.0
                    )
                )
                + prepayment_refund,
                2,
            )
            state.cumulative_reservation_prepayment_refunds = round(
                state.cumulative_reservation_prepayment_refunds
                + prepayment_refund,
                2,
            )
            if self._v26_contract_state():
                commitment.prepayment_refunded = round(
                    commitment.prepayment_refunded + prepayment_refund, 2
                )
                commitment.unearned_revenue = new_prepayment
            refund_succeeded = self._refund(
                state, prepayment_refund, commitment.commitment_id
            )
            if not refund_succeeded and not self._v26_contract_state():
                return {
                    "commitment_id": commitment.commitment_id,
                    "selected_recovery_plan_id": str(plan_id),
                    "status": "refund_failed",
                    "prepayment_refund": prepayment_refund,
                }

        for block in state.reservation_calendar_blocks:
            if (
                block.commitment_id == commitment.commitment_id
                and block.status in {"active", "bound"}
            ):
                block.status = "superseded_by_recovery"
        commitment.lines = replacement_lines
        commitment.total_locked_rent = new_total_rent
        if not self._v26_contract_state():
            commitment.prepayment_collected = new_prepayment
        commitment.outstanding_rent = round(
            new_total_rent - new_prepayment, 2
        )
        commitment.unearned_revenue = new_prepayment
        if self._v26_contract_state():
            commitment.rent_due_at_pickup = commitment.outstanding_rent
        commitment.selected_recovery_plan_id = str(plan_id)
        commitment.recovery_path = "preauthorized_complete_plan"
        request.total_locked_rent = new_total_rent
        request.prepayment_due = new_prepayment
        for line in commitment.lines:
            for day in range(commitment.pickup_day, commitment.return_day):
                state.reservation_calendar_blocks.append(
                    CalendarBlock(
                        block_id=(
                            f"{commitment.commitment_id}-recovery-{plan_id}-"
                            f"{line.line_id}-{day:03d}"
                        ),
                        commitment_id=commitment.commitment_id,
                        line_id=line.line_id,
                        offer_id=line.offer_id,
                        day=day,
                        quantity=line.quantity,
                        authorized_offer_ids=[line.offer_id],
                        selected_offer_id=line.offer_id,
                        promise_profile_id=line.promise_profile_id,
                        profile_schema_version=line.profile_schema_version,
                        service_units=line.service_units,
                        capacity_per_asset=line.capacity_per_asset,
                    )
                )
        self._assert_v26_reservation_financial_invariants(state)
        self._reservation_event(
            state,
            "preauthorized_recovery_plan_selected",
            request_id=request.request_id,
            commitment_id=commitment.commitment_id,
            details={
                "plan_id": str(plan_id),
                "customer_payable_total_rent": new_total_rent,
                "customer_security_deposit_total": round(
                    sum(
                        line.locked_security_deposit
                        for line in commitment.lines
                    ),
                    2,
                ),
                "prepayment_refund": prepayment_refund,
            },
        )
        return {
            "commitment_id": commitment.commitment_id,
            "selected_recovery_plan_id": str(plan_id),
            "recovery_path": commitment.recovery_path,
            "customer_payable_total_rent": new_total_rent,
            **(
                {"refund_status": "paid" if refund_succeeded else "failed"}
                if prepayment_refund > 0
                else {}
            ),
            "customer_security_deposit_total": round(
                sum(
                    line.locked_security_deposit
                    for line in commitment.lines
                ),
                2,
            ),
            "prepayment_collected": new_prepayment,
            "prepayment_refund": prepayment_refund,
            "status": "recovery_pending_until_pickup_cutoff",
            "lines": [
                self._public_reservation_line(line)
                for line in commitment.lines
            ],
        }

    def resolve_reservation_risks(
        self, state: WorldState, actions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not actions:
            raise ValueError("actions must not be empty")
        commitments = {
            row.commitment_id: row for row in state.reservation_commitments
        }
        requests = {row.request_id: row for row in state.reservation_requests}
        results = []
        for action_row in actions:
            if state.done and int(self.dataset.metadata.get("tool_interface_version", 0)) >= 44:
                results.append({
                    "commitment_id": str(action_row.get("commitment_id", "")),
                    "action": str(action_row.get("action", "")),
                    "status": "not_executed", "reason": "business_terminated",
                })
                continue
            commitment_id = str(action_row.get("commitment_id", ""))
            action = str(action_row.get("action", ""))
            if commitment_id not in commitments:
                raise ValueError(f"unknown commitment_id: {commitment_id}")
            commitment = commitments[commitment_id]
            if commitment.status != "at_risk":
                raise ValueError(f"commitment is not at risk: {commitment_id}")
            request = requests[commitment.request_id]
            if action == "inquire_partial_fulfillment":
                terms = self._inquire_partial_fulfillment(state, request, commitment)
                results.append({"commitment_id": commitment_id, "action": action, **terms})
                continue
            if action == "select_partial_fulfillment":
                if not self._v45_reservation_difficulty():
                    raise ValueError("partial fulfillment requires V45")
                if commitment.partial_fulfillment_inquiry_day != state.day:
                    raise ValueError("partial fulfillment response is not current")
                if commitment.partial_fulfillment_accepted is not True:
                    raise ValueError("customer did not accept reduced pickup")
                if commitment.partial_fulfillment_selected:
                    raise ValueError("reduced pickup already selected")
                if self._v54_recovery_state():
                    current_missing, _ = self._current_partial_pickup(state, commitment)
                    if not current_missing or current_missing != commitment.partial_fulfillment_missing_line_ids:
                        raise ValueError("current pickup shortage differs from the customer's reduction consent")
                commitment.partial_fulfillment_selected = True
                self._reservation_event(
                    state, "partial_fulfillment_selected",
                    request_id=request.request_id,
                    commitment_id=commitment_id,
                    details={
                        "missing_line_ids": list(commitment.partial_fulfillment_missing_line_ids),
                        "compensation": commitment.partial_fulfillment_compensation,
                    },
                )
                results.append({"commitment_id": commitment_id, "action": action,
                                "status": "reduced_pickup_pending_until_cutoff"})
                continue
            if action == "inquire_fallback":
                first_inquiry = commitment.fallback_inquiry_day is None
                if first_inquiry:
                    revealed: dict[str, list[str]] = {}
                    for line in commitment.lines:
                        if not line.required:
                            revealed[line.line_id] = []
                            continue
                        latent = (
                            commitment.version_recovery_offer_ids_by_line.get(
                                line.line_id, []
                            )
                            if self._v38_proactive_contingency()
                            and commitment.version_recovery_offer_ids_by_line
                            else request.recovery_offer_ids_by_line.get(
                                line.line_id,
                                request.fallback_offer_ids_by_line.get(
                                    line.line_id, []
                                ),
                            )
                        )
                        already_authorized = set(
                            line.authorized_offer_ids or [line.offer_id]
                        )
                        revealed[line.line_id] = [
                            offer_id
                            for offer_id in latent
                            if offer_id not in already_authorized
                            and self._credible_reservation_offer(state, offer_id)
                        ]
                    commitment.fallback_revealed_offer_ids_by_line = revealed
                    commitment.fallback_inquiry_day = state.day
                options = [
                    {
                        "line_id": line_id,
                        "offer_id": offer_id,
                        "name": self.offers[offer_id].name,
                        "customer_authorized": True,
                        "locked_rent_unchanged": True,
                    }
                    for line_id, offer_ids in (
                        commitment.fallback_revealed_offer_ids_by_line.items()
                    )
                    for offer_id in offer_ids
                ]
                preauthorized_plans = (
                    [
                        self._v24_preauthorized_recovery_plan_view(
                            commitment, plan
                        )
                        for plan in self._eligible_preauthorized_recovery_plans(commitment)
                    ]
                    if self._v24_reservation_operations()
                    else []
                )
                if first_inquiry:
                    self._reservation_event(
                        state,
                        "fallback_inquired",
                        request_id=request.request_id,
                        commitment_id=commitment_id,
                        details={
                            "authorized_offer_ids": sorted(
                                {
                                    offer_id
                                    for offer_ids in commitment.fallback_revealed_offer_ids_by_line.values()
                                    for offer_id in offer_ids
                                }
                            ),
                            **(
                                {
                                    "preauthorized_recovery_plan_ids": [
                                        str(row["plan_id"])
                                        for row in preauthorized_plans
                                    ]
                                }
                                if self._v24_reservation_operations()
                                else {}
                            ),
                        },
                    )
                results.append(
                    {
                        "commitment_id": commitment_id,
                        "action": action,
                        **({
                            "fallback_query_status": self._current_fallback_query_status(commitment),
                            "fallback_inquiry_day": commitment.fallback_inquiry_day,
                        } if self._v29_reservation_accountability() else {}),
                        **(
                            {
                                "preauthorized_recovery_plans": (
                                    preauthorized_plans
                                ),
                                "authorized_line_fallbacks": options,
                                "automatic_selection": False,
                                "candidate_feasibility_evaluated": False,
                            }
                            if self._v24_reservation_operations()
                            else {"authorized_fallbacks": options}
                        ),
                    }
                )
                continue
            if action == "select_preauthorized_plan":
                if not self._v24_reservation_operations():
                    raise ValueError(
                        "select_preauthorized_plan requires tool interface 20"
                    )
                if commitment.fallback_inquiry_day is None:
                    raise ValueError(
                        "inquire_fallback must occur before selecting a recovery plan"
                    )
                plan_id = str(action_row.get("plan_id", ""))
                selected = self._v24_select_preauthorized_recovery_plan(
                    state,
                    request,
                    commitment,
                    plan_id=plan_id,
                )
                results.append(
                    {
                        "commitment_id": commitment_id,
                        "action": action,
                        **selected,
                    }
                )
                continue
            if action == "select_authorized_fallback":
                if commitment.fallback_inquiry_day is None:
                    raise ValueError(
                        "inquire_fallback must occur before selecting a fallback"
                    )
                if (
                    self._v24_reservation_operations()
                    and commitment.recovery_path
                    not in {None, "profile_line_fallback"}
                ):
                    raise ValueError(
                        "complete-plan and line-fallback recovery paths cannot be mixed"
                    )
                offer_id = str(action_row.get("offer_id", ""))
                line_id = str(action_row.get("line_id", ""))
                if not line_id and len(commitment.lines) == 1:
                    line_id = commitment.lines[0].line_id
                authorized = commitment.fallback_revealed_offer_ids_by_line.get(
                    line_id, []
                )
                if (
                    not authorized
                    and not request.recovery_offer_ids_by_line
                    and not (
                        self._v38_proactive_contingency()
                        and commitment.version_recovery_offer_ids_by_line
                    )
                ):
                    authorized = request.fallback_offer_ids_by_line.get(line_id, [])
                if offer_id not in authorized:
                    raise ValueError(
                        f"customer did not authorize fallback for line: {line_id}"
                    )
                for line in commitment.lines:
                    if line.line_id != line_id:
                        continue
                    previous_offer_id = line.selected_offer_id or line.offer_id
                    line.selected_offer_id = offer_id
                    line.fulfillment_mode = "inquired_fallback_selected"
                    if not line.authorized_offer_ids:
                        line.authorized_offer_ids = [line.offer_id]
                    if offer_id not in line.authorized_offer_ids:
                        line.authorized_offer_ids.append(offer_id)
                    for block in state.reservation_calendar_blocks:
                        if (
                            block.commitment_id == commitment_id
                            and block.line_id == line.line_id
                            and block.status in {"active", "bound"}
                        ):
                            if not block.authorized_offer_ids:
                                block.authorized_offer_ids = [block.offer_id]
                            if offer_id not in block.authorized_offer_ids:
                                block.authorized_offer_ids.append(offer_id)
                            block.selected_offer_id = offer_id
                    self._reservation_event(
                        state,
                        "authorized_fallback_selected",
                        request_id=request.request_id,
                        commitment_id=commitment_id,
                        details={
                            "from_offer_id": previous_offer_id,
                            "to_offer_id": offer_id,
                        },
                    )
                if self._v24_reservation_operations():
                    commitment.recovery_path = "profile_line_fallback"
                results.append(
                    {
                        "commitment_id": commitment_id,
                        "action": action,
                        "selected_offer_id": offer_id,
                        "line_id": line_id,
                        "status": "recovery_pending_until_pickup_cutoff",
                    }
                )
                continue
            if action == "cancel_early":
                if self._v29_reservation_accountability():
                    raise ValueError(
                        "use cancel_reservation_commitment for V29 manager "
                        "cancellations"
                    )
                settlement = self._fail_reservation_commitment(
                    state,
                    commitment,
                    status="cancelled_by_manager",
                    cause="early_risk_cancellation",
                    notice_days=max(0, commitment.pickup_day - state.day),
                )
                results.append(
                    {
                        "commitment_id": commitment_id,
                        "action": action,
                        "status": commitment.status,
                        "settlement": settlement,
                    }
                )
                continue
            raise ValueError(
                "action must be inquire_fallback, select_preauthorized_plan, "
                + (
                    "or select_authorized_fallback; use "
                    "cancel_reservation_commitment to cancel"
                    if self._v29_reservation_accountability()
                    else "select_authorized_fallback, or cancel_early"
                )
            )
        return {"actions": results, "cash": state.cash}

    @staticmethod
    def _reservation_capacity_release(
        commitment: ReservationCommitment,
    ) -> dict[str, Any]:
        occupied_through = max(
            commitment.pickup_day,
            commitment.return_day - 1,
        )
        return {
            "pickup_day": int(commitment.pickup_day),
            "occupied_from_day": int(commitment.pickup_day),
            "occupied_through_day": int(occupied_through),
            "scheduled_return_day": int(commitment.return_day),
            "lines": [
                {
                    "line_id": line.line_id,
                    "offer_id": line.selected_offer_id or line.offer_id,
                    "asset_quantity": int(line.quantity),
                    "required": bool(line.required),
                }
                for line in commitment.lines
            ],
        }

    def reservation_cancellation_terms(
        self,
        state: WorldState,
        commitment: ReservationCommitment,
    ) -> dict[str, Any]:
        eligible = bool(
            not state.done
            and commitment.status in {"confirmed", "at_risk"}
            and not commitment.assignments
        )
        result: dict[str, Any] = {
            "available": eligible,
            "commitment_id": commitment.commitment_id,
            "current_status": commitment.status,
        }
        if not eligible:
            result["unavailable_reason"] = "commitment_is_not_confirmed_pre_pickup"
            return result
        terms = self._reservation_failure_terms(
            commitment,
            status="cancelled_by_manager",
            notice_days=max(0, commitment.pickup_day - state.day),
        )
        return {
            **result,
            **terms,
            "capacity_release": self._reservation_capacity_release(commitment),
            "feasibility_evaluated": False,
            "recommendation_included": False,
        }

    def cancel_reservation_commitment(
        self,
        state: WorldState,
        *,
        commitment_id: str,
    ) -> dict[str, Any]:
        if not self._v29_reservation_accountability():
            raise ValueError(
                "cancel_reservation_commitment requires tool interface 25"
            )
        commitment = next(
            (
                row
                for row in state.reservation_commitments
                if row.commitment_id == commitment_id
            ),
            None,
        )
        if commitment is None:
            raise ValueError(f"unknown commitment_id: {commitment_id}")
        preview = self.reservation_cancellation_terms(state, commitment)
        if not preview["available"]:
            raise ValueError(
                "commitment is not a confirmed pre-pickup reservation: "
                f"{commitment_id}"
            )
        previous_status = commitment.status
        origin = "risk_alert" if previous_status == "at_risk" else "proactive"
        settlement = self._fail_reservation_commitment(
            state,
            commitment,
            status="cancelled_by_manager",
            cause=f"{origin}_manager_cancellation",
            notice_days=int(preview["notice_days"]),
        )
        return {
            "commitment_id": commitment_id,
            "previous_status": previous_status,
            "status": commitment.status,
            "cancellation_origin": origin,
            "settlement": settlement,
            "capacity_release": preview["capacity_release"],
            "cash": round(state.cash, 2),
            "business_closed": bool(state.done),
        }

    def _fail_reservation_commitment(
        self,
        state: WorldState,
        commitment: ReservationCommitment,
        *,
        status: str,
        cause: str,
        notice_days: int,
    ) -> dict[str, Any]:
        request = next(
            row
            for row in state.reservation_requests
            if row.request_id == commitment.request_id
        )
        commitment.status = status
        commitment.failure_cause = cause
        request.status = status
        settlement = self._reservation_failure_terms(
            commitment,
            status=status,
            notice_days=notice_days,
        )
        refund = float(settlement["prepayment_refund"])
        state.unearned_revenue_liability = round(
            max(
                0.0,
                state.unearned_revenue_liability
                - refund,
            ),
            2,
        )
        state.current_day_ledger["reservation_prepayment_refunds"] = round(
            float(
                state.current_day_ledger.get(
                    "reservation_prepayment_refunds", 0.0
                )
            )
            + refund,
            2,
        )
        state.cumulative_reservation_prepayment_refunds = round(
            state.cumulative_reservation_prepayment_refunds
            + refund,
            2,
        )
        if status == "pickup_failed":
            state.cumulative_reservation_pickup_failures += 1
        for block in state.reservation_calendar_blocks:
            if block.commitment_id == commitment.commitment_id:
                block.status = "failed" if status == "pickup_failed" else "cancelled"
        compensation = float(settlement["compensation"])
        cancellation_origin = (
            "pickup_failure"
            if status == "pickup_failed"
            else "risk_alert"
            if commitment.risk_exposed_day is not None
            else "proactive"
        )
        self._reservation_event(
            state,
            status,
            request_id=commitment.request_id,
            commitment_id=commitment.commitment_id,
            details={
                "cause": cause,
                "notice_days": notice_days,
                "prepayment_refund": refund,
                "compensation": compensation,
                **(
                    {
                        "cancellation_origin": cancellation_origin,
                        "notice_band": settlement["notice_band"],
                        "current_locked_rent": settlement[
                            "current_locked_rent"
                        ],
                        "required_service_unit_days": settlement[
                            "required_service_unit_days"
                        ],
                        "rent_component": settlement["rent_component"],
                        "service_unit_day_floor_component": settlement[
                            "service_unit_day_floor_component"
                        ],
                        "total_cash_outflow": settlement[
                            "total_cash_outflow"
                        ],
                        "settlement_formula": settlement["formula"],
                    }
                    if self._v29_reservation_accountability()
                    else {}
                ),
            },
        )
        if self._v26_contract_state():
            commitment.prepayment_refunded = round(
                commitment.prepayment_refunded + refund, 2
            )
            commitment.outstanding_rent = 0.0
            commitment.unearned_revenue = 0.0
        self._assert_v26_reservation_financial_invariants(state)
        if not self._refund(state, refund, commitment.commitment_id):
            payment = {}
            if self._v48_service_guarantee():
                payment = self._record_guarantee_payment(state, commitment, kind='whole_failure',
                    line_ids=self._guarantee(commitment)['basis_by_line_cents'], notice_days=notice_days, paid=0.)
                for event in reversed(state.reservation_events):
                    if event.commitment_id==commitment.commitment_id and event.event_type==status:
                        event.details.update(payment);break
            return {**settlement, **payment, "refund_paid": False, "compensation_paid": False}
        paid = True
        actual_compensation_paid = 0.0
        if compensation > 0:
            state.current_day_ledger["reservation_compensation"] = round(
                float(
                    state.current_day_ledger.get(
                        "reservation_compensation", 0.0
                    )
                )
                + compensation,
                2,
            )
            state.cumulative_reservation_compensation = round(
                state.cumulative_reservation_compensation + compensation, 2
            )
            if (
                compensation > state.cash + 1e-9
                and self.config.emergency_credit_limit > 0
            ):
                self._draw_customer_protection_credit(state, compensation)
            cash_before_compensation = state.cash
            compensation_paid = self._charge(
                state,
                compensation,
                reason=f"预约 {commitment.commitment_id} 履约补偿",
            )
            actual_compensation_paid = round(cash_before_compensation - state.cash, 2)
            if self._v26_contract_state() and compensation_paid and not self._v48_service_guarantee():
                commitment.compensation_paid = round(
                    commitment.compensation_paid + compensation, 2
                )
            paid = bool(compensation_paid)
        payment = {}
        if self._v48_service_guarantee():
            payment = self._record_guarantee_payment(state, commitment, kind='whole_failure',
                line_ids=self._guarantee(commitment)['basis_by_line_cents'], notice_days=notice_days, paid=actual_compensation_paid)
            for event in reversed(state.reservation_events):
                if event.commitment_id==commitment.commitment_id and event.event_type==status:
                    event.details.update(payment);break
        return {
            **settlement,
            **payment,
            "refund_paid": True,
            "compensation_paid": paid,
        }

    def _pickup_asset_selection(self, state, commitment):
        """Side-effect-free match for one pickup using the existing allocation order.

        No assets are reserved and other commitments are not simulated. The
        close-time caller applies this separately in the original request order.
        """
        chosen = {}
        reserved_asset_ids = set()
        optional_missing = []
        for line in commitment.lines:
            if line.selected_offer_id is not None:
                candidates = [line.selected_offer_id]
            else:
                authorized = list(line.authorized_offer_ids or [line.offer_id])
                candidates = ([line.staged_offer_id] + [o for o in authorized if o != line.staged_offer_id]
                              if line.staged_offer_id is not None else authorized)
            selection = None
            for offer_id in candidates:
                assets = sorted(
                    (a for a in state.assets if a.offer_id == offer_id and a.status == "available"
                     and (not self._v31_warehouse_capacity() or self._warehouse_asset_held(a))
                     and a.asset_id not in reserved_asset_ids
                     and self._supplier_trial_asset_can_start_contract(state, a, return_day=commitment.return_day)),
                    key=lambda a: (-self._used_life_fraction(a), a.asset_id))
                if len(assets) >= line.quantity:
                    selection = (offer_id, assets[:line.quantity])
                    break
            if selection is None:
                if line.required:
                    return chosen, False, optional_missing
                optional_missing.append(line)
                chosen[line.line_id] = (line.offer_id, [])
                continue
            chosen[line.line_id] = selection
            reserved_asset_ids.update(a.asset_id for a in selection[1])
        return chosen, True, optional_missing

    def _process_reservation_pickups(
        self, state: WorldState, *, finalize_risks: bool = False
    ) -> None:
        if not self.config.reservations_enabled or state.done:
            return
        requests = {row.request_id: row for row in state.reservation_requests}
        for commitment in sorted(state.reservation_commitments, key=lambda row: row.request_id):
            if commitment.status not in {"confirmed", "at_risk"} or commitment.pickup_day > state.day:
                continue
            chosen, possible, optional_missing = self._pickup_asset_selection(state, commitment)
            if self._v45_reservation_difficulty() and possible and optional_missing:
                missing_ids = [line.line_id for line in optional_missing]
                if (
                    not commitment.partial_fulfillment_selected
                    or missing_ids != commitment.partial_fulfillment_missing_line_ids
                ):
                    if not self._v54_recovery_state() or commitment.partial_fulfillment_inquiry_day is None:
                        commitment.partial_fulfillment_missing_line_ids = missing_ids
                    possible = False
            if not possible:
                if (
                    self.config.reservation_disruptions_enabled
                    and not finalize_risks
                ):
                    if commitment.status != "at_risk":
                        commitment.status = "at_risk"
                        commitment.risk_exposed_day = state.day
                        self._reservation_event(
                            state,
                            "pickup_risk_exposed",
                            request_id=commitment.request_id,
                            commitment_id=commitment.commitment_id,
                            details={
                                "recovery_deadline": state.day,
                                "pickup_day": commitment.pickup_day,
                            },
                        )
                        state.inbox.append(
                            f"预约 {commitment.commitment_id} 今日取货存在履约风险；"
                            + ("可在日终前查询客户 fallback、协商可选配件减配或提前取消。"
                               if self._v45_reservation_difficulty()
                               else "可在日终前查询客户 fallback 或提前取消。")
                        )
                    continue
                self._fail_reservation_commitment(
                    state,
                    commitment,
                    status="pickup_failed",
                    cause="unresolved_authorized_offer_assignment",
                    notice_days=0,
                )
                continue
            optional_credit = round(
                sum(
                    line.locked_rent
                    * float(self.config.optional_line_missing_credit_rate)
                    for line in optional_missing
                ),
                2,
            )
            for line in optional_missing:
                line.status = "optional_missing_credited"
                line.credit_amount = round(
                    line.locked_rent
                    * float(self.config.optional_line_missing_credit_rate),
                    2,
                )
            commitment.optional_line_credits = optional_credit
            state.cumulative_optional_line_credits = round(
                state.cumulative_optional_line_credits + optional_credit, 2
            )
            for block in state.reservation_calendar_blocks:
                if (
                    block.commitment_id == commitment.commitment_id
                    and block.status in {"active", "bound"}
                    and any(
                        line.line_id == block.line_id
                        for line in optional_missing
                    )
                ):
                    block.status = "optional_credited"
            deposit_total = 0.0
            trial_contracts: list[tuple[Any, RentalContract]] = []
            for line in commitment.lines:
                selected_offer_id, selected_assets = chosen[line.line_id]
                if not selected_assets:
                    continue
                line.selected_offer_id = selected_offer_id
                if (
                    line.staged_offer_id is not None
                    and selected_offer_id != line.staged_offer_id
                ):
                    state.cumulative_reservation_staged_fallbacks += 1
                    self._reservation_event(
                        state,
                        "staged_offer_unavailable_authorized_fallback_used",
                        request_id=commitment.request_id,
                        commitment_id=commitment.commitment_id,
                        details={
                            "line_id": line.line_id,
                            "staged_offer_id": line.staged_offer_id,
                            "selected_offer_id": selected_offer_id,
                        },
                    )
                if line.fulfillment_mode == "inquired_fallback_selected":
                    line.fulfillment_mode = "inquired_fallback_fulfilled"
                elif selected_offer_id == line.offer_id:
                    line.fulfillment_mode = "primary_fulfilled"
                else:
                    line.fulfillment_mode = (
                        "booking_authorized_substitute_fulfilled"
                    )
                locked_line_deposit = (
                    float(line.locked_security_deposit)
                    if line.locked_security_deposit > 0
                    or line.recovery_source_plan_id is not None
                    else float(state.deposits[selected_offer_id]) * line.quantity
                )
                allocated_deposit = 0.0
                allocated_rent = 0.0
                exact_rents = exact_deposits = None
                if self._v32_service_commitments():
                    selected_assets = sorted(selected_assets, key=lambda a: a.asset_id)
                    exact_rents = split_cents(line.locked_rent, len(selected_assets))
                    exact_deposits = split_cents(locked_line_deposit, len(selected_assets))
                for index, asset in enumerate(selected_assets, start=1):
                    deposit = (
                        exact_deposits[index-1] if exact_deposits is not None
                        else round(locked_line_deposit - allocated_deposit, 2)
                        if index == len(selected_assets)
                        else round(locked_line_deposit / line.quantity, 2)
                    )
                    allocated_deposit = round(allocated_deposit + deposit, 2)
                    deposit_total += deposit
                    contract_rent = (
                        exact_rents[index-1] if exact_rents is not None
                        else round(line.locked_rent - allocated_rent, 2)
                        if index == len(selected_assets)
                        else round(line.locked_rent / line.quantity, 2)
                    )
                    allocated_rent = round(
                        allocated_rent + contract_rent, 2
                    )
                    damage_rng = self._stable_event_rng(
                        _RNG_REQUEST_DAMAGE,
                        *requests[commitment.request_id].event_address,
                        identity=(
                            f"{self._reservation_commitment_rng_identity(commitment.request_id)}:"
                            f"{self._reservation_line_rng_identity(commitment.request_id, line.line_id)}:"
                            f"{asset.asset_id}"
                        ),
                    )
                    event_uniform = float(damage_rng.random())
                    lifecycle_damage: dict[str, float | bool] | None = None
                    severity_uniform: float | None = None
                    if self.config.lifecycle_economics_enabled:
                        severity_uniform = float(
                            self._stable_event_rng(
                                _RNG_REQUEST_DAMAGE_SEVERITY,
                                *requests[commitment.request_id].event_address,
                                identity=(
                                    f"{self._reservation_commitment_rng_identity(commitment.request_id)}:"
                                    f"{self._reservation_line_rng_identity(commitment.request_id, line.line_id)}:"
                                    f"{asset.asset_id}"
                                ),
                            ).random()
                        )
                        lifecycle_damage = self._lifecycle_damage_outcome(
                            asset,
                            start_day=state.day,
                            return_day=commitment.return_day,
                            event_uniform=event_uniform,
                            severity_uniform=severity_uniform,
                        )
                        damaged = bool(lifecycle_damage["damaged"])
                    else:
                        rate = DAMAGE_RATES[asset.damage_risk_band]
                        damaged = bool(
                            event_uniform
                            < self._rental_damage_probability(
                                rate, state.day, commitment.return_day
                            )
                        )
                    asset.status = "rented"
                    asset.used_life_days += line.duration_days
                    asset.used_rental_cycles += 1
                    contract = RentalContract(
                        contract_id=f"rental-{state.next_contract_id:06d}",
                        asset_id=asset.asset_id,
                        offer_id=selected_offer_id,
                        service_line=asset.service_line,
                        start_day=state.day,
                        return_day=commitment.return_day,
                        duration_days=line.duration_days,
                        rent_paid=contract_rent,
                        deposit=deposit,
                        damaged_on_return=damaged,
                        reservation_id=commitment.commitment_id,
                        promised_return_day=commitment.promised_return_day,
                        approved_return_day=commitment.approved_return_day,
                        estimated_return_day=commitment.estimated_return_day,
                        damage_event_uniform=(
                            event_uniform
                            if self.config.lifecycle_economics_enabled
                            else None
                        ),
                        damage_severity_uniform=severity_uniform,
                        damage_condition_loss=(
                            float(lifecycle_damage["condition_loss"])
                            if lifecycle_damage is not None and damaged
                            else None
                        ),
                        damage_wear_at_start=(
                            float(lifecycle_damage["wear_at_start"])
                            if lifecycle_damage is not None
                            else None
                        ),
                        damage_condition_at_start=(
                            float(lifecycle_damage["condition_at_start"])
                            if lifecycle_damage is not None
                            else None
                        ),
                        damage_maintenance_exposure_at_start=(
                            float(
                                lifecycle_damage[
                                    "maintenance_exposure_at_start"
                                ]
                            )
                            if lifecycle_damage is not None
                            else None
                        ),
                    )
                    state.contracts.append(contract)
                    state.next_contract_id += 1
                    if asset.ownership_type == "supplier_trial":
                        contract.supplier_trial_id = asset.supplier_trial_id
                        trial_contracts.append((asset, contract))
                    assignment = ReservationAssignment(
                        assignment_id=(
                            f"{commitment.commitment_id}-{line.line_id}-"
                            f"{index:02d}"
                        ),
                        commitment_id=commitment.commitment_id,
                        line_id=line.line_id,
                        asset_id=asset.asset_id,
                        assigned_day=state.day,
                    )
                    commitment.assignments.append(assignment)
                    line.status = "fulfilled"
                    self._record_rental_activity(
                        state,
                        "rental_started",
                        asset,
                        contract_id=contract.contract_id,
                        reservation_id=commitment.commitment_id,
                        duration_days=line.duration_days,
                        return_day=contract.return_day,
                        rent_paid=contract.rent_paid,
                        deposit=deposit,
                        fulfilled_demand_units=line.customer_units,
                    )
            recognized_rent = round(
                commitment.total_locked_rent - optional_credit, 2
            )
            credited_prepayment = (
                float(commitment.unearned_revenue)
                if self._v26_contract_state()
                else float(commitment.prepayment_collected)
            )
            remaining_rent = round(
                max(0.0, recognized_rent - credited_prepayment), 2
            )
            prepayment_refund = round(
                max(0.0, credited_prepayment - recognized_rent), 2
            )
            state.cash = round(state.cash + remaining_rent + deposit_total, 2)
            state.deposit_liability = round(
                state.deposit_liability + deposit_total, 2
            )
            state.unearned_revenue_liability = round(
                max(
                    0.0,
                    state.unearned_revenue_liability
                    - credited_prepayment,
                ),
                2,
            )
            state.cumulative_rental_revenue = round(
                state.cumulative_rental_revenue + recognized_rent,
                2,
            )
            state.current_day_ledger["reservation_prepayment_recognized"] = round(
                float(
                    state.current_day_ledger.get(
                        "reservation_prepayment_recognized", 0.0
                    )
                )
                + min(credited_prepayment, recognized_rent),
                2,
            )
            state.cumulative_deposits_collected = round(
                state.cumulative_deposits_collected + deposit_total, 2
            )
            self._add(
                state.current_day_ledger,
                "rental_revenue",
                recognized_rent,
            )
            self._add(
                state.current_day_ledger,
                "deposits_collected",
                deposit_total,
            )
            if optional_missing and self._v45_reservation_difficulty():
                compensation = commitment.partial_fulfillment_compensation
                state.current_day_ledger["reservation_compensation"] = round(
                    float(state.current_day_ledger.get("reservation_compensation", 0.0))
                    + compensation, 2)
                state.cumulative_reservation_compensation = round(
                    state.cumulative_reservation_compensation + compensation, 2)
                if compensation > state.cash + 1e-9 and self.config.emergency_credit_limit > 0:
                    self._draw_customer_protection_credit(state, compensation)
                cash_before_compensation = state.cash
                compensation_paid = self._charge(
                    state, compensation, reason=f"预约 {commitment.commitment_id} 减配补偿")
                if self._v48_service_guarantee():
                    self._record_guarantee_payment(state, commitment, kind='partial_pickup',
                        line_ids=commitment.partial_fulfillment_missing_line_ids, notice_days=0,
                        paid=round(cash_before_compensation-state.cash,2))
                if (int(self.dataset.metadata.get("tool_interface_version", 0)) >= 44
                        and "_close_payments" in state.current_day_ledger):
                    state.current_day_ledger["_close_payments"].setdefault(
                        commitment.commitment_id, {}).update(
                            compensation=compensation,
                            compensation_paid=bool(compensation_paid),
                            compensation_amount_paid=round(
                                cash_before_compensation - state.cash, 2))
                if not compensation_paid:
                    return
                if not self._v48_service_guarantee():
                    commitment.compensation_paid = round(
                        commitment.compensation_paid + compensation, 2)
            for asset, contract in trial_contracts:
                trial_fee = self._start_supplier_trial_contract(
                    state, asset, return_day=contract.return_day
                )
                contract.supplier_trial_usage_fee = round(
                    float(trial_fee or 0.0), 2
                )
                if trial_fee is None:
                    return
            for line in commitment.lines:
                selected_offer_id = line.selected_offer_id or line.offer_id
                line_deposit = round(float(line.locked_security_deposit), 2)
                if (
                    line_deposit <= 0
                    and line.recovery_source_plan_id is None
                ):
                    line_deposit = round(
                        float(state.deposits[selected_offer_id]) * line.quantity,
                        2,
                    )
                if line in optional_missing:
                    self._record_attempt(
                        state,
                        self.service_key_for_product_type(line.service_line),
                        "stockout",
                        product_type=line.service_line,
                        offer_id=line.offer_id,
                        quantity=line.customer_units,
                    )
                    state.current_day_ledger["demand"][
                        "requested_line_items"
                    ] += 1
                    state.current_day_ledger["demand"]["unfilled_line_items"] += 1
                    state.cumulative_requested_line_items += 1
                    state.cumulative_unfilled_line_items += 1
                    continue
                self._record_attempt(
                    state,
                    self.service_key_for_product_type(line.service_line),
                    "fulfilled",
                    product_type=self.offers[selected_offer_id].service_line,
                    offer_id=selected_offer_id,
                    revenue=line.locked_rent,
                    deposit=line_deposit,
                    rental_days=line.duration_days * line.quantity,
                    quantity=line.customer_units,
                    channel="reservation",
                )
                state.current_day_ledger["demand"]["requested_line_items"] += 1
                state.current_day_ledger["demand"]["fulfilled_line_items"] += 1
                state.cumulative_requested_line_items += 1
                state.cumulative_fulfilled_line_items += 1
            basket_key = "partial_baskets" if optional_missing else "full_baskets"
            state.current_day_ledger["demand"][basket_key] += 1
            state.current_day_ledger["demand"][
                "operationally_evaluable_baskets"
            ] += 1
            operational_key = (
                "operational_partial_baskets"
                if optional_missing
                else "operational_full_baskets"
            )
            state.current_day_ledger["demand"][operational_key] += 1
            state.cumulative_transactions += 1
            if optional_missing:
                state.cumulative_partial_baskets += 1
                state.cumulative_operational_partial_baskets += 1
            else:
                state.cumulative_full_baskets += 1
                state.cumulative_operational_full_baskets += 1
            state.cumulative_operationally_evaluable_baskets += 1
            if prepayment_refund > 0:
                state.current_day_ledger[
                    "reservation_prepayment_refunds"
                ] = round(
                    float(
                        state.current_day_ledger.get(
                            "reservation_prepayment_refunds", 0.0
                        )
                    )
                    + prepayment_refund,
                    2,
                )
                state.cumulative_reservation_prepayment_refunds = round(
                    state.cumulative_reservation_prepayment_refunds
                    + prepayment_refund,
                    2,
                )
                if self._v26_contract_state():
                    commitment.prepayment_refunded = round(
                        commitment.prepayment_refunded
                        + prepayment_refund,
                        2,
                    )
                if not self._refund(
                    state, prepayment_refund, commitment.commitment_id
                ) and not self._v26_contract_state():
                    return
            commitment.unearned_revenue = 0.0
            if self._v26_contract_state():
                commitment.rent_due_at_pickup = remaining_rent
                commitment.outstanding_rent = 0.0
                if self._v27_extension_cash_audit():
                    commitment.rent_collected_at_pickup = remaining_rent
            else:
                commitment.outstanding_rent = remaining_rent
            commitment.status = "active"
            requests[commitment.request_id].status = "active"
            self._assert_v26_reservation_financial_invariants(state)
            state.cumulative_reservation_pickups += 1
            lines_by_id = {line.line_id: line for line in commitment.lines}
            for block in state.reservation_calendar_blocks:
                if (
                    block.commitment_id == commitment.commitment_id
                    and block.status in {"active", "bound"}
                    and block.line_id in lines_by_id
                ):
                    block.status = "bound"
                    block.selected_offer_id = lines_by_id[
                        block.line_id
                    ].selected_offer_id
            self._reservation_event(
                state,
                "pickup_fulfilled",
                request_id=commitment.request_id,
                commitment_id=commitment.commitment_id,
                details={
                    "assignments": [
                        assignment.asset_id for assignment in commitment.assignments
                    ],
                    "remaining_rent_collected": remaining_rent,
                    "deposits_collected": round(deposit_total, 2),
                    "optional_line_credit": optional_credit,
                    **({"partial_fulfillment_compensation":
                        commitment.partial_fulfillment_compensation}
                       if self._v45_reservation_difficulty() and optional_missing else {}),
                    **({"compensation": commitment.partial_fulfillment_compensation,
                        "compensation_paid": True,
                        "compensation_amount_paid": commitment.partial_fulfillment_compensation}
                       if optional_missing and int(self.dataset.metadata.get("tool_interface_version", 0)) >= 44
                       else {}),
                    "selected_offers": {
                        line.line_id: line.selected_offer_id
                        for line in commitment.lines
                        if line.selected_offer_id is not None
                    },
                    "fulfillment_modes": {
                        line.line_id: line.fulfillment_mode
                        for line in commitment.lines
                        if line.fulfillment_mode is not None
                    },
                },
            )

    def _finalize_reservation_risks(self, state: WorldState) -> None:
        if not self.config.reservation_disruptions_enabled or state.done:
            return
        self._process_reservation_pickups(state, finalize_risks=True)

    def _update_reservation_after_contract_return(
        self, state: WorldState, contract: RentalContract
    ) -> None:
        if not contract.reservation_id:
            return
        commitment = next(
            (
                row
                for row in state.reservation_commitments
                if row.commitment_id == contract.reservation_id
            ),
            None,
        )
        if commitment is None:
            return
        related = [
            row
            for row in state.contracts
            if row.reservation_id == commitment.commitment_id
        ]
        if related and all(row.status == "returned" for row in related):
            commitment.status = "completed"
            if self._v26_contract_state():
                commitment.outstanding_rent = 0.0
                commitment.unearned_revenue = 0.0
            commitment.actual_return_day = max(
                int(row.actual_return_day or row.return_day) for row in related
            )
            for assignment in commitment.assignments:
                assignment.status = "completed"
            for block in state.reservation_calendar_blocks:
                if block.commitment_id == commitment.commitment_id:
                    block.status = "completed"
            request = next(
                row
                for row in state.reservation_requests
                if row.request_id == commitment.request_id
            )
            request.status = "completed"
            self._assert_v26_reservation_financial_invariants(state)
            self._reservation_event(
                state,
                "reservation_completed",
                request_id=request.request_id,
                commitment_id=commitment.commitment_id,
                details={
                    "promised_return_day": commitment.promised_return_day,
                    "approved_return_day": commitment.approved_return_day,
                    "estimated_return_day": commitment.estimated_return_day,
                    "actual_return_day": commitment.actual_return_day,
                },
            )
