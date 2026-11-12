"""V32 opt-in future-service terms. Legacy behavior delegates unchanged."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from dataclasses import replace

from ..service_commitments import ENVIRONMENT_PROTOCOL, POLICY_ID, RNG_NAMESPACE, cents, split_cents
from ..reservation_difficulty import rental_amount, ASSET_PRICING_BASIS, LINE_PRICING_BASIS
from ..wide_date_flexibility import (
    DATE_OFFSETS as WIDE_DATE_OFFSETS,
    ENVIRONMENT_PROTOCOL as WIDE_DATE_ENVIRONMENT_PROTOCOL,
)
from ..capital_balance_tuning import (
    ENVIRONMENT_PROTOCOL as CAPITAL_BALANCE_ENVIRONMENT_PROTOCOL,
)
from ..assortment import ENVIRONMENT_PROTOCOL as OPTIONAL_RETAIL_ENVIRONMENT_PROTOCOL
from ..acquisition_budget import ENVIRONMENT_PROTOCOL as DURABLE_BUDGET_ENVIRONMENT_PROTOCOL
from ..supplier_trial import ENVIRONMENT_PROTOCOL as SUPPLIER_TRIAL_ENVIRONMENT_PROTOCOL
from ..reservation_contingency import (
    ENVIRONMENT_PROTOCOL as PROACTIVE_CONTINGENCY_ENVIRONMENT_PROTOCOL,
    SELECTED_PARAMETERS as CONTINGENCY_PARAMETERS,
    customer_class as contingency_customer_class,
    date_utility_delta as contingency_date_utility_delta,
    initial_date_consideration_probability,
    initial_date_is_acceptable,
)
from ..contingency_actor_interface import (
    ENVIRONMENT_PROTOCOL as CONTINGENCY_ACTOR_INTERFACE_ENVIRONMENT_PROTOCOL,
)
from ..revolving_durable_capital import (
    ENVIRONMENT_PROTOCOL as REVOLVING_DURABLE_CAPITAL_ENVIRONMENT_PROTOCOL,
)
from ..actor_interface_slimming import (
    ENVIRONMENT_PROTOCOL as ACTOR_INTERFACE_SLIMMING_ENVIRONMENT_PROTOCOL,
)
from ..weather_information import (
    ENVIRONMENT_PROTOCOL as WEATHER_INFORMATION_ENVIRONMENT_PROTOCOL,
    forecast_regime,
)


LEGACY_DATE_OFFSETS = [-3, -2, -1, 1, 2, 3]
from ..actor_manual import ENVIRONMENT_PROTOCOL as ACTOR_MANUAL_ENVIRONMENT_PROTOCOL
from ..trip_weather import ENVIRONMENT_PROTOCOL as TRIP_WEATHER_ENVIRONMENT_PROTOCOL, contextualize, public_coverage
from ..reservation_difficulty import ENVIRONMENT_PROTOCOL as RESERVATION_DIFFICULTY_ENVIRONMENT_PROTOCOL
from ..return_dynamics import ENVIRONMENT_PROTOCOL as RETURN_DYNAMICS_ENVIRONMENT_PROTOCOL
from ..autonomous_rentals import ENVIRONMENT_PROTOCOL as AUTONOMOUS_RENTALS_ENVIRONMENT_PROTOCOL
from ..service_guarantee import ENVIRONMENT_PROTOCOL as SERVICE_GUARANTEE_ENVIRONMENT_PROTOCOL
from ..school_holiday import ENVIRONMENT_PROTOCOL as SCHOOL_HOLIDAY_ENVIRONMENT_PROTOCOL
from ..reservation_overlap import ENVIRONMENT_PROTOCOL as OVERLAP_ENVIRONMENT_PROTOCOL
from ..resale_disclosure import ENVIRONMENT_PROTOCOL as V52_ENVIRONMENT_PROTOCOL
from ..operating_history import ENVIRONMENT_PROTOCOL as V53_ENVIRONMENT_PROTOCOL
from ..recovery_state import ENVIRONMENT_PROTOCOL as V54_ENVIRONMENT_PROTOCOL
from ..summer_pressure import ENVIRONMENT_PROTOCOL as SUMMER_PRESSURE_ENVIRONMENT_PROTOCOL
from ..reservation_choice import CompleteContractOption, evaluate_complete_contract_choice, score_complete_store_contract


class ServiceCommitmentsMixin:
    def _v32_service_commitments(self):
        return (self.dataset.metadata.get("environment_protocol") in {
                    ENVIRONMENT_PROTOCOL, WIDE_DATE_ENVIRONMENT_PROTOCOL,
                    CAPITAL_BALANCE_ENVIRONMENT_PROTOCOL,
                    OPTIONAL_RETAIL_ENVIRONMENT_PROTOCOL,
                    DURABLE_BUDGET_ENVIRONMENT_PROTOCOL,
                    SUPPLIER_TRIAL_ENVIRONMENT_PROTOCOL,
                    PROACTIVE_CONTINGENCY_ENVIRONMENT_PROTOCOL,
                    CONTINGENCY_ACTOR_INTERFACE_ENVIRONMENT_PROTOCOL,
                    REVOLVING_DURABLE_CAPITAL_ENVIRONMENT_PROTOCOL,
                    ACTOR_INTERFACE_SLIMMING_ENVIRONMENT_PROTOCOL,
                    WEATHER_INFORMATION_ENVIRONMENT_PROTOCOL,
                    ACTOR_MANUAL_ENVIRONMENT_PROTOCOL,
                    TRIP_WEATHER_ENVIRONMENT_PROTOCOL,
                    RESERVATION_DIFFICULTY_ENVIRONMENT_PROTOCOL,
                    RETURN_DYNAMICS_ENVIRONMENT_PROTOCOL,
                    AUTONOMOUS_RENTALS_ENVIRONMENT_PROTOCOL,
                    SERVICE_GUARANTEE_ENVIRONMENT_PROTOCOL,
                    SCHOOL_HOLIDAY_ENVIRONMENT_PROTOCOL,
                    SUMMER_PRESSURE_ENVIRONMENT_PROTOCOL,
                    OVERLAP_ENVIRONMENT_PROTOCOL,
                    V52_ENVIRONMENT_PROTOCOL,
                    V53_ENVIRONMENT_PROTOCOL,
                    V54_ENVIRONMENT_PROTOCOL,
                }
                and int(self.dataset.metadata.get("tool_interface_version", 0)) >= 28)

    def _validate_service_commitments_configuration(self):
        switches = (self.config.reservation_reschedule_outcomes_enabled,
                    self.config.reservation_date_flexibility_enabled,
                    self.config.reservation_service_offers_enabled)
        if any(type(v) is not bool for v in switches):
            raise ValueError("service commitment switches must be booleans")
        enabled = any(switches)
        if enabled and not self._v32_service_commitments():
                raise ValueError("service commitment mechanisms require an independent V32+ runtime")
        if self._v32_service_commitments():
            policy = self.dataset.metadata.get("service_commitment_policy", {})
            if policy.get("format") != POLICY_ID or not self._v31_warehouse_capacity():
                raise ValueError("V32 requires frozen service and warehouse policies")
            if self.config.reservation_contract_mode != "factorized_line_options_plan_set_v1":
                raise ValueError("V32 requires factorized line quotations")
            probability_keys = [
                "service_offer_attempt_probability",
                "original_dates_unavailable_probability",
            ]
            if not self._v38_proactive_contingency():
                probability_keys.append("date_flexibility_probability")
            for key in probability_keys:
                value = policy.get(key)
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"invalid frozen probability: {key}")
            expected_offsets = (WIDE_DATE_OFFSETS
                if self.dataset.metadata.get("environment_protocol") in {
                    WIDE_DATE_ENVIRONMENT_PROTOCOL, CAPITAL_BALANCE_ENVIRONMENT_PROTOCOL,
                    OPTIONAL_RETAIL_ENVIRONMENT_PROTOCOL,
                    DURABLE_BUDGET_ENVIRONMENT_PROTOCOL,
                    SUPPLIER_TRIAL_ENVIRONMENT_PROTOCOL,
                    PROACTIVE_CONTINGENCY_ENVIRONMENT_PROTOCOL,
                    CONTINGENCY_ACTOR_INTERFACE_ENVIRONMENT_PROTOCOL,
                    REVOLVING_DURABLE_CAPITAL_ENVIRONMENT_PROTOCOL,
                    ACTOR_INTERFACE_SLIMMING_ENVIRONMENT_PROTOCOL,
                    WEATHER_INFORMATION_ENVIRONMENT_PROTOCOL,
                    ACTOR_MANUAL_ENVIRONMENT_PROTOCOL,
                    TRIP_WEATHER_ENVIRONMENT_PROTOCOL,
                    RESERVATION_DIFFICULTY_ENVIRONMENT_PROTOCOL,
                    RETURN_DYNAMICS_ENVIRONMENT_PROTOCOL,
                    AUTONOMOUS_RENTALS_ENVIRONMENT_PROTOCOL,
                    SERVICE_GUARANTEE_ENVIRONMENT_PROTOCOL,
                    SCHOOL_HOLIDAY_ENVIRONMENT_PROTOCOL,
                    SUMMER_PRESSURE_ENVIRONMENT_PROTOCOL,
                    OVERLAP_ENVIRONMENT_PROTOCOL,
                    V52_ENVIRONMENT_PROTOCOL,
                    V53_ENVIRONMENT_PROTOCOL,
                    V54_ENVIRONMENT_PROTOCOL,
                } else LEGACY_DATE_OFFSETS)
            if policy.get("date_offsets") != expected_offsets or policy.get("maximum_complete_plans") != 3:
                raise ValueError("service commitment runtime requires its versioned bounded date and complete-plan space")
            factors = policy.get("market_factors", [])
            if len(factors) != 3 or any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in factors):
                raise ValueError("invalid frozen market factors")
            for key in ("purchase_price_fraction_per_day", "one_day_cleaning_fraction"):
                x = policy.get(key)
                if type(x) not in (int,float) or not math.isfinite(x) or x<=0:
                    raise ValueError(f"invalid frozen reference coefficient: {key}")

    def _service_rng(self, request, identity):
        return self._stable_event_rng(RNG_NAMESPACE, *request.event_address, identity=identity)

    def _v38_initial_alternative_date(self, state, request, original, valid):
        if not valid:
            return None
        primary = next((line for line in request.lines if line.required), request.lines[0])
        session = self._restore_quote_context(
            request.quote_context_by_line[primary.line_id]
        )
        customer_type = contingency_customer_class(
            children=session.children, archetype=session.archetype
        )
        forecasts = {
            int(row["day"]): row
            for row in self.weather_forecast(state, max([original, *valid]) - state.day + 1)
        }
        original_forecast = forecasts[original]
        scored = []
        for day in valid:
            forecast = forecasts[day]
            delta = contingency_date_utility_delta(
                customer_type=customer_type,
                current_day_type=str(self._calendar(original)["day_type"]),
                candidate_day_type=str(self._calendar(day)["day_type"]),
                current_is_school_holiday=bool(self._calendar(original).get('is_school_holiday', False)),
                candidate_is_school_holiday=bool(self._calendar(day).get('is_school_holiday', False)),
                current_condition=(
                    forecast_regime(original_forecast)
                    if self._v42_weather_information_coherence()
                    else str(original_forecast["condition"])
                ),
                candidate_condition=(
                    forecast_regime(forecast)
                    if self._v42_weather_information_coherence()
                    else str(forecast["condition"])
                ),
                shift_days=day - original,
                schedule_flexibility=request.schedule_flexibility,
                forecast_trust=request.forecast_trust,
                occasion_importance=session.occasion_importance,
            )
            if initial_date_is_acceptable(
                delta, request.schedule_flexibility
            ):
                scored.append((int(day), float(delta)))
        if not scored:
            return None
        rng = self._service_rng(request, "initial-date-flexibility-v38")
        if rng.random() >= initial_date_consideration_probability(
            request.schedule_flexibility
        ):
            return None
        temperature = float(CONTINGENCY_PARAMETERS["initial_sampling_temperature"])
        maximum = max(score for _, score in scored)
        weights = [math.exp(temperature * (score - maximum)) for _, score in scored]
        total = sum(weights)
        draw = float(rng.random()) * total
        cumulative = 0.0
        for (day, _), weight in zip(scored, weights):
            cumulative += weight
            if draw <= cumulative:
                return day
        return scored[-1][0]

    def _generate_reservation_requests(self, state):
        known = {r.request_id for r in state.reservation_requests}
        super()._generate_reservation_requests(state)
        if not self._v32_service_commitments():
            return
        policy = self.dataset.metadata["service_commitment_policy"]
        for request in state.reservation_requests:
            if request.request_id in known:
                continue
            original = request.pickup_day
            duration = request.return_day - original
            dates = [original]
            valid = [original + offset for offset in policy["date_offsets"]
                     if self.config.reservation_booking_lead_min_days <= original + offset - state.day
                     <= self.config.reservation_booking_lead_max_days
                     and state.day < original + offset <= self.config.max_days
                     and original + offset + duration <= state.max_days]
            rng = self._service_rng(request, "initial-date-flexibility")
            if self.config.reservation_date_flexibility_enabled and valid:
                if self._v38_proactive_contingency():
                    alternative = self._v38_initial_alternative_date(
                        state, request, original, valid
                    )
                    if alternative is not None:
                        dates.append(alternative)
                elif rng.random() < policy["date_flexibility_probability"]:
                    dates.append(int(rng.choice(valid)))
            forecasts = {int(row["day"]): row for row in self.weather_forecast(state, max(dates)-state.day+1)}
            for day in sorted(dates):
                forecast = forecasts[day]
                request.pickup_date_options.append({"pickup_day": day, "scheduled_return_day": day+duration,
                    "duration_days": duration, "booking_lead_days": day-state.day,
                    "forecast_condition": str(forecast["condition"]),
                    "forecast_accuracy": float(forecast["condition_accuracy"]),
                    **({
                        "forecast_issued_day": int(state.day),
                        "forecast_temp_mean_c": float(forecast["temp_mean_c"]),
                        "forecast_night_low_c": float(forecast["night_low_c"]),
                        "forecast_day_high_c": float(forecast["day_high_c"]),
                    } if self._v42_weather_information_coherence() else {})})
                request.quote_context_by_date[str(day)] = {
                    lid: {**copy.deepcopy(context), "pickup_day": day, "booking_lead_days": day-state.day,
                          "forecast_adversity": self._forecast_adversity(
                              forecast_regime(forecast)
                              if self._v42_weather_information_coherence()
                              else str(forecast["condition"])),
                          "forecast_uncertainty": max(0., 1.-float(forecast["condition_accuracy"]))}
                    for lid, context in request.quote_context_by_line.items()}
            if self._v44_trip_weather():
                for option in request.pickup_date_options:
                    day = int(option["pickup_day"])
                    weather_context = self._trip_weather_context(
                        state, pickup_day=day, duration_days=duration)
                    option["trip_forecast_coverage"] = public_coverage(weather_context)
                    request.quote_context_by_date[str(day)] = {
                        lid: contextualize(context, weather_context)
                        for lid, context in request.quote_context_by_date[str(day)].items()}
                request.quote_context_by_line = copy.deepcopy(
                    request.quote_context_by_date[str(original)])
            self._reservation_event(state, "initial_date_options_frozen", request_id=request.request_id,
                details={"pickup_date_options": copy.deepcopy(request.pickup_date_options),
                         "default_pickup_day": original})
            if self.config.reservation_service_offers_enabled:
                self._generate_fixed_service_offer(state, request)

    def _generate_fixed_service_offer(self, state, request):
        policy = self.dataset.metadata["service_commitment_policy"]
        attempted = self._service_rng(request, "service-offer-route").random() < policy["service_offer_attempt_probability"]
        request.service_offer_audit = {"attempted": bool(attempted)}
        if not attempted:
            return
        factor = float(self._service_rng(request, "market-factor").choice(policy["market_factors"]))
        by_line, daily, reference = {}, {}, {}
        for line in request.lines:
            rows = []
            for pid in sorted(line.eligible_profile_ids):
                for oid in sorted(request.inquiry_profile_member_ids_by_line[line.line_id][pid]):
                    row = self._enriched_pairwise_quote_options(request, line, quote_options=[{
                        "option_id": f"option-{len(rows)+1:03d}", "offer_id": oid,
                        "promise_profile_id": pid, "daily_unit_price": 1.0}])[0]
                    offer = self.offers[oid]
                    purchase = float(offer.consumer_online_offer["unit_price_cny"])
                    cleaning = self.public_offer(oid)["normal_return_cleaning"]
                    clean_cost = float(cleaning["one_day_work_points"]) * float(cleaning["cash_per_work_point"])
                    row["market_reference_line_day"] = row["asset_quantity"] * (
                        purchase * policy["purchase_price_fraction_per_day"] + clean_cost * policy["one_day_cleaning_fraction"])
                    rows.append(row)
            if not rows:
                request.service_offer_audit.update(generated=False, reason="no_public_configuration")
                return
            ref = min(rows, key=lambda r: (r["market_reference_line_day"], r["offer_id"], r["promise_profile_id"]))
            reference[line.line_id] = ref["option_id"]
            daily[line.line_id] = max(1, cents(factor * ref["market_reference_line_day"])) / 100
            for row in rows:
                row["daily_unit_price"] = daily[line.line_id] / row["asset_quantity"]
                row["total_rent"] = (
                    rental_amount(daily[line.line_id], line.duration_days)
                    if self._v45_reservation_difficulty()
                    else cents(daily[line.line_id] * line.duration_days) / 100
                )
            by_line[line.line_id] = rows
        value = self._reservation_bundle_choice_input(state, request, by_line)
        dimensions = [l.store_options for l in value.lines]
        if not all(dimensions):
            request.service_offer_audit.update(generated=False, reason="no_common_date_configuration")
            return
        probe = CompleteContractOption("probe", tuple(d[0] for d in dimensions))
        outside = evaluate_complete_contract_choice(value, (probe,))
        threshold = max(outside.outside_score, outside.abandon_score)
        acceptable, seen = [], set()
        total = 0
        for configuration in itertools.product(*dimensions):
            total += 1
            contract = CompleteContractOption("candidate", tuple(configuration))
            score = score_complete_store_contract(value, contract)
            # Same tie convention as the existing customer kernel (store wins exact ties).
            if score < threshold:
                continue
            raw = {o.line_id: next(r for r in by_line[o.line_id] if r["option_id"] == o.option_id)
                   for o in configuration}
            signature = tuple(sorted((lid, r["offer_id"], r["asset_quantity"]) for lid, r in raw.items()))
            if signature in seen:
                continue
            seen.add(signature)
            is_reference = all(reference[o.line_id] == o.option_id for o in configuration)
            rank = hashlib.sha256(json.dumps([self.config.seed, request.request_id, "service-menu", signature],
                                             separators=(",", ":")).encode()).hexdigest()
            acceptable.append((not is_reference, rank, raw, contract, score))
        request.service_offer_audit.update(generated=bool(acceptable), market_factor=factor,
            internal_configuration_count=total, acceptable_configuration_count=len(acceptable),
            outside_score=outside.outside_score, abandon_score=outside.abandon_score)
        if not acceptable:
            return
        selected = sorted(acceptable, key=lambda x: x[:2])[:policy["maximum_complete_plans"]]
        # Canonical stable public order, never affordability/utility/inventory order.
        selected.sort(key=lambda x: tuple((lid, r["offer_id"], r["promise_profile_id"])
                                         for lid, r in sorted(x[2].items())))
        duration = request.return_day-request.pickup_day
        total_rent = (
            round(sum(rental_amount(daily[l.line_id], l.duration_days)
                      for l in request.lines), 2)
            if self._v45_reservation_difficulty()
            else sum(cents(daily[l.line_id]) * l.duration_days for l in request.lines) / 100
        )
        request.service_offer_terms = {"pricing_basis": (
            "line_first_day_half_additional" if self._v45_reservation_difficulty() else "line_day"),
            "total_rent": total_rent,
            "prepayment_due": cents(total_rent * request.prepayment_rate)/100,
            "line_daily_rents": daily, "duration_days": duration,
            "no_second_customer_acceptance_check": True}
        plans, recovery = [], {}
        required = {l.line_id: l.required for l in request.lines}
        for i, (_, _, raw, contract, score) in enumerate(selected, 1):
            pid = f"plan-{i:02d}"
            lines = [{"line_id": lid, "required": required[lid], "disposition": "quote",
                "line_option_id": r["option_id"], "offer_id": r["offer_id"],
                "promise_profile_id": r["promise_profile_id"], "asset_quantity": r["asset_quantity"],
                "daily_unit_price": r["daily_unit_price"], "pricing_basis": (
                    "line_first_day_half_additional" if self._v45_reservation_difficulty() else "line_day"),
                "line_daily_rent": daily[lid], "line_rent": r["total_rent"],
                "security_deposit": r["locked_security_deposit"]} for lid, r in sorted(raw.items())]
            plans.append({"plan_id": pid, "lines": lines, "total_rent": total_rent,
                "prepayment_due": request.service_offer_terms["prepayment_due"],
                "total_security_deposit": cents(sum(r["security_deposit"] for r in lines))/100})
            recovery[pid] = super()._complete_contract_recovery_offers(state, request, choice_input=value,
                selected_contract=replace(contract, contract_option_id=pid), selected_by_line=raw)
        request.customer_authorized_plans = plans
        request.service_offer_audit["recovery_by_plan_id"] = recovery
        request.status = "pending_service_offer"
        for line in request.lines:
            line.status = "pending_service_offer"
        self._reservation_event(state, "fixed_reward_offer_created", request_id=request.request_id,
            details={"total_rent": total_rent, "complete_plan_count": len(plans),
                     "pickup_date_options": copy.deepcopy(request.pickup_date_options)})

    def _enriched_pairwise_quote_options(self, request, line, *, quote_options):
        result = super()._enriched_pairwise_quote_options(request, line, quote_options=quote_options)
        if request.service_offer_terms:
            daily = request.service_offer_terms["line_daily_rents"][line.line_id]
            for row in result:
                row.update(daily_unit_price=daily / row["asset_quantity"],
                           total_rent=(rental_amount(daily, line.duration_days)
                                       if self._v45_reservation_difficulty()
                                       else cents(daily*line.duration_days)/100),
                           pricing_basis=(LINE_PRICING_BASIS if self._v45_reservation_difficulty()
                                          else "line_day"), line_daily_rent=daily)
        return result

    def _complete_contract_recovery_offers(self, state, request, *, choice_input, selected_contract, selected_by_line):
        if request.service_offer_terms:
            return copy.deepcopy(request.service_offer_audit["recovery_by_plan_id"][selected_contract.contract_option_id])
        return super()._complete_contract_recovery_offers(state, request, choice_input=choice_input,
            selected_contract=selected_contract, selected_by_line=selected_by_line)

    def respond_reservation_service_offers(self, state, decisions):
        if not self._v32_service_commitments():
            raise ValueError("fixed reward service offers require V32")
        if not decisions or len(decisions) > 5 or len({r.get("request_id") for r in decisions}) != len(decisions):
            raise ValueError("decisions require one to five unique request IDs")
        requests = {r.request_id: r for r in state.reservation_requests}
        checked = []
        for row in decisions:
            action = row.get("action")
            expected = {"request_id", "action", "plan_id", "pickup_day"} if action == "accept" else {"request_id", "action"}
            if action not in {"accept", "decline"} or set(row) != expected:
                raise ValueError("accept requires request_id, action, plan_id, pickup_day; decline requires request_id, action only")
            request = requests.get(row["request_id"])
            if request is None or request.status != "pending_service_offer" or request.created_day != state.day:
                raise ValueError("service offer is not pending today")
            if action == "accept":
                if row["plan_id"] not in {p["plan_id"] for p in request.customer_authorized_plans}:
                    raise ValueError("plan is not part of the frozen customer service offer")
                if type(row["pickup_day"]) is not int or row["pickup_day"] not in {d["pickup_day"] for d in request.pickup_date_options}:
                    raise ValueError("pickup_day is not a frozen customer date")
            checked.append((request, row))
        results = []
        for request, row in checked:
            request.decision_day = state.day
            if row["action"] == "decline":
                request.status = "service_offer_declined"
                result = {"request_id": request.request_id, "status": request.status}
            else:
                self._select_initial_date(request, row["pickup_day"])
                request.status = "customer_authorized_single_plan"
                result = self._confirm_factorized_plan(state, request, plan_id=row["plan_id"], origin="manager_accepted_service_offer")
            self._reservation_event(state, "fixed_reward_offer_"+row["action"], request_id=request.request_id,
                commitment_id=request.commitment_id, details={"action": row["action"],
                    **({"plan_id": row["plan_id"], "pickup_day": row["pickup_day"]} if row["action"] == "accept" else {})})
            results.append(self._public_service_money(result) if request.service_offer_terms else result)
        return {"decisions": results, "cash": state.cash}

    @staticmethod
    def _request_on_date(request, contexts):
        return replace(request, quote_context_by_line=contexts, quote_context_by_date={})

    def _bundle_store_option(self, state, request, line, option):
        result = super()._bundle_store_option(state, request, line, option)
        if result is None or len(request.quote_context_by_date) <= 1:
            return result
        variants = tuple(super(ServiceCommitmentsMixin, self)._bundle_store_option(
            state, self._request_on_date(request, request.quote_context_by_date[day]), line, option)
            for day in sorted(request.quote_context_by_date, key=int))
        if any(v is None for v in variants):
            return None
        return replace(result, date_variants=variants)

    def _reservation_bundle_choice_input(self, state, request, enriched_by_line):
        result = super()._reservation_bundle_choice_input(state, request, enriched_by_line)
        if len(request.quote_context_by_date) <= 1:
            return result
        variants = tuple(super(ServiceCommitmentsMixin, self)._reservation_bundle_choice_input(
            state, self._request_on_date(request, request.quote_context_by_date[day]), enriched_by_line)
            for day in sorted(request.quote_context_by_date, key=int))
        return replace(result, date_variants=variants)

    def _confirm_factorized_plan(self, state, request, *, plan_id, origin):
        result = super()._confirm_factorized_plan(state, request, plan_id=plan_id, origin=origin)
        if request.pickup_date_options:
            commitment = next(c for c in state.reservation_commitments if c.commitment_id == request.commitment_id)
            commitment.initial_agreed_pickup_day = commitment.pickup_day
            commitment.initial_agreed_return_day = commitment.return_day
            if request.service_offer_terms:
                for owner in (request, commitment):
                    for line in owner.lines:
                        line.locked_line_daily_rent = request.service_offer_terms["line_daily_rents"][line.line_id]
                        if self._v45_reservation_difficulty():
                            line.locked_line_pricing_basis = "line_first_day_half_additional"
                for row in result.get("lines", []):
                    row.update(
                        pricing_basis=request.service_offer_terms["pricing_basis"],
                        line_daily_rent=request.service_offer_terms["line_daily_rents"][row["line_id"]])
            result.update(pickup_day=commitment.pickup_day, scheduled_return_day=commitment.return_day,
                          initial_agreed_pickup_day=commitment.pickup_day,
                          initial_agreed_return_day=commitment.return_day)
        return result

    def _select_initial_date(self, request, pickup_day):
        option = next(d for d in request.pickup_date_options if d["pickup_day"] == pickup_day)
        request.pickup_day = pickup_day
        request.return_day = option["scheduled_return_day"]
        request.booking_lead_days = option["booking_lead_days"]
        request.forecast_condition = option["forecast_condition"]
        request.forecast_accuracy = option["forecast_accuracy"]
        request.quote_context_by_line = copy.deepcopy(request.quote_context_by_date[str(pickup_day)])

    def confirm_reservation_plans(self, state, selections):
        if not self._v32_service_commitments():
            return super().confirm_reservation_plans(state, selections)
        if not selections or len({r.get("request_id") for r in selections}) != len(selections):
            raise ValueError("selections require unique request IDs")
        requests = {r.request_id: r for r in state.reservation_requests}
        checked = []
        for row in selections:
            if set(row) - {"request_id", "plan_id", "pickup_day"} or not {"request_id", "plan_id"} <= set(row):
                raise ValueError("selection requires request_id, plan_id and a legal pickup_day")
            request = requests.get(row["request_id"])
            if request is None or request.status != "awaiting_manager_confirmation" or request.service_offer_terms:
                raise ValueError("request is not an authorized price inquiry awaiting confirmation")
            if row["plan_id"] not in {p["plan_id"] for p in request.customer_authorized_plans}:
                raise ValueError("plan is not customer authorized")
            if len(request.pickup_date_options) > 1 and "pickup_day" not in row:
                raise ValueError("pickup_day is required when the customer offers two dates")
            day = row.get("pickup_day", request.pickup_day)
            if type(day) is not int or day not in {d["pickup_day"] for d in request.pickup_date_options}:
                raise ValueError("pickup_day must be a frozen customer date")
            checked.append((request, row["plan_id"], day))
        result = []
        for request, plan_id, day in checked:
            self._select_initial_date(request, day)
            result.append(self._confirm_factorized_plan(state, request, plan_id=plan_id, origin="manager_selected"))
        return {"confirmations": result}

    def quote_reservation_inquiries(self, state, decisions):
        result = super().quote_reservation_inquiries(state, decisions)
        if self._v32_service_commitments():
            requests = {r.request_id: r for r in state.reservation_requests}
            for row in result["decisions"]:
                request = requests[row["request_id"]]
                row.update(request_kind="price_inquiry", pickup_date_options=copy.deepcopy(request.pickup_date_options),
                           default_pickup_day=request.original_inquiry_pickup_day,
                           authorization_date_scope="every_advertised_date")
                if request.status == "awaiting_manager_confirmation":
                    row["unanswered_confirmation_default"] = "original_pickup_day_and_customer_preferred_plan"
        return result

    def _reschedule_original_dates_possible(self, state, commitment, *, change_type, proposed_pickup_day):
        if not (self._v32_service_commitments() and self.config.reservation_reschedule_outcomes_enabled
                and change_type == "reschedule"):
            return None
        request = next(r for r in state.reservation_requests if r.request_id == commitment.request_id)
        rng = self._service_rng(request, f"reschedule-viability:{state.day}:{proposed_pickup_day}")
        return bool(rng.random() >= self.dataset.metadata["service_commitment_policy"]["original_dates_unavailable_probability"])

    def _resolve_v32_unavailable_dates(self, state, change, *, expired):
        if change.original_dates_still_possible is not False:
            return {}
        commitment = next(c for c in state.reservation_commitments if c.commitment_id == change.commitment_id)
        if commitment.status not in {"confirmed", "at_risk"}:
            return {}
        refund = float(commitment.unearned_revenue)
        self._cancel_customer_reservation(state, commitment,
            reason="reschedule_unanswered" if expired else "reschedule_declined", change_id=change.change_id)
        for other in state.reservation_change_requests:
            if other.commitment_id == commitment.commitment_id and other.status == "pending":
                other.status = "withdrawn_contract_cancelled"
                other.decision_day = state.day
        return {"contract_status": commitment.status, "prepayment_refund": refund,
                "manager_compensation": 0.0, "released_pickup_day": commitment.pickup_day,
                "released_return_day": commitment.return_day}

    def respond_reservation_changes(self, state, decisions):
        if self._v32_service_commitments():
            changes = {c.change_id: c for c in state.reservation_change_requests}
            commitments = {c.commitment_id: c for c in state.reservation_commitments}
            for decision in decisions:
                change = changes.get(decision.get("change_id"))
                if change is not None:
                    expected = {"active"} if change.change_type == "extension" else {"confirmed", "at_risk"}
                    if commitments[change.commitment_id].status not in expected:
                        raise ValueError("change cannot modify a terminal or incompatible contract")
        return super().respond_reservation_changes(state, decisions)

    def _expire_reservation_requests(self, state):
        super()._expire_reservation_requests(state)
        if self._v32_service_commitments():
            for request in state.reservation_requests:
                if request.status == "pending_service_offer" and request.expires_day == state.day:
                    request.status = "service_offer_expired"
                    request.decision_day = state.day
                    self._reservation_event(state, "fixed_reward_offer_expired", request_id=request.request_id)

    @staticmethod
    def _public_service_money(value):
        if isinstance(value, list):
            return [ServiceCommitmentsMixin._public_service_money(v) for v in value]
        if not isinstance(value, dict):
            return value
        result = {k: ServiceCommitmentsMixin._public_service_money(v) for k, v in value.items()}
        if result.get("pricing_basis") in {"line_day", "line_first_day_half_additional"} or result.get("locked_line_daily_rent") is not None:
            result.pop("daily_unit_price", None)
            result.pop("locked_daily_price", None)
            result.pop("quoted_daily_price", None)
            result["pricing_basis"] = (
                result.get("locked_line_pricing_basis") or result.get("pricing_basis")
                or "line_day"
            )
        return result

    @classmethod
    def _public_reservation_request(cls, request):
        result = super()._public_reservation_request(request)
        if request.pickup_date_options:
            result.update(request_kind="fixed_reward_offer" if request.service_offer_terms else "price_inquiry",
                default_pickup_day=request.original_inquiry_pickup_day,
                pickup_date_options=copy.deepcopy(request.pickup_date_options))
        if request.service_offer_terms:
            result["service_offer_terms"] = copy.deepcopy(request.service_offer_terms)
            result["service_offer_terms"]["complete_plans"] = cls._public_service_money(request.customer_authorized_plans)
            result["available_actions"] = ["accept", "decline"] if request.status == "pending_service_offer" else []
        return cls._public_service_money(result)

    def reservation_inbox(self, state):
        result = super().reservation_inbox(state)
        if self._v32_service_commitments():
            result["service_offers"] = [self._public_reservation_request(r) for r in state.reservation_requests
                                        if r.status == "pending_service_offer" and r.created_day == state.day]
            for notice in result["service_offers"]:
                plans = notice["service_offer_terms"].pop("complete_plans")
                notice["complete_plan_count"] = len(plans)
                notice["choices_tool"] = "view_reservation_quote_workspace"
            result["service_offer_count"] = len(result["service_offers"])
            result["unanswered_service_offer_default"] = "service_offer_expired_without_commitment"
            result["unanswered_plan_confirmation_default"] = "original_pickup_day_and_customer_preferred_plan"
        return result

    def observation(self, state, *, consume_inbox):
        result = super().observation(state, consume_inbox=consume_inbox)
        if self._v32_service_commitments():
            count = sum(r.status == "pending_service_offer" and r.created_day == state.day for r in state.reservation_requests)
            # Business status fields are nested inside the public observation.
            def add(value):
                if isinstance(value, dict):
                    if "reservation_actions_due_today" in value:
                        value["reservation_service_offer_count"] = count
                        value["reservation_actions_due_today"] += count
                    for child in list(value.values()):
                        add(child)
                elif isinstance(value, list):
                    for child in value:
                        add(child)
            add(result)
        return result

    def public_operating_rules(self):
        result = super().public_operating_rules()
        if self._v32_service_commitments():
            result["service_commitments"] = {
                "price_inquiries": "Explicit line prices apply to every advertised date; jointly authorized plans can be confirmed on either date. Unconfirmed authorized quotations bind at day end to the original date and customer preferred plan.",
                "fixed_reward_offers": "Customer's frozen total/line-day rent, exact complete plans and deposits are binding terms. Accept with plan_id and pickup_day or decline; unanswered offers expire without commitment. No inventory feasibility is checked.",
                "date_control": "Initial dates/plan lock on commitment; no proactive free date or plan changes. Later customer reschedules use their own proposed dates.",
                "reschedule_response": "original_dates_still_possible=false means explicit decline or unanswered expiry causes customer cancellation and prepayment refund, not manager default.",
                "line_day_billing": "For fixed reward offers only, entire-line daily rent is fixed across asset quantities and plans. Optional missing lines are credited; extensions bill fulfilled lines at the frozen daily line rent.",
            }
            advance = result["advance_reservations"]
            advance.update(
                one_time_quote="price_inquiry only: explicit factorized line options, one common menu/price for all advertised dates. With two dates even one authorized plan awaits date confirmation. Fixed_reward_offer instead requires accept/decline of frozen customer terms, never a quotation.",
                manager_confirmation="Price inquiries: select authorized plan and advertised pickup_day; at day end unanswered authorized quotes bind on the original date/customer preferred plan. Fixed reward offers: explicit accept with plan_id/pickup_day or decline; unanswered offers expire without commitment.",
                customer_choice_scope="Price inquiries: a common complete plan must be acceptable on every advertised date against the customer's best frozen non-store date option. Fixed reward service offers have already passed this check at creation; acceptance does not repeat it.",
                inquiry_candidates="Price inquiries: frozen Profile/exact quotation choices. Fixed reward offers: up to three frozen complete plans, not independently mixable line options. Full terms are in the request-level workspace.",
                authorization_is_exhaustive_scope="returned price-inquiry authorizations only; the explicitly offered fixed-reward complete-plan set is the entire service offer",
                pricing_basis_by_request_kind={"price_inquiry":"asset_unit_day","fixed_reward_offer":"line_day"},
                service_commitment_controls=dict(result["service_commitments"]),
                service_commitment_switches={"date_flexibility":self.config.reservation_date_flexibility_enabled,
                    "reschedule_decline_outcomes":self.config.reservation_reschedule_outcomes_enabled,
                    "fixed_reward_offers":self.config.reservation_service_offers_enabled},
            )
        return result

    def view_reservation_quote_workspace(self, state, **kwargs):
        result = super().view_reservation_quote_workspace(state, **kwargs)
        if not self._v32_service_commitments():
            return result
        requests = {r.request_id: r for r in state.reservation_requests}
        used_offers, used_profiles = set(), set()
        for row in result["requests"]:
            request = requests[row["request_id"]]
            row.update(request_kind="fixed_reward_offer" if request.service_offer_terms else "price_inquiry",
                pickup_date_options=copy.deepcopy(request.pickup_date_options),
                default_pickup_day=request.original_inquiry_pickup_day)
            if request.service_offer_terms:
                row["service_offer_terms"] = {**copy.deepcopy(request.service_offer_terms),
                    "complete_plans": self._public_service_money(request.customer_authorized_plans)}
                row["available_actions"] = ["accept", "decline"]
                row["configuration_rule"] = "Choose one offered complete plan; do not mix its lines with another plan."
                for line in row["lines"]:
                    line.pop("profile_terms", None)
                for plan in request.customer_authorized_plans:
                    for line in plan["lines"]:
                        used_offers.add(line["offer_id"])
                        used_profiles.add(line["promise_profile_id"])
            else:
                row["available_actions"] = ["quote", "decline"]
                row["authorization_date_scope"] = "every_advertised_date"
                for line in row["lines"]:
                    for term in line["profile_terms"]:
                        used_profiles.add(term["promise_profile_id"])
                        used_offers.update(term["frozen_member_offer_ids"])
        result["profiles_by_id"] = {pid: row for pid, row in result["profiles_by_id"].items() if pid in used_profiles}
        if "offers_by_id" in result:
            result["offers_by_id"] = {oid: row for oid, row in result["offers_by_id"].items() if oid in used_offers}
        if "offer_ids" in result:
            result["offer_ids"] = sorted(used_offers)
        result["quote_constraints"]["applies_to"] = "price_inquiries_only"
        if self._v45_reservation_difficulty():
            result["quote_constraints"].update(
                daily_unit_price="positive_finite_cent_exact_first_day_price_per_asset",
                pricing_basis=ASSET_PRICING_BASIS,
                additional_day_multiplier=0.5,
                additional_day_rounding="half_up_to_cents_per_asset_before_quantity",
            )
        return result

    @classmethod
    def _public_reservation_commitment(cls, commitment):
        return cls._public_service_money(super()._public_reservation_commitment(commitment))

    @classmethod
    def _public_reservation_line(cls, line):
        return cls._public_service_money(super()._public_reservation_line(line))

    @classmethod
    def _v22_reservation_detail_record(cls, request, commitment, **kwargs):
        result = super()._v22_reservation_detail_record(request, commitment, **kwargs)
        if request.pickup_date_options:
            result.update(request_kind="fixed_reward_offer" if request.service_offer_terms else "price_inquiry")
            if "inquiry_terms" in result:
                result["inquiry_terms"]["pickup_date_options"] = copy.deepcopy(request.pickup_date_options)
                if request.service_offer_terms:
                    result["inquiry_terms"]["service_offer_terms"] = {**copy.deepcopy(request.service_offer_terms),
                        "complete_plans": cls._public_service_money(request.customer_authorized_plans)}
            if commitment and "current_terms" in result:
                result["current_terms"].update(initial_agreed_pickup_day=commitment.initial_agreed_pickup_day,
                                              initial_agreed_return_day=commitment.initial_agreed_return_day)
                by_line = {l.line_id: l for l in commitment.lines}
                for row in result["current_terms"].get("locked_lines", []):
                    line = by_line[row["line_id"]]
                    if line.locked_line_daily_rent is not None:
                        row.update(pricing_basis=line.locked_line_pricing_basis or "line_day",
                                   line_daily_rent=line.locked_line_daily_rent)
        return cls._public_service_money(result)

    @classmethod
    def _v22_compact_quote_history_record(cls, request, *, quote_outcome):
        if request.service_offer_terms:
            return {"request_id": request.request_id, "commitment_id": request.commitment_id,
                "request_kind": "fixed_reward_offer", "created_day": request.created_day,
                "quote_day": None, "decision_day": request.decision_day, "quote_action": None,
                "quote_outcome": "service_offer_accepted" if request.commitment_id else request.status,
                "current_request_status": request.status, "pickup_day": request.pickup_day,
                "return_day": request.return_day, "pickup_date_options": copy.deepcopy(request.pickup_date_options),
                "visible_party_context": request.visible_party_context,
                "original_inquiry_pickup_day": request.original_inquiry_pickup_day,
                "original_inquiry_return_day": request.original_inquiry_return_day,
                "lines": [{"line_id": l.line_id, "service_key": l.service_key, "required": l.required,
                           "service_units": l.service_units, "duration_days": l.duration_days,
                           "hard_requirements": copy.deepcopy(l.public_hard_requirements),
                           "stated_preferences": copy.deepcopy(l.public_preferences),
                           "quote_options": []} for l in request.lines],
                "service_offer_terms": copy.deepcopy(request.service_offer_terms),
                "offered_complete_plans": cls._public_service_money(request.customer_authorized_plans),
                "customer_authorized_plan_count": len(request.customer_authorized_plans),
                "customer_authorized_plan_references": [{"plan_id": p["plan_id"]} for p in request.customer_authorized_plans],
                "manager_confirmed_plan_id": request.manager_confirmed_plan_id,
                "decline_feedback": {}}
        result = super()._v22_compact_quote_history_record(request, quote_outcome=quote_outcome)
        if request.pickup_date_options:
            result.update(request_kind="price_inquiry", pickup_date_options=copy.deepcopy(request.pickup_date_options))
        return result

    def view_reservation(self, state, identifiers, **kwargs):
        if self._v32_service_commitments():
            kwargs.setdefault("limit", 10)
            kwargs.setdefault("searchable_quote_history", True)
        result = super().view_reservation(state, identifiers, **kwargs)
        if self._v32_service_commitments() and kwargs.get("view", "index") == "index":
            requests = {r.request_id: r for r in state.reservation_requests}
            # Only index records, not complete plans or hidden request snapshots.
            for row in result.get("reservations", []):
                r = requests[row["request_id"]]
                row.update(request_kind="fixed_reward_offer" if r.service_offer_terms else "price_inquiry",
                           pickup_date_options=copy.deepcopy(r.pickup_date_options))
                c = next((c for c in state.reservation_commitments if c.request_id == r.request_id), None)
                row.update(initial_agreed_pickup_day=c.initial_agreed_pickup_day if c else None,
                           initial_agreed_return_day=c.initial_agreed_return_day if c else None)
        return self._public_service_money(result)

    def _v32_extension_allocations(self, state, commitment, deltas):
        result = {}
        for line in commitment.lines:
            assets = {a.asset_id for a in commitment.assignments if a.line_id == line.line_id}
            contracts = sorted((c for c in state.contracts if c.reservation_id == commitment.commitment_id
                                and c.status == "active" and c.asset_id in assets), key=lambda c: c.contract_id)
            amount = deltas.get(line.line_id, 0.)
            if not contracts:
                if cents(amount):
                    raise AssertionError("extension income lacks actual line assignments")
                continue
            result.update(zip((c.contract_id for c in contracts), split_cents(amount, len(contracts))))
        return result
