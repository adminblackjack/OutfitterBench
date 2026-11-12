"""Diagnostic population knowledge; never receives the live environment or state.

The simulator below uses the production customer kernel on independent synthetic
customers. It knows shared model coefficients, not any real customer's tastes,
future arrivals, world seed, or weather tape. Public request fields are copied
through a whitelist before any prediction is made.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import itertools
import json
import math

import numpy as np

from .config import OutfitterConfig
from .channel_demand import effective_rental_cash_segment
from .data import OutfitterDataset
from .environment import OutfitterEnvironment
from .models import ReservationLine, ReservationRequest, WorldState
from .reservation_choice import (
    BundleStoreOption, CompleteContractOption, evaluate_complete_contract_choice,
)
from .reservation_difficulty import rental_amount
from .runtime.common import DEPOSIT_RATIOS
from .weather_information import consideration_probability, forecast_regime


SAMPLING_SEED = 460925
SAMPLE_COUNT = 128
PRICE_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)


def public_request(request):
    """Intentionally exclude IDs, hidden contexts, histories, and random addresses."""
    fields = ("created_day", "pickup_day", "return_day", "booking_lead_days",
              "visible_party_context", "forecast_condition", "forecast_accuracy",
              "prepayment_rate", "pickup_date_options")
    line_fields = ("line_id", "service_key", "service_units", "duration_days", "required",
                   "hard_requirements", "stated_preferences", "profile_terms")
    return {**{key: copy.deepcopy(request[key]) for key in fields if key in request},
            "lines": [{key: copy.deepcopy(line[key]) for key in line_fields if key in line}
                      for line in request["lines"]]}


def option_key(option):
    return (option["line_id"], option["offer_id"], option["promise_profile_id"])


@dataclass
class Prediction:
    expected_margin: float
    acceptance_probability: float
    sample_count: int
    configuration_count: int


@dataclass
class ChoiceBank:
    """Vectorized exact repricing of the sampled customer utility functions."""
    request: dict
    values: dict
    base_generalized: dict
    deposits: dict
    relative_weights: np.ndarray
    cash_weights: np.ndarray
    cash_scales: np.ndarray
    common_scores: np.ndarray
    threshold: np.ndarray
    duration: int
    sample_weights: np.ndarray
    tail_coefficient: float = 0.

    def scores(self, selected):
        result = self.common_scores.copy()
        rent, deposit = 0.0, 0.0
        for option in selected:
            if option.get("disposition") == "omit":
                continue
            key = option_key(option)
            amount = rental_amount(option["daily_unit_price"], self.duration,
                                   int(option["asset_quantity"]))
            # The stored deposit is already multiplied by deposit friction.
            generalized = amount + self.deposits[key]
            result += self.values[key] - self.relative_weights * math.log(
                max(0.01, generalized) / self.base_generalized[key])
            rent += amount
            deposit += self.deposits[key]
        result -= self.cash_weights * np.log1p((rent + deposit) / self.cash_scales)
        result -= self.tail_coefficient*np.maximum(0.,(rent+deposit)/self.cash_scales-2.)**2
        # Common menus must work on every customer-advertised date.
        return np.min(result, axis=0)

    def evaluate(self, menus, benefit):
        configurations = list(itertools.product(*menus))
        if not 1 <= len(configurations) <= 24:
            raise ValueError("prediction must preserve the actor's 24-plan limit")
        scores = np.stack([self.scores(lines) for lines in configurations])
        gains = np.array([float(benefit(lines)) for lines in configurations])
        # Customers expose only the top three authorized plans. The merchant
        # can choose the best economic option within that returned set.
        ranking = np.argsort(-scores, axis=0, kind="stable")[:3]
        chosen_scores = np.take_along_axis(scores, ranking, axis=0)
        accepted = chosen_scores >= self.threshold[None, :]
        chosen_gains = np.where(accepted, gains[ranking], -np.inf)
        best_gain = np.max(chosen_gains, axis=0)
        has_plan = np.any(accepted, axis=0)
        payoff = np.where(has_plan, best_gain, 0.0)
        return Prediction(float(np.average(payoff,weights=self.sample_weights)),
                          float(np.average(has_plan,weights=self.sample_weights)),
                          len(self.threshold), len(configurations))


class KnowledgeProvider:
    def __init__(self, config: OutfitterConfig, *, samples=SAMPLE_COUNT):
        if not 1 <= int(samples) <= SAMPLE_COUNT:
            raise ValueError("samples must be in [1,128]")
        self.samples = int(samples)
        # The evaluated model/world seed is deliberately discarded.
        self.config = replace(config, seed=SAMPLING_SEED)
        data = OutfitterDataset.load(config.data_dir)
        # Preserve only deterministic calendar fields; remove future physical
        # weather before constructing the synthetic evaluator.
        data.weather = {}
        calendar_keys = {"date", "day_of_year", "weekday", "is_weekend", "day_type", "is_school_holiday",
                         "holiday_id", "holiday_name", "school_holiday"}
        data.weather_calendar = {
            day: {key: value for key, value in row.items() if key in calendar_keys}
            for day, row in data.weather_calendar.items()
        }
        self._env = OutfitterEnvironment(self.config, data)
        def no_truth(*_args, **_kwargs):
            raise AssertionError("knowledge predictor attempted to read weather truth")
        self._env._weather = no_truth
        self._env._temperatures = no_truth
        self._env.weather_forecast = no_truth
        self._env.reset = no_truth
        self._views = {}
        original_views = self._env._channel_offer_views
        def cached_views(state, session, **kwargs):
            if kwargs:
                raise AssertionError("synthetic customer predictor does not use operational branches")
            key = (session.event_address, session.service_key, session.pickup_day)
            if key not in self._views:
                self._views[key] = original_views(state, session)
            return self._views[key]
        self._env._channel_offer_views = cached_views
        self.provenance = {
            "format": "outfitter-known-population-model-v1",
            "sampling_seed": SAMPLING_SEED, "sample_count": self.samples,
            "knows": ["shared_customer_kernel", "catalog", "calendar_arrival_parameters"],
            "forbidden": ["live_state", "real_customer_latents", "world_seed",
                          "future_weather_truth", "future_real_requests", "real_random_tape"],
            "approximation": "public-context conditional population; does not condition on failure to generate a fixed offer",
        }

    @staticmethod
    def _party(request, sample):
        by_service = {line["service_key"]: line for line in request["lines"]}
        family = request.get("visible_party_context") == "family_with_children"
        if "rental:tent" in by_service:
            adults = int(by_service["rental:sleeping_bag"]["service_units"])
            party = int(by_service["rental:sleeping_pad"]["service_units"])
            return "overnight", adults, max(0, party - adults)
        if "rental:camp_table" in by_service:
            party = int(by_service["rental:camp_chair"]["service_units"])
            splits = [(a, party-a) for a in range(2, 5)
                      if (1 <= party-a <= 2 if family else party-a == 0)]
            if not splits:
                raise ValueError("public party context is inconsistent with generated picnic units")
            adults, children = splits[sample % len(splits)]
            return "picnic", adults, children
        return "lightweight", 1, 0

    @staticmethod
    def _condition_session(session, line, date):
        preferences = {p["attribute"]: p["value"] for p in line["stated_preferences"]}
        preferred = session.preferred_group
        use = session.use_case
        if "pad_type" in preferences:
            preferred = preferences["pad_type"]
        if "chair_style" in preferences:
            preferred = "comfort_chair" if preferences["chair_style"] == "comfort" else "lightweight_chair"
        if line["service_key"] == "rental:camp_cookware":
            functions = line["hard_requirements"].get("required_functions", [])
            use = "fry_meal" if "fry" in functions else "cook_meal" if "stew" in functions else "hot_drink"
        context = session.trip_weather_context
        if line["service_key"] == "rental:sleeping_bag":
            cold = context["minimum_covered_forecast_night_low_c"] <= 15
            preferred, use = (("three_season_bag", "cool_night") if cold
                              else ("warm_weather_bag", "warm_night"))
        return replace(session, preferred_group=preferred, use_case=use,
                       portability_sensitive=preferences.get("portability") == "preferred",
                       booking_lead_days=int(date["booking_lead_days"]),
                       pickup_day=int(date["pickup_day"]),
                       forecast_adversity=KnowledgeProvider._adversity(date["forecast_condition"]),
                       forecast_uncertainty=1.0-float(date["forecast_accuracy"]))

    @staticmethod
    def _adversity(condition):
        return {"sunny": 0., "cloudy": .18, "hot": .35, "cold": .45,
                "rainy": .82, "snowy": 1.0}.get(condition, .2)

    def _sample(self, request, date, sample, nonce):
        env = self._env
        archetype, adults, children = self._party(request, sample)
        address = (1_000_000 + nonce, 11, sample)
        duration = int(request["return_day"])-int(request["pickup_day"])
        context = {
            **date["trip_forecast_coverage"], "pickup_day": int(date["pickup_day"]),
            "return_day": int(date["scheduled_return_day"]), "forecasts": [],
        }
        state = WorldState(day=int(date["pickup_day"]), max_days=self.config.max_days+self.config.runoff_days,
            cash=0.0, deposit_liability=0.0, unearned_revenue_liability=0.0,
            prices={oid: 1.0 for oid in env.offers},
            deposits={oid: round(max(20., offer.unit_cost * DEPOSIT_RATIOS[offer.service_line]),2)
                      for oid,offer in env.offers.items() if offer.asset_kind == "durable"},
            rental_service_open={oid:True for oid in env.offers}, assets=[], retail_inventory={})
        sessions, lines = {}, []
        primary = None
        for index, line in enumerate(request["lines"]):
            session = env._make_choice_session(state,
                service_key=line["service_key"], units_requested=int(line["service_units"]),
                duration_days=duration, adults=adults, children=children,
                size_class=line["hard_requirements"].get("size_class", "n/a"),
                archetype=archetype, weather=date["forecast_condition"], request_index=index,
                event_address=address, intent_role="required" if line["required"] else "optional",
                trip_weather_context=context)
            session = self._condition_session(session, line, date)
            if primary is None:
                primary = session
            else:
                session = env._reservation_customer_coherent_session(primary, session)
            sessions[line["line_id"]] = asdict(session)
            lines.append(ReservationLine(line_id=line["line_id"], offer_id="synthetic",
                service_line=line["service_key"].split(":",1)[1], quantity=1,
                customer_units=int(line["service_units"]), service_units=int(line["service_units"]),
                service_key=line["service_key"], duration_days=duration, required=line["required"]))
        synthetic = ReservationRequest(request_id="synthetic-only", created_day=int(request["created_day"]),
            expires_day=int(request["created_day"]), pickup_day=int(date["pickup_day"]),
            return_day=int(date["scheduled_return_day"]), booking_lead_days=int(date["booking_lead_days"]),
            lines=lines, visible_party_context=request["visible_party_context"],
            forecast_condition=date["forecast_condition"], forecast_accuracy=float(date["forecast_accuracy"]),
            prepayment_rate=float(request["prepayment_rate"]), total_locked_rent=0.,prepayment_due=0.,
            event_address=list(address), customer_segment=primary.customer_segment,
            request_budget_multiplier=primary.request_budget_multiplier,
            quote_context_by_line=sessions)
        return state, synthetic

    def choice_bank(self, visible_request, candidates):
        request = public_request(visible_request)
        nonce = int(hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()[:7],16)
        keys = {option_key(o):o for line in candidates for o in line
                if o.get("disposition") != "omit"}
        env, model = self._env, self._env.channel_demand_model
        world = model.worlds[self.config.demand_world_id]
        duration = int(request["return_day"])-int(request["pickup_day"])
        dates = request["pickup_date_options"]
        original_date = next(date for date in dates
                             if int(date["pickup_day"]) == int(request["pickup_day"]))
        regime = forecast_regime({"condition":original_date["forecast_condition"],
                                  "day_high_c":original_date["forecast_day_high_c"]})
        weights = np.empty(self.samples)
        shape = (len(dates),self.samples)
        values = {key:np.full(shape,-np.inf) for key in keys}
        relative,cash_weights,cash_scales,common,threshold = [np.empty(shape) for _ in range(5)]
        deposits = {key:float(option["security_deposit"])*model.deposit_friction_rate
                    for key,option in keys.items()}
        baseline = {key:rental_amount(1.0,duration,int(option["asset_quantity"]))+deposits[key]
                    for key,option in keys.items()}
        enriched = {line["line_id"]:[] for line in request["lines"]}
        for key,option in keys.items():
            enriched[key[0]].append({"option_id":"|".join(key),"offer_id":key[1],
                "promise_profile_id":key[2],"asset_quantity":option["asset_quantity"],
                "daily_unit_price":1.,"total_rent":rental_amount(1.,duration,option["asset_quantity"]),
                "locked_security_deposit":option["security_deposit"]})
        for sample in range(self.samples):
            for d,date in enumerate(dates):
                state, synthetic = self._sample(request,date,sample,nonce)
                value = env._reservation_bundle_choice_input(state,synthetic,enriched)
                if d == 0:
                    context = next(iter(synthetic.quote_context_by_line.values()))
                    weights[sample] = consideration_probability(
                        archetype=context["archetype"], regime=regime,
                        forecast_trust=context["forecast_trust"])
                segment = model.segments[synthetic.customer_segment]
                relative[d,sample] = world.rental_relative_weight * segment.rental_weight_multiplier
                cash_weights[d,sample] = world.absolute_price_weight * segment.absolute_weight_multiplier * value.store_cash_salience
                cash_scales[d,sample] = value.cash_segment.cash_scale_cny
                common[d,sample] = (value.store_service_value-value.return_burden
                    + value.channel_taste_scale*value.channel_tastes["store"])
                required = []
                for line in value.lines:
                    for option in line.store_options:
                        key = (line.line_id,option.offer_id,option.promise_profile_id)
                        values[key][d,sample] = option.non_cash_value
                    if line.required:
                        required.append(line.store_options[0] if line.store_options else
                            BundleStoreOption(line.line_id,"dummy","dummy","dummy",True,-1e9,0.,0.))
                probe = CompleteContractOption("probe",tuple(required),
                    tuple(line.line_id for line in value.lines if not line.required))
                trace = evaluate_complete_contract_choice(value,(probe,))
                threshold[d,sample] = max(trace.outside_score,trace.abandon_score)
        self._views.clear()
        return ChoiceBank(request,values,baseline,deposits,relative,cash_weights,cash_scales,
                          common,np.max(threshold,axis=0),duration,weights,model.rental_budget_tail_coefficient)

    def mean_cash_scale(self, visible_request):
        """Conditional population estimate, not the real customer's budget."""
        request = public_request(visible_request)
        nonce = int(hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:7], 16)
        values = []
        for sample in range(16):
            _, synthetic = self._sample(request, request["pickup_date_options"][0], sample, nonce)
            session = self._env._restore_quote_context(next(iter(synthetic.quote_context_by_line.values())))
            model = self._env.channel_demand_model
            values.append(effective_rental_cash_segment(model, model.segments[synthetic.customer_segment],
                duration_days=session.duration_days, party_equivalent=max(1., session.adults + .6*session.children),
                request_budget_multiplier=synthetic.request_budget_multiplier).cash_scale_cny)
        return float(np.mean(values))
