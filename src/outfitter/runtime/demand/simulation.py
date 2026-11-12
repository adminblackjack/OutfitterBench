from __future__ import annotations

from ..common import (
    Any,
    CLEANING_WIPE_BOXES_PER_REQUEST,
    ChoiceServiceResult,
    Counter,
    WorldState,
    _ARCHETYPE_INDEX,
    _RNG_ARRIVALS,
    _RNG_CUSTOMER_BASKET,
    _RNG_PARTY_COMPOSITION,
    copy,
    np,
    seasonal_index,
)


class DemandSimulationMixin:
    def _serve_customer_v5(
        self,
        state: WorldState,
        basket_rng: np.random.Generator,
        archetype: str,
        weather: str,
        demand_weather: str,
        *,
        event_address: tuple[int, int, int],
    ) -> tuple[int, int, Counter[str]]:
        party_rng = self._event_rng(_RNG_PARTY_COMPOSITION, *event_address)
        day_type = str(self._calendar(state.day)["day_type"])
        adult_ranges = {
            "overnight": (1, 4),
            "picnic": (2, 5),
            "lightweight": (1, 2),
            "supply": (1, 4),
        }
        low, high = adult_ranges[archetype]
        adults = int(party_rng.integers(low, high))
        child_probability = {
            "weekday": 0.08,
            "weekend": 0.15,
            "public_holiday": 0.35,
            "school_holiday": 0.45,
        }[day_type]
        children = (
            int(party_rng.integers(1, 3))
            if archetype != "lightweight" and party_rng.random() < child_probability
            else 0
        )
        if self._v49_school_holiday():
            shape = self._v49_party_shape(state.day, archetype, int(event_address[2]))
            adults, children = shape['adults'], shape['children']
        party_size = adults + children
        requests: dict[tuple[str, str, int | None], int] = {}
        request_roles: dict[tuple[str, str, int | None], str] = {}

        def local_request_multiplier(service_key: str) -> float:
            if self.demand_strategy.channel_enabled:
                return 1.0
            if self.episode_market_profile is None:
                return 1.0
            values = [
                row.request_rate_multiplier
                for key, row in self.episode_market_profile.groups.items()
                if key.startswith(f"{service_key}|")
            ]
            return sum(values) / len(values) if values else 1.0

        def add(
            service_key: str,
            units: int,
            duration: int | None = None,
            size: str = "n/a",
            *,
            role: str = "optional",
        ) -> None:
            if service_key not in self.by_service:
                return
            if service_key == "retail:rain_gear" and size not in {
                "adult",
                "child",
            }:
                raise AssertionError(
                    "formal rainwear demand must have an adult or child size"
                )
            key = (service_key, size, duration)
            requests[key] = max(requests.get(key, 0), max(1, int(units)))
            if role == "required" or key not in request_roles:
                request_roles[key] = role

        def add_sized_rain_gear(*, role: str) -> None:
            if adults > 0:
                add("retail:rain_gear", adults, size="adult", role=role)
            if children > 0:
                add("retail:rain_gear", children, size="child", role=role)

        def maybe(service_key: str, units: int, probability: float, *, duration: int | None = None, size: str = "n/a") -> None:
            adjusted_probability = min(
                1.0, probability * local_request_multiplier(service_key)
            )
            if basket_rng.random() < adjusted_probability:
                add(service_key, units, duration, size)

        duration: int | None = None
        v47_shape=(self._v47_customer_shape(state.day,archetype,int(event_address[2]))
                   if self._v47_autonomous_rentals() and archetype!="supply" else None)
        if archetype == "overnight":
            if day_type == "public_holiday":
                duration = int(basket_rng.choice([2, 3, 4, 5], p=[0.2, 0.35, 0.30, 0.15]))
            else:
                duration = int(basket_rng.choice([1, 2, 3], p=[0.45, 0.40, 0.15]))
            if v47_shape:
                duration=v47_shape["duration"]
                if v47_shape["basecamp"]:
                    add("rental:camp_chair",party_size,duration,role="required")
            add("rental:tent", 1, duration, role="required")
            if adults:
                add("rental:sleeping_bag", adults, duration, "adult_regular", role="required")
            if children:
                add("rental:sleeping_bag", children, duration, "child", role="required")
            add("rental:sleeping_pad", party_size, duration, role="required")
            maybe("rental:camp_lantern", 1, 0.55, duration=duration)
            maybe("rental:camp_cookware", 1, 0.35, duration=duration)
            maybe("retail:bottled_water", party_size, 0.62)
            maybe("retail:energy_snacks", party_size, 0.58)
            maybe("retail:instant_food", party_size, 0.55)
        elif archetype == "picnic":
            duration=v47_shape["duration"] if v47_shape else 1
            add("rental:camp_table", 1, duration, role="required")
            add("rental:camp_chair", party_size, duration, role="required")
            maybe("rental:camp_cookware", 1, 0.35, duration=duration)
            maybe("retail:bottled_water", party_size, 0.62)
            maybe("retail:energy_snacks", party_size, 0.58)
            maybe("retail:garbage_bags", 1, 0.58)
            maybe(
                "retail:cleaning_wipes",
                CLEANING_WIPE_BOXES_PER_REQUEST,
                0.55,
            )
        elif archetype == "lightweight":
            duration = int(basket_rng.choice([1, 2], p=[0.7, 0.3]))
            if v47_shape:
                duration=v47_shape["duration"]
            add("rental:sleeping_bag", 1, duration, "adult_regular", role="required")
            add("rental:sleeping_pad", 1, duration, role="required")
            maybe("rental:camp_lantern", 1, 0.42, duration=duration)
            maybe("retail:bottled_water", 1, 0.62)
            maybe("retail:energy_snacks", 1, 0.62)
        else:
            retail = self._retail_service_keys()
            weights = []
            for service_key in retail:
                weight = local_request_multiplier(service_key)
                if demand_weather == "rainy" and service_key == "retail:rain_gear":
                    weight = 6.0
                if demand_weather == "hot" and service_key in {"retail:bottled_water", "retail:repellent"}:
                    weight = 4.0
                if day_type == "public_holiday" and service_key in {"retail:energy_snacks", "retail:instant_food"}:
                    weight *= 2.0
                weights.append(weight)
            probabilities = np.array(weights, dtype=float) / sum(weights)
            count = min(len(retail), int(basket_rng.integers(1, 4)))
            for service_key in basket_rng.choice(retail, size=count, replace=False, p=probabilities):
                selected_service = str(service_key)
                if selected_service == "retail:rain_gear":
                    add_sized_rain_gear(role="required")
                else:
                    add(
                        selected_service,
                        party_size
                        if selected_service == "retail:bottled_water"
                        else 1,
                        role="required",
                    )

        if demand_weather == "rainy":
            probability = 0.9 if archetype == "supply" else 0.62
            if basket_rng.random() < probability:
                add_sized_rain_gear(
                    role="required" if archetype == "supply" else "optional"
                )
        if demand_weather == "hot":
            maybe("retail:bottled_water", party_size, 0.9 if archetype == "supply" else 0.68)
            maybe("retail:repellent", 1, 0.9 if archetype == "supply" else 0.68)

        requested = 0
        fulfilled = 0
        outcomes: Counter[str] = Counter()
        if any(
            service_key == "retail:rain_gear"
            and size not in {"adult", "child"}
            for service_key, size, _duration in requests
        ):
            raise AssertionError("formal rainwear request retained an invalid size")
        indexed = {
            key: (index, units)
            for index, (key, units) in enumerate(sorted(requests.items()))
        }

        def serve_stage(
            keys: list[tuple[str, str, int | None]],
            *,
            active: bool,
            suppression_reason: str | None = None,
        ) -> list[ChoiceServiceResult]:
            nonlocal requested, fulfilled
            results: list[ChoiceServiceResult] = []
            channel_stage = self.demand_strategy.channel_enabled
            stage_snapshot = None
            if active and channel_stage:
                stage_snapshot = copy.copy(state)
                stage_snapshot.assets = [copy.copy(asset) for asset in state.assets]
                stage_snapshot.retail_inventory = {
                    offer_id: dict(row)
                    for offer_id, row in state.retail_inventory.items()
                }
                stage_snapshot.prices = dict(state.prices)
                stage_snapshot.deposits = dict(state.deposits)
                stage_snapshot.rental_service_open = dict(
                    state.rental_service_open
                )
            for key in sorted(keys):
                request_index, units = indexed[key]
                service_key, size, request_duration = key
                if not active:
                    state.current_day_ledger["internal_basket_intents"].append(
                        {
                            "customer_event_address": list(event_address),
                            "request_index": request_index,
                            "service_key": service_key,
                            "intent_role": request_roles[key],
                            "status": "suppressed",
                            "reason": suppression_reason,
                        }
                    )
                    continue
                session = self._make_choice_session(
                    state,
                    service_key=service_key,
                    units_requested=units,
                    duration_days=request_duration,
                    adults=adults,
                    children=children,
                    size_class=size,
                    archetype=archetype,
                    weather=weather,
                    request_index=request_index,
                    event_address=event_address,
                    intent_role=request_roles[key],
                )
                requested += units
                if channel_stage:
                    if stage_snapshot is None:
                        raise AssertionError("active channel stage lacks snapshot")
                    precomputed = None
                    if not (
                        session.duration_days is not None
                        and state.day + session.duration_days > state.max_days
                    ):
                        precomputed = self.explain_channel_demand(
                            stage_snapshot, session
                        )
                    result = self._serve_choice_session_channel(
                        state,
                        session,
                        outcomes,
                        precomputed=precomputed,
                    )
                else:
                    served = self._serve_choice_session(state, session, outcomes)
                    result = ChoiceServiceResult(
                        fulfilled_units=served,
                        request_satisfied=served == units,
                        internal_channel="store" if served == units else "abandon",
                    )
                fulfilled += result.fulfilled_units
                results.append(result)
                state.current_day_ledger["internal_basket_intents"].append(
                    {
                        "customer_event_address": list(event_address),
                        "request_index": request_index,
                        "service_key": service_key,
                        "intent_role": request_roles[key],
                        "status": "activated",
                        "satisfied": result.request_satisfied,
                    }
                )
            return results

        all_keys = list(indexed)
        if not self.demand_strategy.channel_enabled:
            serve_stage(all_keys, active=True)
            return requested, fulfilled, outcomes

        if archetype == "overnight":
            shelter = [key for key in all_keys if key[0] == "rental:tent"]
            sleep = [
                key
                for key in all_keys
                if key[0] in {"rental:sleeping_bag", "rental:sleeping_pad"}
            ]
            optional = [key for key in all_keys if key not in shelter + sleep]
            shelter_results = serve_stage(shelter, active=True)
            shelter_ok = bool(shelter_results) and all(
                row.request_satisfied for row in shelter_results
            )
            sleep_results = serve_stage(
                sleep,
                active=shelter_ok,
                suppression_reason="required_shelter_abandoned",
            )
            sleep_ok = shelter_ok and bool(sleep_results) and all(
                row.request_satisfied for row in sleep_results
            )
            serve_stage(
                optional,
                active=sleep_ok,
                suppression_reason="required_sleep_stage_abandoned",
            )
        elif archetype == "lightweight":
            sleep = [
                key
                for key in all_keys
                if key[0] in {"rental:sleeping_bag", "rental:sleeping_pad"}
            ]
            optional = [key for key in all_keys if key not in sleep]
            sleep_results = serve_stage(sleep, active=True)
            sleep_ok = bool(sleep_results) and all(
                row.request_satisfied for row in sleep_results
            )
            serve_stage(
                optional,
                active=sleep_ok,
                suppression_reason="required_sleep_stage_abandoned",
            )
        elif archetype == "picnic":
            core = [
                key
                for key in all_keys
                if key[0] in {"rental:camp_table", "rental:camp_chair"}
            ]
            optional = [key for key in all_keys if key not in core]
            core_results = serve_stage(core, active=True)
            viable = any(row.request_satisfied for row in core_results)
            serve_stage(
                optional,
                active=viable,
                suppression_reason="picnic_core_abandoned",
            )
        else:
            serve_stage(all_keys, active=True)
        return requested, fulfilled, outcomes

    def _calendar_arrival_lambda(
        self, archetype: str, calendar: dict[str, Any]
    ) -> float:
        school_structure = {
            "overnight": 0.78,
            "picnic": 1.18,
            "lightweight": 0.90,
            "supply": 1.14,
        }
        weekday_lambda, weekend_lambda = self.rules.archetype_base_lambdas[archetype]
        day_type = str(calendar["day_type"])
        if day_type == "public_holiday":
            holiday_id = str(calendar.get("holiday_id") or "")
            try:
                multiplier = self.rules.holiday_arrival_multipliers[holiday_id]
            except KeyError as error:
                raise ValueError(
                    f"public holiday lacks an arrival multiplier: {holiday_id!r}"
                ) from error
            return weekday_lambda * multiplier
        if bool(calendar["is_weekend"]):
            return weekend_lambda
        if day_type == "school_holiday":
            return weekday_lambda * school_structure[archetype]
        return weekday_lambda

    def _simulate_day(self, state: WorldState) -> None:
        active_archetypes = tuple(
            archetype
            for archetype in self.rules.archetype_base_lambdas
            if self.config.retail_enabled or archetype != "supply"
        )
        if state.day > self.config.max_days:
            state.current_day_ledger["arrivals"] = {
                archetype: 0 for archetype in active_archetypes
            }
            return
        weather = self._weather(state.day)
        demand_weather = self._demand_weather_regime(state.day)
        calendar = self._calendar(state.day)
        source_day = self._source_weather_day(state.day)
        season = seasonal_index(source_day, self.rules.seasonal_anchors)
        counts: dict[str, int] = {}
        for archetype in active_archetypes:
            archetype_index = _ARCHETYPE_INDEX[archetype]
            rate = self._calendar_arrival_lambda(archetype, calendar)
            rate *= self.rules.arrival_scale * self.config.arrival_scale_multiplier
            rate *= season
            rate *= self.rules.weather_multipliers[demand_weather][archetype]
            count = (
                self._potential_arrival_count(state.day, archetype)
                if self._v42_weather_information_coherence()
                else int(
                    self._event_rng(
                        _RNG_ARRIVALS, state.day, archetype_index
                    ).poisson(rate)
                )
            )
            actual_arrivals = 0
            for customer_index in range(count):
                if not self._v42_weather_information_coherence():
                    state.cumulative_customers += 1
                routed_to_reservation, _lead = self._reservation_route(
                    state.day, archetype_index, customer_index
                )
                if archetype != "supply" and routed_to_reservation:
                    # The same latent trip appeared earlier on the advance-booking
                    # tape (possibly choosing an outside channel). It must not be
                    # duplicated as a walk-in on pickup day.
                    continue
                if (
                    self._v42_weather_information_coherence()
                    and not self._customer_weather_considers(
                        archetype=archetype,
                        regime=demand_weather,
                        event_address=(state.day, archetype_index, customer_index),
                        forecast_trust=None,
                        channel="walk-in",
                    )
                ):
                    continue
                actual_arrivals += 1
                if self._v42_weather_information_coherence():
                    state.cumulative_customers += 1
                requested, fulfilled, outcomes = self._serve_customer_v5(
                    state,
                    self._event_rng(
                        _RNG_CUSTOMER_BASKET,
                        state.day,
                        archetype_index,
                        customer_index,
                    ),
                    archetype,
                    weather,
                    demand_weather,
                    event_address=(state.day, archetype_index, customer_index),
                )
                if fulfilled:
                    state.cumulative_transactions += 1
                if requested > 0 and requested == fulfilled:
                    state.current_day_ledger["demand"]["full_baskets"] += 1
                    state.cumulative_full_baskets += 1
                elif fulfilled > 0:
                    state.current_day_ledger["demand"]["partial_baskets"] += 1
                    state.cumulative_partial_baskets += 1
                else:
                    state.current_day_ledger["demand"]["unfilled_baskets"] += 1
                    state.cumulative_unfilled_baskets += 1
                operational_failures = (
                    outcomes["stockout"]
                    + outcomes["service_closed"]
                    + outcomes["unpriced"]
                )
                operational_units = outcomes["fulfilled"] + operational_failures
                if operational_units > 0:
                    demand = state.current_day_ledger["demand"]
                    demand["operationally_evaluable_baskets"] += 1
                    state.cumulative_operationally_evaluable_baskets += 1
                    if operational_failures == 0:
                        demand["operational_full_baskets"] += 1
                        state.cumulative_operational_full_baskets += 1
                    elif outcomes["fulfilled"] > 0:
                        demand["operational_partial_baskets"] += 1
                        state.cumulative_operational_partial_baskets += 1
                    else:
                        demand["operational_unfilled_baskets"] += 1
                        state.cumulative_operational_unfilled_baskets += 1
            counts[archetype] = (
                actual_arrivals
                if self._v42_weather_information_coherence()
                else count
            )
        state.current_day_ledger["arrivals"] = counts
