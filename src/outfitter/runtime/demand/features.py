from __future__ import annotations

from ...reservation_difficulty import rental_amount

from ..common import (
    ACCESSIBILITY_ADJUSTMENTS,
    CHANNEL_QUALITY_SCORES,
    CHOICE_GROUPS_BY_SERVICE,
    ChannelDecisionTrace,
    ChannelEvaluationInput,
    ChannelOfferView,
    ChoiceSession,
    DemandGroupParameters,
    NEED_ADJUSTMENTS,
    QUALITY_SALIENCE,
    QUALITY_SCORES,
    ReferenceDecisionTrace,
    ReferenceEvaluationInput,
    ReferenceOfferView,
    SUBSTITUTE_PENALTIES,
    WorldState,
    _RNG_CHANNEL_AFFINITY,
    _RNG_CHANNEL_BOUNDED_TASTE,
    _RNG_CUSTOMER_SEGMENT,
    _RNG_EXISTING_OPTION,
    _RNG_GROUP_SHOCK,
    _RNG_OFFER_SHOCK,
    _RNG_PROFILE_TASTE,
    _RNG_RENTAL_CONTEXT,
    _RNG_REQUEST_BUDGET,
    _RNG_SERVICE_SHOCK,
    _RNG_SESSION_BUDGET,
    _RNG_SESSION_PREFERENCE,
    _RNG_SUBSTITUTION,
    evaluate_channel_demand,
    evaluate_reference_demand,
    functional_quantity_scale,
    hashlib,
    json,
    math,
    np,
)


class DemandFeaturesMixin:
    def _physical_available_offer_units(
        self, state: WorldState, offer_id: str
    ) -> int:
        offer = self.offers[offer_id]
        if offer.asset_kind == "consumable":
            return int(state.retail_inventory.get(offer_id, {}).get("quantity", 0))
        return sum(
            asset.offer_id == offer_id and asset.status == "available"
            and (not self._v31_warehouse_capacity() or self._warehouse_asset_held(asset))
            for asset in state.assets
        )

    @staticmethod
    def _walk_in_occupied_units(
        state: WorldState, offer_id: str, day: int
    ) -> int:
        return sum(
            contract.status == "active"
            and contract.reservation_id is None
            and contract.offer_id == offer_id
            and contract.start_day <= day < contract.return_day
            for contract in state.contracts
        )

    @staticmethod
    def _walk_in_capacity_limit(
        state: WorldState, offer_id: str, day: int
    ) -> int | None:
        return next(
            (
                int(row.max_walk_in_occupied_units)
                for row in state.walk_in_capacity_limits
                if row.offer_id == offer_id and int(row.day) == int(day)
            ),
            None,
        )

    @staticmethod
    def _compressed_walk_in_capacity_limit_intervals(
        state: WorldState,
        *,
        offer_ids: set[str] | None = None,
        from_day: int | None = None,
        to_day: int | None = None,
    ) -> list[dict[str, Any]]:
        selected = offer_ids or set()
        by_offer: dict[str, list[tuple[int, int]]] = {}
        for row in state.walk_in_capacity_limits:
            if selected and row.offer_id not in selected:
                continue
            if from_day is not None and int(row.day) < int(from_day):
                continue
            if to_day is not None and int(row.day) > int(to_day):
                continue
            by_offer.setdefault(row.offer_id, []).append(
                (int(row.day), int(row.max_walk_in_occupied_units))
            )
        result: list[dict[str, Any]] = []
        for offer_id in sorted(by_offer):
            values = sorted(by_offer[offer_id])
            start = previous_day = values[0][0]
            maximum = values[0][1]
            for day, value in values[1:] + [(values[-1][0] + 2, -1)]:
                if day == previous_day + 1 and value == maximum:
                    previous_day = day
                    continue
                result.append(
                    {
                        "offer_id": offer_id,
                        "from_day": start,
                        "to_day": previous_day,
                        "max_walk_in_occupied_units": maximum,
                    }
                )
                start = previous_day = day
                maximum = value
        return result

    def _walk_in_capacity_slack(
        self, state: WorldState, session: ChoiceSession, offer_id: str
    ) -> int | None:
        if not self._v24_reservation_operations():
            return None
        if int(session.booking_lead_days) > 0:
            return None
        duration = max(1, int(session.duration_days or 1))
        slacks: list[int] = []
        for day in range(state.day, state.day + duration):
            limit = self._walk_in_capacity_limit(state, offer_id, day)
            if limit is None:
                continue
            occupied = self._walk_in_occupied_units(state, offer_id, day)
            slacks.append(max(0, int(limit) - int(occupied)))
        return min(slacks) if slacks else None

    def _available_offer_units(
        self,
        state: WorldState,
        offer_id: str,
        *,
        session: ChoiceSession | None = None,
        reopen_closed: bool = False,
        ignore_walk_in_capacity: bool = False,
        restore_physical: bool = False,
    ) -> int:
        offer = self.offers[offer_id]
        physical = self._physical_available_offer_units(state, offer_id)
        if offer.asset_kind == "consumable":
            if restore_physical and session is not None:
                return max(physical, self._required_resources(session, offer_id))
            return physical
        if not reopen_closed and not state.rental_service_open.get(offer_id, False):
            return 0
        if session is not None and self._v37_supplier_trial():
            return_day = state.day + max(1, int(session.duration_days or 1))
            physical = sum(
                asset.offer_id == offer_id
                and asset.status == "available"
                and (
                    not self._v31_warehouse_capacity()
                    or self._warehouse_asset_held(asset)
                )
                and self._supplier_trial_asset_can_start_contract(
                    state, asset, return_day=return_day
                )
                for asset in state.assets
            )
        if restore_physical and session is not None:
            physical = max(physical, self._required_resources(session, offer_id))
        if session is not None and not ignore_walk_in_capacity:
            slack = self._walk_in_capacity_slack(state, session, offer_id)
            if slack is not None:
                physical = min(physical, slack)
        return physical

    def _required_resources(self, session: ChoiceSession, offer_id: str) -> int:
        offer = self.offers[offer_id]
        if offer.asset_kind == "consumable":
            return session.units_requested
        if session.service_key == "rental:sleeping_pad":
            slots = max(1, int(offer.decision_attributes.get("sleep_slots", 1)))
            return math.ceil(session.units_requested / slots)
        if session.service_key in {"rental:sleeping_bag", "rental:camp_chair"}:
            return session.units_requested
        return 1

    def _line_price(self, state: WorldState, session: ChoiceSession, offer_id: str) -> float:
        offer = self.offers[offer_id]
        resources = self._required_resources(session, offer_id)
        if offer.asset_kind == "consumable":
            return max(0.01, resources * float(state.prices[offer_id]))
        duration = max(1, int(session.duration_days or 1))
        return max(
            0.01,
            resources
            * (
                (rental_amount(float(state.prices[offer_id]), duration)
                 if self._v45_reservation_difficulty()
                 else float(state.prices[offer_id]) * duration)
                + self.rules.deposit_friction * float(state.deposits[offer_id])
            ),
        )

    def _binding_night_low(self, state: WorldState, duration_days: int) -> float:
        last = min(state.max_days, state.day + max(1, duration_days) - 1)
        return min(self._temperatures(day)[1] for day in range(state.day, last + 1))

    def _functional_attribute_fit(
        self, state: WorldState, session: ChoiceSession, offer_id: str
    ) -> float:
        offer = self.offers[offer_id]
        attributes = offer.decision_attributes
        fit = 0.0
        party = session.adults + session.children
        if offer.category_id == "tent":
            capacity = int(attributes.get("capacity_persons", 2))
            comfortable = int(
                attributes.get("comfortable_capacity_persons", capacity)
            )
            if capacity < party:
                fit -= min(3.6, 1.2 * (party - capacity))
            elif party > comfortable:
                fit -= min(0.9, 0.45 * (party - comfortable))
            elif comfortable > party + 2:
                fit -= 0.12 * (comfortable - party - 2)
        elif offer.category_id == "sleeping_bag":
            low = self._customer_night_low(
                state, int(session.duration_days or 1), session.trip_weather_context
            )
            comfort = float(attributes.get("comfort_temp_c", 10.0))
            if comfort > low:
                fit -= min(3.0, (comfort - low) / 5.0)
            elif low - comfort > 12:
                fit -= 0.25
            offered_size = str(attributes.get("size_class", "adult_regular"))
            if session.size_class == "child" and offered_size != "child":
                fit -= 0.7
            if session.size_class != "child" and offered_size == "child":
                fit -= 3.0
        elif offer.category_id == "camp_lantern":
            runtime = float(attributes.get("runtime_hours", 24.0))
            needed = 8.0 * max(1, int(session.duration_days or 1))
            if runtime < needed:
                fit -= min(1.5, (needed - runtime) / 12.0)
        elif offer.category_id == "camp_cookware":
            functions = set(map(str, attributes.get("functions", [])))
            needed = {
                "hot_drink": {"boil"},
                "cook_meal": {"boil", "stew"},
                "fry_meal": {"fry"},
            }.get(session.use_case, {"boil"})
            fit -= 1.2 * len(needed - functions)
        elif offer.category_id == "rain_gear":
            offered_size = str(attributes.get("size_class", "adult"))
            if offered_size not in {
                session.size_class,
                "adult_child_universal",
            }:
                fit -= 1.5 if session.size_class == "child" else 3.0
        return fit

    def _attribute_fit(
        self, state: WorldState, session: ChoiceSession, offer_id: str
    ) -> float:
        """Legacy fit used by the archived nested-logit demand model."""

        offer = self.offers[offer_id]
        tier = offer.perceived_quality_tier or "standard"
        return self._functional_attribute_fit(
            state, session, offer_id
        ) + float(self.rules.quality_values.get(tier, 0.0))

    def _functional_quantity_scale(self, offer_id: str) -> float:
        offer = self.offers[offer_id]
        return functional_quantity_scale(
            offer.category_id,
            offer.choice_group,
            offer.consumer_value_attributes,
        )

    def _consumer_profile_signature(
        self, state: WorldState, offer_id: str
    ) -> str:
        offer = self.offers[offer_id]
        payload = {
            "brand": offer.brand,
            "choice_group": offer.choice_group,
            "consumer_unit": offer.consumer_unit,
            "decision_attributes": offer.decision_attributes,
            "consumer_value_attributes": offer.consumer_value_attributes,
            "perceived_quality_tier": offer.perceived_quality_tier,
            "posted_price": round(float(state.prices[offer_id]), 2),
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]

    def _channel_profile_signature(self, offer_id: str) -> str:
        """Behavioral identity shared across channels and independent of prices."""

        offer = self.offers[offer_id]
        payload = {
            "brand": offer.brand,
            "choice_group": offer.choice_group,
            "consumer_unit": offer.consumer_unit,
            "decision_attributes": offer.decision_attributes,
            "consumer_value_attributes": offer.consumer_value_attributes,
            "perceived_quality_tier": offer.perceived_quality_tier,
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]

    def _reservation_context_keys(self, session: ChoiceSession) -> tuple[str, ...]:
        keys = [
            {
                "overnight": "overnight_camping",
                "picnic": "day_picnic",
                "lightweight": "solo_hiking",
                "supply": "supply_stop",
            }.get(session.archetype, session.visible_party_context)
        ]
        if session.service_key == "retail:rain_gear" and not session.planned_purchase:
            keys.append("emergency_rain")
        if session.service_key == "retail:bottled_water" and self._weather(
            session.event_address[0]
        ) == "hot":
            keys.append("hot_water_need")
        return tuple(dict.fromkeys(keys))

    def _reservation_scale(
        self,
        state: WorldState,
        session: ChoiceSession,
        offer_id: str,
        group: DemandGroupParameters,
        base_reservation: float,
        profile_signature: str,
    ) -> float:
        model = self.demand_model
        market = self.episode_market_profile
        if model is None or market is None:
            raise AssertionError("bounded reservation model is unavailable")
        segment = model.customer_segments[session.customer_segment]
        local = market.groups[group.key]
        context_keys = self._reservation_context_keys(session)
        access_score = (
            float(model.accessibility_scores[group.alternative_accessibility])
            + local.accessibility_shift
            + sum(
                float(model.context_accessibility_scores.get(key, 0.0))
                for key in context_keys
            )
        )
        access_factor = math.exp(-0.25 * access_score)
        access_factor = min(
            model.accessibility_factor_range[1],
            max(model.accessibility_factor_range[0], access_factor),
        )
        urgency_score = sum(
            float(model.context_urgency_scores.get(key, 0.0))
            for key in context_keys
        )
        need_salience = max(
            0.0,
            float(model.need_salience_values[group.need_strength]) + local.need_shift,
        )
        urgency_factor = min(1.60, max(0.80, 1.0 + need_salience * urgency_score))
        fit = self._functional_attribute_fit(state, session, offer_id)
        if str(self.offers[offer_id].choice_group) != session.preferred_group:
            fit -= {
                "low": 1.20,
                "medium": 0.70,
                "high": 0.35,
            }[group.substitution_strength]
        functional_factor = math.exp(
            0.35 * float(segment.functional_weight) * min(1.0, max(-3.6, fit))
        )
        offer = self.offers[offer_id]
        quality_score = float(
            model.quality_scores.get(offer.perceived_quality_tier or "unknown", 0.0)
        )
        quality_factor = math.exp(
            0.18
            * float(segment.quality_weight)
            * float(model.quality_salience_values[group.quality_salience])
            * quality_score
        )
        taste_rng = self._stable_event_rng(
            _RNG_PROFILE_TASTE,
            *session.event_address,
            identity=profile_signature,
        )
        profile_taste = float(taste_rng.uniform(*model.profile_taste_range))
        return (
            base_reservation
            * float(segment.reservation_multiplier)
            * local.reservation_multiplier
            * urgency_factor
            * access_factor
            * self._functional_quantity_scale(offer_id)
            * functional_factor
            * quality_factor
            * profile_taste
        )

    def _reservation_demand_units(
        self, session: ChoiceSession, offer_id: str, group: DemandGroupParameters
    ) -> float:
        if group.value_unit_kind == "retail_sale_unit":
            return float(session.units_requested)
        duration = float(max(1, int(session.duration_days or 1)))
        if group.value_unit_kind == "requested_unit_day":
            return float(session.units_requested) * duration
        return float(self._required_resources(session, offer_id)) * duration

    def neutral_reservation_value(self, offer_id: str) -> float | None:
        """Internal routine-scenario central WTP used by builders and calibration."""

        model = self.demand_model
        if model is None:
            return None
        offer = self.offers[offer_id]
        service_key = self.service_key_for_product_type(offer.service_line)
        group = model.group(service_key, str(offer.choice_group))
        balanced = model.customer_segments["balanced"]
        access_factor = math.exp(
            -0.25
            * float(
                model.accessibility_scores[group.alternative_accessibility]
            )
        )
        access_factor = min(
            model.accessibility_factor_range[1],
            max(model.accessibility_factor_range[0], access_factor),
        )
        quality_score = float(
            model.quality_scores.get(offer.perceived_quality_tier or "unknown", 0.0)
        )
        quality_factor = math.exp(
            0.18
            * float(balanced.quality_weight)
            * float(model.quality_salience_values[group.quality_salience])
            * quality_score
        )
        value = (
            group.reservation_central_cny
            * self._functional_quantity_scale(offer_id)
            * access_factor
            * quality_factor
        )
        if group.value_unit_kind == "requested_unit_day" and offer.category_id == "sleeping_pad":
            value *= max(
                1.0, float(offer.decision_attributes.get("sleep_slots", 1))
            )
        return value

    def neutral_demand_price(
        self,
        state: WorldState,
        offer_id: str,
        *,
        duration_days: int = 1,
    ) -> float | None:
        """Internal price whose generalized transaction cost is at reference.

        Agent tools and public scripted strategies never expose this value.  It
        exists only so the explicitly oracle calibration strategy uses the same
        demand scale as the active production evaluator.
        """

        model = self.reference_demand_model
        if model is None or self.config.demand_world_id == "control":
            return self.neutral_reservation_value(offer_id)
        cell_id = model.offer_assignments.get(offer_id)
        if cell_id is None:
            return None
        offer = self.offers[offer_id]
        duration = 0 if offer.asset_kind == "consumable" else max(
            1, int(duration_days)
        )
        generalized = model.cells[cell_id].generalized_unit_cost(duration)
        shop_deposit = (
            0.0
            if offer.asset_kind == "consumable"
            else float(state.deposits[offer_id])
        )
        cash_equivalent = max(0.01, generalized - 0.05 * shop_deposit)
        return cash_equivalent if duration == 0 else cash_equivalent / duration

    def _offer_utility(
        self, state: WorldState, session: ChoiceSession, offer_id: str
    ) -> float:
        return self._attribute_fit(state, session, offer_id) - session.price_beta * math.log(
            self._line_price(state, session, offer_id) / max(0.01, session.budget)
        )

    def _choice_shock(self, session: ChoiceSession, level: str, identity: str) -> float:
        stream = {
            "service": _RNG_SERVICE_SHOCK,
            "group": _RNG_GROUP_SHOCK,
            "offer": _RNG_OFFER_SHOCK,
        }[level]
        stable_identity = (
            self._offer_rng_identity(identity) if level == "offer" else identity
        )
        return self.random.gumbel(
            stream,
            tuple(session.event_address),
            f"{level}:{stable_identity}",
        )

    def _preferred_group(
        self,
        state: WorldState,
        service_key: str,
        archetype: str,
        weather: str,
        duration_days: int | None,
        portability_sensitive: bool,
        planned_purchase: bool,
        preference_rng: np.random.Generator,
        trip_weather_context: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        groups = self.public_choice_groups().get(service_key) or list(
            CHOICE_GROUPS_BY_SERVICE.get(service_key, ())
        )
        if not groups:
            raise ValueError(f"service has no choice groups: {service_key}")
        use_case = "general"
        if service_key == "rental:tent":
            return "compact_tent", "shelter"
        if service_key == "rental:sleeping_bag":
            low = self._customer_night_low(
                state, int(duration_days or 1), trip_weather_context
            )
            return (
                ("three_season_bag", "cool_night")
                if low <= 15.0
                else ("warm_weather_bag", "warm_night")
            )
        if service_key == "rental:sleeping_pad":
            if portability_sensitive:
                return "closed_cell_foam_pad", "portable_sleep"
            draw = float(preference_rng.random())
            return (
                ("self_inflating_pad", "comfort_sleep")
                if draw < 0.55
                else ("air_mattress", "comfort_sleep")
                if draw < 0.82
                else ("closed_cell_foam_pad", "simple_sleep")
            )
        if service_key == "rental:camp_chair":
            return (
                ("lightweight_chair", "portable_seating")
                if portability_sensitive or archetype == "lightweight"
                else ("comfort_chair", "comfort_seating")
            )
        if service_key == "retail:rain_gear":
            return (
                ("reusable_rainwear", "planned_rain")
                if planned_purchase
                else ("disposable_poncho", "emergency_rain")
            )
        if service_key == "retail:repellent":
            return (
                ("topical_repellent", "direct_protection")
                if preference_rng.random() < 0.65
                else ("wearable_repellent", "wearable_protection")
            )
        if service_key == "retail:energy_snacks":
            return (
                ("compressed_ration", "trip_ration")
                if archetype in {"overnight", "lightweight"}
                else ("snack_portion", "casual_snack")
            )
        if service_key == "rental:camp_cookware":
            use_case = str(
                preference_rng.choice(["hot_drink", "cook_meal", "fry_meal"], p=[0.3, 0.5, 0.2])
            )
        return groups[0], use_case

    def _make_choice_session(
        self,
        state: WorldState,
        *,
        service_key: str,
        units_requested: int,
        duration_days: int | None,
        adults: int,
        children: int,
        size_class: str,
        archetype: str,
        weather: str,
        request_index: int,
        event_address: tuple[int, int, int],
        intent_role: str = "optional",
        trip_weather_context: dict[str, Any] | None = None,
    ) -> ChoiceSession:
        if self._v44_trip_weather() and service_key.startswith("rental:"):
            if trip_weather_context is None:
                trip_weather_context = self._trip_weather_context(
                    state, pickup_day=state.day, duration_days=int(duration_days or 1))
        address = (*event_address, request_index)
        preference_rng = self._event_rng(_RNG_SESSION_PREFERENCE, *address)
        budget_rng = self._event_rng(_RNG_SESSION_BUDGET, *address)
        substitution_rng = self._event_rng(_RNG_SUBSTITUTION, *address)
        portability_sensitive = bool(
            archetype == "lightweight" or preference_rng.random() < 0.22
        )
        planned_purchase = bool(
            archetype != "supply" and preference_rng.random() < 0.55
        )
        preferred, use_case = self._preferred_group(
            state,
            service_key,
            archetype,
            weather,
            duration_days,
            portability_sensitive,
            planned_purchase,
            preference_rng,
            trip_weather_context,
        )
        if service_key == "rental:tent":
            preferred = "family_tent" if adults + children >= 4 else "compact_tent"
        party_effective = adults + 0.6 * children
        proportional = service_key in {
            "rental:sleeping_bag",
            "rental:sleeping_pad",
            "rental:camp_chair",
            "retail:rain_gear",
            "retail:bottled_water",
            "retail:energy_snacks",
            "retail:instant_food",
            "retail:cleaning_wipes",
            "retail:garbage_bags",
        }
        scale = float(units_requested) if proportional else 1.0 + 0.42 * max(0.0, party_effective - 1.0)
        if service_key.startswith("rental:"):
            scale *= max(1, int(duration_days or 1))
        customer_segment = "balanced"
        if (
            self.demand_model is None
            and self.reference_demand_model is None
            and self.channel_demand_model is None
        ):
            base_budget = float(self.rules.base_budgets[service_key]) * scale
            budget = base_budget * math.exp(float(budget_rng.normal(0.0, 0.25)))
            price_beta = self.rules.price_beta_mean * math.exp(
                float(
                    budget_rng.normal(
                        -0.5 * self.rules.price_beta_sigma**2,
                        self.rules.price_beta_sigma,
                    )
                )
            )
        else:
            segment_rng = self._event_rng(_RNG_CUSTOMER_SEGMENT, *address)
            if self.channel_demand_model is not None:
                keys = sorted(self.channel_demand_model.segments)
                probabilities = [
                    self.channel_demand_model.segments[key].share for key in keys
                ]
            elif self.reference_demand_model is not None:
                keys = sorted(self.reference_demand_model.segments)
                probabilities = [
                    self.reference_demand_model.segments[key].share for key in keys
                ]
            else:
                if self.episode_market_profile is None:
                    raise AssertionError("episode market profile is missing")
                keys = sorted(self.episode_market_profile.segment_shares)
                probabilities = [
                    self.episode_market_profile.segment_shares[key] for key in keys
                ]
            customer_segment = str(segment_rng.choice(keys, p=probabilities))
            budget = 0.0
            price_beta = 0.0
        visible_context = (
            "family_with_children"
            if children
            else {
                "overnight": "overnight_camping",
                "picnic": "day_picnic",
                "lightweight": "solo_hiking",
                "supply": "supply_stop",
            }[archetype]
        )
        need_horizon_days, occasion_importance = self._retail_deadline_context(
            service_key=service_key,
            use_case=use_case,
            archetype=archetype,
            weather=weather,
            planned_purchase=planned_purchase,
            intent_role=intent_role,
        )
        rental_context_rng = self._stable_event_rng(
            _RNG_RENTAL_CONTEXT,
            *address,
            identity=service_key,
        )
        booking_lead_days = 0
        forecast_adversity = {
            "sunny": 0.0,
            "cloudy": 0.18,
            "hot": 0.35,
            "cold": 0.45,
            "rainy": 0.82,
            "snowy": 1.0,
        }.get(weather, 0.2)
        forecast_trust = float(rental_context_rng.uniform(0.30, 1.0))
        storage_burden = float(rental_context_rng.uniform(0.0, 1.0))
        future_use_base = {
            "overnight": 0.30,
            "lightweight": 0.35,
            "picnic": 0.12,
            "supply": 0.08,
        }[archetype]
        future_use_width = {
            "overnight": 0.45,
            "lightweight": 0.45,
            "picnic": 0.38,
            "supply": 0.30,
        }[archetype]
        future_use_expectation = min(
            1.0,
            future_use_base
            + future_use_width * float(rental_context_rng.random()),
        )
        return_burden_multiplier = float(
            rental_context_rng.uniform(0.70, 1.35)
        )
        budget_sigma = (
            float(self.channel_demand_model.request_budget_sigma)
            if self.channel_demand_model is not None
            and self.channel_demand_model.format
            == "outfitter-exact-offer-channel-demand-v5-request-bundle"
            else 0.0
        )
        request_budget_rng = self._stable_event_rng(
            _RNG_REQUEST_BUDGET,
            *event_address,
            identity="request-household-budget-v1",
        )
        request_budget_multiplier = math.exp(
            float(
                request_budget_rng.normal(
                    -0.5 * budget_sigma**2,
                    budget_sigma,
                )
            )
        )
        if self.channel_demand_model is not None and self.channel_demand_model.request_budget_bounds is not None:
            from ...autonomous_rentals import truncated_budget
            lower,upper=self.channel_demand_model.request_budget_bounds
            request_budget_multiplier=truncated_budget(float(self._stable_event_rng(
                _RNG_REQUEST_BUDGET,*event_address,identity="request-household-budget-v47").random()),
                sigma=budget_sigma,lower=lower,upper=upper)
        if service_key.startswith("rental:"):
            occasion_importance = 1.10 if intent_role == "required" else 0.75
        return ChoiceSession(
            interaction_id=f"ci-{state.day:03d}-{event_address[1]:02d}-{event_address[2]:04d}-{request_index:02d}",
            service_key=service_key,
            units_requested=max(1, int(units_requested)),
            duration_days=duration_days,
            adults=adults,
            children=children,
            size_class=size_class,
            visible_party_context=visible_context,
            use_case=use_case,
            preferred_group=preferred,
            budget=round(max(0.01, budget), 4),
            price_beta=round(max(0.05, price_beta), 6),
            substitution_tolerance=round(float(substitution_rng.uniform(0.2, 0.9)), 6),
            portability_sensitive=portability_sensitive,
            planned_purchase=planned_purchase,
            event_address=address,
            archetype=archetype,
            customer_segment=customer_segment,
            intent_role=intent_role,
            need_horizon_days=need_horizon_days,
            occasion_importance=occasion_importance,
            booking_lead_days=booking_lead_days,
            pickup_day=state.day,
            forecast_adversity=forecast_adversity,
            forecast_uncertainty=0.0,
            forecast_trust=forecast_trust,
            future_use_expectation=future_use_expectation,
            storage_burden=storage_burden,
            return_burden_multiplier=return_burden_multiplier,
            request_budget_multiplier=request_budget_multiplier,
            trip_weather_context=trip_weather_context,
        )

    @staticmethod
    def _retail_deadline_context(
        *,
        service_key: str,
        use_case: str,
        archetype: str,
        weather: str,
        planned_purchase: bool,
        intent_role: str,
    ) -> tuple[int | None, float]:
        if not service_key.startswith("retail:"):
            return None, 1.0
        required = intent_role == "required"
        importance = 1.0 if required else 0.45
        emergency_rain = (
            service_key == "retail:rain_gear"
            and (use_case == "emergency_rain" or not planned_purchase)
        )
        if emergency_rain:
            importance *= 2.50 if weather in {"rainy", "snowy"} else 1.50
            return 0, round(importance, 6)
        # Optional add-ons have no known occasion deadline.  Treating every
        # supply-stop basket line as same-day urgent made generic wipes, bags,
        # and snacks inherit the economics of a forgotten raincoat.
        if not required:
            return None, round(importance, 6)
        if planned_purchase:
            if service_key == "retail:rain_gear":
                importance *= 0.85
            return 3, round(importance, 6)
        if service_key == "retail:bottled_water":
            importance *= 0.80
            return (0 if archetype == "supply" else 1), round(importance, 6)
        if service_key in {
            "retail:energy_snacks",
            "retail:instant_food",
            "retail:repellent",
        }:
            importance *= 0.70
            return 1, round(importance, 6)
        if service_key in {
            "retail:cleaning_wipes",
            "retail:garbage_bags",
        }:
            importance *= 0.45
            return 1, round(importance, 6)
        return 1, round(importance, 6)

    def _preferred_availability(
        self, state: WorldState, session: ChoiceSession
    ) -> tuple[str, int]:
        offers = self.by_group.get((session.service_key, session.preferred_group), [])
        compatible_offers = [
            offer
            for offer in offers
            if self._reference_hard_compatibility(
                state, session, offer.offer_id
            )[0]
        ]
        available = sum(
            self._available_offer_units(
                state, offer.offer_id, session=session
            )
            for offer in compatible_offers
        )
        if available > 0:
            return "available", available
        carried = any(
            offer.offer_id in state.ever_carried_offer_ids
            for offer in compatible_offers
        )
        return ("stocked_out" if carried else "never_carried"), 0

    def _reference_hard_compatibility(
        self, state: WorldState, session: ChoiceSession, offer_id: str
    ) -> tuple[bool, tuple[str, ...]]:
        """Reject only physically impossible request/offer combinations."""

        offer = self.offers[offer_id]
        attributes = offer.decision_attributes
        reasons: list[str] = []
        party = session.adults + session.children
        if offer.category_id == "tent" and int(
            attributes.get("capacity_persons", 0)
        ) < party:
            reasons.append("tent_capacity_below_party_size")
        if offer.category_id == "sleeping_bag":
            offered = str(attributes.get("size_class", "adult_regular"))
            if session.size_class != "child" and offered == "child":
                reasons.append("adult_request_with_child_sleeping_bag")
        if offer.category_id == "rain_gear":
            offered = str(attributes.get("size_class", "adult"))
            if offered not in {
                session.size_class,
                "adult_child_universal",
            }:
                reasons.append("rainwear_size_class_mismatch")
        if (
            offer.category_id == "camp_chair"
            and session.adults > 0
            and any(token in f"{offer.name} {offer.source_variant_name}" for token in ("儿童", "小童"))
        ):
            reasons.append("adult_request_with_child_only_chair")
        return not reasons, tuple(reasons)

    def _reference_context_adjustment(self, session: ChoiceSession) -> float:
        value = {
            "overnight": 0.10,
            "picnic": 0.0,
            "lightweight": 0.10,
            "supply": -0.10,
        }.get(session.archetype, 0.0)
        if session.service_key == "retail:rain_gear" and not session.planned_purchase:
            value += 0.20
        if session.service_key == "retail:bottled_water" and self._weather(
            session.event_address[0]
        ) == "hot":
            value += 0.15
        return value

    def _reference_offer_views(
        self,
        state: WorldState,
        session: ChoiceSession,
        offer_ids: list[str],
    ) -> tuple[ReferenceOfferView, ...]:
        model = self.reference_demand_model
        if model is None:
            raise AssertionError("reference demand model is unavailable")
        result = []
        duration = max(1, int(session.duration_days or 1))
        for offer_id in sorted(offer_ids):
            offer = self.offers[offer_id]
            required = self._required_resources(session, offer_id)
            available = self._available_offer_units(
                state, offer_id, session=session
            )
            compatible, reasons = self._reference_hard_compatibility(
                state, session, offer_id
            )
            if offer.asset_kind == "consumable":
                shop_cash = required * float(state.prices[offer_id])
                shop_deposit = 0.0
            else:
                shop_cash = (
                    rental_amount(float(state.prices[offer_id]), duration, required)
                    if self._v45_reservation_difficulty()
                    else required * duration * float(state.prices[offer_id])
                )
                shop_deposit = required * float(state.deposits[offer_id])
            shop_generalized = shop_cash + 0.05 * shop_deposit
            cell_id = model.offer_assignments.get(offer_id)
            reference_cost: float | None = None
            if cell_id is not None and cell_id in model.cells:
                try:
                    reference_cost = required * model.cells[
                        cell_id
                    ].generalized_unit_cost(session.duration_days)
                except ValueError:
                    reference_cost = None
            ratio = (
                None
                if reference_cost is None
                else shop_generalized / max(0.01, reference_cost)
            )
            profile_payload = (
                f"{self._consumer_profile_signature(state, offer_id)}:{cell_id or 'missing'}"
            )
            profile_id = hashlib.sha256(
                profile_payload.encode("utf-8")
            ).hexdigest()[:24]
            result.append(
                ReferenceOfferView(
                    offer_id=offer_id,
                    profile_id=profile_id,
                    choice_group=str(offer.choice_group),
                    cell_id=cell_id,
                    available_units=available,
                    required_units=required,
                    hard_compatible=compatible,
                    incompatibility_reasons=reasons,
                    soft_fit=self._functional_attribute_fit(
                        state, session, offer_id
                    ),
                    quality_score=float(
                        QUALITY_SCORES.get(
                            offer.perceived_quality_tier or "unknown", 0.0
                        )
                    ),
                    shop_cash_price=round(shop_cash, 6),
                    shop_deposit=round(shop_deposit, 6),
                    reference_generalized_cost=(
                        None if reference_cost is None else round(reference_cost, 6)
                    ),
                    shop_generalized_cost=round(shop_generalized, 6),
                    price_ratio=None if ratio is None else float(ratio),
                )
            )
        return tuple(result)

    def explain_reference_demand(
        self,
        state: WorldState,
        session: ChoiceSession,
        offer_ids: list[str],
    ) -> tuple[ReferenceDecisionTrace, tuple[ReferenceOfferView, ...]]:
        model = self.reference_demand_model
        if model is None:
            raise ValueError("reference demand model is unavailable")
        world_id = self.config.demand_world_id
        if world_id == "control":
            raise ValueError("control world uses the bounded reservation adapter")
        group_key = f"{session.service_key}|{session.preferred_group}"
        group = model.groups[group_key]
        views = self._reference_offer_views(state, session, offer_ids)
        profile_ids = sorted({row.profile_id for row in views})
        trace = evaluate_reference_demand(
            model,
            ReferenceEvaluationInput(
                service_key=session.service_key,
                preferred_group=session.preferred_group,
                customer_segment=session.customer_segment,
                world_id=world_id,
                group_need_adjustment=NEED_ADJUSTMENTS[group.need_strength],
                accessibility_adjustment=ACCESSIBILITY_ADJUSTMENTS[
                    group.alternative_accessibility
                ],
                context_adjustment=self._reference_context_adjustment(session),
                local_shift=self.reference_local_shifts[group_key],
                quality_salience=QUALITY_SALIENCE[group.quality_salience],
                substitution_penalty=SUBSTITUTE_PENALTIES[
                    group.substitution_strength
                ],
                shop_gumbel=self._choice_shock(session, "service", "shop"),
                outside_gumbel=self._choice_shock(
                    session, "service", "outside"
                ),
                profile_gumbels={
                    profile_id: self._choice_shock(
                        session, "offer", profile_id
                    )
                    for profile_id in profile_ids
                },
                offers=views,
            ),
        )
        return trace, views

    def _select_quoted_offer_reference(
        self,
        state: WorldState,
        session: ChoiceSession,
        offer_ids: list[str],
    ) -> tuple[str, str, dict[str, float], float, float]:
        trace, _views = self.explain_reference_demand(
            state, session, offer_ids
        )
        by_profile = {str(row["profile_id"]): row for row in trace.profiles}
        chosen_profile = (
            trace.selected_profile_id
            if trace.selected_profile_id is not None
            else trace.best_profile_id
        )
        if chosen_profile is None:
            # This path is prevented by formal coverage validation. Preserve a
            # deterministic quote so the environment can record the rejection.
            selected_offer = min(offer_ids)
            selected_group = str(self.offers[selected_offer].choice_group)
            return selected_group, selected_offer, {}, float("-inf"), trace.outside_score
        selected_row = by_profile[chosen_profile]
        selected_offer = str(selected_row["representative_offer_id"])
        utilities = {offer_id: float("-inf") for offer_id in offer_ids}
        utilities.update({
            offer_id: float(row["choice_score"])
            for row in trace.profiles
            for offer_id in row["offer_ids"]
        })
        return (
            str(self.offers[selected_offer].choice_group),
            selected_offer,
            utilities,
            trace.buy_score,
            trace.outside_score,
        )

    def _channel_offer_views(
        self,
        state: WorldState,
        session: ChoiceSession,
        *,
        reopen_closed: bool = False,
        ignore_walk_in_capacity: bool = False,
        restore_physical: bool = False,
    ) -> tuple[ChannelOfferView, ...]:
        if self.channel_demand_model is None:
            raise AssertionError("channel demand model is unavailable")
        rows: list[ChannelOfferView] = []
        duration = max(1, int(session.duration_days or 1))
        for offer in sorted(
            self.by_service.get(session.service_key, []),
            key=lambda row: row.offer_id,
        ):
            online = offer.consumer_online_offer
            if not isinstance(online, dict):
                continue
            compatible, reasons = self._reference_hard_compatibility(
                state, session, offer.offer_id
            )
            resources = self._required_resources(session, offer.offer_id)
            available = self._available_offer_units(
                state,
                offer.offer_id,
                session=session,
                reopen_closed=reopen_closed,
                ignore_walk_in_capacity=ignore_walk_in_capacity,
                restore_physical=restore_physical,
            )
            if session.booking_lead_days > 0 and offer.asset_kind == "durable":
                # Advance requests express willingness to reserve an exact offer,
                # not a same-day inventory quote. Capacity remains the manager's
                # responsibility after acceptance and is never exposed here as a
                # feasibility label.
                available = max(available, resources)
            posted = state.prices.get(offer.offer_id)
            if (
                offer.asset_kind == "durable"
                and not reopen_closed
                and not state.rental_service_open.get(offer.offer_id, False)
            ):
                posted = None
            store_cash = None
            store_deposit = (
                resources * float(state.deposits[offer.offer_id])
                if offer.asset_kind == "durable"
                else 0.0
            )
            if posted is not None:
                if offer.asset_kind == "consumable":
                    store_cash = resources * float(posted)
                else:
                    store_cash = (
                        rental_amount(float(posted), duration, resources)
                        if self._v45_reservation_difficulty()
                        else resources * duration * float(posted)
                    )
            package_price = float(online["package_price_cny"])
            package_units = int(online["consumer_units_per_package"])
            online_cash = math.ceil(resources / package_units) * package_price
            rows.append(
                ChannelOfferView(
                    offer_id=offer.offer_id,
                    profile_id=self._channel_profile_signature(offer.offer_id),
                    choice_group=str(offer.choice_group),
                    kind=(
                        "rental_asset"
                        if offer.asset_kind == "durable"
                        else "retail_item"
                    ),
                    hard_compatible=compatible,
                    incompatibility_reasons=reasons,
                    soft_fit=self._functional_attribute_fit(
                        state, session, offer.offer_id
                    ),
                    quality_score=float(
                        CHANNEL_QUALITY_SCORES.get(
                            offer.perceived_quality_tier or "unknown", 0.0
                        )
                    ),
                    requested_units=resources,
                    available_units=available,
                    store_unit_price=(None if posted is None else float(posted)),
                    store_cash_price=(
                        None if store_cash is None else round(store_cash, 6)
                    ),
                    store_deposit=round(store_deposit, 6),
                    online_package_price=package_price,
                    online_package_units=package_units,
                    online_unit_price=float(online["unit_price_cny"]),
                    online_cash_price=round(online_cash, 6),
                    requested_customer_units=session.units_requested,
                    units_per_resource=(
                        max(
                            1,
                            int(
                                offer.decision_attributes.get(
                                    "sleep_slots", 1
                                )
                            ),
                        )
                        if session.service_key == "rental:sleeping_pad"
                        else 1
                    ),
                )
            )
        return tuple(rows)

    def explain_channel_demand(
        self,
        state: WorldState,
        session: ChoiceSession,
    ) -> tuple[ChannelDecisionTrace, tuple[ChannelOfferView, ...]]:
        views = self._channel_offer_views(state, session)
        return self._evaluate_channel_views(session, views), views

    def _evaluate_channel_views(
        self,
        session: ChoiceSession,
        views: tuple[ChannelOfferView, ...],
    ) -> ChannelDecisionTrace:
        model = self.channel_demand_model
        if model is None:
            raise ValueError("channel demand model is unavailable")
        if self.config.demand_world_id == "control":
            raise ValueError("control world uses the bounded reservation model")
        customer_address = session.event_address[:3]
        affinities = {}
        for channel in ("store", "online", "existing", "abandon"):
            rng = self._stable_event_rng(
                _RNG_CHANNEL_AFFINITY,
                *customer_address,
                identity=channel,
            )
            affinities[channel] = float(rng.normal(0.0, 0.15))
        existing_probability = (
            0.18 if session.service_key.startswith("rental:") else 0.08
        )
        if session.archetype in {"overnight", "lightweight"}:
            existing_probability += 0.04
        existing_rng = self._stable_event_rng(
            _RNG_EXISTING_OPTION,
            *session.event_address,
            identity=session.service_key,
        )
        bounded_tastes = {}
        for channel in ("store", "online", "existing", "abandon"):
            rng = self._stable_event_rng(
                _RNG_CHANNEL_BOUNDED_TASTE,
                *customer_address,
                identity=f"{session.service_key}:{channel}",
            )
            bounded_tastes[channel] = float(rng.uniform(-1.0, 1.0))
        return evaluate_channel_demand(
            model,
            ChannelEvaluationInput(
                service_key=session.service_key,
                preferred_group=session.preferred_group,
                customer_segment=session.customer_segment,
                world_id=self.config.demand_world_id,
                intent_role=session.intent_role,
                existing_option_available=bool(
                    existing_rng.random() < existing_probability
                ),
                channel_affinities=affinities,
                channel_gumbels={
                    channel: self._choice_shock(
                        session, "service", f"channel:{channel}"
                    )
                    for channel in ("store", "online", "existing", "abandon")
                },
                profile_gumbels={
                    profile_id: self._choice_shock(
                        session, "offer", f"channel-profile:{profile_id}"
                    )
                    for profile_id in sorted({row.profile_id for row in views})
                },
                offers=views,
                need_horizon_days=session.need_horizon_days,
                occasion_importance=session.occasion_importance,
                duration_days=session.duration_days,
                booking_lead_days=session.booking_lead_days,
                forecast_adversity=session.forecast_adversity,
                forecast_uncertainty=session.forecast_uncertainty,
                forecast_trust=session.forecast_trust,
                future_use_expectation=session.future_use_expectation,
                storage_burden=session.storage_burden,
                return_burden_multiplier=session.return_burden_multiplier,
                request_budget_multiplier=session.request_budget_multiplier,
                party_equivalent=max(
                    1.0,
                    float(session.adults) + 0.6 * float(session.children),
                ),
                channel_tastes=bounded_tastes,
            ),
        )
