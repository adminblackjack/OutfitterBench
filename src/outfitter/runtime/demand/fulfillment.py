from __future__ import annotations

from ...reservation_difficulty import rental_amount

from ..common import (
    Any,
    ChannelDecisionTrace,
    ChannelOfferView,
    ChoiceServiceResult,
    ChoiceSession,
    Counter,
    DAMAGE_RATES,
    RentalContract,
    WorldState,
    _RNG_PROFILE_CHOICE,
    _RNG_REQUEST_DAMAGE,
    _RNG_REQUEST_DAMAGE_SEVERITY,
    _RNG_RESERVATION_PRICE,
    defaultdict,
    logsumexp,
    math,
    sample_beta_pert,
)


class DemandFulfillmentMixin:
    def _select_quoted_offer(
        self,
        state: WorldState,
        session: ChoiceSession,
        offer_ids: list[str],
    ) -> tuple[str, str, dict[str, float], float, float]:
        return self.demand_strategy.select_quoted_offer(
            self, state, session, offer_ids
        )

    def _select_quoted_offer_legacy(
        self,
        state: WorldState,
        session: ChoiceSession,
        offer_ids: list[str],
    ) -> tuple[str, str, dict[str, float], float, float]:
        utilities = {
            offer_id: self._offer_utility(state, session, offer_id)
            for offer_id in offer_ids
        }
        grouped: dict[str, list[str]] = defaultdict(list)
        for offer_id in offer_ids:
            grouped[str(self.offers[offer_id].choice_group)].append(offer_id)
        inclusive = {
            group: logsumexp(
                (utilities[offer_id] for offer_id in ids),
                scale=self.rules.lambda_offer,
            )
            for group, ids in grouped.items()
        }
        group_systematic = {
            group: inclusive[group]
            + (
                self.rules.preferred_group_bonus
                if group == session.preferred_group
                else -self.rules.substitute_group_penalty
            )
            for group in grouped
        }
        service_inclusive = logsumexp(
            group_systematic.values(), scale=self.rules.lambda_group
        )
        selected_group = max(
            grouped,
            key=lambda group: (
                group_systematic[group]
                + self.rules.lambda_group
                * self._choice_shock(session, "group", group),
                group,
            ),
        )
        selected_offer = max(
            grouped[selected_group],
            key=lambda offer_id: (
                utilities[offer_id]
                + self.rules.lambda_offer
                * self._choice_shock(session, "offer", offer_id),
                offer_id,
            ),
        )
        buy_score = (
            self.service_intercept
            + service_inclusive
            + self._choice_shock(session, "service", "shop")
        )
        outside_score = self._choice_shock(session, "service", "outside")
        return selected_group, selected_offer, utilities, buy_score, outside_score

    def _select_quoted_offer_reservation(
        self,
        state: WorldState,
        session: ChoiceSession,
        offer_ids: list[str],
    ) -> tuple[str, str, dict[str, float], float, float]:
        model = self.demand_model
        if model is None:
            raise AssertionError("bounded reservation model is unavailable")
        group = model.group(session.service_key, session.preferred_group)
        reservation_rng = self._stable_event_rng(
            _RNG_RESERVATION_PRICE,
            *session.event_address,
            identity=group.key,
        )
        base_reservation = sample_beta_pert(
            reservation_rng,
            low=group.reservation_low_cny,
            central=group.reservation_central_cny,
            high=group.reservation_high_cny,
            shape=float(model.variation_shapes[group.customer_variation]),
        )
        profiles: dict[str, list[str]] = defaultdict(list)
        for offer_id in sorted(offer_ids):
            profiles[self._consumer_profile_signature(state, offer_id)].append(
                offer_id
            )
        utilities: dict[str, float] = {}
        accepted: list[tuple[float, str, str]] = []
        all_profiles: list[tuple[float, str, str]] = []
        taste_limit = min(
            abs(model.choice_taste_range[0]),
            abs(model.choice_taste_range[1]),
            float(model.substitution_choice_taste[group.substitution_strength]),
        )
        for signature, ids in sorted(profiles.items()):
            representative = ids[0]
            reservation = self._reservation_scale(
                state,
                session,
                representative,
                group,
                base_reservation,
                signature,
            ) * self._reservation_demand_units(
                session, representative, group
            )
            price = self._line_price(state, session, representative)
            relative_surplus = (reservation - price) / max(0.01, reservation)
            choice_rng = self._stable_event_rng(
                _RNG_PROFILE_CHOICE,
                *session.event_address,
                identity=signature,
            )
            score = relative_surplus + float(
                choice_rng.uniform(-taste_limit, taste_limit)
            )
            for offer_id in ids:
                utilities[offer_id] = score
            record = (score, signature, representative)
            all_profiles.append(record)
            if price <= reservation + 1e-9:
                accepted.append(record)
        pool = accepted or all_profiles
        selected_score, _signature, selected_offer = max(
            pool, key=lambda row: (row[0], row[1])
        )
        selected_group = str(self.offers[selected_offer].choice_group)
        return (
            selected_group,
            selected_offer,
            utilities,
            1.0 if accepted else 0.0,
            0.0 if accepted else 1.0,
        )

    def _rent_from_offer(
        self,
        state: WorldState,
        session: ChoiceSession,
        offer_id: str,
        remaining_units: int,
        outcomes: Counter[str],
    ) -> tuple[int, dict[str, Any]]:
        offer = self.offers[offer_id]
        assets = sorted(
            (
                asset
                for asset in state.assets
                if asset.offer_id == offer_id and asset.status == "available"
                and (not self._v31_warehouse_capacity() or self._warehouse_asset_held(asset))
                and self._supplier_trial_asset_can_start_contract(
                    state,
                    asset,
                    return_day=state.day + max(1, int(session.duration_days or 1)),
                )
            ),
            key=lambda row: (-self._used_life_fraction(row), row.asset_id),
        )
        assets = assets[
            : self._available_offer_units(
                state, offer_id, session=session
            )
        ]
        if session.service_key == "rental:sleeping_pad":
            per_asset = max(1, int(offer.decision_attributes.get("sleep_slots", 1)))
        else:
            per_asset = 1
        needed_assets = min(len(assets), math.ceil(remaining_units / per_asset))
        if session.service_key not in {"rental:sleeping_pad", "rental:sleeping_bag", "rental:camp_chair"}:
            needed_assets = min(1, len(assets))
        duration = max(1, int(session.duration_days or 1))
        fulfilled = 0
        total_rent = 0.0
        total_deposit = 0.0
        contract_ids = []
        for asset in assets[:needed_assets]:
            served = min(per_asset, remaining_units - fulfilled)
            if served <= 0:
                break
            rent = (
                rental_amount(float(state.prices[offer_id]), duration)
                if self._v45_reservation_difficulty()
                else round(float(state.prices[offer_id]) * duration, 2)
            )
            deposit = float(state.deposits[offer_id])
            state.cash = round(state.cash + rent + deposit, 2)
            state.deposit_liability = round(state.deposit_liability + deposit, 2)
            state.cumulative_rental_revenue = round(state.cumulative_rental_revenue + rent, 2)
            state.cumulative_deposits_collected = round(
                state.cumulative_deposits_collected + deposit, 2
            )
            self._add(state.current_day_ledger, "rental_revenue", rent)
            self._add(state.current_day_ledger, "deposits_collected", deposit)
            damage_rng = self._stable_event_rng(
                _RNG_REQUEST_DAMAGE,
                *session.event_address,
                identity=asset.asset_id,
            )
            event_uniform = float(damage_rng.random())
            lifecycle_damage: dict[str, float | bool] | None = None
            severity_uniform: float | None = None
            if self.config.lifecycle_economics_enabled:
                severity_uniform = float(
                    self._stable_event_rng(
                        _RNG_REQUEST_DAMAGE_SEVERITY,
                        *session.event_address,
                        identity=asset.asset_id,
                    ).random()
                )
                lifecycle_damage = self._lifecycle_damage_outcome(
                    asset,
                    start_day=state.day,
                    return_day=state.day + duration,
                    event_uniform=event_uniform,
                    severity_uniform=severity_uniform,
                )
                damaged = bool(lifecycle_damage["damaged"])
            else:
                rate = DAMAGE_RATES[asset.damage_risk_band]
                damaged = bool(
                    event_uniform
                    < self._rental_damage_probability(
                        rate, state.day, state.day + duration
                    )
                )
            asset.status = "rented"
            asset.used_life_days += duration
            asset.used_rental_cycles += 1
            contract = RentalContract(
                contract_id=f"rental-{state.next_contract_id:06d}",
                asset_id=asset.asset_id,
                offer_id=offer_id,
                service_line=asset.service_line,
                start_day=state.day,
                return_day=state.day + duration,
                duration_days=duration,
                rent_paid=rent,
                deposit=deposit,
                damaged_on_return=damaged,
                promised_return_day=state.day + duration,
                approved_return_day=state.day + duration,
                estimated_return_day=state.day + duration,
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
                    float(lifecycle_damage["maintenance_exposure_at_start"])
                    if lifecycle_damage is not None
                    else None
                ),
            )
            state.contracts.append(contract)
            contract_ids.append(contract.contract_id)
            state.next_contract_id += 1
            trial_fee = self._start_supplier_trial_contract(
                state, asset, return_day=contract.return_day
            )
            if asset.ownership_type == "supplier_trial":
                contract.supplier_trial_id = asset.supplier_trial_id
                contract.supplier_trial_usage_fee = round(
                    float(trial_fee or 0.0), 2
                )
            self._record_rental_activity(
                state,
                "rental_started",
                asset,
                contract_id=contract.contract_id,
                duration_days=duration,
                return_day=contract.return_day,
                rent_paid=rent,
                deposit=deposit,
                fulfilled_demand_units=served,
            )
            if trial_fee is None:
                break
            fulfilled += served
            total_rent += rent
            total_deposit += deposit
        if fulfilled:
            self._record_attempt(
                state,
                session.service_key,
                "fulfilled",
                quantity=fulfilled,
                product_type=offer.service_line,
                offer_id=offer_id,
                revenue=round(total_rent, 2),
                deposit=round(total_deposit, 2),
                rental_days=duration * needed_assets,
                outcomes=outcomes,
            )
        return fulfilled, {
            "offer_id": offer_id,
            "quoted_unit_price": round(float(state.prices[offer_id]), 2),
            "offered_units": fulfilled,
            "asset_units": needed_assets,
            "revenue": round(total_rent, 2),
            "retail_cogs": 0.0,
            "rental_days": duration * needed_assets,
            **({"contract_ids": contract_ids} if self.dataset.metadata.get("operating_history_policy") else {}),
        }

    def _sell_from_offer(
        self,
        state: WorldState,
        session: ChoiceSession,
        offer_id: str,
        remaining_units: int,
        outcomes: Counter[str],
    ) -> tuple[int, dict[str, Any]]:
        offer = self.offers[offer_id]
        inventory = state.retail_inventory.get(offer_id, {})
        sold = min(remaining_units, int(inventory.get("quantity", 0)))
        revenue = 0.0
        cogs = 0.0
        if sold:
            unit_cogs = float(inventory.get("avg_unit_cost", offer.unit_cost))
            inventory["quantity"] = int(inventory["quantity"]) - sold
            revenue = round(float(state.prices[offer_id]) * sold, 2)
            cogs = round(unit_cogs * sold, 2)
            state.cash = round(state.cash + revenue, 2)
            state.cumulative_retail_revenue = round(state.cumulative_retail_revenue + revenue, 2)
            self._add(state.current_day_ledger, "retail_revenue", revenue)
            self._record_attempt(
                state,
                session.service_key,
                "fulfilled",
                quantity=sold,
                product_type=offer.service_line,
                offer_id=offer_id,
                revenue=revenue,
                retail_cogs=cogs,
                outcomes=outcomes,
            )
        return sold, {
            "offer_id": offer_id,
            "quoted_unit_price": round(float(state.prices[offer_id]), 2),
            "offered_units": sold,
            "asset_units": 0,
            "revenue": round(revenue, 2),
            "retail_cogs": round(cogs, 2),
            "rental_days": 0,
        }

    def _record_interaction(
        self,
        state: WorldState,
        session: ChoiceSession,
        *,
        preferred_state: str,
        preferred_available_units: int,
        quote_lines: list[dict[str, Any]],
        fulfilled_units: int,
        disposition: str,
    ) -> None:
        from ... import operating_history
        if operating_history.enabled(self.dataset.metadata):
            quote_lines = [operating_history.freeze_quote(self, state, session, line) for line in quote_lines]
        unfulfilled = max(0, session.units_requested - fulfilled_units)
        natural_names = {
            "rental:tent": "帐篷",
            "rental:sleeping_bag": "睡袋",
            "rental:sleeping_pad": "睡垫",
            "rental:camp_chair": "露营椅",
            "rental:camp_table": "露营桌",
            "rental:camp_lantern": "营地灯",
            "rental:camp_cookware": "露营炊具",
            "retail:rain_gear": "雨具",
            "retail:repellent": "驱蚊用品",
            "retail:bottled_water": "瓶装水",
            "retail:energy_snacks": "能量零食",
            "retail:instant_food": "即食食品",
            "retail:cleaning_wipes": "清洁湿巾",
            "retail:garbage_bags": "垃圾袋",
        }
        mode = "rental" if session.service_key.startswith("rental:") else "retail"
        item_name = natural_names[session.service_key]
        party = session.adults + session.children
        duration_text = (
            f"，预计使用 {session.duration_days} 天"
            if session.duration_days is not None
            else ""
        )
        intended_use = {
            "emergency_rain": "临时应急挡雨",
            "planned_rain": "计划性户外防雨",
            "trip_ration": "行程口粮",
            "casual_snack": "随身零食",
            "portable_sleep": "轻量徒步睡眠",
            "comfort_sleep": "露营舒适睡眠",
            "simple_sleep": "基础露营睡眠",
            "portable_seating": "轻量徒步休息",
            "comfort_seating": "营地舒适休息",
            "cool_night": "较冷夜晚保暖",
            "warm_night": "温暖天气睡眠",
        }.get(session.use_case)
        intent_text = f"，用途是{intended_use}" if intended_use else ""
        if mode == "rental":
            request_text = (
                f"想租 {session.units_requested} 件{item_name}，同行 {party} 人"
                f"{duration_text}{intent_text}"
            )
        elif session.service_key == "retail:cleaning_wipes":
            request_text = (
                f"想买 {session.units_requested} 盒{item_name}，同行 {party} 人"
                f"{intent_text}"
            )
        elif session.service_key == "retail:rain_gear":
            audience = "成人" if session.size_class == "adult" else "儿童"
            rainwear_name = "雨披" if session.use_case == "emergency_rain" else "雨衣"
            request_text = (
                f"想买 {session.units_requested} 件{audience}{rainwear_name}，"
                f"同行 {party} 人{intent_text}"
            )
        else:
            request_text = (
                f"想买 {session.units_requested} 份{item_name}，同行 {party} 人"
                f"{intent_text}"
            )
        public_requirements: dict[str, Any] = {
            "requested_units": session.units_requested,
            "party_adults": session.adults,
            "party_children": session.children,
        }
        if session.duration_days is not None:
            public_requirements["duration_days"] = session.duration_days
        if intended_use:
            public_requirements["intended_use"] = intended_use
        if session.size_class not in {"none", "any", "n/a", ""}:
            public_requirements["requested_size"] = {
                "adult_regular": "adult",
                "adult": "adult",
                "child": "child",
                "one_size": "one size",
            }.get(session.size_class, session.size_class.replace("_", " "))
        calendar = self._calendar(state.day)
        row = {
            "interaction_id": session.interaction_id,
            "day": state.day,
            "day_type": calendar["day_type"],
            "holiday_name": calendar.get("holiday_name"),
            "weather": self._weather(state.day),
            "visible_party_context": session.visible_party_context,
            "public_request": {
                "request_id": session.interaction_id,
                "request_text": request_text,
                "requirements": public_requirements,
                "rental_or_retail": mode,
            },
            "request": {
                "service_key": session.service_key,
                "asked_group": session.preferred_group,
                "customer_segment": session.customer_segment,
                "size_class": session.size_class,
                "requested_units": session.units_requested,
                "duration_days": session.duration_days,
            },
            "preferred_availability": {
                "state": preferred_state,
                "available_units": preferred_available_units,
            },
            "quote_lines": quote_lines,
            "response": {
                "accepted_units": fulfilled_units,
                "unfulfilled_units": unfulfilled,
                "disposition": disposition,
            },
        }
        state.current_day_ledger["counter_interactions"].append(row)

    def _public_counter_interaction(self, raw: dict[str, Any]) -> dict[str, Any]:
        from ... import operating_history
        quotes = []
        for line in raw.get("quote_lines", []):
            offer_id = str(line.get("offer_id") or "")
            quotes.append(
                {
                    key: value
                    for key, value in {
                        "offer_id": offer_id,
                        "name": self.offers[offer_id].name if offer_id in self.offers else None,
                        "quoted_unit_price": line.get("quoted_unit_price"),
                        "offered_units": line.get("offered_units"),
                        "revenue": line.get("revenue"),
                        "rental_days": line.get("rental_days"),
                        **({key: line.get(key) for key in operating_history.QUOTE_FIELDS}
                           if operating_history.enabled(self.dataset.metadata) else {}),
                    }.items()
                    if value is not None
                }
            )
        return {
            "interaction_id": raw["interaction_id"],
            "day": raw["day"],
            "day_type": raw["day_type"],
            **(
                {"holiday_name": raw["holiday_name"]}
                if raw.get("holiday_name")
                else {}
            ),
            "weather": raw["weather"],
            "visible_party_context": raw["visible_party_context"],
            "request": dict(raw.get("public_request") or {}),
            "quotes": quotes,
            "response": {
                key: value
                for key, value in raw.get("response", {}).items()
                if key in {"accepted_units", "unfulfilled_units", "disposition"}
            },
        }

    def _record_quote_exposure(
        self,
        state: WorldState,
        service_key: str,
        offer_id: str,
        *,
        quoted_units: int,
    ) -> None:
        offer = self.offers[offer_id]
        metric = state.current_day_ledger["service_metrics"].setdefault(
            service_key,
            {
                "requested_units": 0,
                "fulfilled_units": 0,
                "outside_rejections": 0,
                "stockout_rejections": 0,
                "service_closed_rejections": 0,
                "horizon_rejections": 0,
                "unpriced_rejections": 0,
                "revenue": 0.0,
                "deposits_collected": 0.0,
                "product_fulfillment_breakdown": {},
                "offer_fulfillment_breakdown": {},
            },
        )
        row = metric.setdefault("offer_fulfillment_breakdown", {}).setdefault(
            offer_id,
            {
                "product_type": offer.service_line,
                "fulfilled_units": 0,
                "revenue": 0.0,
                "rental_days": 0,
                "deposits_collected": 0.0,
                "retail_cogs": 0.0,
                "quote_exposures": 0,
                "quoted_units": 0,
            },
        )
        row["quote_exposures"] = int(row.get("quote_exposures", 0)) + 1
        row["quoted_units"] = int(row.get("quoted_units", 0)) + max(
            0, int(quoted_units)
        )

    def _serve_choice_session_channel(
        self,
        state: WorldState,
        session: ChoiceSession,
        outcomes: Counter[str],
        *,
        precomputed: tuple[
            ChannelDecisionTrace, tuple[ChannelOfferView, ...]
        ]
        | None = None,
    ) -> ChoiceServiceResult:
        """Serve one counter request while keeping the chosen outside channel private."""

        demand = state.current_day_ledger["demand"]
        demand["requested_line_items"] += 1
        state.cumulative_requested_line_items += 1
        preferred_state, preferred_available = self._preferred_availability(
            state, session
        )
        if preferred_state == "never_carried":
            demand["preferred_not_carried_units"] += session.units_requested
        elif preferred_state == "stocked_out":
            demand["preferred_stocked_out_units"] += session.units_requested

        if (
            session.duration_days is not None
            and state.day + session.duration_days > state.max_days
        ):
            self._record_attempt(
                state,
                session.service_key,
                "horizon",
                quantity=session.units_requested,
                outcomes=outcomes,
            )
            self._record_interaction(
                state,
                session,
                preferred_state=preferred_state,
                preferred_available_units=preferred_available,
                quote_lines=[],
                fulfilled_units=0,
                disposition="horizon",
            )
            demand["unfilled_line_items"] += 1
            state.cumulative_unfilled_line_items += 1
            return ChoiceServiceResult(
                fulfilled_units=0,
                request_satisfied=False,
                internal_channel="horizon",
            )

        stage_conflict_recomputed = False
        trace, views = (
            self.explain_channel_demand(state, session)
            if precomputed is None
            else precomputed
        )
        if trace.selected_channel == "store" and precomputed is not None:
            current_views = self._channel_offer_views(state, session)
            current_by_id = {row.offer_id: row for row in current_views}
            selected = next(
                (
                    row
                    for row in trace.profiles
                    if row["profile_id"] == trace.selected_profile_id
                ),
                None,
            )
            planned_lines = [] if selected is None else selected.get(
                "store_quote_plan", []
            )
            still_fulfillable = bool(planned_lines) and all(
                str(line["offer_id"]) in current_by_id
                and current_by_id[str(line["offer_id"])].available_units
                >= int(line["resource_units"])
                for line in planned_lines
            )
            if not still_fulfillable:
                stage_conflict_recomputed = True
                views = current_views
                trace = self._evaluate_channel_views(session, current_views)
        view_by_id = {row.offer_id: row for row in views}
        profile_by_id = {
            str(row["profile_id"]): row for row in trace.profiles
        }
        selected_store_profile = (
            None
            if trace.selected_channel != "store"
            or trace.selected_profile_id is None
            else profile_by_id[trace.selected_profile_id]
        )
        state.current_day_ledger["internal_channel_outcomes"].append(
            {
                "interaction_id": session.interaction_id,
                "intent_role": session.intent_role,
                "selected_channel": trace.selected_channel,
                "selected_profile_id": trace.selected_profile_id,
                "selected_offer_id": trace.selected_offer_id,
                "request_satisfied": (
                    trace.selected_channel in {"online", "existing"}
                    or (
                        selected_store_profile is not None
                        and int(
                            selected_store_profile.get(
                                "store_fulfilled_units", 0
                            )
                        )
                        >= session.units_requested
                    )
                ),
                "stage_inventory_conflict_recomputed": stage_conflict_recomputed,
            }
        )

        best_store = (
            None
            if trace.best_store_profile_id is None
            else profile_by_id[trace.best_store_profile_id]
        )

        def public_quote_plan(profile: dict[str, Any] | None) -> list[dict[str, Any]]:
            if profile is None:
                return []
            result = []
            remaining = session.units_requested
            for line in profile.get("store_quote_plan", []):
                offer_id = str(line["offer_id"])
                view = view_by_id[offer_id]
                offered = min(
                    remaining,
                    int(line.get("covered_units", 0)),
                )
                if offered <= 0:
                    continue
                result.append(
                    {
                        "offer_id": offer_id,
                        "quoted_unit_price": round(
                            float(view.store_unit_price), 2
                        ),
                        "offered_units": offered,
                        "asset_units": int(line["resource_units"]),
                        "quote_kind": (
                            "preferred"
                            if view.choice_group == session.preferred_group
                            else "substitute"
                        ),
                    }
                )
                remaining -= offered
            return result

        primary_quotes = public_quote_plan(best_store)
        if trace.selected_channel == "store":
            selected_profile = profile_by_id[str(trace.selected_profile_id)]
            quote_lines: list[dict[str, Any]] = []
            fulfilled = 0
            preferred_fulfilled = 0
            substitute_fulfilled = 0
            for plan_line in selected_profile.get("store_quote_plan", []):
                if fulfilled >= session.units_requested:
                    break
                offer_id = str(plan_line["offer_id"])
                remaining = session.units_requested - fulfilled
                line_requested = min(
                    remaining, int(plan_line.get("covered_units", remaining))
                )
                self._record_quote_exposure(
                    state,
                    session.service_key,
                    offer_id,
                    quoted_units=line_requested,
                )
                if self.offers[offer_id].asset_kind == "durable":
                    served, quote = self._rent_from_offer(
                        state, session, offer_id, line_requested, outcomes
                    )
                else:
                    served, quote = self._sell_from_offer(
                        state, session, offer_id, line_requested, outcomes
                    )
                if served:
                    quote["quote_kind"] = (
                        "preferred"
                        if view_by_id[offer_id].choice_group
                        == session.preferred_group
                        else "substitute"
                    )
                    if quote["quote_kind"] == "preferred":
                        preferred_fulfilled += served
                    else:
                        substitute_fulfilled += served
                    quote_lines.append(quote)
                    fulfilled += served
            demand["preferred_fulfilled_units"] += preferred_fulfilled
            demand["substitute_fulfilled_units"] += substitute_fulfilled
            if fulfilled == session.units_requested:
                disposition = "full_sale"
                demand["fulfilled_line_items"] += 1
                state.cumulative_fulfilled_line_items += 1
            elif fulfilled:
                disposition = "partial_sale"
                demand["partial_line_items"] += 1
                state.cumulative_partial_line_items += 1
            else:
                disposition = "no_quote"
                demand["unfilled_line_items"] += 1
                state.cumulative_unfilled_line_items += 1
            self._record_interaction(
                state,
                session,
                preferred_state=preferred_state,
                preferred_available_units=preferred_available,
                quote_lines=quote_lines,
                fulfilled_units=fulfilled,
                disposition=disposition,
            )
            return ChoiceServiceResult(
                fulfilled_units=fulfilled,
                request_satisfied=fulfilled == session.units_requested,
                internal_channel="store",
            )

        satisfied = trace.selected_channel in {"online", "existing"}
        if primary_quotes:
            for quote in primary_quotes:
                self._record_quote_exposure(
                    state,
                    session.service_key,
                    str(quote["offer_id"]),
                    quoted_units=int(quote["offered_units"]),
                )
            self._record_attempt(
                state,
                session.service_key,
                "outside",
                quantity=session.units_requested,
                outcomes=outcomes,
            )
            demand["declined_after_quote_units"] += session.units_requested
            disposition = "declined_after_quote"
            quotes = primary_quotes
        else:
            beyond_horizon = (
                session.duration_days is not None
                and state.day + session.duration_days > state.max_days
            )
            if self._v24_reservation_operations() and not beyond_horizon:
                outcome = self._v24_unquoted_operational_outcome(
                    state, session, views
                )
            else:
                all_offers = self.by_service.get(session.service_key, [])
                service_closed = (
                    session.service_key.startswith("rental:")
                    and not any(
                        state.rental_service_open.get(offer.offer_id, False)
                        for offer in all_offers
                    )
                )
                compatible_inventory = any(
                    row.hard_compatible and row.available_units > 0
                    for row in views
                )
                priced_inventory = any(
                    row.hard_compatible
                    and row.available_units > 0
                    and row.store_unit_price is not None
                    for row in views
                )
                unpriced = compatible_inventory and not priced_inventory
                outcome = (
                    "horizon"
                    if beyond_horizon
                    else "service_closed"
                    if service_closed
                    else "unpriced"
                    if unpriced
                    else "stockout"
                )
            self._record_attempt(
                state,
                session.service_key,
                outcome,
                quantity=session.units_requested,
                outcomes=outcomes,
            )
            demand["no_quote_units"] += session.units_requested
            if self._v24_reservation_operations():
                disposition = (
                    "outside_without_store_quote"
                    if outcome == "outside"
                    else outcome
                    if outcome
                    in {
                        "horizon",
                        "service_closed",
                        "capacity_policy",
                        "stockout",
                        "unpriced",
                    }
                    else "no_quote"
                )
            else:
                disposition = (
                    outcome
                    if outcome in {"horizon", "service_closed"}
                    else "no_quote"
                )
            quotes = []
        demand["unfilled_line_items"] += 1
        state.cumulative_unfilled_line_items += 1
        self._record_interaction(
            state,
            session,
            preferred_state=preferred_state,
            preferred_available_units=preferred_available,
            quote_lines=quotes,
            fulfilled_units=0,
            disposition=disposition,
        )
        return ChoiceServiceResult(
            fulfilled_units=0,
            request_satisfied=satisfied,
            internal_channel=trace.selected_channel,
        )

    def _v24_unquoted_operational_outcome(
        self,
        state: WorldState,
        session: ChoiceSession,
        actual_views: tuple[Any, ...],
    ) -> str:
        """Attribute a missing store plan by relaxing one constraint layer at a time."""

        if not self._v24_reservation_operations():
            raise AssertionError("V24 rejection attribution used by an older runtime")
        counterfactuals = (
            (
                "service_closed",
                self._channel_offer_views(
                    state,
                    session,
                    reopen_closed=True,
                ),
            ),
            (
                "capacity_policy",
                self._channel_offer_views(
                    state,
                    session,
                    reopen_closed=True,
                    ignore_walk_in_capacity=True,
                ),
            ),
            (
                "stockout",
                self._channel_offer_views(
                    state,
                    session,
                    reopen_closed=True,
                    ignore_walk_in_capacity=True,
                    restore_physical=True,
                ),
            ),
        )
        for outcome, views in counterfactuals:
            trace = self._evaluate_channel_views(session, views)
            if trace.selected_channel == "store":
                return outcome
        compatible_inventory = any(
            row.hard_compatible and row.available_units > 0
            for row in actual_views
        )
        priced_inventory = any(
            row.hard_compatible
            and row.available_units > 0
            and row.store_unit_price is not None
            for row in actual_views
        )
        if compatible_inventory and not priced_inventory:
            return "unpriced"
        return "outside"

    def _serve_choice_session(
        self,
        state: WorldState,
        session: ChoiceSession,
        outcomes: Counter[str],
    ) -> int:
        return self.demand_strategy.serve_session(
            self, state, session, outcomes
        )

    def _serve_choice_session_standard(
        self,
        state: WorldState,
        session: ChoiceSession,
        outcomes: Counter[str],
    ) -> int:
        demand = state.current_day_ledger["demand"]
        demand["requested_line_items"] += 1
        state.cumulative_requested_line_items += 1
        preferred_state, preferred_available = self._preferred_availability(state, session)
        if preferred_state == "never_carried":
            demand["preferred_not_carried_units"] += session.units_requested
        elif preferred_state == "stocked_out":
            demand["preferred_stocked_out_units"] += session.units_requested

        if session.duration_days is not None and state.day + session.duration_days > state.max_days:
            self._record_attempt(
                state, session.service_key, "horizon", quantity=session.units_requested, outcomes=outcomes
            )
            self._record_interaction(
                state,
                session,
                preferred_state=preferred_state,
                preferred_available_units=preferred_available,
                quote_lines=[],
                fulfilled_units=0,
                disposition="horizon",
            )
            demand["unfilled_line_items"] += 1
            state.cumulative_unfilled_line_items += 1
            return 0

        all_offers = self.by_service.get(session.service_key, [])
        if session.service_key.startswith("rental:") and not any(
            state.rental_service_open.get(offer.offer_id, False) for offer in all_offers
        ):
            self._record_attempt(
                state,
                session.service_key,
                "service_closed",
                quantity=session.units_requested,
                outcomes=outcomes,
            )
            self._record_interaction(
                state,
                session,
                preferred_state=preferred_state,
                preferred_available_units=preferred_available,
                quote_lines=[],
                fulfilled_units=0,
                disposition="service_closed",
            )
            demand["unfilled_line_items"] += 1
            state.cumulative_unfilled_line_items += 1
            return 0

        available_ids = sorted(
            offer.offer_id
            for offer in all_offers
            if self._available_offer_units(
                state, offer.offer_id, session=session
            ) > 0
            and offer.offer_id in state.prices
        )
        if not available_ids:
            has_unpriced_inventory = any(
                self._available_offer_units(
                    state, offer.offer_id, session=session
                ) > 0
                and offer.offer_id not in state.prices
                for offer in all_offers
            )
            self._record_attempt(
                state,
                session.service_key,
                "unpriced" if has_unpriced_inventory else "stockout",
                quantity=session.units_requested,
                outcomes=outcomes,
            )
            demand["no_quote_units"] += session.units_requested
            self._record_interaction(
                state,
                session,
                preferred_state=preferred_state,
                preferred_available_units=preferred_available,
                quote_lines=[],
                fulfilled_units=0,
                disposition="no_quote",
            )
            demand["unfilled_line_items"] += 1
            state.cumulative_unfilled_line_items += 1
            return 0

        selected_group, selected_offer, utilities, buy_score, outside_score = self._select_quoted_offer(
            state, session, available_ids
        )
        primary_quote = {
            "offer_id": selected_offer,
            "quoted_unit_price": round(float(state.prices[selected_offer]), 2),
            "offered_units": min(
                session.units_requested,
                self._available_offer_units(
                    state, selected_offer, session=session
                )
                * (
                    int(self.offers[selected_offer].decision_attributes.get("sleep_slots", 1))
                    if session.service_key == "rental:sleeping_pad"
                    else 1
                ),
            ),
            "asset_units": self._available_offer_units(
                state, selected_offer, session=session
            ),
            "quote_kind": "preferred" if selected_group == session.preferred_group else "substitute",
        }
        self._record_quote_exposure(
            state,
            session.service_key,
            selected_offer,
            quoted_units=int(primary_quote["offered_units"]),
        )
        if outside_score >= buy_score:
            self._record_attempt(
                state, session.service_key, "outside", quantity=session.units_requested, outcomes=outcomes
            )
            demand["declined_after_quote_units"] += session.units_requested
            self._record_interaction(
                state,
                session,
                preferred_state=preferred_state,
                preferred_available_units=preferred_available,
                quote_lines=[primary_quote],
                fulfilled_units=0,
                disposition="declined_after_quote",
            )
            demand["unfilled_line_items"] += 1
            state.cumulative_unfilled_line_items += 1
            return 0

        if self.demand_strategy.profile_deduplication:
            by_profile: dict[str, list[str]] = defaultdict(list)
            for offer_id in available_ids:
                if str(self.offers[offer_id].choice_group) != selected_group:
                    continue
                by_profile[
                    self._consumer_profile_signature(state, offer_id)
                ].append(offer_id)
            ordered_profiles = sorted(
                by_profile.items(),
                key=lambda item: (
                    -utilities[item[1][0]],
                    item[0],
                ),
            )
            ranked = [
                offer_id
                for _signature, profile_ids in ordered_profiles
                for offer_id in sorted(profile_ids)
            ]
            selected_adjusted = utilities[selected_offer]
        else:
            ranked = sorted(
                (
                    offer_id
                    for offer_id in available_ids
                    if str(self.offers[offer_id].choice_group) == selected_group
                ),
                key=lambda offer_id: (
                    -(
                        utilities[offer_id]
                        + self.rules.lambda_offer
                        * self._choice_shock(session, "offer", offer_id)
                    ),
                    offer_id,
                ),
            )
            selected_adjusted = utilities[
                selected_offer
            ] + self.rules.lambda_offer * self._choice_shock(
                session, "offer", selected_offer
            )
        quote_lines: list[dict[str, Any]] = []
        fulfilled = 0
        for offer_id in ranked:
            adjusted = utilities[offer_id]
            if not self.demand_strategy.profile_deduplication:
                adjusted += self.rules.lambda_offer * self._choice_shock(
                    session, "offer", offer_id
                )
            if offer_id != selected_offer and selected_adjusted - adjusted > session.substitution_tolerance:
                continue
            remaining = session.units_requested - fulfilled
            if remaining <= 0:
                break
            if self.offers[offer_id].asset_kind == "durable":
                served, quote = self._rent_from_offer(
                    state, session, offer_id, remaining, outcomes
                )
            else:
                served, quote = self._sell_from_offer(
                    state, session, offer_id, remaining, outcomes
                )
            if served:
                if offer_id != selected_offer:
                    self._record_quote_exposure(
                        state,
                        session.service_key,
                        offer_id,
                        quoted_units=served,
                    )
                quote["quote_kind"] = (
                    "preferred"
                    if str(self.offers[offer_id].choice_group) == session.preferred_group
                    else "substitute"
                )
                quote_lines.append(quote)
                fulfilled += served
        missing = session.units_requested - fulfilled
        if missing:
            self._record_attempt(
                state, session.service_key, "stockout", quantity=missing, outcomes=outcomes
            )
        if selected_group == session.preferred_group:
            demand["preferred_fulfilled_units"] += fulfilled
        else:
            demand["substitute_fulfilled_units"] += fulfilled
        if fulfilled == session.units_requested:
            disposition = "full_sale"
            demand["fulfilled_line_items"] += 1
            state.cumulative_fulfilled_line_items += 1
        elif fulfilled:
            disposition = "partial_sale"
            demand["partial_line_items"] += 1
            state.cumulative_partial_line_items += 1
        else:
            disposition = "no_quote"
            demand["unfilled_line_items"] += 1
            state.cumulative_unfilled_line_items += 1
        self._record_interaction(
            state,
            session,
            preferred_state=preferred_state,
            preferred_available_units=preferred_available,
            quote_lines=quote_lines or [primary_quote],
            fulfilled_units=fulfilled,
            disposition=disposition,
        )
        return fulfilled
