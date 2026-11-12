from __future__ import annotations

from .common import (
    Any,
    CLEANING_COST_RULES,
    Counter,
    DAMAGE_RATES,
    DEPOSIT_RATIOS,
    EpisodeMarketProfile,
    LocalGroupProfile,
    PAD_LINES,
    PurchaseOrder,
    RENTAL_HOLDING_RATE,
    RETAIL_HOLDING_RATE,
    RETAIL_SALVAGE_RATE,
    RentalAsset,
    SECONDHAND_GRADE_SCHEDULE,
    SECONDHAND_SELLABLE_STATUSES,
    TERMINAL_HEALTHY_STATUSES,
    WorldState,
    _RNG_EPISODE_MARKET,
    _empty_ledger,
    asdict,
    math,
    median,
    np,
    stable_words,
)
from ..actor_interface_slimming import POLICY_ID as ACTOR_INTERFACE_SLIMMING_POLICY_ID
from ..weather_information import POLICY_ID as WEATHER_INFORMATION_POLICY_ID
from ..reservation_difficulty import POLICY_ID as RESERVATION_DIFFICULTY_POLICY_ID


class RuntimeBaseMixin:
    def _v45_reservation_difficulty(self) -> bool:
        return bool(
            int(self.dataset.metadata.get("tool_interface_version", 0)) >= 40
            and self.dataset.metadata.get("reservation_difficulty_policy")
            == RESERVATION_DIFFICULTY_POLICY_ID
        )

    def _v42_weather_information_coherence(self) -> bool:
        return bool(
            int(self.dataset.metadata.get("tool_interface_version", 0)) >= 37
            and self.dataset.metadata.get("weather_information_policy")
            == WEATHER_INFORMATION_POLICY_ID
        )

    def _v41_actor_interface_slimming(self) -> bool:
        return bool(
            int(self.dataset.metadata.get("tool_interface_version", 0)) >= 36
            and self.dataset.metadata.get("actor_interface_slimming_policy")
            == ACTOR_INTERFACE_SLIMMING_POLICY_ID
        )

    def _v24_reservation_operations(self) -> bool:
        return bool(
            self.config.reservation_contract_mode
            == "factorized_line_options_plan_set_v1"
            and int(self.dataset.metadata.get("tool_interface_version", 0)) >= 20
        )

    def _v25_quote_workspace(self) -> bool:
        return bool(
            self.config.reservation_contract_mode
            == "factorized_line_options_plan_set_v1"
            and int(self.dataset.metadata.get("tool_interface_version", 0)) >= 21
        )

    def _v26_contract_state(self) -> bool:
        return bool(
            self.config.reservation_contract_mode
            == "factorized_line_options_plan_set_v1"
            and int(self.dataset.metadata.get("tool_interface_version", 0)) >= 22
        )

    def _v27_extension_cash_audit(self) -> bool:
        return bool(
            self.config.reservation_contract_mode
            == "factorized_line_options_plan_set_v1"
            and int(self.dataset.metadata.get("tool_interface_version", 0)) >= 23
        )

    def _v28_capacity_summary_audit(self) -> bool:
        return bool(
            self.config.reservation_contract_mode
            == "factorized_line_options_plan_set_v1"
            and int(self.dataset.metadata.get("tool_interface_version", 0)) >= 24
        )

    def _v29_reservation_accountability(self) -> bool:
        return bool(
            self.config.reservation_contract_mode
            == "factorized_line_options_plan_set_v1"
            and int(self.dataset.metadata.get("tool_interface_version", 0)) >= 25
            and self.dataset.metadata.get("liquidation_policy")
            == "flat_nominal_per_unit_v1"
        )

    def _v30_pickup_priority(self) -> bool:
        return bool(
            self._v29_reservation_accountability()
            and int(self.dataset.metadata.get("tool_interface_version", 0)) >= 26
            and self.dataset.metadata.get("reservation_pickup_order")
            == "finalize_due_pickups_before_walk_ins_v1"
        )

    def _reservation_reputation_policy(self) -> dict[str, Any]:
        return dict(
            self.dataset.metadata.get("reservation_service_reputation") or {}
        )

    @staticmethod
    def _service_key_for_line(product_type: str, asset_kind: str) -> str:
        if asset_kind == "consumable":
            return f"retail:{product_type}"
        if product_type in PAD_LINES:
            return "rental:sleeping_pad"
        return f"rental:{product_type}"

    def service_key_for_product_type(self, product_type: str) -> str:
        return self.product_type_to_service_key[product_type]

    def product_types_for_service(self, service_key: str) -> list[str]:
        for row in self._service_registry:
            if row["service_key"] == service_key:
                return list(row["fulfillment_product_types"])
        return []

    def public_service_registry(self) -> list[dict[str, Any]]:
        return [
            {
                "service_key": row["service_key"],
                "kind": row["kind"],
                "fulfillment_product_types": list(row["fulfillment_product_types"]),
            }
            for row in self._service_registry
        ]

    def public_choice_groups(self) -> dict[str, list[str]]:
        return {
            service_key: list(self.catalog.choice_groups.get(service_key, ()))
            for service_key in self.public_service_keys()
        }

    def _reference_prices(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for line, offers in self.by_line.items():
            unit_costs = [offer.unit_cost for offer in offers]
            result[line] = (
                median(unit_costs) / 60.0
                if offers[0].asset_kind == "durable"
                else 1.5 * median(unit_costs)
            )
        return result

    def _public_liquidation_depth_rule(self) -> dict[str, Any]:
        if not self.config.liquidation_depth_enabled:
            return {"enabled": False, "policy": "flat nominal dealer quotes"}
        return {
            "enabled": True,
            "separate_pools": (
                ["rental_asset", "retail_item"]
                if self.config.retail_enabled
                else ["rental_asset"]
            ),
            "rolling_window_days": int(
                self.config.liquidation_depth_window_days
            ),
            "full_rate_until": (
                "the pool-specific full-depth capacity shown in dealer quotes"
            ),
            "marginal_rate_curve": {
                "at_full_depth": 1.0,
                "at_capacity_multiple": float(
                    self.config.liquidation_depth_end_multiple
                ),
                "floor_rate": float(self.config.liquidation_depth_floor_rate),
                "shape": "linear between the two points, then flat",
            },
            "terminal_uses_same_window_and_curve": True,
        }

    def _retail_service_keys(self) -> list[str]:
        return sorted(
            row["service_key"] for row in self._service_registry if row["kind"] == "retail"
        )

    def _event_rng(self, *coordinates: int) -> np.random.Generator:
        """Return the stable random stream assigned to one market event."""
        return self.random.generator(*coordinates)

    def _stable_event_rng(
        self, stream: int, *coordinates: int, identity: str
    ) -> np.random.Generator:
        return self.random.stable_generator(
            stream, *coordinates, identity=identity
        )

    def _offer_rng_identity(self, offer_id: str) -> str:
        """Return an internal listing identity unaffected by public ID changes."""

        return self.offer_rng_identities.get(offer_id, offer_id)

    def supplier_delivery_estimate(self, seller_id: str) -> int:
        """Return one stable, concrete public lead-time estimate per seller."""

        minimum = int(self.config.supplier_delivery_min_days)
        maximum = int(self.config.supplier_delivery_max_days)
        span = maximum - minimum + 1
        return minimum + stable_words(str(seller_id), count=1)[0] % span

    def estimated_order_delivery_day(self, order: PurchaseOrder) -> int:
        if order.estimated_delivery_day is not None:
            return int(order.estimated_delivery_day)
        return order.placed_day + self.supplier_delivery_estimate(order.seller_id)

    def _build_episode_market_profile(self) -> EpisodeMarketProfile:
        model = self.demand_model
        if model is None:
            raise AssertionError("bounded demand model is not loaded")
        expected = {
            f"{service_key}|{group}"
            for service_key, group in self.by_group
        }
        if set(model.groups) != expected:
            raise ValueError(
                "demand model group coverage differs from runtime catalog: "
                f"missing={sorted(expected-set(model.groups))}, "
                f"extra={sorted(set(model.groups)-expected)}"
            )
        max_share = float(model.local_segment_share_max_multiplier)
        raw_shares: dict[str, float] = {}
        for key, segment in sorted(model.customer_segments.items()):
            rng = self._stable_event_rng(
                _RNG_EPISODE_MARKET, identity=f"segment:{key}"
            )
            multiplier = math.exp(
                float(rng.uniform(-math.log(max_share), math.log(max_share)))
            )
            raw_shares[key] = float(segment.share) * multiplier
        total = sum(raw_shares.values())
        segment_shares = {
            key: value / total for key, value in sorted(raw_shares.items())
        }
        groups: dict[str, LocalGroupProfile] = {}
        low_reservation, high_reservation = model.local_reservation_range
        low_shift, high_shift = model.local_additive_shift_range
        for key in sorted(model.groups):
            rng = self._stable_event_rng(
                _RNG_EPISODE_MARKET, identity=f"group:{key}"
            )
            groups[key] = LocalGroupProfile(
                reservation_multiplier=float(
                    rng.uniform(low_reservation, high_reservation)
                ),
                need_shift=float(rng.uniform(low_shift, high_shift)),
                accessibility_shift=float(rng.uniform(low_shift, high_shift)),
                request_rate_multiplier=float(
                    rng.uniform(low_reservation, high_reservation)
                ),
            )
        return EpisodeMarketProfile(segment_shares=segment_shares, groups=groups)

    def _temperatures(self, day: int) -> tuple[float, float, float]:
        calendar = self._calendar(day)
        values = (
            calendar.get("temp_mean_c"),
            calendar.get("night_low_c"),
            calendar.get("day_high_c"),
        )
        if all(value is not None for value in values):
            return tuple(float(value) for value in values)  # type: ignore[return-value]
        month = int(str(calendar["date"])[5:7])
        monthly_mean = {
            1: -3.2, 2: 0.5, 3: 7.5, 4: 14.7, 5: 20.5, 6: 27.2,
            7: 28.9, 8: 27.1, 9: 22.8, 10: 17.0, 11: 7.5, 12: -0.5,
        }
        base_range = {
            1: 10, 2: 11, 3: 12, 4: 12, 5: 11, 6: 10,
            7: 8, 8: 8, 9: 10, 10: 11, 11: 11, 12: 10,
        }
        adjustment = {
            "sunny": 2, "cloudy": -1, "rainy": -3,
            "hot": 0, "cold": 1, "snowy": -2,
        }
        mean = monthly_mean[month]
        span = base_range[month] + adjustment[self._weather(day)]
        return mean, mean - 0.5 * span, mean + 0.5 * span

    def resolved_rules(self) -> dict[str, Any]:
        complete_contract_choice = bool(
            self.config.reservation_contract_mode
            == "complete_contract_options_v1"
            and int(self.dataset.metadata.get("tool_interface_version", 0))
            >= 15
        )
        factorized_plan_choice = bool(
            self.config.reservation_contract_mode
            == "factorized_line_options_plan_set_v1"
            and int(self.dataset.metadata.get("tool_interface_version", 0)) >= 16
        )
        noisy_line_mentions = bool(
            factorized_plan_choice
            and self.plan_set_feedback_policy is not None
            and self.plan_set_feedback_policy.noisy_line_mentions_enabled
        )
        factorized_concern_tags = bool(
            factorized_plan_choice
            and self.plan_set_feedback_policy is not None
            and self.plan_set_feedback_policy.factorized_concern_tags_enabled
        )
        capacity_surface_v22 = bool(
            factorized_plan_choice
            and int(self.dataset.metadata.get("tool_interface_version", 0)) >= 18
        )
        request_bundle_choice = bool(
            factorized_plan_choice
            or complete_contract_choice
            or (
                self.config.reservation_contract_mode
                == "adaptive_pairwise_exact_quotes_v1"
                and int(self.dataset.metadata.get("tool_interface_version", 0))
                >= 14
            )
        )
        if (
            self.channel_demand_model is not None
            and self.config.demand_world_id != "control"
        ):
            model = self.channel_demand_model
            demand_rules: dict[str, Any] = {
                "version": model.format,
                "structure": "exact_offer_hidden_channel_choice_with_role_aware_store_plans",
                "demand_world_id": self.config.demand_world_id,
                "supplier_cost_enters_customer_demand": False,
                "online_price_generates_natural_demand": False,
                "offer_shock_enters_channel_choice": False,
                "agent_observes_hidden_channel": False,
                "duplicate_policy": (
                    "behavioral profiles are deduplicated; compatible profiles may mix "
                    "for quantity fulfillment; inventory beyond requested units has no utility"
                ),
                "multi_unit_policy": {
                    "required": "full fulfillment, compatible-profile mixing allowed",
                    "optional": "partial fulfillment and compatible-profile mixing allowed",
                    "profile_shock_stage": "conditional store-plan choice only",
                },
                "world": asdict(model.worlds[self.config.demand_world_id]),
            }
        elif (
            self.reference_demand_model is not None
            and self.config.demand_world_id != "control"
        ):
            model = self.reference_demand_model
            demand_rules: dict[str, Any] = {
                "version": model.format,
                "structure": "reference_transaction_shop_entry_then_profile_choice",
                "demand_world_id": self.config.demand_world_id,
                "supplier_cost_enters_customer_demand": False,
                "offer_shock_enters_shop_entry": False,
                "hard_compatibility_before_shop_entry": True,
                "duplicate_policy": "unique consumer profiles; inventory depth affects fulfillment only",
                "reference_cells": len(model.cells),
                "reference_cells_synthetic": sum(
                    row.synthetic for row in model.cells.values()
                ),
                "world": asdict(model.worlds[self.config.demand_world_id])
                if self.config.demand_world_id in model.worlds
                else None,
            }
        elif self.demand_model is not None:
            demand_rules: dict[str, Any] = {
                "version": self.demand_model.format,
                "structure": "bounded_beta_pert_reservation_hurdle_then_profile_choice",
                "supplier_cost_enters_customer_demand": False,
                "arrival_scale": round(
                    self.rules.arrival_scale
                    * self.config.arrival_scale_multiplier,
                    6,
                ),
                "profile_identity_excludes": [
                    "offer_id",
                    "supplier_cost",
                    "inventory_depth",
                ],
                "profile_taste_range": list(
                    self.demand_model.profile_taste_range
                ),
                "choice_taste_range": list(
                    self.demand_model.choice_taste_range
                ),
                "episode_market_profile": self.episode_market_profile.to_dict()
                if self.episode_market_profile is not None
                else None,
                "group_parameters": [
                    {
                        "key": key,
                        "value_unit": row.value_unit,
                        "value_unit_kind": row.value_unit_kind,
                        "reservation_low_cny": row.reservation_low_cny,
                        "reservation_central_cny": row.reservation_central_cny,
                        "reservation_high_cny": row.reservation_high_cny,
                        "customer_variation": row.customer_variation,
                        "need_strength": row.need_strength,
                        "quality_salience": row.quality_salience,
                        "alternative_accessibility": row.alternative_accessibility,
                        "substitution_strength": row.substitution_strength,
                    }
                    for key, row in sorted(self.demand_model.groups.items())
                ],
                "duplicate_policy": (
                    "consumer-perceived profiles share bounded taste and aggregate "
                    "inventory; identical offers do not create category demand"
                ),
            }
        else:
            demand_rules = {
                "version": self.rules.version,
                "structure": "legacy_choice_session_stable_hierarchical_gumbel_nested_logit",
                "service_intercept": self.service_intercept,
                "price_beta_mean": self.rules.price_beta_mean,
                "price_beta_sigma": self.rules.price_beta_sigma,
                "lambda_offer": self.rules.lambda_offer,
                "lambda_group": self.rules.lambda_group,
                "deposit_friction": self.rules.deposit_friction,
                "arrival_scale": round(
                    self.rules.arrival_scale
                    * self.config.arrival_scale_multiplier,
                    6,
                ),
                "quality_adjustments": self.rules.quality_values,
                "choice_groups": self.public_choice_groups(),
                "duplicate_policy": "legacy release-level decision-signature dedupe",
            }
        return {
            **(
                {"assortment_mode": self.config.assortment_mode}
                if self.dataset.metadata.get("supported_assortment_modes")
                else {}
            ),
            "demand": demand_rules,
            **({"warehouse_capacity": {
                "policy": self.dataset.metadata["warehouse_capacity_policy"],
                "limit_units": self.config.warehouse_capacity_limit,
                "weights_sha256": self.dataset.warehouse_capacity_weights_sha256,
                "initial_occupied_units": sum(self._warehouse_weight(row["offer_id"]) for row in self.dataset.initial_assets),
                "additional_fee": 0,
            }} if self._v31_warehouse_capacity() else {}),
            **(
                {
                    (
                        "durable_capital"
                        if self._v40_revolving_durable_capital()
                        else "durable_acquisition_budget"
                    ): {
                        "policy": self.dataset.metadata["durable_acquisition_budget_policy"],
                        **(
                            {
                                "configured_additional_durable_capital_headroom": (
                                    self.config.durable_acquisition_budget
                                ),
                                "release_delay_days": 1,
                            }
                            if self._v40_revolving_durable_capital()
                            else {
                                "initial_durable_acquisition_budget": (
                                    self.config.durable_acquisition_budget
                                )
                            }
                        ),
                        "retail_purchases_in_scope": False,
                        (
                            "unused_headroom_has_terminal_value"
                            if self._v40_revolving_durable_capital()
                            else "unused_budget_has_terminal_value"
                        ): False,
                    }
                }
                if self._durable_capital_control_enabled()
                else {}
            ),
            **(
                {"optional_line_omit_mode": "implicit_singleton"}
                if self._v37_supplier_trial()
                else {}
            ),
            **(
                {"supplier_trial": self._public_supplier_trial_policy()}
                if self._v37_supplier_trial()
                else {}
            ),
            "economics": {
                "daily_operating_fee": self.config.daily_fee,
                "retail_daily_holding_rate": self.config.retail_holding_rate,
                "rental_daily_holding_rate": (
                    None
                    if self.config.lifecycle_economics_enabled
                    else self.config.rental_holding_rate
                ),
                "rental_idle_daily_holding_rate": (
                    None
                    if self.config.lifecycle_economics_enabled
                    else self.config.rental_idle_holding_rate
                ),
                "rental_holding_basis": (
                    "opening physically held assets after morning processing; disposed/sold assets excluded; "
                    + (
                        "owned assets pay capital charges, while owned and supplier-trial assets both pay on-site space charges; "
                        if self._v37_supplier_trial()
                        else "capital and on-site space fee rates are unchanged; "
                    )
                    + "same-day sale/disposal remains chargeable"
                    if self._v31_warehouse_capacity()
                    else
                    "opening unsold assets after morning processing: every asset pays "
                    "replacement-cost capital charge and only on-site non-rented assets "
                    "pay storage-unit space charge; same-day sale remains chargeable"
                    if self.config.lifecycle_economics_enabled
                    else "opening assets after morning processing, excluding retired and sold; "
                    "same-day sale or retirement remains chargeable"
                ),
                "cleaning_cost": {
                    "formula": (
                        "offer_world_work_points * cash_per_work_point; the exact same "
                        "work_points drive the FIFO queue and cash charge"
                        if self.config.lifecycle_economics_enabled
                        else "base_cost_by_product_type + extra_per_additional_day "
                        "* max(duration_days - 1, 0)"
                    ),
                    "basis": "once per normally returned rental contract",
                    **(
                        {
                            "cash_per_work_point": float(
                                self.lifecycle_scales.cleaning_cash_per_work_point
                            ),
                            "extra_work_fraction_per_additional_day": float(
                                self.lifecycle_scales.cleaning_extra_work_fraction_per_additional_day
                            ),
                        }
                        if self.config.lifecycle_economics_enabled
                        and self.lifecycle_scales is not None
                        else {
                            "rates": {
                                product_type: {
                                    "base_cost": base_cost,
                                    "extra_per_additional_day": extra_per_day,
                                }
                                for product_type, (base_cost, extra_per_day) in sorted(
                                    CLEANING_COST_RULES.items()
                                )
                            }
                        }
                    ),
                },
                "cleaning_capacity": {
                    "daily_work_points": self.rules.cleaning_capacity_points,
                    "product_work_points": {
                        product_type: {
                            "base": values[0],
                            "extra_per_additional_rental_day": values[1],
                        }
                        for product_type, values in sorted(
                            self.rules.cleaning_work_points.items()
                        )
                    },
                    "queue_order": "return_day_then_asset_id",
                    "partial_work_carries_across_days": True,
                    "scope": (
                        "in-season availability only; it constrains when an asset can "
                        "be rented again"
                    ),
                },
                "repair_cost": (
                    "one severity-derived condition loss times replacement cost, "
                    "subject to the public minimum"
                    if self.config.lifecycle_economics_enabled
                    else "max(10, 0.20 * unit_cost)"
                ),
                "damage_deposit_deduction": (
                    "the same condition-loss repair cash times customer-liability share, "
                    "capped by the frozen contract deposit"
                    if self.config.lifecycle_economics_enabled
                    else "min(deposit, max(5, 0.10 * unit_cost))"
                ),
                "damage_daily_rates": DAMAGE_RATES,
                "terminal_retail_salvage_rate": RETAIL_SALVAGE_RATE,
                "retail_liquidation_market": {
                    "eligible_inventory": "delivered_on_hand_consumer_units",
                    "average_cost_rate": RETAIL_SALVAGE_RATE,
                    "unit_quote_rounding": "ROUND_HALF_UP_to_CNY_0.01_before_multiplying_units",
                    "terminal_valuation_uses_same_unit_quote": True,
                    "in_transit_inventory_sellable": False,
                    "automatic_insolvency_sale": False,
                    "rolling_depth": self._public_liquidation_depth_rule(),
                },
                "secondhand_market": {
                    **(
                        {
                            "nominal_formula": (
                                "replacement_cost * global_new_asset_residual_rate * "
                                "condition_ratio * remaining_combined_wear^global_exponent"
                            ),
                            "uses_rent_or_demand": False,
                        }
                        if self.config.lifecycle_economics_enabled
                        else {
                            "grade_schedule": [
                                {"grade": grade, "minimum_remaining_ratio": threshold, "cost_rate": rate}
                                for grade, threshold, rate in SECONDHAND_GRADE_SCHEDULE
                            ],
                            "damaged_quote": "max(0.05 * unit_cost, healthy_grade_quote - repair_cost)",
                        }
                    ),
                    "sellable_statuses": sorted(SECONDHAND_SELLABLE_STATUSES),
                    "terminal_healthy_statuses": sorted(TERMINAL_HEALTHY_STATUSES),
                    "terminal_damaged_statuses": ["damaged"],
                    **(
                        {
                            "instantaneous_zero_value_statuses": [
                                "rented",
                                "retired",
                                "sold",
                            ],
                            "final_scoring_rental_invariant": (
                                "every admitted rental returns by runoff_end_day before "
                                "the final score is evaluated"
                            ),
                        }
                        if self._v24_reservation_operations()
                        else {
                            "terminal_zero_value_statuses": [
                                "rented",
                                "retired",
                                "sold",
                            ]
                        }
                    ),
                    "terminal_cleaning_policy": (
                        "post-season time is unlimited: each cleaning asset receives "
                        "its healthy dealer quote less its full unpaid scheduled "
                        "cleaning cost; the in-season daily work capacity creates no "
                        "additional terminal discount"
                    ),
                    "rolling_depth": self._public_liquidation_depth_rule(),
                },
                "terminal_in_transit_value": 0.0,
                "diagnostic_valuations": {
                    "official_score": (
                        (
                            "instantaneous liquidation diagnostic; active rented assets "
                            "and in-transit orders are zero. Final scoring occurs only on "
                            "runoff_end_day after every admitted rental has returned"
                        )
                        if self._v24_reservation_operations()
                        else "immediate liquidation value; rented assets and in-transit "
                        "orders are zero"
                    ),
                    "going_concern_business_value": (
                        "official retail salvage plus expected delivered order value "
                        "and expected post-return rental asset value, net of expected "
                        "cleaning and including expected damage deductions"
                    ),
                    "book_net_assets": (
                        "cash minus deposits plus retail book value, remaining-life "
                        "rental book value, and prepaid in-transit order cost"
                    ),
                },
            },
            "operations": {
                "weather_start_day": self.config.weather_start_day,
                "season_start_calendar": self._public_calendar(1),
                "season_end_calendar": self._public_calendar(self.config.max_days),
                **(
                    {
                        "decision_horizon_calendar": self._public_calendar(
                            self.config.max_days
                        ),
                        "runoff_end_calendar": self._public_calendar(
                            self.config.max_days + int(self.config.runoff_days)
                        ),
                        "runoff_end_day": self.config.max_days
                        + int(self.config.runoff_days),
                        "final_scoring_day": self.config.max_days
                        + int(self.config.runoff_days),
                    }
                    if self._v24_reservation_operations()
                    else {}
                ),
                "supplier_delivery_days": {
                    "policy": (
                        "each seller listing shows one concrete estimated delivery "
                        "time; estimates are not guaranteed"
                    ),
                },
                "frozen_supplier_inventory": (
                    "unlimited at every admitted exact offer; live marketplace "
                    "quantity is deliberately not replayed"
                ),
                "rental_asset_dispatch": "highest_combined_wear_first_within_offer",
                "cross_horizon_rentals": (
                    "decision-horizon pickups may return during runoff"
                    if self._v24_reservation_operations()
                    else False
                ),
                "rental_product_types_agent_controllable": True,
                "random_event_layout": "time_and_event_addressed_independent_streams",
                "counter_log_retention_completed_days": 2,
                "decision_horizon_day": self.config.max_days,
                "automatic_runoff_days": int(self.config.runoff_days),
                "preventive_maintenance": {
                    "enabled": bool(self.config.preventive_maintenance_enabled),
                    **(
                        {
                            "trigger": "combined-wear exposure or unsafe condition",
                            "completion_effect": (
                                "moves maintenance wear anchor and grants decaying hazard "
                                "protection; never resets cumulative wear"
                            ),
                            "ordinary_overdue_policy": "risk signal without automatic hold",
                            "safety_hold": "condition ratio below 0.65",
                        }
                        if self.config.lifecycle_economics_enabled
                        else {
                            "interval_days": int(self.config.maintenance_interval_days),
                            "due_window_days": int(self.config.maintenance_due_window_days),
                            "duration_days": int(self.config.maintenance_duration_days),
                            "cost_rate": float(self.config.maintenance_cost_rate),
                        }
                    ),
                },
                "advance_reservations": {
                    "enabled": bool(self.config.reservations_enabled),
                    "latent_rental_trip_share": float(
                        self.config.reservation_share
                    ),
                    "booking_lead_days": [
                        int(self.config.reservation_booking_lead_min_days),
                        int(self.config.reservation_booking_lead_max_days),
                    ],
                    "channel_choice_timing": (
                        "each internally formed complete plan versus the same outside "
                        "alternatives after one factorized manager quote; confirmation "
                        "follows the customer-authorized plan set"
                        if factorized_plan_choice
                        else
                        "one or two explicit complete store contracts versus complete "
                        "outside alternatives after one manager quote"
                        if complete_contract_choice
                        else
                        "one complete store bundle versus complete outside "
                        "alternatives after one manager contract quote"
                        if request_bundle_choice
                        else "after one manager contract quote"
                    ),
                    "inquiry_generation": (
                        "routed latent trips whose required services remain open "
                        "and have hard-compatible public promise profiles; "
                        "independent of posted price and current inventory"
                        if self.config.reservation_contract_mode
                        in {
                            "promise_profiles_v1",
                            "adaptive_pairwise_exact_quotes_v1",
                            "complete_contract_options_v1",
                            "factorized_line_options_plan_set_v1",
                        }
                        else "routed latent trips whose required services remain open "
                        "and have hard-compatible supplier-catalog candidates; "
                        "independent of posted price and current inventory"
                    ),
                    "line_quote_unit": (
                        "one or two binding exact offer/profile/price options per required "
                        "line; optional lines add an explicit omit option"
                        if factorized_plan_choice
                        else
                        "one or two explicit complete store contracts; every quoted line "
                        "contains one binding exact offer/profile and its own per-asset "
                        "per-day price"
                        if complete_contract_choice
                        else
                        "one or two binding exact offer/profile alternatives, each "
                        "with its own per-asset per-day price"
                        if self.config.reservation_contract_mode
                        == "adaptive_pairwise_exact_quotes_v1"
                        else "one selected public promise profile and one per-asset "
                        "per-day price per line"
                        if self.config.reservation_contract_mode
                        == "promise_profiles_v1"
                        else "one per-asset per-day price for all exact candidates"
                    ),
                    "line_quote_completeness": (
                        "required lines need one or two quote options; optional lines need "
                        "zero to two quote options plus exactly one omit; validation is atomic"
                        if factorized_plan_choice
                        else
                        "every complete contract must resolve every inquiry line; required "
                        "lines must quote and optional lines must explicitly quote or omit"
                        if complete_contract_choice
                        else
                        "every required and optional line must receive one or two complete "
                        "exact/profile/cent-exact-price options; missing, duplicate, or "
                        "unknown lines and duplicate exact options reject the whole batch"
                        if self.config.reservation_contract_mode
                        == "adaptive_pairwise_exact_quotes_v1"
                        else "every required and optional line must receive one explicit "
                        "cent-exact price; missing, duplicate, or unknown lines reject "
                        "the whole tool batch"
                    ),
                    "quote_rounds": 1,
                    "primary_offer_and_price_locked_at_confirmation": (
                        self.config.reservation_contract_mode
                        != "promise_profiles_v1"
                    ),
                    "authorized_offer_count": (
                        1
                        if self.config.reservation_contract_mode
                        in {
                            "adaptive_pairwise_exact_quotes_v1",
                            "complete_contract_options_v1",
                            "factorized_line_options_plan_set_v1",
                        }
                        else "all customer-acceptable members of the frozen selected profile"
                        if self.config.reservation_contract_mode
                        == "promise_profiles_v1"
                        else [1, 3]
                    ),
                    "authorization_mode": (
                        "customer_selected_binding_exact"
                        if self.config.reservation_contract_mode
                        in {
                            "adaptive_pairwise_exact_quotes_v1",
                            "complete_contract_options_v1",
                            "factorized_line_options_plan_set_v1",
                        }
                        else "complete_profile_authorization_subset"
                        if self.config.reservation_contract_mode
                        == "promise_profiles_v1"
                        else str(self.config.reservation_authorization_mode)
                    ),
                    "authorization_disclosure": (
                        "quote results disclose at most three fully expanded customer-authorized "
                        "plans and the customer preferred plan; non-returned combinations are "
                        "non-exhaustive and outside choices remain private"
                        if factorized_plan_choice
                        else
                        "inquiries disclose compact profile families and paginated exact "
                        "members; quote results disclose all submitted complete contracts, "
                        "the selected contract if any, and only noisy reported assessments "
                        "for unselected contracts; outside choices remain private"
                        if complete_contract_choice
                        else
                        "inquiries disclose compact profile families and paginated exact "
                        "members; accepted quotes disclose the selected binding exact, "
                        "while rejected quotes disclose no failed line, hidden selection, "
                        "ceiling, or reason"
                        if self.config.reservation_contract_mode
                        == "adaptive_pairwise_exact_quotes_v1"
                        else "inquiries disclose compact profile families; profile members "
                        "are paginated as public catalog facts and authorized members "
                        "are disclosed only after confirmation; willingness ceilings "
                        "and rejection reasons remain hidden"
                        if self.config.reservation_contract_mode
                        == "promise_profiles_v1"
                        else "each inquiry discloses one to three ordered hard-compatible "
                        "supplier-catalog candidates, whether or not currently owned; "
                        "willingness ceilings and rejection reasons remain hidden"
                    ),
                    "walk_in_price_separation": (
                        "posted exact-offer prices apply only to walk-ins and never "
                        "supply or change an advance contract quote"
                    ),
                    "asset_binding": (
                        "the customer-selected exact offer is locked at confirmation; "
                        "the physical asset is assigned at pickup and exact substitution "
                        "requires explicit risk fallback"
                        if self.config.reservation_contract_mode
                        in {
                            "adaptive_pairwise_exact_quotes_v1",
                            "complete_contract_options_v1",
                            "factorized_line_options_plan_set_v1",
                        }
                        else "optional exact-offer staging before pickup without asset "
                        "reservation; pickup tries staged then the remaining frozen "
                        "authorization order; one exact offer supplies each whole line"
                        if self.config.reservation_contract_mode
                        == "promise_profiles_v1"
                        else "pickup morning; one exact offer supplies each whole line "
                        "using the public primary-first authorization order"
                    ),
                    "prepayment_rate": float(
                        self.config.reservation_prepayment_rate
                    ),
                    "prepayment_accounting": (
                        "fully refundable cash receipt and unearned revenue until pickup"
                    ),
                    **(
                        {
                            "extension_accounting": (
                                "the request freezes total and per-line additional "
                                "rent for fulfilled lines only; acceptance collects "
                                "that disclosed amount immediately"
                            )
                        }
                        if self._v27_extension_cash_audit()
                        else {}
                    ),
                    "unanswered_request_default": "expired_unquoted",
                    "calendar_surface": (
                        (
                            "view_rental_capacity raw current assets, explicit pickup/"
                            "occupied-through/return dates, walk-in limits and occupancy, "
                            "public delivery timing, and current service jobs; no feasibility verdict"
                            if self._v24_reservation_operations()
                            else "view_rental_capacity raw current assets, binding intervals, "
                            "public delivery timing, and current service jobs; no feasibility verdict"
                        )
                        if capacity_surface_v22
                        else "raw assets, commitments, and deliveries; no feasibility verdict"
                    ),
                    "weather_and_trip_changes_enabled": bool(
                        self.config.reservation_disruptions_enabled
                    ),
                    **(
                        {
                            "customer_weather_information": (
                                "advance requests use the request-day public forecast; "
                                "walk-ins use arrival-day observed weather; one latent "
                                "customer has only one fixed channel opportunity"
                            ),
                            "forecast_reconsideration": (
                                "a confirmed customer may react once to each newly "
                                "worse public forecast state; retaining the booking "
                                "also records that state as reviewed"
                            ),
                            "non_weather_cancellation": (
                                "an independent customer event may cancel a confirmed "
                                "booking before pickup; it is separate from forecast changes"
                            ),
                        }
                        if self._v42_weather_information_coherence()
                        else {}
                    ),
                    "late_return_probability_after_declined_extension": float(
                        self.config.late_return_probability_after_extension_decline
                    ),
                    "bundle_mode": (
                        "2-4 durable lines; required pickup is atomic and optional "
                        "lines receive independent credits"
                        if self.config.reservation_bundles_enabled
                        else "single-line"
                    ),
                    "bundle_customer_latent_scope": (
                        "segment, forecast trust, overall price sensitivity, and "
                        "return burden are request-level; storage and reuse use one "
                        "request-level base plus category adjustments; use case, exact "
                        "model, quality, portability, and substitution remain line-level"
                    ),
                    "latent_snapshot_semantics": (
                        "immutable inquiry-matched offers are retained separately from "
                        "post-quote authorized and full-contract recovery sets; no "
                        "production per-line price ceiling is generated"
                        if request_bundle_choice
                        else "immutable inquiry offers and ceilings are retained separately "
                        "from post-quote authorized and recovery sets"
                    ),
                    **(
                        {
                            "contract_mode": self.config.reservation_contract_mode,
                            "profile_schema_version": (
                                self.reservation_promise_catalog.schema_version
                                if self.reservation_promise_catalog is not None
                                else None
                            ),
                            "profile_deposit_policy": (
                                self.config.reservation_profile_deposit_policy
                            ),
                            "profile_asset_quantity": (
                                "ceil(service_units / profile capacity_per_asset)"
                            ),
                            "profile_line_mixing": False,
                            "customer_rejection_feedback": (
                                "independent noisy price/equipment concern tags, or one "
                                "exclusive residual, only when the authorized plan set "
                                "is empty"
                                if factorized_concern_tags
                                else "one request-level noisy concern only when the "
                                "authorized plan set is empty"
                                if factorized_plan_choice
                                else
                                "request-shared partial response; every unselected complete "
                                "contract receives one noisy coarse reported assessment"
                                if complete_contract_choice
                                else "none"
                            ),
                            **(
                                {
                                    "required_line_option_count": [1, 2],
                                    "optional_line_quote_option_count": [0, 2],
                                    "optional_line_disposition": ["quote", "omit"],
                                    "maximum_internal_plan_count": 24,
                                    "maximum_returned_authorized_plan_count": 3,
                                    "authorization_is_exhaustive": False,
                                    "customer_choice_scope": (
                                        "each complete plan independently versus one frozen "
                                        "outside plan and abandonment"
                                    ),
                                    "manager_confirmation": (
                                        "choose any returned plan; customer preferred plan "
                                        "auto-confirms at day end"
                                    ),
                                    "negative_feedback_scope": "empty_authorized_set_only",
                                    **(
                                        {
                                            "customer_authorization_reads_operations": False,
                                            "manager_bears_binding_capacity_risk": True,
                                            "same_exact_profile_pair_at_multiple_prices": False,
                                            "quote_batch_validation": (
                                                "atomic_all_invalid_items"
                                            ),
                                        }
                                        if capacity_surface_v22
                                        else {}
                                    ),
                                    **(
                                        {
                                            "reported_concern_tags": [
                                                "price_value",
                                                "equipment_configuration",
                                            ],
                                            "reported_concern_tags_can_cooccur": True,
                                            "residual_concern": "other_or_unclear",
                                            "residual_concern_can_cooccur": False,
                                            "concern_tag_absence_exonerates_axis": False,
                                        }
                                        if factorized_concern_tags
                                        else {
                                            "reported_decline_categories": [
                                                "price_value_concern",
                                                "equipment_fit_concern",
                                                "multiple_or_unclear_concerns",
                                            ]
                                        }
                                    ),
                                    **(
                                        {
                                            "equipment_feedback_line_mention": (
                                                "zero or one noisy subjective "
                                                "customer_mentioned_line_id; only with "
                                                "equipment_configuration"
                                                if factorized_concern_tags
                                                else "zero or one noisy subjective "
                                                "customer_mentioned_line_id"
                                            ),
                                            "line_mention_is_failed_line_truth": False,
                                            "unmentioned_lines_are_exonerated": False,
                                            "line_mention_guarantees_repair": False,
                                        }
                                        if noisy_line_mentions
                                        else {}
                                    ),
                                    "risk_recovery_scope": (
                                        "manager-confirmed full locked plan replay"
                                    ),
                                }
                                if factorized_plan_choice
                                else {}
                            ),
                            **(
                                {
                                    "complete_contract_option_count": [1, 2],
                                    "implicit_cartesian_product": False,
                                    "optional_line_disposition": ["quote", "omit"],
                                    "reported_assessment_is_true_reason": False,
                                    "outside_resolution_disclosed": False,
                                    "customer_choice_scope": (
                                        "one or two explicit complete store contracts; "
                                        "required lines atomic"
                                    ),
                                    "risk_recovery_scope": (
                                        "selected full locked contract replay"
                                    ),
                                }
                                if complete_contract_choice
                                else {}
                            ),
                            **(
                                {
                                    "line_option_count": [1, 2],
                                    "option_order_affects_utility": False,
                                    "selected_exact_locked": True,
                                    **(
                                        {
                                            "customer_choice_scope": (
                                                "request-level complete bundle; required "
                                                "lines atomic; optional lines skippable"
                                            ),
                                            "risk_recovery_scope": (
                                                "full locked bundle replay"
                                            ),
                                        }
                                        if request_bundle_choice
                                        else {}
                                    ),
                                    "quote_choice_history": (
                                        "immutable raw searchable paginated facts"
                                        if int(
                                            self.dataset.metadata.get(
                                                "tool_interface_version", 0
                                            )
                                        )
                                        >= 13
                                        else "immutable raw paginated facts"
                                    ),
                                    **(
                                        {
                                            "quote_history_query": (
                                                "exact request/commitment IDs or public "
                                                "date, rental offer, rental service, "
                                                "aggregate outcome, and menu-size filters; "
                                                "complete bundles; newest first by default"
                                            )
                                        }
                                        if int(
                                            self.dataset.metadata.get(
                                                "tool_interface_version", 0
                                            )
                                        )
                                        >= 13
                                        else {}
                                    ),
                                }
                                if self.config.reservation_contract_mode
                                == "adaptive_pairwise_exact_quotes_v1"
                                else {}
                            ),
                        }
                        if self.config.reservation_contract_mode
                        in {
                            "promise_profiles_v1",
                            "adaptive_pairwise_exact_quotes_v1",
                            "complete_contract_options_v1",
                            "factorized_line_options_plan_set_v1",
                        }
                        else {}
                    ),
                },
            },
        }

    def public_operating_rules(self) -> dict[str, Any]:
        """Business terms needed by the shop manager for bookkeeping and planning."""
        complete_contract_choice = bool(
            self.config.reservation_contract_mode
            == "complete_contract_options_v1"
            and int(self.dataset.metadata.get("tool_interface_version", 0))
            >= 15
        )
        factorized_plan_choice = bool(
            self.config.reservation_contract_mode
            == "factorized_line_options_plan_set_v1"
            and int(self.dataset.metadata.get("tool_interface_version", 0)) >= 16
        )
        noisy_line_mentions = bool(
            factorized_plan_choice
            and self.plan_set_feedback_policy is not None
            and self.plan_set_feedback_policy.noisy_line_mentions_enabled
        )
        factorized_concern_tags = bool(
            factorized_plan_choice
            and self.plan_set_feedback_policy is not None
            and self.plan_set_feedback_policy.factorized_concern_tags_enabled
        )
        capacity_surface_v22 = bool(
            factorized_plan_choice
            and int(self.dataset.metadata.get("tool_interface_version", 0)) >= 18
        )
        request_bundle_choice = bool(
            factorized_plan_choice
            or complete_contract_choice
            or (
                self.config.reservation_contract_mode
                == "adaptive_pairwise_exact_quotes_v1"
                and int(self.dataset.metadata.get("tool_interface_version", 0))
                >= 14
            )
        )
        rules = {
            **(
                {"assortment_mode": self.config.assortment_mode}
                if self.dataset.metadata.get("supported_assortment_modes")
                else {}
            ),
            "season_end_day": self.config.max_days,
            **({"warehouse_capacity": {
                "limit_units": self.config.warehouse_capacity_limit,
                "unit_semantics": "SPACE CONSUMED per stowed whole rental asset, not how many assets fit: larger weight means more warehouse space used. Two assets of weight 6 use 12 units of N. Fixed relative integers, not measured liters.",
                "ownership_scope": "all held rental assets, including rented/cleaning/damaged/repairing/maintenance, plus confirmed inbound quantities",
                "return_space_reserved": True, "retail_in_scope": False,
                "capacity_purchasable": False, "additional_fee": 0,
                "procurement": "Reserve packs times package_units at order acceptance. An over-limit mixed order fails atomically. Receipts transfer inbound to held without adding occupancy.",
                "release": "Only an actual sale or physical disposal releases capacity. Retirement by the manager disposes damaged assets immediately; exhausted assets are automatically disposed on actual return. Records remain auditable. Changing status alone never releases space.",
                "reservations": "Customer commitments do not add physical warehouse occupancy. Warehouse checks do not check rental scheduling or reservation feasibility.",
                "query": "view_warehouse_capacity",
            }} if self._v31_warehouse_capacity() else {}),
            **(
                {
                    (
                        "durable_capital"
                        if self._v40_revolving_durable_capital()
                        else "durable_acquisition_budget"
                    ): (
                        {
                            "configured_additional_durable_capital_headroom": (
                                self.config.durable_acquisition_budget
                            ),
                            "capital_limit": (
                                "initial owned durable historical purchase basis plus "
                                "configured additional headroom"
                            ),
                            "acceptance": (
                                "accepted unreceived owned durable unit basis enters committed capital"
                            ),
                            "delivery": (
                                "received unit basis moves from committed to deployed without "
                                "changing gross occupied capital"
                            ),
                            "release": (
                                "dealer sale or completed owned-asset retirement releases its "
                                "frozen historical basis at the next business day"
                            ),
                            "not_released_by": [
                                "depreciation",
                                "damage",
                                "repair",
                                "cleaning",
                                "maintenance",
                                "rental_use",
                                "temporary_service_closure",
                                "supplier_trial_return",
                            ],
                            "cash_is_independent": True,
                            "revenue_increases_limit": False,
                            "retail_purchases_in_scope": False,
                            "unused_headroom_has_terminal_value": False,
                        }
                        if self._v40_revolving_durable_capital()
                        else {
                            "initial_durable_acquisition_budget": (
                                self.config.durable_acquisition_budget
                            ),
                            "acceptance": (
                                "accepted durable order line cost moves immediately "
                                "into committed budget"
                            ),
                            "delivery": (
                                "a fully delivered order moves its durable amount from "
                                "committed to used without changing remaining budget"
                            ),
                            "replenishment": "none",
                            "not_restored_by": [
                                "rental_revenue",
                                *(
                                    ["retail_revenue"]
                                    if self.config.retail_enabled
                                    else []
                                ),
                                "credit",
                                "asset_sale",
                                "asset_retirement",
                            ],
                            "operating_costs_in_scope": False,
                            "retail_purchases_in_scope": False,
                            "unused_budget_has_terminal_value": False,
                        }
                    )
                }
                if self._durable_capital_control_enabled()
                else {}
            ),
            **(
                {"optional_line_omit_mode": "implicit_singleton"}
                if self._v37_supplier_trial()
                else {}
            ),
            **(
                {"supplier_trial": self._public_supplier_trial_policy()}
                if self._v37_supplier_trial()
                else {}
            ),
            "season_start_calendar": self._public_calendar(1),
            "season_end_calendar": self._public_calendar(self.config.max_days),
            **(
                {
                    "reservation_operations_v24": True,
                    "decision_horizon_day": self.config.max_days,
                    "runoff_end_day": self.config.max_days
                    + int(self.config.runoff_days),
                    "final_scoring_day": self.config.max_days
                    + int(self.config.runoff_days),
                    "decision_horizon_calendar": self._public_calendar(
                        self.config.max_days
                    ),
                    "runoff_end_calendar": self._public_calendar(
                        self.config.max_days + int(self.config.runoff_days)
                    ),
                    "new_order_estimated_delivery_cutoff": (
                        "decision_horizon_day; already accepted orders may realize a "
                        "supplier delay into runoff"
                    ),
                }
                if self._v24_reservation_operations()
                else {}
            ),
            "daily_operating_fee": round(self.config.daily_fee, 2),
            **({"flow_timing_note": (
                "today_morning_flows is fixed after opening processing. Later "
                "customer refunds appear in today_intraday_flows; daily refunds "
                "include both. Refund totals retain the existing shortage/credit "
                "accounting treatment."
            )} if self._v29_reservation_accountability() else {}),
            "opening_price_policy": (
                "all exact offers start unpriced; an unpriced offer cannot be quoted "
                "to or sold to a walk-in until the manager posts a positive price; "
                "advance contract inquiries always use separate explicit line prices"
                if self.channel_demand_model is not None
                else "runtime opening prices are already posted"
            ),
            "management_service_usage_charge": {
                "input_per_million_tokens": self.config.input_token_cost_per_million,
                "output_per_million_tokens": self.config.output_token_cost_per_million,
                "charged_to_cash": self.config.charge_model_usage_to_cash,
            },
            **(
                {
                    "lifecycle_economics": {
                        "status": self.lifecycle_scales.status,
                        "claim_boundary": self.lifecycle_scales.claim_boundary,
                        "cleaning_cash_per_work_point": (
                            self.lifecycle_scales.cleaning_cash_per_work_point
                        ),
                        "annual_capital_rate": self.lifecycle_scales.annual_capital_rate,
                        "daily_space_rate_per_storage_unit": (
                            self.lifecycle_scales.daily_space_rate_per_storage_unit
                        ),
                        "maintenance_interval_wear": (
                            self.lifecycle_scales.maintenance_interval_wear
                        ),
                        "maintenance_due_window_wear": (
                            self.lifecycle_scales.maintenance_due_window_wear
                        ),
                        "maintenance_hazard_reduction": (
                            self.lifecycle_scales.maintenance_hazard_reduction
                        ),
                        "repair_condition_restore_fraction": (
                            self.lifecycle_scales.repair_condition_restore_fraction
                        ),
                        "repair_cash_per_replacement_condition_loss": (
                            self.lifecycle_scales.repair_cash_per_replacement_condition_loss
                        ),
                        "repair_minimum_cash": (
                            self.lifecycle_scales.repair_minimum_cash
                        ),
                        "customer_damage_liability_share": (
                            self.lifecycle_scales.customer_damage_liability_share
                        ),
                        "customer_damage_minimum_deduction": (
                            self.lifecycle_scales.customer_damage_minimum_deduction
                        ),
                        "maintenance_cash_rate_by_severity_tier": dict(
                            self.lifecycle_scales.maintenance_cash_rate_by_severity_tier
                        ),
                        "maintenance_duration_days_by_severity_tier": dict(
                            self.lifecycle_scales.maintenance_duration_days_by_severity_tier
                        ),
                        "new_asset_residual_rate": (
                            self.lifecycle_scales.new_asset_residual_rate
                        ),
                        "residual_wear_exponent": (
                            self.lifecycle_scales.residual_wear_exponent
                        ),
                    }
                }
                if self.config.lifecycle_economics_enabled
                and self.lifecycle_scales is not None
                else {}
            ),
            **(
                {
                    "retail_daily_holding_cost": (
                        f"{self.config.retail_holding_rate} times positive retail inventory "
                        "at average unit cost"
                    )
                }
                if self.config.retail_enabled
                else {}
            ),
            "rental_daily_holding_cost": (
                (
                    "every owned rental asset pays replacement_cost * annual capital rate / "
                    "365 from the opening snapshot; supplier-trial assets pay no capital charge; "
                    if self._v37_supplier_trial()
                    else "every unsold rental asset pays replacement_cost * annual capital rate / 365 from the opening snapshot; "
                )
                + "only on-site non-rented assets additionally "
                "pay storage burden units * daily space rate; a same-day sale does not "
                "avoid the opening charge"
                if self.config.lifecycle_economics_enabled
                else f"{self.config.rental_holding_rate} times the opening unit-cost basis "
                "of rental assets that are not retired or sold after morning processing; "
                "selling or retiring an asset later that day does not avoid that day's charge"
            ),
            "rental_idle_daily_holding_cost": (
                "superseded by the storage-unit space charge in rental_daily_holding_cost"
                if self.config.lifecycle_economics_enabled
                else f"{self.config.rental_idle_holding_rate} times the opening unit-cost "
                "basis of rental assets that are not rented, retired, or sold; this is "
                "charged in addition to the rental capital holding charge"
            ),
            "normal_return_cleaning_cost": (
                "normal returns create one offer/world/weather/duration-adjusted work-point "
                "quantity; the FIFO queue and cash charge use that same quantity"
                if self.config.lifecycle_economics_enabled
                else "normal returns incur a product-dependent cleaning charge, and longer "
                "rentals may cost more to clean"
            ),
            "supplier_inventory": (
                "the approved catalog suppliers can fulfill unlimited packs of every "
                "listed exact offer"
            ),
            "cleaning_capacity": {
                "daily_work_points": self.rules.cleaning_capacity_points,
                "availability": (
                    "returned assets join a FIFO cleaning queue; work can span days and "
                    "the asset cannot be rented until its work is complete; this daily "
                    "capacity limits in-season reuse only"
                ),
            },
            "damaged_asset_repair_cost": (
                "the return's severity-derived condition loss determines one frozen repair "
                "cash amount; paid repair restores only the declared fraction of that loss"
                if self.config.lifecycle_economics_enabled
                else "max(10, 0.20 * asset unit cost)"
            ),
            "damage_deposit_deduction": (
                "the same condition-loss repair amount times customer-liability share, "
                "subject to the minimum and capped by the frozen contract deposit"
                if self.config.lifecycle_economics_enabled
                else "min(contract deposit, max(5, 0.10 * asset unit cost))"
            ),
            "purchase_timeline": (
                (
                    "prepaid on order day; each seller listing shows one concrete estimated "
                    "delivery time, but actual arrival may differ slightly; delivered at "
                    "the start of the actual arrival day; new orders whose estimated "
                    "delivery date exceeds decision_horizon_day are rejected, while an "
                    "already accepted supplier delay may arrive during runoff"
                )
                if self._v24_reservation_operations()
                else "prepaid on order day; each seller listing shows one concrete estimated "
                "delivery time, but actual arrival may differ slightly; delivered at "
                "the start of the actual arrival day; orders whose estimated delivery "
                "date falls after the season closes are not accepted"
            ),
            "normal_return_timeline": (
                "return and refund at the start of the due day; the asset enters the "
                "finite-work FIFO cleaning queue, and its cleaning charge is paid when "
                "the work completes"
            ),
            "advance_reservations": {
                "enabled": bool(self.config.reservations_enabled),
                "request_deadline": "end of the inquiry's creation day",
                "one_time_quote": (
                    "submit one or two exact offer/profile/price options per required line; "
                    "optional lines submit zero to two quote options plus one explicit omit; "
                    "the environment forms at most 24 complete plans and returns at most "
                    "three independently acceptable plans; one auto-confirms, while multiple "
                    "plans await manager choice and default to the customer preferred plan "
                    "at day end"
                    if factorized_plan_choice
                    else
                    "explicitly submit one or two complete contract options or decline the "
                    "whole inquiry; every option resolves every line, required lines quote "
                    "one exact offer/profile/price, optional lines explicitly quote or omit, "
                    "and no implicit Cartesian combinations are created; a willing customer "
                    "reports one noisy assessment for every unselected contract"
                    if complete_contract_choice
                    else
                    "explicitly submit one or two binding exact offer/profile options "
                    "with independent cent-exact per-asset daily prices for every required "
                    "and optional line, or decline the whole inquiry; the customer compares "
                    "the complete store bundle with complete outside alternatives, required "
                    "lines are atomic, optional lines may be skipped, and rejected bundles "
                    "reveal no failed line or hidden outside selection"
                    if request_bundle_choice
                    else "explicitly submit one or two binding exact offer/profile options "
                    "with independent cent-exact per-asset daily prices for every required "
                    "and optional line, or decline the whole inquiry; the customer chooses "
                    "once and rejected bundles reveal no failed line or hidden selection"
                    if self.config.reservation_contract_mode
                    == "adaptive_pairwise_exact_quotes_v1"
                    else "explicitly select one eligible public promise profile and quote "
                    "one cent-exact per-asset daily price for every required and optional "
                    "line, or decline the whole inquiry; walk-in posted prices never "
                    "fill contract lines; customer feedback has no counteroffer or "
                    "rejection reason"
                    if self.config.reservation_contract_mode
                    == "promise_profiles_v1"
                    else "explicitly quote one cent-exact per-unit daily price for every "
                    "required and optional line, or decline the whole inquiry; walk-in "
                    "posted prices never fill contract lines; customer feedback has no "
                    "counteroffer or rejection reason"
                ),
                "inquiry_candidates": (
                    "one normalized request-level quote workspace containing complete "
                    "request-line frozen Profile terms plus deduplicated static Profile "
                    "and exact-offer definitions; dynamic operations remain separate"
                    if factorized_plan_choice and self._v25_quote_workspace()
                    else
                    "compact public promise-profile families per line; exact public members "
                    "are paginated and may be used as factorized line options"
                    if factorized_plan_choice
                    else
                    "compact public promise-profile families per line; exact public members "
                    "are paginated and may be assembled into one or two explicit complete "
                    "contract options"
                    if complete_contract_choice
                    else
                    "compact public promise-profile families per line; exact public "
                    "members are paginated and the manager may quote one or two as "
                    "mutually exclusive binding options"
                    if self.config.reservation_contract_mode
                    == "adaptive_pairwise_exact_quotes_v1"
                    else "compact public promise-profile families per line; exact public "
                    "members are paginated, and the quoted price applies to every "
                    "surviving authorized member of the selected profile"
                    if self.config.reservation_contract_mode
                    == "promise_profiles_v1"
                    else "one to three ordered exact offers per line; the same quoted "
                    "price applies to every surviving authorized offer"
                ),
                "locked_terms": (
                    "only after manager or automatic confirmation: the chosen customer-authorized "
                    "plan's exact offers, profiles, quantities, dates, own rents, and frozen deposits"
                    if factorized_plan_choice
                    else
                    "only after customer acceptance: the selected complete contract, its "
                    "quoted/omitted optional lines, exact offers, profiles, quantities, "
                    "dates, rents, and frozen deposits"
                    if complete_contract_choice
                    else
                    "only after customer acceptance: customer-selected exact offer, "
                    "selected profile/version, computed asset quantity, pickup/return "
                    "dates, exact-option rent, and frozen profile deposit"
                    if self.config.reservation_contract_mode
                    == "adaptive_pairwise_exact_quotes_v1"
                    else "only after customer acceptance: selected profile/version, computed "
                    "asset quantity, complete authorized exact-offer set and order, "
                    "pickup/return dates, quoted rent, and frozen profile deposit"
                    if self.config.reservation_contract_mode
                    == "promise_profiles_v1"
                    else "only after customer acceptance: primary exact offer, authorized "
                    "exact-offer set, quantity, pickup/return dates, quoted rent, "
                    "and primary security deposit"
                ),
                "prepayment": (
                    f"{self.config.reservation_prepayment_rate:.0%} of locked rent; "
                    "fully refundable and recorded as unearned revenue until pickup"
                ),
                    "asset_assignment": (
                        "the selected exact offer is fixed at confirmation and the physical "
                        "asset is assigned at pickup; there is no staging or automatic SKU "
                        "pooling, and a replacement requires explicit risk fallback"
                        if self.config.reservation_contract_mode
                        in {
                            "adaptive_pairwise_exact_quotes_v1",
                            "complete_contract_options_v1",
                            "factorized_line_options_plan_set_v1",
                        }
                    else "an authorized exact offer may be staged before pickup without "
                    "binding assets or reserving capacity; pickup tries staged then the "
                    "remaining frozen order, and one exact offer supplies the whole line; "
                    "confirmed commitments may exceed internally available capacity and "
                    "acceptance returns no feasibility label"
                    if self.config.reservation_contract_mode
                    == "promise_profiles_v1"
                    else "delayed until pickup morning; each whole line uses the first "
                    "publicly authorized exact offer with enough units in primary-first "
                    "order; confirmed commitments may exceed internally available "
                    "capacity and acceptance returns no feasibility label"
                ),
                "pickup_priority": "confirmed reservations before same-day walk-ins",
                **({
                    "final_pickup_cutoff_order": (
                        "at close, finalize due reservation pickups (including explicitly "
                        "selected recovery plans/line fallbacks) before simulating today's "
                        "walk-ins; recovery selection itself does not bind assets; future "
                        "commitments remain the manager's capacity-planning responsibility"
                    ),
                } if self._v30_pickup_priority() else {}),
                **(
                    {
                        "pickup_horizon": (
                            "initial and rescheduled pickup dates cannot exceed the "
                            "manager decision horizon; returns and settlement may "
                            "continue in runoff"
                        )
                    }
                    if self.config.runoff_days > 0
                    else {}
                ),
                "calendar": (
                    (
                        "view_rental_capacity reports raw current assets, explicit pickup/"
                        "occupied-through/return intervals, walk-in policy and occupancy, "
                        "public delivery timing, and current service jobs only; it never "
                        "returns net availability, feasibility, shortage, coverage, or a recommendation"
                        if self._v24_reservation_operations()
                        else "view_rental_capacity reports raw current assets, binding intervals, "
                        "public delivery timing, and current service jobs only; it never "
                        "returns net availability, feasibility, shortage, coverage, or a recommendation"
                    )
                    if capacity_surface_v22
                    else "view_reservation_calendar reports raw facts only and never returns "
                    "feasibility, shortage, coverage, or a recommendation"
                ),
                "weather_changes": (
                    (
                        "advance-request customers use the request-day public forecast, "
                        "while walk-ins use arrival-day observed weather; each potential "
                        "customer has one fixed channel opportunity and does not reappear "
                        "through the other channel. Confirmed customers evaluate each newly "
                        "observed forecast regime once. Only adversity strictly worse than "
                        "every previously evaluated regime in the current commitment version "
                        "can trigger a weather response; keeping the original booking also "
                        "records that regime as evaluated. A customer weather reschedule may "
                        "propose a legal pickup offset from -10 through -3 or +3 through +10 "
                        "days, preserving duration and using only public forecast, calendar, "
                        "date distance, and private customer preferences. A post-pickup "
                        "extension remains +1..3 days; its frozen fulfilled-line rent delta "
                        "is disclosed and collected immediately on acceptance"
                        if self._v42_weather_information_coherence()
                        else (
                            "customers see the same forecast surface as the manager but "
                            "have private forecast trust and schedule flexibility; they "
                            "may cancel, request a +1..3 day reschedule, or request a "
                            "+1..3 day extension whose frozen fulfilled-line rent delta "
                            "is disclosed and collected immediately on acceptance"
                            if self._v27_extension_cash_audit()
                            else "customers see the same forecast surface as the manager but have "
                            "private forecast trust and schedule flexibility; they may cancel, "
                            "request a +1..3 day reschedule, or request a +1..3 day extension"
                        )
                    )
                    if self.config.reservation_disruptions_enabled
                    else "disabled"
                ),
                **(
                    {
                        "non_weather_cancellation": (
                            "a confirmed pre-pickup reservation may be cancelled by one "
                            "independent non-weather customer event. Its hidden absolute event "
                            "day is sampled once when the commitment forms; rescheduling does "
                            "not redraw it, and it is separate from forecast reconsideration"
                        )
                    }
                    if self._v42_weather_information_coherence()
                    else {}
                ),
                "risk_recovery": (
                    (
                        "a newly exposed exact-offer risk remains actionable through one full "
                        "manager decision round; one inquiry returns dormant customer-authorized "
                        "complete plans and same-Profile line fallbacks separately, but selects "
                        "nothing. The manager explicitly chooses one exclusive recovery path. "
                        "A complete-plan recovery never mixes plans and protects each line at "
                        "the lower of its two customer-authorized amounts"
                        if self._v24_reservation_operations()
                        else "a newly exposed exact-offer risk remains actionable through one full "
                        "manager decision round; substitution requires inquiry and explicit "
                        "selection at unchanged price and guarantee, and every hidden candidate "
                        "was authorized by replaying the complete locked bundle"
                    )
                    if request_bundle_choice
                    else "a newly exposed exact-offer risk remains actionable through one full "
                    "manager decision round; substitution requires inquiry and explicit "
                    "selection at unchanged price and guarantee"
                    if self.config.reservation_contract_mode
                    == "adaptive_pairwise_exact_quotes_v1"
                    else "a newly exposed pickup risk remains actionable through one full "
                    "manager decision round; booking-authorized substitution is automatic, "
                    "while undisclosed latent alternatives require inquiry and explicit selection"
                ),
                **(
                    {
                        "walk_in_capacity_policy": (
                            (
                                "optional exact-offer occupied-asset caps over explicit "
                                "dates; set is a direct daily upsert and must not be "
                                "preceded by overlapping clear; batches are atomic and "
                                "order-independent; clear ignores a historical prefix "
                                "without rewriting it; new walk-in contracts must satisfy "
                                "every occupied day, reservations are excluded, existing "
                                "contracts are never cancelled, default is unlimited, and "
                                "no cleaning buffer is inferred"
                                if self._v28_capacity_summary_audit()
                                else "optional exact-offer occupied-asset caps over explicit dates; "
                                "new walk-in contracts must satisfy every occupied day, reservations "
                                "are excluded, existing contracts are never cancelled, default is "
                                "unlimited, and no cleaning buffer is inferred"
                            )
                        )
                    }
                    if self._v24_reservation_operations()
                    else {}
                ),
                "failure_compensation": (
                    {
                        "manager_cancellation": (
                            "max(0.5 * current locked rent * max(0.25, "
                            "1 - notice_days/7), (10 + 2 * required "
                            "service-unit-days) * max(0.35, 1 - notice_days/7))"
                        ),
                        "pickup_failure": (
                            "max(current locked rent, 15 + 3 * required "
                            "service-unit-days)"
                        ),
                        "prepayment_refund": "separate from compensation",
                        "rounding": "cent half-up after max",
                    }
                    if self._v29_reservation_accountability()
                    else f"base {self.config.reservation_failure_compensation_rate:.0%} of "
                    "locked rent at zero notice, declining with notice but never below "
                    "one quarter of that amount"
                ),
                **(
                    {
                        "manager_cancellation": (
                            "cancel_reservation_commitment is available for any "
                            "confirmed pre-pickup commitment; no risk alert is required"
                        ),
                        "reservation_service_reputation": (
                            {
                                "enabled": True,
                                "formal_default_enabled": False,
                                "policy": self._reservation_reputation_policy(),
                                "request_snapshot": (
                                    "frozen at inquiry creation and subtracted "
                                    "once from every complete store plan"
                                ),
                                "inquiry_generation_affected": False,
                            }
                            if self.config.reservation_service_reputation_enabled
                            else {"enabled": False}
                        ),
                    }
                    if self._v29_reservation_accountability()
                    else {}
                ),
                "bundle_pickup": (
                    "all required durable lines assign atomically; unavailable optional "
                    f"lines receive a {self.config.optional_line_missing_credit_rate:.0%} "
                    "locked-line rent credit"
                ),
                **(
                    {
                        "contract_mode": self.config.reservation_contract_mode,
                        "profile_deposit_policy": (
                            self.config.reservation_profile_deposit_policy
                        ),
                        "asset_quantity": (
                            "the environment computes ceil(service_units / "
                            "capacity_per_asset) before locking rent and deposit"
                        ),
                        "line_mixing": False,
                        "profile_option_surface": (
                            "view_reservation_quote_workspace returns normalized frozen "
                            "quote choices without dynamic operations or feasibility"
                            if self._v25_quote_workspace()
                            else "view_reservation_options returns raw paginated facts only"
                        ),
                        **(
                            {
                                "required_line_option_count": [1, 2],
                                "optional_line_quote_option_count": [0, 2],
                                "maximum_internal_plan_count": 24,
                                "maximum_returned_authorized_plan_count": 3,
                                "authorization_is_exhaustive": False,
                                "customer_choice_scope": (
                                    "every complete plan independently against the same "
                                    "outside plan and abandonment"
                                ),
                                "manager_confirmation": (
                                    "any returned plan; customer preferred plan auto-confirms "
                                    "at close if unanswered"
                                ),
                                "cash_and_calendar_timing": "confirmed_plan_only",
                                **(
                                    {
                                        "customer_authorization_reads_inventory_or_operations": False,
                                        "authorized_plan_is_binding": True,
                                        "manager_bears_fulfillment_capacity_and_opportunity_cost": True,
                                        "same_exact_profile_pair_at_multiple_prices": False,
                                        "quote_batch_validation": (
                                            "atomic_all_invalid_items"
                                        ),
                                        "reservation_record_page_limit": 10,
                                    }
                                    if capacity_surface_v22
                                    else {}
                                ),
                                "decline_feedback_scope": "empty_authorized_set_only",
                                **(
                                    {
                                        "reported_concern_tags": [
                                            "price_value",
                                            "equipment_configuration",
                                        ],
                                        "reported_concern_tags_can_cooccur": True,
                                        "residual_concern": "other_or_unclear",
                                        "residual_concern_can_cooccur": False,
                                        "concern_tag_absence_exonerates_axis": False,
                                    }
                                    if factorized_concern_tags
                                    else {
                                        "reported_decline_categories": [
                                            "price_value_concern",
                                            "equipment_fit_concern",
                                            "multiple_or_unclear_concerns",
                                        ]
                                    }
                                ),
                                **(
                                    {
                                        "equipment_feedback_line_mention": (
                                            "zero or one noisy subjective "
                                            "customer_mentioned_line_id; only with "
                                            "equipment_configuration"
                                            if factorized_concern_tags
                                            else "zero or one noisy subjective "
                                            "customer_mentioned_line_id"
                                        ),
                                        "line_mention_is_failed_line_truth": False,
                                        "unmentioned_lines_are_exonerated": False,
                                        "line_mention_guarantees_repair": False,
                                    }
                                    if noisy_line_mentions
                                    else {}
                                ),
                                "outside_resolution_disclosed": False,
                            }
                            if factorized_plan_choice
                            else {}
                        ),
                        **(
                            {
                                "complete_contract_option_count": [1, 2],
                                "implicit_cartesian_product": False,
                                "optional_line_disposition": ["quote", "omit"],
                                "customer_choice_scope": (
                                    "one or two explicit complete store contracts"
                                ),
                                "reported_assessments": [
                                    "price_value_concern",
                                    "contract_configuration_concern",
                                    "multiple_or_unclear_concerns",
                                    "no_specific_concern_other_contract_preferred",
                                ],
                                "assessment_semantics": (
                                    "partial noisy customer report; not a true reason, "
                                    "threshold, feasibility verdict, or action guarantee"
                                ),
                                "outside_resolution_disclosed": False,
                            }
                            if complete_contract_choice
                            else {}
                        ),
                        **(
                            {
                                "line_option_count": [1, 2],
                                **(
                                    {
                                        "customer_choice_scope": (
                                            "request-level complete bundle; required "
                                            "lines atomic; optional lines skippable"
                                        )
                                    }
                                    if request_bundle_choice
                                    else {}
                                ),
                                "choice_history_surface": (
                                    "view_reservation quote_history returns immutable "
                                    "raw submitted options, aggregate outcomes, and "
                                    "accepted selections"
                                    + (
                                        "; it supports exact request/commitment IDs and "
                                        "public filters, applies filters before pagination, "
                                        "returns complete bundles, and defaults to newest first"
                                        if int(
                                            self.dataset.metadata.get(
                                                "tool_interface_version", 0
                                            )
                                        )
                                        >= 13
                                        else ""
                                    )
                                ),
                            }
                            if self.config.reservation_contract_mode
                            == "adaptive_pairwise_exact_quotes_v1"
                            else {}
                        ),
                    }
                    if self.config.reservation_contract_mode
                    in {
                        "promise_profiles_v1",
                        "adaptive_pairwise_exact_quotes_v1",
                        "complete_contract_options_v1",
                        "factorized_line_options_plan_set_v1",
                    }
                    else {}
                ),
            },
            "preventive_maintenance": {
                "enabled": bool(self.config.preventive_maintenance_enabled),
                **(
                    {
                        "trigger": "combined-wear exposure or damaged condition",
                        "overdue_policy": (
                            "ordinary wear overdue raises damage risk but does not force a "
                            "hold; only condition below 0.65 causes maintenance_hold"
                        ),
                        "effect": (
                            "completion moves the wear anchor and provides protection that "
                            "decays with new wear; cumulative wear is never reset"
                        ),
                    }
                    if self.config.lifecycle_economics_enabled
                    else {
                        "inspection_interval_days": int(self.config.maintenance_interval_days),
                        "scheduling_window_days_before_due": int(
                            self.config.maintenance_due_window_days
                        ),
                        "overdue_policy": "maintenance_hold removes the asset from rental capacity",
                        "duration_days": int(self.config.maintenance_duration_days),
                        "cost": f"max(2, {self.config.maintenance_cost_rate} * asset unit cost)",
                    }
                ),
            },
            "automatic_runoff": {
                "decision_horizon_day": self.config.max_days,
                **(
                    {
                        "runoff_end_day": self.config.max_days
                        + int(self.config.runoff_days),
                        "final_scoring_day": self.config.max_days
                        + int(self.config.runoff_days),
                    }
                    if self._v24_reservation_operations()
                    else {}
                ),
                "runoff_days": int(self.config.runoff_days),
                "policy": (
                    (
                        "no new demand, new pickups, or manager/model calls; returns, "
                        "cleaning, maintenance, previously accepted supplier deliveries, "
                        "refunds, holding costs, and settlement continue"
                        if self._v24_reservation_operations()
                        else "no new demand, new pickups, or manager/model calls; returns, "
                        "cleaning, maintenance, refunds, holding costs, and settlement "
                        "continue"
                    )
                    if self.config.runoff_days > 0
                    else (
                        "no new demand or manager/model calls; returns, cleaning, "
                        "maintenance, refunds, holding costs, and settlement continue"
                    )
                ),
            },
            "customer_protection_credit": {
                "limit": round(self.config.emergency_credit_limit, 2),
                "fee_rate": float(self.config.emergency_credit_fee_rate),
                "scope": "required customer refunds only; liability reduces scored value",
            },
            "repair_timeline": (
                "repair is prepaid when requested; asset becomes available at the start of day +2"
            ),
            "season_end_business_value": (
                (
                    "cash - refundable deposit liability + the fixed liquidation dealer's "
                    "per-unit quotes for remaining retail inventory + the dealer's season-close "
                    "values for unsold rental assets; each retail unit quote is 50% of its "
                    "current average acquisition unit cost rounded to the nearest cent before "
                    "multiplying by units; "
                )
                if self.config.retail_enabled
                else (
                    "cash - refundable deposit liability + the fixed liquidation dealer's "
                    "season-close values for unsold rental assets; "
                )
            )
            + (
                "supplier orders still in transit have no season-close value. Post-season "
                "time for scheduled cleaning is unlimited: cleaning assets receive their "
                "healthy quote less the full unpaid cleaning cost, with no extra queue or "
                "daily-capacity discount. Rental nominal dealer value uses only replacement "
                "cost, condition, combined wear and the global resale curve before rolling "
                "market-depth adjustment"
                if self.config.lifecycle_economics_enabled
                else "supplier orders still in transit have no season-close value. Post-season "
                "time for scheduled cleaning is unlimited: cleaning assets receive their "
                "healthy quote less the full unpaid cleaning cost, with no extra queue or "
                "daily-capacity discount"
            ),
            "season_end_rental_asset_valuation": {
                "available": "the dealer's season-close quote for that asset",
                "cleaning": (
                    "the dealer's healthy quote less the full unpaid scheduled cleaning "
                    "cost; post-season time is unlimited, so in-season queue capacity "
                    "causes no further discount"
                ),
                "repairing": (
                    "the dealer's quote after the prepaid repair is completed"
                ),
                "damaged": "the dealer's quote for the asset in its current damaged state",
                "rented": (
                    "zero only in an instantaneous liquidation diagnostic; every "
                    "admitted rental returns by runoff_end_day before final scoring"
                    if self._v24_reservation_operations()
                    else "zero"
                ),
                "retired": "zero",
                "sold": "zero; sale proceeds are already cash",
            },
            "liquidation_dealer": {
                **(
                    {
                        "retail_inventory": (
                            "delivered on-hand consumer units may be sold at a fixed per-unit quote: "
                            "50% of current average acquisition unit cost, rounded to the nearest "
                            "cent before multiplying by units; in-transit orders cannot be sold"
                        )
                    }
                    if self.config.retail_enabled
                    else {}
                ),
                "rental_asset_sellable_statuses": sorted(
                    SECONDHAND_SELLABLE_STATUSES
                ),
                "automatic_insolvency_sale": False,
                "market_depth": self._public_liquidation_depth_rule(),
            },
            "insolvency_rule": (
                "the shop closes immediately and is considered bankrupt if a required "
                "charge or customer refund cannot be paid"
            ),
        }
        return self.actor_assortment_projection(rules)

    def reset(self) -> WorldState:
        assets = [
            RentalAsset(
                asset_id=row["asset_id"],
                offer_id=row["offer_id"],
                service_line=row["service_line"],
                unit_cost=float(row["unit_cost"]),
                used_life_days=int(row["used_life_days"]),
                service_life_limit_days=int(row["service_life_limit_days"]),
                damage_risk_band=row["damage_risk_band"],
                used_rental_cycles=int(row.get("used_rental_cycles", 0)),
                service_life_limit_cycles=(
                    None
                    if row.get("service_life_limit_cycles") is None
                    else int(row["service_life_limit_cycles"])
                ),
                status=str(row.get("status", "available")),
                ready_day=(
                    None if row.get("ready_day") is None else int(row["ready_day"])
                ),
            )
            for row in self.dataset.initial_assets
        ]
        if self.config.lifecycle_economics_enabled:
            for asset in assets:
                projection = self._lifecycle_projection(asset.offer_id)
                prior_day_limit = max(1, asset.service_life_limit_days)
                prior_cycle_limit = max(
                    1,
                    asset.service_life_limit_cycles
                    if asset.service_life_limit_cycles is not None
                    else prior_day_limit,
                )
                used_day_ratio = asset.used_life_days / prior_day_limit
                used_cycle_ratio = asset.used_rental_cycles / prior_cycle_limit
                asset.service_life_limit_days = int(
                    projection["service_life_limit_days"]
                )
                asset.service_life_limit_cycles = int(
                    projection["service_life_limit_cycles"]
                )
                asset.used_life_days = round(
                    used_day_ratio * asset.service_life_limit_days
                )
                asset.used_rental_cycles = round(
                    used_cycle_ratio * asset.service_life_limit_cycles
                )
                asset.next_inspection_due_day = None
                self._initialize_lifecycle_asset_state(asset)
                assert self.lifecycle_scales is not None
                phase_fraction = stable_words(
                    f"lifecycle-maintenance-phase:{asset.asset_id}", count=1
                )[0] / float(2**32 - 1)
                phase_wear = phase_fraction * (
                    self.lifecycle_scales.maintenance_interval_wear
                    + self.lifecycle_scales.maintenance_due_window_wear
                )
                asset.maintenance_wear_anchor = max(
                    0.0, self._combined_wear(asset) - phase_wear
                )
        elif self.config.preventive_maintenance_enabled:
            interval = int(self.config.maintenance_interval_days)
            for asset in assets:
                asset.next_inspection_due_day = max(
                    7,
                    1 + stable_words(asset.asset_id, count=1)[0] % interval,
                )
        retail = {
            row["offer_id"]: {
                "quantity": int(row["quantity"]),
                "avg_unit_cost": float(row["unit_cost"]),
            }
            for row in self.dataset.initial_retail
        }
        prices: dict[str, float] = {}
        initial_price_overrides = {
            str(row["offer_id"]): float(row["price"])
            for row in self.dataset.initial_prices
        }
        deposits: dict[str, float] = {}
        rental_lines: dict[str, bool] = {}
        for offer in self.dataset.offers:
            if offer.offer_id in initial_price_overrides:
                prices[offer.offer_id] = round(
                    initial_price_overrides[offer.offer_id], 2
                )
            elif self.channel_demand_model is None:
                reference_price = (
                    offer.reference_retail_price
                    if offer.asset_kind == "consumable"
                    and offer.reference_retail_price is not None
                    else self.reference_daily_rates[offer.offer_id]
                    if offer.asset_kind == "durable"
                    else offer.unit_cost
                )
                prices[offer.offer_id] = round(float(reference_price), 2)
            if offer.asset_kind == "durable":
                rental_lines[offer.offer_id] = True
                deposits[offer.offer_id] = round(
                    max(20.0, offer.unit_cost * DEPOSIT_RATIOS[offer.service_line]), 2
                )
        state = WorldState(
            day=1,
            max_days=self.config.max_days + int(self.config.runoff_days),
            cash=round(self.config.initial_cash, 2),
            deposit_liability=0.0,
            unearned_revenue_liability=0.0,
            prices=prices,
            deposits=deposits,
            rental_service_open=dict(sorted(rental_lines.items())),
            assets=assets,
            retail_inventory=retail,
            current_day_ledger=_empty_ledger(1),
            decision_horizon_day=self.config.max_days,
            runoff_end_day=self.config.max_days + int(self.config.runoff_days),
            next_asset_id=1 + max(
                (int(asset.asset_id.removeprefix("asset-")) for asset in assets
                 if asset.asset_id.startswith("asset-")
                 and asset.asset_id.removeprefix("asset-").isdigit()),
                default=0,
            ),
            ever_carried_offer_ids=sorted(
                {
                    *(asset.offer_id for asset in assets),
                    *(offer_id for offer_id, row in retail.items() if int(row["quantity"]) > 0),
                }
            ),
        )
        self._initialize_warehouse(state)
        self._initialize_durable_acquisition_budget(state)
        self._initialize_supplier_trials(state)
        self._assert_reservation_contingencies(state)
        if self.config.liquidation_depth_enabled:
            rental_nominal = sum(
                float(quote["offer_price"])
                for asset in state.assets
                if (quote := self.secondhand_quote(asset)) is not None
            )
            retail_nominal = sum(
                self.retail_liquidation_proceeds(
                    float(row["avg_unit_cost"]), int(row["quantity"])
                )
                for row in state.retail_inventory.values()
            )
            fraction = float(self.config.liquidation_depth_capacity_fraction)
            state.liquidation_depth_capacity = {
                "rental_asset": round(max(1.0, fraction * rental_nominal), 2),
                **(
                    {"retail_item": round(max(1.0, fraction * retail_nominal), 2)}
                    if self.config.retail_enabled
                    else {}
                ),
            }
        self._capture_opening_rental_holding_basis(state)
        self._prepare_reservation_day(state)
        self._seal_opening_flows(state)
        self._assert_assortment_state(state)
        self._assert_durable_acquisition_budget(state)
        self._assert_reservation_contingencies(state)
        return state

    def _source_weather_day(self, benchmark_day: int) -> int:
        return self.config.weather_start_day + int(benchmark_day) - 1

    def _weather(self, day: int) -> str:
        source_day = self._source_weather_day(day)
        try:
            return self.dataset.weather[source_day]
        except KeyError as error:
            raise ValueError(
                f"runtime weather is missing source day {source_day} for benchmark day {day}"
            ) from error

    def _demand_weather_regime(self, day: int) -> str:
        """Return the hidden demand regime without exposing thermal labels."""

        sky = self._weather(day)
        if sky in {"rainy", "snowy"}:
            return sky
        _, _, high = self._temperatures(day)
        if high >= 32.0:
            return "hot"
        if high <= 8.0:
            return "cold"
        return sky

    def _calendar(self, day: int) -> dict[str, Any]:
        source_day = self._source_weather_day(day)
        try:
            return self.dataset.weather_calendar[source_day]
        except KeyError as error:
            raise ValueError(
                f"runtime calendar is missing source day {source_day} for benchmark day {day}"
            ) from error

    def _public_calendar(self, day: int) -> dict[str, Any]:
        calendar = self._calendar(day)
        return {
            key: calendar.get(key)
            for key in (
                "date",
                "month_day",
                "weekday",
                "is_weekend",
                "day_type",
                "holiday_name",
                "temp_mean_c",
                "night_low_c",
                "day_high_c",
            )
            if calendar.get(key) is not None
        }

    def actor_assortment_projection(self, value: Any) -> Any:
        """Apply every versioned projection at the actor boundary."""

        projected = value
        if not self.config.retail_enabled:
            dropped = object()

            def project_assortment(item: Any) -> Any:
                if isinstance(item, list):
                    rows = [project_assortment(row) for row in item]
                    return [row for row in rows if row is not dropped]
                if not isinstance(item, dict):
                    return item
                if (
                    item.get("kind") == "retail_item"
                    or str(item.get("service_key") or "").startswith("retail:")
                    or item.get("rental_or_retail") == "retail"
                ):
                    return dropped
                result = {}
                for key, child in item.items():
                    if "retail" in str(key).lower() or key == "rental_or_retail":
                        continue
                    child_projection = project_assortment(child)
                    if child_projection is not dropped:
                        result[key] = child_projection
                return result

            projected = project_assortment(projected)
            if projected is dropped:
                projected = {}

        if self._v41_actor_interface_slimming():
            def is_integrity_key(key: Any) -> bool:
                normalized = str(key).strip().lower()
                return normalized in {"hash", "sha256", "fingerprint"} or any(
                    normalized.endswith(suffix)
                    for suffix in ("_hash", "_sha256", "_fingerprint")
                )

            def project_integrity(item: Any) -> Any:
                if isinstance(item, list):
                    return [project_integrity(row) for row in item]
                if not isinstance(item, dict):
                    return item
                return {
                    key: project_integrity(child)
                    for key, child in item.items()
                    if not is_integrity_key(key)
                }

            projected = project_integrity(projected)
        return projected

    def _assert_assortment_state(self, state: WorldState) -> None:
        if self.config.retail_enabled:
            return
        if state.retail_inventory:
            raise AssertionError("rental-only state contains disabled inventory")
        disabled_totals = (
            state.cumulative_retail_revenue,
            state.cumulative_retail_holding_cost,
            state.cumulative_retail_liquidation_proceeds,
            state.cumulative_retail_liquidation_book_cost,
            state.cumulative_retail_liquidation_loss,
            state.cumulative_retail_units_liquidated,
        )
        if any(abs(float(value)) > 1e-9 for value in disabled_totals):
            raise AssertionError("rental-only state contains disabled financial activity")
        if any(
            str(item.get("offer_id")) not in self.offers
            for order in state.orders
            for item in order.items
        ):
            raise AssertionError("rental-only state contains a disabled order line")

    def observation(self, state: WorldState, *, consume_inbox: bool) -> dict[str, Any]:
        self._assert_assortment_state(state)
        self._assert_warehouse_snapshot(state)
        self._assert_durable_acquisition_budget(state)
        due_tomorrow = sum(
            contract.deposit
            for contract in state.contracts
            if contract.status == "active" and contract.return_day == state.day + 1
        )
        due_7d = sum(
            contract.deposit
            for contract in state.contracts
            if contract.status == "active" and state.day < contract.return_day <= state.day + 7
        )
        statuses = Counter(asset.status for asset in state.assets)
        reservation_quote_request_count = sum(
            request.status == "pending_quote"
            and request.created_day == state.day
            for request in state.reservation_requests
        )
        reservation_plan_confirmation_count = sum(
            request.status == "awaiting_manager_confirmation"
            and request.quote_day == state.day
            for request in state.reservation_requests
        )
        reservation_change_count = sum(
            change.status == "pending" and change.created_day == state.day
            for change in state.reservation_change_requests
        )
        reservation_risk_alert_count = sum(
            commitment.status == "at_risk"
            for commitment in state.reservation_commitments
        )
        result = {
            "day": state.day,
            **({"warehouse_capacity": self._warehouse_summary(state)} if self._v31_warehouse_capacity() else {}),
            **(
                {
                    (
                        "durable_capital"
                        if self._v40_revolving_durable_capital()
                        else "durable_acquisition_budget"
                    ): (
                        self._durable_acquisition_budget_summary(state)
                    )
                }
                if self._durable_capital_control_enabled()
                else {}
            ),
            **(
                {"supplier_trial": self._supplier_trial_summary(state)}
                if self._v37_supplier_trial()
                else {}
            ),
            "calendar": self._public_calendar(state.day),
            **(
                {}
                if self._v24_reservation_operations()
                else {"season_end_day": self.config.max_days}
            ),
            **(
                {
                    "decision_horizon_day": self.config.max_days,
                    "runoff_end_day": state.max_days,
                    "operating_phase": (
                        "decision_horizon"
                        if state.day <= self.config.max_days
                        else "automatic_runoff"
                    ),
                }
                if self.config.runoff_days > 0
                else {}
            ),
            **(
                {
                    "emergency_credit": {
                        "principal_used": round(
                            state.emergency_credit_principal, 2
                        ),
                        "liability": round(
                            state.emergency_credit_liability, 2
                        ),
                        "limit": round(self.config.emergency_credit_limit, 2),
                    }
                }
                if self.config.emergency_credit_limit > 0
                else {}
            ),
            **(
                {
                    "decision_days_remaining_including_today": max(
                        0, self.config.max_days - state.day + 1
                    ),
                    "final_scoring_day": state.max_days,
                }
                if self._v24_reservation_operations()
                else {
                    "season_days_remaining_including_today": max(
                        0, self.config.max_days - state.day + 1
                    )
                }
            ),
            "maximum_new_rental_duration_days": (
                max(0, state.max_days - state.day)
                if state.day <= self.config.max_days
                else 0
            ),
            "weather": self._weather(state.day),
            "cash": round(state.cash, 2),
            "deposit_liability": round(state.deposit_liability, 2),
            **(
                {
                    "unearned_revenue_liability": round(
                        state.unearned_revenue_liability, 2
                    ),
                    **(
                        {
                            "reservation_quote_request_count": (
                                reservation_quote_request_count
                            ),
                            # Compatibility alias retained for old agents. It
                            # has never counted confirmation/change/risk work.
                            "reservation_inbox_count": (
                                reservation_quote_request_count
                            ),
                            "reservation_inbox_count_scope": (
                                "pending_quote_requests_only; deprecated alias "
                                "of reservation_quote_request_count"
                            ),
                            "reservation_plan_confirmation_count": (
                                reservation_plan_confirmation_count
                            ),
                            "reservation_change_count": (
                                reservation_change_count
                            ),
                            "reservation_risk_alert_count": (
                                reservation_risk_alert_count
                            ),
                            "reservation_actions_due_today": (
                                reservation_quote_request_count
                                + reservation_plan_confirmation_count
                                + reservation_change_count
                                + reservation_risk_alert_count
                            ),
                        }
                        if self._v24_reservation_operations()
                        else {
                            "reservation_inbox_count": (
                                reservation_quote_request_count
                            ),
                            "reservation_plan_confirmation_count": (
                                reservation_plan_confirmation_count
                            ),
                        }
                    ),
                    "confirmed_future_reservations": sum(
                        commitment.status == "confirmed"
                        for commitment in state.reservation_commitments
                    ),
                    **(
                        {
                            "reservation_service_reputation": (
                                self._reservation_service_reputation(state)
                            )
                        }
                        if self._v29_reservation_accountability()
                        else {}
                    ),
                }
                if self.config.reservations_enabled
                else {}
            ),
            "maximum_refunds_due_next_morning": round(due_tomorrow, 2),
            "scheduled_cleaning_charges_due_next_morning": (
                self.scheduled_cleaning_charges_due_next_morning(state)
            ),
            "refunds_due_within_7_days": round(due_7d, 2),
            "asset_status_counts": dict(sorted(statuses.items())),
            "rental_offer_open": dict(sorted(state.rental_service_open.items())),
            "retail_units": int(sum(row["quantity"] for row in state.retail_inventory.values())),
            "active_rentals": sum(c.status == "active" for c in state.contracts),
            "orders_in_transit": sum(o.status == "in_transit" for o in state.orders),
            "cleaning_queue": {
                "assets_waiting": len(state.cleaning_queue),
                "remaining_work_points": round(
                    sum(job.remaining_work_points for job in state.cleaning_queue), 2
                ),
                "daily_capacity_points": self.rules.cleaning_capacity_points,
            },
            "today_morning_flows": dict(state.current_day_ledger.get("morning", {})),
            **({
                "today_intraday_flows": dict(state.current_day_ledger.get("intraday", {})),
            } if state.current_day_ledger.get("flow_timing_version") == 1 else {}),
            "cumulative_revenue": round(
                state.cumulative_rental_revenue + state.cumulative_retail_revenue, 2
            ),
            "cumulative_asset_sale_proceeds": round(state.cumulative_asset_sale_proceeds, 2),
            "cumulative_retail_liquidation_proceeds": round(
                state.cumulative_retail_liquidation_proceeds, 2
            ),
            "cumulative_retail_liquidation_loss": round(
                state.cumulative_retail_liquidation_loss, 2
            ),
            "cumulative_retail_units_liquidated": int(
                state.cumulative_retail_units_liquidated
            ),
            "cumulative_costs": {
                "purchases": round(state.cumulative_purchase_cost, 2),
                **(
                    {
                        "supplier_trial_handling_fees": round(
                            state.cumulative_supplier_trial_handling_fees, 2
                        ),
                        "supplier_trial_usage_fees": round(
                            state.cumulative_supplier_trial_usage_fees, 2
                        ),
                    }
                    if self._v37_supplier_trial()
                    else {}
                ),
                "cleaning": round(state.cumulative_cleaning_cost, 2),
                "repairs": round(state.cumulative_repair_cost, 2),
                **(
                    {
                        "maintenance": round(
                            state.cumulative_maintenance_cost, 2
                        )
                    }
                    if self.config.preventive_maintenance_enabled
                    else {}
                ),
                "retail_holding": round(state.cumulative_retail_holding_cost, 2),
                "rental_holding": round(state.cumulative_rental_holding_cost, 2),
                **(
                    {
                        "rental_capital_holding": round(
                            state.cumulative_rental_capital_holding_cost, 2
                        ),
                        "rental_space_holding": round(
                            state.cumulative_rental_space_holding_cost, 2
                        ),
                    }
                    if self.config.lifecycle_economics_enabled
                    and self.config.lifecycle_holding_enabled
                    else {}
                ),
                **(
                    {
                        "rental_idle_holding": round(
                            state.cumulative_rental_idle_holding_cost, 2
                        )
                    }
                    if self.config.rental_idle_holding_rate > 0
                    else {}
                ),
                "operating": round(state.cumulative_fees, 2),
                **(
                    {
                        "reservation_compensation": round(
                            state.cumulative_reservation_compensation, 2
                        ),
                        "optional_line_credits": round(
                            state.cumulative_optional_line_credits, 2
                        ),
                    }
                    if self.config.reservations_enabled
                    else {}
                ),
                **(
                    {"management_service": round(state.cumulative_model_cost, 6)}
                    if self.config.charge_model_usage_to_cash
                    else {}
                ),
            },
            "management_service_usage_estimate": round(
                state.cumulative_model_cost, 6
            ),
            "cumulative_transactions": state.cumulative_transactions,
            "inbox": list(state.inbox),
        }
        if state.daily_history:
            yesterday = state.daily_history[-1]
            result["yesterday"] = self._public_daily_report(yesterday)
            handoff_note = str(yesterday.get("handoff_note") or "").strip()
            if handoff_note:
                result["previous_day_handoff"] = handoff_note
        if consume_inbox:
            state.inbox.clear()
        return self.actor_assortment_projection(result)

    def _public_agent_actions(self, raw: dict[str, Any] | None) -> dict[str, Any]:
        actions = dict(raw or {})

        hidden = {
            "service_key",
            "product_type",
            "choice_group",
            "perceived_quality_tier",
            "decision_signature",
            "decision_attributes",
            "descriptive_attributes",
            "consumer_value_attributes",
            "consumer_value_evidence",
            "reservation_price_cny",
            "alternative_accessibility",
            "customer_segment",
            "episode_market_profile",
        }

        def sanitize(value: Any) -> Any:
            if isinstance(value, list):
                return [sanitize(row) for row in value]
            if not isinstance(value, dict):
                return value
            result = {
                key: sanitize(item)
                for key, item in value.items()
                if key not in hidden
            }
            offer_id = result.get("offer_id")
            if offer_id in self.offers:
                result.setdefault("name", self.offers[str(offer_id)].name)
            return result

        def net_changes(name: str, old_key: str, new_key: str) -> list[dict[str, Any]]:
            values = actions.get(name, {})
            if isinstance(values, list):
                rows = [dict(row) for row in values]
            else:
                rows = [dict(row) for _, row in sorted(dict(values).items())]
            return [
                sanitize(row)
                for row in rows
                if row.get(old_key) != row.get(new_key)
            ]

        rental_status_source = (
            "rental_product_status_changes"
            if "rental_product_status_changes" in actions
            else "rental_offer_status_changes"
        )

        return {
            "price_changes": net_changes(
                "price_changes", "old_price", "new_price"
            ),
            "purchase_orders": [
                sanitize(dict(row)) for row in actions.get("purchase_orders", [])
            ],
            "rental_offer_status_changes": net_changes(
                rental_status_source, "old_open", "new_open"
            ),
            "repairs": [sanitize(dict(row)) for row in actions.get("repairs", [])],
            "retirements": [
                sanitize(dict(row)) for row in actions.get("retirements", [])
            ],
            "asset_sales": [
                sanitize(dict(row)) for row in actions.get("asset_sales", [])
            ],
            "retail_liquidations": [
                sanitize(dict(row))
                for row in actions.get("retail_liquidations", [])
            ],
            **(
                {
                    "walk_in_capacity_limit_changes": [
                        sanitize(dict(row))
                        for row in actions.get(
                            "walk_in_capacity_limit_changes", []
                        )
                    ]
                }
                if self._v24_reservation_operations()
                else {}
            ),
        }

    def _public_daily_report(self, report: dict[str, Any]) -> dict[str, Any]:
        """Return the compact daily summary used in conversational context."""
        demand = report["demand"]
        financial = report["financial"]
        end = report["end"]
        return {
            "day": report["day"],
            **(
                {"calendar": self._public_calendar(int(report["day"]))}
                if report.get("calendar")
                else {}
            ),
            "weather": report["weather"],
            "visitor_count": int(sum(report["arrivals"].values())),
            "completed_transactions": int(
                demand["full_baskets"] + demand["partial_baskets"]
            ),
            "rental_revenue": financial["rental_revenue"],
            "retail_revenue": financial["retail_revenue"],
            **({
                "refunds": financial["refunds"],
                "refunds_by_phase": dict(financial["refunds_by_phase"]),
            } if "refunds_by_phase" in financial else {}),
            "total_revenue": round(
                financial["rental_revenue"] + financial["retail_revenue"], 2
            ),
            "asset_sale_proceeds": financial.get("asset_sale_proceeds", 0.0),
            "assets_sold": financial.get("assets_sold", 0),
            "retail_liquidation_proceeds": financial.get(
                "retail_liquidation_proceeds", 0.0
            ),
            "retail_liquidation_book_cost": financial.get(
                "retail_liquidation_book_cost", 0.0
            ),
            "retail_liquidation_loss": financial.get(
                "retail_liquidation_loss", 0.0
            ),
            "retail_units_liquidated": financial.get(
                "retail_units_liquidated", 0
            ),
            "requested_units": demand["requested_units"],
            "fulfilled_units": demand["fulfilled_units"],
            "unfulfilled_units": int(
                demand.get(
                    "unfulfilled_units",
                    int(demand["requested_units"]) - int(demand["fulfilled_units"]),
                )
            ),
            "requested_line_items": int(demand.get("requested_line_items", 0)),
            "fulfilled_line_items": int(demand.get("fulfilled_line_items", 0)),
            "partial_line_items": int(demand.get("partial_line_items", 0)),
            "unfilled_line_items": int(demand.get("unfilled_line_items", 0)),
            "quote_declined_units": demand["outside_rejections"],
            "stockout_request_units": demand["stockout_rejections"],
            "closed_service_request_units": demand.get(
                "service_closed_rejections", 0
            ),
            **(
                {
                    "capacity_policy_request_units": demand.get(
                        "capacity_policy_rejections", 0
                    )
                }
                if self._v24_reservation_operations()
                else {}
            ),
            "season_end_declined_units": demand.get("horizon_rejections", 0),
            "declined_after_quote_units": demand.get(
                "declined_after_quote_units", demand.get("outside_rejections", 0)
            ),
            "no_quote_units": demand.get("no_quote_units", 0),
            "end_cash": end["cash"],
            "end_deposit_liability": end["deposit_liability"],
            **(
                {
                    "end_unearned_revenue_liability": end[
                        "unearned_revenue_liability"
                    ]
                }
                if "unearned_revenue_liability" in end
                else {}
            ),
            "end_retail_units": end["retail_units"],
            "end_active_rentals": end["active_rentals"],
            "end_asset_status_counts": end["asset_status_counts"],
            "agent_actions": self._public_agent_actions(report.get("agent_actions")),
        }

    def public_service_keys(self) -> list[str]:
        return [row["service_key"] for row in self._service_registry]
