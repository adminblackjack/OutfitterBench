from __future__ import annotations

# Keep stable imports from outfitter.environment working while implementation
# responsibilities live in cohesive internal modules.
from .runtime.common import (
    ChannelDemandModel,
    ChoiceSession,
    DemandModel,
    OutfitterConfig,
    OutfitterDataset,
    RENTAL_ACTIVITY_EVENT_TYPES,
    RENTAL_HOLDING_RATE,
    ReferenceTransactionModel,
    SECONDHAND_GRADE_SCHEDULE,
    V5MarketRules,
    _RNG_EPISODE_MARKET,
    math,
)
from .runtime.base import RuntimeBaseMixin
from .runtime.context import CatalogIndex
from .runtime.queries import QueryMixin
from .runtime.operations import OperationsMixin
from .runtime.randomness import EventRandom
from .runtime.demand.features import DemandFeaturesMixin
from .runtime.demand.fulfillment import DemandFulfillmentMixin
from .runtime.demand.simulation import DemandSimulationMixin
from .runtime.demand.strategy import select_demand_strategy
from .runtime.lifecycle import LifecycleMixin
from .runtime.lifecycle_closure import LifecycleClosureMixin
from .runtime.reservations import ReservationMixin
from .runtime.warehouse import WarehouseMixin
from .runtime.acquisition_budget import AcquisitionBudgetMixin
from .runtime.revolving_durable_capital import RevolvingDurableCapitalMixin
from .runtime.supplier_trial import SupplierTrialMixin
from .runtime.service_commitments import ServiceCommitmentsMixin
from .runtime.reservation_contingency import ReservationContingencyMixin
from .runtime.actor_manual import ActorManualMixin
from .runtime.trip_weather import TripWeatherMixin
from .runtime.return_dynamics import ReturnDynamicsMixin
from .runtime.autonomous_rentals import AutonomousRentalsMixin
from .runtime.service_guarantee import ServiceGuaranteeMixin
from .runtime.school_holiday import SchoolHolidayMixin
from .runtime.reservation_overlap import ReservationOverlapMixin
from .reservation_overlap import ENVIRONMENT_PROTOCOL as OVERLAP_ENVIRONMENT_PROTOCOL
from .resale_disclosure import (ENVIRONMENT_PROTOCOL as V52_ENVIRONMENT_PROTOCOL,
    CORE_RELEASE_ID as V52_CORE_RELEASE_ID, enabled as resale_disclosure_enabled)
from .operating_history import (ENVIRONMENT_PROTOCOL as V53_ENVIRONMENT_PROTOCOL,
    CORE_RELEASE_ID as V53_CORE_RELEASE_ID, enabled as operating_history_enabled)
from .recovery_state import (ENVIRONMENT_PROTOCOL as V54_ENVIRONMENT_PROTOCOL,
    CORE_RELEASE_ID as V54_CORE_RELEASE_ID, enabled as recovery_state_enabled)
from .runtime.summer_pressure import SummerPressureMixin
from .summer_pressure import (ENVIRONMENT_PROTOCOL as V50_ENVIRONMENT_PROTOCOL,
                              policy_document as pressure_policy)
from .school_holiday import (ENVIRONMENT_PROTOCOL as V49_ENVIRONMENT_PROTOCOL,
                            policy_document as school_policy, SEASONAL_ANCHORS)
from .service_guarantee import ENVIRONMENT_PROTOCOL as V48_ENVIRONMENT_PROTOCOL
from .autonomous_rentals import ENVIRONMENT_PROTOCOL as V47_ENVIRONMENT_PROTOCOL
from .return_dynamics import ENVIRONMENT_PROTOCOL as V46_ENVIRONMENT_PROTOCOL
from .trip_weather import ENVIRONMENT_PROTOCOL as V44_ENVIRONMENT_PROTOCOL
from .reservation_difficulty import ENVIRONMENT_PROTOCOL as V45_ENVIRONMENT_PROTOCOL
from .actor_manual import ENVIRONMENT_PROTOCOL as V43_ENVIRONMENT_PROTOCOL
from .lifecycle_economics import LifecycleEconomicScales
from .weather_information import ENVIRONMENT_PROTOCOL as V42_ENVIRONMENT_PROTOCOL


__all__ = [
    "ChoiceSession",
    "OutfitterEnvironment",
    "RENTAL_ACTIVITY_EVENT_TYPES",
    "RENTAL_HOLDING_RATE",
    "SECONDHAND_GRADE_SCHEDULE",
]


class OutfitterEnvironment(
    ReservationOverlapMixin,
    SummerPressureMixin,
    SchoolHolidayMixin,
    ServiceGuaranteeMixin,
    AutonomousRentalsMixin,
    ReturnDynamicsMixin,
    TripWeatherMixin,
    ActorManualMixin,
    ReservationContingencyMixin,
    ServiceCommitmentsMixin,
    SupplierTrialMixin,
    RevolvingDurableCapitalMixin,
    AcquisitionBudgetMixin,
    WarehouseMixin,
    RuntimeBaseMixin,
    QueryMixin,
    LifecycleClosureMixin,
    OperationsMixin,
    DemandFeaturesMixin,
    DemandFulfillmentMixin,
    DemandSimulationMixin,
    ReservationMixin,
    LifecycleMixin,
):
    def business_manual(self):
        manual = super().business_manual()
        if operating_history_enabled(self.dataset.metadata):
            from .actor_contract_v53 import business_manual
            manual = business_manual(self, manual)
            if recovery_state_enabled(self.dataset.metadata):
                manual['reservations']['rules'].append(
                    'Pickup reduction eligibility uses current equipment, not cached shortages. '
                    'missing_optional_line_ids describes the current shortage; '
                    'inquired_missing_optional_line_ids records the omissions actually proposed to the customer. '
                    'Read-only eligibility does not reserve equipment or guarantee pickup. '
                    'A material contract version change invalidates prior reduction consent. '
                    'Within one version, inventory changes do not restore an already-used inquiry right; '
                    'selection and delivery must remain within the accepted reduction terms.')
            return manual
        return manual

    def __init__(self, config: OutfitterConfig, dataset: OutfitterDataset):
        self._initialize_overlap_candidate(dataset)
        self.config = config
        self.source_dataset = dataset
        supported_modes = tuple(
            dataset.metadata.get("supported_assortment_modes") or ()
        )
        if config.assortment_mode == "rental_only" and (
            "rental_only" not in supported_modes
        ):
            raise ValueError(
                "rental_only requires a runtime that explicitly supports assortment projection"
            )
        self.dataset = dataset.active_view(config.assortment_mode)
        dataset = self.dataset
        self._validate_warehouse_configuration()
        self._validate_durable_acquisition_budget_configuration()
        self._validate_supplier_trial_configuration()
        self._validate_service_commitments_configuration()
        self._validate_reservation_contingency_configuration()
        if dataset.metadata.get('environment_protocol') in {V52_ENVIRONMENT_PROTOCOL, V53_ENVIRONMENT_PROTOCOL, V54_ENVIRONMENT_PROTOCOL}:
            new_history = dataset.metadata.get('environment_protocol') != V52_ENVIRONMENT_PROTOCOL
            new_recovery = dataset.metadata.get('environment_protocol') == V54_ENVIRONMENT_PROTOCOL
            if (not resale_disclosure_enabled(dataset.metadata)
                    or dataset.metadata.get('core_release_id') != (V54_CORE_RELEASE_ID if new_recovery else V53_CORE_RELEASE_ID if new_history else V52_CORE_RELEASE_ID)
                    or (new_recovery and not recovery_state_enabled(dataset.metadata))
                    or (new_history and not operating_history_enabled(dataset.metadata))
                    or not config.lifecycle_economics_enabled):
                raise ValueError('V52/V53 requires its versioned public policies and lifecycle economics')
        elif dataset.metadata.get('resale_formula_disclosure_policy') is not None:
            raise ValueError('Resale formula disclosure requires the V52 protocol')
        if dataset.metadata.get('operating_history_policy') is not None and not operating_history_enabled(dataset.metadata):
            raise ValueError('Operating history policy requires V53')
        if dataset.metadata.get('recovery_state_policy') is not None and not recovery_state_enabled(dataset.metadata):
            raise ValueError('Current-pickup recovery policy requires the V54 protocol and interface')
        if dataset.metadata.get('environment_protocol') in {V49_ENVIRONMENT_PROTOCOL, V50_ENVIRONMENT_PROTOCOL, OVERLAP_ENVIRONMENT_PROTOCOL, V52_ENVIRONMENT_PROTOCOL, V53_ENVIRONMENT_PROTOCOL, V54_ENVIRONMENT_PROTOCOL}:
            if not (self._v49_school_holiday() and self._v48_service_guarantee()
                    and self._v47_autonomous_rentals() and self._v45_reservation_difficulty()):
                raise ValueError('V49 requires its calendar policy and inherited V48 rental kernel')
            if dataset.metadata['school_holiday_policy'] != school_policy():
                raise ValueError('V49 school-holiday policy is not the frozen candidate')
            if dataset.market_rules.seasonal_anchors != SEASONAL_ANCHORS:
                raise ValueError('V49 r2 requires its frozen mild seasonal profile')
        if dataset.metadata.get('environment_protocol') in {V50_ENVIRONMENT_PROTOCOL, OVERLAP_ENVIRONMENT_PROTOCOL, V52_ENVIRONMENT_PROTOCOL, V53_ENVIRONMENT_PROTOCOL, V54_ENVIRONMENT_PROTOCOL}:
            if (not self._v50_summer_pressure()
                    or dataset.metadata.get('summer_pressure_policy') != pressure_policy()):
                raise ValueError('V50 requires its frozen D2 summer-pressure policy')
        elif dataset.metadata.get('summer_pressure_policy') is not None:
            raise ValueError('Summer-pressure policy requires the V50 environment protocol')
        if dataset.metadata.get('environment_protocol')==V48_ENVIRONMENT_PROTOCOL:
            if not (self._v48_service_guarantee() and self._v47_autonomous_rentals() and self._v45_reservation_difficulty()):
                raise ValueError('V48 requires its guarantee and inherited autonomous rental kernel')
        if dataset.metadata.get("environment_protocol")==V47_ENVIRONMENT_PROTOCOL:
            if not (self._v47_autonomous_rentals() and self._v45_reservation_difficulty()
                    and self._v44_trip_weather() and self._v43_actor_manual() and not self._v46_return_dynamics()):
                raise ValueError("V47 requires its own autonomous process and inherited V45 demand rules")
        if dataset.metadata.get("environment_protocol") == V46_ENVIRONMENT_PROTOCOL:
            if not (self._v46_return_dynamics() and self._v45_reservation_difficulty()
                    and self._v44_trip_weather() and self._v43_actor_manual()
                    and self._v42_weather_information_coherence()):
                raise ValueError("V46 requires its return process and inherited V45 policies")
        if dataset.metadata.get("environment_protocol") == V45_ENVIRONMENT_PROTOCOL:
            if not (self._v45_reservation_difficulty() and self._v44_trip_weather()
                    and self._v43_actor_manual() and self._v42_weather_information_coherence()):
                raise ValueError("V45 requires its reservation policy and inherited V44 policies")
        if dataset.metadata.get("environment_protocol") == V44_ENVIRONMENT_PROTOCOL:
            if not (self._v44_trip_weather() and self._v43_actor_manual()
                    and self._v42_weather_information_coherence()):
                raise ValueError("V44 requires its forecast policy and inherited manual/weather policies")
        if dataset.metadata.get("environment_protocol") == V43_ENVIRONMENT_PROTOCOL:
            if not (self._v43_actor_manual() and self._v42_weather_information_coherence()):
                raise ValueError("V43 requires its public manual and inherited weather policy")
        if dataset.metadata.get("environment_protocol") == V42_ENVIRONMENT_PROTOCOL:
            if not self._v42_weather_information_coherence():
                raise ValueError("V42 requires its frozen weather-information policy")
            if not self._v41_actor_interface_slimming():
                raise ValueError("V42 requires the inherited V41 actor interface")
        if dataset.metadata.get("environment_protocol") == "outfitter-seasonal-v30-reservation-pickup-priority" and not self._v30_pickup_priority():
            raise ValueError("V30 requires the versioned before-walk-in pickup order")
        self.random = EventRandom(config.seed)
        self.rules: V5MarketRules = dataset.market_rules
        self.demand_model: DemandModel | None = dataset.demand_model
        self.reference_demand_model: ReferenceTransactionModel | None = (
            dataset.reference_demand_model
        )
        self.channel_demand_model: ChannelDemandModel | None = (
            dataset.channel_demand_model
        )
        self.lifecycle_factors = dict(dataset.lifecycle_factors)
        self.lifecycle_scales: LifecycleEconomicScales | None = (
            dataset.lifecycle_scales
        )
        self.reservation_promise_catalog = dataset.reservation_promise_catalog
        self.contract_feedback_policy = dataset.contract_feedback_policy
        self.plan_set_feedback_policy = dataset.plan_set_feedback_policy
        if config.reservation_contract_mode == "complete_contract_options_v1":
            if self.contract_feedback_policy is None:
                raise ValueError(
                    "complete_contract_options_v1 requires a frozen contract "
                    "feedback policy"
                )
            if (
                self.channel_demand_model is None
                or self.channel_demand_model.format
                != "outfitter-exact-offer-channel-demand-v5-request-bundle"
            ):
                raise ValueError(
                    "complete_contract_options_v1 requires request-bundle "
                    "channel demand v5"
                )
        if config.reservation_contract_mode == (
            "factorized_line_options_plan_set_v1"
        ):
            if self.plan_set_feedback_policy is None:
                raise ValueError(
                    "factorized_line_options_plan_set_v1 requires a frozen plan-set "
                    "feedback policy"
                )
            if (
                self.channel_demand_model is None
                or self.channel_demand_model.format
                != "outfitter-exact-offer-channel-demand-v5-request-bundle"
            ):
                raise ValueError(
                    "factorized_line_options_plan_set_v1 requires request-bundle "
                    "channel demand v5"
                )
            tool_interface_version = int(
                dataset.metadata.get("tool_interface_version", 0)
            )
            if (
                tool_interface_version >= 19
                and not self.plan_set_feedback_policy.factorized_concern_tags_enabled
            ):
                raise ValueError(
                    "tool interface 19 requires the v3 factorized-concern "
                    "feedback policy"
                )
            if (
                17 <= tool_interface_version < 19
                and self.plan_set_feedback_policy.factorized_concern_tags_enabled
            ):
                raise ValueError(
                    "the v3 factorized-concern feedback policy requires tool "
                    "interface 19"
                )
            if (
                17 <= tool_interface_version < 19
                and not self.plan_set_feedback_policy.noisy_line_mentions_enabled
            ):
                raise ValueError(
                    "tool interfaces 17-18 require line-enabled feedback"
                )
            if (
                tool_interface_version < 17
                and self.plan_set_feedback_policy.noisy_line_mentions_enabled
            ):
                raise ValueError(
                    "line-enabled feedback requires tool interface 17 or newer"
                )
        lifecycle_metadata = dict(dataset.metadata.get("lifecycle_economics") or {})
        liquidation_policy = str(
            dataset.metadata.get("liquidation_policy") or "legacy_configured_depth"
        )
        flat_liquidation_runtime = liquidation_policy == "flat_nominal_per_unit_v1"
        if flat_liquidation_runtime and config.liquidation_depth_enabled:
            raise ValueError(
                "this runtime fixes flat nominal liquidation and does not allow "
                "liquidation_depth_enabled"
            )
        if lifecycle_metadata.get("enabled_config_required") and not (
            config.lifecycle_economics_enabled
        ):
            raise ValueError(
                "this runtime requires lifecycle_economics_enabled=true"
            )
        if config.lifecycle_economics_enabled:
            if self.lifecycle_scales is None or not self.lifecycle_factors:
                raise ValueError(
                    "lifecycle_economics_enabled requires a lifecycle runtime"
                )
            if config.demand_world_id not in {
                "conservative",
                "base",
                "convenience-heavy",
            }:
                raise ValueError(
                    "lifecycle economics supports conservative, base, or convenience-heavy"
                )
            missing_mechanisms = [
                name
                for name, enabled in (
                    ("weather_wear_enabled", config.weather_wear_enabled),
                    (
                        "preventive_maintenance_enabled",
                        config.preventive_maintenance_enabled,
                    ),
                    ("liquidation_depth_enabled", config.liquidation_depth_enabled),
                )
                if not enabled
                and not (
                    name == "liquidation_depth_enabled"
                    and flat_liquidation_runtime
                )
            ]
            if missing_mechanisms and not config.lifecycle_mechanism_ablation:
                raise ValueError(
                    "lifecycle economics requires enabled mechanisms: "
                    + ", ".join(missing_mechanisms)
                )
            if (
                not config.lifecycle_holding_enabled
                and not config.lifecycle_mechanism_ablation
            ):
                raise ValueError(
                    "disabling lifecycle holding requires lifecycle_mechanism_ablation=true"
                )
        elif config.lifecycle_mechanism_ablation:
            raise ValueError(
                "lifecycle_mechanism_ablation requires lifecycle_economics_enabled"
            )
        reputation_metadata = dict(
            dataset.metadata.get("reservation_service_reputation") or {}
        )
        if config.reservation_service_reputation_enabled:
            if not flat_liquidation_runtime:
                raise ValueError(
                    "the service-reputation experiment requires the V29 runtime"
                )
            if reputation_metadata.get("format") != (
                "outfitter-reservation-service-reputation-v1"
            ):
                raise ValueError(
                    "service reputation requires a frozen V29 reputation policy"
                )
        self.demand_strategy = select_demand_strategy(
            demand_model=self.demand_model,
            reference_demand_model=self.reference_demand_model,
            channel_demand_model=self.channel_demand_model,
            world_id=config.demand_world_id,
        )
        if config.reservations_enabled and not self.demand_strategy.channel_enabled:
            raise ValueError(
                "reservations_enabled requires an exact-offer channel demand runtime"
            )
        profile_metadata = dict(
            dataset.metadata.get("reservation_promise_profiles") or {}
        )
        profile_contract_modes = {
            "promise_profiles_v1",
            "adaptive_pairwise_exact_quotes_v1",
            "complete_contract_options_v1",
            "factorized_line_options_plan_set_v1",
        }
        if config.reservation_contract_mode in profile_contract_modes:
            if self.reservation_promise_catalog is None:
                raise ValueError(
                    "profile reservation contracts require a versioned profile runtime"
                )
            self.reservation_promise_catalog.unit_deposit(
                next(
                    row.profile_id
                    for row in self.reservation_promise_catalog.profiles
                    if row.member_offer_ids
                ),
                str(config.reservation_profile_deposit_policy),
            )
        elif profile_metadata.get("enabled_config_required"):
            raise ValueError(
                "this runtime requires a profile reservation contract mode"
            )
        formal_fee = dataset.metadata.get("formal_daily_fee")
        if formal_fee is not None and not math.isclose(
            float(formal_fee), float(config.daily_fee), abs_tol=1e-9
        ):
            raise ValueError(
                "runtime requires daily_fee="
                f"{float(formal_fee):.2f}, received {float(config.daily_fee):.2f}"
            )
        self.rules.validate()
        self.catalog = CatalogIndex.build(dataset, self._service_key_for_line)
        # Compatibility aliases for research code are references to the
        # catalog-owned collections, not duplicated indexes.
        self.offers = self.catalog.offers
        self.offer_rng_identities = dict(dataset.offer_rng_identities)
        self.by_line = self.catalog.by_line
        self.by_service = self.catalog.by_service
        self.by_group = self.catalog.by_group
        self.product_type_to_service_key = (
            self.catalog.product_type_to_service_key
        )
        self._service_registry = self.catalog.service_registry
        self.reference_prices = self._reference_prices()
        self.reference_daily_rates = {
            offer.offer_id: (
                float(offer.reference_daily_rate)
                if offer.reference_daily_rate is not None
                else self.reference_prices[offer.service_line]
            )
            for offer in dataset.offers
            if offer.asset_kind == "durable"
        }
        # This can be temporarily varied by the local calibration harness. Formal
        # runs use the fixed module constant and record the resolved value.
        self.service_intercept = self.rules.service_intercept
        self.episode_market_profile = (
            self._build_episode_market_profile()
            if self.demand_model is not None
            and not self.demand_strategy.channel_enabled
            else None
        )
        self.reference_local_shifts: dict[str, float] = {}
        if self.reference_demand_model is not None:
            missing_assignments = sorted(
                set(self.offers) - set(self.reference_demand_model.offer_assignments)
            )
            if missing_assignments:
                raise ValueError(
                    "reference demand runtime contains offers without reference cells: "
                    f"{missing_assignments}"
                )
            expected = {
                f"{service_key}|{group}" for service_key, group in self.by_group
            }
            actual = set(self.reference_demand_model.groups)
            if not self._active_group_coverage_is_valid(expected, actual):
                raise ValueError(
                    "reference demand group coverage differs from runtime catalog"
                )
            low, high = self.reference_demand_model.local_shift_range
            for key in sorted(expected):
                rng = self._stable_event_rng(
                    _RNG_EPISODE_MARKET, identity=f"reference-local:{key}"
                )
                self.reference_local_shifts[key] = float(rng.uniform(low, high))
        if self.channel_demand_model is not None:
            if (
                self.config.demand_world_id != "control"
                and self.config.demand_world_id
                not in self.channel_demand_model.worlds
            ):
                raise ValueError(
                    "channel runtime supports control, conservative, base, and "
                    "convenience-heavy demand worlds"
                )
            expected = {
                f"{service_key}|{group}" for service_key, group in self.by_group
            }
            actual = set(self.channel_demand_model.groups)
            if not self._active_group_coverage_is_valid(expected, actual):
                raise ValueError(
                    "channel demand group coverage differs from runtime catalog: "
                    f"missing={sorted(expected-actual)}, "
                    f"extra={sorted(actual-expected)}"
                )
            for offer in self.dataset.offers:
                online = offer.consumer_online_offer
                if not isinstance(online, dict):
                    raise ValueError(
                        f"channel demand offer {offer.offer_id} lacks consumer_online_offer"
                    )
                package_price = float(online.get("package_price_cny", 0.0))
                package_units = int(online.get("consumer_units_per_package", 0))
                unit_price = float(online.get("unit_price_cny", 0.0))
                if (
                    not math.isfinite(package_price)
                    or package_price <= 0
                    or package_units <= 0
                    or not math.isfinite(unit_price)
                    or unit_price <= 0
                    or not math.isclose(
                        unit_price,
                        package_price / package_units,
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    )
                ):
                    raise ValueError(
                        f"channel demand offer {offer.offer_id} has invalid online package units"
                    )

    def _active_group_coverage_is_valid(
        self, expected: set[str], actual: set[str]
    ) -> bool:
        if self.config.retail_enabled:
            return actual == expected
        return expected <= actual and all(
            key.startswith("retail:") for key in actual - expected
        )
