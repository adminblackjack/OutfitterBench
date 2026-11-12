from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Offer:
    offer_id: str
    service_line: str
    category_id: str
    category_label_zh: str
    asset_kind: str
    name: str
    package_price: float
    package_units: int
    unit_cost: float
    seller_id: str
    seller_name: str
    source_title: str
    source_variant_name: str
    evidence: list[dict[str, Any]]
    consumer_unit: dict[str, Any] | None = None
    product_subtype: str | None = None
    fulfills_service_keys: list[str] = field(default_factory=list)
    reference_retail_price: float | None = None
    reference_retail_price_method: str | None = None
    quality_tier: str | None = None
    service_life_limit_days: int | None = None
    service_life_limit_cycles: int | None = None
    damage_risk_band: str | None = None
    lifecycle_prior_scope: str | None = None
    choice_group: str | None = None
    decision_attributes: dict[str, Any] = field(default_factory=dict)
    descriptive_attributes: dict[str, Any] = field(default_factory=dict)
    consumer_value_attributes: dict[str, Any] = field(default_factory=dict)
    consumer_value_evidence: dict[str, Any] = field(default_factory=dict)
    perceived_quality_tier: str | None = None
    quality_tier_provenance: str | None = None
    perceived_quality_provenance: str | None = None
    perceived_quality_confidence: str | None = None
    decision_evidence: dict[str, Any] = field(default_factory=dict)
    unresolved_decision_attributes: list[str] = field(default_factory=list)
    cleaning_work_points: float | None = None
    reference_daily_rate: float | None = None
    decision_signature: str | None = None
    brand: str | None = None
    # A separately versioned customer-facing exact-listing observation.  It may
    # initially match package_price/package_units, but is not an accounting cost.
    consumer_online_offer: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Offer":
        return cls(**value)


@dataclass(slots=True)
class RentalAsset:
    asset_id: str
    offer_id: str
    service_line: str
    unit_cost: float
    used_life_days: int
    service_life_limit_days: int
    damage_risk_band: str
    used_rental_cycles: int = 0
    service_life_limit_cycles: int | None = None
    status: str = "available"
    ready_day: int | None = None
    acquired_day: int = 0
    sold_day: int | None = None
    # Physical removal, not merely an operational status. V31 only.
    disposed_day: int | None = None
    sale_grade: str | None = None
    sale_proceeds: float | None = None
    next_inspection_due_day: int | None = None
    maintenance_started_day: int | None = None
    # Physical condition is separate from additive day/cycle wear. Old states
    # migrate to pristine condition instead of inferring condition from wear.
    condition_ratio: float = 1.0
    maintenance_wear_anchor: float = 0.0
    pending_repair_cost: float | None = None
    repair_target_condition_ratio: float | None = None
    ownership_type: str = "owned"
    supplier_trial_id: str | None = None
    # V40 gross deployed-capital accounting. Integer cents are authoritative;
    # supplier-owned trials remain exempt and keep the basis unset.
    durable_capital_basis_cents: int | None = None
    durable_capital_source_order_id: str | None = None
    durable_capital_source_item_index: int | None = None
    durable_capital_source_unit_index: int | None = None
    capital_exit_day: int | None = None
    capital_exit_cause: str | None = None
    capital_release_effective_day: int | None = None


@dataclass(slots=True)
class RentalContract:
    contract_id: str
    asset_id: str
    offer_id: str
    service_line: str
    start_day: int
    return_day: int
    duration_days: int
    rent_paid: float
    deposit: float
    damaged_on_return: bool
    status: str = "active"
    reservation_id: str | None = None
    promised_return_day: int | None = None
    approved_return_day: int | None = None
    estimated_return_day: int | None = None
    actual_return_day: int | None = None
    damage_event_uniform: float | None = None
    damage_severity_uniform: float | None = None
    damage_condition_loss: float | None = None
    damage_wear_at_start: float | None = None
    damage_condition_at_start: float | None = None
    damage_maintenance_exposure_at_start: float | None = None
    # V37 supplier-owned trial contracts pay explicit use fees rather than
    # consuming owned capital. Defaults keep older serialized states readable.
    supplier_trial_id: str | None = None
    supplier_trial_usage_fee: float = 0.0
    rental_episode_id: str | None = None


@dataclass(slots=True)
class ReservationLine:
    line_id: str
    offer_id: str
    service_line: str
    quantity: int
    customer_units: int
    duration_days: int
    locked_daily_price: float = 0.0
    locked_rent: float = 0.0
    # Before a quote these are the exact offers the customer is willing to
    # consider under one line-level price.  Their order is frozen at inquiry
    # creation and is public; willingness ceilings remain request-private.
    candidate_offer_ids: list[str] = field(default_factory=list)
    quoted_daily_price: float | None = None
    # ``offer_id`` remains the primary exact offer for snapshot compatibility.
    # The ordered public authorization set always starts with this value.
    authorized_offer_ids: list[str] = field(default_factory=list)
    locked_security_deposit: float = 0.0
    selected_offer_id: str | None = None
    fulfillment_mode: str | None = None
    substitution_hint: str | None = None
    required: bool = True
    status: str = "committed"
    credit_amount: float = 0.0
    # V16 promise-profile fields are additive so V15 states remain readable.
    service_key: str | None = None
    service_units: int | None = None
    public_hard_requirements: dict[str, Any] = field(default_factory=dict)
    public_preferences: list[dict[str, Any]] = field(default_factory=list)
    eligible_profile_ids: list[str] = field(default_factory=list)
    eligible_profile_family_ids: list[str] = field(default_factory=list)
    promise_profile_id: str | None = None
    profile_family_id: str | None = None
    profile_schema_version: str | None = None
    capacity_per_asset: int | None = None
    asset_quantity: int | None = None
    staged_offer_id: str | None = None
    # Pairwise V16 quote snapshot. Options are canonicalized and assigned local
    # option IDs before customer evaluation. A rejected request deliberately
    # leaves selected_quote_option_id unset for every line.
    submitted_quote_options: list[dict[str, Any]] = field(default_factory=list)
    selected_quote_option_id: str | None = None
    # V24 keeps the customer-authorized quote distinct from any lower amount
    # protected during an explicit complete-plan recovery.
    authorized_line_rent: float | None = None
    authorized_security_deposit: float | None = None
    recovery_price_credit: float = 0.0
    recovery_deposit_credit: float = 0.0
    recovery_source_plan_id: str | None = None
    # V32 whole-line daily consideration, independent of selected asset count.
    locked_line_daily_rent: float | None = None
    locked_line_pricing_basis: str | None = None


@dataclass(slots=True)
class ReservationRequest:
    request_id: str
    created_day: int
    expires_day: int
    pickup_day: int
    return_day: int
    booking_lead_days: int
    lines: list[ReservationLine]
    visible_party_context: str
    forecast_condition: str
    forecast_accuracy: float
    prepayment_rate: float
    total_locked_rent: float
    prepayment_due: float
    status: str = "pending_quote"
    decision_day: int | None = None
    commitment_id: str | None = None
    # Frozen hidden coordinates support deterministic pickup outcomes without
    # exposing customer-channel state to the manager.
    event_address: list[int] = field(default_factory=list)
    forecast_trust: float = 0.5
    customer_segment: str = "balanced"
    shared_price_beta: float = 0.0
    storage_burden_base: float = 0.5
    future_use_expectation_base: float = 0.0
    return_burden_multiplier: float = 1.0
    request_budget_multiplier: float = 1.0
    schedule_flexibility: float = 0.5
    fallback_offer_ids: list[str] = field(default_factory=list)
    fallback_offer_ids_by_line: dict[str, list[str]] = field(default_factory=dict)
    # Immutable inquiry-time truth. Quote processing must never overwrite it,
    # including when the customer rejects or an optional line is removed.
    inquiry_latent_offer_ids_by_line: dict[str, list[str]] = field(
        default_factory=dict
    )
    inquiry_latent_price_ceiling_by_line: dict[str, dict[str, float]] = field(
        default_factory=dict
    )
    # Post-quote public authorization and hidden recovery are separate from
    # the immutable inquiry snapshot.
    authorized_offer_ids_by_line: dict[str, list[str]] = field(default_factory=dict)
    authorized_price_ceiling_by_line: dict[str, dict[str, float]] = field(
        default_factory=dict
    )
    recovery_offer_ids_by_line: dict[str, list[str]] = field(default_factory=dict)
    recovery_price_ceiling_by_line: dict[str, dict[str, float]] = field(
        default_factory=dict
    )
    latent_snapshot_provenance: str = "immutable_inquiry_v1"
    # Frozen hidden customer contexts and deposit snapshots make quote
    # outcomes independent of any same-day price/assortment changes after the
    # inquiry appeared.  Public reservation views must never expose them.
    quote_context_by_line: dict[str, dict[str, Any]] = field(default_factory=dict)
    candidate_deposit_by_line: dict[str, dict[str, float]] = field(
        default_factory=dict
    )
    quote_action: str | None = None
    quote_day: int | None = None
    reservation_contract_mode: str = "exact_candidates_v15"
    catalog_snapshot_sha256: str | None = None
    promise_profile_schema_version: str | None = None
    profile_deposit_policy_id: str | None = None
    inquiry_public_envelope_by_line: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    inquiry_profile_member_ids_by_line: dict[
        str, dict[str, list[str]]
    ] = field(default_factory=dict)
    inquiry_profile_price_ceiling_by_line: dict[
        str, dict[str, dict[str, float]]
    ] = field(default_factory=dict)
    inquiry_profile_deposit_by_line: dict[str, dict[str, float]] = field(
        default_factory=dict
    )
    # V19 request-level complete contract options.  These snapshots are
    # additive and remain empty for every earlier contract mode.  Public
    # projections expose them only after a valid quote has been processed.
    submitted_contract_options: list[dict[str, Any]] = field(default_factory=list)
    selected_contract_option_id: str | None = None
    reported_contract_assessments: list[dict[str, Any]] = field(
        default_factory=list
    )
    feedback_response_available: bool | None = None
    # Analyzer-only audit data.  It must never enter a manager-visible request,
    # quote result, history row, event, or error.
    contract_feedback_audit: dict[str, Any] = field(default_factory=dict)
    # V20 factorized line options and the capped customer-authorized complete
    # plan set. These remain empty for every earlier contract mode. Only the
    # returned plans are persisted; non-returned internal combinations remain
    # private and non-exhaustively disclosed.
    submitted_line_options: list[dict[str, Any]] = field(default_factory=list)
    customer_authorized_plans: list[dict[str, Any]] = field(default_factory=list)
    customer_preferred_plan_id: str | None = None
    manager_confirmed_plan_id: str | None = None
    plan_confirmation_origin: str | None = None
    manager_confirmation_day: int | None = None
    internal_plan_count: int = 0
    acceptable_plan_count: int = 0
    reported_decline_feedback: dict[str, Any] = field(default_factory=dict)
    plan_set_feedback_audit: dict[str, Any] = field(default_factory=dict)
    # Immutable V26 inquiry dates. They are hidden from generic request
    # projections and exposed only with explicit original/inquiry labels.
    original_inquiry_pickup_day: int | None = None
    original_inquiry_return_day: int | None = None
    # Public V29 experiment snapshot.  It is frozen when the inquiry is
    # created so later same-day cancellations cannot change that customer's
    # evaluation.  The formal V29 environment leaves this mechanism disabled.
    service_reputation_snapshot: dict[str, Any] = field(default_factory=dict)
    pickup_date_options: list[dict[str, Any]] = field(default_factory=list)
    service_offer_terms: dict[str, Any] = field(default_factory=dict)
    # Private, immutable generation/choice data; never included in public DTOs.
    quote_context_by_date: dict[str, dict[str, Any]] = field(default_factory=dict)
    service_offer_audit: dict[str, Any] = field(default_factory=dict)
    # V46 publicly reported itinerary/transport class, frozen before quoting.
    trip_context: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ReservationAssignment:
    assignment_id: str
    commitment_id: str
    line_id: str
    asset_id: str
    assigned_day: int
    status: str = "active"


@dataclass(slots=True)
class ReservationCommitment:
    commitment_id: str
    request_id: str
    confirmed_day: int
    pickup_day: int
    return_day: int
    lines: list[ReservationLine]
    total_locked_rent: float
    prepayment_collected: float
    outstanding_rent: float
    unearned_revenue: float
    status: str = "confirmed"
    assignments: list[ReservationAssignment] = field(default_factory=list)
    original_pickup_day: int | None = None
    original_return_day: int | None = None
    promised_return_day: int | None = None
    approved_return_day: int | None = None
    estimated_return_day: int | None = None
    actual_return_day: int | None = None
    risk_exposed_day: int | None = None
    fallback_inquiry_day: int | None = None
    fallback_revealed_offer_ids_by_line: dict[str, list[str]] = field(
        default_factory=dict
    )
    failure_cause: str | None = None
    optional_line_credits: float = 0.0
    # Dormant complete plans reserve no capacity and are never selected
    # automatically.  They are frozen here so later request projection changes
    # cannot alter the customer's original authorization.
    preauthorized_recovery_plans: list[dict[str, Any]] = field(default_factory=list)
    selected_recovery_plan_id: str | None = None
    recovery_path: str | None = None
    original_confirmed_plan_id: str | None = None
    original_total_locked_rent: float | None = None
    original_total_security_deposit: float | None = None
    # V26 separates historical cash facts from current balances. These fields
    # default cleanly when loading every pre-V26 persisted state.
    rent_due_at_pickup: float | None = None
    prepayment_refunded: float = 0.0
    compensation_paid: float = 0.0
    # V48 original obligation bases and settlement ledger, never reset by version changes.
    service_guarantee: dict[str, Any] | None = None
    # Hidden customer-process state. Public projections must remove these
    # fields; they only prevent a stale forecast from causing repeated
    # reschedule requests with no new information.
    reschedule_forecast_baseline_condition: str | None = None
    reschedule_last_request_condition: str | None = None
    reschedule_last_request_day: int | None = None
    # V27 historical rent receipts. Derived gross/net totals are deliberately
    # not persisted, so public reconciliation cannot drift from components.
    rent_collected_at_pickup: float = 0.0
    post_pickup_rent_collected: float = 0.0
    initial_agreed_pickup_day: int | None = None
    initial_agreed_return_day: int | None = None
    # V38 public term version. It changes only when the customer's contractual
    # dates/equipment or active-trip return date changes.
    commitment_version: int = 1
    contingency_inquiry_used: bool = False
    # Hidden, version-scoped customer authorization for reactive recovery.
    version_recovery_offer_ids_by_line: dict[str, list[str]] = field(
        default_factory=dict
    )
    # V42 hidden customer-process state. These fields are never projected to
    # the manager and prevent repeated reactions to the same public forecast.
    weather_review_pickup_day: int | None = None
    weather_review_baseline_regime: str | None = None
    weather_review_max_adversity: float | None = None
    weather_reviewed_regimes: list[str] = field(default_factory=list)
    natural_cancellation_day: int | None = None
    # V45: consent to a specifically disclosed reduced pickup is version-scoped.
    partial_fulfillment_inquiry_day: int | None = None
    partial_fulfillment_missing_line_ids: list[str] = field(default_factory=list)
    partial_fulfillment_accepted: bool | None = None
    partial_fulfillment_selected: bool = False
    partial_fulfillment_compensation: float = 0.0
    # Private once-per-trip draws; never expose via actor serialization.
    return_process_private: dict[str, Any] = field(default_factory=dict)
    return_notices: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ReservationChangeRequest:
    change_id: str
    commitment_id: str
    change_type: str
    created_day: int
    expires_day: int
    proposed_pickup_day: int
    proposed_return_day: int
    extension_days: int
    status: str = "pending"
    decision_day: int | None = None
    # V27 freezes the deterministic financial consequence when an extension
    # request is created. These remain empty for reschedules and older runs.
    projected_additional_locked_rent: float | None = None
    projected_line_rent_deltas: dict[str, float] = field(default_factory=dict)
    collection_timing: str | None = None
    # Public dates/rent immediately before this proposal. Frozen so later
    # accepted changes cannot rewrite the meaning of earlier timeline entries.
    previous_pickup_day: int | None = None
    previous_return_day: int | None = None
    previous_total_locked_rent: float | None = None
    original_dates_still_possible: bool | None = None
    # V38 freezes which commitment terms the customer request evaluated.
    evaluated_commitment_version: int | None = None
    # V42 evaluator-only snapshot used to reset a newly accepted version's
    # weather baseline without consulting future truth.
    weather_forecast_snapshot_by_day: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class ReservationContingencyInquiry:
    inquiry_id: str
    commitment_id: str
    evaluated_commitment_version: int
    submitted_day: int
    response_day: int
    proposals: list[dict[str, Any]]
    forecast_snapshot_by_day: dict[str, dict[str, Any]]
    status: str = "pending"
    report_id: str | None = None


@dataclass(slots=True)
class ReservationContingencyReport:
    report_id: str
    inquiry_id: str
    commitment_id: str
    evaluated_commitment_version: int
    response_day: int
    expires_day: int
    proposals: list[dict[str, Any]]
    accepted_proposal_ids: list[str] = field(default_factory=list)
    rejected_proposal_ids: list[str] = field(default_factory=list)
    counteroffer: dict[str, Any] | None = None
    outcome: str = "keep_original"
    status: str = "active"
    settlement: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReservationEvent:
    event_id: str
    day: int
    event_type: str
    request_id: str | None = None
    commitment_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CalendarBlock:
    block_id: str
    commitment_id: str
    line_id: str
    offer_id: str
    day: int
    quantity: int
    authorized_offer_ids: list[str] = field(default_factory=list)
    selected_offer_id: str | None = None
    block_type: str = "hard_commitment"
    status: str = "active"
    promise_profile_id: str | None = None
    profile_schema_version: str | None = None
    service_units: int | None = None
    capacity_per_asset: int | None = None
    staged_offer_id: str | None = None


@dataclass(slots=True)
class WalkInCapacityLimit:
    offer_id: str
    day: int
    max_walk_in_occupied_units: int


@dataclass(slots=True)
class PurchaseOrder:
    order_id: str
    seller_id: str
    placed_day: int
    delivery_day: int
    total_cost: float
    items: list[dict[str, Any]]
    estimated_delivery_day: int | None = None
    status: str = "in_transit"
    procurement_mode: str = "owned"
    supplier_trial_id: str | None = None
    supplier_trial_handling_fee: float = 0.0


@dataclass(slots=True)
class SupplierTrialRecord:
    trial_id: str
    offer_id: str
    order_id: str
    accepted_day: int
    handling_fee: float
    status: str = "in_transit"
    asset_id: str | None = None
    delivered_day: int | None = None
    return_deadline: int | None = None
    remaining_contract_starts: int = 3
    return_pending: bool = False
    returned_day: int | None = None


@dataclass(slots=True)
class CleaningJob:
    asset_id: str
    offer_id: str
    product_type: str
    enqueued_day: int
    duration_days: int
    remaining_work_points: float
    cleaning_cost: float


@dataclass(slots=True)
class ToolResult:
    ok: bool
    data: dict[str, Any]
    terminal_day: bool = False


@dataclass(slots=True)
class MemoryDocument:
    name: str
    description: str
    content: str
    created_day: int
    updated_day: int


@dataclass(slots=True)
class WorldState:
    day: int
    max_days: int
    cash: float
    deposit_liability: float
    unearned_revenue_liability: float
    prices: dict[str, float]
    deposits: dict[str, float]
    rental_service_open: dict[str, bool]
    assets: list[RentalAsset]
    retail_inventory: dict[str, dict[str, float]]
    contracts: list[RentalContract] = field(default_factory=list)
    orders: list[PurchaseOrder] = field(default_factory=list)
    durable_acquisition_budget: dict[str, Any] = field(default_factory=dict)
    supplier_trials: list[SupplierTrialRecord] = field(default_factory=list)
    memory_documents: dict[str, MemoryDocument] = field(default_factory=dict)
    inbox: list[str] = field(default_factory=list)
    daily_history: list[dict[str, Any]] = field(default_factory=list)
    current_day_ledger: dict[str, Any] = field(default_factory=dict)
    # Kept so runs written before event-addressed randomness remain loadable.
    rng_state: dict[str, Any] = field(default_factory=dict)
    next_asset_id: int = 1
    next_contract_id: int = 1
    next_order_id: int = 1
    next_supplier_trial_id: int = 1
    done: bool = False
    bankrupt: bool = False
    bankruptcy_shortfall: float = 0.0
    cumulative_rental_revenue: float = 0.0
    cumulative_retail_revenue: float = 0.0
    cumulative_purchase_cost: float = 0.0
    cumulative_supplier_trial_handling_fees: float = 0.0
    cumulative_supplier_trial_usage_fees: float = 0.0
    cumulative_fees: float = 0.0
    cumulative_model_cost: float = 0.0
    cumulative_refunds: float = 0.0
    cumulative_deposits_collected: float = 0.0
    cumulative_damage_deductions: float = 0.0
    cumulative_cleaning_cost: float = 0.0
    cumulative_repair_cost: float = 0.0
    cumulative_retail_holding_cost: float = 0.0
    cumulative_rental_holding_cost: float = 0.0
    cumulative_rental_idle_holding_cost: float = 0.0
    cumulative_rental_capital_holding_cost: float = 0.0
    cumulative_rental_space_holding_cost: float = 0.0
    cumulative_asset_sale_proceeds: float = 0.0
    cumulative_assets_sold: int = 0
    cumulative_retail_liquidation_proceeds: float = 0.0
    cumulative_retail_liquidation_book_cost: float = 0.0
    cumulative_retail_liquidation_loss: float = 0.0
    cumulative_retail_units_liquidated: int = 0
    cumulative_customers: int = 0
    cumulative_transactions: int = 0
    cumulative_full_baskets: int = 0
    cumulative_partial_baskets: int = 0
    cumulative_unfilled_baskets: int = 0
    cumulative_operationally_evaluable_baskets: int = 0
    cumulative_operational_full_baskets: int = 0
    cumulative_operational_partial_baskets: int = 0
    cumulative_operational_unfilled_baskets: int = 0
    cumulative_requested_units: int = 0
    cumulative_fulfilled_units: int = 0
    cumulative_requested_line_items: int = 0
    cumulative_fulfilled_line_items: int = 0
    cumulative_partial_line_items: int = 0
    cumulative_unfilled_line_items: int = 0
    cumulative_outside_rejections: int = 0
    cumulative_stockout_rejections: int = 0
    cumulative_service_closed_rejections: int = 0
    cumulative_capacity_policy_rejections: int = 0
    cumulative_horizon_rejections: int = 0
    cumulative_unpriced_rejections: int = 0
    # Retained as a legacy aggregate so old persisted states remain readable.
    cumulative_closed_rejections: int = 0
    forced_day_ends: int = 0
    parallel_tool_call_anomalies: int = 0
    discarded_tool_calls: int = 0
    read_only_tool_batches: int = 0
    batched_read_only_tool_calls: int = 0
    rejected_tool_batches: int = 0
    rejected_batched_tool_calls: int = 0
    rental_activity: list[dict[str, Any]] = field(default_factory=list)
    cleaning_queue: list[CleaningJob] = field(default_factory=list)
    recent_counter_interactions: list[dict[str, Any]] = field(default_factory=list)
    # V47 evaluator state. Actor projections must whitelist, never serialize it.
    rental_episodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    rental_episode_protocol: str | None = None
    calendar_demand_protocol: str | None = None
    ever_carried_offer_ids: list[str] = field(default_factory=list)
    liquidation_market_history: list[dict[str, Any]] = field(default_factory=list)
    liquidation_depth_capacity: dict[str, float] = field(default_factory=dict)
    reservation_requests: list[ReservationRequest] = field(default_factory=list)
    reservation_commitments: list[ReservationCommitment] = field(default_factory=list)
    reservation_events: list[ReservationEvent] = field(default_factory=list)
    reservation_change_requests: list[ReservationChangeRequest] = field(
        default_factory=list
    )
    reservation_contingency_inquiries: list[ReservationContingencyInquiry] = field(
        default_factory=list
    )
    reservation_contingency_reports: list[ReservationContingencyReport] = field(
        default_factory=list
    )
    reservation_calendar_blocks: list[CalendarBlock] = field(default_factory=list)
    walk_in_capacity_limits: list[WalkInCapacityLimit] = field(default_factory=list)
    next_reservation_event_id: int = 1
    next_reservation_contingency_inquiry_id: int = 1
    next_reservation_contingency_report_id: int = 1
    cumulative_reservation_requests: int = 0
    cumulative_reservation_quotes: int = 0
    cumulative_reservation_customer_quote_declines: int = 0
    cumulative_reservation_manager_quote_declines: int = 0
    cumulative_reservation_optional_lines_removed_at_quote: int = 0
    cumulative_reservations_confirmed: int = 0
    cumulative_reservations_declined: int = 0
    cumulative_reservations_expired: int = 0
    cumulative_reservation_pickups: int = 0
    cumulative_reservation_pickup_failures: int = 0
    cumulative_reservation_prepayments: float = 0.0
    cumulative_reservation_prepayment_refunds: float = 0.0
    cumulative_reservation_customer_cancellations: int = 0
    cumulative_reservation_reschedules: int = 0
    cumulative_reservation_extensions_requested: int = 0
    cumulative_reservation_extensions_approved: int = 0
    cumulative_reservation_late_returns: int = 0
    cumulative_reservation_compensation: float = 0.0
    cumulative_reservation_staging_actions: int = 0
    cumulative_reservation_staged_fallbacks: int = 0
    cumulative_reservation_contingency_inquiries: int = 0
    cumulative_reservation_contingencies_applied: int = 0
    cumulative_reservation_contingency_customer_cancellations: int = 0
    decision_horizon_day: int | None = None
    runoff_end_day: int | None = None
    emergency_credit_principal: float = 0.0
    emergency_credit_liability: float = 0.0
    cumulative_emergency_credit_fees: float = 0.0
    cumulative_maintenance_cost: float = 0.0
    cumulative_maintenance_actions: int = 0
    cumulative_optional_line_credits: float = 0.0
    warehouse_capacity: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data.get("rental_episodes"):
            data.pop("rental_episodes",None)
        if data.get("rental_episode_protocol") is None:
            data.pop("rental_episode_protocol",None)
        if data.get("calendar_demand_protocol") is None:
            data.pop("calendar_demand_protocol", None)
        # New mechanisms do not alter the persisted wire shape of legacy runs.
        for request in data.get("reservation_requests", []):
            for key in ("pickup_date_options", "service_offer_terms", "quote_context_by_date", "service_offer_audit"):
                if not request.get(key):
                    request.pop(key, None)
        for commitment in data.get("reservation_commitments", []):
            if commitment.get("service_guarantee") is None:
                commitment.pop("service_guarantee", None)
            for key in ("initial_agreed_pickup_day", "initial_agreed_return_day"):
                if commitment.get(key) is None:
                    commitment.pop(key, None)
            if commitment.get("commitment_version") == 1:
                commitment.pop("commitment_version", None)
            if not commitment.get("contingency_inquiry_used"):
                commitment.pop("contingency_inquiry_used", None)
            if not commitment.get("version_recovery_offer_ids_by_line"):
                commitment.pop("version_recovery_offer_ids_by_line", None)
            for key in (
                "weather_review_pickup_day",
                "weather_review_baseline_regime",
                "weather_review_max_adversity",
                "natural_cancellation_day",
            ):
                if commitment.get(key) is None:
                    commitment.pop(key, None)
            if not commitment.get("weather_reviewed_regimes"):
                commitment.pop("weather_reviewed_regimes", None)
        for owner in data.get("reservation_requests", []) + data.get("reservation_commitments", []):
            for line in owner.get("lines", []):
                if line.get("locked_line_daily_rent") is None:
                    line.pop("locked_line_daily_rent", None)
        for change in data.get("reservation_change_requests", []):
            if change.get("original_dates_still_possible") is None:
                change.pop("original_dates_still_possible", None)
            if change.get("evaluated_commitment_version") is None:
                change.pop("evaluated_commitment_version", None)
            if not change.get("weather_forecast_snapshot_by_day"):
                change.pop("weather_forecast_snapshot_by_day", None)
        if not data.get("warehouse_capacity"):
            data.pop("warehouse_capacity", None)
        if not data.get("durable_acquisition_budget"):
            data.pop("durable_acquisition_budget", None)
        if not data.get("supplier_trials"):
            data.pop("supplier_trials", None)
            data.pop("next_supplier_trial_id", None)
            data.pop("cumulative_supplier_trial_handling_fees", None)
            data.pop("cumulative_supplier_trial_usage_fees", None)
        if not data.get("reservation_contingency_inquiries"):
            data.pop("reservation_contingency_inquiries", None)
            data.pop("next_reservation_contingency_inquiry_id", None)
            data.pop("cumulative_reservation_contingency_inquiries", None)
        if not data.get("reservation_contingency_reports"):
            data.pop("reservation_contingency_reports", None)
            data.pop("next_reservation_contingency_report_id", None)
        if not data.get("cumulative_reservation_contingencies_applied"):
            data.pop("cumulative_reservation_contingencies_applied", None)
        if not data.get("cumulative_reservation_contingency_customer_cancellations"):
            data.pop(
                "cumulative_reservation_contingency_customer_cancellations", None
            )
        for asset in data["assets"]:
            if asset.get("disposed_day") is None:
                asset.pop("disposed_day", None)
            for key in (
                "durable_capital_basis_cents",
                "durable_capital_source_order_id",
                "durable_capital_source_item_index",
                "durable_capital_source_unit_index",
                "capital_exit_day",
                "capital_exit_cause",
                "capital_release_effective_day",
            ):
                if asset.get(key) is None:
                    asset.pop(key, None)
        for key in (
            "cumulative_rental_capital_holding_cost",
            "cumulative_rental_space_holding_cost",
            "cumulative_reservation_staging_actions",
            "cumulative_reservation_staged_fallbacks",
            "cumulative_capacity_policy_rejections",
        ):
            if data.get(key) == 0.0:
                data.pop(key, None)
        if not data.get("walk_in_capacity_limits"):
            data.pop("walk_in_capacity_limits", None)
        profile_line_defaults = {
            "service_key": None,
            "service_units": None,
            "public_hard_requirements": {},
            "public_preferences": [],
            "eligible_profile_ids": [],
            "eligible_profile_family_ids": [],
            "promise_profile_id": None,
            "profile_family_id": None,
            "profile_schema_version": None,
            "capacity_per_asset": None,
            "asset_quantity": None,
            "staged_offer_id": None,
        }
        pairwise_line_defaults = {
            "submitted_quote_options": [],
            "selected_quote_option_id": None,
        }
        recovery_line_defaults = {
            "authorized_line_rent": None,
            "authorized_security_deposit": None,
            "recovery_price_credit": 0.0,
            "recovery_deposit_credit": 0.0,
            "recovery_source_plan_id": None,
        }
        profile_request_defaults = {
            "reservation_contract_mode": "exact_candidates_v15",
            "catalog_snapshot_sha256": None,
            "promise_profile_schema_version": None,
            "profile_deposit_policy_id": None,
            "inquiry_public_envelope_by_line": {},
            "inquiry_profile_member_ids_by_line": {},
            "inquiry_profile_price_ceiling_by_line": {},
            "inquiry_profile_deposit_by_line": {},
        }
        complete_contract_request_defaults = {
            "submitted_contract_options": [],
            "selected_contract_option_id": None,
            "reported_contract_assessments": [],
            "feedback_response_available": None,
            "contract_feedback_audit": {},
        }
        factorized_plan_request_defaults = {
            "submitted_line_options": [],
            "customer_authorized_plans": [],
            "customer_preferred_plan_id": None,
            "manager_confirmed_plan_id": None,
            "plan_confirmation_origin": None,
            "manager_confirmation_day": None,
            "internal_plan_count": 0,
            "acceptable_plan_count": 0,
            "reported_decline_feedback": {},
            "plan_set_feedback_audit": {},
        }
        for request in data.get("reservation_requests", []):
            # V18 adds a hidden request budget. Keep every V17/V16 serialized
            # request and frozen ChoiceSession byte-shape unchanged when the
            # additive value is its legacy neutral default.
            if request.get("request_budget_multiplier") == 1.0:
                request.pop("request_budget_multiplier", None)
            for context in dict(
                request.get("quote_context_by_line") or {}
            ).values():
                if context.get("request_budget_multiplier") == 1.0:
                    context.pop("request_budget_multiplier", None)
            exact = (
                request.get("reservation_contract_mode", "exact_candidates_v15")
                == "exact_candidates_v15"
            )
            if exact:
                for key, default in profile_request_defaults.items():
                    if request.get(key) == default:
                        request.pop(key, None)
            for key, default in complete_contract_request_defaults.items():
                if request.get(key) == default:
                    request.pop(key, None)
            for key, default in factorized_plan_request_defaults.items():
                if request.get(key) == default:
                    request.pop(key, None)
            for line in request.get("lines", []):
                if exact:
                    for key, default in profile_line_defaults.items():
                        if line.get(key) == default:
                            line.pop(key, None)
                if (
                    request.get(
                        "reservation_contract_mode", "exact_candidates_v15"
                    )
                    != "adaptive_pairwise_exact_quotes_v1"
                ):
                    for key, default in pairwise_line_defaults.items():
                        if line.get(key) == default:
                            line.pop(key, None)
                for key, default in recovery_line_defaults.items():
                    if line.get(key) == default:
                        line.pop(key, None)
        for change in data.get("reservation_change_requests", []):
            for key in (
                "previous_pickup_day", "previous_return_day",
                "previous_total_locked_rent",
            ):
                if change.get(key) is None:
                    change.pop(key, None)
        for commitment in data.get("reservation_commitments", []):
            for key, default in {
                "preauthorized_recovery_plans": [],
                "selected_recovery_plan_id": None,
                "recovery_path": None,
                "original_confirmed_plan_id": None,
                "original_total_locked_rent": None,
                "original_total_security_deposit": None,
            }.items():
                if commitment.get(key) == default:
                    commitment.pop(key, None)
            for line in commitment.get("lines", []):
                if not line.get("promise_profile_id"):
                    for key, default in profile_line_defaults.items():
                        if line.get(key) == default:
                            line.pop(key, None)
                for key, default in pairwise_line_defaults.items():
                    if line.get(key) == default:
                        line.pop(key, None)
                for key, default in recovery_line_defaults.items():
                    if line.get(key) == default:
                        line.pop(key, None)
        for block in data.get("reservation_calendar_blocks", []):
            if not block.get("promise_profile_id"):
                for key in (
                    "promise_profile_id",
                    "profile_schema_version",
                    "service_units",
                    "capacity_per_asset",
                    "staged_offer_id",
                ):
                    if block.get(key) is None:
                        block.pop(key, None)
        # Keep the frozen V12 wire shape byte-stable while allowing V13 states
        # to persist only lifecycle values that differ from migration defaults.
        for asset in data.get("assets", []):
            defaults = {
                "condition_ratio": 1.0,
                "maintenance_wear_anchor": 0.0,
                "pending_repair_cost": None,
                "repair_target_condition_ratio": None,
                "ownership_type": "owned",
                "supplier_trial_id": None,
                "durable_capital_basis_cents": None,
                "durable_capital_source_order_id": None,
                "durable_capital_source_item_index": None,
                "durable_capital_source_unit_index": None,
                "capital_exit_day": None,
                "capital_exit_cause": None,
                "capital_release_effective_day": None,
            }
            for key, default in defaults.items():
                if asset.get(key) == default:
                    asset.pop(key, None)
        for order in data.get("orders", []):
            for key, default in {
                "procurement_mode": "owned",
                "supplier_trial_id": None,
                "supplier_trial_handling_fee": 0.0,
            }.items():
                if order.get(key) == default:
                    order.pop(key, None)
        for contract in data.get("contracts", []):
            for key in (
                "damage_event_uniform",
                "damage_severity_uniform",
                "damage_condition_loss",
                "damage_wear_at_start",
                "damage_condition_at_start",
                "damage_maintenance_exposure_at_start",
            ):
                if contract.get(key) is None:
                    contract.pop(key, None)
            if contract.get("supplier_trial_id") is None:
                contract.pop("supplier_trial_id", None)
            if contract.get("rental_episode_id") is None:
                contract.pop("rental_episode_id",None)
            if contract.get("supplier_trial_usage_fee") == 0.0:
                contract.pop("supplier_trial_usage_fee", None)
        return data

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WorldState":
        data = dict(value)
        data.setdefault("unearned_revenue_liability", 0.0)

        def reservation_line(raw: dict[str, Any]) -> ReservationLine:
            # V9 persisted an inquiry-time walk-in reference. V10 removes the
            # construct completely; accepting the legacy key here keeps old
            # completed runs analyzable without carrying it into active state.
            migrated = dict(raw)
            migrated.pop("reference_daily_price", None)
            return ReservationLine(**migrated)

        def reservation_request(raw: dict[str, Any]) -> ReservationRequest:
            migrated = dict(raw)
            legacy_offer_ids = migrated.pop("latent_offer_ids_by_line", None)
            legacy_ceilings = migrated.pop("latent_price_ceiling_by_line", None)
            if "inquiry_latent_offer_ids_by_line" not in migrated:
                migrated["inquiry_latent_offer_ids_by_line"] = {
                    str(line_id): list(offer_ids)
                    for line_id, offer_ids in dict(legacy_offer_ids or {}).items()
                }
                migrated["inquiry_latent_price_ceiling_by_line"] = {
                    str(line_id): {
                        str(offer_id): float(ceiling)
                        for offer_id, ceiling in dict(ceilings).items()
                    }
                    for line_id, ceilings in dict(legacy_ceilings or {}).items()
                }
                migrated["latent_snapshot_provenance"] = (
                    "legacy_post_quote_mixed"
                    if migrated.get("commitment_id")
                    else "legacy_inquiry_snapshot"
                )
            fallback = dict(migrated.get("fallback_offer_ids_by_line") or {})
            request_lines = list(migrated.get("lines") or [])
            migrated.setdefault(
                "authorized_offer_ids_by_line",
                {
                    str(line.get("line_id", "")): list(
                        line.get("authorized_offer_ids") or []
                    )
                    for line in request_lines
                    if line.get("authorized_offer_ids")
                },
            )
            inquiry_ceilings = dict(
                migrated.get("inquiry_latent_price_ceiling_by_line") or {}
            )
            migrated.setdefault(
                "authorized_price_ceiling_by_line",
                {
                    str(line_id): {
                        str(offer_id): float(
                            dict(inquiry_ceilings.get(line_id) or {}).get(
                                offer_id, 0.0
                            )
                        )
                        for offer_id in offer_ids
                    }
                    for line_id, offer_ids in dict(
                        migrated.get("authorized_offer_ids_by_line") or {}
                    ).items()
                },
            )
            migrated.setdefault(
                "recovery_offer_ids_by_line",
                {
                    str(line_id): list(offer_ids)
                    for line_id, offer_ids in fallback.items()
                },
            )
            migrated.setdefault(
                "recovery_price_ceiling_by_line",
                {
                    str(line_id): {
                        str(offer_id): float(
                            dict(inquiry_ceilings.get(line_id) or {}).get(
                                offer_id, 0.0
                            )
                        )
                        for offer_id in offer_ids
                    }
                    for line_id, offer_ids in fallback.items()
                },
            )
            contexts = dict(migrated.get("quote_context_by_line") or {})
            first_context = next(iter(contexts.values()), {})
            migrated.setdefault(
                "customer_segment",
                str(first_context.get("customer_segment", "balanced")),
            )
            migrated.setdefault(
                "shared_price_beta", float(first_context.get("price_beta", 0.0))
            )
            migrated.setdefault(
                "storage_burden_base",
                float(first_context.get("storage_burden", 0.5)),
            )
            migrated.setdefault(
                "future_use_expectation_base",
                float(first_context.get("future_use_expectation", 0.0)),
            )
            migrated.setdefault(
                "return_burden_multiplier",
                float(first_context.get("return_burden_multiplier", 1.0)),
            )
            migrated.setdefault(
                "request_budget_multiplier",
                float(first_context.get("request_budget_multiplier", 1.0)),
            )
            return ReservationRequest(
                **{
                    **migrated,
                    "lines": [
                        line
                        if isinstance(line, ReservationLine)
                        else reservation_line(line)
                        for line in migrated.get("lines", [])
                    ],
                }
            )

        data["assets"] = [RentalAsset(**row) for row in data.get("assets", [])]
        data["contracts"] = [RentalContract(**row) for row in data.get("contracts", [])]
        data["orders"] = [PurchaseOrder(**row) for row in data.get("orders", [])]
        data["supplier_trials"] = [
            row
            if isinstance(row, SupplierTrialRecord)
            else SupplierTrialRecord(**row)
            for row in data.get("supplier_trials", [])
        ]
        data["cleaning_queue"] = [
            row if isinstance(row, CleaningJob) else CleaningJob(**row)
            for row in data.get("cleaning_queue", [])
        ]
        data["reservation_requests"] = [
            row
            if isinstance(row, ReservationRequest)
            else reservation_request(row)
            for row in data.get("reservation_requests", [])
        ]
        data["reservation_commitments"] = [
            row
            if isinstance(row, ReservationCommitment)
            else ReservationCommitment(
                **{
                    **row,
                    "lines": [
                        line
                        if isinstance(line, ReservationLine)
                        else reservation_line(line)
                        for line in row.get("lines", [])
                    ],
                    "assignments": [
                        assignment
                        if isinstance(assignment, ReservationAssignment)
                        else ReservationAssignment(**assignment)
                        for assignment in row.get("assignments", [])
                    ],
                }
            )
            for row in data.get("reservation_commitments", [])
        ]
        data["reservation_events"] = [
            row if isinstance(row, ReservationEvent) else ReservationEvent(**row)
            for row in data.get("reservation_events", [])
        ]
        data["reservation_change_requests"] = [
            row
            if isinstance(row, ReservationChangeRequest)
            else ReservationChangeRequest(**row)
            for row in data.get("reservation_change_requests", [])
        ]
        data["reservation_contingency_inquiries"] = [
            row
            if isinstance(row, ReservationContingencyInquiry)
            else ReservationContingencyInquiry(**row)
            for row in data.get("reservation_contingency_inquiries", [])
        ]
        data["reservation_contingency_reports"] = [
            row
            if isinstance(row, ReservationContingencyReport)
            else ReservationContingencyReport(**row)
            for row in data.get("reservation_contingency_reports", [])
        ]
        data["reservation_calendar_blocks"] = [
            row if isinstance(row, CalendarBlock) else CalendarBlock(**row)
            for row in data.get("reservation_calendar_blocks", [])
        ]
        data["walk_in_capacity_limits"] = [
            row
            if isinstance(row, WalkInCapacityLimit)
            else WalkInCapacityLimit(**row)
            for row in data.get("walk_in_capacity_limits", [])
        ]
        legacy_memory = data.pop("memory", {})
        raw_documents = data.get("memory_documents")
        if raw_documents is None:
            raw_documents = {
                str(name): {
                    "name": str(name),
                    "description": "",
                    "content": str(content),
                    "created_day": 0,
                    "updated_day": 0,
                }
                for name, content in dict(legacy_memory).items()
            }
        data["memory_documents"] = {
            str(name): (
                document
                if isinstance(document, MemoryDocument)
                else MemoryDocument(**document)
            )
            for name, document in dict(raw_documents).items()
        }
        return cls(**data)
