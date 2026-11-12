from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "runtime_data"
ASSORTMENT_MODES = ("rental_only", "rental_and_retail")


@dataclass(slots=True)
class OutfitterConfig:
    data_dir: str = str(DEFAULT_DATA_DIR)
    runs_dir: str = "runs/outfitter"
    seed: int = 42
    max_days: int = 120
    weather_start_day: int = 1
    initial_cash: float = 10_000.0
    daily_fee: float = 60.0
    rental_holding_rate: float = 0.0005
    rental_idle_holding_rate: float = 0.0
    retail_holding_rate: float = 0.001
    # V35 projects one shared sealed catalog into one of two benchmark tracks.
    # The legacy default preserves historical full-store configurations.
    assortment_mode: str = "rental_and_retail"
    # V31 only. None is an explicitly uncapped calibration control, not a
    # purchasable warehouse upgrade. The episode freezes this value at reset.
    warehouse_capacity_limit: int | None = None
    # Versioned durable-capital control. V36-V39 interpret a finite value as a
    # non-replenishing acquisition budget; V40 interprets it as additional
    # headroom above initial owned durable basis. None is uncapped calibration.
    durable_acquisition_budget: float | None = None
    # V37 only. Disabled fields are omitted from legacy serialized configs.
    supplier_trial_enabled: bool = False
    supplier_trial_max_accepted: int = 4
    supplier_trial_max_active: int = 1
    supplier_trial_max_contract_starts: int = 3
    supplier_trial_window_days: int = 21
    supplier_trial_max_unit_cost: float = 1500.0
    supplier_trial_handling_fee: float = 10.0
    supplier_trial_usage_fee_multiplier: float = 1.0
    liquidation_depth_enabled: bool = False
    liquidation_depth_window_days: int = 7
    liquidation_depth_capacity_fraction: float = 0.20
    liquidation_depth_end_multiple: float = 4.0
    liquidation_depth_floor_rate: float = 0.55
    reservations_enabled: bool = False
    reservation_contract_mode: str = "exact_candidates_v15"
    reservation_profile_deposit_policy: str | None = None
    reservation_share: float = 0.65
    reservation_booking_lead_min_days: int = 1
    reservation_booking_lead_max_days: int = 14
    reservation_prepayment_rate: float = 0.25
    reservation_authorization_mode: str = "credible_assortment"
    reservation_authorized_alternative_max: int = 2
    reservation_disruptions_enabled: bool = False
    # V32 independent, opt-in mechanisms. Legacy configurations remain inert.
    reservation_reschedule_outcomes_enabled: bool = False
    reservation_date_flexibility_enabled: bool = False
    reservation_service_offers_enabled: bool = False
    # V38 only. Disabled fields are omitted from legacy serialized configs.
    reservation_contingency_enabled: bool = False
    late_return_probability_after_extension_decline: float = 0.12
    reservation_failure_compensation_rate: float = 0.50
    # V29 can expose a bounded service-reputation experiment.  Formal V29
    # configurations leave it disabled; the policy parameters are frozen in
    # runtime metadata rather than supplied by the evaluated manager.
    reservation_service_reputation_enabled: bool = False
    weather_wear_enabled: bool = False
    reservation_bundles_enabled: bool = False
    optional_line_missing_credit_rate: float = 1.0
    preventive_maintenance_enabled: bool = False
    lifecycle_economics_enabled: bool = False
    lifecycle_mechanism_ablation: bool = False
    lifecycle_holding_enabled: bool = True
    maintenance_interval_days: int = 28
    maintenance_due_window_days: int = 3
    maintenance_duration_days: int = 1
    maintenance_cost_rate: float = 0.02
    runoff_days: int = 0
    emergency_credit_limit: float = 0.0
    emergency_credit_fee_rate: float = 0.25
    arrival_scale_multiplier: float = 1.0
    demand_world_id: str = "base"
    supplier_delivery_min_days: int = 2
    supplier_delivery_max_days: int = 7
    supplier_delivery_early_days: int = 1
    supplier_delivery_late_days: int = 2
    # Retained for loading older configs; supplier-specific delivery profiles
    # supersede this scalar in the current environment.
    delivery_days: int = 2
    model: str = "deepseek-v4-flash"
    base_url: str | None = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 1.0
    max_output_tokens: int = 8192
    request_timeout_seconds: float = 1200.0
    max_retries: int = 8
    request_rate_limit_per_minute: int | None = None
    rate_limit_retry_seconds: float = 60.0
    reasoning_effort: str | None = "max"
    thinking_type: str | None = None
    provider_extra_body: dict[str, Any] = field(default_factory=dict)
    parallel_tool_calls: bool = True
    max_read_only_batch_size: int = 5
    max_model_calls: int = 10_000
    max_iterations_per_day: int = 50
    context_budget_chars: int = 100_000
    input_token_cost_per_million: float = 0.0
    output_token_cost_per_million: float = 100.0
    charge_model_usage_to_cash: bool = True

    def validate(self) -> None:
        if self.assortment_mode not in ASSORTMENT_MODES:
            raise ValueError(
                "assortment_mode must be rental_only or rental_and_retail"
            )
        if self.warehouse_capacity_limit is not None and (
            type(self.warehouse_capacity_limit) is not int or self.warehouse_capacity_limit <= 0
        ):
            raise ValueError("warehouse_capacity_limit must be a positive integer or null")
        if self.durable_acquisition_budget is not None:
            budget = self.durable_acquisition_budget
            if (
                isinstance(budget, bool)
                or not isinstance(budget, (int, float))
                or not math.isfinite(float(budget))
                or float(budget) < 0
                or round(float(budget), 2) != float(budget)
            ):
                raise ValueError(
                    "durable_acquisition_budget must be a nonnegative cent-exact amount or null"
                )
        if type(self.supplier_trial_enabled) is not bool:
            raise ValueError("supplier_trial_enabled must be boolean")
        for name in (
            "supplier_trial_max_accepted",
            "supplier_trial_max_active",
            "supplier_trial_max_contract_starts",
            "supplier_trial_window_days",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "supplier_trial_max_unit_cost",
            "supplier_trial_handling_fee",
            "supplier_trial_usage_fee_multiplier",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        data = Path(self.data_dir).expanduser()
        expected = {
            "catalog.jsonl",
            "initial_assets.jsonl",
            "initial_retail_inventory.jsonl",
            "weather.jsonl",
            "metadata.json",
        }
        missing = sorted(name for name in expected if not (data / name).is_file())
        if missing:
            raise FileNotFoundError(f"runtime data is missing files: {missing}")
        if not 1 <= self.max_days <= 365:
            raise ValueError("max_days must be in [1, 365]")
        if not 1 <= self.weather_start_day <= 365:
            raise ValueError("weather_start_day must be in [1, 365]")
        if not 0 <= int(self.runoff_days) <= 14:
            raise ValueError("runoff_days must be in [0, 14]")
        if self.weather_start_day + self.max_days + self.runoff_days - 1 > 365:
            raise ValueError(
                "weather_start_day + max_days + runoff_days must fit within 365 weather days"
            )
        if self.initial_cash <= 0 or self.daily_fee < 0 or self.delivery_days < 1:
            raise ValueError("cash/delivery settings are invalid")
        for name in (
            "rental_holding_rate",
            "rental_idle_holding_rate",
            "retail_holding_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0 or value > 0.05:
                raise ValueError(f"{name} must be finite and within [0, 0.05]")
        if not 1 <= int(self.liquidation_depth_window_days) <= 30:
            raise ValueError("liquidation_depth_window_days must be in [1, 30]")
        if not math.isfinite(float(self.liquidation_depth_capacity_fraction)) or not (
            0.01 <= float(self.liquidation_depth_capacity_fraction) <= 1.0
        ):
            raise ValueError(
                "liquidation_depth_capacity_fraction must be in [0.01, 1.0]"
            )
        if not math.isfinite(float(self.liquidation_depth_end_multiple)) or (
            float(self.liquidation_depth_end_multiple) <= 1.0
        ):
            raise ValueError("liquidation_depth_end_multiple must exceed 1")
        if not math.isfinite(float(self.liquidation_depth_floor_rate)) or not (
            0.0 < float(self.liquidation_depth_floor_rate) <= 1.0
        ):
            raise ValueError("liquidation_depth_floor_rate must be in (0, 1]")
        if not math.isfinite(float(self.reservation_share)) or not (
            0.0 <= float(self.reservation_share) <= 1.0
        ):
            raise ValueError("reservation_share must be in [0, 1]")
        if self.reservation_contract_mode not in {
            "exact_candidates_v15",
            "promise_profiles_v1",
            "adaptive_pairwise_exact_quotes_v1",
            "complete_contract_options_v1",
            "factorized_line_options_plan_set_v1",
        }:
            raise ValueError("invalid reservation_contract_mode")
        profile_policies = {
            "canonical_reference_v1",
            "flat20_v1",
            "flat40_v1",
        }
        profile_contract_modes = {
            "promise_profiles_v1",
            "adaptive_pairwise_exact_quotes_v1",
            "complete_contract_options_v1",
            "factorized_line_options_plan_set_v1",
        }
        if self.reservation_contract_mode in profile_contract_modes:
            if not self.reservations_enabled:
                raise ValueError(
                    "profile reservation contracts require reservations_enabled"
                )
            if self.reservation_profile_deposit_policy not in profile_policies:
                raise ValueError(
                    "profile reservation contracts require an explicit frozen profile "
                    "deposit policy"
                )
        elif self.reservation_profile_deposit_policy is not None:
            raise ValueError(
                "reservation_profile_deposit_policy is valid only in "
                "a profile reservation contract mode"
            )
        if not (
            1
            <= int(self.reservation_booking_lead_min_days)
            <= int(self.reservation_booking_lead_max_days)
            <= 14
        ):
            raise ValueError(
                "reservation booking lead bounds must satisfy 1 <= min <= max <= 14"
            )
        if not math.isfinite(float(self.reservation_prepayment_rate)) or not (
            0.0 <= float(self.reservation_prepayment_rate) <= 1.0
        ):
            raise ValueError("reservation_prepayment_rate must be in [0, 1]")
        if self.reservation_authorization_mode not in {
            "credible_assortment",
            "risk_only",
            "all_latent",
        }:
            raise ValueError("invalid reservation_authorization_mode")
        if not 0 <= int(self.reservation_authorized_alternative_max) <= 2:
            raise ValueError(
                "reservation_authorized_alternative_max must be in [0, 2]"
            )
        if self.reservation_disruptions_enabled and not self.reservations_enabled:
            raise ValueError(
                "reservation_disruptions_enabled requires reservations_enabled"
            )
        if not isinstance(self.reservation_contingency_enabled, bool):
            raise ValueError("reservation_contingency_enabled must be boolean")
        if self.reservation_contingency_enabled and not self.reservations_enabled:
            raise ValueError(
                "reservation_contingency_enabled requires reservations_enabled"
            )
        if not isinstance(self.reservation_service_reputation_enabled, bool):
            raise ValueError(
                "reservation_service_reputation_enabled must be boolean"
            )
        if self.reservation_service_reputation_enabled and not (
            self.reservations_enabled and self.reservation_disruptions_enabled
        ):
            raise ValueError(
                "reservation service reputation requires enabled reservation "
                "disruptions"
            )
        for name in (
            "late_return_probability_after_extension_decline",
            "reservation_failure_compensation_rate",
            "optional_line_missing_credit_rate",
            "maintenance_cost_rate",
            "emergency_credit_fee_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 7 <= int(self.maintenance_interval_days) <= 90:
            raise ValueError("maintenance_interval_days must be in [7, 90]")
        if not 1 <= int(self.maintenance_due_window_days) <= 14:
            raise ValueError("maintenance_due_window_days must be in [1, 14]")
        if not 1 <= int(self.maintenance_duration_days) <= 7:
            raise ValueError("maintenance_duration_days must be in [1, 7]")
        if (
            not math.isfinite(float(self.emergency_credit_limit))
            or float(self.emergency_credit_limit) < 0
        ):
            raise ValueError("emergency_credit_limit must be finite and non-negative")
        if not (
            2 <= self.supplier_delivery_min_days
            <= self.supplier_delivery_max_days
            <= 7
        ):
            raise ValueError("supplier delivery bounds must satisfy 2 <= min <= max <= 7")
        if not 0 <= self.supplier_delivery_early_days <= 1:
            raise ValueError("supplier delivery early allowance must be in [0, 1]")
        if not 0 <= self.supplier_delivery_late_days <= 2:
            raise ValueError("supplier delivery late allowance must be in [0, 2]")
        if not math.isfinite(self.arrival_scale_multiplier) or not (
            0.25 <= self.arrival_scale_multiplier <= 2.0
        ):
            raise ValueError("arrival_scale_multiplier must be in [0.25, 2.0]")
        if self.demand_world_id not in {
            "control",
            "conservative",
            "base",
            "convenience-heavy",
            "extreme-tail-stress",
        }:
            raise ValueError("invalid demand_world_id")
        if self.reasoning_effort not in {None, "high", "max"}:
            raise ValueError("reasoning_effort must be high, max, or null")
        if self.thinking_type not in {None, "enabled", "disabled"}:
            raise ValueError("thinking_type must be enabled, disabled, or null")
        if not isinstance(self.provider_extra_body, dict) or not all(
            isinstance(key, str) for key in self.provider_extra_body
        ):
            raise ValueError("provider_extra_body must be a mapping with string keys")
        conflicts = []
        if self.thinking_type is not None and "thinking" in self.provider_extra_body:
            conflicts.append("thinking")
        if (
            self.reasoning_effort is not None
            and "reasoning_effort" in self.provider_extra_body
        ):
            conflicts.append("reasoning_effort")
        if conflicts:
            joined = ", ".join(sorted(conflicts))
            raise ValueError(
                f"provider_extra_body conflicts with explicit provider settings: {joined}"
            )
        try:
            json.dumps(self.provider_extra_body, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "provider_extra_body must contain JSON-compatible values"
            ) from error
        if not 1 <= self.max_read_only_batch_size <= 10:
            raise ValueError("max_read_only_batch_size must be in [1, 10]")
        if self.max_iterations_per_day <= 0 or self.max_model_calls <= 0:
            raise ValueError("model-call limits must be positive")
        for value in (
            self.input_token_cost_per_million,
            self.output_token_cost_per_million,
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("token prices must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.warehouse_capacity_limit is None:
            result.pop("warehouse_capacity_limit", None)
        if self.durable_acquisition_budget is None:
            result.pop("durable_acquisition_budget", None)
        if not self.supplier_trial_enabled:
            for key in tuple(result):
                if key.startswith("supplier_trial_"):
                    result.pop(key, None)
        if not self.reservation_contingency_enabled:
            result.pop("reservation_contingency_enabled", None)
        return result

    @property
    def retail_enabled(self) -> bool:
        return self.assortment_mode == "rental_and_retail"

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "OutfitterConfig":
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> "OutfitterConfig":
        raw = {} if path is None else yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("config YAML root must be a mapping")
        return cls.from_dict(raw)
