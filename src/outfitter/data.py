from __future__ import annotations

from .jsonio import read_jsonl as _jsonl

import json
import hashlib
import re
import shutil
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .models import Offer
from .demand import DemandModel
from .channel_demand import ChannelDemandModel
from .lifecycle_economics import LifecycleEconomicScales
from .reference_demand import ReferenceTransactionModel
from .reservation_profiles import ReservationPromiseCatalog
from .reservation_feedback import ContractFeedbackPolicy, PlanSetFeedbackPolicy
from .v5 import (
    BENCHMARK_CALENDAR_YEAR,
    RUNTIME_FORMAT,
    V5MarketRules,
    calendar_for_source_day,
    day_type_for_source_day,
    decision_signature,
    infer_v5_offer_fields,
    legacy_2023_holiday_for_source_day,
)


V6_RUNTIME_FORMAT = "outfitter-runtime-v6-reservations"
V7_RUNTIME_FORMAT = "outfitter-runtime-v7-disjunctive-reservations"
V8_RUNTIME_FORMAT = "outfitter-runtime-v8-contract-quotes"
V9_RUNTIME_FORMAT = "outfitter-runtime-v9-explicit-contract-quotes"
V10_RUNTIME_FORMAT = "outfitter-runtime-v10-coherent-bundle-latents"
V11_RUNTIME_FORMAT = "outfitter-runtime-v11-compact-ids-pagination"
V12_RUNTIME_FORMAT = "outfitter-runtime-v12-lifecycle-closure"
V13_RUNTIME_FORMAT = "outfitter-runtime-v13-reservation-promise-profiles"

@dataclass(slots=True)
class OutfitterDataset:
    root: Path
    offers: list[Offer]
    offer_rng_identities: dict[str, str]
    initial_assets: list[dict[str, Any]]
    initial_retail: list[dict[str, Any]]
    weather: dict[int, str]
    weather_calendar: dict[int, dict[str, Any]]
    metadata: dict[str, Any]
    market_rules: V5MarketRules
    initial_prices: list[dict[str, Any]]
    demand_model: DemandModel | None = None
    reference_demand_model: ReferenceTransactionModel | None = None
    channel_demand_model: ChannelDemandModel | None = None
    lifecycle_factors: dict[str, dict[str, Any]] = field(default_factory=dict)
    lifecycle_scales: LifecycleEconomicScales | None = None
    reservation_promise_catalog: ReservationPromiseCatalog | None = None
    contract_feedback_policy: ContractFeedbackPolicy | None = None
    plan_set_feedback_policy: PlanSetFeedbackPolicy | None = None
    warehouse_capacity_weights: dict[str, int] = field(default_factory=dict)
    warehouse_capacity_weights_sha256: str | None = None

    def active_view(self, assortment_mode: str) -> "OutfitterDataset":
        """Project one sealed full catalog into the selected operating assortment.

        The source files and model objects remain shared and hash-bound.  Only
        actor-operable offers and their opening state are projected.
        """

        if assortment_mode == "rental_and_retail":
            return self
        if assortment_mode != "rental_only":
            raise ValueError(f"unsupported assortment mode: {assortment_mode}")
        active_offers = [row for row in self.offers if row.asset_kind == "durable"]
        active_ids = {row.offer_id for row in active_offers}
        if not active_offers or any(
            str(row.get("offer_id")) not in active_ids for row in self.initial_assets
        ):
            raise ValueError("rental-only projection has invalid durable opening assets")
        return replace(
            self,
            offers=active_offers,
            offer_rng_identities={
                key: value
                for key, value in self.offer_rng_identities.items()
                if key in active_ids
            },
            initial_retail=[],
            initial_prices=[
                row for row in self.initial_prices
                if str(row.get("offer_id")) in active_ids
            ],
        )

    @classmethod
    def load(cls, root: str | Path) -> "OutfitterDataset":
        path = Path(root).expanduser().resolve()
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        runtime_format = metadata.get("format")
        if runtime_format not in {
            "outfitter-runtime-v1",
            "outfitter-runtime-v2",
            RUNTIME_FORMAT,
            V6_RUNTIME_FORMAT,
            V7_RUNTIME_FORMAT,
            V8_RUNTIME_FORMAT,
            V9_RUNTIME_FORMAT,
            V10_RUNTIME_FORMAT,
            V11_RUNTIME_FORMAT,
            V12_RUNTIME_FORMAT,
            V13_RUNTIME_FORMAT,
        }:
            raise ValueError("unsupported Outfitter runtime data format")
        if runtime_format in {
            RUNTIME_FORMAT,
            V6_RUNTIME_FORMAT,
            V7_RUNTIME_FORMAT,
            V8_RUNTIME_FORMAT,
            V9_RUNTIME_FORMAT,
            V10_RUNTIME_FORMAT,
            V11_RUNTIME_FORMAT,
            V12_RUNTIME_FORMAT,
            V13_RUNTIME_FORMAT,
        }:
            expected_hashes = metadata.get("content_sha256")
            if not isinstance(expected_hashes, dict) or not expected_hashes:
                raise ValueError("v5 runtime metadata must contain content_sha256")
            for name, expected in sorted(expected_hashes.items()):
                target = path / str(name)
                if not target.is_file():
                    raise ValueError(f"v5 runtime hash target is missing: {name}")
                actual = hashlib.sha256(target.read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f"v5 runtime content hash mismatch: {name}")
        offers = [Offer.from_dict(row) for row in _jsonl(path / "catalog.jsonl")]
        if len({row.offer_id for row in offers}) != len(offers):
            raise ValueError("runtime catalog contains duplicate offer IDs")
        offer_rng_identities = {
            offer.offer_id: offer.offer_id for offer in offers
        }
        if runtime_format in {
            V11_RUNTIME_FORMAT,
            V12_RUNTIME_FORMAT,
            V13_RUNTIME_FORMAT,
        }:
            registry_path = path / "offer_id_registry.json"
            if not registry_path.is_file():
                raise ValueError("v11 runtime is missing offer_id_registry.json")
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            if registry.get("format") != "outfitter-offer-id-registry-v1":
                raise ValueError("unsupported offer ID registry format")
            entries = list(registry.get("entries") or [])
            registry_ids = [str(row.get("offer_id") or "") for row in entries]
            identities = [
                str(row.get("listing_identity_sha256") or "") for row in entries
            ]
            first_seen_ids = [
                str(row.get("first_seen_source_offer_id") or "")
                for row in entries
            ]
            if (
                len(set(registry_ids)) != len(registry_ids)
                or len(set(identities)) != len(identities)
                or any(
                    re.fullmatch(r"offer-[0-9]{3,}", value) is None
                    for value in registry_ids
                )
                or any(not value for value in identities)
                or any(not value for value in first_seen_ids)
            ):
                raise ValueError(
                    "offer ID registry contains invalid or duplicate entries"
                )
            active_ids = {
                str(row["offer_id"])
                for row in entries
                if bool(row.get("active"))
            }
            if active_ids != {offer.offer_id for offer in offers}:
                raise ValueError(
                    "active offer ID registry entries do not match catalog"
                )
            offer_rng_identities = {
                str(row["offer_id"]): str(row["first_seen_source_offer_id"])
                for row in entries
                if bool(row.get("active"))
            }
        if runtime_format in {
            RUNTIME_FORMAT,
            V6_RUNTIME_FORMAT,
            V7_RUNTIME_FORMAT,
            V8_RUNTIME_FORMAT,
            V9_RUNTIME_FORMAT,
            V10_RUNTIME_FORMAT,
            V11_RUNTIME_FORMAT,
            V12_RUNTIME_FORMAT,
            V13_RUNTIME_FORMAT,
        }:
            if not 1 <= len(offers) <= 110:
                raise ValueError("v5 runtime catalog must contain between 1 and 110 offers")
        elif len(offers) != 48:
            raise ValueError("legacy runtime catalog must contain 48 unique offers")
        if runtime_format == "outfitter-runtime-v2":
            for offer in offers:
                if offer.asset_kind != "consumable":
                    continue
                if (
                    not isinstance(offer.consumer_unit, dict)
                    or not str(offer.consumer_unit.get("label") or "").strip()
                    or not offer.product_subtype
                    or len(offer.fulfills_service_keys) != 1
                    or offer.reference_retail_price is None
                    or offer.reference_retail_price <= 0
                    or offer.quality_tier not in {"economy", "standard", "premium"}
                ):
                    raise ValueError(
                        f"retail offer {offer.offer_id} is missing v2 consumer semantics"
                    )
        for offer in offers:
            if not offer.choice_group or not offer.decision_signature:
                group, decision, descriptive, tier = infer_v5_offer_fields(
                    category_id=offer.category_id,
                    service_line=offer.service_line,
                    name=offer.name,
                    product_subtype=offer.product_subtype,
                    quality_tier=offer.quality_tier,
                    evidence=offer.evidence,
                )
                offer.choice_group = group
                offer.decision_attributes = decision
                offer.descriptive_attributes = descriptive
                offer.perceived_quality_tier = tier
                offer.decision_signature = decision_signature(group, decision, tier)
            if offer.perceived_quality_tier not in {
                "economy",
                "standard",
                "premium",
                "unknown",
            }:
                raise ValueError(f"offer {offer.offer_id} has invalid perceived quality tier")
        weather_rows = _jsonl(path / "weather.jsonl")
        fallback_start = date(BENCHMARK_CALENDAR_YEAR, 1, 1)
        calendar: dict[int, dict[str, Any]] = {}
        for row in weather_rows:
            day = int(row["day"])
            raw_date = str(row.get("date") or (fallback_start + timedelta(days=day - 1)).isoformat())
            parsed = date.fromisoformat(raw_date)
            fallback_calendar = (
                calendar_for_source_day(day)
                if parsed.year == BENCHMARK_CALENDAR_YEAR
                else {}
            )
            legacy_holiday_id, legacy_holiday_name = (
                legacy_2023_holiday_for_source_day(day)
                if parsed.year == 2023
                else (None, None)
            )
            calendar[day] = {
                "date": parsed.isoformat(),
                "month_day": parsed.strftime("%m-%d"),
                "weekday": row.get("weekday") or parsed.strftime("%A"),
                "is_weekend": bool(row.get("is_weekend", parsed.weekday() >= 5)),
                "day_type": row.get("day_type")
                or day_type_for_source_day(
                    day,
                    is_weekend=bool(row.get("is_weekend", parsed.weekday() >= 5)),
                ),
                "holiday_id": row.get("holiday_id")
                or fallback_calendar.get("holiday_id")
                or legacy_holiday_id,
                "holiday_name": row.get("holiday_name")
                or fallback_calendar.get("holiday_name")
                or legacy_holiday_name,
                "temp_mean_c": row.get("temp_mean_c"),
                "night_low_c": row.get("night_low_c"),
                "day_high_c": row.get("day_high_c"),
            }
        rules_path = path / "market_rules.json"
        market_rules = (
            V5MarketRules.from_dict(json.loads(rules_path.read_text(encoding="utf-8")))
            if rules_path.is_file()
            else V5MarketRules()
        )
        market_rules.validate()
        initial_prices_path = path / "initial_prices.jsonl"
        demand_model_path = path / "demand_model.json"
        reference_demand_path = path / "reference_transaction_model.json"
        channel_demand_path = path / "channel_demand_model.json"
        lifecycle_factor_path = path / "lifecycle_factors.jsonl"
        lifecycle_scales_path = path / "lifecycle_economic_scales.json"
        reservation_profile_path = path / "reservation_promise_profiles.json"
        contract_feedback_path = path / "reservation_contract_feedback.json"
        plan_set_feedback_path = path / "reservation_plan_set_feedback.json"
        lifecycle_factors: dict[str, dict[str, Any]] = {}
        lifecycle_scales: LifecycleEconomicScales | None = None
        reservation_promise_catalog: ReservationPromiseCatalog | None = None
        contract_feedback_policy: ContractFeedbackPolicy | None = None
        plan_set_feedback_policy: PlanSetFeedbackPolicy | None = None
        if runtime_format in {V12_RUNTIME_FORMAT, V13_RUNTIME_FORMAT}:
            if not lifecycle_factor_path.is_file() or not lifecycle_scales_path.is_file():
                raise ValueError("v12 lifecycle runtime is missing lifecycle factor files")
            lifecycle_rows = _jsonl(lifecycle_factor_path)
            lifecycle_factors = {
                str(row.get("offer_id") or ""): dict(row) for row in lifecycle_rows
            }
            durable_ids = {
                offer.offer_id for offer in offers if offer.asset_kind == "durable"
            }
            if (
                not lifecycle_factors
                or "" in lifecycle_factors
                or len(lifecycle_factors) != len(lifecycle_rows)
                or set(lifecycle_factors) != durable_ids
            ):
                raise ValueError(
                    "lifecycle factor coverage must exactly match durable offers"
                )
            lifecycle_scales = LifecycleEconomicScales.from_dict(
                json.loads(lifecycle_scales_path.read_text(encoding="utf-8"))
            )
        if runtime_format == V13_RUNTIME_FORMAT:
            if not reservation_profile_path.is_file():
                raise ValueError(
                    "v13 reservation promise runtime is missing its profile catalog"
                )
            reservation_promise_catalog = ReservationPromiseCatalog.from_dict(
                json.loads(reservation_profile_path.read_text(encoding="utf-8"))
            )
            catalog_sha256 = hashlib.sha256(
                (path / "catalog.jsonl").read_bytes()
            ).hexdigest()
            if reservation_promise_catalog.source_catalog_sha256 != catalog_sha256:
                raise ValueError(
                    "reservation promise profile source catalog hash mismatch"
                )
            reservation_promise_catalog.validate(offers)
            if contract_feedback_path.is_file():
                contract_feedback_policy = ContractFeedbackPolicy.from_dict(
                    json.loads(
                        contract_feedback_path.read_text(encoding="utf-8")
                    )
                )
            if plan_set_feedback_path.is_file():
                plan_set_feedback_policy = PlanSetFeedbackPolicy.from_dict(
                    json.loads(
                        plan_set_feedback_path.read_text(encoding="utf-8")
                    )
                )
        warehouse_weights = {}
        warehouse_hash = None
        if metadata.get("warehouse_capacity_policy") is not None:
            from .warehouse_capacity import WAREHOUSE_POLICY, validate_weights
            if metadata["warehouse_capacity_policy"] != WAREHOUSE_POLICY:
                raise ValueError("unsupported warehouse capacity policy")
            weights_path = path / "warehouse_capacity_weights.json"
            warehouse_weights = validate_weights(
                json.loads(weights_path.read_text(encoding="utf-8")), path / "catalog.jsonl",
                {offer.offer_id for offer in offers if offer.asset_kind == "durable"},
            )
            warehouse_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
            if metadata.get("content_sha256", {}).get(weights_path.name) != warehouse_hash:
                raise ValueError("warehouse weights must be included in frozen runtime hashes")
        return cls(
            warehouse_capacity_weights=warehouse_weights,
            warehouse_capacity_weights_sha256=warehouse_hash,
            root=path,
            offers=offers,
            offer_rng_identities=offer_rng_identities,
            initial_assets=_jsonl(path / "initial_assets.jsonl"),
            initial_retail=_jsonl(path / "initial_retail_inventory.jsonl"),
            weather={
                int(row["day"]): (
                    str(row["condition"])
                    if str(row["condition"]) in {"sunny", "cloudy", "rainy", "snowy"}
                    else "cloudy"
                )
                for row in weather_rows
            },
            weather_calendar=calendar,
            metadata=metadata,
            market_rules=market_rules,
            initial_prices=(
                _jsonl(initial_prices_path) if initial_prices_path.is_file() else []
            ),
            demand_model=(
                DemandModel.from_dict(
                    json.loads(demand_model_path.read_text(encoding="utf-8"))
                )
                if demand_model_path.is_file()
                else None
            ),
            reference_demand_model=(
                ReferenceTransactionModel.from_dict(
                    json.loads(reference_demand_path.read_text(encoding="utf-8"))
                )
                if reference_demand_path.is_file()
                else None
            ),
            channel_demand_model=(
                ChannelDemandModel.from_dict(
                    json.loads(channel_demand_path.read_text(encoding="utf-8"))
                )
                if channel_demand_path.is_file()
                else None
            ),
            lifecycle_factors=lifecycle_factors,
            lifecycle_scales=lifecycle_scales,
            reservation_promise_catalog=reservation_promise_catalog,
            contract_feedback_policy=contract_feedback_policy,
            plan_set_feedback_policy=plan_set_feedback_policy,
        )

    def snapshot_to(self, target: str | Path) -> Path:
        destination = Path(target)
        destination.mkdir(parents=True, exist_ok=True)
        for source in sorted(self.root.iterdir()):
            if source.is_file():
                shutil.copy2(source, destination / source.name)
        return destination

    def require_weather_coverage(self, through_day: int, *, start_day: int = 1) -> None:
        required = set(range(int(start_day), int(start_day) + int(through_day)))
        missing = sorted(required - set(self.weather))
        if missing:
            raise ValueError(
                f"runtime weather does not cover source days {start_day} through "
                f"{int(start_day) + int(through_day) - 1}; "
                f"first missing day is {missing[0]}"
            )

    @property
    def by_id(self) -> dict[str, Offer]:
        return {offer.offer_id: offer for offer in self.offers}

    @property
    def service_lines(self) -> tuple[str, ...]:
        return tuple(sorted({offer.service_line for offer in self.offers}))
