"""Agent-facing product projection and internal/public field-difference audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .environment import OutfitterEnvironment
from .csvio import write_csv as _write_csv
from .tools import ToolExecutor


FORBIDDEN_AGENT_KEYS = frozenset(
    {
        "quote_context_by_date",
        "service_offer_audit",
        "market_reference_line_day",
        "product_type",
        "product_subtype",
        "choice_group",
        "service_key",
        "perceived_quality_tier",
        "perceived_quality_confidence",
        "perceived_quality_provenance",
        "quality_tier",
        "quality_tier_provenance",
        "decision_signature",
        "decision_attributes",
        "descriptive_attributes",
        "consumer_value_attributes",
        "consumer_value_evidence",
        "reservation_price_cny",
        "reservation_low_cny",
        "reservation_central_cny",
        "reservation_high_cny",
        "alternative_accessibility",
        "customer_segment",
        "episode_market_profile",
        "demand_model",
        "reference_transaction_model",
        "reference_transaction_evidence",
        "reference_cell",
        "reference_cell_id",
        "reference_class_id",
        "reference_generalized_cost",
        "price_to_reference_ratio",
        "q_ref",
        "beta_effective",
        "demand_world_id",
        "elasticity",
        "entry_probability",
        "preferred_availability",
        "preferred_group",
        "asked_group",
        "preferred_group_state",
        "quote_kind",
        "reference_daily_rate",
        "consumer_online_offer",
        "online_choice",
        "online_offer_id",
        "need_horizon_days",
        "occasion_importance",
        "online_late_probability",
        "expected_lateness_loss",
        "immediate_fulfillment_value",
        "markup_reference_weight",
        "outside_option",
        "selected_channel",
        "internal_channel",
        "channel_scores",
        "channel_systematic",
        "internal_channel_outcomes",
        "internal_basket_intents",
        "service_life_limit_days",
        "service_life_limit_cycles",
        "damage_risk_band",
        "lifecycle_prior_scope",
        "holiday_id",
        "holiday_arrival_multiplier",
        "weather_source_date",
        "weather_source_condition",
        "thermal_regime",
    }
)


def is_integrity_identifier_key(value: Any) -> bool:
    normalized = str(value).strip().lower()
    return normalized in {"hash", "sha256", "fingerprint"} or any(
        normalized.endswith(suffix)
        for suffix in ("_hash", "_sha256", "_fingerprint")
    )


def forbidden_agent_paths(value: Any, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}"
            if str(key) in FORBIDDEN_AGENT_KEYS or is_integrity_identifier_key(key):
                paths.append(path)
            if str(key) == "name" and str(item) in FORBIDDEN_AGENT_KEYS:
                paths.append(path)
            paths.extend(forbidden_agent_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(forbidden_agent_paths(item, f"{prefix}[{index}]"))
    return paths


def assert_agent_visible(value: Any, *, allowed_paths: frozenset[str] = frozenset()) -> None:
    paths = [path for path in forbidden_agent_paths(value) if path not in allowed_paths]
    if paths:
        raise ValueError(f"agent-visible payload contains internal fields: {paths}")


def write_product_visibility_exports(
    environment: OutfitterEnvironment, output_dir: Path
) -> tuple[Path, Path]:
    public_rows = []
    state = environment.reset()
    tool_executor = ToolExecutor(environment)
    for offer in sorted(environment.dataset.offers, key=lambda row: row.offer_id):
        result = tool_executor.execute(
            state, "inspect_offer", {"offer_id": offer.offer_id}
        )
        if not result.ok:
            raise ValueError(f"inspect_offer projection failed: {result.data}")
        public = result.data["offer"]
        # These exact paths are already public contractual lifecycle facts;
        # similarly named hidden model fields elsewhere remain prohibited.
        assert_agent_visible(public, allowed_paths=frozenset({
            "$.lifecycle_profile.service_life_limit_days",
            "$.lifecycle_profile.service_life_limit_cycles",
        }))
        public_rows.append(
            {
                "offer_id": public["offer_id"],
                "kind": public["kind"],
                "name": public["name"],
                "brand": public.get("brand"),
                "source_title": public["source_title"],
                "source_variant_name": public["source_variant_name"],
                "seller_id": public["seller_id"],
                "seller_name": public["seller_name"],
                "estimated_delivery_days": public["estimated_delivery_days"],
                "package_price": public["package_price"],
                "package_units": public["package_units"],
                "supplier_unit_cost": public["supplier_unit_cost"],
                "current_price": public["current_price"],
                "fixed_deposit": public.get("fixed_deposit"),
                "accepting_new_rentals": public.get("accepting_new_rentals"),
                "consumer_unit_json": json.dumps(
                    public.get("consumer_unit"), ensure_ascii=False, sort_keys=True
                ),
                "specifications_json": json.dumps(
                    public["specifications"], ensure_ascii=False, sort_keys=True
                ),
                "source_evidence_json": json.dumps(
                    public.get("source_evidence", []),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "normal_return_cleaning_json": json.dumps(
                    public.get("normal_return_cleaning"),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                **({"holding_costs_json": json.dumps(public.get("holding_costs"), ensure_ascii=False, sort_keys=True),
                    "lifecycle_profile_json": json.dumps(public.get("lifecycle_profile"), ensure_ascii=False, sort_keys=True)}
                   if environment.dataset.metadata.get("operating_history_policy") else {}),
            }
        )
    sales_rows = []
    for offer in sorted(environment.dataset.offers, key=lambda row: row.offer_id):
        row = {
                "offer_id": offer.offer_id,
                "product_type": offer.service_line,
                "choice_group": offer.choice_group,
                "service_key": offer.fulfills_service_keys[0],
                "asset_kind": offer.asset_kind,
                "perceived_quality_tier": offer.perceived_quality_tier,
                "perceived_quality_confidence": offer.perceived_quality_confidence,
                "decision_signature": offer.decision_signature,
                "decision_attributes_json": json.dumps(
                    offer.decision_attributes, ensure_ascii=False, sort_keys=True
                ),
                "descriptive_attributes_json": json.dumps(
                    offer.descriptive_attributes, ensure_ascii=False, sort_keys=True
                ),
                "consumer_value_attributes_json": json.dumps(
                    offer.consumer_value_attributes,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "reference_daily_rate": offer.reference_daily_rate,
                "service_life_limit_days": offer.service_life_limit_days,
                "service_life_limit_cycles": offer.service_life_limit_cycles,
                "damage_risk_band": offer.damage_risk_band,
                "cleaning_work_points": offer.cleaning_work_points,
                "consumer_online_offer_json": json.dumps(
                    offer.consumer_online_offer,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        if environment.demand_model is None:
            row["reference_retail_price"] = offer.reference_retail_price
        if environment.reference_demand_model is not None:
            row["reference_cell_id"] = (
                environment.reference_demand_model.offer_assignments.get(
                    offer.offer_id
                )
            )
        sales_rows.append(row)
    public_path = output_dir / "agent_visible_product_data.csv"
    sales_path = output_dir / "sales_model_product_data.csv"
    _write_csv(public_path, public_rows)
    _write_csv(sales_path, sales_rows)
    return sales_path, public_path
