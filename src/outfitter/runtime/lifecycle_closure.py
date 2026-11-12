from __future__ import annotations

from typing import Any

from ..lifecycle_economics import (
    cleaning_terms,
    combined_wear_fraction,
    condition_loss_from_severity_uniform,
    damage_event_probability,
    damage_terms,
    holding_terms,
    maintenance_status,
    nominal_residual_value,
)
from ..models import RentalAsset, RentalContract


class LifecycleClosureMixin:
    """Runtime projection of the audited K/A/D/V lifecycle relationships."""

    def _lifecycle_factor(self, offer_id: str) -> dict[str, Any]:
        try:
            return self.lifecycle_factors[offer_id]
        except KeyError as error:
            raise ValueError(f"missing lifecycle factor for {offer_id}") from error

    def _lifecycle_projection(self, offer_id: str) -> dict[str, Any]:
        factor = self._lifecycle_factor(offer_id)
        try:
            return dict(factor["world_projections"][self.config.demand_world_id])
        except KeyError as error:
            raise ValueError(
                f"missing lifecycle world {self.config.demand_world_id} for {offer_id}"
            ) from error

    @staticmethod
    def _condition_ratio(asset: RentalAsset) -> float:
        return max(0.0, min(1.0, float(asset.condition_ratio)))

    def _combined_wear(self, asset: RentalAsset) -> float:
        return combined_wear_fraction(
            used_rental_days=asset.used_life_days,
            completed_rental_cycles=asset.used_rental_cycles,
            service_life_limit_days=asset.service_life_limit_days,
            service_life_limit_cycles=(
                asset.service_life_limit_cycles
                if asset.service_life_limit_cycles is not None
                else asset.service_life_limit_days
            ),
        )

    def _lifecycle_maintenance_status(self, asset: RentalAsset) -> dict[str, Any]:
        assert self.lifecycle_scales is not None
        return maintenance_status(
            combined_wear=self._combined_wear(asset),
            maintenance_wear_anchor=asset.maintenance_wear_anchor,
            condition_ratio=self._condition_ratio(asset),
            scales=self.lifecycle_scales,
        )

    def _damage_weather_multipliers(
        self, start_day: int, return_day: int
    ) -> list[float]:
        rates = {
            "sunny": 1.0,
            "cloudy": 1.0,
            "hot": 1.15,
            "cold": 1.25,
            "rainy": 1.55,
            "snowy": 1.75,
        }
        return [
            rates.get(self._demand_weather_regime(day), 1.0)
            for day in range(start_day, return_day)
        ]

    def _lifecycle_damage_outcome(
        self,
        asset: RentalAsset,
        *,
        start_day: int,
        return_day: int,
        event_uniform: float,
        severity_uniform: float,
        wear_at_start: float | None = None,
        condition_at_start: float | None = None,
        maintenance_exposure_at_start: float | None = None,
    ) -> dict[str, float | bool]:
        assert self.lifecycle_scales is not None
        factor = self._lifecycle_factor(asset.offer_id)
        projection = self._lifecycle_projection(asset.offer_id)
        wear = self._combined_wear(asset) if wear_at_start is None else wear_at_start
        condition = (
            self._condition_ratio(asset)
            if condition_at_start is None
            else condition_at_start
        )
        exposure = (
            max(0.0, wear - min(wear, asset.maintenance_wear_anchor))
            if maintenance_exposure_at_start is None
            else maintenance_exposure_at_start
        )
        duration = max(1, int(return_day) - int(start_day))
        probability = damage_event_probability(
            hazard_tier=str(factor["d_damage"]["hazard_tier"]),
            duration_days=duration,
            combined_wear=wear,
            condition_ratio=condition,
            wear_since_maintenance=exposure,
            world_hazard_multiplier=float(
                projection["damage_hazard_multiplier"]
            ),
            daily_hazard_multipliers=self._damage_weather_multipliers(
                int(start_day), int(start_day) + duration
            ),
            scales=self.lifecycle_scales,
        )
        damaged = float(event_uniform) < probability
        condition_loss = (
            condition_loss_from_severity_uniform(
                severity_tier=str(factor["d_damage"]["severity_tier"]),
                severity_uniform=float(severity_uniform),
                world_severity_multiplier=float(
                    projection["damage_severity_multiplier"]
                ),
                scales=self.lifecycle_scales,
            )
            if damaged
            else 0.0
        )
        return {
            "damaged": damaged,
            "damage_probability": round(probability, 10),
            "condition_loss": round(condition_loss, 10),
            "wear_at_start": round(wear, 10),
            "condition_at_start": round(condition, 10),
            "maintenance_exposure_at_start": round(exposure, 10),
        }

    def _refresh_contract_lifecycle_damage(
        self, asset: RentalAsset, contract: RentalContract
    ) -> None:
        if not self.config.lifecycle_economics_enabled:
            return
        if (
            contract.damage_event_uniform is None
            or contract.damage_severity_uniform is None
            or contract.damage_wear_at_start is None
            or contract.damage_condition_at_start is None
            or contract.damage_maintenance_exposure_at_start is None
        ):
            raise ValueError(
                f"lifecycle contract is missing frozen damage inputs: {contract.contract_id}"
            )
        outcome = self._lifecycle_damage_outcome(
            asset,
            start_day=contract.start_day,
            return_day=contract.return_day,
            event_uniform=contract.damage_event_uniform,
            severity_uniform=contract.damage_severity_uniform,
            wear_at_start=contract.damage_wear_at_start,
            condition_at_start=contract.damage_condition_at_start,
            maintenance_exposure_at_start=(
                contract.damage_maintenance_exposure_at_start
            ),
        )
        contract.damaged_on_return = bool(outcome["damaged"])
        contract.damage_condition_loss = (
            float(outcome["condition_loss"])
            if contract.damaged_on_return
            else None
        )

    def _lifecycle_return_damage_terms(
        self, asset: RentalAsset, contract: RentalContract
    ) -> dict[str, float]:
        assert self.lifecycle_scales is not None
        if contract.damage_condition_loss is None:
            raise ValueError(
                f"damaged lifecycle contract has no condition loss: {contract.contract_id}"
            )
        terms = damage_terms(
            replacement_cost=asset.unit_cost,
            condition_ratio_before=self._condition_ratio(asset),
            security_deposit=contract.deposit,
            condition_loss=contract.damage_condition_loss,
            scales=self.lifecycle_scales,
        )
        repair_cash = round(float(terms["repair_cash_cost"]), 2)
        deduction = round(
            min(float(contract.deposit), float(terms["deposit_deduction"])), 2
        )
        return {
            **terms,
            "repair_cash_cost": repair_cash,
            "deposit_deduction": deduction,
            "uncovered_repair_cost": round(
                max(0.0, repair_cash - deduction), 2
            ),
        }

    def _initialize_lifecycle_asset_state(self, asset: RentalAsset) -> None:
        """Project legacy opening damage without deriving condition from wear."""

        asset.condition_ratio = self._condition_ratio(asset)
        asset.maintenance_wear_anchor = max(
            0.0, float(asset.maintenance_wear_anchor)
        )
        if asset.status != "damaged" or asset.pending_repair_cost is not None:
            return
        assert self.lifecycle_scales is not None
        factor = self._lifecycle_factor(asset.offer_id)
        projection = self._lifecycle_projection(asset.offer_id)
        loss = condition_loss_from_severity_uniform(
            severity_tier=str(factor["d_damage"]["severity_tier"]),
            severity_uniform=0.5,
            world_severity_multiplier=float(
                projection["damage_severity_multiplier"]
            ),
            scales=self.lifecycle_scales,
        )
        terms = damage_terms(
            replacement_cost=asset.unit_cost,
            condition_ratio_before=asset.condition_ratio,
            security_deposit=0.0,
            condition_loss=loss,
            scales=self.lifecycle_scales,
        )
        asset.condition_ratio = float(terms["condition_ratio_after_damage"])
        asset.pending_repair_cost = float(terms["repair_cash_cost"])
        asset.repair_target_condition_ratio = float(
            terms["condition_ratio_after_repair"]
        )

    def _lifecycle_cleaning_terms(
        self,
        asset: RentalAsset,
        *,
        duration_days: int,
        start_day: int | None = None,
        return_day: int | None = None,
    ) -> dict[str, float]:
        assert self.lifecycle_scales is not None
        projection = self._lifecycle_projection(asset.offer_id)
        weather_multiplier = (
            self._return_weather_cleaning_multiplier(start_day, return_day)
            if start_day is not None and return_day is not None
            else 1.0
        )
        terms = cleaning_terms(
            one_day_work_points=float(
                projection["one_day_cleaning_work_points"]
            ),
            duration_days=duration_days,
            return_weather_multiplier=weather_multiplier,
            scales=self.lifecycle_scales,
        )
        return {
            **terms,
            "cash_cost": round(float(terms["cash_cost"]), 2),
            "weather_multiplier": round(weather_multiplier, 6),
        }

    def _lifecycle_holding_terms(
        self, asset: RentalAsset, *, stored_on_site: bool
    ) -> dict[str, float]:
        assert self.lifecycle_scales is not None
        projection = self._lifecycle_projection(asset.offer_id)
        return holding_terms(
            replacement_cost=asset.unit_cost,
            storage_burden_units=float(projection["storage_burden_units"]),
            stored_on_site=stored_on_site,
            scales=self.lifecycle_scales,
        )

    def _lifecycle_nominal_residual(
        self, asset: RentalAsset, *, condition_ratio: float | None = None
    ) -> float:
        assert self.lifecycle_scales is not None
        return nominal_residual_value(
            replacement_cost=asset.unit_cost,
            condition_ratio=(
                self._condition_ratio(asset)
                if condition_ratio is None
                else max(0.0, min(1.0, float(condition_ratio)))
            ),
            combined_wear=self._combined_wear(asset),
            scales=self.lifecycle_scales,
        )

    def _lifecycle_rented_going_concern_terms(
        self, asset: RentalAsset, contract: RentalContract
    ) -> dict[str, float]:
        assert self.lifecycle_scales is not None
        outcome = self._lifecycle_damage_outcome(
            asset,
            start_day=contract.start_day,
            return_day=contract.return_day,
            event_uniform=0.0,
            severity_uniform=0.5,
            wear_at_start=contract.damage_wear_at_start,
            condition_at_start=contract.damage_condition_at_start,
            maintenance_exposure_at_start=(
                contract.damage_maintenance_exposure_at_start
            ),
        )
        probability = float(outcome["damage_probability"])
        damage = damage_terms(
            replacement_cost=asset.unit_cost,
            condition_ratio_before=self._condition_ratio(asset),
            security_deposit=contract.deposit,
            condition_loss=float(outcome["condition_loss"]),
            scales=self.lifecycle_scales,
        )
        healthy = self._lifecycle_nominal_residual(asset)
        damaged = self._lifecycle_nominal_residual(
            asset,
            condition_ratio=float(damage["condition_ratio_after_damage"]),
        )
        cleaning = float(
            self._lifecycle_cleaning_terms(
                asset,
                duration_days=contract.duration_days,
                start_day=contract.start_day,
                return_day=contract.return_day,
            )["cash_cost"]
        )
        deduction = round(float(damage["deposit_deduction"]), 2)
        recoverable = (1.0 - probability) * healthy + probability * damaged
        expected_cleaning = (1.0 - probability) * cleaning
        expected_deduction = probability * deduction
        return {
            "recoverable_value": recoverable,
            "expected_cleaning_cost": expected_cleaning,
            "expected_damage_deduction": expected_deduction,
            "net_value": recoverable - expected_cleaning + expected_deduction,
        }
