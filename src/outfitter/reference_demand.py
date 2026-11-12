"""Reference-transaction demand model and pure, auditable decision evaluator."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


REFERENCE_DEMAND_FORMAT = "outfitter-reference-transaction-demand-v1"
REFERENCE_WORLDS_FORMAT = "outfitter-reference-demand-worlds-v1"


@dataclass(frozen=True, slots=True)
class ReferenceDemandWorld:
    world_id: str
    q_ref: float
    beta: float
    participates_in_economic_evaluation: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReferenceDemandWorld":
        row = cls(**value)
        if not row.world_id or not 0 < row.q_ref < 1 or row.beta <= 1:
            raise ValueError("reference demand world requires q_ref in (0,1) and beta > 1")
        return row


DEFAULT_REFERENCE_WORLDS = {
    row.world_id: row
    for row in (
        ReferenceDemandWorld("conservative", 0.35, 4.0),
        ReferenceDemandWorld("base", 0.50, 2.5),
        ReferenceDemandWorld("convenience-heavy", 0.65, 2.2),
        ReferenceDemandWorld(
            "extreme-tail-stress",
            0.65,
            1.5,
            participates_in_economic_evaluation=False,
        ),
    )
}


@dataclass(frozen=True, slots=True)
class ReferenceSegment:
    share: float
    beta_multiplier: float
    functional_weight: float
    quality_weight: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReferenceSegment":
        row = cls(**value)
        if min(
            row.share,
            row.beta_multiplier,
            row.functional_weight,
            row.quality_weight,
        ) <= 0:
            raise ValueError("reference segment values must be positive")
        return row


DEFAULT_REFERENCE_SEGMENTS = {
    "price_sensitive": ReferenceSegment(0.35, 1.20, 1.00, 0.60),
    "balanced": ReferenceSegment(0.35, 1.00, 1.00, 1.00),
    "quality_seeking": ReferenceSegment(0.15, 0.90, 1.10, 1.50),
    "need_driven": ReferenceSegment(0.15, 0.85, 1.25, 0.80),
}


@dataclass(frozen=True, slots=True)
class ReferenceGroupPolicy:
    service_key: str
    choice_group: str
    need_strength: str
    quality_salience: str
    alternative_accessibility: str
    substitution_strength: str

    @property
    def key(self) -> str:
        return f"{self.service_key}|{self.choice_group}"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReferenceGroupPolicy":
        row = cls(**value)
        for name in (
            "need_strength",
            "quality_salience",
            "alternative_accessibility",
            "substitution_strength",
        ):
            if getattr(row, name) not in {"low", "medium", "high"}:
                raise ValueError(f"invalid reference group ordinal: {name}")
        return row


@dataclass(frozen=True, slots=True)
class ReferenceCell:
    cell_id: str
    service_key: str
    choice_group: str
    transaction_kind: str
    cash_price_by_duration: dict[int, float]
    deposit: float
    quote_ids: tuple[str, ...]
    independent_source_count: int
    confidence: str
    synthetic: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReferenceCell":
        raw = dict(value)
        raw["cash_price_by_duration"] = {
            int(key): float(item)
            for key, item in dict(raw.get("cash_price_by_duration") or {}).items()
        }
        raw["quote_ids"] = tuple(map(str, raw.get("quote_ids") or ()))
        row = cls(**raw)
        if row.transaction_kind not in {"rental_contract", "retail_sale"}:
            raise ValueError("invalid reference transaction kind")
        if not row.cash_price_by_duration or any(
            duration < 0 or not math.isfinite(price) or price <= 0
            for duration, price in row.cash_price_by_duration.items()
        ):
            raise ValueError("reference price schedule must be finite and positive")
        if row.deposit < 0 or row.confidence not in {"low", "medium", "high"}:
            raise ValueError("invalid reference cell deposit/confidence")
        if not row.synthetic and (
            row.independent_source_count < 3 or len(row.quote_ids) < 3
        ):
            raise ValueError("formal reference cells require three independent quotes")
        return row

    def generalized_unit_cost(self, duration_days: int | None) -> float:
        duration = 0 if self.transaction_kind == "retail_sale" else max(
            1, int(duration_days or 1)
        )
        try:
            cash = self.cash_price_by_duration[duration]
        except KeyError as error:
            raise ValueError(
                f"reference cell {self.cell_id} lacks duration {duration}"
            ) from error
        return cash + 0.05 * self.deposit


@dataclass(frozen=True, slots=True)
class ReferenceTransactionModel:
    format: str
    cells: dict[str, ReferenceCell]
    offer_assignments: dict[str, str]
    groups: dict[str, ReferenceGroupPolicy]
    segments: dict[str, ReferenceSegment]
    worlds: dict[str, ReferenceDemandWorld]
    local_shift_range: tuple[float, float] = (-0.10, 0.10)
    entry_adjustment_bounds: tuple[float, float] = (-0.50, 0.50)
    profile_shock_scale: float = 0.25

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReferenceTransactionModel":
        raw = dict(value)
        cells = [ReferenceCell.from_dict(dict(row)) for row in raw.pop("reference_cells", [])]
        groups = [
            ReferenceGroupPolicy.from_dict(dict(row))
            for row in raw.pop("group_policies", [])
        ]
        raw["cells"] = {row.cell_id: row for row in cells}
        raw["groups"] = {row.key: row for row in groups}
        raw["segments"] = {
            str(key): ReferenceSegment.from_dict(dict(row))
            for key, row in dict(raw.get("segments") or {}).items()
        }
        raw["worlds"] = {
            str(key): ReferenceDemandWorld.from_dict(dict(row))
            for key, row in dict(raw.get("worlds") or {}).items()
        }
        raw["offer_assignments"] = {
            str(key): str(item)
            for key, item in dict(raw.get("offer_assignments") or {}).items()
        }
        for name in ("local_shift_range", "entry_adjustment_bounds"):
            raw[name] = tuple(map(float, raw.get(name, getattr(cls, name, ()))))
        model = cls(**raw)
        model.validate()
        return model

    def validate(self) -> None:
        if self.format != REFERENCE_DEMAND_FORMAT:
            raise ValueError(f"unsupported reference demand format: {self.format}")
        if set(self.segments) != set(DEFAULT_REFERENCE_SEGMENTS):
            raise ValueError("reference demand model must define four fixed segments")
        if not math.isclose(
            sum(row.share for row in self.segments.values()), 1.0, abs_tol=1e-9
        ):
            raise ValueError("reference segment shares must sum to one")
        if not {"conservative", "base", "convenience-heavy"} <= set(self.worlds):
            raise ValueError("reference demand model lacks formal worlds")
        missing_cells = sorted(set(self.offer_assignments.values()) - set(self.cells))
        if missing_cells:
            raise ValueError(f"reference offer assignments use unknown cells: {missing_cells}")
        if len(self.local_shift_range) != 2 or self.local_shift_range[0] > self.local_shift_range[1]:
            raise ValueError("invalid reference local shift range")
        if (
            len(self.entry_adjustment_bounds) != 2
            or self.entry_adjustment_bounds[0] > self.entry_adjustment_bounds[1]
        ):
            raise ValueError("invalid reference entry adjustment bounds")
        if self.profile_shock_scale <= 0:
            raise ValueError("profile shock scale must be positive")
        for world in self.worlds.values():
            if world.participates_in_economic_evaluation:
                minimum_beta = world.beta * min(
                    row.beta_multiplier for row in self.segments.values()
                )
                if minimum_beta <= 1:
                    raise ValueError("formal effective beta must exceed one")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "reference_cells": [
                {
                    **asdict(self.cells[key]),
                    "cash_price_by_duration": {
                        str(duration): price
                        for duration, price in sorted(
                            self.cells[key].cash_price_by_duration.items()
                        )
                    },
                    "quote_ids": list(self.cells[key].quote_ids),
                }
                for key in sorted(self.cells)
            ],
            "offer_assignments": dict(sorted(self.offer_assignments.items())),
            "group_policies": [asdict(self.groups[key]) for key in sorted(self.groups)],
            "segments": {
                key: asdict(self.segments[key]) for key in sorted(self.segments)
            },
            "worlds": {
                key: asdict(self.worlds[key]) for key in sorted(self.worlds)
            },
            "local_shift_range": list(self.local_shift_range),
            "entry_adjustment_bounds": list(self.entry_adjustment_bounds),
            "profile_shock_scale": self.profile_shock_scale,
        }


@dataclass(frozen=True, slots=True)
class ReferenceOfferView:
    offer_id: str
    profile_id: str
    choice_group: str
    cell_id: str | None
    available_units: int
    required_units: int
    hard_compatible: bool
    incompatibility_reasons: tuple[str, ...]
    soft_fit: float
    quality_score: float
    shop_cash_price: float
    shop_deposit: float
    reference_generalized_cost: float | None
    shop_generalized_cost: float
    price_ratio: float | None


@dataclass(frozen=True, slots=True)
class ReferenceEvaluationInput:
    service_key: str
    preferred_group: str
    customer_segment: str
    world_id: str
    group_need_adjustment: float
    accessibility_adjustment: float
    context_adjustment: float
    local_shift: float
    quality_salience: float
    substitution_penalty: float
    shop_gumbel: float
    outside_gumbel: float
    profile_gumbels: dict[str, float]
    offers: tuple[ReferenceOfferView, ...]


@dataclass(frozen=True, slots=True)
class ReferenceDecisionTrace:
    format: str
    world_id: str
    service_key: str
    preferred_group: str
    customer_segment: str
    q_ref: float
    beta_effective: float
    context_terms: dict[str, float]
    context_adjustment_total: float
    profiles: tuple[dict[str, Any], ...]
    best_profile_id: str | None
    entry_logit: float | None
    entry_probability: float
    shop_gumbel: float
    outside_gumbel: float
    buy_score: float
    outside_score: float
    bought: bool
    selected_profile_id: str | None
    selected_offer_id: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


NEED_ADJUSTMENTS = {"low": -0.10, "medium": 0.0, "high": 0.10}
ACCESSIBILITY_ADJUSTMENTS = {"low": 0.20, "medium": 0.0, "high": -0.20}
QUALITY_SALIENCE = {"low": 0.5, "medium": 1.0, "high": 1.5}
SUBSTITUTE_PENALTIES = {"low": 1.20, "medium": 0.70, "high": 0.35}
QUALITY_SCORES = {"economy": -0.5, "standard": 0.0, "premium": 0.5, "unknown": 0.0}


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def evaluate_reference_demand(
    model: ReferenceTransactionModel,
    value: ReferenceEvaluationInput,
) -> ReferenceDecisionTrace:
    """Evaluate one request without mutating inventory or business state."""

    try:
        world = model.worlds[value.world_id]
        segment = model.segments[value.customer_segment]
        group = model.groups[f"{value.service_key}|{value.preferred_group}"]
    except KeyError as error:
        raise ValueError(f"reference evaluation lacks policy: {error}") from error
    beta = world.beta * segment.beta_multiplier
    context_terms = {
        "need": float(value.group_need_adjustment),
        "accessibility": float(value.accessibility_adjustment),
        "context": float(value.context_adjustment),
        "local_shift": float(value.local_shift),
    }
    context_total = min(
        model.entry_adjustment_bounds[1],
        max(model.entry_adjustment_bounds[0], sum(context_terms.values())),
    )
    eligible = [
        row
        for row in value.offers
        if row.hard_compatible
        and row.available_units >= row.required_units
        and row.price_ratio is not None
        and row.reference_generalized_cost is not None
    ]
    if not eligible:
        return ReferenceDecisionTrace(
            format="outfitter-reference-demand-trace-v1",
            world_id=world.world_id,
            service_key=value.service_key,
            preferred_group=value.preferred_group,
            customer_segment=value.customer_segment,
            q_ref=world.q_ref,
            beta_effective=beta,
            context_terms=context_terms,
            context_adjustment_total=context_total,
            profiles=tuple(),
            best_profile_id=None,
            entry_logit=None,
            entry_probability=0.0,
            shop_gumbel=value.shop_gumbel,
            outside_gumbel=value.outside_gumbel,
            buy_score=float("-inf"),
            outside_score=value.outside_gumbel,
            bought=False,
            selected_profile_id=None,
            selected_offer_id=None,
            reason="no_reference_covered_compatible_offer",
        )

    by_profile: dict[str, list[ReferenceOfferView]] = {}
    for row in eligible:
        by_profile.setdefault(row.profile_id, []).append(row)
    profile_rows = []
    for profile_id, rows in sorted(by_profile.items()):
        representative = min(rows, key=lambda row: (float(row.price_ratio or math.inf), row.offer_id))
        ratio = float(representative.price_ratio)
        soft_fit_term = min(
            0.0,
            max(-0.5, 0.15 * segment.functional_weight * representative.soft_fit),
        )
        substitute = (
            0.0
            if representative.choice_group == value.preferred_group
            else value.substitution_penalty
        )
        entry_systematic = -beta * math.log(ratio) + soft_fit_term - substitute
        quality_term = (
            0.18
            * segment.quality_weight
            * value.quality_salience
            * representative.quality_score
        )
        choice_systematic = (
            segment.functional_weight * representative.soft_fit
            + quality_term
            - beta * math.log(ratio)
            - substitute
        )
        shock = float(value.profile_gumbels.get(profile_id, 0.0))
        profile_rows.append(
            {
                "profile_id": profile_id,
                "offer_ids": sorted(row.offer_id for row in rows),
                "representative_offer_id": representative.offer_id,
                "choice_group": representative.choice_group,
                "cell_id": representative.cell_id,
                "price_ratio": ratio,
                "shop_cash_price": representative.shop_cash_price,
                "shop_deposit": representative.shop_deposit,
                "shop_generalized_cost": representative.shop_generalized_cost,
                "reference_generalized_cost": representative.reference_generalized_cost,
                "soft_fit": representative.soft_fit,
                "soft_fit_entry_term": soft_fit_term,
                "quality_score": representative.quality_score,
                "quality_choice_term": quality_term,
                "substitute_penalty": substitute,
                "entry_systematic": entry_systematic,
                "choice_systematic": choice_systematic,
                "profile_gumbel": shock,
                "choice_score": choice_systematic + model.profile_shock_scale * shock,
            }
        )
    best = max(profile_rows, key=lambda row: (row["entry_systematic"], row["profile_id"]))
    entry_logit = _logit(world.q_ref) + float(best["entry_systematic"]) + context_total
    probability = _sigmoid(entry_logit)
    buy_score = entry_logit + value.shop_gumbel
    outside_score = value.outside_gumbel
    bought = buy_score > outside_score
    selected = (
        max(profile_rows, key=lambda row: (row["choice_score"], row["profile_id"]))
        if bought
        else None
    )
    return ReferenceDecisionTrace(
        format="outfitter-reference-demand-trace-v1",
        world_id=world.world_id,
        service_key=value.service_key,
        preferred_group=value.preferred_group,
        customer_segment=value.customer_segment,
        q_ref=world.q_ref,
        beta_effective=beta,
        context_terms=context_terms,
        context_adjustment_total=context_total,
        profiles=tuple(profile_rows),
        best_profile_id=str(best["profile_id"]),
        entry_logit=entry_logit,
        entry_probability=probability,
        shop_gumbel=value.shop_gumbel,
        outside_gumbel=outside_score,
        buy_score=buy_score,
        outside_score=outside_score,
        bought=bought,
        selected_profile_id=(None if selected is None else str(selected["profile_id"])),
        selected_offer_id=(
            None if selected is None else str(selected["representative_offer_id"])
        ),
        reason="bought" if bought else "outside_option",
    )


def default_reference_model_payload(
    *,
    reference_cells: list[dict[str, Any]],
    offer_assignments: dict[str, str],
    group_policies: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format": REFERENCE_DEMAND_FORMAT,
        "reference_cells": reference_cells,
        "offer_assignments": offer_assignments,
        "group_policies": group_policies,
        "segments": {
            key: asdict(row) for key, row in DEFAULT_REFERENCE_SEGMENTS.items()
        },
        "worlds": {
            key: asdict(row) for key, row in DEFAULT_REFERENCE_WORLDS.items()
        },
        "local_shift_range": [-0.10, 0.10],
        "entry_adjustment_bounds": [-0.50, 0.50],
        "profile_shock_scale": 0.25,
    }
