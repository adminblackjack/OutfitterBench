"""Noisy, action-aligned feedback for unselected complete contracts."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any


CONTRACT_FEEDBACK_FORMAT = "outfitter-complete-contract-feedback-v1"
PLAN_SET_FEEDBACK_FORMAT = "outfitter-factorized-plan-set-feedback-v1"
PLAN_SET_FEEDBACK_FORMAT_V2 = (
    "outfitter-factorized-plan-set-feedback-v2-noisy-line-mention"
)
PLAN_SET_FEEDBACK_FORMAT_V3 = (
    "outfitter-factorized-plan-set-feedback-v3-independent-concern-tags"
)
PRICE_VALUE_TAG = "price_value"
EQUIPMENT_CONFIGURATION_TAG = "equipment_configuration"
OTHER_OR_UNCLEAR_TAG = "other_or_unclear"
PUBLIC_FACTORIZED_CONCERN_TAGS = (
    PRICE_VALUE_TAG,
    EQUIPMENT_CONFIGURATION_TAG,
)
PRICE_CONCERN = "price_value_concern"
CONFIGURATION_CONCERN = "contract_configuration_concern"
MULTIPLE_CONCERNS = "multiple_or_unclear_concerns"
NO_SPECIFIC_CONCERN = "no_specific_concern_other_contract_preferred"
EQUIPMENT_FIT_CONCERN = "equipment_fit_concern"
PUBLIC_ASSESSMENT_CATEGORIES = (
    PRICE_CONCERN,
    CONFIGURATION_CONCERN,
    MULTIPLE_CONCERNS,
    NO_SPECIFIC_CONCERN,
)
PUBLIC_PLAN_SET_DECLINE_CATEGORIES = (
    PRICE_CONCERN,
    EQUIPMENT_FIT_CONCERN,
    MULTIPLE_CONCERNS,
)


@dataclass(frozen=True, slots=True)
class ContractFeedbackPolicy:
    response_probability: float
    price_discount_fraction: float
    reporting_fidelity: float
    price_gain_scale: float = 1.0
    configuration_gain_scale: float = 1.0
    contrast_sharpness: float = 3.0
    mixed_signal_weight: float = 0.5
    dominance_ratio: float = 1.35
    minimum_action_relevance: float = 0.20
    relevance_scale_floor: float = 0.25

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ContractFeedbackPolicy":
        if value.get("format") != CONTRACT_FEEDBACK_FORMAT:
            raise ValueError("unsupported complete-contract feedback format")
        policy = cls(
            response_probability=float(value["response_probability"]),
            price_discount_fraction=float(value["price_discount_fraction"]),
            reporting_fidelity=float(value["reporting_fidelity"]),
            price_gain_scale=float(value.get("price_gain_scale", 1.0)),
            configuration_gain_scale=float(
                value.get("configuration_gain_scale", 1.0)
            ),
            contrast_sharpness=float(value.get("contrast_sharpness", 3.0)),
            mixed_signal_weight=float(value.get("mixed_signal_weight", 0.5)),
            dominance_ratio=float(value.get("dominance_ratio", 1.35)),
            minimum_action_relevance=float(
                value.get("minimum_action_relevance", 0.20)
            ),
            relevance_scale_floor=float(
                value.get("relevance_scale_floor", 0.25)
            ),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        for name in (
            "response_probability",
            "price_discount_fraction",
            "reporting_fidelity",
        ):
            raw = float(getattr(self, name))
            if not math.isfinite(raw) or not 0.0 < raw < 1.0:
                raise ValueError(f"{name} must be finite and within (0, 1)")
        for name in (
            "price_gain_scale",
            "configuration_gain_scale",
            "contrast_sharpness",
            "mixed_signal_weight",
        ):
            raw = float(getattr(self, name))
            if not math.isfinite(raw) or raw <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.dominance_ratio) or self.dominance_ratio <= 1.0:
            raise ValueError("dominance_ratio must exceed 1")
        if (
            not math.isfinite(self.minimum_action_relevance)
            or self.minimum_action_relevance < 0.0
        ):
            raise ValueError("minimum_action_relevance must be non-negative")
        if (
            not math.isfinite(self.relevance_scale_floor)
            or self.relevance_scale_floor <= 0.0
        ):
            raise ValueError("relevance_scale_floor must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": CONTRACT_FEEDBACK_FORMAT,
            "response_probability": self.response_probability,
            "price_discount_fraction": self.price_discount_fraction,
            "reporting_fidelity": self.reporting_fidelity,
            "price_gain_scale": self.price_gain_scale,
            "configuration_gain_scale": self.configuration_gain_scale,
            "contrast_sharpness": self.contrast_sharpness,
            "mixed_signal_weight": self.mixed_signal_weight,
            "dominance_ratio": self.dominance_ratio,
            "minimum_action_relevance": self.minimum_action_relevance,
            "relevance_scale_floor": self.relevance_scale_floor,
        }


@dataclass(frozen=True, slots=True)
class PlanSetFeedbackPolicy:
    response_probability: float
    reporting_fidelity: float | None = None
    price_burden_scale: float | None = None
    equipment_fit_regret_scale: float | None = None
    contrast_sharpness: float | None = 2.5
    mixed_signal_weight: float | None = 0.5
    feedback_format: str = PLAN_SET_FEEDBACK_FORMAT
    line_mention_probability: float | None = None
    line_mention_content_fidelity: float | None = None
    line_mention_contrast_sharpness: float | None = None
    line_mention_salience_scale: float | None = None
    price_curve_discount_fractions: tuple[float, ...] | None = None
    price_action_scale: float | None = None
    equipment_action_scale: float | None = None
    equipment_action_top_k: int | None = None
    tag_probability_floor: float | None = None
    tag_probability_ceiling: float | None = None
    tag_relevance_threshold_ratio: float | None = None
    tag_sharpness: float | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PlanSetFeedbackPolicy":
        feedback_format = str(value.get("format") or "")
        if feedback_format not in {
            PLAN_SET_FEEDBACK_FORMAT,
            PLAN_SET_FEEDBACK_FORMAT_V2,
            PLAN_SET_FEEDBACK_FORMAT_V3,
        }:
            raise ValueError("unsupported factorized-plan-set feedback format")
        policy = cls(
            response_probability=float(value["response_probability"]),
            reporting_fidelity=(
                float(value["reporting_fidelity"])
                if "reporting_fidelity" in value
                else None
            ),
            price_burden_scale=(
                float(value["price_burden_scale"])
                if "price_burden_scale" in value
                else None
            ),
            equipment_fit_regret_scale=(
                float(value["equipment_fit_regret_scale"])
                if "equipment_fit_regret_scale" in value
                else None
            ),
            contrast_sharpness=(
                float(value.get("contrast_sharpness", 2.5))
                if feedback_format != PLAN_SET_FEEDBACK_FORMAT_V3
                else None
            ),
            mixed_signal_weight=(
                float(value.get("mixed_signal_weight", 0.5))
                if feedback_format != PLAN_SET_FEEDBACK_FORMAT_V3
                else None
            ),
            feedback_format=feedback_format,
            line_mention_probability=(
                float(value["line_mention_probability"])
                if "line_mention_probability" in value
                else None
            ),
            line_mention_content_fidelity=(
                float(value["line_mention_content_fidelity"])
                if "line_mention_content_fidelity" in value
                else None
            ),
            line_mention_contrast_sharpness=(
                float(value["line_mention_contrast_sharpness"])
                if "line_mention_contrast_sharpness" in value
                else None
            ),
            line_mention_salience_scale=(
                float(value["line_mention_salience_scale"])
                if "line_mention_salience_scale" in value
                else None
            ),
            price_curve_discount_fractions=(
                tuple(
                    float(row)
                    for row in value["price_curve_discount_fractions"]
                )
                if "price_curve_discount_fractions" in value
                else None
            ),
            price_action_scale=(
                float(value["price_action_scale"])
                if "price_action_scale" in value
                else None
            ),
            equipment_action_scale=(
                float(value["equipment_action_scale"])
                if "equipment_action_scale" in value
                else None
            ),
            equipment_action_top_k=(
                int(value["equipment_action_top_k"])
                if "equipment_action_top_k" in value
                else None
            ),
            tag_probability_floor=(
                float(value["tag_probability_floor"])
                if "tag_probability_floor" in value
                else None
            ),
            tag_probability_ceiling=(
                float(value["tag_probability_ceiling"])
                if "tag_probability_ceiling" in value
                else None
            ),
            tag_relevance_threshold_ratio=(
                float(value["tag_relevance_threshold_ratio"])
                if "tag_relevance_threshold_ratio" in value
                else None
            ),
            tag_sharpness=(
                float(value["tag_sharpness"])
                if "tag_sharpness" in value
                else None
            ),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        response = float(self.response_probability)
        if not math.isfinite(response) or not 0.0 < response < 1.0:
            raise ValueError(
                "response_probability must be finite and within (0, 1)"
            )
        line_values = (
            self.line_mention_probability,
            self.line_mention_content_fidelity,
            self.line_mention_contrast_sharpness,
            self.line_mention_salience_scale,
        )
        tag_values = (
            self.price_curve_discount_fractions,
            self.price_action_scale,
            self.equipment_action_scale,
            self.equipment_action_top_k,
            self.tag_probability_floor,
            self.tag_probability_ceiling,
            self.tag_relevance_threshold_ratio,
            self.tag_sharpness,
        )
        if self.feedback_format == PLAN_SET_FEEDBACK_FORMAT_V3:
            if any(value is not None for value in (
                self.reporting_fidelity,
                self.price_burden_scale,
                self.equipment_fit_regret_scale,
                self.contrast_sharpness,
                self.mixed_signal_weight,
            )):
                raise ValueError("v3 concern tags cannot carry legacy category fields")
            if any(value is None for value in (*line_values, *tag_values)):
                raise ValueError(
                    "v3 concern tags require every tag and line-mention field"
                )
            discounts = tuple(self.price_curve_discount_fractions or ())
            if (
                not discounts
                or discounts != tuple(sorted(set(discounts)))
                or any(
                    not math.isfinite(value) or not 0.0 < value < 1.0
                    for value in discounts
                )
            ):
                raise ValueError(
                    "price_curve_discount_fractions must be unique, increasing, "
                    "finite values within (0, 1)"
                )
            for name in (
                "price_action_scale",
                "equipment_action_scale",
                "tag_relevance_threshold_ratio",
                "tag_sharpness",
            ):
                raw = float(getattr(self, name))
                if not math.isfinite(raw) or raw <= 0.0:
                    raise ValueError(f"{name} must be finite and positive")
            if int(self.equipment_action_top_k or 0) != 2:
                raise ValueError("v3 equipment_action_top_k must equal 2")
            floor = float(self.tag_probability_floor)
            ceiling = float(self.tag_probability_ceiling)
            if not (
                math.isfinite(floor)
                and math.isfinite(ceiling)
                and 0.0 < floor < ceiling < 1.0
            ):
                raise ValueError(
                    "tag probability floor/ceiling must satisfy 0 < floor < "
                    "ceiling < 1"
                )
        else:
            if any(value is not None for value in tag_values):
                raise ValueError("legacy plan-set feedback cannot carry v3 tag fields")
            for name in ("reporting_fidelity",):
                raw = float(getattr(self, name))
                if not math.isfinite(raw) or not 0.0 < raw < 1.0:
                    raise ValueError(f"{name} must be finite and within (0, 1)")
            for name in (
                "price_burden_scale",
                "equipment_fit_regret_scale",
                "contrast_sharpness",
                "mixed_signal_weight",
            ):
                raw = float(getattr(self, name))
                if not math.isfinite(raw) or raw <= 0.0:
                    raise ValueError(f"{name} must be finite and positive")
        if self.feedback_format == PLAN_SET_FEEDBACK_FORMAT:
            if any(value is not None for value in line_values):
                raise ValueError("v1 plan-set feedback cannot enable line mentions")
            return
        if self.feedback_format not in {
            PLAN_SET_FEEDBACK_FORMAT_V2,
            PLAN_SET_FEEDBACK_FORMAT_V3,
        }:
            raise ValueError("unsupported factorized-plan-set feedback format")
        if any(value is None for value in line_values):
            raise ValueError("line-enabled feedback requires all line-mention fields")
        for name in (
            "line_mention_probability",
            "line_mention_content_fidelity",
        ):
            raw = float(getattr(self, name))
            if not math.isfinite(raw) or not 0.0 < raw < 1.0:
                raise ValueError(f"{name} must be finite and within (0, 1)")
        for name in (
            "line_mention_contrast_sharpness",
            "line_mention_salience_scale",
        ):
            raw = float(getattr(self, name))
            if not math.isfinite(raw) or raw <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def noisy_line_mentions_enabled(self) -> bool:
        return self.feedback_format in {
            PLAN_SET_FEEDBACK_FORMAT_V2,
            PLAN_SET_FEEDBACK_FORMAT_V3,
        }

    @property
    def factorized_concern_tags_enabled(self) -> bool:
        return self.feedback_format == PLAN_SET_FEEDBACK_FORMAT_V3

    def to_dict(self) -> dict[str, Any]:
        value = {
            "format": self.feedback_format,
            "response_probability": self.response_probability,
        }
        if self.factorized_concern_tags_enabled:
            value.update(
                {
                    "price_curve_discount_fractions": list(
                        self.price_curve_discount_fractions or ()
                    ),
                    "price_action_scale": self.price_action_scale,
                    "equipment_action_scale": self.equipment_action_scale,
                    "equipment_action_top_k": self.equipment_action_top_k,
                    "tag_probability_floor": self.tag_probability_floor,
                    "tag_probability_ceiling": self.tag_probability_ceiling,
                    "tag_relevance_threshold_ratio": (
                        self.tag_relevance_threshold_ratio
                    ),
                    "tag_sharpness": self.tag_sharpness,
                }
            )
        else:
            value.update(
                {
                    "reporting_fidelity": self.reporting_fidelity,
                    "price_burden_scale": self.price_burden_scale,
                    "equipment_fit_regret_scale": (
                        self.equipment_fit_regret_scale
                    ),
                    "contrast_sharpness": self.contrast_sharpness,
                    "mixed_signal_weight": self.mixed_signal_weight,
                }
            )
        if self.noisy_line_mentions_enabled:
            value.update(
                {
                    "line_mention_probability": self.line_mention_probability,
                    "line_mention_content_fidelity": (
                        self.line_mention_content_fidelity
                    ),
                    "line_mention_contrast_sharpness": (
                        self.line_mention_contrast_sharpness
                    ),
                    "line_mention_salience_scale": (
                        self.line_mention_salience_scale
                    ),
                }
            )
        return value


def factorized_concern_uniform(
    *, seed: int, request_id: str, stream: str
) -> float:
    """Independent stable V23 tag tape shared by production and calibration."""

    if stream not in {"price-tag", "equipment-tag"}:
        raise ValueError("unsupported factorized-concern random stream")
    digest = hashlib.sha256(
        f"v23-{stream}|{int(seed)}|{request_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def factorized_concern_tag_probability(
    policy: PlanSetFeedbackPolicy,
    *,
    tag: str,
    action_gain: float,
) -> float:
    """Map one customer-only action-evidence axis to a noisy report chance."""

    if not policy.factorized_concern_tags_enabled:
        raise ValueError("factorized concern tags require a v3 feedback policy")
    if tag == PRICE_VALUE_TAG:
        scale = float(policy.price_action_scale)
    elif tag == EQUIPMENT_CONFIGURATION_TAG:
        scale = float(policy.equipment_action_scale)
    else:
        raise ValueError(f"unsupported factorized concern tag: {tag}")
    relevance = math.log1p(max(0.0, float(action_gain)) / scale)
    threshold = math.log1p(float(policy.tag_relevance_threshold_ratio))
    logit = float(policy.tag_sharpness) * (relevance - threshold)
    logistic = (
        1.0 / (1.0 + math.exp(-logit))
        if logit >= 0.0
        else math.exp(logit) / (1.0 + math.exp(logit))
    )
    return float(policy.tag_probability_floor) + (
        float(policy.tag_probability_ceiling)
        - float(policy.tag_probability_floor)
    ) * logistic


def reported_factorized_concerns(
    policy: PlanSetFeedbackPolicy,
    *,
    price_action_gain: float,
    equipment_action_gain: float,
    price_uniform: float,
    equipment_uniform: float,
) -> tuple[str, ...]:
    """Sample zero, one, or two independently reported actionable concerns.

    The residual tag is returned only when neither actionable concern is
    reported. Canonical tuple order has no ranking or strength semantics.
    """

    for name, value in (
        ("price_uniform", price_uniform),
        ("equipment_uniform", equipment_uniform),
    ):
        if not 0.0 <= float(value) < 1.0:
            raise ValueError(f"{name} must be within [0, 1)")
    concerns = []
    if float(price_uniform) < factorized_concern_tag_probability(
        policy,
        tag=PRICE_VALUE_TAG,
        action_gain=price_action_gain,
    ):
        concerns.append(PRICE_VALUE_TAG)
    if float(equipment_uniform) < factorized_concern_tag_probability(
        policy,
        tag=EQUIPMENT_CONFIGURATION_TAG,
        action_gain=equipment_action_gain,
    ):
        concerns.append(EQUIPMENT_CONFIGURATION_TAG)
    return tuple(concerns or [OTHER_OR_UNCLEAR_TAG])


def line_mention_uniform(*, seed: int, request_id: str, stream: str) -> float:
    """Independent stable tape shared by production and offline validation."""

    if stream not in {"availability", "content"}:
        raise ValueError("unsupported line-mention random stream")
    digest = hashlib.sha256(
        f"mention-{stream}|{int(seed)}|{request_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def reported_customer_mentioned_line_id(
    policy: PlanSetFeedbackPolicy,
    *,
    equipment_fit_regret_by_line: dict[str, float],
    availability_uniform: float,
    content_uniform: float,
) -> str | None:
    """Sample one subjective line mention without exposing a failed-line truth."""

    if not policy.noisy_line_mentions_enabled:
        return None
    for name, value in (
        ("availability_uniform", availability_uniform),
        ("content_uniform", content_uniform),
    ):
        if not 0.0 <= float(value) < 1.0:
            raise ValueError(f"{name} must be within [0, 1)")
    if float(availability_uniform) >= float(policy.line_mention_probability):
        return None
    salience = {
        str(line_id): max(0.0, float(value))
        for line_id, value in equipment_fit_regret_by_line.items()
    }
    line_ids = sorted(salience)
    if not line_ids:
        return None
    scale = float(policy.line_mention_salience_scale)
    sharpness = float(policy.line_mention_contrast_sharpness)
    logits = {
        line_id: sharpness * math.log1p(salience[line_id] / scale)
        for line_id in line_ids
    }
    maximum = max(logits.values())
    weights = {
        line_id: math.exp(logits[line_id] - maximum) for line_id in line_ids
    }
    total = sum(weights.values())
    fidelity = float(policy.line_mention_content_fidelity)
    probabilities = {
        line_id: fidelity * weights[line_id] / total
        + (1.0 - fidelity) / len(line_ids)
        for line_id in line_ids
    }
    cursor = 0.0
    for line_id in line_ids:
        cursor += probabilities[line_id]
        if float(content_uniform) < cursor + 1e-12:
            return line_id
    return line_ids[-1]


@dataclass(frozen=True, slots=True)
class ContractAssessmentDiagnostic:
    contract_option_id: str
    contract_score: float
    reference_score: float
    external_score: float
    price_gain: float
    configuration_gain: float
    another_store_contract_selected: bool


def latent_assessment_category(
    policy: ContractFeedbackPolicy,
    diagnostic: ContractAssessmentDiagnostic,
) -> str:
    """Classify an internal action-sensitivity topic, not a true reason."""

    price = max(0.0, float(diagnostic.price_gain)) / float(
        policy.price_gain_scale
    )
    configuration = max(
        0.0, float(diagnostic.configuration_gain)
    ) / float(policy.configuration_gain_scale)
    maximum = max(price, configuration)
    no_specific_allowed = bool(diagnostic.another_store_contract_selected)
    if maximum < float(policy.minimum_action_relevance):
        return NO_SPECIFIC_CONCERN if no_specific_allowed else MULTIPLE_CONCERNS
    if price >= float(policy.dominance_ratio) * max(configuration, 1e-12):
        return PRICE_CONCERN
    if configuration >= float(policy.dominance_ratio) * max(price, 1e-12):
        return CONFIGURATION_CONCERN
    return MULTIPLE_CONCERNS


def reported_assessment_category(
    policy: ContractFeedbackPolicy,
    diagnostic: ContractAssessmentDiagnostic,
    *,
    report_uniform: float,
) -> tuple[str, str]:
    """Return (reported, latent) using noisy action-relevance weights.

    Response availability is shared by a request, but callers use a stable
    contract-specific uniform for assessment content.  The possible public
    categories still depend on whether another store contract was selected, so
    the special comparative label never appears after both options were rejected.
    """

    if not 0.0 <= float(report_uniform) < 1.0:
        raise ValueError("report_uniform must be within [0, 1)")
    allowed = [PRICE_CONCERN, CONFIGURATION_CONCERN, MULTIPLE_CONCERNS]
    if diagnostic.another_store_contract_selected:
        allowed.append(NO_SPECIFIC_CONCERN)
    # These counterfactual actions have different natural utility scales: a
    # modest percentage discount is smaller than the best one-line equipment
    # replacement.  Compare each gain to a scale frozen by the same-tape panel.
    # Dividing both by the decision deficit created a >0.93 common-denominator
    # correlation in screening and made the public categories indistinguishable.
    price_relevance = max(0.0, float(diagnostic.price_gain)) / float(
        policy.price_gain_scale
    )
    configuration_relevance = max(
        0.0, float(diagnostic.configuration_gain)
    ) / float(policy.configuration_gain_scale)
    price_signal = math.log1p(price_relevance)
    configuration_signal = math.log1p(configuration_relevance)
    contrast = price_signal - configuration_signal
    sharpness = float(policy.contrast_sharpness)
    signals = {
        PRICE_CONCERN: 0.20 + math.exp(sharpness * contrast),
        CONFIGURATION_CONCERN: 0.20 + math.exp(-sharpness * contrast),
        MULTIPLE_CONCERNS: (
            0.35
            + float(policy.mixed_signal_weight)
            * math.exp(-sharpness * abs(contrast))
            + 0.35 / (1.0 + price_signal + configuration_signal)
        ),
    }
    if diagnostic.another_store_contract_selected:
        selected_advantage = max(
            float(policy.relevance_scale_floor),
            float(diagnostic.reference_score) - float(diagnostic.external_score),
        )
        customer_viability = max(
            0.0,
            float(diagnostic.contract_score) - float(diagnostic.external_score),
        ) / selected_advantage
        signals[NO_SPECIFIC_CONCERN] = 0.15 + math.log1p(customer_viability)
    signal_total = sum(signals[row] for row in allowed)
    signal_probabilities = {
        row: signals[row] / signal_total for row in allowed
    }
    uniform_probability = 1.0 / len(allowed)
    probabilities = {
        row: (
            float(policy.reporting_fidelity) * signal_probabilities[row]
            + (1.0 - float(policy.reporting_fidelity)) * uniform_probability
        )
        for row in allowed
    }
    latent = max(allowed, key=lambda row: (signals[row], row))
    cursor = 0.0
    for row in allowed:
        cursor += probabilities[row]
        if report_uniform < cursor + 1e-12:
            return row, latent
    return allowed[-1], latent


def reported_plan_set_decline_category(
    policy: ContractFeedbackPolicy | PlanSetFeedbackPolicy,
    *,
    price_burden: float,
    equipment_fit_regret: float,
    report_uniform: float,
) -> tuple[str, str]:
    """Return one noisy request-level concern after an empty authorized set.

    The inputs are separately scaled subjective salience measures. They are not
    ground-truth causes, thresholds, or guarantees that one action will produce
    a transaction.
    """

    if not 0.0 <= float(report_uniform) < 1.0:
        raise ValueError("report_uniform must be within [0, 1)")
    price_scale = (
        float(policy.price_burden_scale)
        if isinstance(policy, PlanSetFeedbackPolicy)
        else float(policy.price_gain_scale)
    )
    equipment_scale = (
        float(policy.equipment_fit_regret_scale)
        if isinstance(policy, PlanSetFeedbackPolicy)
        else float(policy.configuration_gain_scale)
    )
    price_signal = math.log1p(max(0.0, float(price_burden)) / price_scale)
    equipment_signal = math.log1p(
        max(0.0, float(equipment_fit_regret)) / equipment_scale
    )
    contrast = price_signal - equipment_signal
    sharpness = float(policy.contrast_sharpness)
    signals = {
        PRICE_CONCERN: 0.20 + math.exp(sharpness * contrast),
        EQUIPMENT_FIT_CONCERN: 0.20 + math.exp(-sharpness * contrast),
        MULTIPLE_CONCERNS: (
            0.35
            + float(policy.mixed_signal_weight)
            * math.exp(-sharpness * abs(contrast))
            + 0.35 / (1.0 + price_signal + equipment_signal)
        ),
    }
    latent = max(
        PUBLIC_PLAN_SET_DECLINE_CATEGORIES,
        key=lambda category: (signals[category], category),
    )
    total = sum(signals.values())
    uniform_probability = 1.0 / len(PUBLIC_PLAN_SET_DECLINE_CATEGORIES)
    probabilities = {
        category: (
            float(policy.reporting_fidelity) * signals[category] / total
            + (1.0 - float(policy.reporting_fidelity)) * uniform_probability
        )
        for category in PUBLIC_PLAN_SET_DECLINE_CATEGORIES
    }
    cursor = 0.0
    for category in PUBLIC_PLAN_SET_DECLINE_CATEGORIES:
        cursor += probabilities[category]
        if report_uniform < cursor + 1e-12:
            return category, latent
    return PUBLIC_PLAN_SET_DECLINE_CATEGORIES[-1], latent
