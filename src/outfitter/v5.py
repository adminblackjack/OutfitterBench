from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Iterable


RUNTIME_FORMAT = "outfitter-runtime-v5"
SCHEMA_VERSION = "outfitter-decision-schema-v5"
MARKET_RULES_VERSION = "outfitter-market-rules-v6"
CLEANING_WIPE_BOXES_PER_REQUEST = 1

BENCHMARK_CALENDAR_YEAR = 2026
HOLIDAY_NAMES_ZH: dict[str, str] = {
    "new_year": "元旦",
    "spring_festival": "春节",
    "qingming": "清明节",
    "labor_day": "劳动节",
    "dragon_boat": "端午节",
    "mid_autumn": "中秋节",
    "national_day": "国庆节",
}
PUBLIC_HOLIDAY_RANGES_2026: tuple[tuple[str, date, date], ...] = (
    ("new_year", date(2026, 1, 1), date(2026, 1, 3)),
    ("spring_festival", date(2026, 2, 15), date(2026, 2, 23)),
    ("qingming", date(2026, 4, 4), date(2026, 4, 6)),
    ("labor_day", date(2026, 5, 1), date(2026, 5, 5)),
    ("dragon_boat", date(2026, 6, 19), date(2026, 6, 21)),
    ("mid_autumn", date(2026, 9, 25), date(2026, 9, 27)),
    ("national_day", date(2026, 10, 1), date(2026, 10, 7)),
)
SCHOOL_HOLIDAY_START_2026 = date(2026, 7, 1)
SCHOOL_HOLIDAY_END_2026 = date(2026, 8, 31)

CHOICE_GROUPS_BY_SERVICE: dict[str, tuple[str, ...]] = {
    "rental:tent": ("compact_tent", "family_tent"),
    "rental:sleeping_bag": ("warm_weather_bag", "three_season_bag"),
    "rental:sleeping_pad": (
        "closed_cell_foam_pad",
        "self_inflating_pad",
        "air_mattress",
    ),
    "rental:camp_chair": ("lightweight_chair", "comfort_chair"),
    "rental:camp_table": ("camp_table",),
    "rental:camp_lantern": ("camp_lantern",),
    "rental:camp_cookware": ("camp_cookware",),
    "retail:rain_gear": ("disposable_poncho", "reusable_rainwear"),
    "retail:repellent": ("topical_repellent", "wearable_repellent"),
    "retail:bottled_water": ("bottled_water",),
    "retail:energy_snacks": ("compressed_ration", "snack_portion"),
    "retail:instant_food": ("instant_food",),
    "retail:cleaning_wipes": ("cleaning_wipes",),
    "retail:garbage_bags": ("garbage_bags",),
}

DECISION_ATTRIBUTE_NAMES: dict[str, tuple[str, ...]] = {
    "tent": ("comfortable_capacity_persons", "capacity_persons"),
    "sleeping_bag": ("comfort_temp_c", "size_class"),
    "sleeping_pad": ("pad_type", "sleep_slots"),
    "camp_chair": ("chair_style",),
    "camp_table": (),
    "camp_lantern": ("runtime_hours",),
    "camp_cookware": ("functions",),
    "rain_gear": ("form", "size_class", "reusable"),
    "repellent": ("form",),
    "bottled_water": ("volume_ml",),
    "energy_snacks": ("snack_type",),
    "instant_food": ("preparation",),
    "cleaning_wipes": ("sheets_per_unit",),
    "garbage_bags": ("bags_per_unit",),
}


@dataclass(frozen=True, slots=True)
class V5MarketRules:
    version: str = MARKET_RULES_VERSION
    arrival_scale: float = 2.2
    archetype_base_lambdas: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "overnight": (0.7, 1.4),
            "picnic": (0.8, 1.6),
            "lightweight": (0.5, 0.9),
            "supply": (1.5, 2.3),
        }
    )
    weather_multipliers: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "sunny": {
                "overnight": 1.25,
                "picnic": 1.30,
                "lightweight": 1.20,
                "supply": 1.05,
            },
            "cloudy": {
                "overnight": 1.0,
                "picnic": 1.0,
                "lightweight": 1.0,
                "supply": 1.0,
            },
            "hot": {
                "overnight": 0.75,
                "picnic": 0.60,
                "lightweight": 0.80,
                "supply": 1.10,
            },
            "rainy": {
                "overnight": 0.30,
                "picnic": 0.20,
                "lightweight": 0.35,
                "supply": 0.90,
            },
            "cold": {
                "overnight": 0.60,
                "picnic": 0.45,
                "lightweight": 0.70,
                "supply": 0.95,
            },
            "snowy": {
                "overnight": 0.15,
                "picnic": 0.10,
                "lightweight": 0.20,
                "supply": 0.80,
            },
        }
    )
    seasonal_anchors: tuple[tuple[int, float], ...] = (
        (91, 0.80),
        (111, 1.15),
        (135, 1.25),
        (166, 1.10),
        (196, 0.80),
        (227, 0.90),
        (258, 1.20),
        (279, 1.15),
        (288, 0.85),
    )
    holiday_arrival_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "new_year": 1.30,
            "spring_festival": 1.20,
            "qingming": 1.45,
            "labor_day": 1.75,
            "dragon_boat": 1.50,
            "mid_autumn": 1.60,
            "national_day": 2.00,
        }
    )
    lambda_offer: float = 0.25
    lambda_group: float = 0.70
    service_intercept: float = -0.35
    price_beta_mean: float = 2.4
    price_beta_sigma: float = 0.28
    deposit_friction: float = 0.05
    preferred_group_bonus: float = 0.65
    substitute_group_penalty: float = 0.55
    quality_values: dict[str, float] = field(
        default_factory=lambda: {
            "economy": -0.15,
            "standard": 0.0,
            "premium": 0.18,
            "unknown": 0.0,
        }
    )
    base_budgets: dict[str, float] = field(
        default_factory=lambda: {
            "rental:tent": 42.0,
            "rental:sleeping_bag": 15.0,
            "rental:sleeping_pad": 8.0,
            "rental:camp_chair": 6.0,
            "rental:camp_table": 12.0,
            "rental:camp_lantern": 10.0,
            "rental:camp_cookware": 10.0,
            "retail:rain_gear": 22.0,
            "retail:repellent": 24.0,
            "retail:bottled_water": 3.0,
            "retail:energy_snacks": 9.0,
            "retail:instant_food": 28.0,
            # One sale unit is a sealed outer box/pack, normally 100 wipes.
            "retail:cleaning_wipes": 15.0,
            "retail:garbage_bags": 7.0,
        }
    )
    cleaning_capacity_points: float = 30.0
    cleaning_work_points: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "camp_chair": (1.0, 0.10),
            "camp_lantern": (1.0, 0.10),
            "camp_table": (1.5, 0.15),
            "closed_cell_foam_pad": (1.5, 0.15),
            "air_mattress": (2.0, 0.20),
            "self_inflating_pad": (2.0, 0.20),
            "camp_cookware": (2.5, 0.25),
            "sleeping_bag": (4.5, 0.40),
            "tent": (6.0, 0.55),
        }
    )
    forecast_accuracy_by_lead: tuple[float, ...] = (
        1.0,
        0.90,
        0.80,
        0.70,
        0.65,
        0.60,
        0.55,
        0.50,
    )
    forecast_temperature_sigma_by_lead: tuple[float, ...] = (
        0.0,
        1.2,
        1.8,
        2.4,
        2.9,
        3.4,
        3.8,
        4.2,
    )

    def validate(self) -> None:
        if not 0 < self.lambda_offer <= self.lambda_group <= 1:
            raise ValueError("choice scales must satisfy 0 < lambda_offer <= lambda_group <= 1")
        if self.arrival_scale <= 0 or self.cleaning_capacity_points <= 0:
            raise ValueError("arrival scale and cleaning capacity must be positive")
        if set(self.archetype_base_lambdas) != {
            "overnight",
            "picnic",
            "lightweight",
            "supply",
        }:
            raise ValueError("market rules must define the four fixed archetypes")
        if len(self.forecast_accuracy_by_lead) != 8:
            raise ValueError("forecast accuracy must cover leads 0 through 7")
        if len(self.forecast_temperature_sigma_by_lead) != 8:
            raise ValueError("forecast temperature sigma must cover leads 0 through 7")
        if set(self.holiday_arrival_multipliers) != set(HOLIDAY_NAMES_ZH):
            raise ValueError("market rules must define every 2026 public-holiday multiplier")
        if any(value <= 0 for value in self.holiday_arrival_multipliers.values()):
            raise ValueError("holiday arrival multipliers must be positive")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["seasonal_anchors"] = [list(row) for row in self.seasonal_anchors]
        value["forecast_accuracy_by_lead"] = list(self.forecast_accuracy_by_lead)
        value["forecast_temperature_sigma_by_lead"] = list(
            self.forecast_temperature_sigma_by_lead
        )
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "V5MarketRules":
        raw = dict(value)
        legacy_holiday_multiplier = raw.pop("holiday_multiplier", None)
        if "holiday_arrival_multipliers" not in raw and legacy_holiday_multiplier is not None:
            raw["holiday_arrival_multipliers"] = {
                holiday_id: float(legacy_holiday_multiplier)
                for holiday_id in HOLIDAY_NAMES_ZH
            }
        for key in (
            "seasonal_anchors",
            "forecast_accuracy_by_lead",
            "forecast_temperature_sigma_by_lead",
        ):
            if key in raw:
                raw[key] = tuple(tuple(row) if isinstance(row, list) else row for row in raw[key])
        if "archetype_base_lambdas" in raw:
            raw["archetype_base_lambdas"] = {
                key: tuple(pair) for key, pair in raw["archetype_base_lambdas"].items()
            }
        if "cleaning_work_points" in raw:
            raw["cleaning_work_points"] = {
                key: tuple(pair) for key, pair in raw["cleaning_work_points"].items()
            }
        rules = cls(**raw)
        rules.validate()
        return rules


def stable_words(value: str, *, count: int = 4) -> tuple[int, ...]:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return tuple(
        int.from_bytes(digest[index * 4 : index * 4 + 4], "big")
        for index in range(count)
    )


def gumbel_from_uniform(value: float) -> float:
    clipped = min(1.0 - 1e-15, max(1e-15, float(value)))
    return -math.log(-math.log(clipped))


def logsumexp(values: Iterable[float], *, scale: float = 1.0) -> float:
    rows = [float(value) / scale for value in values]
    if not rows:
        return -math.inf
    peak = max(rows)
    return scale * (peak + math.log(sum(math.exp(value - peak) for value in rows)))


def seasonal_index(source_day: int, anchors: Iterable[tuple[int, float]]) -> float:
    rows = sorted((int(day), float(value)) for day, value in anchors)
    if source_day <= rows[0][0]:
        return rows[0][1]
    if source_day >= rows[-1][0]:
        return rows[-1][1]
    for (left_day, left), (right_day, right) in zip(rows, rows[1:], strict=False):
        if left_day <= source_day <= right_day:
            weight = (source_day - left_day) / (right_day - left_day)
            return left + weight * (right - left)
    raise AssertionError("seasonal anchor interpolation failed")


def public_holiday_for_date(value: date) -> tuple[str | None, str | None]:
    for holiday_id, start, end in PUBLIC_HOLIDAY_RANGES_2026:
        if start <= value <= end:
            return holiday_id, HOLIDAY_NAMES_ZH[holiday_id]
    return None, None


def calendar_classification(value: date) -> dict[str, Any]:
    is_weekend = value.weekday() >= 5
    holiday_id, holiday_name = public_holiday_for_date(value)
    if holiday_id is not None:
        day_type = "public_holiday"
    elif is_weekend:
        day_type = "weekend"
    elif SCHOOL_HOLIDAY_START_2026 <= value <= SCHOOL_HOLIDAY_END_2026:
        day_type = "school_holiday"
        holiday_name = "暑假"
    else:
        day_type = "weekday"
    return {
        "date": value.isoformat(),
        "month_day": value.strftime("%m-%d"),
        "weekday": value.strftime("%A"),
        "is_weekend": is_weekend,
        "day_type": day_type,
        "holiday_id": holiday_id,
        "holiday_name": holiday_name,
    }


def calendar_for_source_day(source_day: int) -> dict[str, Any]:
    if not 1 <= int(source_day) <= 365:
        raise ValueError("source day must be in [1, 365]")
    target = date(BENCHMARK_CALENDAR_YEAR, 1, 1) + timedelta(days=int(source_day) - 1)
    return calendar_classification(target)


def legacy_2023_holiday_for_source_day(source_day: int) -> tuple[str | None, str | None]:
    day = int(source_day)
    if day == 95:
        holiday_id = "qingming"
    elif 119 <= day <= 123:
        holiday_id = "labor_day"
    elif 173 <= day <= 175:
        holiday_id = "dragon_boat"
    elif 272 <= day <= 279:
        holiday_id = "national_day"
    else:
        return None, None
    return holiday_id, HOLIDAY_NAMES_ZH[holiday_id]


def day_type_for_source_day(source_day: int, *, is_weekend: bool | None = None) -> str:
    """Compatibility helper for callers that still address frozen weather by day-of-year."""

    classification = calendar_for_source_day(source_day)
    if is_weekend is not None and classification["day_type"] not in {
        "public_holiday",
        "school_holiday",
    }:
        return "weekend" if is_weekend else "weekday"
    return str(classification["day_type"])


def _fact_map(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        str(row.get("name")): row.get("value")
        for row in evidence
        if row.get("name") and row.get("value") is not None
    }


def _first_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
    return None if match is None else float(match.group(0))


def positive_capacity_range(value: Any) -> tuple[int, int] | None:
    """Parse a sourced tent-capacity value into comfort and hard maximums."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = int(value)
        return (number, number) if number > 0 else None
    text = str(value or "").strip()
    if not text:
        return None
    range_match = re.search(
        r"(?<!\d)(\d+)\s*(?:-|~|～|至|到)\s*(\d+)\s*(?:人)?",
        text,
    )
    if range_match is not None:
        low, high = sorted(map(int, range_match.groups()))
        return (low, high) if low > 0 else None
    person_match = re.search(r"(?<!\d)(\d+)\s*人", text)
    if person_match is not None:
        number = int(person_match.group(1))
        return (number, number) if number > 0 else None
    if re.fullmatch(r"\s*\d+(?:\.0+)?\s*", text):
        number = int(float(text))
        return (number, number) if number > 0 else None
    return None


def tent_capacity_range_from_offer(row: dict[str, Any]) -> tuple[int, int] | None:
    """Recover an audited capacity range from a current or legacy runtime row."""

    descriptive = dict(row.get("descriptive_attributes") or {})
    candidates: list[Any] = [descriptive.get("capacity_persons")]
    evidence = dict(row.get("decision_evidence") or {}).get("capacity_persons")
    if isinstance(evidence, dict):
        for source in evidence.get("sources") or []:
            if isinstance(source, dict):
                candidates.extend((source.get("evidence"), source.get("source_value")))
    candidates.extend(
        (
            row.get("source_variant_name"),
            row.get("name"),
            row.get("source_title"),
            dict(row.get("decision_attributes") or {}).get("capacity_persons"),
        )
    )
    for candidate in candidates:
        parsed = positive_capacity_range(candidate)
        if parsed is not None:
            return parsed
    return None


def normalize_tent_capacity_offer(row: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a release/runtime offer to range-aware internal tent capacity."""

    result = dict(row)
    if str(result.get("category_id")) != "tent":
        return result
    parsed = tent_capacity_range_from_offer(result)
    if parsed is None:
        return result
    comfortable, maximum = parsed
    attributes = dict(result.get("decision_attributes") or {})
    attributes.update(
        comfortable_capacity_persons=comfortable,
        capacity_persons=maximum,
    )
    prior = dict(result.get("decision_evidence") or {}).get("capacity_persons")
    sources = list(prior.get("sources") or []) if isinstance(prior, dict) else []
    result["decision_attributes"] = attributes
    result["decision_evidence"] = {
        **dict(result.get("decision_evidence") or {}),
        "comfortable_capacity_persons": {
            "derivation": "lower endpoint of an explicit sourced tent-capacity range",
            "sources": sources,
            "value": comfortable,
        },
        "capacity_persons": {
            "derivation": "upper endpoint of an explicit sourced tent-capacity range",
            "sources": sources,
            "value": maximum,
        },
    }
    result["decision_signature"] = decision_signature(
        str(result.get("choice_group") or ""),
        attributes,
        str(result.get("perceived_quality_tier") or "unknown"),
    )
    return result


def infer_v5_offer_fields(
    *,
    category_id: str,
    service_line: str,
    name: str,
    product_subtype: str | None,
    quality_tier: str | None,
    evidence: list[dict[str, Any]],
) -> tuple[str, dict[str, Any], dict[str, Any], str]:
    """Compatibility projection for pre-v5 artifacts.

    Released v5 data is produced by the audited extractor.  This projection is
    deliberately conservative and exists so the engine and migration tests can
    run against archived v2 data while the paid extraction campaign is pending.
    """

    facts = _fact_map(evidence)
    text = f"{name} {product_subtype or ''}".casefold()
    decision: dict[str, Any] = {}
    descriptive = dict(facts)

    if category_id == "tent":
        comfortable, maximum = positive_capacity_range(
            facts.get("capacity_persons")
        ) or (2, 2)
        decision.update(
            comfortable_capacity_persons=comfortable,
            capacity_persons=maximum,
        )
        group = "compact_tent" if maximum <= 3 else "family_tent"
    elif category_id == "sleeping_bag":
        comfort = _first_number(facts.get("comfort_temp_c"))
        if comfort is None:
            comfort = _first_number(facts.get("temperature_rating_c"))
        if comfort is None:
            comfort = 15.0 if "夏季" in name else 5.0
        decision.update(comfort_temp_c=float(comfort), size_class="adult_regular")
        group = "warm_weather_bag" if comfort >= 10 else "three_season_bag"
    elif category_id == "sleeping_pad":
        subtype = product_subtype or service_line
        slots = 3 if "三人" in name else 2 if "双人" in name else 1
        decision.update(pad_type=subtype, sleep_slots=slots)
        group = subtype
    elif category_id == "camp_chair":
        weight = _first_number(facts.get("weight_kg"))
        style = "comfort" if any(token in text for token in ("躺", "脚托", "豪华")) else "lightweight"
        if weight is not None and weight >= 3:
            style = "comfort"
        decision["chair_style"] = style
        group = f"{style}_chair"
    elif category_id == "camp_lantern":
        decision["runtime_hours"] = _first_number(facts.get("runtime_hours")) or 24.0
        group = "camp_lantern"
    elif category_id == "camp_cookware":
        contents = str(facts.get("set_contents") or name)
        functions = []
        if any(token in contents for token in ("壶", "煮", "锅")):
            functions.append("boil")
        if any(token in contents for token in ("煎", "炒")):
            functions.append("fry")
        if any(token in contents for token in ("汤", "炖", "蒸")):
            functions.append("stew")
        decision["functions"] = sorted(set(functions or ["boil"]))
        group = "camp_cookware"
    elif category_id == "camp_table":
        group = "camp_table"
    elif category_id == "rain_gear":
        subtype = product_subtype or service_line
        reusable = bool(facts.get("reusable", subtype != "disposable_poncho"))
        size_value = str(facts.get("size") or "adult").casefold()
        size_class = "child" if any(token in size_value for token in ("child", "儿童")) else "adult"
        decision.update(form=subtype, size_class=size_class, reusable=reusable)
        group = "disposable_poncho" if not reusable else "reusable_rainwear"
    elif category_id == "repellent":
        form = str(facts.get("form") or product_subtype or service_line)
        decision["form"] = form
        group = "wearable_repellent" if any(token in form for token in ("bracelet", "手环")) else "topical_repellent"
    elif category_id == "energy_snacks":
        subtype = product_subtype or service_line
        decision["snack_type"] = subtype
        group = "compressed_ration" if "compressed" in subtype else "snack_portion"
    elif category_id == "bottled_water":
        decision["volume_ml"] = int(_first_number(facts.get("volume_ml")) or 500)
        group = "bottled_water"
    elif category_id == "instant_food":
        decision["preparation"] = str(facts.get("preparation_method") or "self_heating")
        group = "instant_food"
    elif category_id == "cleaning_wipes":
        decision["sheets_per_unit"] = int(_first_number(facts.get("sheet_count")) or 1)
        group = "cleaning_wipes"
    elif category_id == "garbage_bags":
        decision["bags_per_unit"] = int(_first_number(facts.get("bag_count")) or 1)
        group = "garbage_bags"
    else:
        group = service_line

    tier = (
        quality_tier
        if quality_tier in {"economy", "standard", "premium", "unknown"}
        else "unknown"
    )
    return group, decision, descriptive, tier


def decision_signature(choice_group: str, attributes: dict[str, Any], tier: str) -> str:
    normalized = repr((choice_group, sorted(attributes.items()), tier))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
