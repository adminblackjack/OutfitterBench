"""Versioned public promise profiles for advance-rental contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

from .models import Offer


PROFILE_CATALOG_FORMAT = "outfitter-reservation-promise-profiles-v1"
PROFILE_SCHEMA_VERSION = "reservation-promise-profile-v1"
PROFILE_DEPOSIT_POLICIES = {
    "canonical_reference_v1": {
        "kind": "frozen_profile_table",
        "description": (
            "Initial value is the V15 exact deposit of the lowest-cost active "
            "member; an existing non-null value is immutable across catalog updates."
        ),
    },
    "flat20_v1": {
        "kind": "flat_per_asset",
        "unit_deposit_cny": 20.0,
    },
    "flat40_v1": {
        "kind": "flat_per_asset",
        "unit_deposit_cny": 40.0,
    },
}

_REFERENCE_DEPOSIT_RATIOS = {
    "tent": 0.60,
    "sleeping_bag": 0.35,
    "closed_cell_foam_pad": 0.25,
    "self_inflating_pad": 0.40,
    "air_mattress": 0.55,
    "camp_table": 0.30,
    "camp_chair": 0.30,
    "camp_lantern": 0.40,
    "camp_cookware": 0.25,
}
_BAG_TEMPERATURE_THRESHOLDS = (-10, 0, 5, 15, 20)
_LANTERN_RUNTIME_THRESHOLDS = (6, 8, 16, 24, 32, 40)
_PAD_TYPES = (
    "air_mattress",
    "closed_cell_foam_pad",
    "self_inflating_pad",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _is_child_chair(offer: Offer) -> bool:
    text = f"{offer.name} {offer.source_title} {offer.source_variant_name}"
    return any(token in text for token in ("儿童", "小童"))


def _reference_exact_deposit(offer: Offer) -> float:
    return round(
        max(
            20.0,
            float(offer.unit_cost)
            * float(_REFERENCE_DEPOSIT_RATIOS[offer.service_line]),
        ),
        2,
    )


@dataclass(frozen=True, slots=True)
class ReservationPromiseProfile:
    profile_id: str
    family_id: str
    service_key: str
    public_guarantee: dict[str, Any]
    capacity_per_asset: int
    member_offer_ids: tuple[str, ...]
    unit_deposit_cny_by_policy: dict[str, float | None]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReservationPromiseProfile":
        return cls(
            profile_id=str(value["profile_id"]),
            family_id=str(value["family_id"]),
            service_key=str(value["service_key"]),
            public_guarantee=dict(value["public_guarantee"]),
            capacity_per_asset=int(value["capacity_per_asset"]),
            member_offer_ids=tuple(map(str, value.get("member_offer_ids", []))),
            unit_deposit_cny_by_policy={
                str(policy): (None if amount is None else float(amount))
                for policy, amount in dict(
                    value.get("unit_deposit_cny_by_policy") or {}
                ).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "family_id": self.family_id,
            "service_key": self.service_key,
            "public_guarantee": self.public_guarantee,
            "capacity_per_asset": self.capacity_per_asset,
            "member_offer_ids": list(self.member_offer_ids),
            "unit_deposit_cny_by_policy": dict(
                sorted(self.unit_deposit_cny_by_policy.items())
            ),
        }


@dataclass(frozen=True, slots=True)
class ReservationPromiseCatalog:
    schema_version: str
    source_catalog_sha256: str
    deposit_policies: dict[str, dict[str, Any]]
    profiles: tuple[ReservationPromiseProfile, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReservationPromiseCatalog":
        if value.get("format") != PROFILE_CATALOG_FORMAT:
            raise ValueError("unsupported reservation promise profile format")
        catalog = cls(
            schema_version=str(value["schema_version"]),
            source_catalog_sha256=str(value["source_catalog_sha256"]),
            deposit_policies={
                str(key): dict(row)
                for key, row in dict(value.get("deposit_policies") or {}).items()
            },
            profiles=tuple(
                ReservationPromiseProfile.from_dict(row)
                for row in value.get("profiles", [])
            ),
        )
        return catalog

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": PROFILE_CATALOG_FORMAT,
            "schema_version": self.schema_version,
            "source_catalog_sha256": self.source_catalog_sha256,
            "deposit_policies": dict(sorted(self.deposit_policies.items())),
            "profiles": [row.to_dict() for row in self.profiles],
        }

    @property
    def by_id(self) -> dict[str, ReservationPromiseProfile]:
        return {row.profile_id: row for row in self.profiles}

    def unit_deposit(self, profile_id: str, policy_id: str) -> float:
        if policy_id not in self.deposit_policies:
            raise ValueError(f"unknown reservation profile deposit policy: {policy_id}")
        try:
            value = self.by_id[profile_id].unit_deposit_cny_by_policy[policy_id]
        except KeyError as error:
            raise ValueError(f"unknown reservation promise profile: {profile_id}") from error
        if value is None or not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"profile {profile_id} has no active deposit for policy {policy_id}"
            )
        return round(float(value), 2)

    def eligible_profiles(self, public_envelope: dict[str, Any]) -> list[str]:
        service_key = str(public_envelope["service_key"])
        hard = dict(public_envelope.get("hard_requirements") or {})
        duration = int(public_envelope.get("duration_days", 1))
        profile_ids: set[str]
        if service_key == "rental:tent":
            party = int(hard["minimum_capacity_persons"])
            profile_ids = {f"tent.capacity_{party}plus.v1"}
        elif service_key == "rental:sleeping_bag":
            size = str(hard.get("size_class", "adult_regular"))
            profile_ids = {
                f"sleeping_bag.{size}.comfort_{threshold}c_or_lower.v1"
                for threshold in _BAG_TEMPERATURE_THRESHOLDS
            }
        elif service_key == "rental:sleeping_pad":
            profile_ids = {
                f"sleeping_pad.{pad_type}.slots_{slots}.v1"
                for pad_type in _PAD_TYPES
                for slots in (1, 2, 3)
            }
        elif service_key == "rental:camp_chair":
            audience = str(hard.get("audience", "adult"))
            profile_ids = {
                f"chair.{audience}.{style}.v1"
                for style in ("lightweight", "comfort")
            }
        elif service_key == "rental:camp_cookware":
            functions = tuple(sorted(map(str, hard["required_functions"])))
            name = "_".join(functions)
            profile_ids = {f"cookware.{name}.v1"}
        elif service_key == "rental:camp_lantern":
            needed = 8 * max(1, duration)
            profile_ids = {f"lantern.runtime_{needed}h_plus.v1"}
        elif service_key == "rental:camp_table":
            profile_ids = {"table.general.v1"}
        else:
            profile_ids = set()
        by_id = self.by_id
        return sorted(
            profile_id
            for profile_id in profile_ids
            if profile_id in by_id and by_id[profile_id].member_offer_ids
        )

    def validate(self, offers: Iterable[Offer]) -> None:
        offer_rows = list(offers)
        offers_by_id = {row.offer_id: row for row in offer_rows}
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported reservation promise profile schema version")
        if set(self.deposit_policies) != set(PROFILE_DEPOSIT_POLICIES):
            raise ValueError("reservation profile deposit policies are incomplete")
        profile_ids = [row.profile_id for row in self.profiles]
        if len(profile_ids) != len(set(profile_ids)) or profile_ids != sorted(profile_ids):
            raise ValueError("reservation promise profile IDs must be unique and sorted")
        covered: set[str] = set()
        for profile in self.profiles:
            if profile.capacity_per_asset < 1:
                raise ValueError(f"invalid profile capacity: {profile.profile_id}")
            if list(profile.member_offer_ids) != sorted(set(profile.member_offer_ids)):
                raise ValueError(f"profile members are not unique and sorted: {profile.profile_id}")
            if set(profile.unit_deposit_cny_by_policy) != set(
                PROFILE_DEPOSIT_POLICIES
            ):
                raise ValueError(f"profile deposit table is incomplete: {profile.profile_id}")
            for policy_id, amount in profile.unit_deposit_cny_by_policy.items():
                if amount is None:
                    if profile.member_offer_ids:
                        raise ValueError(
                            f"active profile has no {policy_id} deposit: {profile.profile_id}"
                        )
                    continue
                if not math.isfinite(float(amount)) or float(amount) <= 0:
                    raise ValueError(f"invalid profile deposit: {profile.profile_id}")
            for offer_id in profile.member_offer_ids:
                offer = offers_by_id.get(offer_id)
                if offer is None or offer.asset_kind != "durable":
                    raise ValueError(
                        f"profile contains an unknown non-durable offer: {offer_id}"
                    )
                if not _offer_matches_profile(offer, profile.profile_id):
                    raise ValueError(
                        f"profile membership contradicts public evidence: "
                        f"{profile.profile_id}/{offer_id}"
                    )
                covered.add(offer_id)
        durable_ids = {
            row.offer_id for row in offer_rows if row.asset_kind == "durable"
        }
        if covered != durable_ids:
            raise ValueError(
                "reservation promise profiles do not exactly cover durable offers: "
                f"missing={sorted(durable_ids-covered)}, extra={sorted(covered-durable_ids)}"
            )


def _profile_definitions() -> list[tuple[str, str, str, dict[str, Any], int]]:
    rows: list[tuple[str, str, str, dict[str, Any], int]] = []
    for capacity in range(1, 7):
        rows.append(
            (
                f"tent.capacity_{capacity}plus.v1",
                "tent.capacity.v1",
                "rental:tent",
                {"minimum_capacity_persons": capacity},
                capacity,
            )
        )
    for size in ("adult_regular", "child"):
        for threshold in _BAG_TEMPERATURE_THRESHOLDS:
            rows.append(
                (
                    f"sleeping_bag.{size}.comfort_{threshold}c_or_lower.v1",
                    f"sleeping_bag.{size}.temperature.v1",
                    "rental:sleeping_bag",
                    {"size_class": size, "comfort_temp_c_max": threshold},
                    1,
                )
            )
    for pad_type in _PAD_TYPES:
        for slots in (1, 2, 3):
            rows.append(
                (
                    f"sleeping_pad.{pad_type}.slots_{slots}.v1",
                    f"sleeping_pad.{pad_type}.capacity.v1",
                    "rental:sleeping_pad",
                    {"pad_type": pad_type, "sleep_slots": slots},
                    slots,
                )
            )
    for audience in ("adult", "child"):
        for style in ("lightweight", "comfort"):
            rows.append(
                (
                    f"chair.{audience}.{style}.v1",
                    f"chair.{audience}.style.v1",
                    "rental:camp_chair",
                    {"audience": audience, "chair_style": style},
                    1,
                )
            )
    for functions in (("boil",), ("fry",), ("boil", "stew")):
        name = "_".join(functions)
        rows.append(
            (
                f"cookware.{name}.v1",
                "cookware.function.v1",
                "rental:camp_cookware",
                {"guaranteed_functions": list(functions)},
                1,
            )
        )
    for runtime in _LANTERN_RUNTIME_THRESHOLDS:
        rows.append(
            (
                f"lantern.runtime_{runtime}h_plus.v1",
                "lantern.runtime.v1",
                "rental:camp_lantern",
                {"minimum_runtime_hours": runtime},
                1,
            )
        )
    rows.append(
        (
            "table.general.v1",
            "table.general.v1",
            "rental:camp_table",
            {"service": "camp_table"},
            1,
        )
    )
    return sorted(rows, key=lambda row: row[0])


def _offer_matches_profile(offer: Offer, profile_id: str) -> bool:
    attributes = offer.decision_attributes
    if profile_id.startswith("tent.capacity_"):
        capacity = int(profile_id.split("capacity_", 1)[1].split("plus", 1)[0])
        return offer.category_id == "tent" and int(
            attributes.get("capacity_persons", 0)
        ) >= capacity
    if profile_id.startswith("sleeping_bag."):
        parts = profile_id.split(".")
        size = parts[1]
        threshold = int(parts[2].split("comfort_", 1)[1].split("c_", 1)[0])
        return (
            offer.category_id == "sleeping_bag"
            and str(attributes.get("size_class", "adult_regular")) == size
            and float(attributes.get("comfort_temp_c", math.inf)) <= threshold
        )
    if profile_id.startswith("sleeping_pad."):
        parts = profile_id.split(".")
        slots = int(parts[2].split("slots_", 1)[1])
        return (
            offer.category_id == "sleeping_pad"
            and str(attributes.get("pad_type", offer.service_line)) == parts[1]
            and int(attributes.get("sleep_slots", 1)) == slots
        )
    if profile_id.startswith("chair."):
        parts = profile_id.split(".")
        audience = "child" if _is_child_chair(offer) else "adult"
        return (
            offer.category_id == "camp_chair"
            and audience == parts[1]
            and str(attributes.get("chair_style", "comfort")) == parts[2]
        )
    if profile_id.startswith("cookware."):
        needed = set(profile_id.split(".")[1].split("_"))
        available = set(map(str, attributes.get("functions", [])))
        return offer.category_id == "camp_cookware" and needed <= available
    if profile_id.startswith("lantern.runtime_"):
        threshold = int(profile_id.split("runtime_", 1)[1].split("h_", 1)[0])
        return offer.category_id == "camp_lantern" and float(
            attributes.get("runtime_hours", 0)
        ) >= threshold
    if profile_id == "table.general.v1":
        return offer.category_id == "camp_table"
    return False


def build_reservation_promise_catalog(
    offers: Iterable[Offer],
    *,
    source_catalog_sha256: str,
    previous: ReservationPromiseCatalog | None = None,
) -> ReservationPromiseCatalog:
    offer_rows = sorted(
        (row for row in offers if row.asset_kind == "durable"),
        key=lambda row: row.offer_id,
    )
    previous_by_id = {} if previous is None else previous.by_id
    profiles: list[ReservationPromiseProfile] = []
    for profile_id, family_id, service_key, guarantee, capacity in _profile_definitions():
        members = tuple(
            row.offer_id
            for row in offer_rows
            if _offer_matches_profile(row, profile_id)
        )
        prior = previous_by_id.get(profile_id)
        prior_reference = (
            None
            if prior is None
            else prior.unit_deposit_cny_by_policy.get("canonical_reference_v1")
        )
        if prior_reference is not None:
            canonical = float(prior_reference)
        elif members:
            cheapest = min(
                (row for row in offer_rows if row.offer_id in members),
                key=lambda row: (float(row.unit_cost), row.offer_id),
            )
            canonical = _reference_exact_deposit(cheapest)
        else:
            canonical = None
        profiles.append(
            ReservationPromiseProfile(
                profile_id=profile_id,
                family_id=family_id,
                service_key=service_key,
                public_guarantee=guarantee,
                capacity_per_asset=capacity,
                member_offer_ids=members,
                unit_deposit_cny_by_policy={
                    "canonical_reference_v1": canonical,
                    "flat20_v1": 20.0,
                    "flat40_v1": 40.0,
                },
            )
        )
    catalog = ReservationPromiseCatalog(
        schema_version=PROFILE_SCHEMA_VERSION,
        source_catalog_sha256=source_catalog_sha256,
        deposit_policies={
            key: dict(value) for key, value in PROFILE_DEPOSIT_POLICIES.items()
        },
        profiles=tuple(profiles),
    )
    catalog.validate(offer_rows)
    return catalog


def catalog_semantics_sha256(catalog: ReservationPromiseCatalog) -> str:
    semantic = [
        {
            "profile_id": row.profile_id,
            "family_id": row.family_id,
            "service_key": row.service_key,
            "public_guarantee": row.public_guarantee,
            "capacity_per_asset": row.capacity_per_asset,
            "unit_deposit_cny_by_policy": row.unit_deposit_cny_by_policy,
        }
        for row in catalog.profiles
    ]
    return hashlib.sha256(_canonical_json(semantic)).hexdigest()

