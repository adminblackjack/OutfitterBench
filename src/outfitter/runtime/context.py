from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from ..data import OutfitterDataset
from ..models import Offer


@dataclass(slots=True)
class CatalogIndex:
    """Static catalog relationships shared by runtime services.

    The contained dictionaries and lists intentionally remain the same mutable
    objects exposed by the historical environment API. Research fixtures can
    therefore add synthetic offers explicitly, while normal episodes build the
    index only once.
    """

    offers: dict[str, Offer]
    by_line: dict[str, list[Offer]]
    by_service: dict[str, list[Offer]]
    by_group: dict[tuple[str, str], list[Offer]]
    product_type_to_service_key: dict[str, str]
    service_registry: list[dict[str, Any]]
    choice_groups: dict[str, tuple[str, ...]]

    @classmethod
    def build(
        cls,
        dataset: OutfitterDataset,
        fallback_service_key: Callable[[str, str], str],
    ) -> "CatalogIndex":
        by_line: dict[str, list[Offer]] = defaultdict(list)
        for offer in dataset.offers:
            by_line[offer.service_line].append(offer)

        product_type_to_service_key: dict[str, str] = {}
        for product_type, offers in by_line.items():
            explicit = {
                key
                for offer in offers
                for key in offer.fulfills_service_keys
                if str(key).strip()
            }
            if len(explicit) > 1:
                raise ValueError(
                    f"product type {product_type} maps to multiple service keys: "
                    f"{sorted(explicit)}"
                )
            product_type_to_service_key[product_type] = (
                next(iter(explicit))
                if explicit
                else fallback_service_key(product_type, offers[0].asset_kind)
            )

        grouped: dict[str, list[str]] = defaultdict(list)
        for product_type, service_key in product_type_to_service_key.items():
            grouped[service_key].append(product_type)
        service_registry = [
            {
                "service_key": service_key,
                "kind": service_key.split(":", 1)[0],
                "fulfillment_product_types": sorted(product_types),
            }
            for service_key, product_types in sorted(grouped.items())
        ]

        by_service: dict[str, list[Offer]] = defaultdict(list)
        by_group: dict[tuple[str, str], list[Offer]] = defaultdict(list)
        for offer in dataset.offers:
            service_key = product_type_to_service_key[offer.service_line]
            by_service[service_key].append(offer)
            by_group[(service_key, str(offer.choice_group))].append(offer)
        choice_groups = {
            service_key: tuple(
                sorted(str(offer.choice_group) for offer in offers)
            )
            for service_key, offers in by_service.items()
        }
        # Catalog validation guarantees one row per exact offer, but choice
        # groups repeat across offers and are exposed as unique stable values.
        choice_groups = {
            key: tuple(dict.fromkeys(values))
            for key, values in choice_groups.items()
        }
        return cls(
            offers=dataset.by_id,
            by_line=by_line,
            by_service=by_service,
            by_group=by_group,
            product_type_to_service_key=product_type_to_service_key,
            service_registry=service_registry,
            choice_groups=choice_groups,
        )
