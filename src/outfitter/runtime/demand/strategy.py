from __future__ import annotations

from collections import Counter
from typing import Any


class DemandStrategy:
    """Mode-specific dispatch chosen once when an environment is created."""

    mode = "unknown"
    channel_enabled = False
    reference_enabled = False
    profile_deduplication = False

    def select_quoted_offer(
        self,
        environment: Any,
        state: Any,
        session: Any,
        offer_ids: list[str],
    ) -> tuple[str, str, dict[str, float], float, float]:
        raise NotImplementedError

    def serve_session(
        self,
        environment: Any,
        state: Any,
        session: Any,
        outcomes: Counter[str],
    ) -> int:
        return environment._serve_choice_session_standard(
            state, session, outcomes
        )


class LegacyV5Strategy(DemandStrategy):
    mode = "legacy_v5"

    def select_quoted_offer(
        self,
        environment: Any,
        state: Any,
        session: Any,
        offer_ids: list[str],
    ) -> tuple[str, str, dict[str, float], float, float]:
        return environment._select_quoted_offer_legacy(
            state, session, offer_ids
        )


class BoundedStrategy(DemandStrategy):
    mode = "bounded"
    profile_deduplication = True

    def select_quoted_offer(
        self,
        environment: Any,
        state: Any,
        session: Any,
        offer_ids: list[str],
    ) -> tuple[str, str, dict[str, float], float, float]:
        return environment._select_quoted_offer_reservation(
            state, session, offer_ids
        )


class ReferenceStrategy(DemandStrategy):
    mode = "reference"
    reference_enabled = True
    profile_deduplication = True

    def select_quoted_offer(
        self,
        environment: Any,
        state: Any,
        session: Any,
        offer_ids: list[str],
    ) -> tuple[str, str, dict[str, float], float, float]:
        return environment._select_quoted_offer_reference(
            state, session, offer_ids
        )


class ChannelStrategy(BoundedStrategy):
    mode = "channel"
    channel_enabled = True

    def serve_session(
        self,
        environment: Any,
        state: Any,
        session: Any,
        outcomes: Counter[str],
    ) -> int:
        return environment._serve_choice_session_channel(
            state, session, outcomes
        ).fulfilled_units


def select_demand_strategy(
    *,
    demand_model: Any,
    reference_demand_model: Any,
    channel_demand_model: Any,
    world_id: str,
) -> DemandStrategy:
    if channel_demand_model is not None and world_id != "control":
        return ChannelStrategy()
    if reference_demand_model is not None and world_id != "control":
        return ReferenceStrategy()
    if demand_model is not None or reference_demand_model is not None:
        return BoundedStrategy()
    return LegacyV5Strategy()
