from __future__ import annotations

from typing import Any


class ReservationRecoveryBatchValidationError(ValueError):
    """Recovery validation failed after the entire business state was restored."""


class ReservationQuoteBatchValidationError(ValueError):
    """Structured V22 quote validation failure raised before any mutation."""

    def __init__(self, invalid_items: list[dict[str, Any]]):
        super().__init__("reservation quote batch contains invalid items")
        self.invalid_items = [dict(row) for row in invalid_items]


class ReservationContingencyVersionError(ValueError):
    """Structured, atomic V39 commitment-version mismatch."""

    def __init__(
        self, *, commitment_id: str, received_version: object, current_version: int
    ):
        super().__init__("commitment version does not match current terms")
        self.commitment_id = str(commitment_id)
        self.received_version = received_version
        self.current_version = int(current_version)
