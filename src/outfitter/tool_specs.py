from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

from .environment import RENTAL_ACTIVITY_EVENT_TYPES
from .supplier_trial import ACTIVITY_EVENT_TYPES as SUPPLIER_TRIAL_ACTIVITY_EVENT_TYPES
from .reservation_feedback import (
    EQUIPMENT_CONFIGURATION_TAG,
    OTHER_OR_UNCLEAR_TAG,
    PRICE_VALUE_TAG,
    PUBLIC_ASSESSMENT_CATEGORIES,
    PUBLIC_PLAN_SET_DECLINE_CATEGORIES,
)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    effect: Literal["read_only", "state_change", "terminal"]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}
V31_WAREHOUSE_CAPACITY_SPEC = ToolSpec(
    "view_warehouse_capacity",
    "View fixed warehouse capacity including return-space reservations for every owned rental asset and confirmed inbound unit. Global totals are never filtered; per-SKU counts and fixed integer weights are paginated. Rented/cleaning/repairing assets still occupy capacity; actual sale/disposal releases it. Retail and customer reservations do not add occupancy. No availability or scheduling verdict.",
    {"type": "object", "properties": {
        "offer_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "cursor": {"type": "integer", "minimum": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 50},
    }, "additionalProperties": False}, "read_only",
)
V38_REQUEST_CONTINGENCY_SPEC = ToolSpec(
    "request_reservation_contingency",
    "Ask a customer about one or two exact changes to a confirmed pre-pickup commitment. Each commitment has one successful inquiry for its entire lifetime. Proposals preserve duration and locked money terms; dates must be 2-14 days ahead and exact-SKU changes must remain inside the current Promise Profile. Customer response arrives next morning. The batch validates atomically and does not check store feasibility.",
    {
        "type": "object",
        "properties": {
            "inquiries": {
                "type": "array",
                "minItems": 1,
                "maxItems": 25,
                "items": {
                    "type": "object",
                    "properties": {
                        "commitment_id": {"type": "string"},
                        "commitment_version": {"type": "integer", "minimum": 1},
                        "proposals": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 2,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "proposal_id": {"type": "string", "minLength": 1, "maxLength": 60},
                                    "pickup_day": {"type": "integer", "minimum": 1},
                                    "return_day": {"type": "integer", "minimum": 2},
                                    "line_changes": {
                                        "type": "array",
                                        "maxItems": 4,
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "line_id": {"type": "string"},
                                                "offer_id": {"type": "string"},
                                                "promise_profile_id": {"type": "string"},
                                            },
                                            "required": ["line_id", "offer_id", "promise_profile_id"],
                                            "additionalProperties": False,
                                        },
                                    },
                                },
                                "required": ["proposal_id", "pickup_day", "return_day"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["commitment_id", "commitment_version", "proposals"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["inquiries"],
        "additionalProperties": False,
    },
    "state_change",
)
V38_APPLY_CONTINGENCY_SPEC = ToolSpec(
    "apply_reservation_contingency",
    "Apply one customer-accepted proposal or the customer's date-only counteroffer from a contingency report published today. Acceptance never changes a contract automatically. The full batch validates atomically; omitting a report keeps the original contract and the report expires at close.",
    {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 25,
                "items": {
                    "type": "object",
                    "properties": {
                        "report_id": {"type": "string"},
                        "selection_id": {"type": "string"},
                    },
                    "required": ["report_id", "selection_id"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    },
    "state_change",
)
HISTORY_PERIOD = {
    "type": "string",
    "enum": ["last_7_days", "last_30_days", "season_to_date"],
}
ITEMS = {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "offer_id": {"type": "string"},
            "packs": {"type": "integer", "minimum": 1, "maximum": 100},
        },
        "required": ["offer_id", "packs"],
        "additionalProperties": False,
    },
}
LIQUIDATION_REQUEST = {
    "type": "object",
    "properties": {
        "rental_asset_ids": {
            "type": "array",
            "maxItems": 25,
            "uniqueItems": True,
            "items": {"type": "string"},
        },
        "retail_items": {
            "type": "array",
            "maxItems": 25,
            "items": {
                "type": "object",
                "properties": {
                    "offer_id": {"type": "string"},
                    "units": {"type": "integer", "minimum": 1},
                },
                "required": ["offer_id", "units"],
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}
TOOL_SPECS = (
    ToolSpec(
        "business_status",
        "Get an authoritative current snapshot of cash, deposit obligations, activity, and season progress. The compact opening report is provided only once per business day; call this tool whenever a fresh overview is needed.",
        EMPTY,
        "read_only",
    ),
    ToolSpec(
        "business_history",
        "View factual operating totals or daily records for completed days, or successful manager actions for one current or completed day.",
        {
            "type": "object",
            "properties": {
                "view": {"type": "string", "enum": ["summary", "actions"]},
                "period": HISTORY_PERIOD,
                "horizon_days": {"type": "integer", "minimum": 1, "maximum": 30},
                "end_day": {"type": "integer", "minimum": 1},
                "granularity": {"type": "string", "enum": ["totals", "daily"]},
                "day": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
        "read_only",
    ),
    ToolSpec(
        "finance_history",
        "View factual daily financial records for completed business days.",
        {
            "type": "object",
            "properties": {
                "horizon_days": {"type": "integer", "minimum": 1, "maximum": 30},
                "period": HISTORY_PERIOD,
                "end_day": {"type": "integer", "minimum": 1},
                "granularity": {"type": "string", "enum": ["totals", "daily"]},
            },
            "additionalProperties": False,
        },
        "read_only",
    ),
    ToolSpec(
        "service_history",
        "View aggregate sales results or exact-offer performance for completed days.",
        {
            "type": "object",
            "properties": {
                "horizon_days": {"type": "integer", "minimum": 1, "maximum": 30},
                "period": HISTORY_PERIOD,
                "end_day": {"type": "integer", "minimum": 1},
                "granularity": {
                    "type": "string",
                    "enum": ["totals", "daily", "offer"],
                },
                "offer_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 20,
                    "uniqueItems": True,
                    "items": {"type": "string"},
                },
                "cursor": {"type": "integer", "minimum": 0},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 20,
                },
            },
            "additionalProperties": False,
        },
        "read_only",
    ),
    ToolSpec(
        "counter_log",
        "Inspect a page of exact requests, availability, quotes, and responses from the two most recently completed business days. A current or unavailable day returns available=false rather than a tool error.",
        {
            "type": "object",
            "properties": {
                "day": {"type": "integer", "minimum": 1},
                "cursor": {"type": "integer", "minimum": 0},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                },
            },
            "additionalProperties": False,
        },
        "read_only",
    ),
    ToolSpec(
        "demand_report",
        "Query long-run counter demand by up to three observable dimensions.",
        {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "uniqueItems": True,
                    "items": {
                        "type": "string",
                        "enum": [
                            "request_text",
                            "rental_or_retail",
                            "requested_size",
                            "disposition",
                            "day_type",
                            "holiday_name",
                            "weather",
                            "visible_party_context",
                            "day",
                            "week",
                        ],
                    },
                },
                "filters": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {},
                    },
                },
                "last_n_days": {"type": "integer", "minimum": 1, "maximum": 200},
                "from_day": {"type": "integer", "minimum": 1},
                "to_day": {"type": "integer", "minimum": 1},
                "cursor": {"type": "integer", "minimum": 0},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 20,
                },
            },
            "required": ["group_by"],
            "additionalProperties": False,
        },
        "read_only",
    ),
    ToolSpec(
        "weather_forecast",
        "View the current weather forecast through fourteen days ahead.",
        {
            "type": "object",
            "properties": {"horizon_days": {"type": "integer", "minimum": 1, "maximum": 14}},
            "additionalProperties": False,
        },
        "read_only",
    ),
    ToolSpec(
        "search_catalog",
        "Browse approved exact supplier offers using natural-language product text, source specifications, kind, or offer ID. Omitting view while providing query searches offers; limit is harmless on the compact index.",
        {
            "type": "object",
            "properties": {
                "view": {"type": "string", "enum": ["index", "offers"]},
                "query": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["rental_asset", "retail_item"],
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 10,
                },
                "cursor": {"type": "integer", "minimum": 0},
            },
            "additionalProperties": False,
        },
        "read_only",
    ),
    ToolSpec(
        "catalog_report",
        "Filter exact supplier offers by natural-language text, kind, or neutral source specification. query and kind may be supplied directly or inside filters.",
        {
            "type": "object",
            "properties": {
                "filters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": ["rental_asset", "retail_item"],
                        },
                    },
                    "additionalProperties": False,
                },
                "query": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["rental_asset", "retail_item"],
                },
                "specification_filters": {
                    "type": "array",
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "op": {"type": "string", "enum": ["eq", "lte", "gte", "contains"]},
                            "value": {},
                        },
                        "required": ["name", "op", "value"],
                        "additionalProperties": False,
                    },
                },
                "offset": {"type": "integer", "minimum": 0},
                "cursor": {"type": "integer", "minimum": 0},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 10,
                },
            },
            "additionalProperties": False,
        },
        "read_only",
    ),
    ToolSpec(
        "inspect_offer",
        "Inspect one exact catalog offer and its source evidence.",
        {
            "type": "object",
            "properties": {"offer_id": {"type": "string"}},
            "required": ["offer_id"],
            "additionalProperties": False,
        },
        "read_only",
    ),
    ToolSpec(
        "place_purchase_order",
        "Prepay packs from one seller for scheduled delivery before the outdoor season closes.",
        {
            "type": "object",
            "properties": {"seller_id": {"type": "string"}, "items": ITEMS},
            "required": ["seller_id", "items"],
            "additionalProperties": False,
        },
        "state_change",
    ),
    ToolSpec(
        "purchase_order_status",
        "View in-transit orders by default, delivered history by explicit filter, or one exact order.",
        {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "status": {"type": "string", "enum": ["in_transit", "delivered"]},
                "placed_day_from": {"type": "integer", "minimum": 1},
                "placed_day_to": {"type": "integer", "minimum": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "additionalProperties": False,
        },
        "read_only",
    ),
    ToolSpec(
        "view_retail_inventory",
        "View retail inventory and in-transit retail quantities by exact offer. Rental equipment is available only through view_rental_assets.",
        {
            "type": "object",
            "properties": {
                "offer_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 20,
                    "uniqueItems": True,
                    "items": {"type": "string"},
                },
            },
            "additionalProperties": False,
        },
        "read_only",
    ),
    ToolSpec(
        "view_rental_assets",
        "View rental equipment by complete offer summary (default) or paginated individual asset detail.",
        {
            "type": "object",
            "properties": {
                "view": {"type": "string", "enum": ["summary", "asset"]},
                "status": {
                    "type": "string",
                    "enum": [
                        "available",
                        "rented",
                        "cleaning",
                        "damaged",
                        "repairing",
                        "maintenance",
                        "maintenance_hold",
                        "retired",
                        "sold",
                    ],
                },
                "offer_id": {"type": "string"},
                "cursor": {"type": "integer", "minimum": 0},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 20,
                },
            },
            "additionalProperties": False,
        },
        "read_only",
    ),
    ToolSpec(
        "view_active_rentals",
        "View active rentals and future refundable deposits.",
        EMPTY,
        "read_only",
    ),
    ToolSpec(
        "reservation_inbox",
        "View today's frozen advance-rental requests and their end-of-day response deadline.",
        EMPTY,
        "read_only",
    ),
    ToolSpec(
        "view_reservation_calendar",
        "View up to 14 days of raw disjunctive exact-offer commitment blocks, current asset states, and supplier-delivery facts. This tool does not compute feasibility, shortage, coverage, or advice.",
        {
            "type": "object",
            "properties": {
                "offer_ids": {
                    "type": "array",
                    "maxItems": 20,
                    "uniqueItems": True,
                    "items": {"type": "string"},
                },
                "from_day": {"type": "integer", "minimum": 1},
                "to_day": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
        "read_only",
    ),
    ToolSpec(
        "view_reservation",
        "Browse a paginated compact reservation index, inspect details for at most three exact request/commitment/change IDs, or page the factual timeline for exactly one such ID.",
        {
            "type": "object",
            "properties": {
                "view": {
                    "type": "string",
                    "enum": ["index", "detail", "timeline"],
                },
                "identifiers": {
                    "type": "array",
                    "maxItems": 3,
                    "uniqueItems": True,
                    "items": {"type": "string"},
                },
                "cursor": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "additionalProperties": False,
        },
        "read_only",
    ),
    ToolSpec(
        "quote_reservation_inquiries",
        "Explicitly quote or decline today's reservation inquiries once, in the supplied order. A quote must provide one positive, cent-exact per-unit daily price for every required and optional line. The same line price applies to all of its exact candidates. Walk-in posted prices are separate and never fill missing contract prices. The environment computes rent and returns only request-level customer rejection or accepted locked terms; it never returns willingness ceilings, a rejection reason, or a feasibility verdict.",
        {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 50,
                    "items": {
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "request_id": {"type": "string"},
                                    "action": {
                                        "type": "string",
                                        "enum": ["quote"],
                                    },
                                    "line_prices": {
                                        "type": "array",
                                        "minItems": 1,
                                        "maxItems": 4,
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "line_id": {"type": "string"},
                                                "daily_unit_price": {
                                                    "type": "number",
                                                    "exclusiveMinimum": 0,
                                                    "multipleOf": 0.01,
                                                },
                                            },
                                            "required": [
                                                "line_id",
                                                "daily_unit_price",
                                            ],
                                            "additionalProperties": False,
                                        },
                                    },
                                },
                                "required": [
                                    "request_id",
                                    "action",
                                    "line_prices",
                                ],
                                "additionalProperties": False,
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "request_id": {"type": "string"},
                                    "action": {
                                        "type": "string",
                                        "enum": ["decline"],
                                    },
                                },
                                "required": ["request_id", "action"],
                                "additionalProperties": False,
                            },
                        ],
                    },
                }
            },
            "required": ["decisions"],
            "additionalProperties": False,
        },
        "state_change",
    ),
    ToolSpec(
        "respond_reservation_changes",
        "Accept or decline today's customer reschedule or extension requests. This tool does not process new reservation inquiries.",
        {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 50,
                    "items": {
                        "type": "object",
                        "properties": {
                            "change_id": {"type": "string"},
                            "action": {
                                "type": "string",
                                "enum": ["accept", "decline"],
                            },
                        },
                        "required": ["change_id", "action"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["decisions"],
            "additionalProperties": False,
        },
        "state_change",
    ),
    ToolSpec(
        "resolve_reservation_risks",
        "Handle a visible at-risk commitment by first asking which still-hidden frozen alternatives intersect the shop's current credible assortment, selecting a returned exact offer, or cancelling early with refund and notice-based compensation. Inquiry is idempotent and results report state transitions, not feasibility verdicts.",
        {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 25,
                    "items": {
                        "type": "object",
                        "properties": {
                            "commitment_id": {"type": "string"},
                            "action": {
                                "type": "string",
                                "enum": [
                                    "inquire_fallback",
                                    "select_authorized_fallback",
                                    "cancel_early",
                                ],
                            },
                            "offer_id": {"type": "string"},
                            "line_id": {"type": "string"},
                        },
                        "required": ["commitment_id", "action"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["actions"],
            "additionalProperties": False,
        },
        "state_change",
    ),
    ToolSpec(
        "view_rental_activity",
        "View recorded rental, return, cleaning, repair, retirement, acquisition, and sale events.",
        {
            "type": "object",
            "properties": {
                "horizon_days": {"type": "integer", "minimum": 1, "maximum": 30},
                "end_day": {"type": "integer", "minimum": 1},
                "offer_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 10,
                    "uniqueItems": True,
                    "items": {"type": "string"},
                },
                "asset_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 25,
                    "uniqueItems": True,
                    "items": {"type": "string"},
                },
                "event_types": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": {
                        "type": "string",
                        "enum": list(RENTAL_ACTIVITY_EVENT_TYPES),
                    },
                },
                "cursor": {
                    "type": "string",
                    "pattern": "^rental-event-[0-9]{6,}$",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["horizon_days"],
            "additionalProperties": False,
        },
        "read_only",
    ),
    ToolSpec(
        "view_liquidation_quotes",
        "Get today's fixed-dealer quotes for selected delivered retail units and/or available or damaged rental assets. At least one input list must be non-empty; invalid input rejects the quote but the diagnostic returns every invalid item, its status, and actual on-hand retail units.",
        LIQUIDATION_REQUEST,
        "read_only",
    ),
    ToolSpec(
        "sell_to_liquidation_dealer",
        "Atomically sell selected delivered retail units and/or available or damaged rental assets to the fixed dealer. At least one input list must be non-empty; any invalid item rejects the whole sale.",
        LIQUIDATION_REQUEST,
        "state_change",
    ),
    ToolSpec(
        "set_prices",
        "Set per-day rental prices or per-unit retail prices for exact offers.",
        {
            "type": "object",
            "properties": {
                "prices": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "offer_id": {"type": "string"},
                            "price": {"type": "number", "exclusiveMinimum": 0},
                        },
                        "required": ["offer_id", "price"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["prices"],
            "additionalProperties": False,
        },
        "state_change",
    ),
    ToolSpec(
        "set_rental_offer_status",
        "Open or close exact rental offers for new bookings without changing existing contracts.",
        {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "offer_id": {"type": "string"},
                            "open": {"type": "boolean"},
                        },
                        "required": ["offer_id", "open"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        },
        "state_change",
    ),
    ToolSpec(
        "repair_or_retire_assets",
        "Repair damaged assets (costs cash, two days) or retire them immediately.",
        {
            "type": "object",
            "properties": {
                "repair_asset_ids": {"type": "array", "items": {"type": "string"}},
                "retire_asset_ids": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
        "state_change",
    ),
    ToolSpec(
        "schedule_maintenance",
        "Schedule assets whose public inspection window is open for preventive maintenance. Maintenance is prepaid and temporarily removes the exact assets from rental capacity.",
        {
            "type": "object",
            "properties": {
                "asset_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 50,
                    "uniqueItems": True,
                    "items": {"type": "string"},
                }
            },
            "required": ["asset_ids"],
            "additionalProperties": False,
        },
        "state_change",
    ),
    ToolSpec(
        "memory_list",
        "List a bounded page of persistent private memory document metadata. This does not return document content.",
        {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 20,
                },
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "prefix": {"type": "string", "maxLength": 64},
                "sort": {
                    "type": "string",
                    "enum": ["updated_desc", "name_asc"],
                    "default": "updated_desc",
                },
            },
            "additionalProperties": False,
        },
        "read_only",
    ),
    ToolSpec(
        "memory_search",
        "Search persistent private memory document names, descriptions, and content using literal keywords. Returns bounded document metadata, not document content.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                    "description": "Whitespace-separated literal keywords; every keyword must match.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 20,
                },
                "offset": {"type": "integer", "minimum": 0, "default": 0},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "read_only",
    ),
    ToolSpec(
        "memory_read",
        "Read the complete content of one persistent memory document by exact name.",
        {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 64,
                    "pattern": "^[A-Za-z0-9_-]+$",
                }
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        "read_only",
    ),
    ToolSpec(
        "memory_write",
        "Create one persistent memory document, or fully replace the description and content of an existing document with the same name.",
        {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 64,
                    "pattern": "^[A-Za-z0-9_-]+$",
                },
                "description": {"type": "string", "minLength": 1, "maxLength": 200},
                "content": {"type": "string", "maxLength": 40000},
            },
            "required": ["name", "description", "content"],
            "additionalProperties": False,
        },
        "state_change",
    ),
    ToolSpec(
        "memory_delete",
        "Permanently delete one persistent memory document by exact name. A missing document returns an error.",
        {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 64,
                    "pattern": "^[A-Za-z0-9_-]+$",
                }
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        "state_change",
    ),
    ToolSpec(
        "close_business_day",
        "Finish today's decisions and advance the shop to the next business morning. An optional bounded handoff note is visible during the next business day only.",
        {
            "type": "object",
            "properties": {
                "handoff_note": {
                    "type": "string",
                    "maxLength": 3000,
                }
            },
            "additionalProperties": False,
        },
        "terminal",
    ),
)


PROFILE_QUOTE_SPEC = ToolSpec(
    "quote_reservation_inquiries",
    "Quote or decline today's promise-profile reservation inquiries once. Every quoted required and optional line must select one eligible public promise profile and provide one positive cent-exact per-asset daily price. The environment computes asset quantity, rent, prepayment, and the frozen profile deposit. Customer rejection remains terminal and returns no reason or counteroffer.",
    {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 50,
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "request_id": {"type": "string"},
                                "action": {
                                    "type": "string",
                                    "enum": ["quote"],
                                },
                                "lines": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 4,
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "line_id": {"type": "string"},
                                            "promise_profile_id": {
                                                "type": "string"
                                            },
                                            "daily_unit_price": {
                                                "type": "number",
                                                "exclusiveMinimum": 0,
                                                "multipleOf": 0.01,
                                            },
                                        },
                                        "required": [
                                            "line_id",
                                            "promise_profile_id",
                                            "daily_unit_price",
                                        ],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["request_id", "action", "lines"],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {
                                "request_id": {"type": "string"},
                                "action": {
                                    "type": "string",
                                    "enum": ["decline"],
                                },
                            },
                            "required": ["request_id", "action"],
                            "additionalProperties": False,
                        },
                    ]
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    },
    "state_change",
)

PAIRWISE_RESERVATION_VIEW_SPEC = ToolSpec(
    "view_reservation",
    "Browse a paginated compact reservation index, inspect details for at most three exact request/commitment/change IDs, page the factual timeline for exactly one ID, or page immutable raw quote/choice history. Quote history contains only public request context, submitted terms, aggregate outcomes, and accepted selections; it never summarizes preferences or exposes hidden customer state.",
    {
        "type": "object",
        "properties": {
            "view": {
                "type": "string",
                "enum": ["index", "detail", "timeline", "quote_history"],
            },
            "identifiers": {
                "type": "array",
                "maxItems": 3,
                "uniqueItems": True,
                "items": {"type": "string"},
            },
            "cursor": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        "additionalProperties": False,
    },
    "read_only",
)

SEARCHABLE_PAIRWISE_RESERVATION_VIEW_SPEC = ToolSpec(
    "view_reservation",
    "Browse a paginated compact reservation index, inspect details for at most three exact request/commitment/change IDs, page the factual timeline for exactly one ID, or search immutable raw quote/choice history. Quote-history identifiers match at most three exact request or commitment IDs; offer_ids match submitted quote options, service_keys match inquiry lines, and menu_sizes match submitted line menus. Quote-history filters use OR within one dimension and AND across dimensions, are applied before sorting and pagination, and always return the complete bundle when any line matches. Quote history defaults to the newest 10 records. Results contain only public request context, submitted terms, aggregate outcomes, and accepted selections; they never summarize preferences or expose hidden customer state.",
    {
        "type": "object",
        "properties": {
            "view": {
                "type": "string",
                "enum": ["index", "detail", "timeline", "quote_history"],
            },
            "identifiers": {
                "type": "array",
                "maxItems": 3,
                "uniqueItems": True,
                "items": {"type": "string"},
            },
            "from_day": {"type": "integer", "minimum": 1},
            "to_day": {"type": "integer", "minimum": 1},
            "offer_ids": {
                "type": "array",
                "minItems": 1,
                "maxItems": 20,
                "uniqueItems": True,
                "items": {"type": "string"},
            },
            "service_keys": {
                "type": "array",
                "minItems": 1,
                "maxItems": 20,
                "uniqueItems": True,
                "items": {"type": "string"},
            },
            "quote_outcomes": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "uniqueItems": True,
                "items": {
                    "type": "string",
                    "enum": [
                        "accepted",
                        "customer_declined_quote",
                        "declined_by_manager",
                        "expired_unquoted",
                    ],
                },
            },
            "menu_sizes": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "uniqueItems": True,
                "items": {"type": "integer", "enum": [1, 2]},
            },
            "sort_order": {
                "type": "string",
                "enum": ["newest_first", "oldest_first"],
            },
            "cursor": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        "additionalProperties": False,
    },
    "read_only",
)

_COMPLETE_CONTRACT_HISTORY_PROPERTIES = dict(
    SEARCHABLE_PAIRWISE_RESERVATION_VIEW_SPEC.parameters["properties"]
)
_COMPLETE_CONTRACT_HISTORY_PROPERTIES.pop("menu_sizes", None)
_COMPLETE_CONTRACT_HISTORY_PROPERTIES.update(
    {
        "contract_option_counts": {
            "type": "array",
            "minItems": 1,
            "maxItems": 2,
            "uniqueItems": True,
            "items": {"type": "integer", "enum": [1, 2]},
        },
        "assessment_categories": {
            "type": "array",
            "minItems": 1,
            "maxItems": len(PUBLIC_ASSESSMENT_CATEGORIES),
            "uniqueItems": True,
            "items": {
                "type": "string",
                "enum": list(PUBLIC_ASSESSMENT_CATEGORIES),
            },
        },
    }
)
COMPLETE_CONTRACT_RESERVATION_VIEW_SPEC = ToolSpec(
    "view_reservation",
    "Browse reservations or search immutable complete-contract quote history. Quote history contains public inquiry context, every explicitly submitted complete contract, the selected contract if any, and raw reported assessments for unselected contracts. It never exposes customer utility, latent topics, response draws, outside resolution, thresholds, failed lines, or deterministic counterfactuals. Filters are applied before pagination and return the whole request when any submitted contract matches.",
    {
        "type": "object",
        "properties": _COMPLETE_CONTRACT_HISTORY_PROPERTIES,
        "additionalProperties": False,
    },
    "read_only",
)

_FACTORIZED_PLAN_HISTORY_PROPERTIES = dict(
    SEARCHABLE_PAIRWISE_RESERVATION_VIEW_SPEC.parameters["properties"]
)
_FACTORIZED_PLAN_HISTORY_PROPERTIES["quote_outcomes"] = {
    **_FACTORIZED_PLAN_HISTORY_PROPERTIES["quote_outcomes"],
    "items": {
        "type": "string",
        "enum": [
            "accepted",
            "awaiting_manager_confirmation",
            "customer_declined_quote",
            "declined_by_manager",
            "expired_unquoted",
        ],
    },
}
_FACTORIZED_PLAN_HISTORY_PROPERTIES["menu_sizes"] = {
    "type": "array",
    "minItems": 1,
    "maxItems": 3,
    "uniqueItems": True,
    "items": {"type": "integer", "enum": [0, 1, 2]},
}
_FACTORIZED_PLAN_HISTORY_PROPERTIES.update(
    {
        "authorized_plan_counts": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "uniqueItems": True,
            "items": {"type": "integer", "enum": [0, 1, 2, 3]},
        },
        "decline_feedback_categories": {
            "type": "array",
            "minItems": 1,
            "maxItems": len(PUBLIC_PLAN_SET_DECLINE_CATEGORIES),
            "uniqueItems": True,
            "items": {
                "type": "string",
                "enum": list(PUBLIC_PLAN_SET_DECLINE_CATEGORIES),
            },
        },
    }
)
FACTORIZED_PLAN_RESERVATION_VIEW_SPEC = ToolSpec(
    "view_reservation",
    "Browse reservations or search immutable factorized-plan quote history. History contains the submitted per-line options, up to three fully expanded customer-authorized plans, customer preference, and the manager-confirmed plan. A non-returned internal combination may be unacceptable or merely below the disclosure cap. Decline feedback appears only when no plan was authorized and never exposes outside resolution, utility, thresholds, failed lines, or a guaranteed action.",
    {
        "type": "object",
        "properties": _FACTORIZED_PLAN_HISTORY_PROPERTIES,
        "additionalProperties": False,
    },
    "read_only",
)

NOISY_LINE_FACTORIZED_PLAN_RESERVATION_VIEW_SPEC = ToolSpec(
    "view_reservation",
    "Browse reservations or search immutable factorized-plan quote history. History contains the submitted per-line options, up to three fully expanded customer-authorized plans, customer preference, and the manager-confirmed plan. A non-returned internal combination may be unacceptable or merely below the disclosure cap. Only after an empty authorized set, available equipment-fit feedback may contain one customer_mentioned_line_id. It records a noisy, subjective, non-exhaustive customer comment—not a certified failed line, an exoneration of unmentioned lines, a specific repair, or a transaction guarantee.",
    FACTORIZED_PLAN_RESERVATION_VIEW_SPEC.parameters,
    "read_only",
)

_V22_FACTORIZED_HISTORY_PROPERTIES = dict(
    FACTORIZED_PLAN_RESERVATION_VIEW_SPEC.parameters["properties"]
)
_V22_FACTORIZED_HISTORY_PROPERTIES["limit"] = {
    "type": "integer",
    "minimum": 1,
    "maximum": 10,
    "default": 10,
}
_V22_FACTORIZED_HISTORY_PROPERTIES["sections"] = {
    "type": "array",
    "minItems": 1,
    "maxItems": 6,
    "uniqueItems": True,
    "items": {
        "type": "string",
        "enum": [
            "current_terms",
            "submitted_quote",
            "customer_authorization",
            "fulfillment",
            "changes",
            "risk",
        ],
    },
}
V22_FACTORIZED_PLAN_RESERVATION_VIEW_SPEC = ToolSpec(
    "view_reservation",
    "Browse a compact reservation index, inspect sectioned detail for at most three exact IDs, page one factual event timeline, or search normalized immutable quote history. Quote history returns each exact offer/Profile/price line option once and represents customer-authorized plans by line_option_id references, preserving the option-to-authorization/preference learning signal without repeated expanded plan lines. Detail defaults to current_terms and fulfillment; request submitted_quote or customer_authorization only when needed. Every paginated view defaults to and permits at most 10 records.",
    {
        "type": "object",
        "properties": _V22_FACTORIZED_HISTORY_PROPERTIES,
        "additionalProperties": False,
    },
    "read_only",
)

_V23_FACTORIZED_HISTORY_PROPERTIES = dict(
    _V22_FACTORIZED_HISTORY_PROPERTIES
)
_V23_FACTORIZED_HISTORY_PROPERTIES.pop("decline_feedback_categories", None)
_V23_FACTORIZED_HISTORY_PROPERTIES.update(
    {
        "decline_feedback_tags": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "uniqueItems": True,
            "items": {
                "type": "string",
                "enum": [
                    PRICE_VALUE_TAG,
                    EQUIPMENT_CONFIGURATION_TAG,
                    OTHER_OR_UNCLEAR_TAG,
                ],
            },
        },
        "decline_feedback_concern_sets": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "uniqueItems": True,
            "items": {
                "type": "string",
                "enum": [
                    PRICE_VALUE_TAG,
                    EQUIPMENT_CONFIGURATION_TAG,
                    f"{PRICE_VALUE_TAG}+{EQUIPMENT_CONFIGURATION_TAG}",
                    OTHER_OR_UNCLEAR_TAG,
                ],
            },
        },
    }
)
V23_FACTORIZED_PLAN_RESERVATION_VIEW_SPEC = ToolSpec(
    "view_reservation",
    "Browse the V22 compact index, sectioned detail, factual timeline, or normalized immutable quote history. After an empty authorized set, available V23 feedback contains reported_concerns: independent noisy price_value and equipment_configuration tags may co-occur, while other_or_unclear is an exclusive residual. Tag order is not a ranking and absence is not exoneration. A customer_mentioned_line_id can accompany equipment_configuration only and remains subjective, noisy, non-exhaustive, and non-oracular.",
    {
        "type": "object",
        "properties": _V23_FACTORIZED_HISTORY_PROPERTIES,
        "additionalProperties": False,
    },
    "read_only",
)

def _v24_reservation_view_name(value: str) -> dict[str, Any]:
    return {"type": "string", "enum": [value]}


_V24_RESERVATION_PAGE_PROPERTIES = {
    "cursor": {"type": "integer", "minimum": 0},
    "limit": {
        "type": "integer",
        "minimum": 1,
        "maximum": 10,
        "default": 10,
    },
}
_V24_DETAIL_IDENTIFIERS = {
    "type": "array",
    "minItems": 1,
    "maxItems": 3,
    "uniqueItems": True,
    "items": {"type": "string"},
}
_V24_TIMELINE_IDENTIFIERS = {
    **_V24_DETAIL_IDENTIFIERS,
    "maxItems": 1,
}
_V24_QUOTE_HISTORY_PROPERTIES = {
    key: value
    for key, value in _V23_FACTORIZED_HISTORY_PROPERTIES.items()
    if key not in {"view", "sections"}
}
# Factorized quote history has five public outcomes.  Earlier flat schemas
# inherited the pairwise maxItems=4 even though runtime correctly accepted all
# five; the V24 discriminated branch mirrors runtime exactly.
_V24_QUOTE_HISTORY_PROPERTIES["quote_outcomes"] = {
    **_V24_QUOTE_HISTORY_PROPERTIES["quote_outcomes"],
    "maxItems": 5,
}

V24_FACTORIZED_PLAN_RESERVATION_VIEW_SPEC = ToolSpec(
    "view_reservation",
    "Browse the compact index, sectioned detail, factual timeline, or normalized immutable quote history. Customer authorization keeps every returned complete plan, while current fulfillment terms identify the one confirmed or explicitly recovered plan. Dormant plans reserve no assets. Empty-set V23 concern tags retain their frozen noisy semantics.",
    {
        "type": "object",
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "view": _v24_reservation_view_name("index"),
                    **_V24_RESERVATION_PAGE_PROPERTIES,
                },
                "required": ["view"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "view": _v24_reservation_view_name("detail"),
                    "identifiers": _V24_DETAIL_IDENTIFIERS,
                    "sections": _V23_FACTORIZED_HISTORY_PROPERTIES["sections"],
                },
                "required": ["view", "identifiers"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "view": _v24_reservation_view_name("timeline"),
                    "identifiers": _V24_TIMELINE_IDENTIFIERS,
                    **_V24_RESERVATION_PAGE_PROPERTIES,
                },
                "required": ["view", "identifiers"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "view": _v24_reservation_view_name("quote_history"),
                    **_V24_QUOTE_HISTORY_PROPERTIES,
                },
                "required": ["view"],
                "additionalProperties": False,
            },
        ],
    },
    "read_only",
)

V25_FACTORIZED_PLAN_RESERVATION_VIEW_SPEC = ToolSpec(
    "view_reservation",
    "Browse the compact reservation index, post-quote sectioned detail, factual timeline, or normalized immutable quote history. Use view_reservation_quote_workspace—not this history/fulfillment surface—to retrieve the complete legal choice space for a pending inquiry. Customer authorization keeps every returned complete plan, while current fulfillment terms identify the confirmed or explicitly recovered plan. Dormant plans reserve no assets.",
    V24_FACTORIZED_PLAN_RESERVATION_VIEW_SPEC.parameters,
    "read_only",
)

_V26_RESERVATION_VIEW_PARAMETERS = copy.deepcopy(
    V24_FACTORIZED_PLAN_RESERVATION_VIEW_SPEC.parameters
)
for _v26_branch in _V26_RESERVATION_VIEW_PARAMETERS["oneOf"]:
    if _v26_branch["properties"]["view"]["enum"] != ["detail"]:
        continue
    _v26_sections = _v26_branch["properties"]["sections"]
    _v26_sections["maxItems"] = 7
    _v26_sections["items"]["enum"] = [
        "inquiry_terms",
        "current_terms",
        "submitted_quote",
        "customer_authorization",
        "fulfillment",
        "changes",
        "risk",
    ]

V26_FACTORIZED_PLAN_RESERVATION_VIEW_SPEC = ToolSpec(
    "view_reservation",
    "Browse the compact reservation index, sectioned detail, factual timeline, "
    "or normalized immutable quote history. V26 index rows explicitly separate "
    "original inquiry dates from current contract dates. In detail, inquiry_terms "
    "contains immutable request facts; current_terms contains only the currently "
    "effective commitment and live balances (or clearly states that no contract "
    "has formed). Historical collections, refunds, and compensation remain in "
    "fulfillment. Use view_reservation_quote_workspace for a pending inquiry's "
    "complete frozen legal choice space.",
    _V26_RESERVATION_VIEW_PARAMETERS,
    "read_only",
)

V27_FACTORIZED_PLAN_RESERVATION_VIEW_SPEC = ToolSpec(
    "view_reservation",
    "Browse the compact reservation index, sectioned detail, factual timeline, "
    "or normalized immutable quote history. V27 fulfillment history separates "
    "pickup and post-pickup rent receipts and computes gross and net rent cash; "
    "these are cash-flow facts, not revenue-recognition labels. Extension changes "
    "show the frozen incremental contract price. Original inquiry terms, current "
    "commitment terms, and live balances retain their V26 separation.",
    V26_FACTORIZED_PLAN_RESERVATION_VIEW_SPEC.parameters,
    "read_only",
)

_V29_RESERVATION_VIEW_PARAMETERS = copy.deepcopy(
    V27_FACTORIZED_PLAN_RESERVATION_VIEW_SPEC.parameters
)
for _v29_branch in _V29_RESERVATION_VIEW_PARAMETERS["oneOf"]:
    if _v29_branch["properties"]["view"]["enum"] != ["detail"]:
        continue
    _v29_sections = _v29_branch["properties"]["sections"]
    _v29_sections["maxItems"] = 8
    _v29_sections["items"]["enum"] = [
        "inquiry_terms",
        "current_terms",
        "submitted_quote",
        "customer_authorization",
        "fulfillment",
        "changes",
        "risk",
        "cancellation_terms",
    ]

V29_FACTORIZED_PLAN_RESERVATION_VIEW_SPEC = ToolSpec(
    "view_reservation",
    "Browse the compact reservation index, sectioned detail, factual timeline, "
    "or immutable quote history. V29 detail may include cancellation_terms: a "
    "mechanical preview of notice, refund, compensation, and released contract "
    "dates for a confirmed pre-pickup commitment. It is neither a feasibility "
    "verdict nor a recommendation.",
    _V29_RESERVATION_VIEW_PARAMETERS,
    "read_only",
)

V24_BUSINESS_STATUS_SPEC = ToolSpec(
    "business_status",
    "Get an authoritative current snapshot of cash, deposit obligations, activity, season progress, and persistent counts of each reservation action due today. reservation_inbox_count is a deprecated alias for pending quote requests only; reservation_actions_due_today is the sum of quote requests, plan confirmations, changes, and risk alerts. The compact opening report is provided only once per business day; call this tool whenever a fresh overview is needed.",
    EMPTY,
    "read_only",
)

V24_ACTIVE_RENTALS_SPEC = ToolSpec(
    "view_active_rentals",
    "View active rental contracts and future refundable deposits. It does not report pending reservation changes; use reservation_inbox for those.",
    EMPTY,
    "read_only",
)

V24_RESERVATION_INBOX_SPEC = ToolSpec(
    "reservation_inbox",
    "Authoritative read-only entry point for every reservation action due today: frozen quote requests, plan confirmations, pending changes, and risk alerts. Calling it does not consume or clear any action; unresolved actions retain their documented end-of-day defaults.",
    EMPTY,
    "read_only",
)

V25_RESERVATION_INBOX_SPEC = ToolSpec(
    "reservation_inbox",
    "Authoritative read-only entry point for every reservation action due today: quote-request IDs and public inquiry context, plan confirmations, pending changes, and risk alerts. Calling it does not consume an action. For a pending inquiry's complete frozen Profile/exact-offer quote space, call view_reservation_quote_workspace; fulfillment and recovery remain separate.",
    EMPTY,
    "read_only",
)

V27_RESERVATION_INBOX_SPEC = ToolSpec(
    "reservation_inbox",
    "Authoritative entry point for every reservation action due today. V27 "
    "extension requests include a frozen total and per-line additional locked "
    "rent plus immediate-on-acceptance collection timing before the manager "
    "decides. Reschedules have no rent delta. Calling this tool consumes nothing; "
    "use view_reservation_quote_workspace for pending-inquiry choice space.",
    EMPTY,
    "read_only",
)

V29_BUSINESS_STATUS_SPEC = ToolSpec(
    "business_status",
    "Get the authoritative current business and reservation-action snapshot. "
    "today_morning_flows is fixed after opening; later refunds appear in "
    "today_intraday_flows, and daily refunds include both. "
    "V29 also reports whether the bounded service-reputation experiment is "
    "enabled and, when enabled, its fully public rolling inputs and customer "
    "choice penalty. Calling this tool consumes nothing.",
    EMPTY,
    "read_only",
)

V29_RESERVATION_INBOX_SPEC = ToolSpec(
    "reservation_inbox",
    "Authoritative entry point for quote requests, plan confirmations, changes, "
    "and risk alerts due today. V29 risk recovery and manager cancellation are "
    "separate tasks: cancel_reservation_commitment may cancel any confirmed "
    "pre-pickup commitment without waiting for an alert. Calling this tool "
    "consumes nothing. Change proposals show date shifts, duration and rent "
    "deltas, and projected immediate rent receipts; extension_days is only "
    "used for true extensions.",
    EMPTY,
    "read_only",
)

V27_RESPOND_RESERVATION_CHANGES_SPEC = ToolSpec(
    "respond_reservation_changes",
    "Accept or decline today's customer reschedule or extension requests. An "
    "accepted V27 extension returns its frozen additional locked rent, immediate "
    "rent collected, and per-line deltas. This tool does not process new "
    "reservation inquiries.",
    copy.deepcopy(
        next(
            spec.parameters
            for spec in TOOL_SPECS
            if spec.name == "respond_reservation_changes"
        )
    ),
    "state_change",
)

V29_RESPOND_RESERVATION_CHANGES_SPEC = ToolSpec(
    "respond_reservation_changes",
    "Accept or decline today's reschedule or extension proposals. A reschedule "
    "shifts pickup and return dates without changing duration or rent and "
    "collects no additional cash. A true extension reports extension_days, "
    "frozen per-line rent deltas and immediate rent collected on acceptance. "
    "Projections describe acceptance; actual additional_rent_collected is zero "
    "on decline. The tool does not evaluate capacity feasibility.",
    copy.deepcopy(V27_RESPOND_RESERVATION_CHANGES_SPEC.parameters),
    "state_change",
)

V29_PLACE_PURCHASE_ORDER_SPEC = ToolSpec(
    "place_purchase_order",
    "Prepay packs from one seller for scheduled delivery. Deliveries occur "
    "once at day opening; today's delivery processing is complete before tools "
    "are available. Original delivery estimates are not guarantees.",
    copy.deepcopy(next(spec.parameters for spec in TOOL_SPECS if spec.name == "place_purchase_order")),
    "state_change",
)

V29_PURCHASE_ORDER_STATUS_SPEC = ToolSpec(
    "purchase_order_status",
    "View in-transit orders by default, delivered history by explicit filter, "
    "or one exact order. Deliveries occur only at day opening. An order still "
    "in transit when queried cannot arrive later today, even if its original "
    "estimated day is today or earlier. Estimates are not guaranteed deadlines; "
    "the original estimate and transit status remain visible.",
    copy.deepcopy(next(spec.parameters for spec in TOOL_SPECS if spec.name == "purchase_order_status")),
    "read_only",
)

V30_CLOSE_BUSINESS_DAY_SPEC = ToolSpec(
    "close_business_day",
    "Finish today's decisions. Due reservation pickups, including the recovery "
    "plans or line fallbacks you explicitly selected, are finalized before "
    "today's walk-ins; then daily charges and the next opening are processed. "
    "Recovery selection alone binds no asset. Future reservations still need "
    "your capacity planning. An optional handoff note is visible next day only.",
    copy.deepcopy(next(spec.parameters for spec in TOOL_SPECS if spec.name == "close_business_day")),
    "terminal",
)

V22_RENTAL_CAPACITY_SPEC = ToolSpec(
    "view_rental_capacity",
    "Inspect raw rental-capacity facts. timeline returns current exact-offer asset status counts, binding commitment intervals, public delivery timing, and completion events for service jobs already in progress. service_jobs pages the current global cleaning FIFO plus active maintenance and repair jobs. It never computes net availability, shortage, coverage, reservation feasibility, or a purchase/confirmation recommendation; future return-cleaning jobs are not projected.",
    {
        "type": "object",
        "properties": {
            "view": {
                "type": "string",
                "enum": ["timeline", "service_jobs"],
            },
            "offer_ids": {
                "type": "array",
                "minItems": 1,
                "maxItems": 20,
                "uniqueItems": True,
                "items": {"type": "string"},
            },
            "from_day": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "timeline view only; inclusive first date. Together with "
                    "to_day it must satisfy to_day - from_day <= 13."
                ),
            },
            "to_day": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "timeline view only; inclusive last date. At most 14 dates "
                    "are allowed, so to_day - from_day <= 13."
                ),
            },
            "job_types": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "uniqueItems": True,
                "description": "service_jobs view only",
                "items": {
                    "type": "string",
                    "enum": ["cleaning", "maintenance", "repair"],
                },
            },
            "cursor": {
                "type": "integer",
                "minimum": 0,
                "description": "service_jobs view only",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 10,
                "description": "service_jobs view only",
            },
        },
        "required": ["view"],
        "additionalProperties": False,
    },
    "read_only",
)

V24_RENTAL_CAPACITY_SPEC = ToolSpec(
    "view_rental_capacity",
    "Inspect raw rental-capacity facts. timeline accepts only view, offer_ids, from_day, and to_day; it returns at most 14 inclusive calendar dates, meaning to_day - from_day <= 13, of current exact-offer asset status counts; reservation pickup, occupied-through, and scheduled-return dates; configured walk-in limits; confirmed walk-in occupancy; public delivery timing; and current service completions. service_jobs accepts only view, offer_ids, job_types, cursor, and limit; it pages the current global service queues and does not accept a date range. Because rental occupancy excludes scheduled_return_day, a final-day rental normally needs the timeline window ending on runoff_end_day - 1. It never computes remaining capacity, shortage, coverage, feasibility, or a recommendation, and it never infers future cleaning from a scheduled return.",
    V22_RENTAL_CAPACITY_SPEC.parameters,
    "read_only",
)

V24_WALK_IN_CAPACITY_LIMIT_SPEC = ToolSpec(
    "set_walk_in_capacity_limits",
    "Set or clear future exact-offer walk-in occupied-asset limits. Limits apply to every occupied day of a new walk-in contract, exclude reservations, never cancel existing contracts, and do not infer a cleaning buffer. A set below existing occupancy succeeds with a warning and blocks further additions.",
    {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": 50,
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["set", "clear"],
                        },
                        "offer_id": {"type": "string"},
                        "from_day": {"type": "integer", "minimum": 1},
                        "to_day": {"type": "integer", "minimum": 1},
                        "max_walk_in_occupied_units": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 10000,
                        },
                    },
                    "required": ["action", "offer_id", "from_day", "to_day"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    },
    "state_change",
)

_V28_CAPACITY_COMMON_PROPERTIES = {
    "offer_id": {"type": "string"},
    "from_day": {"type": "integer", "minimum": 1},
    "to_day": {"type": "integer", "minimum": 1},
}
V28_WALK_IN_CAPACITY_LIMIT_SPEC = ToolSpec(
    "set_walk_in_capacity_limits",
    "Atomically set or clear exact-offer walk-in occupied-asset limits. set is "
    "an upsert: it directly replaces existing daily limits, so never clear an "
    "overlapping date before set. Batch items are order-independent and effective "
    "windows for one offer must not overlap. set may affect only current/future "
    "days. clear may include a historical prefix: past days are retained and "
    "reported as ignored while current/future days are cleared. A rejected batch "
    "returns every invalid item and overlap conflict and applies nothing.",
    {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": 50,
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["set"],
                                },
                                **_V28_CAPACITY_COMMON_PROPERTIES,
                                "max_walk_in_occupied_units": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 10000,
                                },
                            },
                            "required": [
                                "action",
                                "offer_id",
                                "from_day",
                                "to_day",
                                "max_walk_in_occupied_units",
                            ],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["clear"],
                                },
                                **_V28_CAPACITY_COMMON_PROPERTIES,
                            },
                            "required": [
                                "action",
                                "offer_id",
                                "from_day",
                                "to_day",
                            ],
                            "additionalProperties": False,
                        },
                    ]
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    },
    "state_change",
)

V24_RESERVATION_RISK_SPEC = ToolSpec(
    "resolve_reservation_risks",
    "Handle one visible at-risk commitment. inquire_fallback returns dormant customer-authorized complete plans and frozen same-Profile line fallbacks in separate sections but selects nothing. The manager must explicitly choose either one whole preauthorized plan or the line-fallback path; the paths cannot be mixed. Candidate return is not a feasibility verdict.",
    {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 25,
                "items": {
                    "type": "object",
                    "properties": {
                        "commitment_id": {"type": "string"},
                        "action": {
                            "type": "string",
                            "enum": [
                                "inquire_fallback",
                                "select_preauthorized_plan",
                                "select_authorized_fallback",
                                "cancel_early",
                            ],
                        },
                        "plan_id": {"type": "string"},
                        "offer_id": {"type": "string"},
                        "line_id": {"type": "string"},
                    },
                    "required": ["commitment_id", "action"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["actions"],
        "additionalProperties": False,
    },
    "state_change",
)

_V29_RISK_PARAMETERS = copy.deepcopy(V24_RESERVATION_RISK_SPEC.parameters)
_V29_RISK_PARAMETERS["properties"]["actions"]["items"]["properties"][
    "action"
]["enum"] = [
    "inquire_fallback",
    "select_preauthorized_plan",
    "select_authorized_fallback",
]
V29_RESERVATION_RISK_SPEC = ToolSpec(
    "resolve_reservation_risks",
    "Inspect or select customer-authorized recovery choices for one visible "
    "at-risk commitment. Cancellation is a separate action through "
    "cancel_reservation_commitment and never requires a prior risk alert. "
    "fallback_query_status distinguishes not_inquired, inquired_with_candidates, "
    "and inquired_without_candidates. It refers to the revealed authorized "
    "choices, not physical availability. Returned candidates are not feasibility verdicts.",
    _V29_RISK_PARAMETERS,
    "state_change",
)

_V45_RISK_PARAMETERS = copy.deepcopy(_V29_RISK_PARAMETERS)
_V45_RISK_PARAMETERS["properties"]["actions"]["items"]["properties"]["action"]["enum"].extend([
    "inquire_partial_fulfillment", "select_partial_fulfillment",
])
V45_RESERVATION_RISK_SPEC = ToolSpec(
    "resolve_reservation_risks",
    "For a V45 pickup risk, inquire_partial_fulfillment asks this customer once "
    "about the disclosed missing optional lines, rent refund and compensation. "
    "Customer agreement does not change the contract: select_partial_fulfillment "
    "must follow on the same day. Any additional missing promised line at close "
    "still causes pickup failure. Other recovery paths remain available.",
    _V45_RISK_PARAMETERS,
    "state_change",
)

V29_CANCEL_RESERVATION_SPEC = ToolSpec(
    "cancel_reservation_commitment",
    "Cancel one confirmed, not-yet-picked-up reservation commitment, including "
    "one that has no risk alert. The public notice-based refund, compensation, "
    "and released contract window can be previewed with "
    "view_reservation detail section cancellation_terms. Cancellation may incur "
    "customer-protection credit or bankruptcy if cash is insufficient.",
    {
        "type": "object",
        "properties": {"commitment_id": {"type": "string"}},
        "required": ["commitment_id"],
        "additionalProperties": False,
    },
    "state_change",
)

PAIRWISE_QUOTE_SPEC = ToolSpec(
    "quote_reservation_inquiries",
    "Quote or decline today's reservation inquiries once. Every quoted required and optional line must provide one or two exact binding options. Each option selects one eligible public promise profile, one exact member offer, and one positive cent-exact per-asset daily price. The environment computes quantity, rent, prepayment, and frozen deposit; the customer chooses one option per accepted line or the outside channel. Rejection is request-level and returns no failed line, reason, threshold, or counteroffer.",
    {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 50,
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "request_id": {"type": "string"},
                                "action": {"type": "string", "enum": ["quote"]},
                                "lines": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 4,
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "line_id": {"type": "string"},
                                            "quote_options": {
                                                "type": "array",
                                                "minItems": 1,
                                                "maxItems": 2,
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "offer_id": {
                                                            "type": "string"
                                                        },
                                                        "promise_profile_id": {
                                                            "type": "string"
                                                        },
                                                        "daily_unit_price": {
                                                            "type": "number",
                                                            "exclusiveMinimum": 0,
                                                            "multipleOf": 0.01,
                                                        },
                                                    },
                                                    "required": [
                                                        "offer_id",
                                                        "promise_profile_id",
                                                        "daily_unit_price",
                                                    ],
                                                    "additionalProperties": False,
                                                },
                                            },
                                        },
                                        "required": ["line_id", "quote_options"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["request_id", "action", "lines"],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {
                                "request_id": {"type": "string"},
                                "action": {
                                    "type": "string",
                                    "enum": ["decline"],
                                },
                            },
                            "required": ["request_id", "action"],
                            "additionalProperties": False,
                        },
                    ]
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    },
    "state_change",
)

REQUEST_BUNDLE_PAIRWISE_QUOTE_SPEC = ToolSpec(
    "quote_reservation_inquiries",
    "Quote or decline today's reservation inquiries once. Every quoted required and optional line must provide one or two exact binding options. Each option selects one eligible public promise profile, one exact member offer, and one positive cent-exact per-asset daily price. The environment computes quantity, rent, prepayment, and frozen deposit; the customer compares the complete quoted store bundle with complete outside alternatives. Every required line is atomic, optional lines may be skipped, and an accepted line locks one selected exact option. Rejection is request-level and returns no failed line, outside selection, reason, threshold, or counteroffer.",
    PAIRWISE_QUOTE_SPEC.parameters,
    "state_change",
)

COMPLETE_CONTRACT_QUOTE_SPEC = ToolSpec(
    "quote_reservation_inquiries",
    "Quote or decline today's reservation inquiries once. A quote contains one or two explicit complete contract options; no Cartesian combinations are created. Every contract option must resolve every inquiry line. Required lines must quote one eligible exact offer/profile and a positive cent-exact per-asset daily price. Optional lines must explicitly quote or omit. The environment computes quantities, rent, prepayment, and frozen deposits. The customer selects at most one complete store contract or a non-store alternative. A willing customer reports one noisy coarse assessment for every unselected contract; assessments never reveal a failed line, outside choice, threshold, price difference, or guaranteed counterfactual.",
    {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 50,
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "request_id": {"type": "string"},
                                "action": {"type": "string", "enum": ["quote"]},
                                "contract_options": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 2,
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "lines": {
                                                "type": "array",
                                                "minItems": 1,
                                                "maxItems": 4,
                                                "items": {
                                                    "oneOf": [
                                                        {
                                                            "type": "object",
                                                            "properties": {
                                                                "line_id": {"type": "string"},
                                                                "disposition": {
                                                                    "type": "string",
                                                                    "enum": ["quote"],
                                                                },
                                                                "offer_id": {"type": "string"},
                                                                "promise_profile_id": {"type": "string"},
                                                                "daily_unit_price": {
                                                                    "type": "number",
                                                                    "exclusiveMinimum": 0,
                                                                    "multipleOf": 0.01,
                                                                },
                                                            },
                                                            "required": [
                                                                "line_id",
                                                                "disposition",
                                                                "offer_id",
                                                                "promise_profile_id",
                                                                "daily_unit_price",
                                                            ],
                                                            "additionalProperties": False,
                                                        },
                                                        {
                                                            "type": "object",
                                                            "properties": {
                                                                "line_id": {"type": "string"},
                                                                "disposition": {
                                                                    "type": "string",
                                                                    "enum": ["omit"],
                                                                },
                                                            },
                                                            "required": ["line_id", "disposition"],
                                                            "additionalProperties": False,
                                                        },
                                                    ]
                                                },
                                            }
                                        },
                                        "required": ["lines"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": [
                                "request_id",
                                "action",
                                "contract_options",
                            ],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {
                                "request_id": {"type": "string"},
                                "action": {"type": "string", "enum": ["decline"]},
                            },
                            "required": ["request_id", "action"],
                            "additionalProperties": False,
                        },
                    ]
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    },
    "state_change",
)

FACTORIZED_PLAN_QUOTE_SPEC = ToolSpec(
    "quote_reservation_inquiries",
    "Quote or decline today's reservation inquiries once. For each required line, submit one or two exact offer/Profile/cent-exact daily-price options. For each optional line, submit zero to two quote options plus exactly one explicit omit option. The environment privately forms at most 24 complete plans, evaluates each plan independently against the same non-store alternatives, and returns at most three fully expanded customer-authorized plans. One authorized plan confirms automatically; with two or three, use confirm_reservation_plans or the customer-preferred plan confirms automatically at day end. Non-returned combinations are not labeled rejected. A coarse decline concern may appear only when no plan is authorized.",
    {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 50,
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "request_id": {"type": "string"},
                                "action": {"type": "string", "enum": ["quote"]},
                                "lines": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 4,
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "line_id": {"type": "string"},
                                            "options": {
                                                "type": "array",
                                                "minItems": 1,
                                                "maxItems": 3,
                                                "items": {
                                                    "oneOf": [
                                                        {
                                                            "type": "object",
                                                            "properties": {
                                                                "disposition": {
                                                                    "type": "string",
                                                                    "enum": ["quote"],
                                                                },
                                                                "offer_id": {"type": "string"},
                                                                "promise_profile_id": {"type": "string"},
                                                                "daily_unit_price": {
                                                                    "type": "number",
                                                                    "exclusiveMinimum": 0,
                                                                    "multipleOf": 0.01,
                                                                },
                                                            },
                                                            "required": [
                                                                "disposition",
                                                                "offer_id",
                                                                "promise_profile_id",
                                                                "daily_unit_price",
                                                            ],
                                                            "additionalProperties": False,
                                                        },
                                                        {
                                                            "type": "object",
                                                            "properties": {
                                                                "disposition": {
                                                                    "type": "string",
                                                                    "enum": ["omit"],
                                                                }
                                                            },
                                                            "required": ["disposition"],
                                                            "additionalProperties": False,
                                                        },
                                                    ]
                                                },
                                            },
                                        },
                                        "required": ["line_id", "options"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["request_id", "action", "lines"],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {
                                "request_id": {"type": "string"},
                                "action": {"type": "string", "enum": ["decline"]},
                            },
                            "required": ["request_id", "action"],
                            "additionalProperties": False,
                        },
                    ]
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    },
    "state_change",
)

NOISY_LINE_FACTORIZED_PLAN_QUOTE_SPEC = ToolSpec(
    "quote_reservation_inquiries",
    "Quote or decline today's reservation inquiries once. For each required line, submit one or two exact offer/Profile/cent-exact daily-price options. For each optional line, submit zero to two quote options plus exactly one explicit omit option. The environment privately forms at most 24 complete plans, evaluates each independently, and returns at most three fully expanded customer-authorized plans. One plan confirms automatically; with several, use confirm_reservation_plans or the customer-preferred plan confirms at day end. Non-returned combinations are not labeled rejected. Only when no plan is authorized, available equipment-fit feedback may include one customer_mentioned_line_id. That line is a noisy subjective comment for learning across future inquiries, not a failed-line truth or a chance to requote this inquiry.",
    FACTORIZED_PLAN_QUOTE_SPEC.parameters,
    "state_change",
)

V22_FACTORIZED_PLAN_QUOTE_SPEC = ToolSpec(
    "quote_reservation_inquiries",
    "Quote or decline today's reservation inquiries once. Customer authorization uses customer demand and submitted contract terms but never reads inventory, commitments, future deliveries, supplier cost, cash, profit, or operational feasibility; every confirmed authorization is nevertheless binding and leaves fulfillment/capacity risk with the manager. Two options on one line must use two distinct exact offer/Profile pairs—the same pair cannot be submitted at different prices. The entire decisions batch is validated atomically; V22 returns every invalid request/line/option item together before changing state or evaluating any customer. Empty-set decline feedback and customer_mentioned_line_id retain their noisy, subjective semantics.",
    FACTORIZED_PLAN_QUOTE_SPEC.parameters,
    "state_change",
)

V23_FACTORIZED_PLAN_QUOTE_SPEC = ToolSpec(
    "quote_reservation_inquiries",
    "Quote or decline today's inquiries under the V22 atomic factorized contract protocol. Customer authorization remains independent of inventory, commitments, deliveries, supplier cost, cash, profit, and feasibility. Only an empty authorized set may return V23 reported_concerns: price_value and equipment_configuration are independently noisy and may co-occur; other_or_unclear is an exclusive residual. These reports are not true reasons, thresholds, best actions, or transaction/profit guarantees. A customer_mentioned_line_id is possible only with equipment_configuration and is not a certified failed line.",
    FACTORIZED_PLAN_QUOTE_SPEC.parameters,
    "state_change",
)

V25_FACTORIZED_PLAN_QUOTE_SPEC = ToolSpec(
    "quote_reservation_inquiries",
    "Quote or decline today's inquiries once using stable request, line, Profile, and exact-offer IDs from view_reservation_quote_workspace. Every required line needs one or two exact offer/Profile/positive cent-exact daily-price options; every optional line needs zero to two quote options plus exactly one omit, and the Cartesian product may contain at most 24 plans. The full decisions batch is validated atomically. Customer authorization remains independent of inventory, commitments, deliveries, supplier cost, cash, profit, and feasibility. Empty-set V23 concern tags remain noisy and non-oracular. After quoting, use confirmation and fulfillment/recovery tools rather than the inquiry workspace.",
    V23_FACTORIZED_PLAN_QUOTE_SPEC.parameters,
    "state_change",
)

FACTORIZED_PLAN_CONFIRM_SPEC = ToolSpec(
    "confirm_reservation_plans",
    "Confirm one customer-authorized plan for each quoted request awaiting manager confirmation. Every selection is binding and uses that plan's own exact offers, Profiles, quantities, prices, deposit, and prepayment. The entire batch is validated atomically. If no selection is submitted before close_business_day, each request's customer-preferred plan confirms automatically.",
    {
        "type": "object",
        "properties": {
            "selections": {
                "type": "array",
                "minItems": 1,
                "maxItems": 50,
                "items": {
                    "type": "object",
                    "properties": {
                        "request_id": {"type": "string"},
                        "plan_id": {"type": "string"},
                    },
                    "required": ["request_id", "plan_id"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["selections"],
        "additionalProperties": False,
    },
    "state_change",
)

PROFILE_OPTIONS_SPEC = ToolSpec(
    "view_reservation_options",
    "Inspect the frozen public promise profiles for one inquiry line, page the exact public members of one eligible profile, or page the frozen authorized exact members after confirmation. Results include raw catalog, asset-state, order, cost, deposit, and delivery facts but never acceptance probabilities, hidden ceilings, shortages, feasibility, or advice.",
    {
        "type": "object",
        "properties": {
            "request_id": {"type": "string"},
            "line_id": {"type": "string"},
            "view": {
                "type": "string",
                "enum": ["profiles", "profile_members", "authorized_members"],
            },
            "profile_family_id": {"type": "string"},
            "promise_profile_id": {"type": "string"},
            "cursor": {"type": "integer", "minimum": 0},
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "default": 20,
            },
        },
        "required": ["request_id", "line_id", "view"],
        "additionalProperties": False,
    },
    "read_only",
)

V25_QUOTE_WORKSPACE_SPEC = ToolSpec(
    "view_reservation_quote_workspace",
    "Retrieve the complete frozen legal quote space for one or more current-day inquiries. Results normalize complete requests, request-line-specific Profile terms and frozen exact-member IDs, deduplicated static Profile definitions, and optionally deduplicated compact public offer definitions. They never include current prices, inventory, contracts, occupancy, orders, service jobs, customer latent state, acceptance predictions, recommendations, or feasibility. pending_page uses a stable request-ID keyset cursor and never splits a request; requests atomically resolves one to five explicit pending request IDs. compact is the default offer projection; references_only must be explicitly requested. Use inspect_offer for full source specifications and evidence, and use operational tools separately for fulfillment risk.",
    {
        "type": "object",
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["pending_page"],
                    },
                    "cursor": {"type": "string"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 5,
                    },
                    "offer_projection": {
                        "type": "string",
                        "enum": ["compact", "references_only"],
                        "default": "compact",
                    },
                },
                "required": ["mode"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["requests"],
                    },
                    "request_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 5,
                        "uniqueItems": True,
                        "items": {"type": "string"},
                    },
                    "offer_projection": {
                        "type": "string",
                        "enum": ["compact", "references_only"],
                        "default": "compact",
                    },
                },
                "required": ["mode", "request_ids"],
                "additionalProperties": False,
            },
        ],
    },
    "read_only",
)

PROFILE_STAGE_SPEC = ToolSpec(
    "stage_reservation_line",
    "Choose which frozen authorized exact offer should be tried first for one confirmed promise-profile line. Staging is allowed only before pickup, binds no asset, reserves no capacity, does not change price or deposit, and falls through to the remaining authorized order if unavailable.",
    {
        "type": "object",
        "properties": {
            "commitment_id": {"type": "string"},
            "line_id": {"type": "string"},
            "authorized_offer_id": {"type": "string"},
        },
        "required": ["commitment_id", "line_id", "authorized_offer_id"],
        "additionalProperties": False,
    },
    "state_change",
)


def _project_assortment_specs(
    rows: tuple[ToolSpec, ...], assortment_mode: str
) -> tuple[ToolSpec, ...]:
    if assortment_mode == "rental_and_retail":
        return rows
    if assortment_mode != "rental_only":
        raise ValueError(f"unsupported assortment mode: {assortment_mode}")
    projected: list[ToolSpec] = []
    rental_liquidation = {
        "type": "object",
        "properties": {
            "rental_asset_ids": {
                "type": "array",
                "minItems": 1,
                "maxItems": 25,
                "uniqueItems": True,
                "items": {"type": "string"},
            }
        },
        "required": ["rental_asset_ids"],
        "additionalProperties": False,
    }
    for spec in rows:
        if spec.name == "view_retail_inventory":
            continue
        parameters = copy.deepcopy(spec.parameters)
        description = spec.description
        if spec.name == "search_catalog":
            parameters["properties"].pop("kind", None)
            description = (
                "Browse approved exact rental-equipment supplier offers using "
                "natural-language product text, source specifications, or offer ID."
            )
        elif spec.name == "catalog_report":
            parameters["properties"].pop("kind", None)
            filters = parameters["properties"].get("filters", {})
            filters.get("properties", {}).pop("kind", None)
            description = (
                "Filter exact rental-equipment supplier offers by natural text or "
                "neutral source specification."
            )
        elif spec.name == "demand_report":
            dimensions = parameters["properties"]["group_by"]["items"]["enum"]
            parameters["properties"]["group_by"]["items"]["enum"] = [
                value for value in dimensions if value != "rental_or_retail"
            ]
        elif spec.name in {
            "view_liquidation_quotes", "sell_to_liquidation_dealer",
        }:
            parameters = copy.deepcopy(rental_liquidation)
            description = (
                "Get today's fixed-dealer quotes for selected available or damaged "
                "rental assets. Invalid input rejects the quote and returns all item diagnostics."
                if spec.name == "view_liquidation_quotes"
                else "Atomically sell selected available or damaged rental assets to "
                "the fixed dealer. Any invalid item rejects the whole sale."
            )
        elif spec.name == "set_prices":
            description = "Set per-day rental prices for exact equipment offers."
        elif spec.name == "place_purchase_order":
            description = description.replace(
                "The entire mixed order is rejected",
                "The entire order is rejected",
            ).replace(" Retail line cost is exempt.", "")
        elif spec.name == "view_warehouse_capacity":
            description = (
                "View fixed warehouse capacity including return-space reservations "
                "for every owned rental asset and confirmed inbound unit. Global "
                "totals are never filtered; per-SKU counts and fixed integer weights "
                "are paginated. Rented, cleaning, repairing, or maintenance assets "
                "still occupy capacity; actual sale or disposal releases it. Customer "
                "reservations do not add occupancy. No availability or scheduling verdict."
            )
        projected.append(ToolSpec(spec.name, description, parameters, spec.effect))
    return tuple(projected)


def tool_specs_for_mode(
    contract_mode: str,
    *,
    tool_interface_version: int | None = None,
    assortment_mode: str = "rental_and_retail",
) -> tuple[ToolSpec, ...]:
    if contract_mode == "exact_candidates_v15":
        return _project_assortment_specs(TOOL_SPECS, assortment_mode)
    if contract_mode not in {
        "promise_profiles_v1",
        "adaptive_pairwise_exact_quotes_v1",
        "complete_contract_options_v1",
        "factorized_line_options_plan_set_v1",
    }:
        raise ValueError(f"unsupported reservation contract mode: {contract_mode}")
    rows: list[ToolSpec] = []
    v24_description_replacements = {
        "business_status": (
            V29_BUSINESS_STATUS_SPEC
            if tool_interface_version is not None
            and tool_interface_version >= 25
            else V24_BUSINESS_STATUS_SPEC
        ),
        "view_active_rentals": V24_ACTIVE_RENTALS_SPEC,
        "reservation_inbox": (
            V29_RESERVATION_INBOX_SPEC
            if tool_interface_version is not None
            and tool_interface_version >= 25
            else V27_RESERVATION_INBOX_SPEC
            if tool_interface_version is not None
            and tool_interface_version >= 23
            else V25_RESERVATION_INBOX_SPEC
            if tool_interface_version is not None
            and tool_interface_version >= 21
            else V24_RESERVATION_INBOX_SPEC
        ),
    }
    if tool_interface_version is not None and tool_interface_version >= 23:
        v24_description_replacements["respond_reservation_changes"] = (
            V29_RESPOND_RESERVATION_CHANGES_SPEC
            if tool_interface_version >= 25
            else V27_RESPOND_RESERVATION_CHANGES_SPEC
        )
    if tool_interface_version is not None and tool_interface_version >= 25:
        v24_description_replacements["place_purchase_order"] = V29_PLACE_PURCHASE_ORDER_SPEC
        v24_description_replacements["purchase_order_status"] = V29_PURCHASE_ORDER_STATUS_SPEC
    if tool_interface_version is not None and tool_interface_version >= 26:
        v24_description_replacements["close_business_day"] = V30_CLOSE_BUSINESS_DAY_SPEC
    for spec in TOOL_SPECS:
        if (
            contract_mode == "factorized_line_options_plan_set_v1"
            and tool_interface_version is not None
            and tool_interface_version >= 20
            and spec.name in v24_description_replacements
        ):
            rows.append(v24_description_replacements[spec.name])
            continue
        if (
            contract_mode == "factorized_line_options_plan_set_v1"
            and tool_interface_version is not None
            and tool_interface_version >= 18
            and spec.name == "view_reservation_calendar"
        ):
            rows.append(
                V24_RENTAL_CAPACITY_SPEC
                if tool_interface_version >= 20
                else V22_RENTAL_CAPACITY_SPEC
            )
            continue
        if (
            contract_mode in {
                "adaptive_pairwise_exact_quotes_v1",
                "complete_contract_options_v1",
                "factorized_line_options_plan_set_v1",
            }
            and spec.name == "view_reservation"
        ):
            rows.append(
                V29_FACTORIZED_PLAN_RESERVATION_VIEW_SPEC
                if contract_mode == "factorized_line_options_plan_set_v1"
                and tool_interface_version is not None
                and tool_interface_version >= 25
                else V27_FACTORIZED_PLAN_RESERVATION_VIEW_SPEC
                if contract_mode == "factorized_line_options_plan_set_v1"
                and tool_interface_version is not None
                and tool_interface_version >= 23
                else V26_FACTORIZED_PLAN_RESERVATION_VIEW_SPEC
                if contract_mode == "factorized_line_options_plan_set_v1"
                and tool_interface_version is not None
                and tool_interface_version >= 22
                else V25_FACTORIZED_PLAN_RESERVATION_VIEW_SPEC
                if contract_mode == "factorized_line_options_plan_set_v1"
                and tool_interface_version is not None
                and tool_interface_version >= 21
                else V24_FACTORIZED_PLAN_RESERVATION_VIEW_SPEC
                if contract_mode == "factorized_line_options_plan_set_v1"
                and tool_interface_version is not None
                and tool_interface_version >= 20
                else V23_FACTORIZED_PLAN_RESERVATION_VIEW_SPEC
                if contract_mode == "factorized_line_options_plan_set_v1"
                and tool_interface_version is not None
                and tool_interface_version >= 19
                else V22_FACTORIZED_PLAN_RESERVATION_VIEW_SPEC
                if contract_mode == "factorized_line_options_plan_set_v1"
                and tool_interface_version is not None
                and tool_interface_version >= 18
                else NOISY_LINE_FACTORIZED_PLAN_RESERVATION_VIEW_SPEC
                if contract_mode == "factorized_line_options_plan_set_v1"
                and tool_interface_version is not None
                and tool_interface_version >= 17
                else FACTORIZED_PLAN_RESERVATION_VIEW_SPEC
                if contract_mode == "factorized_line_options_plan_set_v1"
                else COMPLETE_CONTRACT_RESERVATION_VIEW_SPEC
                if contract_mode == "complete_contract_options_v1"
                and tool_interface_version is not None
                and tool_interface_version >= 15
                else SEARCHABLE_PAIRWISE_RESERVATION_VIEW_SPEC
                if tool_interface_version is not None
                and tool_interface_version >= 13
                else PAIRWISE_RESERVATION_VIEW_SPEC
            )
        elif spec.name == "quote_reservation_inquiries":
            rows.extend(
                (
                    (
                        V25_QUOTE_WORKSPACE_SPEC
                        if contract_mode
                        == "factorized_line_options_plan_set_v1"
                        and tool_interface_version is not None
                        and tool_interface_version >= 21
                        else PROFILE_OPTIONS_SPEC
                    ),
                    (
                    V25_FACTORIZED_PLAN_QUOTE_SPEC
                    if contract_mode == "factorized_line_options_plan_set_v1"
                    and tool_interface_version is not None
                    and tool_interface_version >= 21
                    else V23_FACTORIZED_PLAN_QUOTE_SPEC
                    if contract_mode == "factorized_line_options_plan_set_v1"
                    and tool_interface_version is not None
                    and tool_interface_version >= 19
                    else V22_FACTORIZED_PLAN_QUOTE_SPEC
                    if contract_mode == "factorized_line_options_plan_set_v1"
                    and tool_interface_version is not None
                    and tool_interface_version >= 18
                    else NOISY_LINE_FACTORIZED_PLAN_QUOTE_SPEC
                    if contract_mode == "factorized_line_options_plan_set_v1"
                    and tool_interface_version is not None
                    and tool_interface_version >= 17
                    else FACTORIZED_PLAN_QUOTE_SPEC
                    if contract_mode == "factorized_line_options_plan_set_v1"
                    else COMPLETE_CONTRACT_QUOTE_SPEC
                    if contract_mode == "complete_contract_options_v1"
                    else PROFILE_QUOTE_SPEC
                    if contract_mode == "promise_profiles_v1"
                        else (
                            REQUEST_BUNDLE_PAIRWISE_QUOTE_SPEC
                            if tool_interface_version is not None
                            and tool_interface_version >= 14
                            else PAIRWISE_QUOTE_SPEC
                        )
                    ),
                    *(
                        (FACTORIZED_PLAN_CONFIRM_SPEC,)
                        if contract_mode
                        == "factorized_line_options_plan_set_v1"
                        else ()
                    ),
                    *(
                        (PROFILE_STAGE_SPEC,)
                        if contract_mode == "promise_profiles_v1"
                        else ()
                    ),
                )
            )
        elif (
            spec.name == "resolve_reservation_risks"
            and contract_mode == "factorized_line_options_plan_set_v1"
            and tool_interface_version is not None
            and tool_interface_version >= 20
        ):
            rows.append(
                V45_RESERVATION_RISK_SPEC
                if tool_interface_version >= 40
                else V29_RESERVATION_RISK_SPEC
                if tool_interface_version >= 25
                else V24_RESERVATION_RISK_SPEC
            )
        else:
            rows.append(spec)
    if (
        contract_mode == "factorized_line_options_plan_set_v1"
        and tool_interface_version is not None
        and tool_interface_version >= 20
    ):
        rows.append(
            V28_WALK_IN_CAPACITY_LIMIT_SPEC
            if tool_interface_version >= 24
            else V24_WALK_IN_CAPACITY_LIMIT_SPEC
        )
        if tool_interface_version >= 25:
            rows.append(V29_CANCEL_RESERVATION_SPEC)
        if tool_interface_version >= 27:
            rows.append(V31_WAREHOUSE_CAPACITY_SPEC)
            updated = []
            for spec in rows:
                parameters = copy.deepcopy(spec.parameters)
                description = spec.description
                if spec.name == "place_purchase_order":
                    description += " V31: rental packs times package_units reserve fixed warehouse capacity immediately, including return space while rented. The entire mixed order is rejected if over capacity. Check view_warehouse_capacity; no reservation feasibility is checked."
                    if tool_interface_version >= 31:
                        description += (
                            " V40: accepted unreceived owned durable unit basis enters the fixed gross deployed-capital envelope. Retail and supplier-trial lines are exempt. Cash and capital headroom are independent; over-limit orders fail atomically. Permanent owned-asset exits release their frozen basis next business day. Check business_status."
                            if tool_interface_version >= 35
                            else " V36: accepted durable line cost also consumes the fixed non-replenishing durable acquisition budget. Retail line cost is exempt. The order is rejected atomically when its durable subtotal exceeds the remaining budget; check business_status."
                        )
                elif spec.name == "repair_or_retire_assets":
                    description += " V31: retirement physically disposes the selected damaged assets, releases warehouse capacity once and retains an audit event."
                elif spec.name == "view_rental_activity":
                    parameters["properties"]["event_types"]["items"]["enum"].append("asset_disposed")
                updated.append(ToolSpec(spec.name, description, parameters, spec.effect))
            rows = updated
    if tool_interface_version is not None and tool_interface_version >= 28 and contract_mode == "factorized_line_options_plan_set_v1":
        rows.append(ToolSpec("respond_reservation_service_offers",
            "Accept or decline today's fixed-reward customer service offers, NOT price inquiries. Accept one frozen complete plan and one advertised pickup_day; cash and commitment form immediately without another customer price test. No feasibility is checked. Do not mix plans or submit prices. Unanswered offers expire without a commitment. Entire batch validates before any change.",
            {"type": "object", "properties": {"decisions": {"type": "array", "minItems": 1, "maxItems": 5,
                "items": {"oneOf": [
                    {"type": "object", "properties": {"request_id": {"type": "string"}, "action": {"const": "accept"},
                        "plan_id": {"type": "string"}, "pickup_day": {"type": "integer", "minimum": 1}},
                     "required": ["request_id", "action", "plan_id", "pickup_day"], "additionalProperties": False},
                    {"type": "object", "properties": {"request_id": {"type": "string"}, "action": {"const": "decline"}},
                     "required": ["request_id", "action"], "additionalProperties": False}]}}},
             "required": ["decisions"], "additionalProperties": False}, "state_change"))
        updated = []
        for spec in rows:
            parameters = copy.deepcopy(spec.parameters)
            description = spec.description
            if spec.name == "confirm_reservation_plans":
                parameters["properties"]["selections"]["items"]["properties"]["pickup_day"] = {"type": "integer", "minimum": 1}
                description += " V32: include pickup_day when there are two advertised dates, even for one authorized plan. Authorization covers both dates. If unanswered, the original inquiry date and customer preferred plan bind at day end. Fixed reward offers use respond_reservation_service_offers instead."
            elif spec.name == "quote_reservation_inquiries":
                description = "For price_inquiry requests only. Explicitly quote each required/optional line using the frozen factorized schema. One menu and its prices cover every advertised date. Customer authorization is common to those dates. With two dates even a single authorized plan awaits confirm_reservation_plans with pickup_day. At day end an unanswered authorized quote binds at the original inquiry date and customer preferred plan. Existing noisy feedback appears only for rejected price inquiries; no requotes. Fixed-reward offers use respond_reservation_service_offers."
            elif spec.name == "view_reservation_quote_workspace":
                description += " V32: includes pending fixed-reward service offers with exact immutable complete plans/line-day rent/deposits and price inquiries with full quotation choices. Date options are customer-authorized dates, not stock advice. Do not mix lines across service-offer plans."
            elif spec.name == "reservation_inbox":
                description += " V32 also returns service_offers and their count; these require accept/decline, not quotations."
            elif spec.name == "respond_reservation_changes":
                description += " V32: read original_dates_still_possible and on_decline/on_expiry. If the original trip is impossible, decline or expiry causes customer cancellation/refund, not manager default."
            elif spec.name == "view_reservation":
                description += " V32 quote_history also contains separately typed fixed_reward_offer records, not agent quotations; filter service_offer_accepted/declined/expired or request IDs. inquiry_terms contains immutable date options and offered amounts/plans."
                for branch in parameters.get("oneOf", []):
                    field = branch.get("properties", {}).get("quote_outcomes")
                    if field:
                        field["items"]["enum"] += ["service_offer_accepted", "service_offer_declined", "service_offer_expired"]
                        field["maxItems"] = 8
            updated.append(ToolSpec(spec.name, description, parameters, spec.effect))
        rows = updated
    if (
        tool_interface_version is not None
        and tool_interface_version >= 32
        and contract_mode == "factorized_line_options_plan_set_v1"
    ):
        updated = []
        for spec in rows:
            parameters = copy.deepcopy(spec.parameters)
            description = spec.description
            if spec.name == "quote_reservation_inquiries":
                description = (
                    "For price_inquiry requests only. Submit every inquiry line. "
                    "Each required line needs one or two exact offer/Profile/positive "
                    "cent-exact price options and cannot be omitted. Each optional line "
                    "may submit zero to two quote options; the environment always adds "
                    "one implicit omit. One explicit omit remains accepted for backward "
                    "compatibility and normalizes to that same singleton; duplicate omits "
                    "are invalid. The Cartesian product may contain at most 24 plans. "
                    "Customer authorization is not an inventory or feasibility verdict."
                )
                quote_branch = parameters["properties"]["decisions"]["items"]["oneOf"][0]
                quote_branch["properties"]["lines"]["items"]["properties"]["options"]["minItems"] = 0
            elif spec.name == "place_purchase_order":
                parameters["properties"]["procurement_mode"] = {
                    "type": "string",
                    "enum": ["owned", "supplier_trial"],
                    "default": "owned",
                    "description": (
                        "owned performs a normal purchase. supplier_trial requests "
                        "one pack containing one never-owned durable asset under the "
                        "public bounded trial policy."
                    ),
                }
                description += (
                    " V37: procurement_mode defaults to owned. supplier_trial accepts "
                    "exactly one single-unit durable pack, charges the public handling "
                    "fee, occupies the trial slot and warehouse immediately, and never "
                    "consumes durable acquisition budget."
                )
            elif spec.name == "view_rental_activity":
                parameters["properties"]["event_types"]["items"]["enum"] += list(
                    SUPPLIER_TRIAL_ACTIVITY_EVENT_TYPES
                )
            elif spec.name == "view_rental_assets":
                parameters["properties"]["status"]["enum"].append(
                    "supplier_returned"
                )
                description += (
                    " V37 asset detail identifies owned versus supplier-trial "
                    "equipment and exposes the applicable trial deadline, remaining "
                    "contract starts, and return state."
                )
            updated.append(ToolSpec(spec.name, description, parameters, spec.effect))
        rows = updated
        rows.append(
            ToolSpec(
                "return_supplier_trial_assets",
                "Return active supplier-owned trial assets. Available assets leave "
                "immediately; rented, cleaning, damaged, repairing, or maintenance "
                "assets become return-pending and leave after required service. Early "
                "return never restores the four-trial allowance. The entire batch is "
                "validated atomically.",
                {
                    "type": "object",
                    "properties": {
                        "asset_ids": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 10,
                            "uniqueItems": True,
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["asset_ids"],
                    "additionalProperties": False,
                },
                "state_change",
            )
        )
    if (
        tool_interface_version is not None
        and tool_interface_version >= 33
        and contract_mode == "factorized_line_options_plan_set_v1"
    ):
        rows.extend((V38_REQUEST_CONTINGENCY_SPEC, V38_APPLY_CONTINGENCY_SPEC))
        updated = []
        for spec in rows:
            description = spec.description
            if spec.name == "reservation_inbox":
                description += (
                    " V38 also returns contingency_reports published today; accepted "
                    "changes require apply_reservation_contingency before close."
                )
            elif spec.name == "view_reservation":
                description += (
                    " V38 responses may also include contingency_history for matching "
                    "commitment, inquiry, or report identifiers."
                )
            updated.append(
                ToolSpec(spec.name, description, copy.deepcopy(spec.parameters), spec.effect)
            )
        rows = updated
    if (
        tool_interface_version is not None
        and tool_interface_version >= 34
        and contract_mode == "factorized_line_options_plan_set_v1"
    ):
        updated = []
        for spec in rows:
            parameters = copy.deepcopy(spec.parameters)
            description = spec.description
            if spec.name == "weather_forecast":
                parameters["properties"]["horizon_days"]["maximum"] = 15
                parameters["properties"]["horizon_days"]["description"] = (
                    "Inclusive date count. Use 15 for today plus fourteen "
                    "future days."
                )
                description = (
                    "View public weather and calendar forecasts for an inclusive "
                    "window. horizon_days=15 returns today plus fourteen future days."
                )
            elif spec.name == "view_rental_capacity":
                parameters["properties"]["from_day"]["description"] = (
                    "timeline only; inclusive first date. Together with to_day "
                    "it must satisfy to_day - from_day <= 14."
                )
                parameters["properties"]["to_day"]["description"] = (
                    "timeline only; inclusive last date. At most 15 dates are "
                    "allowed, so to_day - from_day <= 14."
                )
                description = (
                    "Inspect raw rental-capacity facts. timeline accepts view, "
                    "offer_ids, from_day and to_day and may cover at most 15 "
                    "inclusive dates—today plus fourteen future days when begun "
                    "today. service_jobs retains its paginated current-queue view. "
                    "No net availability, feasibility, or recommendation is computed."
                )
            elif spec.name == "view_rental_assets":
                properties = parameters["properties"]
                common = {
                    "status": copy.deepcopy(properties["status"]),
                    "offer_id": copy.deepcopy(properties["offer_id"]),
                }
                parameters = {
                    "type": "object",
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "view": {"const": "summary"},
                                **copy.deepcopy(common),
                            },
                            "required": ["view"],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {
                                "view": {"const": "asset"},
                                **copy.deepcopy(common),
                                "cursor": copy.deepcopy(properties["cursor"]),
                                "limit": copy.deepcopy(properties["limit"]),
                            },
                            "required": ["view"],
                            "additionalProperties": False,
                        },
                    ]
                }
                description = (
                    "View rental equipment using one explicit mode. summary accepts "
                    "only view, status and offer_id. asset additionally accepts cursor "
                    "and limit. Pagination is invalid for summary."
                )
            elif spec.name == "reservation_inbox":
                description = (
                    "Authoritative entry point for today's reservation work and "
                    "customer updates. contingency_updates_today contains every "
                    "contingency result published today; actionable_contingency_reports "
                    "contains only reports that can be applied before close. Only the "
                    "actionable count contributes to reservation_actions_due_today."
                )
            elif spec.name == "business_status":
                description += (
                    " V39 actionable_contingency_report_count counts only reports "
                    "that currently require an apply decision."
                )
            elif spec.name == "view_reservation":
                description += (
                    " V39 index and current_terms expose the current commitment "
                    "version, proactive-inquiry eligibility and deadline. Confirmed "
                    "line detail exposes contingency_same_profile_offer_ids: the exact "
                    "same-Profile SKU IDs that are syntactically legal contingency "
                    "substitutions, without a feasibility or recommendation judgment."
                )
            updated.append(ToolSpec(spec.name, description, parameters, spec.effect))
        rows = updated
    if (
        tool_interface_version is not None
        and tool_interface_version >= 35
        and contract_mode == "factorized_line_options_plan_set_v1"
    ):
        updated = []
        for spec in rows:
            description = spec.description.replace(
                "durable acquisition budget",
                "gross deployed durable-capital envelope",
            )
            if spec.name == "business_status":
                description += (
                    " V40 durable_capital reports initial owned basis, configured "
                    "additional headroom, fixed limit, deployed and committed basis, "
                    "next-day pending release, and remaining headroom."
                )
            elif spec.name == "view_rental_assets":
                description += (
                    " V40 asset rows expose each asset's historical durable capital "
                    "basis, deployed/pending/released/exempt status, and any exit and "
                    "next-day release date."
                )
            elif spec.name == "repair_or_retire_assets":
                description += (
                    " V40 completed retirement of an owned asset releases its frozen "
                    "capital basis on the next business day, never immediately."
                )
            elif spec.name == "sell_to_liquidation_dealer":
                description += (
                    " V40 sale cash is immediate but an owned asset's frozen capital "
                    "basis becomes reusable only on the next business day."
                )
            elif spec.name == "purchase_order_status":
                description += (
                    " V40 owned orders report their frozen durable capital basis; "
                    "unreceived unit basis is committed capital."
                )
            updated.append(
                ToolSpec(
                    spec.name,
                    description,
                    copy.deepcopy(spec.parameters),
                    spec.effect,
                )
            )
        rows = updated
    if (
        tool_interface_version is not None
        and tool_interface_version >= 36
        and contract_mode == "factorized_line_options_plan_set_v1"
    ):
        updated = []
        for spec in rows:
            description = spec.description
            if spec.name == "set_walk_in_capacity_limits":
                description += (
                    " V41 returns only this batch's applied items, the merged "
                    "current/future effective intervals for touched offers, compact "
                    "warnings, and whether state changed. It never echoes other offers "
                    "or expired history. Use view_rental_capacity with explicit offer "
                    "IDs and dates to query capacity limits."
                )
            elif spec.name == "view_rental_capacity":
                description += (
                    " V41 walk-in limit intervals are clipped to the requested date "
                    "window and adjacent equal-valued intervals are merged."
                )
            updated.append(
                ToolSpec(
                    spec.name,
                    description,
                    copy.deepcopy(spec.parameters),
                    spec.effect,
                )
            )
        rows = updated
    if tool_interface_version is not None and tool_interface_version >= 38:
        from .actor_manual import TOPICS
        rows.append(ToolSpec(
            "read_business_rules",
            "Read public business rules; omit topics for the directory.",
            {"type": "object", "properties": {
                "topics": {"type": "array", "items": {"type": "string", "enum": list(TOPICS)},
                           "minItems": 1, "maxItems": len(TOPICS), "uniqueItems": True},
            }, "additionalProperties": False},
            "read_only",
        ))
    if tool_interface_version is not None and tool_interface_version >= 51:
        from .operating_history import tool_parameters
        rows = [ToolSpec(spec.name, spec.description, tool_parameters(spec.name, spec.parameters), spec.effect)
                for spec in rows]
    if tool_interface_version is not None and tool_interface_version >= 52:
        from .recovery_state import query_parameters
        rows = [ToolSpec(spec.name, spec.description, query_parameters(spec.name, spec.parameters), spec.effect)
                for spec in rows]
    projected = _project_assortment_specs(tuple(rows), assortment_mode)
    if tool_interface_version is not None and tool_interface_version >= 40:
        from .actor_tool_text import V45_DESCRIPTIONS
        descriptions = V45_DESCRIPTIONS
        if tool_interface_version >= 51:
            from .operating_history import TOOL_DESCRIPTIONS
            descriptions = {**V45_DESCRIPTIONS, **TOOL_DESCRIPTIONS}
        return tuple(
            ToolSpec(
                spec.name,
                descriptions.get(spec.name, spec.description)
                + (" Retail prices are per consumer unit."
                   if spec.name == "set_prices" and assortment_mode == "rental_and_retail"
                   else ""),
                spec.parameters,
                spec.effect,
            )
            for spec in projected
        )
    return projected
